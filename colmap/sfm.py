"""COLMAP SfM: feature extraction, matching, sparse reconstruction, undistortion.

Wraps pycolmap 4.0.3 into a single-class pipeline. Extractor and matcher are
selected independently via the `extractor` and `matcher` config fields.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import threading
from pathlib import Path

import pycolmap
from tqdm.auto import tqdm

from config import ColmapConfig

logger = logging.getLogger(__name__)

# Several pycolmap calls have glog emit multiple INFO lines per image
# (incremental_mapping 5/image, SIFT extract_features 6/image), which floods the
# terminal at large N. Capture fd 2 into a pipe and drive a tqdm bar off the
# caller-supplied regex; every glog INFO is swallowed by the drain thread. Fatals
# still surface as Python exceptions, which is enough.
_REG_RE = re.compile(r"Registering image #\d+ \(num_reg_frames=(\d+)\)")
_EXTRACT_RE = re.compile(r"Processed file \[(\d+)/\d+\]")


@contextlib.contextmanager
def _glog_progress(total: int, desc: str, pattern: re.Pattern):
    """Redirect C++ glog stderr to a pipe; drive a tqdm bar from `pattern`'s
    first capture group (the running count).

    The bar writes to saved_fd (a second dup of the original stderr) so it never
    feeds back into the pipe -> drain -> tqdm.write -> pipe loop.
    """
    saved_fd = os.dup(2)
    bar_fd = os.dup(saved_fd)
    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 2)
    os.close(w_fd)

    bar_stream = os.fdopen(bar_fd, "w", buffering=1)
    bar = tqdm(total=total, desc=desc, unit="img", file=bar_stream)

    def drain() -> None:
        with os.fdopen(r_fd, "r", errors="replace") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    bar.n = min(int(m.group(1)), total)
                    bar.refresh()

    th = threading.Thread(target=drain, daemon=True)
    th.start()

    try:
        yield
    finally:
        # dup2 closes the old fd 2 (the pipe's write end) first; combined with the
        # w_fd we already closed, that hands the drain side its EOF.
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        th.join(timeout=3.0)
        bar.close()


class ColmapSfM:
    """COLMAP SfM pipeline via pycolmap 4.0.3."""

    def __init__(self, config: ColmapConfig) -> None:
        self.config = config

    def run(self) -> None:
        if not self.config.input_images.exists():
            raise FileNotFoundError(f"Images not found: {self.config.input_images}")

        if self.config.colmap_dir.exists():
            shutil.rmtree(self.config.colmap_dir)
        self.config.colmap_dir.mkdir(parents=True)

        self._extract_and_match()
        model_path = self._sparse_reconstruction()
        self._undistort(model_path)

        logger.info("COLMAP SfM complete: %s", self.config.colmap_dir)

    def _extract_and_match(self) -> None:
        if self.config.extractor == "superpoint":
            self._extract_and_match_hloc()
            return

        family = "SIFT" if self.config.extractor == "sift" else "ALIKED"
        extractor_members = pycolmap.FeatureExtractorType.__members__
        matcher_members = pycolmap.FeatureMatcherType.__members__
        extractor_type = extractor_members[self.config.extractor.upper()]
        matcher_type = matcher_members[f"{family}_{self.config.matcher.upper()}"]
        camera_mode = pycolmap.CameraMode.__members__[self.config.camera_mode.upper()]
        reader_options = pycolmap.ImageReaderOptions(
            camera_model=self.config.camera_model.upper(),
        )

        db = self.config.colmap_database

        logger.info(
            "Extracting features (extractor=%s, matcher=%s, max_num_features=%d, "
            "camera_mode=%s, camera_model=%s)",
            self.config.extractor, self.config.matcher, self.config.max_num_features,
            self.config.camera_mode, self.config.camera_model,
        )
        extraction_options = pycolmap.FeatureExtractionOptions(type=extractor_type)
        if family == "SIFT":
            extraction_options.sift.max_num_features = self.config.max_num_features
        else:
            extraction_options.aliked.max_num_features = self.config.max_num_features
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        n_images = sum(
            1 for p in self.config.input_images.iterdir()
            if p.is_file() and p.suffix.lower() in image_exts
        )
        with _glog_progress(n_images, f"{family} extract", _EXTRACT_RE):
            pycolmap.extract_features(
                db,
                self.config.input_images,
                camera_mode=camera_mode,
                reader_options=reader_options,
                extraction_options=extraction_options,
                device=pycolmap.Device.cuda,
            )

        matching_options = pycolmap.FeatureMatchingOptions(type=matcher_type)
        if self.config.pair_strategy == "vocab_tree":
            logger.info(
                "Matching features via vocab tree (extractor=%s, matcher=%s, "
                "num_images=%d, num_nearest_neighbors=%d, vocab_tree=%s)",
                self.config.extractor, self.config.matcher,
                self.config.num_images, self.config.num_nearest_neighbors,
                self.config.vocab_tree_path,
            )
            pairing_options = pycolmap.VocabTreePairingOptions(
                vocab_tree_path=self.config.vocab_tree_path,
                num_images=self.config.num_images,
                num_nearest_neighbors=self.config.num_nearest_neighbors,
            )
            pycolmap.match_vocabtree(
                db,
                matching_options=matching_options,
                pairing_options=pairing_options,
                device=pycolmap.Device.cuda,
            )
        else:
            logger.info(
                "Matching features exhaustively (extractor=%s, matcher=%s)",
                self.config.extractor, self.config.matcher,
            )
            pycolmap.match_exhaustive(
                db,
                matching_options=matching_options,
                device=pycolmap.Device.cuda,
            )

    def _extract_and_match_hloc(self) -> None:
        # pycolmap.FeatureExtractorType has no SUPERPOINT, so extractor+matcher
        # must both be run out-of-band via hloc, and the results written into
        # database.db before pycolmap.incremental_mapping picks it up.
        from hloc import (
            extract_features,
            match_features,
            pairs_from_exhaustive,
            pairs_from_retrieval,
        )
        from hloc.triangulation import (
            estimation_and_geometric_verification,
            import_features,
            import_matches,
        )

        self.config.hloc_dir.mkdir(parents=True, exist_ok=True)

        feat_conf = {
            "model": {
                "name": "superpoint",
                "nms_radius": self.config.nms_radius,
                "max_keypoints": self.config.max_num_features,
            },
            "preprocessing": {
                "grayscale": True,
                "resize_max": self.config.resize_max,
                "resize_force": self.config.resize_force,
            },
        }
        match_conf = {
            "output": "matches-superpoint-lightglue",
            "model": {
                "name": "lightglue",
                "features": "superpoint",
                "depth_confidence": self.config.lightglue_depth_confidence,
                "width_confidence": self.config.lightglue_width_confidence,
            },
        }

        logger.info(
            "Extracting SuperPoint features (max_keypoints=%d, nms_radius=%d, resize_max=%d, resize_force=%s)",
            self.config.max_num_features, self.config.nms_radius,
            self.config.resize_max, self.config.resize_force,
        )
        extract_features.main(
            conf=feat_conf,
            image_dir=self.config.input_images,
            feature_path=self.config.hloc_features,
        )

        if self.config.pair_strategy == "retrieval":
            logger.info(
                "Extracting global descriptors (%s) for retrieval pairing",
                self.config.retrieval_model,
            )
            retrieval_conf = extract_features.confs[self.config.retrieval_model]
            extract_features.main(
                conf=retrieval_conf,
                image_dir=self.config.input_images,
                feature_path=self.config.hloc_global_features,
            )
            logger.info(
                "Building retrieval pairs (model=%s, num_matched=%d)",
                self.config.retrieval_model, self.config.num_matched,
            )
            pairs_from_retrieval.main(
                descriptors=self.config.hloc_global_features,
                output=self.config.hloc_pairs,
                num_matched=self.config.num_matched,
            )
        else:
            logger.info("Building exhaustive pairs")
            pairs_from_exhaustive.main(
                output=self.config.hloc_pairs, features=self.config.hloc_features,
            )

        logger.info(
            "Matching features (LightGlue, depth_confidence=%s, width_confidence=%s)",
            self.config.lightglue_depth_confidence,
            self.config.lightglue_width_confidence,
        )
        match_features.main(
            conf=match_conf,
            pairs=self.config.hloc_pairs,
            features=self.config.hloc_features,
            matches=self.config.hloc_matches,
        )

        db_path = self.config.colmap_database
        with pycolmap.Database.open(db_path):
            pass
        pycolmap.import_images(
            db_path,
            self.config.input_images,
            pycolmap.CameraMode.__members__[self.config.camera_mode.upper()],
            options=pycolmap.ImageReaderOptions(
                camera_model=self.config.camera_model.upper(),
            ),
        )
        with pycolmap.Database.open(db_path) as db:
            image_ids = {image.name: image.image_id for image in db.read_all_images()}
            import_features(image_ids, db, self.config.hloc_features)
            import_matches(image_ids, db, self.config.hloc_pairs, self.config.hloc_matches)
        # Mapper needs two_view_geometries populated; skipping this reproduces
        # the "ba_config.NumImages() >= 2" crash seen with ALIKED+LightGlue.
        estimation_and_geometric_verification(db_path, self.config.hloc_pairs)

    def _sparse_reconstruction(self) -> Path:
        sparse_parent = self.config.colmap_sparse_parent
        sparse_parent.mkdir(parents=True)

        if self.config.mapper == "global":
            maps = self._global_mapping(sparse_parent)
        else:
            with pycolmap.Database.open(self.config.colmap_database) as db:
                n_images = len(db.read_all_images())
            logger.info("Running incremental SfM (%d images)...", n_images)
            with _glog_progress(n_images, "Incremental SfM", _REG_RE):
                maps = pycolmap.incremental_mapping(
                    self.config.colmap_database,
                    self.config.input_images,
                    sparse_parent,
                )
        if not maps:
            raise RuntimeError("COLMAP produced no reconstruction")

        largest_id = max(maps, key=lambda mid: len(maps[mid].images))
        largest = maps[largest_id]
        logger.info(
            "SfM: %d model(s); largest=%d (%d images, %d points)",
            len(maps), largest_id, len(largest.images), len(largest.points3D),
        )

        target_dir = self.config.colmap_sparse  # sparse/0
        largest_dir = sparse_parent / str(largest_id)
        if largest_dir != target_dir:
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(str(largest_dir), str(target_dir))
        for sub in sparse_parent.iterdir():
            if sub != target_dir and sub.is_dir():
                shutil.rmtree(sub)

        return target_dir

    def _global_mapping(self, sparse_parent: Path) -> dict:
        # GLOMAP is sensitive to prior_focal_length; calibrate_view_graph estimates
        # focal lengths from existing two-view geometries and upgrades them to
        # CALIBRATED, which the global mapper requires for reliable precision.
        logger.info("Calibrating view graph focal lengths...")
        pycolmap.calibrate_view_graph(self.config.colmap_database)

        if not self.config.glomap_refine:
            logger.info("Running global SfM (GLOMAP)...")
            return pycolmap.global_mapping(
                self.config.colmap_database,
                self.config.input_images,
                sparse_parent,
            )

        # Hybrid: GLOMAP bootstraps poses quickly; incremental then runs its BA
        # outlier filter (unregisters flipped low-parallax cameras) and re-registers
        # them via PnP against GLOMAP's 3D tracks. Stash GLOMAP output in a
        # sibling subdir so sparse_parent only contains the final incremental models.
        glomap_parent = sparse_parent / "_glomap"
        glomap_parent.mkdir()
        logger.info("Running global SfM (GLOMAP)...")
        glomap_maps = pycolmap.global_mapping(
            self.config.colmap_database, self.config.input_images, glomap_parent,
        )
        if not glomap_maps:
            return glomap_maps
        largest_id = max(glomap_maps, key=lambda mid: len(glomap_maps[mid].images))
        logger.info(
            "Refining GLOMAP output via incremental mapper (input model=%d)...",
            largest_id,
        )
        # pycolmap.incremental_mapping with input_path writes the refined model
        # files flat into output_path (not output_path/<id>/), so point it at
        # sparse_parent/0 directly and remap the returned id to 0 for the
        # caller's move/cleanup logic.
        model_dir = sparse_parent / "0"
        model_dir.mkdir()
        refined_maps = pycolmap.incremental_mapping(
            self.config.colmap_database,
            self.config.input_images,
            model_dir,
            input_path=glomap_parent / str(largest_id),
        )
        if not refined_maps:
            shutil.rmtree(model_dir)
            return refined_maps
        return {0: next(iter(refined_maps.values()))}

    def _undistort(self, model_path: Path) -> None:
        self.config.colmap_dense.mkdir(parents=True)

        logger.info("Undistorting images...")
        # undistort_images logs one glog INFO per image, flooding at large N. Lift
        # the threshold to WARNING for the call; other stages (GLOMAP /
        # incremental) keep the default level.
        prev_level = pycolmap.logging.minloglevel
        pycolmap.logging.minloglevel = pycolmap.logging.WARNING
        try:
            pycolmap.undistort_images(
                self.config.colmap_dense,
                model_path,
                self.config.input_images,
            )
        finally:
            pycolmap.logging.minloglevel = prev_level


if __name__ == "__main__":
    from utils.cli import run_stage

    run_stage(__file__, "COLMAP SfM", ColmapConfig.from_yaml, ColmapSfM)
