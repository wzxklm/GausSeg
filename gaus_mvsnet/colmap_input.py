"""COLMAP sparse model -> MVSNet scene layout (cams/, pair.txt).

Produces what the depth network reads: one camera file per view holding the
extrinsic, intrinsic and depth range, plus the reference/source view pairing
scored as in Yao et al. Nothing under the COLMAP output is modified — views
that survive filtering keep their original image, and the index -> image name
mapping travels with the returned views rather than by renaming files.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap

from config import GausMvsNetConfig

logger = logging.getLogger(__name__)

_NUM_SOURCE_VIEWS = 10
_DEPTH_PERCENTILE = 0.01
# A registered view whose depth span collapses to near nothing sits almost at
# the scene origin: a pose failure that would otherwise poison its cost volume.
_OUTLIER_RANGE_RATIO = 0.01
_PARALLEL_MIN_VIEWS = 20


@dataclass
class MvsView:
    """One reference view of the MVSNet-format scene."""

    index: int
    image_name: str
    K: np.ndarray
    extrinsic: np.ndarray
    depth_min: float
    depth_interval: float
    depth_num: int
    depth_max: float
    source_views: list[tuple[int, float]]


_WORKER: dict[str, object] = {}


def _init_score_worker(centers, point_sets, theta0, sigma1, sigma2) -> None:
    _WORKER.update(
        centers=centers, point_sets=point_sets,
        theta0=theta0, sigma1=sigma1, sigma2=sigma2,
    )


def _score_worker(pair: tuple[int, int]) -> tuple[int, int, float]:
    i, j = pair
    return i, j, _view_score(
        i, j,
        _WORKER["centers"], _WORKER["point_sets"],
        _WORKER["theta0"], _WORKER["sigma1"], _WORKER["sigma2"],
    )


def _view_score(
    i: int,
    j: int,
    centers: dict[int, np.ndarray],
    point_sets: dict[int, dict[int, np.ndarray]],
    theta0: float,
    sigma1: float,
    sigma2: float,
) -> float:
    """Yao et al. view-selection score: triangulation angles over shared points."""
    ids_i, ids_j = point_sets[i]["ids"], point_sets[j]["ids"]
    common = np.intersect1d(ids_i, ids_j, assume_unique=True)
    if common.size == 0:
        return 0.0

    xyz = point_sets[i]["xyz"][np.searchsorted(ids_i, common)]
    di = centers[i] - xyz
    dj = centers[j] - xyz
    cos_theta = np.clip(
        np.sum(di * dj, axis=1)
        / (np.linalg.norm(di, axis=1) * np.linalg.norm(dj, axis=1)),
        -1.0, 1.0,
    )
    thetas = np.degrees(np.arccos(cos_theta))
    sigma = np.where(thetas <= theta0, sigma1, sigma2)
    return float(np.sum(np.exp(-((thetas - theta0) ** 2) / (2 * sigma**2))))


class ColmapMvsInput:
    """Builds the MVSNet-format scene the depth network consumes."""

    def __init__(self, config: GausMvsNetConfig) -> None:
        self.config = config
        self._points: dict[str, dict[str, np.ndarray]] = {}
        # OpenMVS numbers its images by sorted name, and .dmap files are
        # keyed by that number — not by the view index, which skips
        # filtered views.
        self.openmvs_ids: dict[str, int] = {}

    def build(self) -> list[MvsView]:
        cfg = self.config
        if not cfg.colmap_sparse.is_dir():
            raise FileNotFoundError(f"COLMAP sparse model not found: {cfg.colmap_sparse}")

        recon = pycolmap.Reconstruction(str(cfg.colmap_sparse))
        if not recon.images:
            raise RuntimeError(f"No registered images in {cfg.colmap_sparse}")
        logger.info(
            "COLMAP model: %d cameras, %d images, %d points",
            len(recon.cameras), len(recon.images), len(recon.points3D),
        )

        self.openmvs_ids = {
            name: i for i, name in enumerate(sorted(recon.images[k].name for k in recon.images), start=1)
        }
        views = self._collect_views(recon)
        views = self._drop_outlier_views(views)
        self._select_source_views(views)
        self._write_scene(views)
        return views

    def _collect_views(self, recon: pycolmap.Reconstruction) -> list[MvsView]:
        views: list[MvsView] = []
        for index, image_id in enumerate(sorted(recon.images)):
            image = recon.images[image_id]
            K = np.asarray(recon.cameras[image.camera_id].calibration_matrix(), dtype=np.float64)

            extrinsic = np.eye(4, dtype=np.float64)
            extrinsic[:3, :4] = image.cam_from_world().matrix()

            ids, xyz = [], []
            for p in image.points2D:
                if p.has_point3D():
                    ids.append(p.point3D_id)
                    xyz.append(recon.points3D[p.point3D_id].xyz)
            ids = np.asarray(ids, dtype=np.int64)
            xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
            if xyz.shape[0] < 2:
                logger.warning("View %d (%s) sees %d points; skipped", index, image.name, xyz.shape[0])
                continue

            z = np.sort((extrinsic[:3, :3] @ xyz.T + extrinsic[:3, 3:4])[2])
            depth_min = float(z[int(len(z) * _DEPTH_PERCENTILE)])
            depth_max = float(z[int(len(z) * (1.0 - _DEPTH_PERCENTILE))])
            if not depth_min > 0 or depth_max <= depth_min:
                logger.warning(
                    "View %d (%s) has a degenerate depth range [%.4f, %.4f]; skipped",
                    index, image.name, depth_min, depth_max,
                )
                continue

            depth_num = self._hypothesis_count(K, extrinsic, depth_min, depth_max)
            views.append(MvsView(
                index=index,
                image_name=image.name,
                K=K,
                extrinsic=extrinsic,
                depth_min=depth_min,
                # Raw step: the dataset applies `interval_scale` when it turns
                # this into hypotheses, so scaling here would cancel it out.
                depth_interval=(depth_max - depth_min) / (depth_num - 1),
                depth_num=depth_num,
                depth_max=depth_max,
                source_views=[],
            ))
            order = np.argsort(ids)
            self._points[image.name] = {"ids": ids[order], "xyz": xyz[order]}
        return views

    def _hypothesis_count(
        self, K: np.ndarray, extrinsic: np.ndarray, depth_min: float, depth_max: float
    ) -> int:
        """Depth hypotheses that keep one step under a pixel at the near plane.

        Steps are uniform in depth but the criterion is the disparity a
        one-pixel ray shift spans at `depth_min`, so a scene imaged at a longer
        focal length gets proportionally more hypotheses.
        """
        R, t = extrinsic[:3, :3], extrinsic[:3, 3]
        K_inv, R_inv = np.linalg.inv(K), np.linalg.inv(R)
        cx, cy = K[0, 2], K[1, 2]
        p_center = R_inv @ (K_inv @ np.array([cx, cy, 1.0]) * depth_min - t)
        p_shift = R_inv @ (K_inv @ np.array([cx + 1.0, cy, 1.0]) * depth_min - t)
        step = float(np.linalg.norm(p_shift - p_center))
        if step <= 0:
            return self.config.num_depth
        count = (1 / depth_min - 1 / depth_max) / (1 / depth_min - 1 / (depth_min + step))
        return max(2, int(round(count)))

    def _drop_outlier_views(self, views: list[MvsView]) -> list[MvsView]:
        spans = np.array([v.depth_max - v.depth_min for v in views])
        median = float(np.median(spans))
        if median <= 0:
            return views

        threshold = median * _OUTLIER_RANGE_RATIO
        kept, dropped = [], []
        for v in views:
            (kept if (v.depth_max - v.depth_min) >= threshold else dropped).append(v)
        if not dropped:
            return views

        for v in dropped:
            logger.info(
                "Dropping view %s: depth span %.6f below %.6f",
                v.image_name, v.depth_max - v.depth_min, threshold,
            )
            self._points.pop(v.image_name, None)
        logger.info("Kept %d/%d views after outlier filtering", len(kept), len(views))
        # Indices must stay contiguous: pair.txt and the cam file names index
        # into the surviving list, not into COLMAP's image ids.
        for new_index, view in enumerate(kept):
            view.index = new_index
        return kept

    def _select_source_views(self, views: list[MvsView]) -> None:
        cfg = self.config
        n = len(views)
        if n < 2:
            raise RuntimeError(f"Need at least 2 views for MVS, got {n}")

        centers = {
            v.index: -v.extrinsic[:3, :3].T @ v.extrinsic[:3, 3] for v in views
        }
        point_sets = {v.index: self._points[v.image_name] for v in views}

        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        score = np.zeros((n, n), dtype=np.float64)
        workers = mp.cpu_count()
        if n >= _PARALLEL_MIN_VIEWS and workers > 1:
            chunksize = max(1, len(pairs) // (workers * 4))
            with mp.Pool(
                processes=workers,
                initializer=_init_score_worker,
                initargs=(centers, point_sets, cfg.view_selection_theta0,
                          cfg.view_selection_sigma1, cfg.view_selection_sigma2),
            ) as pool:
                for i, j, s in pool.imap_unordered(_score_worker, pairs, chunksize=chunksize):
                    score[i, j] = score[j, i] = s
        else:
            for i, j in pairs:
                s = _view_score(
                    i, j, centers, point_sets, cfg.view_selection_theta0,
                    cfg.view_selection_sigma1, cfg.view_selection_sigma2,
                )
                score[i, j] = score[j, i] = s

        for view in views:
            ranked = np.argsort(score[view.index])[::-1][:_NUM_SOURCE_VIEWS]
            view.source_views = [(int(k), float(score[view.index, k])) for k in ranked]
        logger.info("Scored %d view pairs across %d views", len(pairs), n)

    def _write_scene(self, views: list[MvsView]) -> None:
        cfg = self.config
        if cfg.mvs_cams_dir.exists():
            for stale in cfg.mvs_cams_dir.glob("*_cam.txt"):
                stale.unlink()
        cfg.mvs_cams_dir.mkdir(parents=True, exist_ok=True)

        for view in views:
            lines = ["extrinsic"]
            lines += [" ".join(f"{v:.10f}" for v in row) for row in view.extrinsic]
            lines += ["", "intrinsic"]
            lines += [" ".join(f"{v:.10f}" for v in row) for row in view.K]
            lines += ["", f"{view.depth_min:.10f} {view.depth_interval:.10f} "
                          f"{view.depth_num:.10f} {view.depth_max:.10f}", ""]
            (cfg.mvs_cams_dir / f"{view.index:08d}_cam.txt").write_text("\n".join(lines))

        with cfg.mvs_pair_file.open("w") as f:
            f.write(f"{len(views)}\n")
            for view in views:
                f.write(f"{view.index}\n{len(view.source_views)} ")
                f.write(" ".join(f"{i} {s:f}" for i, s in view.source_views))
                f.write("\n")

        logger.info("MVSNet scene written: %d views → %s", len(views), cfg.mvs_dir)
