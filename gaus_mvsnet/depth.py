"""Gaus-MVSNet depth inference stage.

Runs the cascade MVS backbone over a COLMAP reconstruction and writes the
depth products the rest of the pipeline consumes: a fused point cloud for
QS-SAM3D to segment, and one OpenMVS .dmap per view, which carries the same
depth to the mesh stage and to the QS-SAM3D converter.

Inference uses the backbone alone. The 3D Gaussian Splatting auxiliary branch
of the paper supervises training only and has no role here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import GausMvsNetConfig
from gaus_mvsnet.colmap_input import ColmapMvsInput, MvsView
from gaus_mvsnet.dataset import MvsInferenceDataset
from gaus_mvsnet.fusion import DepthFusion
from gaus_mvsnet.model.casmvsnet import CascadeMVSNet_eval
from utils.depth import write_dmap
from utils.gpu import gpu_cleanup
from utils.pointcloud import voxel_downsample, write_ply

logger = logging.getLogger(__name__)


class GausMvsNetDepth:
    """Depth estimation and fusion from a COLMAP reconstruction."""

    def __init__(self, config: GausMvsNetConfig) -> None:
        self.config = config

    def run(self) -> None:
        cfg = self.config
        checkpoint = Path(cfg.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Gaus-MVSNet checkpoint not found: {checkpoint}. "
                "See the README for where to download it."
            )
        if not cfg.colmap_images.is_dir():
            raise FileNotFoundError(f"COLMAP images not found: {cfg.colmap_images}")

        cfg.mvs_dir.mkdir(parents=True, exist_ok=True)

        scene = ColmapMvsInput(cfg)
        views = scene.build()
        dataset = MvsInferenceDataset(
            views=views,
            image_dir=cfg.colmap_images,
            num_view=cfg.num_view,
            num_depth=cfg.num_depth,
            interval_scale=cfg.interval_scale,
            max_h=cfg.max_h,
            max_w=cfg.max_w,
        )

        with gpu_cleanup():
            self._predict(dataset, checkpoint)

        points, colors = DepthFusion(cfg, views, dataset.geometry).run()
        self._write_pointcloud(points, colors)
        self._write_dmaps(views, dataset.geometry, scene.openmvs_ids)

        logger.info("Gaus-MVSNet depth complete: %s", cfg.mvs_dir)

    def _predict(self, dataset: MvsInferenceDataset, checkpoint: Path) -> None:
        cfg = self.config
        if not torch.cuda.is_available():
            raise RuntimeError("Gaus-MVSNet inference requires a CUDA GPU")

        model = CascadeMVSNet_eval(
            refine=False,
            ndepths=cfg.ndepths_list,
            depth_interals_ratio=cfg.depth_inter_r_list,
            share_cr=False,
            cr_base_chs=cfg.cr_base_chs_list,
        )
        model.load_state_dict(_read_state_dict(checkpoint), strict=True)
        model.cuda().eval()

        cfg.depth_dir.mkdir(parents=True, exist_ok=True)
        cfg.confidence_dir.mkdir(parents=True, exist_ok=True)

        # One sample per forward: the cost volume at inference resolution
        # dominates VRAM, and batching it buys nothing on a single card.
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        with torch.no_grad():
            for sample in tqdm(loader, desc="Estimating depth", unit="view"):
                outputs = model(
                    sample["imgs"].cuda(),
                    {k: v.cuda() for k, v in sample["proj_matrices"].items()},
                    sample["depth_values"].cuda(),
                )
                index = int(sample["index"][0])
                depth = outputs["depth"][0].cpu().numpy().astype(np.float32)
                confidence = (
                    outputs["photometric_confidence"][0].cpu().numpy().astype(np.float32)
                )
                np.save(cfg.depth_dir / f"{index:08d}.npy", depth)
                np.save(cfg.confidence_dir / f"{index:08d}.npy", confidence)

        del model

    def _write_pointcloud(self, points: np.ndarray, colors: np.ndarray) -> None:
        cfg = self.config
        raw = cfg.mvs_dir / "fused_pointcloud.ply"
        write_ply(raw, points, colors)
        logger.info("Fused %d points → %s", points.shape[0], raw)
        voxel_downsample(
            input_path=raw,
            output_path=cfg.final_pointcloud,
            target_count=cfg.downsample_target,
            max_iter=cfg.downsample_max_iter,
            tolerance=cfg.downsample_tolerance,
            batch_size=cfg.downsample_batch_size,
        )

    def _write_dmaps(
        self,
        views: list[MvsView],
        geometry: dict[int, dict],
        openmvs_ids: dict[str, int],
    ) -> None:
        """Export each view's depth in OpenMVS's .dmap format.

        The exported depth is the one fusion kept, not the raw network output,
        so the per-frame geometry QS-SAM3D back-projects and the mesh OpenMVS
        builds both agree with `final_pointcloud.ply`.

        OpenMVS indexes depth maps into the images it loaded itself, and it
        crashes rather than rescales when a map's size differs from the
        resolution it settled on, so maps go out at the source resolution and
        the mesh stage pins OpenMVS to that same size.
        """
        cfg = self.config
        if cfg.dmap_dir.exists():
            for stale in cfg.dmap_dir.glob("depth*.dmap"):
                stale.unlink()
        cfg.dmap_dir.mkdir(parents=True, exist_ok=True)

        by_index = {v.index: v for v in views}
        for view in tqdm(views, desc="Writing .dmap", unit="view"):
            filtered = cfg.filtered_depth_dir / f"{view.index:08d}.npy"
            if not filtered.is_file():
                logger.warning("View %d survived no fusion; .dmap skipped", view.index)
                continue
            depth = np.load(filtered)
            source_w, source_h = geometry[view.index]["source_size"]
            if depth.shape != (source_h, source_w):
                depth = cv2.resize(
                    depth, (source_w, source_h), interpolation=cv2.INTER_NEAREST
                )

            valid = depth[depth > 0]
            if valid.size == 0:
                logger.warning("View %d has no valid depth; .dmap skipped", view.index)
                continue

            neighbours = [
                openmvs_ids[by_index[i].image_name]
                for i, _ in view.source_views
                if i in by_index
            ]
            write_dmap(
                path=cfg.dmap_dir / f"depth{openmvs_ids[view.image_name]:04d}.dmap",
                depth=depth,
                # Resolved by OpenMVS against its working folder, which sits
                # beside the COLMAP output.
                image_name=f"../colmap/dense/images/{view.image_name}",
                image_id=openmvs_ids[view.image_name],
                neighbor_ids=neighbours,
                K=view.K,
                R=view.extrinsic[:3, :3],
                C=-view.extrinsic[:3, :3].T @ view.extrinsic[:3, 3],
                image_size=(source_w, source_h),
                d_min=float(valid.min()),
                d_max=float(valid.max()),
            )
        logger.info("Wrote %d depth maps → %s", len(views), cfg.dmap_dir)


def _read_state_dict(checkpoint: Path) -> dict:
    """Unwrap a training checkpoint, accepting a bare state dict as well."""
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


if __name__ == "__main__":
    from utils.cli import run_stage

    run_stage(
        __file__, "Gaus-MVSNet depth inference", GausMvsNetConfig.from_yaml, GausMvsNetDepth
    )
