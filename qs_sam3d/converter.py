"""COLMAP + OpenMVS → QS-SAM3D format converter."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image

from config import QsSam3dConfig
from utils.depth import pose_camera_to_world, read_dmap

logger = logging.getLogger(__name__)


def _filter_depth_by_scene(
    depth: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    scene_points: np.ndarray,
    tau: float,
) -> np.ndarray:
    dh, dw = depth.shape
    P_cam = (scene_points - C) @ R.T
    z = P_cam[:, 2]
    front = z > 0
    P_cam = P_cam[front]
    z = z[front]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = np.round(fx * P_cam[:, 0] / z + cx).astype(np.int64)
    v = np.round(fy * P_cam[:, 1] / z + cy).astype(np.int64)
    in_bounds = (u >= 0) & (u < dw) & (v >= 0) & (v < dh)
    u, v, z = u[in_bounds], v[in_bounds], z[in_bounds]
    Z_ref = np.zeros((dh, dw), dtype=np.float32)
    # sort by -z so farther points write first and are overwritten by nearer ones
    order = np.argsort(-z)
    Z_ref[v[order], u[order]] = z[order].astype(np.float32)
    out = depth.copy()
    bad = (Z_ref == 0) | (np.abs(out - Z_ref) > tau * Z_ref)
    out[bad] = 0.0
    return out


class QsSam3dConverter:
    """Build the SAM3D scene directory from COLMAP + OpenMVS outputs."""

    def __init__(self, config: QsSam3dConfig) -> None:
        self.config = config

    def run(self) -> None:
        cfg = self.config
        if not cfg.openmvs_dir.is_dir():
            raise FileNotFoundError(f"OpenMVS output not found: {cfg.openmvs_dir}")
        if not cfg.final_pointcloud.is_file():
            raise FileNotFoundError(f"Final point cloud not found: {cfg.final_pointcloud}")

        dmap_files = sorted(cfg.openmvs_dir.glob("depth*.dmap"))
        if not dmap_files:
            raise RuntimeError(f"No .dmap files in {cfg.openmvs_dir}")

        scene_ply = cfg.openmvs_dir / "scene_dense.ply"
        if not scene_ply.is_file():
            raise FileNotFoundError(f"Fused scene point cloud not found: {scene_ply}")
        scene_points = np.asarray(
            o3d.io.read_point_cloud(str(scene_ply)).points, dtype=np.float64
        )

        if cfg.sam3d_dir.exists():
            shutil.rmtree(cfg.sam3d_dir)
        cfg.sam3d_dir.mkdir(parents=True)
        self._init_output()
        k_matrices = [
            self._convert_frame(idx, p, scene_points) for idx, p in enumerate(dmap_files)
        ]
        mean_K = np.mean(np.stack(k_matrices, axis=0), axis=0)
        np.savetxt(cfg.sam3d_intrinsic_global, mean_K, fmt="%.6f")
        self._convert_pointcloud()

        logger.info("Converter complete: %d frames → %s", len(dmap_files), cfg.sam3d_dir)

    def _init_output(self) -> None:
        cfg = self.config
        dirs = (
            cfg.sam3d_color_dir,
            cfg.sam3d_depth_dir,
            cfg.sam3d_pose_dir,
            cfg.sam3d_intrinsics_dir,
        )
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _convert_frame(
        self, idx: int, dmap_path: Path, scene_points: np.ndarray
    ) -> np.ndarray:
        cfg = self.config
        dm = read_dmap(dmap_path)
        iw, ih = dm["image_width"], dm["image_height"]
        dw, dh = dm["depth_width"], dm["depth_height"]

        src = self._resolve_image(dm["image_name"])
        shutil.copy2(src, cfg.sam3d_color_dir / f"{idx}.jpg")

        # OpenMVS writes pre-fusion .dmap values; enforce cross-view consistency
        # against scene_dense.ply using dm["K"] at dmap-native resolution.
        depth = _filter_depth_by_scene(
            dm["depth"], dm["K"], dm["R"], dm["C"],
            scene_points, cfg.fusion_consistency_threshold,
        )
        if (dw, dh) != (iw, ih):
            # NEAREST preserves the zero-means-invalid sentinel
            depth = cv2.resize(depth, (iw, ih), interpolation=cv2.INTER_NEAREST)
        depth_mm = np.zeros(depth.shape, dtype=np.uint16)
        valid = (depth > 0) & np.isfinite(depth)
        depth_mm[valid] = (depth[valid] * cfg.depth_shift).astype(np.uint16)
        cv2.imwrite(str(cfg.sam3d_depth_dir / f"{idx}.png"), depth_mm)

        T_c2w = pose_camera_to_world(dm["R"], dm["C"])
        np.savetxt(cfg.sam3d_pose_dir / f"{idx}.txt", T_c2w, fmt="%.18e")

        # dm["K"] applies at (iw, ih) but the copied color image is at its source
        # resolution; scale K so it matches the saved color image, which is the
        # invariant downstream SAM back-projection relies on.
        with Image.open(src) as im:
            src_w, src_h = im.size
        sx, sy = src_w / iw, src_h / ih
        K = dm["K"].copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        K4 = np.eye(4, dtype=np.float64)
        K4[:3, :3] = K
        np.savetxt(cfg.sam3d_intrinsics_dir / f"{idx}.txt", K4, fmt="%.6f")
        return K4

    def _resolve_image(self, image_name: str) -> Path:
        """OpenMVS paths are relative to its workdir; fall back to COLMAP dense/images."""
        p = Path(image_name)
        if not p.is_absolute():
            p = (self.config.openmvs_dir / p).resolve()
        if p.is_file():
            return p
        fallback = self.config.colmap_images / Path(image_name).name
        if fallback.is_file():
            return fallback
        raise FileNotFoundError(f"Image from .dmap not found: tried {p} and {fallback}")

    def _convert_pointcloud(self) -> None:
        pcd = o3d.io.read_point_cloud(str(self.config.final_pointcloud))
        if not pcd.has_points():
            raise RuntimeError(f"Empty point cloud: {self.config.final_pointcloud}")

        coords = np.asarray(pcd.points, dtype=np.float32)
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        else:
            colors = np.full((coords.shape[0], 3), 255, dtype=np.uint8)

        torch.save(
            {"coord": coords, "color": colors},
            self.config.sam3d_pointcloud,
            pickle_protocol=4,
        )
        logger.info(
            "Saved pointcloud.pth: %d points → %s",
            coords.shape[0], self.config.sam3d_pointcloud,
        )


if __name__ == "__main__":
    from utils.cli import run_stage

    run_stage(__file__, "QS-SAM3D format converter", QsSam3dConfig.from_yaml, QsSam3dConverter)
