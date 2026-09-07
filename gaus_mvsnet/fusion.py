"""Depth-map filtering and fusion into a single colored point cloud.

Implements the fusion described in the paper: a pixel survives when its
photometric confidence clears `tau_c` and at least `num_consistent` source
views agree with it under forward-backward reprojection (`tau_p` px) and
relative depth deviation (`tau_delta`). Surviving depths are averaged over the
views that agreed before being back-projected.

The surviving depth is also written back per view, because it — not the raw
network output — is what produced the point cloud, and the .dmap files that
carry depth to the mesh and segmentation stages have to agree with it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from config import GausMvsNetConfig
from gaus_mvsnet.colmap_input import MvsView

logger = logging.getLogger(__name__)


def _reproject(
    depth_ref: np.ndarray,
    K_ref: np.ndarray,
    extrinsic_ref: np.ndarray,
    depth_src: np.ndarray,
    K_src: np.ndarray,
    extrinsic_src: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Round-trip reference pixels through the source view.

    Returns the depth the round trip lands on and where it lands, both in the
    reference frame.
    """
    h, w = depth_ref.shape
    x_ref, y_ref = np.meshgrid(np.arange(w), np.arange(h))
    x_ref, y_ref = x_ref.reshape(-1), y_ref.reshape(-1)
    ones = np.ones_like(x_ref)

    xyz_ref = np.linalg.inv(K_ref) @ (np.vstack((x_ref, y_ref, ones)) * depth_ref.reshape(-1))
    xyz_src = (extrinsic_src @ np.linalg.inv(extrinsic_ref) @ np.vstack((xyz_ref, ones)))[:3]
    k_xyz_src = K_src @ xyz_src
    xy_src = k_xyz_src[:2] / k_xyz_src[2:3]

    sampled = cv2.remap(
        depth_src,
        xy_src[0].reshape(h, w).astype(np.float32),
        xy_src[1].reshape(h, w).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
    )

    xyz_back = np.linalg.inv(K_src) @ (np.vstack((xy_src, ones)) * sampled.reshape(-1))
    xyz_reprojected = (
        extrinsic_ref @ np.linalg.inv(extrinsic_src) @ np.vstack((xyz_back, ones))
    )[:3]
    k_xyz_reprojected = K_ref @ xyz_reprojected
    xy_reprojected = k_xyz_reprojected[:2] / k_xyz_reprojected[2:3]

    return (
        xyz_reprojected[2].reshape(h, w).astype(np.float32),
        xy_reprojected[0].reshape(h, w).astype(np.float32),
        xy_reprojected[1].reshape(h, w).astype(np.float32),
    )


def _consistency(
    depth_ref: np.ndarray,
    K_ref: np.ndarray,
    extrinsic_ref: np.ndarray,
    depth_src: np.ndarray,
    K_src: np.ndarray,
    extrinsic_src: np.ndarray,
    reproj_threshold: float,
    depth_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth_ref.shape
    x_ref, y_ref = np.meshgrid(np.arange(w), np.arange(h))

    depth_reprojected, x_reprojected, y_reprojected = _reproject(
        depth_ref, K_ref, extrinsic_ref, depth_src, K_src, extrinsic_src
    )
    distance = np.sqrt((x_reprojected - x_ref) ** 2 + (y_reprojected - y_ref) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.abs(depth_reprojected - depth_ref) / depth_ref
    mask = (distance < reproj_threshold) & (relative < depth_threshold)
    depth_reprojected[~mask] = 0
    return mask, depth_reprojected


class DepthFusion:
    """Fuses per-view depth maps into one point cloud."""

    def __init__(
        self,
        config: GausMvsNetConfig,
        views: list[MvsView],
        geometry: dict[int, dict],
    ) -> None:
        self.config = config
        self.views = {v.index: v for v in views}
        self.geometry = geometry

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        self.config.filtered_depth_dir.mkdir(parents=True, exist_ok=True)
        points, colors = [], []
        for view in tqdm(list(self.views.values()), desc="Fusing depth maps", unit="view"):
            result = self._fuse_view(view)
            if result is None:
                continue
            points.append(result[0])
            colors.append(result[1])

        if not points:
            raise RuntimeError("No pixel survived depth fusion; check the filter thresholds")

        return np.concatenate(points, axis=0), np.concatenate(colors, axis=0)

    def _fuse_view(self, view: MvsView) -> tuple[np.ndarray, np.ndarray] | None:
        cfg = self.config
        depth_ref = self._load(cfg.depth_dir, view.index)
        confidence = self._load(cfg.confidence_dir, view.index)
        K_ref = self.geometry[view.index]["K"]

        photo_mask = confidence > cfg.prob_threshold
        agreed = np.zeros_like(depth_ref, dtype=np.int32)
        depth_sum = depth_ref.copy()

        for source_index, _ in view.source_views:
            source = self.views.get(source_index)
            if source is None:
                continue
            mask, depth_reprojected = _consistency(
                depth_ref, K_ref, view.extrinsic,
                self._load(cfg.depth_dir, source_index),
                self.geometry[source_index]["K"], source.extrinsic,
                cfg.reproj_threshold, cfg.depth_threshold,
            )
            agreed += mask.astype(np.int32)
            depth_sum += depth_reprojected

        geo_mask = agreed >= cfg.num_consistent
        final_mask = photo_mask & geo_mask
        logger.debug(
            "View %d: photo %.3f, geo %.3f, kept %.3f",
            view.index, photo_mask.mean(), geo_mask.mean(), final_mask.mean(),
        )
        if not final_mask.any():
            logger.warning("View %d contributed no points", view.index)
            return None

        depth_avg = depth_sum / (agreed + 1)
        np.save(
            cfg.filtered_depth_dir / f"{view.index:08d}.npy",
            np.where(final_mask, depth_avg, 0.0).astype(np.float32),
        )

        # Integer pixel indices against a principal point in COLMAP's
        # convention (centre at i+0.5) put every point half a pixel off the
        # ray it belongs to. Kept deliberately: it is the convention the
        # MVSNet-family code this backbone comes from uses throughout, so the
        # cloud stays comparable with the published numbers. At the scenes
        # here the bias is ~0.05% of scene extent, far under the distance
        # threshold that later assigns labels.
        height, width = depth_ref.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys, zs = x[final_mask], y[final_mask], depth_avg[final_mask]

        xyz_camera = np.linalg.inv(K_ref) @ (np.vstack((xs, ys, np.ones_like(xs))) * zs)
        xyz_world = (
            np.linalg.inv(view.extrinsic) @ np.vstack((xyz_camera, np.ones_like(xs)))
        )[:3]

        color = self._reference_color(view, width, height)[final_mask]
        return xyz_world.T.astype(np.float32), color.astype(np.float32)

    def _reference_color(self, view: MvsView, width: int, height: int) -> np.ndarray:
        path = self.config.colmap_images / view.image_name
        image = np.asarray(Image.open(path), dtype=np.float32) / 255.0
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return image

    @staticmethod
    def _load(directory: Path, index: int) -> np.ndarray:
        path = directory / f"{index:08d}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing depth-stage output: {path}")
        return np.load(path)
