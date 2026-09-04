"""Point-cloud / mesh / segmentation visualization.

Used by QS-SAM3D. Produces:
  - Colored 3D PLYs (point clouds and meshes, segment palette or RGB-blended)
  - Per-frame colored 2D masks and RGB overlays
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import open3d as o3d
from PIL import Image

from utils.pointcloud import write_ply

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def generate_segment_colors(num_segments: int, seed: int) -> np.ndarray:
    """Golden-angle hue spacing with seeded S/V variation. (N,3) RGB in [0,1]."""
    rng = np.random.default_rng(seed)
    hsv = np.empty((num_segments, 3), dtype=np.float32)
    # 137.508° = 360°·(1-1/φ); golden-angle hue steps stay maximally separated
    # for any palette size without clustering, unlike uniform hue division.
    hsv[:, 0] = (np.arange(num_segments) * 137.508 % 360) / 360.0
    # Keep S ≥ 0.7 and V ≥ 0.8 so colors stay vivid/bright enough to read on
    # dark PLY backgrounds; jitter the remaining range so adjacent golden-angle
    # hues still look distinct at small marker sizes.
    hsv[:, 1] = 0.7 + 0.3 * rng.random(num_segments)
    hsv[:, 2] = 0.8 + 0.2 * rng.random(num_segments)
    return mcolors.hsv_to_rgb(hsv).astype(np.float32)


def visualize_partition(
    coords: np.ndarray,
    group_ids: np.ndarray,
    save_path: Path | str,
    seed: int,
) -> None:
    """Save a PLY where each group ID has a distinct color; -1 points are black."""
    group_ids = group_ids.reshape(-1).astype(np.int64)
    valid = group_ids[group_ids != -1]
    num_groups = int(valid.max()) + 1 if valid.size else 0

    palette = generate_segment_colors(num_groups, seed) if num_groups else np.zeros((0, 3))
    colors = np.zeros((group_ids.shape[0], 3), dtype=np.float32)
    mask = group_ids >= 0
    colors[mask] = palette[group_ids[mask]]
    write_ply(save_path, coords, colors)


def visualize_mesh_partition(
    mesh: o3d.geometry.TriangleMesh,
    group_ids: np.ndarray,
    save_path: Path | str,
    seed: int,
) -> None:
    """Same coloring rule as `visualize_partition`, but writes a mesh with faces."""
    group_ids = group_ids.reshape(-1).astype(np.int64)
    n_verts = len(mesh.vertices)
    if group_ids.shape[0] != n_verts:
        raise ValueError(
            f"group_ids length {group_ids.shape[0]} != mesh vertex count {n_verts}"
        )

    valid = group_ids[group_ids != -1]
    num_groups = int(valid.max()) + 1 if valid.size else 0
    palette = generate_segment_colors(num_groups, seed) if num_groups else np.zeros((0, 3))
    colors = np.zeros((n_verts, 3), dtype=np.float32)
    mask = group_ids >= 0
    colors[mask] = palette[group_ids[mask]]

    out = o3d.geometry.TriangleMesh()
    out.vertices = mesh.vertices
    out.triangles = mesh.triangles
    if mesh.has_vertex_normals():
        out.vertex_normals = mesh.vertex_normals
    out.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(save_path), out)


def visualize_mesh_segmentation(
    mesh: o3d.geometry.TriangleMesh,
    group_ids: np.ndarray,
    save_path: Path | str,
    alpha: float,
    beta: float,
    seed: int,
) -> None:
    """Mesh PLY: alpha*vertex_color + beta*segment_color blend; -1 keeps original RGB."""
    group_ids = group_ids.reshape(-1).astype(np.int64)
    n_verts = len(mesh.vertices)
    if group_ids.shape[0] != n_verts:
        raise ValueError(
            f"group_ids length {group_ids.shape[0]} != mesh vertex count {n_verts}"
        )
    if not mesh.has_vertex_colors():
        raise ValueError(f"Mesh has no vertex colors; cannot blend at {save_path}")
    rgb = np.asarray(mesh.vertex_colors, dtype=np.float32)  # [0, 1]

    blended = rgb.copy()
    valid = group_ids[group_ids != -1]
    if valid.size:
        palette = generate_segment_colors(int(valid.max()) + 1, seed)
        mask = group_ids >= 0
        blended[mask] = alpha * rgb[mask] + beta * palette[group_ids[mask]]
    np.clip(blended, 0.0, 1.0, out=blended)

    out = o3d.geometry.TriangleMesh()
    out.vertices = mesh.vertices
    out.triangles = mesh.triangles
    if mesh.has_vertex_normals():
        out.vertex_normals = mesh.vertex_normals
    out.vertex_colors = o3d.utility.Vector3dVector(blended.astype(np.float64))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(save_path), out)


def visualize_masks_2d(
    mask_dir: Path | str,
    rgb_dir: Path | str,
    output_dir: Path | str,
    alpha: float,
    beta: float,
    seed: int,
) -> int:
    """Render colored masks + RGB overlays for every `.png` mask in `mask_dir`.

    For each mask file `<stem>.png`, searches `rgb_dir` for `<stem>.{jpg,png}`.
    Returns the number of mask files processed.
    """
    mask_dir = Path(mask_dir)
    rgb_dir = Path(rgb_dir)
    out = Path(output_dir)

    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        logger.warning("No mask files in %s", mask_dir)
        return 0

    colored_dir = out / "colored_masks"
    overlay_dir = out / "overlay_results"
    colored_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    palette = (generate_segment_colors(256, seed) * 255).astype(np.uint8)

    for mask_file in mask_files:
        mask_img = np.array(Image.open(mask_file))
        rgb_img = _find_rgb(rgb_dir, mask_file.stem)
        if rgb_img is not None and rgb_img.shape[:2] != mask_img.shape:
            rgb_img = cv2.resize(rgb_img, (mask_img.shape[1], mask_img.shape[0]))

        colored = np.zeros((*mask_img.shape, 3), dtype=np.uint8)
        valid = mask_img >= 0
        if valid.any():
            colored[valid] = palette[mask_img[valid].astype(np.intp) % len(palette)]

        overlay = (
            cv2.addWeighted(rgb_img, alpha, colored, beta, 0)
            if rgb_img is not None
            else colored
        )

        cv2.imwrite(
            str(colored_dir / f"{mask_file.stem}_colored.png"),
            cv2.cvtColor(colored, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(overlay_dir / f"{mask_file.stem}_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

    logger.info("Rendered %d 2D mask overlays to %s", len(mask_files), out)
    return len(mask_files)


def _find_rgb(rgb_dir: Path, stem: str) -> np.ndarray | None:
    for ext in (".jpg", ".jpeg", ".png"):
        p = rgb_dir / f"{stem}{ext}"
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None
