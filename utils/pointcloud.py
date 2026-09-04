"""Point cloud I/O and GPU voxel-grid downsampling.

`voxel_downsample` reduces a PLY to a target point count via binary-searched
voxel size, with GPU batch hashing + aggregation. 21-bit-per-axis voxel hash
supports up to ~2M cells per axis before collision.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

logger = logging.getLogger(__name__)


def read_ply(path: Path | str) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a PLY. Returns (points Nx3 float32, colors Nx3 float32 or None)."""
    pcd = o3d.io.read_point_cloud(str(path))
    if not pcd.has_points():
        raise RuntimeError(f"No points in {path}")
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
    return points, colors


def write_ply(path: Path | str, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write points (and optional colors in [0,1]) to a binary PLY."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def voxel_downsample(
    input_path: Path | str,
    output_path: Path | str,
    target_count: int,
    max_iter: int = 30,
    tolerance: float = 0.01,
    batch_size: int = 20_000_000,
) -> None:
    """Voxel-downsample a PLY to `target_count` points (GPU+CPU hybrid).

    If the input already has fewer than `target_count` points, writes the input
    unchanged.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("voxel_downsample requires CUDA GPU")

    points_np, colors_np = read_ply(input_path)
    n = points_np.shape[0]
    logger.info("Loaded %d points from %s", n, input_path)

    if n <= target_count:
        logger.info("Already at/below target (%d ≤ %d); copying", n, target_count)
        write_ply(output_path, points_np, colors_np)
        return

    device = torch.device("cuda")
    points_gpu = torch.from_numpy(points_np).to(device)
    origin = points_gpu.min(dim=0, keepdim=True)[0]
    bbox_diag = torch.norm(points_gpu.max(dim=0)[0] - origin[0]).item()

    voxel_size = _binary_search_voxel_size(
        points_gpu, origin, target_count, bbox_diag, max_iter, tolerance, batch_size,
    )

    colors_gpu = (
        torch.from_numpy(colors_np).to(device) if colors_np is not None else None
    )
    points_out, colors_out = _aggregate(
        points_gpu, colors_gpu, origin, voxel_size, batch_size,
    )

    write_ply(
        output_path,
        points_out.numpy(),
        colors_out.numpy() if colors_out is not None else None,
    )
    logger.info("Wrote %d points to %s", points_out.shape[0], output_path)

    del points_gpu, origin
    if colors_gpu is not None:
        del colors_gpu
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Internal: voxel hashing + aggregation
# ---------------------------------------------------------------------------


def _voxel_hash(points: torch.Tensor, origin: torch.Tensor, voxel_size: float) -> torch.Tensor:
    vi = torch.floor((points - origin) / voxel_size).long()
    return (
        ((vi[:, 0] & 0x1FFFFF) << 42)
        | ((vi[:, 1] & 0x1FFFFF) << 21)
        | (vi[:, 2] & 0x1FFFFF)
    )


def _count_voxels(
    points_gpu: torch.Tensor, origin: torch.Tensor, voxel_size: float, batch_size: int,
) -> int:
    parts = []
    for i in range(0, points_gpu.shape[0], batch_size):
        h = _voxel_hash(points_gpu[i:i + batch_size], origin, voxel_size)
        parts.append(torch.unique(h).cpu())
    return torch.unique(torch.cat(parts)).shape[0]


def _binary_search_voxel_size(
    points_gpu: torch.Tensor,
    origin: torch.Tensor,
    target: int,
    bbox_diag: float,
    max_iter: int,
    tol: float,
    batch_size: int,
) -> float:
    lo, hi = 1e-6, bbox_diag / 10.0
    best = hi
    for i in range(max_iter):
        mid = (lo + hi) / 2.0
        if mid < 1e-7:
            break
        count = _count_voxels(points_gpu, origin, mid, batch_size)
        logger.info("  [iter %d/%d] voxel=%.6f -> count=%d", i + 1, max_iter, mid, count)
        if abs(count - target) / target < tol:
            return mid
        if count > target:
            lo = mid
        else:
            hi = mid
        best = mid
        torch.cuda.empty_cache()
    return best


def _aggregate(
    points_gpu: torch.Tensor,
    colors_gpu: torch.Tensor | None,
    origin: torch.Tensor,
    voxel_size: float,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Batch GPU per-voxel sum/count, then CPU merge across batches."""
    device = points_gpu.device
    n = points_gpu.shape[0]
    has_colors = colors_gpu is not None

    hashes_parts: list[torch.Tensor] = []
    pc_parts: list[torch.Tensor] = []
    color_parts: list[torch.Tensor] = []

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        pts = points_gpu[i:end]
        h = _voxel_hash(pts, origin, voxel_size)
        uniq, inv = torch.unique(h, return_inverse=True)
        m = uniq.shape[0]
        inv3 = inv.unsqueeze(1).expand(-1, 3)

        sums = torch.zeros((m, 3), dtype=torch.float32, device=device)
        sums.scatter_add_(0, inv3, pts)
        cnts = torch.zeros(m, dtype=torch.float32, device=device)
        cnts.scatter_add_(0, inv, torch.ones_like(inv, dtype=torch.float32))

        hashes_parts.append(uniq.cpu())
        pc_parts.append(torch.cat([sums, cnts.unsqueeze(1)], dim=1).cpu())

        if has_colors:
            cs = torch.zeros((m, 3), dtype=torch.float32, device=device)
            cs.scatter_add_(0, inv3, colors_gpu[i:end])
            color_parts.append(cs.cpu())

        torch.cuda.empty_cache()

    all_h = torch.cat(hashes_parts)
    all_pc = torch.cat(pc_parts)
    all_col = torch.cat(color_parts) if has_colors else None

    order = torch.argsort(all_h)
    sh = all_h[order]
    spc = all_pc[order]
    scol = all_col[order] if all_col is not None else None

    _, inv = torch.unique_consecutive(sh, return_inverse=True)
    m = int(inv.max().item()) + 1
    inv3 = inv.unsqueeze(1).expand(-1, 3)

    sums = torch.zeros((m, 3), dtype=torch.float32)
    sums.scatter_add_(0, inv3, spc[:, :3])
    cnts = torch.zeros(m, dtype=torch.float32)
    cnts.scatter_add_(0, inv, spc[:, 3])
    points_out = sums / cnts.unsqueeze(1)

    colors_out = None
    if scol is not None:
        cs = torch.zeros((m, 3), dtype=torch.float32)
        cs.scatter_add_(0, inv3, scol)
        colors_out = cs / cnts.unsqueeze(1)

    return points_out, colors_out
