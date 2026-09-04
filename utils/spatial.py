"""Spatial utilities: voxelization, GPU KNN matching, group-label helpers.

Used by QS-SAM3D segmentation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import open3d as o3d
import pointops
import torch


# ---------------------------------------------------------------------------
# Voxelize — train-mode voxel downsampling with FNV hashing
# ---------------------------------------------------------------------------


class Voxelize:
    """Downsample a point-cloud dict to one point per voxel.

    `data_dict` must contain `"coord"` (N,3). Only fields listed in `keys` are
    subsampled; other entries pass through unchanged. `"coord"` is always
    subsampled, regardless of `keys`.
    """

    def __init__(self, voxel_size: float, keys: Sequence[str]) -> None:
        self.voxel_size = float(voxel_size)
        self.keys = tuple(keys)

    def __call__(self, data_dict: dict) -> dict:
        coord = data_dict["coord"]
        discrete = np.floor(coord / self.voxel_size).astype(np.int64)
        discrete -= discrete.min(axis=0)
        keys_hash = _fnv_hash(discrete.astype(np.uint64))

        idx_sort = np.argsort(keys_hash)
        _, count = np.unique(keys_hash[idx_sort], return_counts=True)
        idx_unique = idx_sort[np.cumsum(np.insert(count, 0, 0)[:-1])]

        out = dict(data_dict)
        for k in ("coord", *self.keys):
            if k in data_dict:
                out[k] = data_dict[k][idx_unique]
        return out


def _fnv_hash(arr: np.ndarray) -> np.ndarray:
    """FNV64-1A hash per row."""
    h = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        h *= np.uint64(1099511628211)
        h = np.bitwise_xor(h, arr[:, j])
    return h


# ---------------------------------------------------------------------------
# GPU KNN matching
# ---------------------------------------------------------------------------

# pointops.knn_query is brute O(N*M) on GPU — blocked 27-neighbor pruning wins
# quickly once combined point count grows. 50K matches the legacy threshold that
# was validated against full-scale DTU/ScanNet merges.
_BLOCKED_THRESHOLD = 50_000
_BLOCK_TARGET_POINTS = 100_000


def get_matching_indices(
    pcd_query: o3d.geometry.PointCloud,
    pcd_target: o3d.geometry.PointCloud,
    search_radius: float,
) -> list[list[int]]:
    """For each point in `pcd_query`, find its nearest neighbor in `pcd_target`
    within `search_radius`. Returns `[[i_query, j_target], ...]`."""
    points0 = np.asarray(pcd_query.points, dtype=np.float64)
    points1 = np.asarray(pcd_target.points, dtype=np.float64)
    if points0.size == 0 or points1.size == 0:
        return []

    if points0.shape[0] + points1.shape[0] < _BLOCKED_THRESHOLD:
        return _match_brute_gpu(points0, points1, search_radius)
    return _match_blocked_gpu(points0, points1, search_radius)


def _match_brute_gpu(points0: np.ndarray, points1: np.ndarray, r: float) -> list[list[int]]:
    with torch.no_grad():
        q = torch.from_numpy(points0).cuda().contiguous().float()
        t = torch.from_numpy(points1).cuda().contiguous().float()
        q_off = torch.tensor([q.shape[0]], device="cuda", dtype=torch.int32)
        t_off = torch.tensor([t.shape[0]], device="cuda", dtype=torch.int32)
        idx, dis = pointops.knn_query(1, t, t_off, q, q_off)  # type: ignore[misc]
    idx_np = idx.cpu().numpy()[:, 0]
    dis_np = dis.cpu().numpy()[:, 0]
    keep = np.where(dis_np < r)[0]
    return [[int(i), int(idx_np[i])] for i in keep]


def _match_blocked_gpu(points0: np.ndarray, points1: np.ndarray, r: float) -> list[list[int]]:
    """Spatial-block GPU KNN — each query only searches its 27-neighbor blocks."""
    bbox = points1.max(axis=0) - points1.min(axis=0) + 1e-6
    surface = 2 * (bbox[0] * bbox[1] + bbox[1] * bbox[2] + bbox[0] * bbox[2])
    density = len(points1) / surface
    block = max((_BLOCK_TARGET_POINTS / density) ** 0.5, 2 * r)

    b1 = np.floor(points1 / block).astype(np.int32)
    target_blocks: dict[tuple, list[int]] = {}
    for i, key in enumerate(map(tuple, b1)):
        target_blocks.setdefault(key, []).append(i)

    b0 = np.floor(points0 / block).astype(np.int32)
    query_blocks: dict[tuple, list[int]] = {}
    for i, key in enumerate(map(tuple, b0)):
        query_blocks.setdefault(key, []).append(i)

    matches: list[list[int]] = []
    for key, q_idx_list in query_blocks.items():
        candidates: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    candidates.extend(
                        target_blocks.get((key[0] + dx, key[1] + dy, key[2] + dz), [])
                    )
        if not candidates:
            continue
        q_idx = np.asarray(q_idx_list, dtype=np.int64)
        c_idx = np.asarray(candidates, dtype=np.int64)

        with torch.no_grad():
            q_pts = torch.from_numpy(points0[q_idx]).cuda().contiguous().float()
            c_pts = torch.from_numpy(points1[c_idx]).cuda().contiguous().float()
            q_off = torch.tensor([q_pts.shape[0]], device="cuda", dtype=torch.int32)
            c_off = torch.tensor([c_pts.shape[0]], device="cuda", dtype=torch.int32)
            idx, dis = pointops.knn_query(1, c_pts, c_off, q_pts, q_off)  # type: ignore[misc]
        idx_np = idx.cpu().numpy()[:, 0]
        dis_np = dis.cpu().numpy()[:, 0]

        for i in np.where(dis_np < r)[0]:
            matches.append([int(q_idx[i]), int(c_idx[idx_np[i]])])
    return matches


# ---------------------------------------------------------------------------
# Group-label helpers
# ---------------------------------------------------------------------------


def num_to_natural(group_ids: np.ndarray) -> np.ndarray:
    """Remap non-negative IDs to contiguous `[0, N)`; preserves -1."""
    g = group_ids.copy()
    if np.all(g == -1):
        return g
    unique = np.unique(g[g != -1])
    lut = np.full(int(unique.max()) + 2, -1, dtype=np.int64)
    lut[unique + 1] = np.arange(len(unique))
    return lut[g + 1]


def remove_small_group(group_ids: np.ndarray, min_size: int) -> np.ndarray:
    """Set group IDs to -1 where the group contains fewer than `min_size` points."""
    out = group_ids.copy()
    unique, counts = np.unique(out, return_counts=True)
    for uid, c in zip(unique, counts):
        if c < min_size:
            out[out == uid] = -1
    return out


def pairwise_indices(length: int) -> list[list[int]]:
    """Generate adjacent pairs [[0,1], [2,3], ...]; last is singleton if odd."""
    return [[i, i + 1] if i + 1 < length else [i] for i in range(0, length, 2)]


def make_open3d_pcd(coord: np.ndarray) -> o3d.geometry.PointCloud | None:
    """Wrap (N,3) coords in an Open3D PointCloud. Returns None if empty or NaN."""
    if coord.shape[0] == 0 or np.isnan(coord).any():
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    return pcd
