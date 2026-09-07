"""OpenMVS .dmap depth-map reader.

Reference: OpenMVS 2.4.0 libs/MVS/DepthMap.cpp ExportDepthDataRaw.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np


_DMAP_MAGIC = 0x5244  # 'D','R' little-endian
_HEADER_FMT = "<HBBIIIIff"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 28


def read_dmap(path: Path | str) -> dict[str, Any]:
    """Read an OpenMVS .dmap file. Returns a dict with depth, camera, image metadata.

    Keys:
      depth: (depth_h, depth_w) float32, meters (0 = invalid)
      image_width, image_height: int — source (undistorted) image resolution
      depth_width, depth_height: int
      image_name: str — relative path to source image (from OpenMVS workdir)
      image_id: int — MVS reference image ID
      neighbor_ids: list[int]
      K: (3,3) float64 — intrinsic at image_width/height
      R: (3,3) float64 — world-to-camera rotation
      C: (3,) float64 — camera center in world
      d_min, d_max: float
    """
    path = Path(path)
    with path.open("rb") as f:
        name, type_, _pad, iw, ih, dw, dh, d_min, d_max = struct.unpack(
            _HEADER_FMT, f.read(_HEADER_SIZE)
        )
        if name != _DMAP_MAGIC:
            raise ValueError(f"{path}: not a .dmap (magic {name:#06x} != {_DMAP_MAGIC:#06x})")
        if not type_ & 0x1:
            raise ValueError(f"{path}: HAS_DEPTH flag not set in type={type_:#x}")

        (n_fn,) = struct.unpack("<H", f.read(2))
        image_name = f.read(n_fn).decode("utf-8", errors="replace")
        (n_ids,) = struct.unpack("<I", f.read(4))
        ids = np.frombuffer(f.read(4 * n_ids), dtype=np.uint32)
        K = np.frombuffer(f.read(8 * 9), dtype=np.float64).reshape(3, 3).copy()
        R = np.frombuffer(f.read(8 * 9), dtype=np.float64).reshape(3, 3).copy()
        C = np.frombuffer(f.read(8 * 3), dtype=np.float64).copy()
        depth_bytes = f.read(4 * dw * dh)
        if len(depth_bytes) != 4 * dw * dh:
            raise ValueError(
                f"{path}: truncated depth (expected {4 * dw * dh} bytes, got {len(depth_bytes)})"
            )
        depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(dh, dw).copy()

    return dict(
        depth=depth,
        image_width=int(iw), image_height=int(ih),
        depth_width=int(dw), depth_height=int(dh),
        image_name=image_name,
        image_id=int(ids[0]) if len(ids) else -1,
        neighbor_ids=[int(x) for x in ids[1:]],
        K=K, R=R, C=C,
        d_min=float(d_min), d_max=float(d_max),
    )


def pose_camera_to_world(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """OpenMVS (R, C) → 4x4 camera-to-world matrix.

    OpenMVS stores R as world-to-camera rotation and C as camera center in world.
    Camera-to-world is `[R^T | C; 0 | 1]`.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.T
    T[:3, 3] = C
    return T


def write_dmap(
    path: Path | str,
    depth: np.ndarray,
    image_name: str,
    image_id: int,
    neighbor_ids: list[int],
    K: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    image_size: tuple[int, int],
    d_min: float,
    d_max: float,
) -> None:
    """Write an OpenMVS .dmap file, inverse of `read_dmap`.

    `image_size` is (width, height) of the image the depth was estimated on,
    and `K` must be the intrinsic at that resolution. Only HAS_DEPTH is set:
    OpenMVS fuses depth-only maps, and the normal/confidence planes it would
    otherwise read are not produced by an external estimator.
    """
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    dh, dw = depth.shape
    iw, ih = image_size
    encoded = image_name.encode("utf-8")
    ids = np.asarray([image_id, *neighbor_ids], dtype=np.uint32)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack(
            _HEADER_FMT, _DMAP_MAGIC, 0x1, 0, iw, ih, dw, dh, d_min, d_max
        ))
        f.write(struct.pack("<H", len(encoded)))
        f.write(encoded)
        f.write(struct.pack("<I", len(ids)))
        f.write(ids.tobytes())
        f.write(np.ascontiguousarray(K, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(R, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(C, dtype=np.float64).tobytes())
        f.write(depth.tobytes())
