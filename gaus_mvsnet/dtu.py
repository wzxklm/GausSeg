"""DTU training samples in the MVSNet preprocessing layout.

Each sample is one reference view, its first `num_view - 1` source views from
`Cameras/pair.txt`, and one lighting condition, at 640x512. The ground-truth
depth and mask of the reference view are loaded for the logged accuracy
metrics only; no loss reads them.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_NUM_LIGHTS = 7
# Rectified images are 640x512; the raw 1600x1200 depth maps are halved and
# centre-cropped to match.
_CROP_H, _CROP_W = 512, 640


class DtuTrainDataset(Dataset):
    """Reference/source samples for unsupervised training."""

    def __init__(
        self,
        dtu_root: Path,
        depths_raw: Path,
        train_list: Path,
        num_view: int,
        num_depth: int,
        interval_scale: float,
    ) -> None:
        self.dtu_root = Path(dtu_root)
        self.depths_raw = Path(depths_raw)
        self.num_view = num_view
        self.num_depth = num_depth
        self.interval_scale = interval_scale
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )

        scans = [
            line.strip() for line in Path(train_list).read_text().splitlines() if line.strip()
        ]
        pairs = _read_pairs(self.dtu_root / "Cameras" / "pair.txt")
        self.samples = [
            (scan, light, ref, src)
            for scan in scans
            for ref, src in pairs
            for light in range(_NUM_LIGHTS)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        scan, light, ref_view, src_views = self.samples[idx]
        view_ids = [ref_view] + src_views[: self.num_view - 1]

        imgs, centered, projections = [], [], []
        for i, vid in enumerate(view_ids):
            # Image files number views from 1, camera and depth files from 0.
            image = Image.open(
                self.dtu_root
                / "Rectified"
                / f"{scan}_train"
                / f"rect_{vid + 1:03d}_{light}_r5000.png"
            ).convert("RGB")
            imgs.append(self.normalize(image).numpy())
            rgb = np.asarray(image, dtype=np.float32) / 255.0
            centered.append(_standardize(rgb).transpose(2, 0, 1))

            intrinsics, extrinsics, depth_min, depth_interval = self._read_cam(
                self.dtu_root / "Cameras" / "train" / f"{vid:08d}_cam.txt"
            )
            projection = np.zeros((2, 4, 4), dtype=np.float32)
            projection[0] = extrinsics
            projection[1, :3, :3] = intrinsics
            projections.append(projection)

            if i == 0:
                ref_rgb = rgb.transpose(2, 0, 1)
                depth_values = np.arange(
                    depth_min,
                    depth_min + depth_interval * self.num_depth,
                    depth_interval,
                    dtype=np.float32,
                )
                depth = _read_pfm(self.depths_raw / scan / f"depth_map_{vid:04d}.pfm")
                mask = np.asarray(
                    Image.open(self.depths_raw / scan / f"depth_visual_{vid:04d}.png"), np.float32
                )

        # Camera files carry intrinsics for the 1/4-resolution stage.
        stage1 = np.stack(projections)
        stage2 = stage1.copy()
        stage2[:, 1, :2, :] *= 2
        stage3 = stage1.copy()
        stage3[:, 1, :2, :] *= 4

        return {
            "imgs": np.stack(imgs),
            "center_imgs": np.stack(centered),
            "ref_rgb": ref_rgb,
            "proj_matrices": {"stage1": stage1, "stage2": stage2, "stage3": stage3},
            "depth_values": depth_values,
            "depth": _multiscale(_crop_to_images(depth)),
            "mask": _multiscale(_crop_to_images((mask > 10).astype(np.float32))),
        }

    def _read_cam(self, path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
        lines = path.read_text().splitlines()
        extrinsics = np.array(" ".join(lines[1:5]).split(), dtype=np.float32).reshape(4, 4)
        intrinsics = np.array(" ".join(lines[7:10]).split(), dtype=np.float32).reshape(3, 3)
        depth_min, depth_interval = (float(x) for x in lines[11].split()[:2])
        return intrinsics, extrinsics, depth_min, depth_interval * self.interval_scale


def _read_pairs(path: Path) -> list[tuple[int, list[int]]]:
    lines = path.read_text().splitlines()
    pairs = []
    for k in range(int(lines[0])):
        ref = int(lines[1 + 2 * k])
        # "<count> <id> <score> <id> <score> ..."
        src = [int(x) for x in lines[2 + 2 * k].split()[1::2]]
        pairs.append((ref, src))
    return pairs


def _standardize(img: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance per channel: the photometric loss compares these."""
    mean = img.mean(axis=(0, 1), keepdims=True)
    std = np.sqrt(img.var(axis=(0, 1), keepdims=True))
    return (img - mean) / (std + 1e-8)


def _crop_to_images(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
    h, w = img.shape
    top, left = (h - _CROP_H) // 2, (w - _CROP_W) // 2
    return img[top : top + _CROP_H, left : left + _CROP_W]


def _multiscale(img: np.ndarray) -> dict[str, np.ndarray]:
    h, w = img.shape
    return {
        "stage1": cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
        "stage2": cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
        "stage3": img,
    }


def _read_pfm(path: Path) -> np.ndarray:
    """Single-channel PFM -> float32 [H, W], top row first."""
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        if header != "Pf":
            raise ValueError(f"{path}: expected a single-channel PFM ('Pf'), got {header!r}")
        dims = re.match(r"^(\d+)\s(\d+)\s$", f.readline().decode("utf-8"))
        if dims is None:
            raise ValueError(f"{path}: malformed PFM header")
        width, height = map(int, dims.groups())
        scale = float(f.readline().rstrip())
        data = np.fromfile(f, "<f" if scale < 0 else ">f")
    return np.flipud(data.reshape(height, width)).astype(np.float32)
