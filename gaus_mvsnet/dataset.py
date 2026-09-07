"""Inference-time sample construction for the cascade MVS backbone.

Each sample is one reference view plus its highest-scoring source views,
resized to a multiple of 32 and paired with per-stage projection matrices.
The geometry each view is resized to is computed once in `__init__` so the
fusion and .dmap stages can reuse it without re-deriving the scale factors.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from gaus_mvsnet.colmap_input import MvsView

logger = logging.getLogger(__name__)

# The cascade halves the cost volume twice, so both sides must stay divisible
# by 32 for the stage feature maps to line up with the warped volume.
_SIZE_BASE = 32
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class MvsInferenceDataset(Dataset):
    """Reference/source view samples in the layout the backbone expects."""

    def __init__(
        self,
        views: list[MvsView],
        image_dir: Path,
        num_view: int,
        num_depth: int,
        interval_scale: float,
        max_h: int,
        max_w: int,
    ) -> None:
        self.views = views
        self.image_dir = Path(image_dir)
        self.num_view = num_view
        self.num_depth = num_depth
        self.interval_scale = interval_scale
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

        self.geometry: dict[int, dict] = {}
        for view in views:
            path = self._image_path(view)
            width, height = Image.open(path).size
            new_w, new_h = _fitted_size(width, height, max_w, max_h)
            K = view.K.copy()
            K[0, :] *= new_w / width
            K[1, :] *= new_h / height
            self.geometry[view.index] = {
                "source_size": (width, height),
                "size": (new_w, new_h),
                "K": K,
            }

    def _image_path(self, view: MvsView) -> Path:
        path = self.image_dir / view.image_name
        if not path.is_file():
            raise FileNotFoundError(f"Image for view {view.index} not found: {path}")
        return path

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, idx: int) -> dict:
        view = self.views[idx]
        by_index = {v.index: v for v in self.views}

        source_ids = [i for i, _ in view.source_views if i in by_index][: self.num_view - 1]
        if not source_ids:
            raise RuntimeError(f"View {view.index} has no usable source view")
        # A view with too few neighbours repeats its best one: the backbone's
        # variance fusion needs a fixed view count, and a repeat contributes
        # nothing rather than corrupting the cost volume.
        while len(source_ids) < self.num_view - 1:
            source_ids.append(source_ids[0])

        target_w, target_h = self.geometry[view.index]["size"]
        imgs, proj_matrices = [], []
        for view_index in [view.index, *source_ids]:
            member = by_index[view_index]
            img = np.asarray(Image.open(self._image_path(member)), dtype=np.float32) / 255.0
            source_w, source_h = self.geometry[view_index]["source_size"]

            K = member.K.copy()
            K[0, :] *= target_w / source_w
            K[1, :] *= target_h / source_h
            img = cv2.resize(img, (target_w, target_h))

            proj = np.zeros((2, 4, 4), dtype=np.float32)
            proj[0] = member.extrinsic
            # Stage 1 runs at quarter resolution; stages 2 and 3 scale back up.
            proj[1, :3, :3] = K
            proj[1, :2, :] /= 4.0
            proj_matrices.append(proj)
            imgs.append(self.transform(img))

        proj_matrices = np.stack(proj_matrices)
        stage2 = proj_matrices.copy()
        stage2[:, 1, :2, :] *= 2
        stage3 = proj_matrices.copy()
        stage3[:, 1, :2, :] *= 4

        interval = (view.depth_max - view.depth_min) / self.num_depth * self.interval_scale
        depth_values = (
            view.depth_min + np.arange(self.num_depth, dtype=np.float32) * interval
        ).astype(np.float32)

        return {
            "imgs": torch.stack(imgs),
            "proj_matrices": {
                "stage1": proj_matrices,
                "stage2": stage2,
                "stage3": stage3,
            },
            "depth_values": depth_values,
            "index": view.index,
        }


def _fitted_size(width: int, height: int, max_w: int, max_h: int) -> tuple[int, int]:
    """Largest size within (max_w, max_h) whose sides are multiples of 32.

    Images already under the cap are cropped down to the grid rather than
    upscaled, so a small capture is never interpolated up to the cap.
    """
    if height > max_h or width > max_w:
        scale = min(max_h / height, max_w / width)
    else:
        scale = 1.0
    new_w = int(scale * width) // _SIZE_BASE * _SIZE_BASE
    new_h = int(scale * height) // _SIZE_BASE * _SIZE_BASE
    if new_w <= 0 or new_h <= 0:
        raise ValueError(
            f"Image {width}x{height} is smaller than the {_SIZE_BASE}px cascade grid"
        )
    return new_w, new_h
