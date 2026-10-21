"""Cross-view photometric consistency loss L_PC."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gaus_mvsnet.model.warping import SSIM, gradient, inverse_warping

# Upstream RC-MVSNet weights; `pc_norm` in the config rescales the total.
_RECONSTRUCTION_WEIGHT = 12.0
_SSIM_WEIGHT = 6.0
# Only the first two source views enter the SSIM term, as upstream.
_SSIM_VIEWS = 2
# Added where a source view does not cover the pixel, so the per-pixel min skips it.
_INVALID = 1e4
# Images are full resolution; the two coarse cascade stages predict depth at 1/4 and 1/2.
_STAGE_SCALE = {0: 0.25, 1: 0.5}


def _reprojection_error(
    warped: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """0.5 * photometric + 0.5 * image-gradient Smooth-L1 per pixel, [B, H, W, 1]."""
    photo = F.smooth_l1_loss(warped * mask, ref * mask, reduction="none").mean(-1, keepdim=True)
    ref_dx, ref_dy = gradient(ref)
    warped_dx, warped_dy = gradient(warped)
    # Compare gradients only between two covered pixels, so the mask edge adds no gradient.
    mask_dx = mask[:, :, 1:] * mask[:, :, :-1]
    mask_dy = mask[:, 1:] * mask[:, :-1]
    grad_x = F.smooth_l1_loss(warped_dx * mask_dx, ref_dx * mask_dx, reduction="none").mean(
        -1, keepdim=True
    )
    grad_y = F.smooth_l1_loss(warped_dy * mask_dy, ref_dy * mask_dy, reduction="none").mean(
        -1, keepdim=True
    )
    grad = F.pad(grad_x, (0, 0, 0, 1)) + F.pad(grad_y, (0, 0, 0, 0, 0, 1))
    return 0.5 * photo + 0.5 * grad


class PhotometricLoss(nn.Module):
    """L_PC at one cascade stage.

    Every source view is warped into the reference view with the predicted
    depth; each pixel takes the source view with the lowest reprojection error,
    and the error is averaged over pixels at least one view covers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ssim = SSIM()

    def forward(
        self, imgs: torch.Tensor, cams: torch.Tensor, depth: torch.Tensor, stage_idx: int
    ) -> torch.Tensor:
        imgs = torch.unbind(imgs, 1)
        cams = torch.unbind(cams, 1)
        scale = _STAGE_SCALE.get(stage_idx)

        def to_stage(img: torch.Tensor) -> torch.Tensor:
            # Nearest resampling: bilinear would blur the edges the loss compares.
            if scale is not None:
                img = F.interpolate(img, scale_factor=scale, recompute_scale_factor=True)
            return img.permute(0, 2, 3, 1)

        ref_img = to_stage(imgs[0])
        errors = []
        ssim_loss = 0.0
        for view in range(1, len(imgs)):
            warped, mask = inverse_warping(to_stage(imgs[view]), cams[0], cams[view], depth)
            errors.append(_reprojection_error(warped, ref_img, mask) + _INVALID * (1 - mask))
            if view <= _SSIM_VIEWS:
                ssim_loss = ssim_loss + torch.mean(self.ssim(ref_img, warped, mask))

        best = torch.stack(errors, dim=-1).min(dim=-1).values
        covered = (best < _INVALID).float()
        reconstruction = (best * covered).sum() / covered.sum().clamp_min(1.0)
        return _RECONSTRUCTION_WEIGHT * reconstruction + _SSIM_WEIGHT * ssim_loss


class MultiStagePhotometricLoss(nn.Module):
    """Weighted sum of L_PC over the cascade stages."""

    def __init__(self, stage_weights: list[float]) -> None:
        super().__init__()
        self.stage_weights = stage_weights
        self.loss = PhotometricLoss()

    def forward(
        self, outputs: dict, imgs: torch.Tensor, proj_matrices: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        total = 0.0
        for stage_idx, weight in enumerate(self.stage_weights):
            key = f"stage{stage_idx + 1}"
            total = total + weight * self.loss(
                imgs, proj_matrices[key], outputs[key]["depth"], stage_idx
            )
        return total
