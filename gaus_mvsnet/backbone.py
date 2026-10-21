"""Cascade backbone for training: CasMVSNet with a variance-only cost volume."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gaus_mvsnet.model.casmvsnet import CascadeMVSNet_eval
from gaus_mvsnet.model.modules import depth_regression, homo_warping


class _VarianceDepthNet(nn.Module):
    """Per-stage depth from the variance of the warped feature volumes.

    Replaces the vendored DepthNet, which also assembles image volumes for
    RC-MVSNet's NeRF branch. It holds no parameters, so the state dict stays
    identical to `CascadeMVSNet_eval` and checkpoints load into inference with
    `strict=True`.
    """

    def forward(
        self,
        features: list[torch.Tensor],
        proj_matrices: torch.Tensor,
        depth_values: torch.Tensor,
        num_depth: int,
        cost_regularization: nn.Module,
        **_,
    ) -> dict[str, torch.Tensor]:
        proj_matrices = torch.unbind(proj_matrices, 1)
        num_views = len(features)
        ref_proj = _full_projection(proj_matrices[0])

        ref_volume = features[0].unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume**2
        del ref_volume
        for src_fea, src_proj in zip(features[1:], proj_matrices[1:]):
            warped = homo_warping(src_fea, _full_projection(src_proj), ref_proj, depth_values)
            volume_sum = volume_sum + warped
            volume_sq_sum = volume_sq_sum + warped**2
            del warped
        variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        prob_volume = F.softmax(cost_regularization(variance).squeeze(1), dim=1)
        return {"depth": depth_regression(prob_volume, depth_values=depth_values)}


def _full_projection(proj: torch.Tensor) -> torch.Tensor:
    """[B, 2, 4, 4] (extrinsic, intrinsic) -> [B, 4, 4] K @ [R | t]."""
    full = proj[:, 0].clone()
    full[:, :3, :4] = torch.matmul(proj[:, 1, :3, :3], proj[:, 0, :3, :4])
    return full


class GausMvsNetBackbone(CascadeMVSNet_eval):
    """`CascadeMVSNet_eval` with the variance-only DepthNet."""

    def __init__(
        self, ndepths: list[int], depth_interals_ratio: list[float], cr_base_chs: list[int]
    ) -> None:
        super().__init__(
            refine=False,
            ndepths=ndepths,
            depth_interals_ratio=depth_interals_ratio,
            share_cr=False,
            grad_method="detach",
            cr_base_chs=cr_base_chs,
        )
        self.DepthNet = _VarianceDepthNet()
