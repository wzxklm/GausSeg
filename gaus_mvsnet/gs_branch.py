"""3D Gaussian Splatting auxiliary branch (training only).

Every batch, one Gaussian per pixel of the final-stage depth D, in the
reference camera frame: mean = back-projected pixel centre, covariance =
||grad D||^2 I, colour = reference RGB. Opacity is the only quantity fitted:
first with D frozen, then held fixed while L_RC and L_DC are evaluated with D
attached. Nothing is kept across batches.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from gsplat import rasterization


def _gaussian_window(
    size: int = 11, sigma: float = 1.5, channels: int = 3, device: str | torch.device = "cpu"
) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = (g[:, None] @ g[None, :])[None, None]
    return window.expand(channels, 1, size, size).contiguous()


def _ssim(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    pad = window.shape[-1] // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=c)
    mu_y = F.conv2d(y, window, padding=pad, groups=c)
    sigma_x = F.conv2d(x * x, window, padding=pad, groups=c) - mu_x**2
    sigma_y = F.conv2d(y * y, window, padding=pad, groups=c) - mu_y**2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=c) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return (num / den).mean()


def _depth_gradient_magnitude(depth: torch.Tensor) -> torch.Tensor:
    """||grad D|| by central differences, [B, H, W]."""
    d = F.pad(depth[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
    gx = (d[:, 1:-1, 2:] - d[:, 1:-1, :-2]) * 0.5
    gy = (d[:, 2:, 1:-1] - d[:, :-2, 1:-1]) * 0.5
    return torch.sqrt(gx**2 + gy**2 + 1e-12)


def _backproject(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """[B, H, W] depth -> [B, H*W, 3] points in the camera frame."""
    b, h, w = depth.shape
    v, u = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    # gsplat samples pixel (i, j) at (j + 0.5, i + 0.5)
    pix = torch.stack([u + 0.5, v + 0.5, torch.ones_like(u)], dim=-1).reshape(1, -1, 3)
    rays = pix @ torch.inverse(K).transpose(1, 2)
    return rays * depth.reshape(b, -1, 1)


class GaussianAuxBranch:
    """L_RC and L_DC from per-batch Gaussians built on the backbone's depth."""

    def __init__(
        self,
        inner_steps: int,
        inner_lr: float,
        opacity_init: float,
        ssim_lambda: float,
        w_rc: float,
        w_dc: float,
        dc_unit: float,
        dc_beta: float,
    ) -> None:
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.opacity_logit_init = math.log(opacity_init / (1 - opacity_init))
        self.ssim_lambda = ssim_lambda
        self.w_rc = w_rc
        self.w_dc = w_dc
        self.dc_unit = dc_unit
        self.dc_beta = dc_beta
        self._window: torch.Tensor | None = None

    def _render(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        K: torch.Tensor,
        h: int,
        w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quats = torch.zeros(means.shape[0], 4, device=means.device)
        quats[:, 0] = 1.0
        viewmat = torch.eye(4, device=means.device)[None]
        out, _, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmat,
            K[None],
            w,
            h,
            render_mode="RGB+D",
            packed=False,
        )
        # "D" is the unnormalized alpha-weighted depth sum
        return out[0, ..., :3].permute(2, 0, 1), out[0, ..., 3]

    def _losses(
        self, rgb_hat: torch.Tensor, z_hat: torch.Tensor, rgb: torch.Tensor, z_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        l1 = (rgb_hat - rgb).abs().mean()
        d_ssim = 1.0 - _ssim(rgb_hat[None], rgb[None], self._window)
        loss_rc = (1 - self.ssim_lambda) * l1 + self.ssim_lambda * d_ssim
        valid = z_target > 0
        loss_dc = F.smooth_l1_loss(
            z_hat[valid] / self.dc_unit, z_target[valid] / self.dc_unit, beta=self.dc_beta
        )
        return loss_rc, loss_dc

    def _weighted(self, loss_rc: torch.Tensor, loss_dc: torch.Tensor) -> torch.Tensor:
        return self.w_rc * loss_rc + self.w_dc * loss_dc

    def __call__(
        self, depth: torch.Tensor, rgb: torch.Tensor, K: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """depth [B, H, W] (with grad), rgb [B, 3, H, W] in [0, 1], K [B, 3, 3] at depth
        resolution.

        Returns the weighted auxiliary loss and values for the training log.
        """
        if self._window is None or self._window.device != depth.device:
            self._window = _gaussian_window(channels=3, device=depth.device)
        b, h, w = depth.shape
        means_all = _backproject(depth, K)
        scales_all = _depth_gradient_magnitude(depth).reshape(b, -1)

        total_rc = total_dc = 0.0
        stats = {"inner_loss_init": 0.0, "inner_loss_final": 0.0, "mean_alpha": 0.0}
        for i in range(b):
            means = means_all[i]
            scales = scales_all[i][:, None].expand(-1, 3)
            colors = rgb[i].reshape(3, -1).t()
            z_target = depth[i].detach()

            logit = torch.full(
                (means.shape[0],), self.opacity_logit_init, device=depth.device, requires_grad=True
            )
            inner_opt = torch.optim.Adam([logit], lr=self.inner_lr)
            means_d, scales_d = means.detach(), scales.detach()
            for step in range(self.inner_steps):
                inner_opt.zero_grad(set_to_none=True)
                rgb_hat, z_hat = self._render(
                    means_d, scales_d, torch.sigmoid(logit), colors, K[i], h, w
                )
                inner_loss = self._weighted(*self._losses(rgb_hat, z_hat, rgb[i], z_target))
                inner_loss.backward()
                inner_opt.step()
                if step == 0:
                    stats["inner_loss_init"] += inner_loss.item() / b

            opacities = torch.sigmoid(logit).detach()
            rgb_hat, z_hat = self._render(means, scales, opacities, colors, K[i], h, w)
            loss_rc, loss_dc = self._losses(rgb_hat, z_hat, rgb[i], z_target)
            stats["inner_loss_final"] += self._weighted(loss_rc, loss_dc).item() / b
            stats["mean_alpha"] += opacities.mean().item() / b
            total_rc = total_rc + loss_rc / b
            total_dc = total_dc + loss_dc / b

        stats["loss_rc"] = float(total_rc)
        stats["loss_dc"] = float(total_dc)
        return self._weighted(total_rc, total_dc), stats
