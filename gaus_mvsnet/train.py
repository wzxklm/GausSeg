"""Gaus-MVSNet unsupervised training on DTU.

    L = w_pc * L_PC + w_rc * L_RC + w_dc * L_DC

L_PC is the cross-view photometric loss on every cascade stage; L_RC and L_DC
come from the 3D Gaussian Splatting auxiliary branch on the final-stage depth.
No loss uses ground-truth depth. Checkpoints hold the backbone only, in the
parameter layout the inference stage loads.
"""

from __future__ import annotations

import csv
import logging
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import GausMvsNetTrainConfig
from gaus_mvsnet.backbone import GausMvsNetBackbone
from gaus_mvsnet.dtu import DtuTrainDataset
from gaus_mvsnet.gs_branch import GaussianAuxBranch
from gaus_mvsnet.photometric import MultiStagePhotometricLoss

logger = logging.getLogger(__name__)


class GausMvsNetTrainer:
    """Train the cascade backbone with the photometric and 3DGS losses."""

    def __init__(self, config: GausMvsNetTrainConfig) -> None:
        self.config = config

    def run(self) -> None:
        cfg = self.config
        if not torch.cuda.is_available():
            raise RuntimeError("Gaus-MVSNet training requires a CUDA GPU")
        for path in (
            cfg.dtu_root / "Cameras" / "pair.txt",
            cfg.dtu_root / "Rectified",
            cfg.dtu_depths_raw,
            cfg.train_list,
        ):
            if not path.exists():
                raise FileNotFoundError(f"DTU training data not found: {path}")

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.benchmark = True
        cfg.log_dir.mkdir(parents=True, exist_ok=True)

        dataset = DtuTrainDataset(
            dtu_root=cfg.dtu_root,
            depths_raw=cfg.dtu_depths_raw,
            train_list=cfg.train_list,
            num_view=cfg.num_view,
            num_depth=cfg.num_depth,
            interval_scale=cfg.interval_scale,
        )
        loader = DataLoader(
            dataset,
            cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        if len(loader) == 0:
            raise ValueError(f"{len(dataset)} samples is fewer than one batch of {cfg.batch_size}")
        logger.info("%d training samples, %d iterations per epoch", len(dataset), len(loader))

        model = GausMvsNetBackbone(
            cfg.ndepths_list, cfg.depth_inter_r_list, cfg.cr_base_chs_list
        ).cuda()
        pc_loss = MultiStagePhotometricLoss(cfg.pc_stage_weights_list).cuda()
        branch = None
        if cfg.aux_branch:
            branch = GaussianAuxBranch(
                inner_steps=cfg.inner_steps,
                inner_lr=cfg.inner_lr,
                opacity_init=cfg.opacity_init,
                ssim_lambda=cfg.ssim_lambda,
                w_rc=cfg.w_rc,
                w_dc=cfg.w_dc,
                dc_unit=cfg.dc_unit,
                dc_beta=cfg.dc_beta,
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
        start_epoch = 0
        if cfg.resume is not None:
            state = torch.load(cfg.resume, map_location="cpu")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = state["epoch"] + 1
            logger.info("Resumed from %s at epoch %d", cfg.resume, start_epoch)
        scheduler = self._scheduler(optimizer, len(loader), start_epoch)

        global_step = start_epoch * len(loader)
        with open(cfg.log_file, "a", newline="") as log_file:
            writer = None
            for epoch in range(start_epoch, cfg.epochs):
                model.train()
                for batch_idx, sample in enumerate(loader):
                    row = self._step(model, pc_loss, branch, optimizer, sample)
                    scheduler.step()
                    row = {"epoch": epoch, "step": global_step, **row}

                    if writer is None:
                        writer = csv.DictWriter(
                            log_file, fieldnames=list(row), extrasaction="ignore"
                        )
                        if log_file.tell() == 0:
                            writer.writeheader()
                    writer.writerow(row)
                    if global_step % cfg.summary_freq == 0:
                        log_file.flush()
                        values = {k: v for k, v in row.items() if k not in ("epoch", "step")}
                        logger.info(
                            "epoch %d iter %d/%d | " + ", ".join(f"{k}=%.4g" for k in values),
                            epoch,
                            batch_idx,
                            len(loader),
                            *values.values(),
                        )

                    global_step += 1
                    if cfg.max_steps is not None and global_step >= cfg.max_steps:
                        break

                ckpt = cfg.log_dir / f"model_{epoch:06d}.ckpt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    ckpt,
                )
                logger.info("Saved %s", ckpt)
                if cfg.max_steps is not None and global_step >= cfg.max_steps:
                    break

        logger.info("Gaus-MVSNet training complete: %s", cfg.log_dir)

    def _step(
        self,
        model: GausMvsNetBackbone,
        pc_loss: MultiStagePhotometricLoss,
        branch: GaussianAuxBranch | None,
        optimizer: torch.optim.Optimizer,
        sample: dict,
    ) -> dict[str, float]:
        cfg = self.config
        t0 = time.time()
        sample = _to_cuda(sample)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(sample["imgs"], sample["proj_matrices"], sample["depth_values"])
        loss_pc = pc_loss(outputs, sample["center_imgs"], sample["proj_matrices"]) / cfg.pc_norm
        loss = cfg.w_pc * loss_pc
        row = {"lr": optimizer.param_groups[0]["lr"], "loss_pc": loss_pc.item()}
        if branch is not None:
            K = sample["proj_matrices"]["stage3"][:, 0, 1, :3, :3]
            aux_loss, stats = branch(outputs["depth"], sample["ref_rgb"], K)
            loss = loss + aux_loss
            row.update(stats)

        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        row["loss"] = loss.item()
        row["step_ms"] = (time.time() - t0) * 1000
        row["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        row.update(
            _depth_metrics(
                outputs["depth"].detach(), sample["depth"]["stage3"], sample["mask"]["stage3"]
            )
        )
        return row

    def _scheduler(
        self, optimizer: torch.optim.Optimizer, iters_per_epoch: int, start_epoch: int
    ) -> torch.optim.lr_scheduler.LambdaLR:
        cfg = self.config
        milestones = [e * iters_per_epoch for e in cfg.lr_epochs_list]

        def factor(it: int) -> float:
            warm = 1.0 if it >= cfg.warmup_iters else 1.0 / 3 + (2.0 / 3) * it / cfg.warmup_iters
            return warm / cfg.lr_divisor ** sum(it >= m for m in milestones)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        for _ in range(start_epoch * iters_per_epoch):
            scheduler.step()
        return scheduler


def _to_cuda(x):
    if isinstance(x, dict):
        return {k: _to_cuda(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.cuda(non_blocking=True)
    return x


@torch.no_grad()
def _depth_metrics(
    depth: torch.Tensor, depth_gt: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    err = (depth - depth_gt).abs()[mask > 0.5]
    if err.numel() == 0:
        return {}
    return {
        "abs_err_mm": err.mean().item(),
        "thres2mm_err": (err > 2).float().mean().item(),
        "thres4mm_err": (err > 4).float().mean().item(),
        "thres8mm_err": (err > 8).float().mean().item(),
    }


if __name__ == "__main__":
    from utils.cli import run_stage

    run_stage(
        __file__,
        "Gaus-MVSNet unsupervised training",
        GausMvsNetTrainConfig.from_yaml,
        GausMvsNetTrainer,
    )
