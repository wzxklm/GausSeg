"""Gaus-MVSNet training smoke test.

Trains for a few iterations on the DTU data `config.yaml` points at, with a
small batch, and verifies that the losses stay finite and that the checkpoint
loads into the inference backbone with `strict=True`.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import math
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from config import GausMvsNetTrainConfig  # noqa: E402
from gaus_mvsnet.model.casmvsnet import CascadeMVSNet_eval  # noqa: E402
from gaus_mvsnet.train import GausMvsNetTrainer  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config.yaml"
STEPS = 20


def test_train() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = dataclasses.replace(
            GausMvsNetTrainConfig.from_yaml(CONFIG_PATH),
            log_dir=Path(tmp),
            batch_size=1,
            num_view=4,
            max_steps=STEPS,
            resume=None,
        )

        GausMvsNetTrainer(config).run()

        with open(config.log_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == STEPS, f"{len(rows)} logged steps, expected {STEPS}"
        for key in ("loss", "loss_pc", "loss_rc", "loss_dc"):
            assert all(math.isfinite(float(r[key])) for r in rows), f"non-finite {key}"

        ckpt = config.log_dir / "model_000000.ckpt"
        assert ckpt.is_file(), f"missing {ckpt}"
        backbone = CascadeMVSNet_eval(
            refine=False,
            ndepths=config.ndepths_list,
            depth_interals_ratio=config.depth_inter_r_list,
            share_cr=False,
            cr_base_chs=config.cr_base_chs_list,
        )
        backbone.load_state_dict(torch.load(ckpt, map_location="cpu")["model"], strict=True)

        final = float(rows[-1]["loss"])
        print(f"[test_train] OK — {STEPS} steps, final loss {final:.4f}, checkpoint loads")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    test_train()
