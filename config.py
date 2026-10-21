"""Configuration system: YAML -> dataclass."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class GausMvsNetTrainConfig:
    """Gaus-MVSNet unsupervised training configuration."""

    # Data
    dtu_root: Path
    dtu_depths_raw: Path
    train_list: Path
    num_view: int
    num_depth: int
    interval_scale: float
    # Backbone
    ndepths: str
    depth_inter_r: str
    cr_base_chs: str
    # Optimisation
    log_dir: Path
    epochs: int
    lr: float
    lr_epochs: str
    lr_divisor: float
    warmup_iters: int
    batch_size: int
    num_workers: int
    seed: int
    max_steps: int | None
    resume: Path | None
    summary_freq: int
    # Losses
    w_pc: float
    w_rc: float
    w_dc: float
    pc_norm: float
    pc_stage_weights: str
    ssim_lambda: float
    dc_unit: float
    dc_beta: float
    aux_branch: bool
    # Inner opacity loop
    inner_steps: int
    inner_lr: float
    opacity_init: float

    # Derived paths
    log_file: Path = field(init=False)

    def __post_init__(self) -> None:
        self.dtu_root = Path(self.dtu_root)
        self.dtu_depths_raw = Path(self.dtu_depths_raw)
        self.train_list = Path(self.train_list)
        self.log_dir = Path(self.log_dir)
        if self.resume is not None:
            self.resume = Path(self.resume)
        self.log_file = self.log_dir / "train_log.csv"

        stages = (len(self.ndepths_list), len(self.depth_inter_r_list), len(self.cr_base_chs_list))
        if len(set(stages)) != 1:
            raise ValueError(
                f"Cascade stage counts disagree: ndepths={stages[0]}, "
                f"depth_inter_r={stages[1]}, cr_base_chs={stages[2]}"
            )
        if len(self.pc_stage_weights_list) != self.num_stage:
            raise ValueError(
                f"pc_stage_weights has {len(self.pc_stage_weights_list)} entries "
                f"for {self.num_stage} cascade stages"
            )
        if self.num_view < 2:
            raise ValueError(f"num_view must be >= 2, got {self.num_view}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.inner_steps < 1:
            raise ValueError(f"inner_steps must be >= 1, got {self.inner_steps}")
        if not 0.0 < self.opacity_init < 1.0:
            raise ValueError(f"opacity_init must be in (0, 1), got {self.opacity_init}")
        for name in ("dc_unit", "dc_beta", "pc_norm", "lr_divisor"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1 or null, got {self.max_steps}")

    @property
    def ndepths_list(self) -> list[int]:
        return [int(x) for x in self.ndepths.split(",") if x]

    @property
    def depth_inter_r_list(self) -> list[float]:
        return [float(x) for x in self.depth_inter_r.split(",") if x]

    @property
    def cr_base_chs_list(self) -> list[int]:
        return [int(x) for x in self.cr_base_chs.split(",") if x]

    @property
    def lr_epochs_list(self) -> list[int]:
        return [int(x) for x in self.lr_epochs.split(",") if x]

    @property
    def pc_stage_weights_list(self) -> list[float]:
        return [float(x) for x in self.pc_stage_weights.split(",") if x]

    @property
    def num_stage(self) -> int:
        return len(self.ndepths_list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GausMvsNetTrainConfig:
        raw = _load_yaml(path)
        return cls(**{k: v for section in _SECTIONS for k, v in raw[section].items()})


_SECTIONS = ("data", "backbone", "train", "loss", "inner_loop")
# Resolved against the config file's directory, so a machine-specific copy
# (e.g. config.local.yaml) can live anywhere.
_PATH_KEYS = (
    ("data", "dtu_root"),
    ("data", "dtu_depths_raw"),
    ("data", "train_list"),
    ("train", "log_dir"),
    ("train", "resume"),
)


def _load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    for section, key in _PATH_KEYS:
        value = raw.get(section, {}).get(key)
        if value and not Path(value).is_absolute():
            raw[section][key] = str((path.parent / value).resolve())
    return raw
