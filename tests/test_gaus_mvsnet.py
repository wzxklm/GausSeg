"""Gaus-MVSNet depth inference single-module test.

Runs depth estimation and fusion on the COLMAP output for the scene
`config.yaml` points at, and verifies the point cloud and the .dmap files the
mesh and segmentation stages read. Requires `test_colmap` to have run first
(or the COLMAP dense/ outputs already present), and the checkpoint named by
`gaus_mvsnet.checkpoint`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import GausMvsNetConfig  # noqa: E402
from gaus_mvsnet.depth import GausMvsNetDepth  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config.yaml"


def test_gaus_mvsnet() -> None:
    config = GausMvsNetConfig.from_yaml(CONFIG_PATH)

    GausMvsNetDepth(config).run()

    for p in (config.mvs_pair_file, config.final_pointcloud):
        assert p.is_file(), f"missing {p}"

    cams = sorted(config.mvs_cams_dir.glob("*_cam.txt"))
    dmaps = sorted(config.dmap_dir.glob("depth*.dmap"))
    assert cams, f"no camera files in {config.mvs_cams_dir}"
    assert len(dmaps) == len(cams), f"{len(dmaps)} .dmap files for {len(cams)} views"

    print(
        f"[test_gaus_mvsnet] OK — {len(dmaps)} depth maps + fused cloud at {config.mvs_dir}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    test_gaus_mvsnet()
