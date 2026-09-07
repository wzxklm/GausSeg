"""OpenMVS dense reconstruction single-module test.

Runs OpenMVS on the COLMAP output for the scene `config.yaml` points at and
verifies the expected .mvs / .ply files exist. Requires `test_colmap` to have
run first (or the COLMAP dense/ outputs already present), and under
`mesh_only` also `test_gaus_mvsnet`, whose depth maps it fuses.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import OpenMVSConfig  # noqa: E402
from openmvs.dense import OpenMVSDense  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config.yaml"


def test_openmvs() -> None:
    config = OpenMVSConfig.from_yaml(CONFIG_PATH)

    OpenMVSDense(config).run()

    for p in (
        config.openmvs_scene,
        config.openmvs_dense_mvs,
        config.openmvs_dense_ply,
        config.openmvs_mesh_ply,
        config.final_pointcloud,
    ):
        assert p.is_file(), f"missing {p}"

    print(
        f"[test_openmvs] OK — dense PLY + mesh PLY + final PLY at {config.openmvs_dir}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    test_openmvs()
