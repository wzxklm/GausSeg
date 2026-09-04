"""COLMAP SfM single-module test.

Runs COLMAP on the scene `config.yaml` points at and verifies the expected
output files exist. Runnable standalone (`python tests/test_colmap.py`) or
imported by `tests/test_all.py`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from colmap.sfm import ColmapSfM  # noqa: E402
from config import ColmapConfig  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config.yaml"


def test_colmap() -> None:
    config = ColmapConfig.from_yaml(CONFIG_PATH)

    ColmapSfM(config).run()

    assert config.colmap_sparse.is_dir(), f"missing {config.colmap_sparse}"
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        assert (config.colmap_sparse / name).is_file(), f"missing sparse/0/{name}"

    assert config.colmap_images.is_dir(), f"missing {config.colmap_images}"
    undistorted = list(config.colmap_images.glob("*.jpg"))
    assert undistorted, f"no undistorted images in {config.colmap_images}"

    print(
        f"[test_colmap] OK — sparse/0 + {len(undistorted)} undistorted images "
        f"at {config.colmap_dir}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    test_colmap()
