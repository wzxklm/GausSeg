"""QS-SAM3D single-module test: converter + segmentation.

Assumes COLMAP + OpenMVS outputs already exist (run `test_colmap` +
`test_openmvs` first, or run as part of `test_all.py`). Executes the full
stage 3-4 chain on the scene `config.yaml` points at and verifies its outputs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import QsSam3dConfig  # noqa: E402
from qs_sam3d.converter import QsSam3dConverter  # noqa: E402
from qs_sam3d.segmentation import QsSam3dSegmentation  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config.yaml"


def test_qs_sam3d_converter() -> None:
    config = QsSam3dConfig.from_yaml(CONFIG_PATH)
    QsSam3dConverter(config).run()

    scene_dir = config.sam3d_dir
    for sub in ("color", "depth", "pose", "intrinsics"):
        d = scene_dir / sub
        assert d.is_dir(), f"missing {d}"
        assert any(d.iterdir()), f"empty {d}"
    assert (scene_dir / "intrinsics" / "intrinsic_depth.txt").is_file()
    assert config.sam3d_pointcloud.is_file(), f"missing {config.sam3d_pointcloud}"

    n_frames = len(list((scene_dir / "color").glob("*.jpg")))
    print(f"[test_qs_sam3d_converter] OK — {n_frames} frames at {scene_dir}")


def test_qs_sam3d_segmentation() -> None:
    config = QsSam3dConfig.from_yaml(CONFIG_PATH)
    QsSam3dSegmentation(config).run()

    assert config.sam3d_segmentation.is_file(), f"missing {config.sam3d_segmentation}"
    assert config.sam3d_segmentation_ply.is_file(), f"missing {config.sam3d_segmentation_ply}"
    assert config.sam3d_mesh_segmentation_ply.is_file(), (
        f"missing {config.sam3d_mesh_segmentation_ply}"
    )
    assert config.sam3d_merged_cloud.is_file(), f"missing {config.sam3d_merged_cloud}"
    assert config.sam3d_segmented_mesh_ply.is_file(), (
        f"missing {config.sam3d_segmented_mesh_ply}"
    )

    print(f"[test_qs_sam3d_segmentation] OK — segmentation at {config.sam3d_segmentation}")


def test_qs_sam3d() -> None:
    test_qs_sam3d_converter()
    test_qs_sam3d_segmentation()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    test_qs_sam3d()
