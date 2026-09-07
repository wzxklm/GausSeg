"""Full-pipeline test: imports and runs each single-module test in order."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

from test_colmap import test_colmap  # noqa: E402
from test_gaus_mvsnet import test_gaus_mvsnet  # noqa: E402
from test_openmvs import test_openmvs  # noqa: E402
from test_qs_sam3d import test_qs_sam3d  # noqa: E402


def test_all() -> None:
    test_colmap()
    test_gaus_mvsnet()
    test_openmvs()
    test_qs_sam3d()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    test_all()
    print("\n=== ALL TESTS PASSED ===")
