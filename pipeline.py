"""Pipeline orchestrator: 5-step sequential execution."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from tqdm.contrib.logging import logging_redirect_tqdm

from config import PipelineConfig
from utils.gpu import gpu_cleanup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step definitions (lazy-imported modules)
# ---------------------------------------------------------------------------

STEPS = {
    "colmap": {
        "name": "COLMAP SfM",
        "module": "colmap.sfm",
        "class": "ColmapSfM",
        "config_attr": "colmap",
    },
    "openmvs": {
        "name": "OpenMVS Dense",
        "module": "openmvs.dense",
        "class": "OpenMVSDense",
        "config_attr": "openmvs",
    },
    "qs_sam3d_convert": {
        "name": "QS-SAM3D Convert",
        "module": "qs_sam3d.converter",
        "class": "QsSam3dConverter",
        "config_attr": "qs_sam3d",
    },
    "qs_sam3d_segment": {
        "name": "QS-SAM3D Segment",
        "module": "qs_sam3d.segmentation",
        "class": "QsSam3dSegmentation",
        "config_attr": "qs_sam3d",
    },
}

MODE_STEPS = {
    "reconstruct": ["colmap", "openmvs"],
    "segment": ["qs_sam3d_convert", "qs_sam3d_segment"],
    "all": ["colmap", "openmvs", "qs_sam3d_convert", "qs_sam3d_segment"],
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """Sequential pipeline orchestrator with lazy imports and GPU cleanup."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(self, mode: str) -> None:
        step_keys = MODE_STEPS.get(mode)
        if step_keys is None:
            raise ValueError(f"Unknown mode: {mode!r}. Choose from {list(MODE_STEPS)}")

        logger.info("Pipeline starting — mode=%s, steps=%d", mode, len(step_keys))
        for i, key in enumerate(step_keys, 1):
            step = STEPS[key]
            logger.info("[%d/%d] %s ...", i, len(step_keys), step["name"])
            t0 = time.perf_counter()

            try:
                with gpu_cleanup():
                    self._run_step(step)
            except Exception:
                logger.exception("[%d/%d] %s FAILED", i, len(step_keys), step["name"])
                raise

            elapsed = time.perf_counter() - t0
            logger.info("[%d/%d] %s done (%.1fs)", i, len(step_keys), step["name"], elapsed)

        logger.info("Pipeline finished — mode=%s", mode)

    def _run_step(self, step: dict) -> None:
        """Lazy-import module and execute step."""
        import importlib

        mod = importlib.import_module(step["module"])
        cls = getattr(mod, step["class"])
        cfg = getattr(self.config, step["config_attr"])
        runner = cls(cfg)
        runner.run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GausSeg reconstruction and segmentation pipeline")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.yaml")
    parser.add_argument("--mode", choices=list(MODE_STEPS), default="all", help="Pipeline mode")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = PipelineConfig.from_yaml(args.config)
    pipeline = Pipeline(config)
    # Bare logger output tears the tqdm bars apart; the redirect routes every
    # record through tqdm.write, printing it on its own line above active bars.
    with logging_redirect_tqdm():
        pipeline.run(args.mode)


if __name__ == "__main__":
    main()
