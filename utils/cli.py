"""Shared CLI runner for stage modules that can be executed standalone.

Each stage module's `__main__` block duplicates the same argparse +
logging.basicConfig + `sys.path.insert` preamble. `run_stage` captures
that pattern so per-stage `__main__` blocks reduce to a single call.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, TypeVar

from tqdm.contrib.logging import logging_redirect_tqdm

T = TypeVar("T")


def run_stage(
    module_file: str,
    description: str,
    config_loader: Callable[[Path], T],
    runner: Callable[[T], object],
) -> None:
    """Parse `--config`, configure logging, then build and run a stage.

    `module_file` is the stage module's `__file__`; its parent is added to
    `sys.path` so `from config import ...` resolves when run standalone.
    `config_loader` maps a YAML path to a config; `runner` maps that config
    to an object whose `.run()` is invoked.
    """
    project_root = Path(module_file).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "config.yaml",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Bare logger output tears the tqdm bars apart; the redirect routes every
    # record through tqdm.write, printing it on its own line above active bars.
    with logging_redirect_tqdm():
        runner(config_loader(args.config)).run()
