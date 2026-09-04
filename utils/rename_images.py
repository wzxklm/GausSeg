"""Rename images in a directory to the `{:08d}.ext` convention.

Sorts by filename only, which is correct when source names are lexicographically
chronological (camera timestamps such as `20260421_010505.jpg` are). Other sort
strategies (exif/mtime/natural) can be added later as `--sort` options.

python -m utils.rename_images --dir data/<scene>/inputs/images
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def rename_by_name(images_dir: Path, dry_run: bool = False) -> None:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images dir not found: {images_dir}")

    files = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        raise FileNotFoundError(f"no images found in {images_dir}")

    targets = [
        images_dir / f"{i:08d}{p.suffix.lower()}"
        for i, p in enumerate(files)
    ]

    if all(src == dst for src, dst in zip(files, targets)):
        logger.info("already in target format, nothing to do (%d files)", len(files))
        return

    foreign = set(images_dir.iterdir()) - set(files)
    collisions = [dst for dst in targets if dst in foreign]
    if collisions:
        raise RuntimeError(
            f"{len(collisions)} target name(s) collide with non-image files, "
            f"e.g. {collisions[0].name}"
        )

    if dry_run:
        for src, dst in zip(files, targets):
            if src != dst:
                logger.info("[dry-run] %s -> %s", src.name, dst.name)
        return

    # Two-pass via temp suffix to avoid clobbering when source names
    # overlap with the target {:08d} space.
    tmp_paths = [p.with_name(f".rename_tmp_{i:08d}{p.suffix.lower()}") for i, p in enumerate(files)]
    for src, tmp in zip(files, tmp_paths):
        src.rename(tmp)
    for tmp, dst in zip(tmp_paths, targets):
        tmp.rename(dst)

    logger.info("renamed %d files in %s", len(files), images_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename images to {:08d}.ext by sorted filename.")
    parser.add_argument("--dir", type=Path, required=True, help="images directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    rename_by_name(args.dir, dry_run=args.dry_run)
