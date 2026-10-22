"""Stage stable documentation without dropping existing versioned archives."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def stage(site: Path, archives: Path, destination: Path) -> None:
    """Combine a fresh site and existing archives in a new deployment directory."""
    for source in (site, archives):
        if not source.is_dir():
            raise ValueError(f"documentation input is not a directory: {source}")
        if destination.resolve().is_relative_to(source.resolve()):
            raise ValueError(f"deployment destination overlaps its input: {destination}")
    if destination.exists():
        raise FileExistsError(f"deployment destination already exists: {destination}")
    if (site / "releases").exists():
        raise ValueError("fresh documentation must not already contain release archives")
    shutil.copytree(site, destination, symlinks=True)
    shutil.copytree(archives, destination / "releases", symlinks=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("site", type=Path)
    parser.add_argument("archives", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    stage(args.site, args.archives, args.destination)


if __name__ == "__main__":
    main()
