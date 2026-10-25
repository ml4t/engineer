from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci.stage_docs import stage


@pytest.mark.parametrize("versions", [("0.1.6",), ("0.1.5", "0.1.6")])
def test_stable_docs_stage_keeps_archived_bytes_without_stale_stable_pages(
    tmp_path: Path, versions: tuple[str, ...]
) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("new stable docs\n")
    (site / "release.json").write_text('{"version":"0.1.7"}\n')
    previous = tmp_path / "website" / "static" / "docs" / "engineer"
    archives = previous / "releases"
    for version in versions:
        archive = archives / version
        archive.mkdir(parents=True)
        (archive / "release.json").write_text(f'{{"version":"{version}"}}\n')
        (archive / ".hidden").write_bytes(b"hidden archive file\n")
        (archive / "asset.bin").write_bytes(b"\x00\xffarchived binary\n")
    (previous / "obsolete.html").write_text("obsolete stable page\n")
    destination = tmp_path / "stable-site"

    subprocess.run(
        [sys.executable, "scripts/ci/stage_docs.py", str(site), str(archives), str(destination)],
        check=True,
    )

    assert (destination / "index.html").read_text() == "new stable docs\n"
    assert (destination / "release.json").read_text() == '{"version":"0.1.7"}\n'
    assert not (destination / "obsolete.html").exists()
    assert not (site / "releases").exists()
    for version in versions:
        archive = destination / "releases" / version
        assert (archive / "release.json").read_text() == f'{{"version":"{version}"}}\n'
        assert (archive / ".hidden").read_bytes() == b"hidden archive file\n"
        assert (archive / "asset.bin").read_bytes() == b"\x00\xffarchived binary\n"
        assert not (archive / "releases").exists()


@pytest.mark.parametrize("missing", ["site", "archives"])
def test_docs_stage_refuses_missing_checkout_or_build_before_creating_output(
    tmp_path: Path, missing: str
) -> None:
    site, archives, destination = (tmp_path / name for name in ("site", "archives", "output"))
    for source in (site, archives):
        if source.name != missing:
            source.mkdir()
    with pytest.raises(ValueError, match="input is not a directory"):
        stage(site, archives, destination)
    assert not destination.exists()


def test_docs_stage_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    site, archives, destination = (tmp_path / name for name in ("site", "archives", "output"))
    for directory in (site, archives, destination):
        directory.mkdir()
    (destination / "keep").write_text("existing work\n")
    with pytest.raises(FileExistsError, match="already exists"):
        stage(site, archives, destination)
    assert (destination / "keep").read_text() == "existing work\n"


@pytest.mark.parametrize("source", ["site", "archives"])
def test_docs_stage_refuses_recursive_output_inside_inputs(tmp_path: Path, source: str) -> None:
    site, archives = (tmp_path / name for name in ("site", "archives"))
    site.mkdir()
    archives.mkdir()
    with pytest.raises(ValueError, match="overlaps its input"):
        stage(site, archives, tmp_path / source / "output")


def test_docs_stage_refuses_an_archive_tree_in_the_fresh_versioned_site(tmp_path: Path) -> None:
    site, archives = (tmp_path / name for name in ("site", "archives"))
    (site / "releases").mkdir(parents=True)
    archives.mkdir()
    destination = tmp_path / "output"
    with pytest.raises(ValueError, match="must not already contain"):
        stage(site, archives, destination)
    assert not destination.exists()
