from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_files(candidate_dir: Path) -> tuple[Path, Path]:
    dist_dir = candidate_dir / "dist"
    wheels = tuple(dist_dir.glob("*.whl"))
    sdists = tuple(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(
            f"candidate must contain one wheel and one sdist, found {len(wheels)} and {len(sdists)}"
        )
    return wheels[0], sdists[0]


def _wheel_identity(wheel: Path) -> tuple[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise ValueError(f"wheel must contain one METADATA file: {wheel}")
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise ValueError(f"wheel metadata is missing Name or Version: {wheel}")
    return name, version


def _sdist_identity(sdist: Path) -> tuple[str, str]:
    with tarfile.open(sdist, "r:gz") as archive:
        metadata_members = [
            member
            for member in archive.getmembers()
            if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO")
        ]
        if len(metadata_members) != 1:
            raise ValueError(f"sdist must contain one top-level PKG-INFO file: {sdist}")
        stream = archive.extractfile(metadata_members[0])
        if stream is None:
            raise ValueError(f"cannot read sdist metadata: {sdist}")
        metadata = BytesParser().parsebytes(stream.read())
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise ValueError(f"sdist metadata is missing Name or Version: {sdist}")
    return name, version


def _artifact_record(path: Path) -> dict[str, str | int]:
    return {"filename": path.name, "sha256": _sha256(path), "size": path.stat().st_size}


def create(candidate_dir: Path, commit_sha: str, git_tree: str) -> None:
    wheel, sdist = _distribution_files(candidate_dir)
    wheel_identity = _wheel_identity(wheel)
    sdist_identity = _sdist_identity(sdist)
    if wheel_identity != sdist_identity:
        raise ValueError(
            f"wheel identity {wheel_identity!r} does not match sdist identity {sdist_identity!r}"
        )
    name, version = wheel_identity
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "commit_sha": commit_sha,
        "git_tree": git_tree,
        "name": name,
        "version": version,
        "artifacts": [_artifact_record(wheel), _artifact_record(sdist)],
    }
    (candidate_dir / "candidate.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_manifest(candidate_dir: Path) -> dict[str, Any]:
    value = json.loads((candidate_dir / "candidate.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("candidate manifest must be a JSON object")
    return value


def verify(
    candidate_dir: Path,
    *,
    expected_commit: str | None = None,
    expected_tree: str | None = None,
    expected_tag: str | None = None,
) -> None:
    manifest = _load_manifest(candidate_dir)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported candidate schema: {manifest.get('schema_version')!r}")
    if expected_commit is not None and manifest.get("commit_sha") != expected_commit:
        raise ValueError("candidate commit does not match the requested release commit")
    if expected_tree is not None and manifest.get("git_tree") != expected_tree:
        raise ValueError("candidate tree does not match the requested release tree")
    if expected_tag is not None and expected_tag != f"v{manifest.get('version')}":
        raise ValueError(
            f"release tag {expected_tag!r} does not match candidate version {manifest.get('version')!r}"
        )

    wheel, sdist = _distribution_files(candidate_dir)
    actual_files = {wheel.name: wheel, sdist.name: sdist}
    records = manifest.get("artifacts")
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("candidate manifest must describe exactly two artifacts")
    recorded_files = {record.get("filename"): record for record in records}
    if set(recorded_files) != set(actual_files):
        raise ValueError("candidate manifest artifact inventory does not match dist directory")
    for filename, path in actual_files.items():
        record = recorded_files[filename]
        if record.get("sha256") != _sha256(path) or record.get("size") != path.stat().st_size:
            raise ValueError(f"candidate artifact integrity check failed: {filename}")

    identity = _wheel_identity(wheel)
    if identity != _sdist_identity(sdist):
        raise ValueError("candidate wheel and sdist identities do not match")
    if identity != (manifest.get("name"), manifest.get("version")):
        raise ValueError("candidate metadata does not match the manifest")


def compare_wheel(
    candidate_dir: Path,
    rebuilt_dir: Path,
    *,
    require_byte_identical: bool = False,
) -> None:
    candidate_wheel, _ = _distribution_files(candidate_dir)
    rebuilt_wheels = tuple(rebuilt_dir.glob("*.whl"))
    if len(rebuilt_wheels) != 1:
        raise ValueError(f"rebuilt directory must contain one wheel, found {len(rebuilt_wheels)}")
    rebuilt_wheel = rebuilt_wheels[0]
    if candidate_wheel.name != rebuilt_wheel.name:
        raise ValueError("rebuilt wheel filename does not match the candidate")
    with (
        zipfile.ZipFile(candidate_wheel) as candidate_archive,
        zipfile.ZipFile(rebuilt_wheel) as rebuilt_archive,
    ):
        candidate_members = {
            name: candidate_archive.read(name) for name in candidate_archive.namelist()
        }
        rebuilt_members = {name: rebuilt_archive.read(name) for name in rebuilt_archive.namelist()}
    if candidate_members != rebuilt_members:
        differing_members = sorted(
            name
            for name in candidate_members.keys() | rebuilt_members.keys()
            if candidate_members.get(name) != rebuilt_members.get(name)
        )
        raise ValueError(f"rebuilt wheel contents do not match the candidate: {differing_members}")
    if require_byte_identical and _sha256(candidate_wheel) != _sha256(rebuilt_wheel):
        raise ValueError("rebuilt wheel is not byte-identical to the candidate")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("candidate_dir", type=Path)
    create_parser.add_argument("--commit-sha", required=True)
    create_parser.add_argument("--git-tree", required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("candidate_dir", type=Path)
    verify_parser.add_argument("--expected-commit")
    verify_parser.add_argument("--expected-tree")
    verify_parser.add_argument("--expected-tag")

    compare_parser = subparsers.add_parser("compare-wheel")
    compare_parser.add_argument("candidate_dir", type=Path)
    compare_parser.add_argument("rebuilt_dir", type=Path)
    compare_parser.add_argument("--require-byte-identical", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "create":
        create(args.candidate_dir, args.commit_sha, args.git_tree)
    elif args.command == "verify":
        verify(
            args.candidate_dir,
            expected_commit=args.expected_commit,
            expected_tree=args.expected_tree,
            expected_tag=args.expected_tag,
        )
    else:
        compare_wheel(
            args.candidate_dir,
            args.rebuilt_dir,
            require_byte_identical=args.require_byte_identical,
        )


if __name__ == "__main__":
    main()
