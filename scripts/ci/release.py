"""Preflight and verify an ml4t-engineer PyPI publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

PYPI_URL = "https://pypi.org/pypi/{name}/{version}/json"
USER_AGENT = "ml4t-engineer-release-verifier/1.0"


def _package(name: str, version: str) -> dict[str, Any]:
    request = Request(
        PYPI_URL.format(name=name, version=version), headers={"User-Agent": USER_AGENT}
    )
    with urlopen(request, timeout=20) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("PyPI response must be a JSON object")
    return value


def require_version_absent(name: str, version: str) -> None:
    try:
        _package(name, version)
    except HTTPError as error:
        if error.code == 404:
            return
        raise
    raise ValueError(f"{name} {version} already exists on PyPI")


def verify_publication(candidate_dir: Path) -> None:
    manifest = json.loads((candidate_dir / "candidate.json").read_text(encoding="utf-8"))
    package = _package(manifest["name"], manifest["version"])
    info = package.get("info")
    if not isinstance(info, dict):
        raise ValueError("PyPI response has no project metadata")
    if (info.get("name"), info.get("version")) != (manifest["name"], manifest["version"]):
        raise ValueError("PyPI project identity does not match the candidate manifest")
    metadata = {
        "author_email": info.get("author_email"),
        "classifiers": sorted(info.get("classifiers", [])),
        "description": info.get("summary"),
        "keywords": sorted(
            keyword.strip() for keyword in (info.get("keywords") or "").split(",") if keyword
        ),
        "license": info.get("license"),
        "maintainer_email": info.get("maintainer_email"),
        "project_urls": info.get("project_urls"),
        "requires_python": info.get("requires_python"),
    }
    if metadata != manifest.get("metadata"):
        raise ValueError("PyPI public metadata does not match the candidate manifest")

    published = {
        item.get("filename"): (item.get("digests", {}).get("sha256"), item.get("size"))
        for item in package.get("urls", [])
        if isinstance(item, dict)
    }
    expected = {item["filename"]: (item["sha256"], item["size"]) for item in manifest["artifacts"]}
    if published != expected:
        raise ValueError("PyPI artifacts do not match the candidate manifest")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    absent = subparsers.add_parser("require-absent")
    absent.add_argument("name")
    absent.add_argument("version")
    verify = subparsers.add_parser("verify")
    verify.add_argument("candidate_dir", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "require-absent":
        require_version_absent(args.name, args.version)
    else:
        verify_publication(args.candidate_dir)


if __name__ == "__main__":
    main()
