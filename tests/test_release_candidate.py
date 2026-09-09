from __future__ import annotations

import json
import subprocess
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

import pytest

from scripts.ci import candidate, check_readme_links, release, verify_docs_deployment


@pytest.fixture(scope="module")
def candidate_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("candidate")
    subprocess.run(["uv", "build", "--out-dir", str(directory / "dist")], check=True)
    candidate.create(directory, commit_sha="a" * 40, git_tree="b" * 40)
    return directory


def test_candidate_manifest_binds_both_artifacts_to_source(candidate_dir: Path) -> None:
    manifest = json.loads((candidate_dir / "candidate.json").read_text(encoding="utf-8"))
    candidate.verify(
        candidate_dir,
        expected_commit="a" * 40,
        expected_tree="b" * 40,
        expected_tag=f"v{manifest['version']}",
    )
    assert manifest["name"] == "ml4t-engineer"
    assert manifest["version"]
    assert manifest["metadata"]["description"] == (
        "Feature engineering, labeling, alternative bars, and leakage-safe datasets for financial ML."
    )
    assert manifest["metadata"]["author_email"] == "Stefan Jansen <stefan@applied-ai.com>"
    assert manifest["metadata"]["maintainer_email"] == "Stefan Jansen <pm@ml4trading.io>"
    assert len(manifest["artifacts"]) == 2
    assert all(len(record["sha256"]) == 64 for record in manifest["artifacts"])


def test_candidate_verification_is_atomic_for_wrong_identity_and_bytes(
    candidate_dir: Path,
) -> None:
    with pytest.raises(ValueError, match="does not match candidate version"):
        candidate.verify(candidate_dir, expected_tag="v0.0.0")

    wheel = next((candidate_dir / "dist").glob("*.whl"))
    original = wheel.read_bytes()
    try:
        wheel.write_bytes(original + b"modified")
        with pytest.raises(ValueError, match="integrity check failed"):
            candidate.verify(candidate_dir)
    finally:
        wheel.write_bytes(original)


def test_distributions_contain_public_metadata_and_required_files(candidate_dir: Path) -> None:
    wheel, sdist = candidate._distribution_files(candidate_dir)
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode()
        members = set(archive.namelist())
    assert "Author-email: Stefan Jansen <stefan@applied-ai.com>" in metadata
    assert "Maintainer-email: Stefan Jansen <pm@ml4trading.io>" in metadata
    assert "ml4t/engineer/py.typed" in members

    with tarfile.open(sdist, "r:gz") as archive:
        members = {member.name.partition("/")[2] for member in archive.getmembers()}
    assert {"README.md", "LICENSE", "CHANGELOG.md"} <= members


def test_readme_link_checker_validates_local_and_http_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "LICENSE").write_text("license\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("[local](LICENSE) [remote](https://example.test/docs)\n", encoding="utf-8")
    requests: list[Request] = []

    def open_url(request: Request, *, timeout: int) -> BytesIO:
        requests.append(request)
        assert timeout == 20
        response = BytesIO(b"ok")
        response.status = 200  # type: ignore[attr-defined]
        return response

    monkeypatch.setattr(check_readme_links, "urlopen", open_url)
    check_readme_links.check(readme)
    assert requests[0].get_header("User-agent") == check_readme_links.USER_AGENT


def test_docs_verifier_checks_release_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {"commit": "a" * 40, "library": "engineer", "version": "0.1.4"}
    responses = iter(({**expected, "commit": "old"}, expected))
    requests: list[Request] = []

    def open_url(request: Request, *, timeout: int) -> BytesIO:
        requests.append(request)
        assert timeout == 20
        return BytesIO(json.dumps(next(responses)).encode())

    monkeypatch.setattr(verify_docs_deployment, "urlopen", open_url)
    monkeypatch.setattr(verify_docs_deployment.time, "sleep", lambda _: None)
    verify_docs_deployment.verify(
        ("https://www.ml4trading.io/docs/engineer/release.json",),
        expected,
        attempts=2,
        retry_seconds=0,
    )
    assert [parse_qs(urlparse(request.full_url).query)["attempt"] for request in requests] == [
        ["0"],
        ["1"],
    ]


def test_release_preflight_accepts_only_a_missing_pypi_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_package(_name: str, _version: str) -> dict[str, object]:
        raise HTTPError("https://example.test", 404, "missing", {}, None)

    monkeypatch.setattr(release, "_package", missing_package)
    release.require_version_absent("ml4t-engineer", "9.9.9")

    monkeypatch.setattr(release, "_package", lambda _name, _version: {})
    with pytest.raises(ValueError, match="already exists"):
        release.require_version_absent("ml4t-engineer", "9.9.9")


def test_pypi_publication_must_match_candidate_manifest(
    monkeypatch: pytest.MonkeyPatch,
    candidate_dir: Path,
) -> None:
    manifest = json.loads((candidate_dir / "candidate.json").read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    response = {
        "info": {
            "author_email": metadata["author_email"],
            "classifiers": metadata["classifiers"],
            "keywords": ",".join(metadata["keywords"]),
            "license": metadata["license"],
            "maintainer_email": metadata["maintainer_email"],
            "name": manifest["name"],
            "project_urls": metadata["project_urls"],
            "requires_python": metadata["requires_python"],
            "summary": metadata["description"],
            "version": manifest["version"],
        },
        "urls": [
            {
                "filename": record["filename"],
                "digests": {"sha256": record["sha256"]},
                "size": record["size"],
            }
            for record in manifest["artifacts"]
        ],
    }
    monkeypatch.setattr(release, "_package", lambda _name, _version: response)
    release.verify_publication(candidate_dir)

    response["urls"][0]["digests"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifacts do not match"):
        release.verify_publication(candidate_dir)


def test_pypi_install_verification_retries_index_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter(
        (
            subprocess.CompletedProcess([], 1),
            subprocess.CompletedProcess([], 1),
            subprocess.CompletedProcess([], 0),
        )
    )
    commands: list[list[str]] = []
    sleeps: list[int] = []

    def run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[bytes]:
        assert check is False
        commands.append(command)
        return next(results)

    monkeypatch.setattr(release.subprocess, "run", run)
    monkeypatch.setattr(release.time, "sleep", sleeps.append)

    release.verify_install(
        "ml4t-engineer",
        "0.1.4",
        Path("tests/documentation_workflows.py"),
        attempts=3,
        retry_seconds=7,
    )

    assert len(commands) == 3
    assert commands[0] == [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--refresh-package",
        "ml4t-engineer",
        "--with",
        "ml4t-engineer==0.1.4",
        "python",
        "tests/documentation_workflows.py",
        "--readme-only",
    ]
    assert sleeps == [7, 7]
