from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(sys.platform != "linux", reason="the Docker deployment action runs on Ubuntu")
def test_release_workflow_preserves_old_archives_through_actual_deploy_action(
    tmp_path: Path,
) -> None:
    def command(*args: str, cwd: Path = tmp_path, env: dict[str, str] | None = None) -> str:
        return subprocess.run(
            args, cwd=cwd, env=env, check=True, capture_output=True, text=True
        ).stdout

    remote = tmp_path / "website.git"
    website = tmp_path / ".docs-website"
    command("git", "init", "--bare", "--initial-branch=main", str(remote))
    command("git", "clone", str(remote), str(website))
    previous = website / "static/docs/engineer"
    archive = previous / "releases/0.1.6"
    archive.mkdir(parents=True)
    (archive / "release.json").write_bytes(b'{"version":"0.1.6"}\n')
    (archive / ".hidden").write_bytes(b"archived hidden file\n")
    (archive / "asset.bin").write_bytes(b"\x00\xffarchived binary\n")
    (previous / "obsolete.html").write_text("obsolete stable page\n")
    sibling = website / "static/docs/data/index.html"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("another library\n")
    command("git", "add", ".", cwd=website)
    command(
        "git",
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.test",
        "commit",
        "-m",
        "Seed docs history",
        cwd=website,
    )
    command("git", "push", "origin", "main", cwd=website)
    site = tmp_path / "site"
    site.mkdir()
    (site / "release.json").write_bytes(b'{"version":"0.1.7"}\n')
    (site / "index.html").write_text("new stable page\n")
    config = tmp_path / "gitconfig"
    command(
        "git",
        "config",
        "--file",
        str(config),
        f"url.{remote.as_uri()}.insteadOf",
        "https://fixture:fixture-token@github.test/fixture/website.git",
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(config),
        "GIT_CONFIG_NOSYSTEM": "1",
        "TMPDIR": str(scratch),
        "SSH_DEPLOY_KEY": "",
        "API_TOKEN_GITHUB": "fixture-token",
        "GITHUB_REPOSITORY": "ml4t/engineer",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_REF": "refs/heads/main",
    }
    workflow = yaml.load(
        (ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader
    )
    for step in workflow["jobs"]["docs"]["steps"]:
        if step.get("name") == "Stage stable documentation with existing archives":
            producer = tmp_path / "scripts/ci/stage_docs.py"
            producer.parent.mkdir(parents=True)
            shutil.copyfile(ROOT / "scripts/ci/stage_docs.py", producer)
            command(*shlex.split(step["run"]), env=env)
        action = step.get("uses", "")
        if not action.startswith("cpina/github-action-push-to-another-repository@"):
            continue
        pin = action.split("@", maxsplit=1)[1]
        entrypoint = ROOT / "tests/fixtures" / f"docs-deploy-{pin[:8]}" / "entrypoint.sh"
        inputs = step["with"]
        target = inputs["target-directory"].replace(
            "${{ needs.validate.outputs.version }}", "0.1.7"
        )
        command(
            "bash",
            str(entrypoint),
            "",
            str(tmp_path / inputs["source-directory"]),
            "fixture",
            "website",
            "github.test",
            "fixture@example.test",
            "Fixture",
            "",
            "main",
            "Deploy docs fixture",
            target,
            "false",
            env=env,
        )

    def published(path: str) -> str:
        return command("git", "--git-dir", str(remote), "show", f"main:{path}")

    assert published("static/docs/engineer/releases/0.1.6/release.json") == '{"version":"0.1.6"}\n'
    assert published("static/docs/engineer/releases/0.1.6/.hidden") == "archived hidden file\n"
    assert (
        subprocess.check_output(
            [
                "git",
                "--git-dir",
                str(remote),
                "show",
                "main:static/docs/engineer/releases/0.1.6/asset.bin",
            ]
        )
        == b"\x00\xffarchived binary\n"
    )
    assert published("static/docs/engineer/release.json") == '{"version":"0.1.7"}\n'
    assert published("static/docs/engineer/releases/0.1.7/release.json") == '{"version":"0.1.7"}\n'
    assert published("static/docs/data/index.html") == "another library\n"
    paths = command("git", "--git-dir", str(remote), "ls-tree", "-r", "--name-only", "main")
    assert "static/docs/engineer/obsolete.html" not in paths
    assert "releases/0.1.7/releases/" not in paths
