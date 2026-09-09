"""Validate every local and HTTP link in the root README."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

LINK = re.compile(r"(?<!!)\[[^]]+\]\((?P<target>[^)]+)\)")
USER_AGENT = "ml4t-engineer-readme-check/1.0"


def targets(readme: Path) -> tuple[str, ...]:
    """Return distinct README link targets in source order."""
    return tuple(
        dict.fromkeys(match["target"].strip() for match in LINK.finditer(readme.read_text()))
    )


def validate_local(root: Path, target: str) -> None:
    """Require a relative link to resolve inside the repository."""
    parsed = urlparse(target)
    path = (root / unquote(parsed.path)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"README link escapes the repository: {target}") from error
    if not path.exists():
        raise ValueError(f"README link does not exist: {target}")


def validate_http(target: str, *, attempts: int = 3, retry_seconds: float = 1.0) -> None:
    """Require an HTTP target to return a non-error response."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = Request(target, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=20) as response:
                if response.status < 400:
                    return
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = error
        if attempt + 1 < attempts:
            time.sleep(retry_seconds)
    raise RuntimeError(f"README link failed: {target}: {last_error}")


def check(readme: Path, *, check_http: bool = True) -> None:
    """Validate every Markdown link in *readme*."""
    root = readme.resolve().parent
    for target in targets(readme):
        scheme = urlparse(target).scheme
        if scheme in {"http", "https"}:
            if check_http:
                validate_http(target)
        elif not scheme:
            validate_local(root, target)
        else:
            raise ValueError(f"README uses unsupported link scheme: {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("readme", type=Path, nargs="?", default=Path("README.md"))
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    check(args.readme, check_http=not args.local_only)


if __name__ == "__main__":
    main()
