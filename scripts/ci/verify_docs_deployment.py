"""Verify that public documentation identifies the release being published."""

from __future__ import annotations

import json
import os
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

USER_AGENT = "ml4t-engineer-release-verifier/1.0"


def _read_identity(url: str, commit: str, attempt: int) -> object:
    query = urlencode({"commit": commit, "attempt": attempt})
    request = Request(
        f"{url}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def verify(
    urls: tuple[str, ...],
    expected: dict[str, str],
    *,
    attempts: int = 20,
    retry_seconds: float = 15,
) -> None:
    if attempts < 1:
        raise ValueError("attempts must be positive")

    observed: list[object] = []
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            observed = [_read_identity(url, expected["commit"], attempt) for url in urls]
            last_error = None
            if all(value == expected for value in observed):
                return
        except (OSError, URLError, ValueError) as error:
            last_error = error

        if attempt + 1 < attempts:
            time.sleep(retry_seconds)

    raise RuntimeError(
        f"deployed documentation identity did not match {expected!r}; "
        f"last observed {observed!r}; last error {last_error!r}"
    )


def main() -> None:
    expected = {
        "commit": os.environ["RELEASE_COMMIT"],
        "version": os.environ["RELEASE_VERSION"],
        "library": "engineer",
    }
    verify(
        (
            "https://www.ml4trading.io/docs/engineer/release.json",
            f"https://www.ml4trading.io/docs/engineer/releases/{expected['version']}/release.json",
        ),
        expected,
    )


if __name__ == "__main__":
    main()
