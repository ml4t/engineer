"""A feature's first evaluation in a process must equal its second.

``polars.Expr.rolling_map`` does not advance its window while the callback is being
compiled, so a feature whose callback calls a lazily-jitted kernel used to return the
first window's value for the entire series the first time a process evaluated it.
Nothing raised: the column came out near-constant. On a 600-row series, five features
returned two distinct values on their first use and the full set on every later one.

That is what made ``etfs/03_financial_features`` fail its withheld-rows check in CI
and pass when re-run by hand: CI compiles, a re-run loads the compiled kernel from
disk. It was read as a point-in-time leak and as CPU contention; it was neither.

Each case runs in its own interpreter with its own empty Numba cache directory,
because "the first call in a process, with nothing compiled" is the condition.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

# feature -> the expression to build, as source, against columns `close` and `returns`
FEATURES = {
    "hurst_exponent": 'regime.hurst_exponent("close", period=100)',
    "coefficient_of_variation": 'statistics.coefficient_of_variation("close", window=50)',
    "rolling_cv_zscore": 'statistics.rolling_cv_zscore("close", window=20)',
    "variance_ratio": 'statistics.variance_ratio("returns", window=50)',
    "rolling_kl_divergence": 'statistics.rolling_kl_divergence("returns", window=50)',
    "rolling_wasserstein": 'statistics.rolling_wasserstein("returns", window=50)',
    "rolling_drift": 'statistics.rolling_drift("returns", window=50)',
    "rolling_entropy": 'ml.rolling_entropy("returns", window=100)',
    "rolling_entropy_lz": 'ml.rolling_entropy_lz("returns", window=100)',
    "rolling_entropy_plugin": 'ml.rolling_entropy_plugin("returns", window=100)',
    "wma": 'trend.wma("close", period=30)',
    "volatility_percentile_rank": 'volatility.volatility_percentile_rank("returns", lookback=100)',
}

PROBE = textwrap.dedent(
    """
    import json, logging, warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    import numpy as np
    import polars as pl
    from ml4t.engineer.features import ml, regime, statistics, trend, volatility

    rng = np.random.default_rng(11)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))
    returns = np.diff(close, prepend=close[0]) / close
    frame = pl.DataFrame({{"close": close, "returns": returns}})

    def evaluate():
        values = frame.select(({expr}).alias("v")).to_series().to_numpy().astype(float)
        return values.tolist()

    first = evaluate()
    second = evaluate()
    print(json.dumps([first, second]))
    """
)


def _run(expr: str, tmp_path) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the feature twice in a fresh interpreter with an empty Numba cache."""
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path), PYTHONWARNINGS="ignore")
    completed = subprocess.run(
        [sys.executable, "-c", PROBE.format(expr=expr)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(f"probe failed:\n{completed.stderr[-2000:]}")
    first, second = json.loads(completed.stdout.strip().splitlines()[-1])
    return np.asarray(first), np.asarray(second)


@pytest.mark.parametrize("name", sorted(FEATURES))
def test_first_evaluation_matches_the_second(name: str, tmp_path) -> None:
    first, second = _run(FEATURES[name], tmp_path)
    finite = first[np.isfinite(first)]
    n = finite.size

    assert n > 100, f"{name} produced almost nothing to compare"
    np.testing.assert_allclose(
        first,
        second,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
        err_msg=f"{name} changes numerical output between its cold and second evaluation",
    )
    distinct = len(np.unique(finite))
    assert distinct > 2, f"{name} came out near-constant ({distinct} distinct values over {n} rows)"
