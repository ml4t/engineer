"""Apply fractional differencing without relying on repository data files."""

import numpy as np
import polars as pl

from ml4t.engineer.features.fdiff import (
    fdiff_diagnostics,
    ffdiff,
    find_optimal_d,
)


def main() -> None:
    """Compare a random-walk price with its fractionally differenced form."""
    rng = np.random.default_rng(42)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 1_500)))
    data = pl.DataFrame({"close": prices})

    search = find_optimal_d(
        data["close"],
        d_range=(0.0, 1.0),
        step=0.1,
        adf_pvalue_threshold=0.05,
    )
    selected_d = search["optimal_d"]
    transformed = data.with_columns(close_ffd=ffdiff("close", d=selected_d))
    diagnostics = fdiff_diagnostics(data["close"], d=selected_d)

    assert len(transformed) == len(data)
    assert transformed["close_ffd"].is_not_nan().any()
    print(f"optimal_d={selected_d:.2f}")
    print(f"adf_pvalue={diagnostics['adf_pvalue']:.6f}")
    print(f"correlation={diagnostics['correlation']:.6f}")
    print(f"weights={diagnostics['n_weights']}")
    print("fdiff_example=pass")


if __name__ == "__main__":
    main()
