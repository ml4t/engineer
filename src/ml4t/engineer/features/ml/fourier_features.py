import numpy as np
import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="fourier_features",
    category="ml",
    description="Fourier Features - spectral features for ML",
    lookback=252,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def fourier_features(
    close: pl.Expr | str,
    n_components: int = 10,
    period: int | None = None,
) -> dict[str, pl.Expr]:
    """Extract Fourier features for capturing periodic patterns.

    Useful for capturing complex seasonal patterns in price data.

    Parameters
    ----------
    close : pl.Expr | str
        Time series column
    n_components : int, default 10
        Number of Fourier components
    period : int, optional
        Base period (if None, assumes daily = 390 minutes for US markets)

    Returns
    -------
    dict[str, pl.Expr]
        Dictionary of Fourier features

    Raises
    ------
    ValueError
        If n_components or period are not positive
    TypeError
        If n_components or period are not integers
    """
    # Validate inputs
    validate_window(n_components, min_window=1, name="n_components")
    if period is not None:
        validate_window(period, min_window=1, name="period")

    pl.col(close) if isinstance(close, str) else close

    if period is None:
        period = 390  # Trading minutes in a day

    features = {}

    # Create time index (assumes sequential data)
    t = pl.int_range(pl.len()).cast(pl.Float64)

    for k in range(1, n_components + 1):
        # Fourier basis functions
        features[f"fourier_sin_{k}"] = (2 * np.pi * k * t / period).sin()
        features[f"fourier_cos_{k}"] = (2 * np.pi * k * t / period).cos()

    return features
