import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_list_length,
)


@feature(
    name="create_lag_features",
    category="ml",
    description="Create Lag Features - lagged close for ML",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def create_lag_features(
    feature: pl.Expr | str,
    lags: list[int] | None = None,
    include_diff: bool = True,
    include_ratio: bool = False,
) -> dict[str, pl.Expr]:
    """Create lagged features with optional differences and ratios.

    Generates multiple lag features from a single column, useful for
    capturing temporal dependencies.

    Parameters
    ----------
    feature : pl.Expr | str
        Feature to lag
    lags : List[int], optional
        Lag periods (default: [1, 2, 3, 5, 10])
    include_diff : bool, default True
        Whether to include differences from lagged close
    include_ratio : bool, default False
        Whether to include ratios to lagged close

    Returns
    -------
    dict[str, pl.Expr]
        Dictionary of lagged features

    Raises
    ------
    ValueError
        If lags is empty or contains non-positive close
    TypeError
        If lags contains non-integers or boolean parameters are not boolean
    """
    if lags is None:
        lags = [1, 2, 3, 5, 10]

    # Validate inputs
    validate_list_length(lags, min_length=1, name="lags")
    for i, lag in enumerate(lags):
        if not isinstance(lag, int):
            raise TypeError(f"lags[{i}] must be an integer, got {type(lag).__name__}")
        if lag <= 0:
            raise ValueError(f"lags[{i}] must be positive, got {lag}")

    if not isinstance(include_diff, bool):
        raise TypeError(
            f"include_diff must be a boolean, got {type(include_diff).__name__}",
        )
    if not isinstance(include_ratio, bool):
        raise TypeError(
            f"include_ratio must be a boolean, got {type(include_ratio).__name__}",
        )

    feature = pl.col(feature) if isinstance(feature, str) else feature

    lag_features = {}

    for lag in lags:
        # Basic lag
        lag_features[f"lag_{lag}"] = feature.shift(lag)

        if include_diff:
            # Difference from lag
            lag_features[f"diff_{lag}"] = feature - feature.shift(lag)

        if include_ratio:
            # Ratio to lag
            lag_features[f"ratio_{lag}"] = feature / (feature.shift(lag) + 1e-10)

    return lag_features
