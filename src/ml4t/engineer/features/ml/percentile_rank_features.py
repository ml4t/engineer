import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_list_length,
    validate_window,
)


@feature(
    name="percentile_rank_features",
    category="ml",
    description="Percentile Rank Features - rank-based normalization",
    lookback=0,
    normalized=True,
    value_range=(0.0, 100.0),
    formula="",
    ta_lib_compatible=False,
)
def percentile_rank_features(
    feature: pl.Expr | str,
    windows: list[int] | None = None,
    method: str = "average",
) -> dict[str, pl.Expr]:
    """Calculate rolling percentile ranks.

    Shows where current value stands relative to recent history,
    useful for normalization and regime identification.

    Parameters
    ----------
    feature : pl.Expr | str
        Feature to rank
    windows : List[int], optional
        Rolling window sizes (default: [20, 50, 100])
    method : str, default "average"
        Ranking method

    Returns
    -------
    dict[str, pl.Expr]
        Dictionary of percentile rank features

    Raises
    ------
    ValueError
        If windows is empty or contains non-positive close
    TypeError
        If windows contains non-integers or method is not a string
    """
    if windows is None:
        windows = [20, 50, 100]

    # Validate inputs
    validate_list_length(windows, min_length=1, name="windows")
    for i, window in enumerate(windows):
        if not isinstance(window, int):
            raise TypeError(
                f"windows[{i}] must be an integer, got {type(window).__name__}",
            )
        validate_window(window, min_window=2, name=f"windows[{i}]")

    if not isinstance(method, str):
        raise TypeError(f"method must be a string, got {type(method).__name__}")

    feature = pl.col(feature) if isinstance(feature, str) else feature

    ranks = {}

    for window in windows:
        # Calculate rolling percentile rank
        # Count how many close in the window are less than or equal to current value
        # This properly calculates percentile rank within each rolling window
        rank = feature.rolling_map(
            lambda x: ((x[-1] > x[:-1]).sum() + 0.5 * (x[-1] == x[:-1]).sum()) / len(x) * 100
            if len(x) > 0
            else None,
            window_size=window,
        )
        ranks[f"rank_{window}"] = rank

    return ranks
