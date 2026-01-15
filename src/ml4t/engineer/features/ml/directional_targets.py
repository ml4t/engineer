# mypy: disable-error-code="misc"
import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_list_length,
)
from ml4t.engineer.logging import logged_feature


@logged_feature("directional_targets", warn_threshold_ms=200.0, log_data_quality=True)
@feature(
    name="directional_targets",
    category="ml",
    description="Directional Targets - classification labels for ML",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def directional_targets(
    returns: pl.Expr | str,
    thresholds: list[float] | None = None,
    horizon: int | None = None,
) -> dict[str, pl.Expr]:
    """Create directional classification targets.

    Converts continuous returns into classification targets based on thresholds,
    useful for directional prediction models.

    Parameters
    ----------
    returns : pl.Expr | str
        Returns column
    thresholds : List[float], optional
        Thresholds for classification (default: [0.0, 0.001, 0.002, 0.005])
    horizon : int, optional
        If provided, shifts targets by this horizon for future prediction

    Returns
    -------
    dict[str, pl.Expr]
        Dictionary of classification targets

    Raises
    ------
    ValueError
        If thresholds is empty, contains negative close, or horizon is not positive
    TypeError
        If thresholds contains non-numeric close or horizon is not an integer
    """
    if thresholds is None:
        thresholds = [0.0, 0.001, 0.002, 0.005]

    # Validate inputs
    validate_list_length(thresholds, min_length=1, name="thresholds")
    for i, thresh in enumerate(thresholds):
        if not isinstance(thresh, (int, float)):
            raise TypeError(
                f"thresholds[{i}] must be numeric, got {type(thresh).__name__}",
            )
        if thresh < 0:
            raise ValueError(f"thresholds[{i}] must be non-negative, got {thresh}")

    if horizon is not None:
        if not isinstance(horizon, int):
            raise TypeError(f"horizon must be an integer, got {type(horizon).__name__}")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")

    returns = pl.col(returns) if isinstance(returns, str) else returns

    if horizon is not None:
        returns = returns.shift(-horizon)

    targets = {}

    # Binary classification for each threshold
    for thresh in thresholds:
        thresh_bps = int(thresh * 10000)
        if thresh == 0:
            # Simple up/down classification
            # Use cast to preserve nulls (when returns is null, result is null)
            targets["target_direction"] = (returns > 0).cast(pl.Int32)
        else:
            # Three-class: down, neutral, up
            # Preserve nulls: when returns is null, comparisons return null
            targets[f"target_{thresh_bps}bps"] = (
                pl.when(returns.is_null())
                .then(pl.lit(None, dtype=pl.Int32))
                .when(returns > thresh)
                .then(2)
                .when(returns < -thresh)
                .then(0)
                .otherwise(1)
            )

    return targets
