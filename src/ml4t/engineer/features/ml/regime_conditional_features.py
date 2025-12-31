import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_list_length,
)


@feature(
    name="regime_conditional_features",
    category="ml",
    description="Regime Conditional Features - regime-dependent features",
    lookback=60,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def regime_conditional_features(
    feature: pl.Expr | str,
    regime: pl.Expr | str,
    regime_values: list[int] | None = None,
) -> dict[str, pl.Expr]:
    """Create regime-conditional features.

    Splits features based on market regime, allowing models to learn
    different behaviors in different market conditions.

    Parameters
    ----------
    feature : pl.Expr | str
        Feature to condition on regime
    regime : pl.Expr | str
        Regime indicator column
    regime_values : List[int], optional
        Possible regime close (default: [-1, 0, 1])

    Returns
    -------
    dict[str, pl.Expr]
        Dictionary of conditional features

    Raises
    ------
    ValueError
        If regime_values is empty
    TypeError
        If regime_values is not a list or contains non-integers
    """
    if regime_values is None:
        regime_values = [-1, 0, 1]

    # Validate inputs
    validate_list_length(regime_values, min_length=1, name="regime_values")
    for i, val in enumerate(regime_values):
        if not isinstance(val, int):
            raise TypeError(
                f"regime_values[{i}] must be an integer, got {type(val).__name__}",
            )

    feature = pl.col(feature) if isinstance(feature, str) else feature
    regime = pl.col(regime) if isinstance(regime, str) else regime

    conditional = {}

    for regime_val in regime_values:
        regime_name = {-1: "bear", 0: "neutral", 1: "bull"}.get(
            regime_val,
            str(regime_val),
        )
        conditional[f"feat_{regime_name}"] = (
            pl.when(regime == regime_val).then(feature).otherwise(0)
        )

    return conditional
