# mypy: disable-error-code="operator,assignment,union-attr,arg-type"
import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_list_length,
    validate_window,
)


@feature(
    name="interaction_features",
    category="ml",
    description="Interaction Features - feature products for ML",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def interaction_features(
    features: list[pl.Expr | str],
    max_degree: int = 2,
    include_bias: bool = False,
) -> dict[str, pl.Expr]:
    """Create polynomial interaction features.

    Generates interaction terms between features for capturing
    non-linear relationships.

    Parameters
    ----------
    features : List[pl.Expr | str]
        List of features to create interactions for
    max_degree : int, default 2
        Maximum degree of interactions
    include_bias : bool, default False
        Whether to include bias term (constant 1)

    Returns
    -------
    dict[str, pl.Expr]
        Dictionary of interaction features

    Raises
    ------
    ValueError
        If features is empty or max_degree is not positive
    TypeError
        If features is not a list, max_degree is not an integer, or include_bias is not boolean
    """
    # Validate inputs
    validate_list_length(features, min_length=1, name="features")
    validate_window(max_degree, min_window=1, name="max_degree")
    if not isinstance(include_bias, bool):
        raise TypeError(
            f"include_bias must be a boolean, got {type(include_bias).__name__}",
        )

    # Convert strings to expressions
    expr_features: list[pl.Expr] = [pl.col(f) if isinstance(f, str) else f for f in features]

    interactions = {}

    if include_bias:
        interactions["bias"] = pl.lit(1.0)

    # Add original features
    for i, feat in enumerate(expr_features):
        interactions[f"feat_{i}"] = feat

    if max_degree >= 2:
        # Pairwise interactions
        for i in range(len(expr_features)):
            for j in range(i, len(expr_features)):
                interactions[f"feat_{i}_x_feat_{j}"] = expr_features[i] * expr_features[j]

    if max_degree >= 3:
        # Three-way interactions
        for i in range(len(expr_features)):
            for j in range(i, len(expr_features)):
                for k in range(j, len(expr_features)):
                    interactions[f"feat_{i}_x_feat_{j}_x_feat_{k}"] = (
                        expr_features[i] * expr_features[j] * expr_features[k]
                    )

    return interactions
