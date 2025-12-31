"""Meta-labeling for signal correctness assessment.

Meta-labeling is a technique from López de Prado (2018) that creates a secondary
model to predict whether the primary trading signal will be profitable. This allows:

1. Filtering low-confidence trades
2. Sizing positions based on confidence
3. Improving strategy Sharpe ratio without changing signal logic

The meta-label is binary:
- 1: The primary signal direction matched the subsequent return (profitable)
- 0: The primary signal direction opposed the subsequent return (unprofitable)

References
----------
.. [1] López de Prado, M. (2018). "Advances in Financial Machine Learning".
       Chapter 3: Meta-Labeling.
"""

from typing import Literal

import polars as pl

__all__ = [
    "meta_labels",
    "apply_meta_model",
    "compute_bet_size",
]


def meta_labels(
    data: pl.DataFrame,
    signal_col: str,
    return_col: str,
    threshold: float = 0.0,
) -> pl.DataFrame:
    """Create meta-labels indicating if primary signal was profitable.

    Meta-labels assess whether the primary model's directional prediction
    was correct. The primary signal indicates direction (+1 long, -1 short),
    and the meta-label indicates if taking that direction was profitable.

    Parameters
    ----------
    data : pl.DataFrame
        Input DataFrame containing signal and return columns.
    signal_col : str
        Column name containing the primary signal. Values should be:
        - Positive: Long signal
        - Negative: Short signal
        - Zero: No signal (will produce NaN meta-label)
    return_col : str
        Column name containing the forward returns to evaluate against.
        These should be the returns achieved after the signal was generated.
    threshold : float, default 0.0
        Minimum return threshold to consider a trade profitable.
        Use positive value to require returns exceed transaction costs.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with added 'meta_label' column:
        - 1: Signal was profitable (signal * return > threshold)
        - 0: Signal was unprofitable (signal * return <= threshold)
        - null: No signal (signal == 0)

    Notes
    -----
    The meta-label is computed as:

    .. math::

        \\text{meta\\_label} = \\mathbb{1}[\\text{signal} \\cdot \\text{return} > \\text{threshold}]

    This creates a binary classification target for a meta-model that predicts
    whether to act on the primary signal.

    Examples
    --------
    >>> import polars as pl
    >>> from ml4t.engineer.labeling import meta_labels
    >>>
    >>> df = pl.DataFrame({
    ...     "signal": [1, -1, 1, -1, 0],
    ...     "fwd_return": [0.02, -0.01, -0.01, -0.02, 0.01],
    ... })
    >>> result = meta_labels(df, "signal", "fwd_return")
    >>> # Row 0: long + positive return = 1 (profitable)
    >>> # Row 1: short + negative return = 1 (profitable)
    >>> # Row 2: long + negative return = 0 (unprofitable)
    >>> # Row 3: short + negative return = 0 (unprofitable, short lost)
    >>> # Row 4: no signal = null

    References
    ----------
    .. [1] López de Prado, M. (2018). "Advances in Financial Machine Learning".
           Wiley. Chapter 3.
    """
    signal = pl.col(signal_col)
    returns = pl.col(return_col)

    # Compute signed return (positive if signal direction was profitable)
    signed_return = signal.sign() * returns

    # Meta-label: 1 if profitable, 0 if not, null if no signal
    meta_label = (
        pl.when(signal == 0)
        .then(None)
        .when(signed_return > threshold)
        .then(1)
        .otherwise(0)
        .alias("meta_label")
    )

    return data.with_columns(meta_label)


def compute_bet_size(
    probability: pl.Expr | str,
    method: Literal["linear", "sigmoid", "discrete"] = "sigmoid",
    scale: float = 1.0,
    threshold: float = 0.5,
) -> pl.Expr:
    """Compute bet size from meta-model probability.

    Transforms the meta-model's predicted probability of success into
    a bet sizing coefficient. Higher probability leads to larger positions.

    Parameters
    ----------
    probability : pl.Expr | str
        Column containing meta-model probability predictions [0, 1].
    method : {"linear", "sigmoid", "discrete"}, default "sigmoid"
        Bet sizing function:
        - "linear": bet_size = 2 * (prob - 0.5), range [-1, 1]
        - "sigmoid": bet_size = (1 + e^(-scale*(prob-0.5)))^-1 * 2 - 1
        - "discrete": bet_size = 1 if prob > threshold else 0
    scale : float, default 1.0
        Scaling factor for sigmoid. Higher values create sharper cutoff.
        Ignored for "linear" and "discrete" methods.
    threshold : float, default 0.5
        Probability threshold for "discrete" method.
        Ignored for "linear" and "sigmoid" methods.

    Returns
    -------
    pl.Expr
        Bet size coefficient, typically in range [0, 1] or [-1, 1].

    Notes
    -----
    The bet size methods are:

    **Linear**: Simple linear scaling centered at 0.5
    .. math::

        \\text{bet\\_size} = 2 \\cdot (p - 0.5)

    **Sigmoid**: S-curve that concentrates bets near extremes
    .. math::

        \\text{bet\\_size} = \\frac{2}{1 + e^{-s \\cdot (p - 0.5)}} - 1

    **Discrete**: Binary sizing based on threshold
    .. math::

        \\text{bet\\_size} = \\mathbb{1}[p > \\text{threshold}]

    Examples
    --------
    >>> import polars as pl
    >>> from ml4t.engineer.labeling import compute_bet_size
    >>>
    >>> df = pl.DataFrame({"prob": [0.3, 0.5, 0.7, 0.9]})
    >>> df.with_columns(
    ...     compute_bet_size("prob", method="linear").alias("linear"),
    ...     compute_bet_size("prob", method="sigmoid", scale=5.0).alias("sigmoid"),
    ...     compute_bet_size("prob", method="discrete", threshold=0.6).alias("discrete"),
    ... )
    """
    prob = pl.col(probability) if isinstance(probability, str) else probability

    if method == "linear":
        # Linear: 0.0 -> -1, 0.5 -> 0, 1.0 -> 1
        return (prob - 0.5) * 2

    elif method == "sigmoid":
        # Sigmoid: S-curve centered at 0.5, scaled to [-1, 1]
        # Using polars expressions for the sigmoid transform
        x = (prob - 0.5) * scale
        sigmoid = 1 / (1 + (-x).exp())
        return sigmoid * 2 - 1

    elif method == "discrete":
        # Discrete: binary 0/1 based on threshold
        return pl.when(prob > threshold).then(1.0).otherwise(0.0)

    else:
        msg = f"Unknown method: {method}. Use 'linear', 'sigmoid', or 'discrete'."
        raise ValueError(msg)


def apply_meta_model(
    data: pl.DataFrame,
    primary_signal_col: str,
    meta_probability_col: str,
    bet_size_method: Literal["linear", "sigmoid", "discrete"] = "sigmoid",
    scale: float = 5.0,
    threshold: float = 0.5,
    output_col: str = "sized_signal",
) -> pl.DataFrame:
    """Apply meta-model probability to size primary signal bets.

    Combines the primary model's directional signal with the meta-model's
    confidence estimate to produce a sized position signal.

    Parameters
    ----------
    data : pl.DataFrame
        Input DataFrame with signal and probability columns.
    primary_signal_col : str
        Column with primary model signal (typically +1, -1, or 0).
    meta_probability_col : str
        Column with meta-model predicted probability [0, 1].
    bet_size_method : {"linear", "sigmoid", "discrete"}, default "sigmoid"
        Method to convert probability to bet size. See `compute_bet_size`.
    scale : float, default 5.0
        Scaling factor for sigmoid method.
    threshold : float, default 0.5
        Threshold for discrete method.
    output_col : str, default "sized_signal"
        Name for the output column.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with added sized signal column:
        sized_signal = sign(primary_signal) * bet_size(probability)

    Notes
    -----
    The sized signal is computed as:

    .. math::

        \\text{sized\\_signal} = \\text{sign}(\\text{signal}) \\cdot f(\\text{probability})

    where f() is the bet sizing function.

    The output can be used directly as position weights in a backtest,
    where the sign indicates direction and magnitude indicates conviction.

    Examples
    --------
    >>> import polars as pl
    >>> from ml4t.engineer.labeling import apply_meta_model
    >>>
    >>> df = pl.DataFrame({
    ...     "signal": [1, -1, 1, -1],
    ...     "meta_prob": [0.8, 0.3, 0.5, 0.9],
    ... })
    >>> result = apply_meta_model(df, "signal", "meta_prob")
    >>> # High prob + long signal -> strong positive
    >>> # Low prob + short signal -> weak negative (may filter)
    >>> # 0.5 prob + any signal -> near zero (uncertain)

    See Also
    --------
    meta_labels : Create meta-labels for training meta-model.
    compute_bet_size : Underlying bet sizing functions.
    """
    signal = pl.col(primary_signal_col)

    # Compute bet size from probability
    bet_size = compute_bet_size(
        meta_probability_col,
        method=bet_size_method,
        scale=scale,
        threshold=threshold,
    )

    # Sized signal: direction from primary, magnitude from meta
    sized_signal = signal.sign() * bet_size.abs()

    return data.with_columns(sized_signal.alias(output_col))
