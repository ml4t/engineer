"""Finite observation-window methods with explicit missing-value definitions.

These methods are independent of the recursively seeded TA-Lib indicators.
Window sizes count input observations, including missing observations. Expressions
must be applied within the caller's asset groups, as compute_features does.
"""

import math
from numbers import Real
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from ml4t.engineer._numba import jit
from ml4t.engineer.core.decorators import feature


def _period(period: int, *, minimum: int = 1) -> int:
    if isinstance(period, bool) or not isinstance(period, int) or period < minimum:
        raise ValueError(f"period must be an integer >= {minimum}, got {period!r}")
    return period


def _ready(expression: pl.Expr, period: int) -> pl.Expr:
    return pl.when(pl.int_range(pl.len()) >= period - 1).then(expression)


def _numeric(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Float64).fill_nan(None)


def _difference(price: pl.Expr, reference: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    if dtype.is_integer():
        price, reference = price.cast(pl.Int128), reference.cast(pl.Int128)
    if dtype == pl.Int128 or isinstance(dtype, pl.Decimal):
        same_sign = ((price >= 0) & (reference >= 0)) | ((price <= 0) & (reference <= 0))
        safe_price = pl.when(same_sign).then(price).otherwise(0)
        safe_reference = pl.when(same_sign).then(reference).otherwise(0)
        exact = (safe_price - safe_reference).cast(pl.Float64)
        return (
            pl.when(same_sign)
            .then(exact)
            .otherwise(price.cast(pl.Float64) - reference.cast(pl.Float64))
        )
    if dtype.is_integer():
        return (price - reference).cast(pl.Float64)
    return price.cast(pl.Float64) - reference.cast(pl.Float64)


def _paired_difference(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    def left_type(value: pl.Expr, dtype: pl.DataType) -> pl.Expr:
        def right_type(reference: pl.Expr, reference_dtype: pl.DataType) -> pl.Expr:
            if dtype.is_integer() and reference_dtype.is_integer():
                return _difference(value, reference, pl.Int128())
            if isinstance(dtype, pl.Decimal) and isinstance(reference_dtype, pl.Decimal):
                return _difference(value, reference, dtype)
            return value.cast(pl.Float64) - reference.cast(pl.Float64)

        return right.pipe_with_dtype(right_type)

    return left.pipe_with_dtype(left_type)


@feature(
    name="absolute_percent_change",
    category="momentum",
    description="Fractional change with absolute reference denominator",
    normalized=True,
    value_range=(float("-inf"), float("inf")),
    formula="(value - previous) / abs(previous)",
)
def absolute_percent_change(close: str, period: int = 1) -> pl.Expr:
    """Fractional change of signed values; not percentage points or signed-denominator returns."""
    _period(period)
    value = pl.col(close)
    previous = value.shift(period)
    return _paired_difference(value, previous) / previous.cast(pl.Float64).abs()


def exponential_decay(
    *,
    decay_rate: float | None = None,
    span: float | None = None,
    halflife: float | None = None,
    center_of_mass: float | None = None,
) -> float:
    """Resolve exactly one finite-weight selector, with no implicit recursive-EMA substitution."""
    choices = {
        "decay_rate": decay_rate,
        "span": span,
        "halflife": halflife,
        "center_of_mass": center_of_mass,
    }
    supplied = [(name, value) for name, value in choices.items() if value is not None]
    if len(supplied) != 1:
        raise ValueError("Declare exactly one of decay_rate, span, halflife, center_of_mass")
    name, value = supplied[0]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    if name == "span":
        if value <= 1:
            raise ValueError("span must be > 1")
        value = 1 - 2 / (1 + value)
    elif name == "halflife":
        if value <= 0:
            raise ValueError("halflife must be positive")
        value = math.exp(math.log(0.5) / value)
    elif name == "center_of_mass":
        if value <= 0:
            raise ValueError("center_of_mass must be positive")
        value = 1 - 1 / (1 + value)
    if not 0 < value <= 1:
        raise ValueError("Resolved decay_rate must be in (0, 1]")
    return value


@feature(
    name="rsi_cutler",
    category="momentum",
    description="Finite simple-average RSI over period changes, omitting missing changes",
    normalized=True,
    value_range=(0.0, 100.0),
    formula="100 * sum(positive changes) / sum(absolute changes)",
)
def rsi_cutler(close: str, period: int = 14) -> pl.Expr:
    """Cutler RSI over period changes (period+1 prices); flat windows yield NaN.

    Missing quotes invalidate their adjacent changes. The remaining changes in
    the full observation window are reduced without filling the missing prices.
    Integer changes are subtracted in Int128 before floating conversion.
    """
    _period(period)

    def change(price: pl.Expr, dtype: pl.DataType) -> pl.Expr:
        return _difference(price, price.shift(1), dtype)

    delta = pl.col(close).pipe_with_dtype(change).fill_nan(None)
    gain = delta.clip(lower_bound=0).rolling_mean(period, min_samples=1)
    loss = (-delta.clip(upper_bound=0)).rolling_mean(period, min_samples=1)
    return _ready(100 * gain / (gain + loss), period + 1)


@feature(
    name="finite_mean",
    category="trend",
    description="Full-window simple mean with NaN/null omission and valid-count denominator",
    formula="sum(valid prices) / count(valid prices)",
)
def finite_mean(close: str, period: int = 20) -> pl.Expr:
    """Require period rows, then average the nonmissing values in that window."""
    _period(period)
    return _ready(_numeric(close).rolling_mean(period, min_samples=1), period)


@feature(
    name="finite_weighted_mean",
    category="trend",
    description="Finite weighted mean with independently reduced weight denominator",
    input_type="OHLCV",
    formula="nansum(close * volume) / nansum(volume)",
)
def finite_weighted_mean(close: str, volume: str, period: int = 20) -> pl.Expr:
    """Weights paired with missing prices remain in the independent denominator.

    This explicit definition reproduces WeightedAverageValue/VWAP in Pipeline.
    Callers requiring a paired-valid denominator must mask weights themselves.
    """
    _period(period)
    numerator = (_numeric(close) * _numeric(volume)).rolling_sum(period, min_samples=0)
    denominator = _numeric(volume).rolling_sum(period, min_samples=0)
    return _ready(numerator / denominator, period)


@feature(
    name="average_dollar_volume",
    category="volume",
    description="Finite price-times-volume sum divided by the full observation count",
    input_type="OHLCV",
    formula="nansum(close * volume) / period",
)
def average_dollar_volume(close: str, volume: str, period: int = 20) -> pl.Expr:
    """Mean dollar volume with fixed-count denominator; no currency/basis conversion."""
    _period(period)
    total = (_numeric(close) * _numeric(volume)).rolling_sum(period, min_samples=0)
    return _ready(total / period, period)


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _weighted_windows(
    values: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    omit: bool,
) -> npt.NDArray[np.float64]:
    width = len(weights)
    result = np.full(len(values), np.nan)
    weight_sum = weights.sum()
    for end in range(width - 1, len(values)):
        total = 0.0
        missing = False
        for j in range(width):
            value = values[end - width + 1 + j]
            if np.isnan(value):
                missing = True
            else:
                total += (weights[j] / weight_sum) * value
        if missing and not omit:
            continue
        result[end] = total
    return result


def _variance_windows(
    values: Any,
    missing: npt.NDArray[np.bool_],
    weights: npt.NDArray[np.float64],
    omit: bool,
    bias_corrected: bool,
) -> npt.NDArray[np.float64]:
    width = len(weights)
    result = np.full(len(values), np.nan)
    for end in range(width - 1, len(values)):
        start = end - width + 1
        first = -1
        weight_sum, denominator = 0.0, 0.0
        positive, negative, has_missing = False, False, False
        for j in range(width):
            if missing[start + j]:
                has_missing = True
                continue
            if first < 0:
                first = start + j
            value = values[start + j]
            positive = positive or value > 0
            negative = negative or value < 0
            # This equals W**2 - sum(w**2), without subtracting nearly equal numbers.
            denominator += 2 * weight_sum * weights[j]
            weight_sum += weights[j]
        if first < 0 or (has_missing and not omit):
            continue
        if bias_corrected and denominator <= 0:
            continue
        anchor = values[first]
        deltas = np.zeros(width)
        scale = 0.0
        for j in range(width):
            if missing[start + j]:
                continue
            value = values[start + j]
            if positive and negative:
                delta = float(value)
            elif value >= anchor:
                delta = float(value - anchor)
            else:
                delta = -float(anchor - value)
            deltas[j] = delta
            scale = max(scale, abs(delta))
        if not np.isfinite(scale):
            continue
        if scale == 0:
            result[end] = 0.0
            continue
        mean = 0.0
        for j in range(width):
            if not missing[start + j]:
                mean += (weights[j] / weight_sum) * (deltas[j] / scale)
        variance = 0.0
        for j in range(width):
            if not missing[start + j]:
                residual = deltas[j] / scale - mean
                variance += (weights[j] / weight_sum) * residual * residual
        if bias_corrected:
            variance = variance * weight_sum * weight_sum / denominator
        result[end] = scale * np.sqrt(variance)
    return result


_variance_numba = jit(nopython=True, cache=True)(_variance_windows)


def _variance_expression(
    close: str | pl.Expr,
    weights: npt.NDArray[np.float64],
    *,
    omit: bool = False,
    bias_corrected: bool = True,
) -> pl.Expr:
    expression = pl.col(close) if isinstance(close, str) else close

    def compute(values: pl.Series) -> pl.Series:
        missing = values.is_null().to_numpy()
        if values.dtype in (pl.Float32, pl.Float64):
            missing = missing | values.is_nan().fill_null(False).to_numpy()
        if values.dtype in (pl.Int128, pl.Null) or isinstance(values.dtype, pl.Decimal):
            result = _variance_windows(values.to_list(), missing, weights, omit, bias_corrected)
        else:
            result = _variance_numba(
                values.fill_null(0).to_numpy(), missing, weights, omit, bias_corrected
            )
        return pl.Series(result)

    # Keep raw integer/Decimal values through window-local subtraction. A global
    # float anchor loses later one-tick variation after a large price-level change.
    return expression.map_batches(compute, return_dtype=pl.Float64)


def _weighted_expression(
    close: str | pl.Expr,
    weights: npt.NDArray[np.float64],
    *,
    variance: bool = False,
    omit: bool = False,
) -> pl.Expr:
    expression = pl.col(close) if isinstance(close, str) else close
    if variance:
        return _variance_expression(expression, weights)
    return expression.cast(pl.Float64).map_batches(
        lambda values: pl.Series(_weighted_windows(values.to_numpy(), weights, omit)),
        return_dtype=pl.Float64,
    )


@feature(
    name="linear_weighted_mean",
    category="trend",
    description="Finite linear weights 1..period with fixed denominator and NaN omission",
    formula="nansum(price * [1..period]) / (period * (period+1) / 2)",
)
def linear_weighted_mean(close: str, period: int = 20) -> pl.Expr:
    """Linear weighted mean; a missing price does not remove its denominator weight."""
    _period(period)
    return _weighted_expression(close, np.arange(1, period + 1, dtype=np.float64), omit=True)


def _exponential_weights(period: int, decay_rate: float) -> npt.NDArray[np.float64]:
    _period(period)
    if (
        isinstance(decay_rate, bool)
        or not isinstance(decay_rate, int | float)
        or not math.isfinite(decay_rate)
        or not 0 < decay_rate <= 1
    ):
        raise ValueError("decay_rate must be finite and in (0, 1]")
    # The common power cancels in both statistics. Newest weight 1 avoids
    # needless underflow from the reference's decay**(period+1)..decay**2.
    return np.power(float(decay_rate), np.arange(period - 1, -1, -1, dtype=np.float64))


@feature(
    name="finite_ewma",
    category="trend",
    description="Finite normalized exponential weighted mean; missing values propagate",
    formula="sum(price * decay_rate**age) / sum(decay_rate**age)",
)
def finite_ewma(close: str, period: int = 20, decay_rate: float = 0.9) -> pl.Expr:
    """Finite-window EWMA, distinct from recursively seeded ema."""
    return _weighted_expression(close, _exponential_weights(period, decay_rate))


@feature(
    name="finite_ewm_std",
    category="statistics",
    description="Finite exponential weighted standard deviation with weighted bias correction",
    formula="sqrt(weighted_variance * sum(w)**2 / (sum(w)**2-sum(w**2)))",
)
def finite_ewm_std(close: str, period: int = 20, decay_rate: float = 0.9) -> pl.Expr:
    """Bias-corrected finite weighted std; undefined effective degrees yield NaN."""
    return _weighted_expression(close, _exponential_weights(period, decay_rate), variance=True)


@feature(
    name="annualized_volatility",
    category="volatility",
    description="Population standard deviation of supplied returns with explicit annualization",
    input_type="returns",
    formula="nanstd(returns, ddof=0) * sqrt(annualization_factor)",
)
def annualized_volatility(
    returns: str | pl.Expr, period: int = 252, annualization_factor: float = 252.0
) -> pl.Expr:
    """Full-window population volatility; period counts supplied return observations."""
    _period(period)
    if (
        isinstance(annualization_factor, bool)
        or not isinstance(annualization_factor, int | float)
        or not math.isfinite(annualization_factor)
        or annualization_factor <= 0
    ):
        raise ValueError("annualization_factor must be positive and finite")
    volatility = _variance_expression(returns, np.ones(period), omit=True, bias_corrected=False)
    return volatility * math.sqrt(annualization_factor)


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _drawdown_windows(values: npt.NDArray[np.float64], period: int) -> npt.NDArray[np.float64]:
    result = np.full(len(values), np.nan)
    for end in range(period - 1, len(values)):
        peak = np.nan
        largest = -np.inf
        selected_peak = np.nan
        selected_end = np.nan
        for j in range(end - period + 1, end + 1):
            value = values[j]
            if np.isnan(value):
                continue
            peak = np.fmax(peak, value)
            difference = peak - value
            if difference > largest:
                largest = difference
                selected_peak, selected_end = peak, value
        if selected_end != 0:
            result[end] = (selected_peak - selected_end) / selected_end
        elif selected_peak > 0:
            result[end] = np.inf
    return result


@feature(
    name="finite_max_drawdown",
    category="risk",
    description="Pipeline absolute-drop selection with trough-price denominator",
    formula="(peak - trough) / trough at first largest absolute price drop",
)
def finite_max_drawdown(close: str, period: int = 20) -> pl.Expr:
    """Legacy finite drawdown definition, not portfolio/peak-denominator drawdown."""
    _period(period)
    return (
        pl.col(close)
        .cast(pl.Float64)
        .map_batches(
            lambda values: pl.Series(_drawdown_windows(values.to_numpy(), period)),
            return_dtype=pl.Float64,
        )
    )


@feature(
    name="finite_bollinger_bands",
    category="volatility",
    description="Finite NaN-omitting mean and population-standard-deviation bands",
    formula="middle=nanmean(close); lower/upper=middle +/- k*nanstd(close,ddof=0)",
)
def finite_bollinger_bands(close: str, period: int = 20, k: float = 2.0) -> dict[str, pl.Expr]:
    """Population bands requiring period rows; no smoothing or sample correction."""
    _period(period)
    if isinstance(k, bool) or not isinstance(k, int | float) or not math.isfinite(k) or k < 0:
        raise ValueError("k must be finite and nonnegative")
    middle = finite_mean(close, period)
    distance = _variance_expression(close, np.ones(period), omit=True, bias_corrected=False) * k
    return {"lower": middle - distance, "middle": middle, "upper": middle + distance}


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _extreme_position(
    values: npt.NDArray[np.float64], period: int, maximum: bool
) -> npt.NDArray[np.float64]:
    result = np.full(len(values), np.nan)
    for end in range(period - 1, len(values)):
        chosen = np.nan
        index = -1
        for j in range(period):
            value = values[end - period + 1 + j]
            if not np.isnan(value) and (
                index == -1 or (value > chosen if maximum else value < chosen)
            ):
                chosen, index = value, j
        if index >= 0:
            result[end] = 100.0 * index / (period - 1)
    return result


@feature(
    name="finite_aroon",
    category="momentum",
    description="Finite Aroon with first extreme ties and period-1 denominator",
    input_type="HL",
    normalized=True,
    value_range=(0.0, 100.0),
    formula="100 * first extreme position / (period-1)",
)
def finite_aroon(high: str, low: str, period: int = 25) -> dict[str, pl.Expr]:
    """First-tie Aroon fields; period counts prices, not changes."""
    _period(period, minimum=2)

    def position(column: str, maximum: bool) -> pl.Expr:
        return (
            pl.col(column)
            .fill_nan(None)
            .rank("dense")
            .cast(pl.Float64)
            .map_batches(
                lambda values: pl.Series(_extreme_position(values.to_numpy(), period, maximum)),
                return_dtype=pl.Float64,
            )
        )

    return {"down": position(low, False), "up": position(high, True)}


@feature(
    name="fast_stochastic",
    category="momentum",
    description="Unsmoothed finite stochastic K with NaN-omitting extrema",
    input_type="HLC",
    normalized=True,
    value_range=(0.0, 100.0),
    formula="100 * (close - nanmin(low)) / (nanmax(high) - nanmin(low))",
)
def fast_stochastic(high: str, low: str, close: str, period: int = 14) -> pl.Expr:
    """Unsmoothed K; zero ranges stay undefined rather than being replaced by zero."""
    _period(period)
    upper = pl.col(high).fill_nan(None).rolling_max(period, min_samples=1)
    lower = pl.col(low).fill_nan(None).rolling_min(period, min_samples=1)
    numerator = _paired_difference(pl.col(close).fill_nan(None), lower)
    denominator = _paired_difference(upper, lower)
    return _ready(100 * numerator / denominator, period)


@feature(
    name="finite_ichimoku",
    category="trend",
    description="Finite unshifted Ichimoku spans and explicit within-window chikou position",
    input_type="HLC",
)
def finite_ichimoku(
    high: str,
    low: str,
    close: str,
    period: int = 52,
    tenkan_length: int = 9,
    kijun_length: int = 26,
    chikou_position: int = 26,
) -> dict[str, pl.Expr]:
    """Pipeline's unshifted fields, not a chart's displaced Ichimoku cloud.

    A zero tenkan/kijun length selects the whole window, matching Python's
    legacy [-0:] slice. Chikou is an index from the oldest price in the window,
    not a negative shift; its index must be in [0, period).
    """
    _period(period)
    for name, value in (("tenkan_length", tenkan_length), ("kijun_length", kijun_length)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= period:
            raise ValueError(f"{name} must be an integer in [0, period]")
    if (
        isinstance(chikou_position, bool)
        or not isinstance(chikou_position, int)
        or not 0 <= chikou_position < period
    ):
        raise ValueError("chikou_position must be an integer in [0, period)")

    def midpoint(width: int) -> pl.Expr:
        return (_numeric(high).rolling_max(width) + _numeric(low).rolling_min(width)) / 2

    tenkan = midpoint(tenkan_length or period)
    kijun = midpoint(kijun_length or period)
    outputs = {
        "tenkan_sen": tenkan,
        "kijun_sen": kijun,
        "senkou_span_a": (tenkan + kijun) / 2,
        "senkou_span_b": midpoint(period),
        "chikou_span": pl.col(close).shift(period - 1 - chikou_position),
    }
    return {name: _ready(value, period) for name, value in outputs.items()}


@feature(
    name="finite_macd_signal",
    category="momentum",
    description="MACD signal formed from three finite normalized exponential windows",
    formula="finite_ewma(finite_ewma(fast)-finite_ewma(slow),signal)",
)
def finite_macd_signal(
    close: str, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> pl.Expr:
    """Finite MACD signal; length-one means use their sole observation."""
    for name, value in (
        ("fast_period", fast_period),
        ("slow_period", slow_period),
        ("signal_period", signal_period),
    ):
        try:
            _period(value)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc
    if slow_period <= fast_period:
        raise ValueError("slow_period must be greater than fast_period")

    def weights(width: int) -> npt.NDArray[np.float64]:
        if width == 1:
            return np.ones(1)
        return _exponential_weights(width, 1 - 2 / (1 + width))

    difference = _weighted_expression(close, weights(fast_period)) - _weighted_expression(
        close, weights(slow_period)
    )
    return _weighted_expression(difference, weights(signal_period))


@feature(
    name="finite_true_range",
    category="volatility",
    description="Two-price true range with NaN-omitting horizontal components",
    input_type="HLC",
    formula="nanmax(high-low, abs(high-prev_close), abs(low-prev_close))",
)
def finite_true_range(high: str, low: str, close: str) -> pl.Expr:
    """One-bar true range after its previous price; no arbitrary larger-window reduction."""
    upper, lower = pl.col(high).fill_nan(None), pl.col(low).fill_nan(None)
    previous = pl.col(close).fill_nan(None).shift(1)
    return _ready(
        pl.max_horizontal(
            _paired_difference(upper, lower),
            _paired_difference(upper, previous).abs(),
            _paired_difference(lower, previous).abs(),
        ),
        2,
    )
