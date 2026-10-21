"""Zipline factor-name recipes over native expressions, without importing Zipline.

These return Polars expressions or named output expressions, not Pipeline Terms.
Validate the input feed/grid once, keep historical inputs, then apply output
screens. Dated source eligibility belongs to native feature preparation.
"""

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

from ml4t.engineer.features import events, paired, rolling
from ml4t.engineer.features.utils import arithmetic

Expression = pl.Expr | dict[str, pl.Expr]

__all__ = ["factor_expression", "return_factor_features"]


def return_factor_features(
    data: pl.DataFrame | pl.LazyFrame,
    name: str,
    *,
    target: Mapping[str, Any],
    returns_length: int = 2,
    correlation_length: int | None = None,
    regression_length: int | None = None,
    allowed_missing_percentage: float | None = None,
    mask: str | pl.Expr | None = None,
    **native_options: Any,
) -> pl.DataFrame | pl.LazyFrame:
    """Map return-factor constructors to native keyed price preparation.

    target contains stable physical IDs, not an opaque Zipline Asset. Declare
    price_basis, feed_spec/calendar and column mappings through native_options.
    The mask changes the statistic output only, retaining target input history.
    This is a dataframe recipe, not a Pipeline interpreter or legacy constructor.
    """
    methods = {
        "RollingPearsonOfReturns": "pearson",
        "RollingSpearmanOfReturns": "spearman",
        "RollingLinearRegressionOfReturns": "regression",
        "SimpleBeta": "beta",
    }
    if name not in methods:
        raise ValueError(f"Unsupported return-factor recipe {name!r}")
    method = methods[name]
    rolling._period(returns_length, minimum=2)
    if mask is not None and not isinstance(mask, str | pl.Expr):
        raise TypeError("mask must be an input column name or expression")
    correlation = method in {"pearson", "spearman"}
    width = correlation_length if correlation else regression_length
    wrong_length = regression_length if correlation else correlation_length
    if width is None or wrong_length is not None:
        raise ValueError("Declare only the correlation_length or regression_length for this recipe")
    rolling._period(width, minimum=3 if method == "beta" else 2)
    allowed_missing = 0
    if method == "beta":
        if returns_length != 2:
            raise ValueError("SimpleBeta uses daily (two-price) returns")
        fraction = 0.25 if allowed_missing_percentage is None else allowed_missing_percentage
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, int | float)
            or not 0 <= fraction <= 1
        ):
            raise ValueError("allowed_missing_percentage must be in [0,1]")
        allowed_missing = int(width * fraction)
    elif allowed_missing_percentage is not None:
        raise ValueError("Only SimpleBeta has allowed_missing_percentage")
    output = paired.target_return_features(
        data,
        target=target,
        period=width,
        returns_periods=returns_length - 1,
        method=method,
        allowed_missing=allowed_missing,
        **native_options,
    )
    if mask is None:
        return output
    predicate = pl.col(mask) if isinstance(mask, str) else mask
    column = f"{native_options.get('output', 'target')}_{method}"
    return output.with_columns(pl.when(predicate).then(pl.col(column)).alias(column))


@dataclass(frozen=True)
class _Recipe:
    function: Callable[..., Expression]
    columns: tuple[str, ...]
    window: int | None
    required_inputs: bool = False
    order: tuple[int, ...] | None = None
    window_arg: str | None = "period"
    adjustment: int = 0


def _clip(value: str, min_bound: float, max_bound: float) -> pl.Expr:
    return pl.col(value).clip(min_bound, max_bound)


def _latest(value: str) -> pl.Expr:
    return pl.col(value)


_RECIPES = {
    "Returns": _Recipe(arithmetic.returns, ("close",), None, window_arg="periods", adjustment=-1),
    "DailyReturns": _Recipe(arithmetic.returns, ("close",), 2, window_arg="periods", adjustment=-1),
    "PercentChange": _Recipe(
        rolling.absolute_percent_change, ("close",), None, True, adjustment=-1
    ),
    "RateOfChangePercentage": _Recipe(
        arithmetic.percentage_change, ("close",), None, True, window_arg="periods", adjustment=-1
    ),
    "SimpleMovingAverage": _Recipe(rolling.finite_mean, ("close",), None, True),
    "WeightedAverageValue": _Recipe(rolling.finite_weighted_mean, ("close", "volume"), None, True),
    "VWAP": _Recipe(rolling.finite_weighted_mean, ("close", "volume"), None),
    "AverageDollarVolume": _Recipe(rolling.average_dollar_volume, ("close", "volume"), None),
    "LinearWeightedMovingAverage": _Recipe(rolling.linear_weighted_mean, ("close",), None, True),
    "MaxDrawdown": _Recipe(rolling.finite_max_drawdown, ("close",), None, True),
    "AnnualizedVolatility": _Recipe(rolling.annualized_volatility, ("returns",), 252),
    "RSI": _Recipe(rolling.rsi_cutler, ("close",), 15, adjustment=-1),
    "BollingerBands": _Recipe(rolling.finite_bollinger_bands, ("close",), None),
    "Aroon": _Recipe(rolling.finite_aroon, ("low", "high"), None, order=(1, 0)),
    "FastStochasticOscillator": _Recipe(
        rolling.fast_stochastic, ("close", "low", "high"), 14, order=(2, 1, 0)
    ),
    "IchimokuKinkoHyo": _Recipe(rolling.finite_ichimoku, ("high", "low", "close"), 52),
    "TrueRange": _Recipe(rolling.finite_true_range, ("high", "low", "close"), 2, window_arg=None),
    "Clip": _Recipe(_clip, ("close",), 1, True, window_arg=None),
    "Latest": _Recipe(_latest, ("close",), 1, True, window_arg=None),
    "RollingPearson": _Recipe(paired.rolling_pearson, ("value", "target"), None, True),
    "RollingSpearman": _Recipe(paired.rolling_spearman, ("value", "target"), None, True),
    "RollingLinearRegression": _Recipe(paired.rolling_regression, ("value", "target"), None, True),
    "SimpleBeta": _Recipe(paired.rolling_beta, ("returns", "target_returns"), None, True),
    "ExponentialWeightedMovingAverage": _Recipe(rolling.finite_ewma, ("close",), None, True),
    "ExponentialWeightedMovingStdDev": _Recipe(rolling.finite_ewm_std, ("close",), None, True),
    "MovingAverageConvergenceDivergenceSignal": _Recipe(
        rolling.finite_macd_signal, ("close",), None, window_arg=None
    ),
}
_ALIASES = {
    "EWMA": "ExponentialWeightedMovingAverage",
    "EWMSTD": "ExponentialWeightedMovingStdDev",
    "MACDSignal": "MovingAverageConvergenceDivergenceSignal",
}
_FACTORIES = {
    "from_span": "span",
    "from_halflife": "halflife",
    "from_center_of_mass": "center_of_mass",
}
_EVENTS: dict[str, Literal["since", "until"]] = {
    "BusinessDaysSincePreviousEvent": "since",
    "BusinessDaysUntilNextEvent": "until",
}


def _inputs(inputs: Sequence[str] | None, columns: tuple[str, ...], required: bool) -> list[str]:
    if inputs is None:
        if required:
            raise ValueError("This factor requires explicit input column names")
        return list(columns)
    if (
        not isinstance(inputs, Sequence)
        or isinstance(inputs, str)
        or len(inputs) != len(columns)
        or any(not isinstance(c, str) or not c for c in inputs)
    ):
        raise ValueError(f"inputs must contain {len(columns)} nonempty column names")
    return list(inputs)


def factor_expression(
    name: str,
    *,
    inputs: Sequence[str] | None = None,
    window_length: int | None = None,
    mask: str | pl.Expr | None = None,
    **params: Any,
) -> Expression:
    """Map a supported factor name to one native calculation and explicit field names.

    window_length counts prices/inputs as in Pipeline; return/RSI recipes translate
    N prices to N-1 changes. A mask controls output only and never removes warmup
    history. False/null predicate values yield native null outputs; mask=None
    leaves outputs unmasked. Factory suffixes support
    the three EWMA/EWMSTD constructors. Opaque Terms/Assets are not executed.
    """
    if not isinstance(name, str):
        raise TypeError("name must identify a supported factor recipe")
    pieces = name.split(".")
    if len(pieces) > 2:
        raise ValueError("Unsupported factor constructor")
    canonical = _ALIASES.get(pieces[0], pieces[0])
    factory = pieces[1] if len(pieces) == 2 else None
    if factory is not None and (
        factory not in _FACTORIES
        or canonical not in {"ExponentialWeightedMovingAverage", "ExponentialWeightedMovingStdDev"}
    ):
        raise ValueError("Only the declared finite exponential constructors are supported")
    if mask is not None and not isinstance(mask, str | pl.Expr):
        raise TypeError("mask must be an input column name or expression")
    if canonical in _EVENTS:
        if isinstance(window_length, bool) or window_length not in (None, 0):
            raise ValueError("Event weekday counts do not use rolling windows")
        column = _inputs(inputs, ("event_date",), True)[0]
        output: Expression = events.event_business_days(
            column, direction=_EVENTS[canonical], **params
        )
    else:
        if canonical not in _RECIPES:
            raise ValueError(
                f"Unsupported numeric recipe {name!r}; port custom calculations to native expressions/registered features"
            )
        route = _RECIPES[canonical]
        columns = _inputs(inputs, route.columns, route.required_inputs)
        if route.order is not None:
            columns = [columns[index] for index in route.order]
        options = dict(params)
        if canonical.startswith("ExponentialWeightedMoving"):
            if factory is not None and _FACTORIES[factory] not in options:
                raise ValueError(f"{factory} requires {_FACTORIES[factory]}")
            selectors = {
                key: options.pop(key)
                for key in ("decay_rate", "span", "halflife", "center_of_mass")
                if key in options
            }
            options["decay_rate"] = rolling.exponential_decay(**selectors)
        if canonical == "IchimokuKinkoHyo":
            for old, new in (
                ("tenkan_sen_length", "tenkan_length"),
                ("kijun_sen_length", "kijun_length"),
                ("chikou_span_length", "chikou_position"),
            ):
                if old in options:
                    if new in options:
                        raise ValueError(f"Conflicting {old}/{new}")
                    options[new] = options.pop(old)
        lengths = (
            ("correlation_length",)
            if canonical in {"RollingPearson", "RollingSpearman"}
            else (
                ("regression_length",)
                if canonical in {"RollingLinearRegression", "SimpleBeta"}
                else ()
            )
        )
        for legacy in lengths:
            if legacy in options:
                if window_length is not None:
                    raise ValueError("Declare one window_length or legacy statistical length")
                window_length = options.pop(legacy)
        width = window_length if window_length is not None else route.window
        if route.window_arg is not None:
            if width is None:
                raise ValueError("Declare window_length")
            rolling._period(width)
            if route.window_arg in options:
                raise ValueError("Use window_length rather than a second native window argument")
            options[route.window_arg] = width + route.adjustment
        elif canonical == "MovingAverageConvergenceDivergenceSignal":
            if window_length is not None:
                raise ValueError("MACDSignal derives its window from the three periods")
        else:
            assert width is not None
            rolling._period(width)
            if canonical == "TrueRange" and width != 2:
                raise ValueError("TrueRange requires its two-price definition")
        if canonical == "SimpleBeta":
            assert width is not None
            if width < 3:
                raise ValueError("SimpleBeta regression_length must be >=3")
            fraction = options.pop("allowed_missing_percentage", 0.25)
            if (
                isinstance(fraction, bool)
                or not isinstance(fraction, int | float)
                or not 0 <= fraction <= 1
            ):
                raise ValueError("allowed_missing_percentage must be in [0,1]")
            if "allowed_missing" in options:
                raise ValueError("Declare allowed_missing_percentage only once")
            options["allowed_missing"] = int(width * fraction)
        signature = inspect.signature(route.function)
        column_params = list(signature.parameters)[: len(columns)]
        unknown = set(options) - (set(signature.parameters) - set(column_params))
        if unknown:
            raise ValueError(f"Unknown factor options: {sorted(unknown)}")
        if canonical == "AnnualizedVolatility" and inputs is None:
            output = route.function(arithmetic.returns("close"), **options)
            assert isinstance(output, pl.Expr) and width is not None
            output = rolling._ready(output, width + 1)
        else:
            output = route.function(*columns, **options)
        if canonical in {"Clip", "Latest"}:
            assert isinstance(output, pl.Expr) and width is not None
            output = rolling._ready(output, width)
    if mask is None:
        return output
    predicate = pl.col(mask) if isinstance(mask, str) else mask
    if isinstance(output, dict):
        return {key: pl.when(predicate).then(value) for key, value in output.items()}
    return pl.when(predicate).then(output)
