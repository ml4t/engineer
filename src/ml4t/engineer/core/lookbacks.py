"""Authoritative warm-up calculations for registered features."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

LookbackCalculator = Callable[[Mapping[str, Any]], int]


def _constant(value: int) -> LookbackCalculator:
    return lambda _params: value


def _parameter(name: str, adjustment: int = 0) -> LookbackCalculator:
    return lambda params: int(params[name]) + adjustment


def _scaled_parameter(name: str, multiplier: int, adjustment: int = 0) -> LookbackCalculator:
    return lambda params: multiplier * int(params[name]) + adjustment


def _maximum(*names: str, adjustment: int = 0) -> LookbackCalculator:
    return lambda params: max(int(params[name]) for name in names) + adjustment


def _sum(*names: str, adjustment: int = 0) -> LookbackCalculator:
    return lambda params: sum(int(params[name]) for name in names) + adjustment


def _half_window(name: str = "window") -> LookbackCalculator:
    return lambda params: int(params[name]) // 2 - 1


def _optional_half_window(params: Mapping[str, Any]) -> int:
    window = params["window"]
    return 0 if window is None else int(window) // 2 - 1


def _maximum_sequence(name: str, default: Sequence[int], adjustment: int = 0) -> LookbackCalculator:
    def calculate(params: Mapping[str, Any]) -> int:
        values = params[name]
        resolved = default if values is None else values
        return max(int(value) for value in resolved) + adjustment

    return calculate


def _stochastic(params: Mapping[str, Any]) -> int:
    return int(params["fastk_period"]) + 2 * int(params["slowk_period"]) - 3


def _stochf(params: Mapping[str, Any]) -> int:
    return int(params["fastk_period"]) + int(params["fastd_period"]) - 2


def _stochrsi(params: Mapping[str, Any]) -> int:
    return int(params["timeperiod"]) + int(params["fastk_period"]) + int(params["fastd_period"]) - 2


def _rolling_cv_zscore(params: Mapping[str, Any]) -> int:
    return int(params["window"]) * int(params["lookback_multiplier"]) // 2 - 1


def _variance_ratio(params: Mapping[str, Any]) -> int:
    return max(int(params["window"]) // 2, int(params["q"]) + 2) - 1


def _ulcer_index(params: Mapping[str, Any]) -> int:
    minimum = int(params["window"]) // 2
    return 2 * minimum - 2


FEATURE_LOOKBACKS: dict[str, LookbackCalculator] = {
    # Volume
    "ad": _constant(0),
    "adosc": _parameter("slowperiod", -1),
    "obv": _constant(0),
    # Math
    "maximum": _parameter("timeperiod", -1),
    "minimum": _parameter("timeperiod", -1),
    "summation": _parameter("timeperiod", -1),
    # Price transforms
    "avgprice": _constant(0),
    "medprice": _constant(0),
    "midprice": _parameter("timeperiod", -1),
    "typprice": _constant(0),
    "wclprice": _constant(0),
    # Trend
    "dema": _scaled_parameter("period", 2, -2),
    "donchian_channels": _parameter("period", -1),
    "ema": _parameter("period", -1),
    "kama": _parameter("timeperiod"),
    "midpoint": _parameter("timeperiod", -1),
    "sma": _parameter("period", -1),
    "t3": _scaled_parameter("timeperiod", 6, -6),
    "tema": _scaled_parameter("period", 3, -3),
    "trima": _parameter("period", -1),
    "wma": _parameter("period", -1),
    # Momentum
    "adx": _scaled_parameter("period", 2, -1),
    "adxr": _scaled_parameter("timeperiod", 3, -2),
    "apo": _parameter("slow_period", -1),
    "aroon": _parameter("timeperiod"),
    "aroonosc": _parameter("timeperiod"),
    "bop": _constant(0),
    "cci": _parameter("period", -1),
    "cmo": _parameter("timeperiod"),
    "dx": _parameter("timeperiod"),
    "imi": _parameter("timeperiod", -1),
    "macd": _parameter("slow_period", 7),
    "macdfix": _parameter("signalperiod", 24),
    "mfi": _parameter("period"),
    "minus_di": _parameter("timeperiod"),
    "minus_dm": _parameter("timeperiod", -1),
    "mom": lambda params: 10 if params["period"] is None else int(params["period"]),
    "plus_di": _parameter("timeperiod"),
    "plus_dm": _parameter("timeperiod", -1),
    "ppo": _parameter("slow_period", -1),
    "roc": _parameter("period"),
    "rocp": _parameter("timeperiod"),
    "rocr": _parameter("timeperiod"),
    "rocr100": _parameter("timeperiod"),
    "rsi": _parameter("period"),
    "sar": _constant(1),
    "stochastic": _stochastic,
    "stochf": _stochf,
    "stochrsi": _stochrsi,
    "trix": _scaled_parameter("timeperiod", 3, -2),
    "ultosc": _maximum("timeperiod1", "timeperiod2", "timeperiod3"),
    "willr": _parameter("period", -1),
    # Volatility
    "atr": _parameter("period"),
    "bollinger_bands": _parameter("period", -1),
    "conditional_volatility_ratio": _parameter("period", -1),
    "ewma_volatility": _constant(1),
    "garch_forecast": _constant(0),
    "garman_klass_volatility": _parameter("period", -1),
    "natr": _parameter("period"),
    "parkinson_volatility": _parameter("period", -1),
    "realized_volatility": _parameter("period", -1),
    "rogers_satchell_volatility": _parameter("period", -1),
    "trange": _constant(1),
    "volatility_of_volatility": _sum("vol_period", "vov_period"),
    "volatility_percentile_rank": _sum("period", "lookback", adjustment=-1),
    "volatility_regime_probability": _sum("period", "lookback", adjustment=-1),
    "yang_zhang_volatility": _parameter("period"),
    # Microstructure
    "amihud_illiquidity": _parameter("period", -1),
    "bid_ask_imbalance": _parameter("period", -1),
    "book_depth_ratio": _constant(0),
    "effective_tick_rule": _constant(0),
    "kyle_lambda": _scaled_parameter("period", 2, -2),
    "order_flow_imbalance": _constant(0),
    "price_impact_ratio": _parameter("period", -1),
    "quote_stuffing_indicator": _scaled_parameter("period", 4, -1),
    "realized_spread": _parameter("period", -1),
    "roll_spread_estimator": _scaled_parameter("period", 2),
    "trade_intensity": _parameter("period", -1),
    "volume_at_price_ratio": _parameter("period", -1),
    "volume_synchronicity": _scaled_parameter("period", 2, -1),
    "volume_weighted_price_momentum": _parameter("period"),
    "weighted_mid_price": _constant(0),
    # ML
    "create_lag_features": _maximum_sequence("lags", [1, 2, 3, 5, 10]),
    "cyclical_encode": _constant(0),
    "directional_targets": _constant(0),
    "ffdiff": _constant(0),
    "fourier_features": _constant(0),
    "interaction_features": _constant(0),
    "multi_horizon_returns": _maximum_sequence("horizons", [1, 5, 10, 30, 60]),
    "percentile_rank_features": _maximum_sequence("windows", [20, 50, 100], -1),
    "regime_conditional_features": _constant(0),
    "rolling_entropy": _half_window(),
    "rolling_entropy_lz": _half_window(),
    "rolling_entropy_plugin": _half_window(),
    "time_decay_weights": _constant(0),
    "volatility_adjusted_returns": _parameter("vol_lookback", -1),
    # Regime
    "choppiness_index": _parameter("period", -1),
    "fractal_efficiency": _parameter("period"),
    "hurst_exponent": _parameter("period", -1),
    "trend_intensity_index": _parameter("period", -1),
    # Risk
    "downside_deviation": _half_window(),
    "higher_moments": _parameter("window", -1),
    "maximum_drawdown": _optional_half_window,
    "risk_adjusted_returns": _half_window(),
    "tail_ratio": _half_window(),
    "ulcer_index": _ulcer_index,
    # Statistics
    "avgdev": _parameter("timeperiod", -1),
    "coefficient_of_variation": _half_window(),
    "linearreg": _parameter("timeperiod", -1),
    "linearreg_angle": _parameter("timeperiod", -1),
    "linearreg_intercept": _parameter("timeperiod", -1),
    "linearreg_slope": _parameter("timeperiod", -1),
    "rolling_cv_zscore": _rolling_cv_zscore,
    "rolling_drift": _half_window(),
    "rolling_kl_divergence": _half_window(),
    "rolling_wasserstein": _half_window(),
    "stddev": _parameter("period", -1),
    "tsf": _parameter("timeperiod", -1),
    "var": _parameter("timeperiod", -1),
    "variance_ratio": _variance_ratio,
}


def bind_feature_lookback(
    name: str,
    default_parameters: Mapping[str, Any],
    declared: int | str | Callable[..., int] | None,
) -> Callable[..., int]:
    """Bind a first-party calculation or normalize a third-party declaration."""
    calculator = FEATURE_LOOKBACKS.get(name)
    if calculator is not None:
        if declared is not None:
            raise TypeError(
                f"Feature '{name}' has an authoritative lookback calculation; "
                "remove the decorator lookback argument"
            )

        def calculate(**overrides: Any) -> int:
            value = calculator({**default_parameters, **overrides})
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Feature '{name}' produced invalid lookback {value!r}")
            return value

        return calculate

    if declared is None:
        raise TypeError(f"Feature '{name}' must declare a lookback calculation")
    if isinstance(declared, int):
        return lambda **_overrides: declared
    if isinstance(declared, str):
        if not declared.strip():
            raise TypeError(f"Feature '{name}': lookback parameter name cannot be empty")
        return lambda **overrides: int(overrides.get(declared, default_parameters.get(declared, 1)))
    if callable(declared):
        return declared
    raise TypeError(f"Feature '{name}' has invalid lookback declaration {declared!r}")


__all__ = ["FEATURE_LOOKBACKS", "LookbackCalculator", "bind_feature_lookback"]
