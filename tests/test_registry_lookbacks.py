"""Runtime contracts for feature registry lookback metadata."""

import inspect
import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.engineer import compute_features
from ml4t.engineer.core.dispatch import COLUMN_ARG_MAP
from ml4t.engineer.core.lookbacks import FEATURE_LOOKBACKS
from ml4t.engineer.core.registry import get_registry


@pytest.fixture(scope="module")
def lookback_data() -> pl.DataFrame:
    """Create finite, non-degenerate inputs long enough for every default."""
    row_count = 700
    index = np.arange(row_count, dtype=np.float64)
    rng = np.random.default_rng(20260731)
    returns = 0.001 * np.sin(index / 7.0) + rng.normal(0.0, 0.003, row_count)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = close * (1.0 + rng.normal(0.0, 0.001, row_count))
    spread = 0.2 + 0.05 * np.sin(index / 11.0)
    start = datetime(2024, 1, 1)

    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=int(offset)) for offset in index],
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": 1000.0 + 100.0 * np.sin(index / 5.0) + index % 17,
            "returns": returns,
            "bid_price": close - 0.05,
            "ask_price": close + 0.05,
            "bid_size": 500.0 + index % 23,
            "ask_size": 450.0 + (index * 3) % 29,
            "num_trades": 50.0 + index % 11,
        }
    )


def _value_is_usable(series: pl.Series, row: int) -> bool:
    value = series[row]
    if value is None:
        return False
    if series.dtype.base_type() == pl.Struct:
        return all(
            _value_is_usable(series.struct.field(field.name), row) for field in series.dtype.fields
        )
    return not isinstance(value, float) or math.isfinite(value)


def _first_usable_row(
    data: pl.DataFrame,
    feature_name: str,
    params: dict[str, object] | None = None,
) -> int | None:
    spec: str | dict[str, object]
    if params is None:
        spec = feature_name
    else:
        spec = {"name": feature_name, "params": params}
    result = compute_features(data, [spec])
    output_columns = [name for name in result.columns if name not in data.columns]
    return next(
        (
            row
            for row in range(result.height)
            if all(_value_is_usable(result[name], row) for name in output_columns)
        ),
        None,
    )


def test_first_party_lookback_table_covers_registry_exactly() -> None:
    registry_names = set(get_registry().list_all())
    assert set(FEATURE_LOOKBACKS) == registry_names


def test_metadata_defaults_match_function_defaults() -> None:
    for name in get_registry().list_all():
        metadata = get_registry().get(name)
        assert metadata is not None
        signature = inspect.signature(metadata.func)

        for parameter_name, parameter in signature.parameters.items():
            if parameter_name in COLUMN_ARG_MAP:
                assert parameter_name not in metadata.parameters
            elif parameter.default is not inspect.Parameter.empty:
                assert metadata.parameters[parameter_name] == parameter.default
            else:
                assert parameter_name in metadata.parameters, (
                    f"{name}.{parameter_name} has no execution default"
                )


def test_default_lookbacks_match_first_usable_outputs(lookback_data: pl.DataFrame) -> None:
    mismatches = []
    for name in get_registry().list_all():
        metadata = get_registry().get(name)
        assert metadata is not None
        expected = metadata.lookback()
        actual = _first_usable_row(lookback_data, name)
        if actual != expected:
            mismatches.append((name, expected, actual))

    assert not mismatches, f"Lookback mismatches: {mismatches}"


CONFIGURED_LOOKBACK_CASES: dict[str, dict[str, object]] = {
    "adosc": {"fastperiod": 2, "slowperiod": 5},
    "adx": {"period": 5},
    "adxr": {"timeperiod": 5},
    "amihud_illiquidity": {"period": 5},
    "apo": {"fast_period": 3, "slow_period": 6},
    "aroon": {"timeperiod": 5},
    "aroonosc": {"timeperiod": 5},
    "atr": {"period": 5},
    "avgdev": {"timeperiod": 5},
    "bid_ask_imbalance": {"period": 5},
    "bollinger_bands": {"period": 5},
    "cci": {"period": 5},
    "choppiness_index": {"period": 5},
    "cmo": {"timeperiod": 5},
    "coefficient_of_variation": {"window": 10},
    "conditional_volatility_ratio": {"period": 5},
    "create_lag_features": {"lags": [2, 4]},
    "dema": {"period": 5},
    "donchian_channels": {"period": 5},
    "downside_deviation": {"window": 20},
    "dx": {"timeperiod": 5},
    "ema": {"period": 5},
    "fractal_efficiency": {"period": 5},
    "garman_klass_volatility": {"period": 5},
    "higher_moments": {"window": 20},
    "hurst_exponent": {"period": 20},
    "imi": {"timeperiod": 5},
    "kama": {"timeperiod": 5},
    "kyle_lambda": {"period": 5},
    "linearreg": {"timeperiod": 5},
    "linearreg_angle": {"timeperiod": 5},
    "linearreg_intercept": {"timeperiod": 5},
    "linearreg_slope": {"timeperiod": 5},
    "macd": {"fast_period": 3, "slow_period": 6},
    "macdfix": {"signalperiod": 5},
    "maximum": {"timeperiod": 5},
    "maximum_drawdown": {"window": 20},
    "mfi": {"period": 5},
    "midpoint": {"timeperiod": 5},
    "midprice": {"timeperiod": 5},
    "minimum": {"timeperiod": 5},
    "minus_di": {"timeperiod": 5},
    "minus_dm": {"timeperiod": 5},
    "mom": {"period": 5},
    "multi_horizon_returns": {"horizons": [2, 4]},
    "natr": {"period": 5},
    "parkinson_volatility": {"period": 5},
    "percentile_rank_features": {"windows": [5, 10]},
    "plus_di": {"timeperiod": 5},
    "plus_dm": {"timeperiod": 5},
    "ppo": {"fast_period": 3, "slow_period": 6},
    "price_impact_ratio": {"period": 5},
    "quote_stuffing_indicator": {"period": 3},
    "realized_spread": {"period": 5},
    "realized_volatility": {"period": 5},
    "risk_adjusted_returns": {"window": 20},
    "roc": {"period": 5},
    "rocp": {"timeperiod": 5},
    "rocr": {"timeperiod": 5},
    "rocr100": {"timeperiod": 5},
    "rogers_satchell_volatility": {"period": 5},
    "roll_spread_estimator": {"period": 5},
    "rolling_cv_zscore": {"window": 10, "lookback_multiplier": 4},
    "rolling_drift": {"window": 20},
    "rolling_entropy": {"window": 10},
    "rolling_entropy_lz": {"window": 20},
    "rolling_entropy_plugin": {"window": 10},
    "rolling_kl_divergence": {"window": 20},
    "rolling_wasserstein": {"window": 20},
    "rsi": {"period": 5},
    "sma": {"period": 5},
    "stddev": {"period": 5},
    "stochastic": {"fastk_period": 5, "slowk_period": 2, "slowd_period": 3},
    "stochf": {"fastk_period": 5, "fastd_period": 3},
    "stochrsi": {"timeperiod": 5, "fastk_period": 3, "fastd_period": 2},
    "summation": {"timeperiod": 5},
    "t3": {"timeperiod": 3},
    "tail_ratio": {"window": 20},
    "tema": {"period": 5},
    "trade_intensity": {"period": 5},
    "trend_intensity_index": {"period": 5},
    "trima": {"period": 5},
    "trix": {"timeperiod": 5},
    "tsf": {"timeperiod": 5},
    "ulcer_index": {"window": 20},
    "ultosc": {"timeperiod1": 3, "timeperiod2": 5, "timeperiod3": 7},
    "var": {"timeperiod": 5},
    "variance_ratio": {"window": 20, "q": 3},
    "volatility_adjusted_returns": {"vol_lookback": 5},
    "volatility_of_volatility": {"vol_period": 5, "vov_period": 5},
    "volatility_percentile_rank": {"period": 5, "lookback": 10},
    "volatility_regime_probability": {"period": 5, "lookback": 10},
    "volume_at_price_ratio": {"period": 5},
    "volume_synchronicity": {"period": 5},
    "volume_weighted_price_momentum": {"period": 5},
    "willr": {"period": 5},
    "wma": {"period": 5},
    "yang_zhang_volatility": {"period": 5},
}


@pytest.mark.parametrize(
    ("feature_name", "params"),
    CONFIGURED_LOOKBACK_CASES.items(),
    ids=CONFIGURED_LOOKBACK_CASES,
)
def test_configured_lookbacks_match_first_usable_outputs(
    lookback_data: pl.DataFrame,
    feature_name: str,
    params: dict[str, object],
) -> None:
    metadata = get_registry().get(feature_name)
    assert metadata is not None
    assert _first_usable_row(lookback_data, feature_name, params) == metadata.lookback(**params)


def test_optional_and_alternate_paths_match_lookbacks(lookback_data: pl.DataFrame) -> None:
    cases: list[tuple[str, dict[str, object]]] = [
        ("maximum_drawdown", {"window": None}),
        ("quote_stuffing_indicator", {"period": 3, "num_trades": "num_trades"}),
    ]
    for feature_name, params in cases:
        metadata = get_registry().get(feature_name)
        assert metadata is not None
        assert _first_usable_row(lookback_data, feature_name, params) == metadata.lookback(**params)
