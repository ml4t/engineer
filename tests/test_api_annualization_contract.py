from datetime import UTC, datetime, timedelta
from math import sqrt
from statistics import mean, stdev

import numpy as np
import pandas_market_calendars as mcal
import polars as pl
import pytest
import yaml
from ml4t.specs import FeedSpec

from ml4t.engineer import compute_features

RETURNS = np.array(
    [
        0.012,
        -0.008,
        0.006,
        -0.003,
        0.015,
        -0.011,
        0.004,
        0.009,
        -0.005,
        0.007,
        -0.002,
        0.013,
        -0.009,
        0.005,
        0.011,
        -0.006,
        0.008,
        -0.004,
        0.014,
        -0.007,
    ]
)


def prices(times):
    frames = []
    for symbol, multiplier in [("A", 1.0), ("B", 1.7)]:
        close = 100 * np.cumprod(np.r_[1.0, 1.0 + RETURNS * multiplier])
        frames.append(
            pl.DataFrame(
                {
                    "timestamp": times,
                    "symbol": symbol,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1.0,
                }
            )
        )
    return pl.concat(frames)


def feed(calendar):
    return FeedSpec(
        time_contract_version=2,
        calendar=calendar,
        entity_col="symbol",
        data_frequency="1h",
        timestamp_semantics="bar_close",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label="right",
        partial_bar_policy="include",
    )


def actual_times(calendar):
    schedule = mcal.get_calendar(calendar).schedule("2025-07-07", "2025-07-17")
    return list(
        mcal.date_range(schedule, frequency="1h", closed="right", force_close=True)[
            :21
        ].to_pydatetime()
    )


def request_file(tmp_path, requests, serialized):
    if not serialized:
        return requests
    path = tmp_path / "features.yaml"
    path.write_text(yaml.safe_dump({"features": requests}))
    return path


def materialize(frame):
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


@pytest.mark.parametrize("calendar", ["NYSE", "24/7"])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("serialized", [False, True])
def test_real_price_pipeline_uses_explicit_scale_not_calendar_or_bar_frequency(
    tmp_path, calendar, lazy, serialized
):
    raw = prices(actual_times(calendar))
    data = raw.lazy() if lazy else raw
    checked = compute_features(data, [], feed_spec=feed(calendar))
    returns = checked.with_columns(pl.col("close").pct_change().over("symbol").alias("returns"))
    requests = [
        {
            "name": "realized_volatility",
            "params": {"period": 3, "annualize": False},
            "output": "raw",
        },
        {"name": "realized_volatility", "params": {"period": 3}, "output": "legacy_default"},
    ]
    requests += [
        {
            "name": "realized_volatility",
            "params": {"period": 3, "trading_periods": scale},
            "output": f"v{scale}",
        }
        for scale in (252, 365, 8760)
    ]
    output = materialize(
        compute_features(
            returns, request_file(tmp_path, requests, serialized), feed_spec=feed(calendar)
        )
    )
    assert output.select(raw.columns).equals(raw)
    for symbol in ("A", "B"):
        asset = output.filter(pl.col("symbol") == symbol)
        values = asset["returns"].drop_nulls().to_list()
        expected = stdev(values[-3:])
        assert asset["raw"][-1] == pytest.approx(expected, rel=1e-11)
        assert asset["legacy_default"][-1] == pytest.approx(expected * sqrt(252), rel=1e-11)
        for scale in (252, 365, 8760):
            assert asset[f"v{scale}"][-1] == pytest.approx(expected * sqrt(scale), rel=1e-11)


@pytest.mark.parametrize("lazy", [False, True])
def test_row_mode_annualization_is_invariant_to_irregular_and_reordered_timestamps(
    monkeypatch, lazy
):
    import sys

    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    start = datetime(2025, 7, 11, tzinfo=UTC)
    outputs = []
    requests = [
        {
            "name": "realized_volatility",
            "params": {"period": 3, "trading_periods": scale},
            "output": f"v{scale}",
        }
        for scale in (252, 365, 8760)
    ]
    for offsets in [list(range(21)), [i * i for i in range(21)], list(reversed(range(21)))]:
        raw = prices([start + timedelta(hours=h) for h in offsets])
        data = raw.lazy() if lazy else raw
        returns = data.with_columns(pl.col("close").pct_change().over("symbol").alias("returns"))
        result = materialize(
            compute_features(returns, requests, feed_spec=feed("NYSE"), calendar=None)
        )
        assert result.select(raw.columns).equals(raw)
        outputs.append(result.select([f"v{scale}" for scale in (252, 365, 8760)]))
    assert outputs[0].equals(outputs[1])
    assert outputs[0].equals(outputs[2])


@pytest.mark.parametrize("scale", [252, 365, 8760])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("serialized", [False, True])
def test_public_risk_workflow_retains_explicit_annualization_and_period_risk_free_rate(
    tmp_path, scale, lazy, serialized
):
    raw = prices(actual_times("24/7"))
    data = raw.lazy() if lazy else raw
    checked = compute_features(data, [], feed_spec=feed("24/7"))
    returns = checked.with_columns(pl.col("close").pct_change().over("symbol").alias("returns"))
    requests = [
        {
            "name": "risk_adjusted_returns",
            "params": {"window": 20, "risk_free_rate": 0.052, "trading_periods": scale},
            "output": "risk",
        }
    ]
    output = materialize(
        compute_features(
            returns, request_file(tmp_path, requests, serialized), feed_spec=feed("24/7")
        )
    )
    assert output.select(raw.columns).equals(raw)
    for symbol in ("A", "B"):
        asset = output.filter(pl.col("symbol") == symbol)
        values = asset["returns"].drop_nulls().to_numpy()
        rf = 0.052 / scale
        excess = mean(values) - rf
        downside = np.minimum(values - rf, 0)
        close = asset["close"].to_numpy()
        peaks = np.maximum.accumulate(close)
        drawdown = abs(np.min((close - peaks) / peaks))
        gains = np.sum(values[values > rf] - rf)
        losses = np.sum(rf - values[values <= rf])
        expected = {
            "sharpe_ratio": excess / (stdev(values) + 1e-10) * sqrt(scale),
            "sortino_ratio": excess / (sqrt(np.mean(downside**2)) + 1e-10) * sqrt(scale),
            "calmar_ratio": mean(values) * scale / (drawdown + 1e-10),
            "omega_ratio": gains / losses,
        }
        for metric, value in expected.items():
            assert asset[f"risk_{metric}"][-1] == pytest.approx(value, rel=1e-9)
