"""Adjustment arithmetic must not invalidate a consistent bar or repair bad prices."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.core.exceptions import DataValidationError
from ml4t.engineer.labeling import triple_barrier_labels


def _bars():
    # The adjusted close is supplied directly; the other prices use the adjustment ratio.
    raw_close, adjusted_close = 133.41, 116.987526
    factor = adjusted_close / raw_close
    return pl.DataFrame(
        {
            "timestamp": [datetime(2020, 1, 27) + timedelta(days=i) for i in range(3)],
            "open": [(raw_close - 0.25) * factor] * 3,
            "high": [raw_close * factor] * 3,
            "low": [(raw_close - 1) * factor] * 3,
            "close": [adjusted_close] * 3,
        }
    )


def _label(frame, side=1):
    return triple_barrier_labels(
        frame,
        LabelingConfig.triple_barrier(
            upper_barrier=0.03, lower_barrier=0.02, max_holding_period=1, side=side
        ),
        price_col="close",
        high_col="high",
        low_col="low",
        open_col="open",
    )


@pytest.mark.parametrize("side", [-1, 1])
def test_ratio_adjustment_roundoff_does_not_reject_a_consistent_bar(side):
    frame = _bars()
    before = frame.clone()
    assert frame["close"][0] - frame["high"][0] == 1.4210854715202004e-14
    output = _label(frame, side)
    assert output.height == frame.height
    assert output["label"][0] == 0
    assert output["label_time"][0] == frame["timestamp"][1]
    assert_frame_equal(output.select(frame.columns), frame)
    assert_frame_equal(frame, before)


@pytest.mark.parametrize("column", ["open", "close"])
@pytest.mark.parametrize("boundary,direction", [("high", np.inf), ("low", 0.0)])
@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e12])
def test_ulp_range_slack_accepts_roundoff_without_changing_input(
    column, boundary, direction, scale
):
    frame = _bars().with_columns(pl.col(pl.Float64) * scale)
    frame = frame.with_columns(pl.col("high").alias("close"))
    rounded = np.nextafter(frame[boundary][0], direction)
    frame = frame.with_columns(pl.lit(rounded).alias(column))
    before = frame.clone()
    output = _label(frame)
    assert_frame_equal(output.select(frame.columns), frame)
    assert_frame_equal(frame, before)


@pytest.mark.parametrize("column", ["open", "close"])
@pytest.mark.parametrize("boundary,direction", [("high", np.inf), ("low", 0.0)])
@pytest.mark.parametrize("steps", [8, 9, 32])
def test_range_validation_enforces_the_documented_roundoff_limit(
    column, boundary, direction, steps
):
    frame = _bars().with_columns(pl.col("high").alias("close"))
    candidate = frame[boundary][0]
    for _ in range(steps):
        candidate = np.nextafter(candidate, direction)
    frame = frame.with_columns(pl.lit(candidate).alias(column))
    before = frame.clone()
    if steps == 8:
        output = _label(frame)
        assert_frame_equal(output.select(frame.columns), frame)
    else:
        with pytest.raises(DataValidationError, match="low-high range"):
            _label(frame)
    assert_frame_equal(frame, before)


@pytest.mark.parametrize("column", ["open", "close", "high", "low"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, 0.0, -1.0])
def test_roundoff_tolerance_keeps_nonfinite_and_nonpositive_prices_invalid(column, invalid):
    frame = _bars().with_columns(pl.col("high").alias("close"))
    frame = frame.with_columns(pl.lit(invalid).alias(column))
    with pytest.raises(DataValidationError, match="finite|positive|NaN"):
        _label(frame)


def test_roundoff_tolerance_does_not_accept_inverted_high_and_low():
    frame = _bars().with_columns(pl.col("low").alias("high"), pl.col("high").alias("low"))
    with pytest.raises(DataValidationError, match="high prices"):
        _label(frame)
