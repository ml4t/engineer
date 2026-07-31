"""Behavioral tests for every public imbalance-bar variant."""

from datetime import datetime, timedelta
from functools import partial

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.bars.imbalance import (
    FixedTickImbalanceBarSampler,
    FixedVolumeImbalanceBarSampler,
    ImbalanceBarSampler,
    TickImbalanceBarSampler,
    WindowTickImbalanceBarSampler,
    WindowVolumeImbalanceBarSampler,
    _calculate_fixed_tick_imbalance_bars_nb,
    _calculate_fixed_volume_imbalance_bars_nb,
    _calculate_tick_imbalance_bars_nb,
    _calculate_window_tick_imbalance_bars_nb,
    _calculate_window_volume_imbalance_bars_nb,
)
from ml4t.engineer.core.exceptions import DataValidationError


def _trades(
    sides: list[float],
    volumes: list[float] | None = None,
) -> pl.DataFrame:
    if volumes is None:
        volumes = [100.0] * len(sides)
    start = datetime(2026, 1, 1, 9, 30)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(seconds=index) for index in range(len(sides))],
            "price": [100.0 + index / 10 for index in range(len(sides))],
            "volume": volumes,
            "side": sides,
        }
    )


@pytest.mark.parametrize(
    "factory",
    [
        partial(TickImbalanceBarSampler, expected_ticks_per_bar=5),
        partial(ImbalanceBarSampler, expected_ticks_per_bar=5),
        partial(FixedTickImbalanceBarSampler, threshold=3),
        partial(FixedVolumeImbalanceBarSampler, threshold=250.0),
        partial(WindowTickImbalanceBarSampler, initial_expected_t=3, tick_window=3),
        partial(WindowVolumeImbalanceBarSampler, initial_expected_t=3, tick_window=3),
    ],
)
def test_every_imbalance_sampler_rejects_missing_side_and_handles_empty(factory) -> None:
    sampler = factory()
    without_side = _trades([1.0]).drop("side")

    with pytest.raises(DataValidationError, match="require.*'side' column"):
        sampler.sample(without_side)

    empty = _trades([])
    result = sampler.sample(empty)
    assert result.is_empty()
    assert result.schema["timestamp"] == pl.Datetime


@pytest.mark.parametrize(
    ("factory", "expected_message"),
    [
        (partial(TickImbalanceBarSampler, expected_ticks_per_bar=0), "must be positive"),
        (partial(TickImbalanceBarSampler, expected_ticks_per_bar=5, alpha=0), "alpha"),
        (partial(TickImbalanceBarSampler, expected_ticks_per_bar=5, alpha=1.1), "alpha"),
        (
            partial(TickImbalanceBarSampler, expected_ticks_per_bar=5, initial_p_buy=-0.1),
            "initial_p_buy",
        ),
        (
            partial(TickImbalanceBarSampler, expected_ticks_per_bar=5, min_bars_warmup=-1),
            "min_bars_warmup",
        ),
        (partial(ImbalanceBarSampler, expected_ticks_per_bar=5, min_bars_warmup=-1), "warmup"),
        (partial(FixedTickImbalanceBarSampler, threshold=0), "positive"),
        (partial(FixedVolumeImbalanceBarSampler, threshold=0), "positive"),
        (partial(WindowTickImbalanceBarSampler, initial_expected_t=0), "initial_expected_t"),
        (
            partial(WindowTickImbalanceBarSampler, initial_expected_t=5, bar_window=0),
            "bar_window",
        ),
        (
            partial(WindowTickImbalanceBarSampler, initial_expected_t=5, tick_window=0),
            "tick_window",
        ),
        (partial(WindowVolumeImbalanceBarSampler, initial_expected_t=0), "initial_expected_t"),
        (
            partial(WindowVolumeImbalanceBarSampler, initial_expected_t=5, bar_window=0),
            "bar_window",
        ),
        (
            partial(WindowVolumeImbalanceBarSampler, initial_expected_t=5, tick_window=0),
            "tick_window",
        ),
    ],
)
def test_imbalance_sampler_constructor_boundaries(factory, expected_message) -> None:
    with pytest.raises(ValueError, match=expected_message):
        factory()


def test_tick_imbalance_kernel_updates_estimates_after_warmup() -> None:
    sides = np.ones(12, dtype=np.float64)

    indices, thresholds, cumulative, expected_t, p_buy = _calculate_tick_imbalance_bars_nb(
        sides,
        initial_expected_t=3.0,
        initial_p_buy=0.75,
        alpha=0.5,
        min_bars_warmup=0,
    )

    assert indices.tolist() == sorted(indices.tolist())
    assert len(indices) >= 2
    assert np.all(cumulative >= thresholds)
    assert expected_t[-1] != 3.0
    assert p_buy[-1] > 0.75


def test_fixed_imbalance_kernels_reset_after_each_threshold() -> None:
    sides = np.ones(7, dtype=np.float64)
    volumes = np.full(7, 100.0)

    tick_indices, tick_theta = _calculate_fixed_tick_imbalance_bars_nb(sides, 3.0)
    volume_indices, volume_theta = _calculate_fixed_volume_imbalance_bars_nb(volumes, sides, 250.0)

    assert tick_indices.tolist() == [2, 5]
    assert tick_theta.tolist() == [3.0, 3.0]
    assert volume_indices.tolist() == [2, 5]
    assert volume_theta.tolist() == [300.0, 300.0]


def test_window_kernels_wait_for_history_then_use_bounded_bar_window() -> None:
    sides = np.ones(12, dtype=np.float64)
    volumes = np.full(12, 100.0)

    tick = _calculate_window_tick_imbalance_bars_nb(sides, 3, bar_window=1, tick_window=3)
    volume = _calculate_window_volume_imbalance_bars_nb(
        volumes,
        sides,
        3,
        bar_window=1,
        tick_window=3,
    )

    assert tick[0][0] >= 2
    assert volume[0][0] >= 2
    assert len(tick[0]) >= 2
    assert len(volume[0]) >= 2
    assert np.all(tick[1] > 0)
    assert np.all(volume[1] > 0)


def test_window_kernels_do_not_form_bars_from_zero_expected_imbalance() -> None:
    sides = np.array([1.0, -1.0] * 4)
    volumes = np.full(8, 100.0)

    tick = _calculate_window_tick_imbalance_bars_nb(sides, 3, bar_window=2, tick_window=2)
    volume = _calculate_window_volume_imbalance_bars_nb(
        volumes,
        sides,
        3,
        bar_window=2,
        tick_window=2,
    )

    assert tick[0].size == 0
    assert volume[0].size == 0


@pytest.mark.parametrize(
    ("sampler", "diagnostic_columns"),
    [
        (
            TickImbalanceBarSampler(
                expected_ticks_per_bar=3,
                initial_p_buy=0.75,
                min_bars_warmup=0,
            ),
            {"tick_imbalance", "expected_imbalance", "expected_t", "p_buy"},
        ),
        (
            ImbalanceBarSampler(
                expected_ticks_per_bar=3,
                initial_p_buy=0.75,
                min_bars_warmup=0,
            ),
            {"imbalance", "expected_imbalance", "expected_t", "p_buy", "v_plus", "e_v"},
        ),
        (
            FixedTickImbalanceBarSampler(threshold=3),
            {"tick_imbalance", "cumulative_theta", "threshold"},
        ),
        (
            FixedVolumeImbalanceBarSampler(threshold=250.0),
            {"volume_imbalance", "cumulative_theta", "threshold"},
        ),
        (
            WindowTickImbalanceBarSampler(initial_expected_t=3, bar_window=1, tick_window=3),
            {"tick_imbalance", "expected_imbalance", "expected_t", "p_buy"},
        ),
        (
            WindowVolumeImbalanceBarSampler(initial_expected_t=3, bar_window=1, tick_window=3),
            {"volume_imbalance", "expected_imbalance", "expected_t", "imbalance_factor"},
        ),
    ],
)
def test_every_imbalance_sampler_preserves_all_ticks_with_incomplete_bar(
    sampler,
    diagnostic_columns,
) -> None:
    data = _trades([1.0] * 10)

    complete = sampler.sample(data)
    all_bars = sampler.sample(data, include_incomplete=True)

    assert diagnostic_columns <= set(all_bars.columns)
    assert complete["tick_count"].sum() <= len(data)
    assert all_bars["tick_count"].sum() == len(data)
    assert all_bars["volume"].sum() == pytest.approx(data["volume"].sum())


@pytest.mark.parametrize(
    "sampler",
    [
        TickImbalanceBarSampler(expected_ticks_per_bar=100, initial_p_buy=1.0),
        ImbalanceBarSampler(expected_ticks_per_bar=100, initial_p_buy=1.0),
        FixedTickImbalanceBarSampler(threshold=100),
        FixedVolumeImbalanceBarSampler(threshold=100_000.0),
        WindowTickImbalanceBarSampler(initial_expected_t=100, tick_window=20),
        WindowVolumeImbalanceBarSampler(initial_expected_t=100, tick_window=20),
    ],
)
def test_no_completed_imbalance_bar_returns_schema_and_optional_remainder(sampler) -> None:
    data = _trades([1.0, -1.0] * 3)

    completed = sampler.sample(data)
    with_remainder = sampler.sample(data, include_incomplete=True)

    assert completed.is_empty()
    assert len(with_remainder) == 1
    assert with_remainder["tick_count"].item() == len(data)
