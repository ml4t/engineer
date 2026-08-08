"""Adversarial tests for the shared public bar-sampler input contract."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from functools import partial

import polars as pl
import pytest

from ml4t.engineer.bars import (
    DollarBarSampler,
    DollarBarSamplerOriginal,
    DollarRunBarSampler,
    FixedTickImbalanceBarSampler,
    FixedTickRunBarSampler,
    FixedVolumeImbalanceBarSampler,
    ImbalanceBarSampler,
    ImbalanceBarSamplerOriginal,
    TickBarSampler,
    TickBarSamplerOriginal,
    TickImbalanceBarSampler,
    TickRunBarSampler,
    VolumeBarSampler,
    VolumeBarSamplerOriginal,
    VolumeRunBarSampler,
    WindowTickImbalanceBarSampler,
    WindowVolumeImbalanceBarSampler,
)
from ml4t.engineer.bars.base import BarSampler
from ml4t.engineer.core.exceptions import DataValidationError

SamplerFactory = Callable[[], BarSampler]

SAMPLER_FACTORIES: list[pytest.ParameterSet] = [
    pytest.param(partial(TickBarSampler, ticks_per_bar=10_000), id="tick"),
    pytest.param(partial(VolumeBarSampler, volume_per_bar=10_000), id="volume"),
    pytest.param(partial(DollarBarSampler, dollars_per_bar=1_000_000), id="dollar"),
    pytest.param(
        partial(
            ImbalanceBarSampler,
            expected_ticks_per_bar=10_000,
            initial_p_buy=0.75,
        ),
        id="imbalance",
    ),
    pytest.param(
        partial(TickImbalanceBarSampler, expected_ticks_per_bar=10_000),
        id="tick-imbalance",
    ),
    pytest.param(
        partial(FixedTickImbalanceBarSampler, threshold=10_000),
        id="fixed-tick-imbalance",
    ),
    pytest.param(
        partial(FixedVolumeImbalanceBarSampler, threshold=10_000),
        id="fixed-volume-imbalance",
    ),
    pytest.param(
        partial(WindowTickImbalanceBarSampler, initial_expected_t=10_000),
        id="window-tick-imbalance",
    ),
    pytest.param(
        partial(WindowVolumeImbalanceBarSampler, initial_expected_t=10_000),
        id="window-volume-imbalance",
    ),
    pytest.param(
        partial(TickRunBarSampler, expected_ticks_per_bar=10_000),
        id="tick-run",
    ),
    pytest.param(partial(FixedTickRunBarSampler, threshold=10_000), id="fixed-tick-run"),
    pytest.param(
        partial(VolumeRunBarSampler, expected_ticks_per_bar=10_000),
        id="volume-run",
    ),
    pytest.param(
        partial(DollarRunBarSampler, expected_ticks_per_bar=10_000),
        id="dollar-run",
    ),
    pytest.param(partial(TickBarSamplerOriginal, ticks_per_bar=10_000), id="tick-original"),
    pytest.param(
        partial(VolumeBarSamplerOriginal, volume_per_bar=10_000),
        id="volume-original",
    ),
    pytest.param(
        partial(DollarBarSamplerOriginal, dollars_per_bar=1_000_000),
        id="dollar-original",
    ),
    pytest.param(
        partial(
            ImbalanceBarSamplerOriginal,
            expected_ticks_per_bar=10_000,
            initial_p_buy=0.75,
        ),
        id="imbalance-original",
    ),
]


def _trades(timestamps: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "price": [10.0 + index for index in range(len(timestamps))],
            "volume": [1.0] * len(timestamps),
            "side": [1 if index % 2 == 0 else -1 for index in range(len(timestamps))],
        }
    )


@pytest.mark.parametrize("factory", SAMPLER_FACTORIES)
def test_every_public_sampler_rejects_unsorted_trades_before_state_change(
    factory: SamplerFactory,
) -> None:
    """Row order cannot redefine stateful bar chronology."""
    start = datetime(2024, 1, 1, 9, 30)
    data = _trades([start + timedelta(seconds=2), start, start + timedelta(seconds=1)])
    sampler = factory()
    initial_state = vars(sampler).copy()

    with pytest.raises(DataValidationError, match="sorted in nondecreasing order"):
        sampler.sample(data, include_incomplete=True)

    assert vars(sampler) == initial_state


@pytest.mark.parametrize("factory", SAMPLER_FACTORIES)
def test_every_public_sampler_checks_chronology_across_dataframe_chunks(
    factory: SamplerFactory,
) -> None:
    """Physical chunk boundaries cannot hide a timestamp reversal."""
    start = datetime(2024, 1, 1, 9, 30)
    data = pl.concat(
        [
            _trades([start, start + timedelta(seconds=2)]),
            _trades([start + timedelta(seconds=1), start + timedelta(seconds=3)]),
        ],
        rechunk=False,
    )
    assert data.n_chunks() == 2

    with pytest.raises(DataValidationError, match="sorted in nondecreasing order"):
        factory().sample(data)


@pytest.mark.parametrize("factory", SAMPLER_FACTORIES)
def test_every_public_sampler_preserves_input_order_for_equal_timestamps(
    factory: SamplerFactory,
) -> None:
    """Equal timestamps are valid and retain their stable input order."""
    timestamp = datetime(2024, 1, 1, 9, 30)
    data = _trades([timestamp, timestamp, timestamp])

    bars = factory().sample(data, include_incomplete=True)

    assert not bars.is_empty()
    assert bars["open"][0] == 10.0
    assert bars["close"][-1] == 12.0


@pytest.mark.parametrize("factory", SAMPLER_FACTORIES)
def test_every_public_sampler_has_one_typed_zero_bar_schema(factory: SamplerFactory) -> None:
    """Empty and insufficient streams produce identical typed schemas."""
    empty = pl.DataFrame(
        schema={
            "timestamp": pl.Datetime("us"),
            "price": pl.Float64,
            "volume": pl.Float64,
            "side": pl.Int8,
        }
    )
    insufficient = _trades([datetime(2024, 1, 1, 9, 30)])
    sampler = factory()

    from_empty = sampler.sample(empty)
    from_insufficient = sampler.sample(insufficient)

    assert from_empty.schema == from_insufficient.schema
    assert from_empty.width > 0
    assert pl.Null not in from_empty.schema.values()


@pytest.mark.parametrize("price", [float("nan"), float("inf"), 0.0, -1.0])
def test_non_finite_or_nonpositive_prices_are_rejected(price: float) -> None:
    """Prices must be finite and positive."""
    data = _trades([datetime(2024, 1, 1, 9, 30)]).with_columns(pl.lit(price).alias("price"))

    with pytest.raises(DataValidationError, match="Price values"):
        TickBarSampler(1).sample(data)


@pytest.mark.parametrize("volume", [float("nan"), float("inf"), -1.0])
def test_non_finite_or_negative_volumes_are_rejected(volume: float) -> None:
    """Volumes must be finite and nonnegative."""
    data = _trades([datetime(2024, 1, 1, 9, 30)]).with_columns(pl.lit(volume).alias("volume"))

    with pytest.raises(DataValidationError, match="Volume values"):
        TickBarSampler(1).sample(data)


def test_zero_volume_is_valid() -> None:
    """A zero-volume trade is finite and does not corrupt OHLC output."""
    data = _trades([datetime(2024, 1, 1, 9, 30)]).with_columns(pl.lit(0.0).alias("volume"))

    bars = TickBarSampler(1).sample(data)

    assert bars["volume"].to_list() == [0.0]


@pytest.mark.parametrize("side", [0.0, 7.0, float("nan"), float("inf"), None])
def test_invalid_side_encodings_are_rejected(side: float | None) -> None:
    """Trade side is exactly buy 1 or sell -1."""
    data = _trades([datetime(2024, 1, 1, 9, 30)]).with_columns(pl.lit(side).alias("side"))

    with pytest.raises(DataValidationError, match="Side values"):
        VolumeBarSampler(1).sample(data)


def test_timestamp_must_be_datetime_and_non_null() -> None:
    """Trade chronology requires concrete Datetime values."""
    timestamp = datetime(2024, 1, 1, 9, 30)
    valid = _trades([timestamp])
    integer_timestamp = valid.with_columns(pl.lit(1).alias("timestamp"))
    null_timestamp = valid.with_columns(pl.lit(None, dtype=pl.Datetime).alias("timestamp"))

    with pytest.raises(DataValidationError, match="Datetime dtype"):
        TickBarSampler(1).sample(integer_timestamp)
    with pytest.raises(DataValidationError, match="null"):
        TickBarSampler(1).sample(null_timestamp)
