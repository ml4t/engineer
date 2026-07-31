"""Donchian channel expression contracts."""

import polars as pl
import pytest

from ml4t.engineer.core.exceptions import InvalidParameterError
from ml4t.engineer.features.trend.donchian import (
    donchian_channels,
    donchian_lower,
    donchian_middle,
    donchian_upper,
)


def test_donchian_helpers_match_combined_channel() -> None:
    data = pl.DataFrame(
        {
            "high": [3.0, 5.0, 4.0, 7.0],
            "low": [1.0, 2.0, 0.0, 3.0],
        }
    )
    upper, lower, middle = donchian_channels(period=2)
    result = data.select(
        upper.alias("combined_upper"),
        lower.alias("combined_lower"),
        middle.alias("combined_middle"),
        donchian_upper(period=2).alias("upper"),
        donchian_lower(period=2).alias("lower"),
        donchian_middle(period=2).alias("middle"),
    )

    assert result["upper"].equals(result["combined_upper"])
    assert result["lower"].equals(result["combined_lower"])
    assert result["middle"].equals(result["combined_middle"])
    assert result["upper"].to_list() == [None, 5.0, 5.0, 7.0]
    assert result["lower"].to_list() == [None, 1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "factory",
    [donchian_channels, donchian_upper, donchian_lower, donchian_middle],
)
def test_donchian_helpers_reject_nonpositive_period(factory: object) -> None:
    with pytest.raises(InvalidParameterError, match="period must be >= 1"):
        factory(period=0)  # type: ignore[operator]
