"""Base class for information-driven bar samplers."""

from abc import ABC, abstractmethod
from typing import Any

import polars as pl

from ml4t.engineer.core.exceptions import DataValidationError


class BarSampler(ABC):
    """Abstract base class for bar samplers.

    Bar samplers transform irregularly spaced tick data into
    regularly sampled bars based on various criteria (ticks, volume, etc).
    """

    @abstractmethod
    def sample(
        self,
        data: pl.DataFrame,
        include_incomplete: bool = False,
    ) -> pl.DataFrame:
        """Sample bars from tick data.

        Parameters
        ----------
        data : pl.DataFrame
            Tick data with columns: timestamp, price, volume, side
        include_incomplete : bool, default False
            Whether to include incomplete final bar

        Returns
        -------
        pl.DataFrame
            Sampled bars with OHLCV and additional information
        """

    def _validate_data(self, data: pl.DataFrame) -> None:
        """Validate input data has required columns.

        Parameters
        ----------
        data : pl.DataFrame
            Input data to validate

        Raises
        ------
        DataValidationError
            If required columns are missing
        """
        if not isinstance(data, pl.DataFrame):
            raise DataValidationError("data must be a Polars DataFrame")

        required_cols = {"timestamp", "price", "volume"}
        missing_cols = required_cols - set(data.columns)

        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")

        # Check for empty data
        if len(data) == 0:
            return

        null_columns = sorted(column for column in required_cols if data[column].null_count())
        if null_columns:
            raise DataValidationError(f"Required columns contain null values: {null_columns}")

        # Check data types
        if not data["price"].dtype.is_numeric():
            raise DataValidationError("Price column must be numeric")

        if not data["volume"].dtype.is_numeric():
            raise DataValidationError("Volume column must be numeric")

        if not isinstance(data["timestamp"].dtype, pl.Datetime):
            raise DataValidationError("Timestamp column must use a Polars Datetime dtype")

        if data["price"].is_nan().any() or data["price"].is_infinite().any():
            raise DataValidationError("Price values must be finite")
        if (data["price"] <= 0).any():
            raise DataValidationError("Price values must be positive")

        if data["volume"].is_nan().any() or data["volume"].is_infinite().any():
            raise DataValidationError("Volume values must be finite")
        if (data["volume"] < 0).any():
            raise DataValidationError("Volume values must be nonnegative")

        if not data["timestamp"].is_sorted():
            raise DataValidationError("Timestamp values must be sorted in nondecreasing order")

        if "side" in data.columns:
            if data["side"].null_count():
                raise DataValidationError("Side values must not be null")
            if not data["side"].dtype.is_numeric():
                raise DataValidationError("Side column must be numeric")
            if not data["side"].is_in([-1, 1]).all():
                raise DataValidationError("Side values must be either -1 or 1")

    @staticmethod
    def _empty_result(columns: list[str]) -> pl.DataFrame:
        """Create a typed empty bar result."""
        count_columns = {"buy_count", "run_length", "sell_count", "tick_count"}
        schema = {}
        for column in columns:
            if column == "timestamp":
                schema[column] = pl.Datetime("us")
            elif column in count_columns:
                schema[column] = pl.UInt32
            else:
                schema[column] = pl.Float64
        return pl.DataFrame(schema=schema)

    def _create_ohlcv_bar(
        self,
        ticks: pl.DataFrame,
        additional_cols: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create OHLCV bar from ticks.

        Parameters
        ----------
        ticks : pl.DataFrame
            Tick data for this bar
        additional_cols : dict, optional
            Additional columns to include in bar

        Returns
        -------
        dict
            Bar data as dictionary
        """
        if len(ticks) == 0:
            return {}

        bar = {
            "timestamp": ticks["timestamp"][0],
            "open": ticks["price"][0],
            "high": ticks["price"].max(),
            "low": ticks["price"].min(),
            "close": ticks["price"][-1],
            "volume": ticks["volume"].sum(),
            "tick_count": len(ticks),
        }

        if additional_cols:
            bar.update(additional_cols)

        return bar


__all__ = ["BarSampler"]
