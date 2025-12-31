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
        required_cols = {"timestamp", "price", "volume"}
        missing_cols = required_cols - set(data.columns)

        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")

        # Check for empty data
        if len(data) == 0:
            return

        # Check data types
        if not data["price"].dtype.is_numeric():
            raise DataValidationError("Price column must be numeric")

        if not data["volume"].dtype.is_numeric():
            raise DataValidationError("Volume column must be numeric")

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
