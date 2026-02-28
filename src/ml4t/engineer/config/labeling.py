# mypy: disable-error-code="misc,no-any-return,arg-type"
"""Labeling configuration with Pydantic v2 serialization.

This module provides config classes for ML labeling methods that extend BaseConfig,
enabling JSON/YAML serialization for reproducible experiments.

Migrates from simple dataclasses in labeling/barriers.py to full Pydantic support.

Examples
--------
>>> from ml4t.engineer.config import LabelingConfig
>>>
>>> # Create and serialize config
>>> config = LabelingConfig.triple_barrier(
...     upper_barrier=0.02,
...     lower_barrier=0.01,
...     max_holding_period=20,
... )
>>> config.to_yaml("labeling_config.yaml")
>>>
>>> # Load and use
>>> config = LabelingConfig.from_yaml("labeling_config.yaml")
>>> labels = triple_barrier_labels(df, config=config)
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Literal

from pydantic import AliasChoices, Field, field_validator

from ml4t.engineer.config.base import BaseConfig
from ml4t.engineer.config.data_contract import DataContractConfig


class LabelingConfig(BaseConfig):
    """Unified configuration for all labeling methods.

    Extends BaseConfig for full JSON/YAML serialization support.
    Supports multiple labeling methods via the `method` discriminator.

    All barrier distances are specified as POSITIVE values representing
    the distance from the entry price. The position side determines
    the direction of the barriers.

    Attributes
    ----------
    method : str
        Labeling method: "triple_barrier", "atr_barrier", "fixed_horizon",
        "trend_scanning", "percentile"
    price_col : str
        Price column for barrier calculations (typically 'close')
    timestamp_col : str
        Timestamp column for duration calculations

    Triple Barrier Parameters
    -------------------------
    upper_barrier : float | str | None
        Upper barrier distance or column name for dynamic barriers
    lower_barrier : float | str | None
        Lower barrier distance or column name for dynamic barriers
    max_holding_period : int | str
        Maximum holding period in bars or column name
    side : int | str | None
        Position side: 1 (long), -1 (short), 0/None (symmetric)
    trailing_stop : bool | float | str
        Enable trailing stop or specify percentage/column

    ATR Barrier Parameters
    ----------------------
    atr_tp_multiple : float
        ATR multiplier for take profit (e.g., 2.0 = 2x ATR)
    atr_sl_multiple : float
        ATR multiplier for stop loss (e.g., 1.0 = 1x ATR)
    atr_period : int
        ATR calculation period (Wilder's default: 14)

    Fixed Horizon Parameters
    ------------------------
    horizon : int
        Forward-looking period in bars
    return_method : str
        Return calculation: "returns", "log_returns", "binary"
    threshold : float | None
        Binary classification threshold

    Trend Scanning Parameters
    -------------------------
    min_horizon : int
        Minimum lookforward period
    max_horizon : int
        Maximum lookforward period
    t_value_threshold : float
        T-statistic threshold for trend significance

    Examples
    --------
    >>> # Triple barrier with fixed barriers
    >>> config = LabelingConfig(
    ...     method="triple_barrier",
    ...     upper_barrier=0.02,
    ...     lower_barrier=0.01,
    ...     max_holding_period=20,
    ...     side=1,
    ... )
    >>> config.to_yaml("config.yaml")

    >>> # ATR-adjusted barriers
    >>> config = LabelingConfig.atr_barrier(
    ...     atr_tp_multiple=2.0,
    ...     atr_sl_multiple=1.0,
    ...     max_holding_period=20,
    ... )

    >>> # Load from file
    >>> config = LabelingConfig.from_yaml("config.yaml")
    """

    # Discriminator for labeling method
    method: Literal[
        "triple_barrier",
        "atr_barrier",
        "fixed_horizon",
        "trend_scanning",
        "percentile",
    ] = Field("triple_barrier", description="Labeling method to use")

    # Common parameters
    price_col: str = Field("close", description="Price column for calculations")
    timestamp_col: str = Field("timestamp", description="Timestamp column")
    group_col: str | list[str] | None = Field(
        None,
        validation_alias=AliasChoices("group_col", "symbol_col", "ticker_col", "asset_col"),
        description="Grouping column(s) for panel labeling (e.g., symbol, ticker)",
    )
    data_contract: DataContractConfig | None = Field(
        None,
        validation_alias=AliasChoices("data_contract", "contract"),
        description="Optional shared dataframe column contract for cross-library workflows",
    )

    # Triple barrier parameters
    upper_barrier: float | str | None = Field(
        None,
        description="Upper barrier distance (positive) or column name for dynamic barriers",
    )
    lower_barrier: float | str | None = Field(
        None,
        description="Lower barrier distance (positive) or column name for dynamic barriers",
    )
    max_holding_period: int | str | timedelta = Field(
        10,
        description=(
            "Maximum holding period: int (bars), duration string ('1h', '4h', '1d'), "
            "timedelta, or column name. Time-based values are converted to per-event "
            "bar counts during labeling, allowing adaptive horizons for irregular data."
        ),
    )
    side: int | str | None = Field(
        1,
        description="Position side: 1 (long), -1 (short), 0/None (symmetric)",
    )
    trailing_stop: bool | float | str = Field(
        False,
        description="Trailing stop: False, percentage (float), or column name",
    )

    # Weight scheme (for sample weighting)
    weight_scheme: str = Field(
        "equal",
        description="Weighting scheme: 'equal', 'returns', 'time_decay'",
    )
    weight_decay_rate: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Decay rate for time-based weighting",
    )

    # ATR barrier parameters
    atr_tp_multiple: float = Field(
        2.0,
        gt=0.0,
        description="ATR multiplier for take profit target",
    )
    atr_sl_multiple: float = Field(
        1.0,
        gt=0.0,
        description="ATR multiplier for stop loss",
    )
    atr_period: int = Field(
        14,
        ge=1,
        description="ATR calculation period",
    )

    # Fixed horizon parameters
    horizon: int = Field(
        10,
        ge=1,
        description="Forward-looking period for fixed horizon labels",
    )
    return_method: Literal["returns", "log_returns", "binary"] = Field(
        "returns",
        description="Return calculation method",
    )
    threshold: float | None = Field(
        None,
        description="Binary classification threshold",
    )

    # Trend scanning parameters
    min_horizon: int = Field(
        5,
        ge=1,
        description="Minimum lookforward period for trend scanning",
    )
    max_horizon: int = Field(
        20,
        ge=1,
        description="Maximum lookforward period for trend scanning",
    )
    t_value_threshold: float = Field(
        2.0,
        gt=0.0,
        description="T-statistic threshold for trend significance",
    )

    # Percentile labeling parameters
    percentile_window: int = Field(
        252,
        ge=1,
        description="Rolling window for percentile calculation",
    )
    n_bins: int = Field(
        3,
        ge=2,
        description="Number of bins for multi-class labels",
    )

    @field_validator("side")
    @classmethod
    def validate_side(cls, v: int | str | None) -> int | str | None:
        """Validate side is valid."""
        if isinstance(v, int) and v not in (-1, 0, 1):
            raise ValueError("side must be -1, 0, 1, or a column name")
        return v

    @field_validator("max_horizon")
    @classmethod
    def validate_max_horizon(cls, v: int, info: Any) -> int:
        """Ensure max_horizon >= min_horizon."""
        # Note: info.data contains already-validated fields
        if "min_horizon" in info.data and v < info.data["min_horizon"]:
            raise ValueError("max_horizon must be >= min_horizon")
        return v

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def triple_barrier(
        cls,
        upper_barrier: float | str | None = 0.02,
        lower_barrier: float | str | None = 0.01,
        max_holding_period: int | str | timedelta = 20,
        side: int | str | None = 1,
        trailing_stop: bool | float | str = False,
        **kwargs: Any,
    ) -> LabelingConfig:
        """Create triple barrier labeling config.

        Parameters
        ----------
        upper_barrier : float | str | None
            Take profit barrier (2% = 0.02) or column name
        lower_barrier : float | str | None
            Stop loss barrier (1% = 0.01) or column name
        max_holding_period : int | str | timedelta
            Maximum holding period:
            - int: Number of bars
            - str: Duration string ('4h', '1d') or column name
            - timedelta: Python timedelta object
        side : int | str | None
            Position direction: 1 (long), -1 (short)
        trailing_stop : bool | float | str
            Enable trailing stop

        Returns
        -------
        LabelingConfig
            Configured for triple barrier method

        Examples
        --------
        >>> config = LabelingConfig.triple_barrier(0.02, 0.01, 20)
        >>> config.to_yaml("triple_barrier.yaml")

        >>> # Time-based max holding period
        >>> config = LabelingConfig.triple_barrier(0.02, 0.01, "4h")

        >>> # Using timedelta
        >>> from datetime import timedelta
        >>> config = LabelingConfig.triple_barrier(0.02, 0.01, timedelta(hours=4))
        """
        return cls(
            method="triple_barrier",
            upper_barrier=upper_barrier,
            lower_barrier=lower_barrier,
            max_holding_period=max_holding_period,
            side=side,
            trailing_stop=trailing_stop,
            **kwargs,
        )

    @classmethod
    def atr_barrier(
        cls,
        atr_tp_multiple: float = 2.0,
        atr_sl_multiple: float = 1.0,
        atr_period: int = 14,
        max_holding_period: int | str | timedelta = 20,
        side: int | str | None = 1,
        trailing_stop: bool = False,
        **kwargs: Any,
    ) -> LabelingConfig:
        """Create ATR-adjusted barrier labeling config.

        Volatility-adaptive barriers that adjust to market conditions.

        Parameters
        ----------
        atr_tp_multiple : float
            ATR multiplier for take profit (e.g., 2.0 = 2x ATR)
        atr_sl_multiple : float
            ATR multiplier for stop loss (e.g., 1.0 = 1x ATR)
        atr_period : int
            ATR calculation period (default: 14)
        max_holding_period : int | str | timedelta
            Maximum holding period:
            - int: Number of bars
            - str: Duration string ('4h', '1d') or column name
            - timedelta: Python timedelta object
        side : int | str | None
            Position direction: 1 (long), -1 (short)
        trailing_stop : bool
            Enable trailing stop

        Returns
        -------
        LabelingConfig
            Configured for ATR barrier method

        Examples
        --------
        >>> config = LabelingConfig.atr_barrier(2.0, 1.0, 14)
        >>> config.to_yaml("atr_barrier.yaml")

        >>> # Time-based max holding period
        >>> config = LabelingConfig.atr_barrier(2.0, 1.0, 14, max_holding_period="4h")
        """
        return cls(
            method="atr_barrier",
            atr_tp_multiple=atr_tp_multiple,
            atr_sl_multiple=atr_sl_multiple,
            atr_period=atr_period,
            max_holding_period=max_holding_period,
            side=side,
            trailing_stop=trailing_stop,
            **kwargs,
        )

    @classmethod
    def fixed_horizon(
        cls,
        horizon: int = 10,
        return_method: Literal["returns", "log_returns", "binary"] = "returns",
        threshold: float | None = None,
        **kwargs: Any,
    ) -> LabelingConfig:
        """Create fixed horizon labeling config.

        Simple forward-looking returns over a fixed period.

        Parameters
        ----------
        horizon : int
            Forward-looking period in bars
        return_method : str
            "returns", "log_returns", or "binary"
        threshold : float | None
            Threshold for binary classification

        Returns
        -------
        LabelingConfig
            Configured for fixed horizon method

        Examples
        --------
        >>> config = LabelingConfig.fixed_horizon(10, "binary", threshold=0.0)
        """
        return cls(
            method="fixed_horizon",
            horizon=horizon,
            return_method=return_method,
            threshold=threshold,
            **kwargs,
        )

    @classmethod
    def trend_scanning(
        cls,
        min_horizon: int = 5,
        max_horizon: int = 20,
        t_value_threshold: float = 2.0,
        **kwargs: Any,
    ) -> LabelingConfig:
        """Create trend scanning labeling config.

        De Prado's trend scanning method using t-statistics.

        Parameters
        ----------
        min_horizon : int
            Minimum lookforward period
        max_horizon : int
            Maximum lookforward period
        t_value_threshold : float
            T-statistic threshold for trend significance

        Returns
        -------
        LabelingConfig
            Configured for trend scanning method

        Examples
        --------
        >>> config = LabelingConfig.trend_scanning(5, 20, 2.0)
        """
        return cls(
            method="trend_scanning",
            min_horizon=min_horizon,
            max_horizon=max_horizon,
            t_value_threshold=t_value_threshold,
            **kwargs,
        )

__all__ = [
    "LabelingConfig",
]

_REMOVED_EXPORTS = {
    "BarrierLabelingConfig": (
        "ml4t.engineer.config.labeling.BarrierLabelingConfig has been removed. "
        "Use ml4t.engineer.config.LabelingConfig instead."
    )
}


def __getattr__(name: str) -> object:
    if name in _REMOVED_EXPORTS:
        raise ImportError(_REMOVED_EXPORTS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
