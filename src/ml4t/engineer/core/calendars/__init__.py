"""
Exchange calendars for time-aware feature engineering.

This module provides calendar-aware functionality for:
- Rolling windows over trading sessions (not calendar days)
- Holiday-aware feature calculations
- Market-hours filtering
- Cross-validation split timing
"""

from .base import ExchangeCalendar
from .crypto import CryptoCalendar
from .equity import EquityCalendar

__all__ = [
    "CryptoCalendar",
    "EquityCalendar",
    "ExchangeCalendar",
]
