#!/usr/bin/env python3
"""Generate golden reference files for regression testing.

Usage:
    python validation/golden/generate_golden.py [--feature NAME]

Only run this script when intentionally updating feature calculations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from ml4t.engineer.features.momentum import mom, roc, rsi
from ml4t.engineer.features.statistics import stddev
from ml4t.engineer.features.trend import ema, sma, wma
from ml4t.engineer.features.volatility import atr, natr
from ml4t.engineer.features.volume import ad, obv

# Output directory
GOLDEN_DATA_DIR = Path(__file__).parent / "data"

# Reference data: deterministic for reproducibility
np.random.seed(42)
N = 500
RETURNS = np.random.randn(N) * 0.02
CLOSE = 100.0 * np.exp(np.cumsum(RETURNS))
HIGH = CLOSE * (1 + np.abs(np.random.randn(N)) * 0.01)
LOW = CLOSE * (1 - np.abs(np.random.randn(N)) * 0.01)
OPEN = (HIGH + LOW) / 2
HIGH = np.maximum.reduce([OPEN, CLOSE, HIGH])
LOW = np.minimum.reduce([OPEN, CLOSE, LOW])
VOLUME = np.random.randint(100000, 1000000, N).astype(np.float64)

# Feature definitions: name -> (function, kwargs)
FEATURES = {
    # Trend
    "sma_20": (sma, {"close": CLOSE, "period": 20}),
    "sma_50": (sma, {"close": CLOSE, "period": 50}),
    "ema_20": (ema, {"close": CLOSE, "period": 20}),
    "ema_50": (ema, {"close": CLOSE, "period": 50}),
    "wma_20": (wma, {"close": CLOSE, "period": 20}),
    # Momentum
    "rsi_14": (rsi, {"close": CLOSE, "period": 14}),
    "roc_10": (roc, {"close": CLOSE, "period": 10}),
    "mom_10": (mom, {"close": CLOSE, "period": 10}),
    # Volatility
    "atr_14": (atr, {"high": HIGH, "low": LOW, "close": CLOSE, "period": 14}),
    "natr_14": (natr, {"high": HIGH, "low": LOW, "close": CLOSE, "period": 14}),
    "stddev_20": (stddev, {"close": CLOSE, "period": 20}),
    # Volume
    "obv": (obv, {"close": CLOSE, "volume": VOLUME}),
    "ad": (ad, {"high": HIGH, "low": LOW, "close": CLOSE, "volume": VOLUME}),
}


def generate_golden(feature_name: str | None = None) -> None:
    """Generate golden files for features."""
    GOLDEN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    features_to_generate = {feature_name: FEATURES[feature_name]} if feature_name else FEATURES

    for name, (func, kwargs) in features_to_generate.items():
        print(f"Generating {name}...")

        # Compute feature
        result = func(**kwargs)

        # Create DataFrame with input data for context
        df = pl.DataFrame(
            {
                "close": CLOSE,
                "high": HIGH,
                "low": LOW,
                "open": OPEN,
                "volume": VOLUME,
                "feature": result,
            }
        )

        # Save as Parquet
        output_path = GOLDEN_DATA_DIR / f"{name}.parquet"
        df.write_parquet(output_path)
        print(f"  -> {output_path}")

    print(f"\nGenerated {len(features_to_generate)} golden files.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate golden reference files")
    parser.add_argument("--feature", type=str, help="Specific feature to generate")
    args = parser.parse_args()

    if args.feature and args.feature not in FEATURES:
        print(f"Unknown feature: {args.feature}")
        print(f"Available: {list(FEATURES.keys())}")
        return

    generate_golden(args.feature)


if __name__ == "__main__":
    main()
