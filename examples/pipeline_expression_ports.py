"""Native ports of elementwise math and daily summaries, not a graph interpreter."""

import json

import numpy as np
import polars as pl


def elementary_features(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    value = pl.col("value")
    functions = {
        "sin": value.sin(),
        "cos": value.cos(),
        "tan": value.tan(),
        "arcsin": value.arcsin(),
        "arccos": value.arccos(),
        "arctan": value.arctan(),
        "sinh": value.sinh(),
        "cosh": value.cosh(),
        "tanh": value.tanh(),
        "arcsinh": value.arcsinh(),
        "arccosh": value.arccosh(),
        "arctanh": value.arctanh(),
        "log": value.log(),
        "log10": value.log(10),
        "log1p": value.log1p(),
        "exp": value.exp(),
        "expm1": np.expm1(value),
        "sqrt": value.sqrt(),
        "abs": value.abs(),
    }
    return frame.with_columns(expr.alias(name) for name, expr in functions.items())


def daily_summary(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    value = pl.when(pl.col("eligible").fill_null(False)).then(pl.col("value").fill_nan(None))
    return (
        frame.group_by("decision_time")
        .agg(
            value.mean().alias("mean"),
            value.std(ddof=0).alias("stddev"),
            value.max().alias("max"),
            value.min().alias("min"),
            value.median().alias("median"),
            value.sum().alias("sum"),
            value.count().alias("notnull_count"),
        )
        .sort("decision_time")
    )


if __name__ == "__main__":
    values = pl.DataFrame(
        {
            "decision_time": [1] * 4 + [2] * 2,
            "value": [1.0, 3.0, float("nan"), 999.0, None, 5.0],
            "eligible": [True, True, True, False, True, False],
        }
    )
    summary = daily_summary(values)
    assert summary["mean"].to_list() == [2.0, None]
    assert summary["stddev"].to_list() == [1.0, None]
    assert summary["sum"].to_list() == [4.0, 0.0]
    assert summary["notnull_count"].to_list() == [2, 0]
    elementary = elementary_features(pl.DataFrame({"value": [1e-20, None]}))
    assert elementary["expm1"].to_list() == [1e-20, None]
    print(json.dumps(summary.to_dicts(), ensure_ascii=True, default=str))
