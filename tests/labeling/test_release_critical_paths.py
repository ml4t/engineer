from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.core.exceptions import DataValidationError
from ml4t.engineer.labeling.atr_barriers import _unique_internal_column, atr_triple_barrier_labels
from ml4t.engineer.labeling.calendar import (
    PandasMarketCalendar,
    SimpleTradingCalendar,
    _calendar_session_ids,
    _temporary_column_name,
    _validate_explicit_calendar,
    calendar_aware_labels,
)
from ml4t.engineer.labeling.horizon_labels import (
    _time_based_horizon_labels,
    _trend_scanning_single_group,
    fixed_time_horizon_labels,
    trend_scanning_labels,
)
from ml4t.engineer.labeling.meta_labels import (
    _validate_finite_number,
    _validate_numeric_column,
    apply_meta_model,
    meta_labels,
)
from ml4t.engineer.labeling.numba_ops import (
    _calculate_barrier_prices,
    _calculate_label_return,
    _check_barrier_touch,
    _initialize_trailing_stop,
    _process_single_event,
    _resolve_barrier_exit,
    _update_trailing_stop,
)
from ml4t.engineer.labeling.percentile_labels import (
    _canonical_horizon,
    _temporary_column,
    _validate_lookback_window,
    rolling_percentile_binary_labels,
)
from ml4t.engineer.labeling.triple_barrier import (
    _finite_float_array,
    _prepare_barrier_arrays,
    _validate_ohlc_arrays,
    triple_barrier_labels,
)
from ml4t.engineer.labeling.uniqueness import (
    _expected_uniqueness_for_candidate,
    build_concurrency,
    calculate_label_uniqueness,
    calculate_sample_weights,
    sequential_bootstrap,
)
from ml4t.engineer.labeling.utils import (
    _get_future_price_lookup,
    duration_to_polars_expr,
    resolve_group_cols,
    resolve_timestamp_col,
)


def test_internal_column_helpers_avoid_collisions() -> None:
    columns = {"temporary", "temporary_"}
    assert _unique_internal_column(columns, "temporary") == "temporary__"
    assert "temporary__" in columns
    assert _temporary_column_name(["temporary", "temporary_1"], "temporary") == "temporary_2"
    assert _temporary_column(["temporary", "temporary_1"], "temporary") == "temporary_2"


def _ohlc_data(rows: int = 5) -> pl.DataFrame:
    timestamps = pl.datetime_range(
        datetime(2024, 1, 1),
        datetime(2024, 1, 1) + timedelta(minutes=rows - 1),
        "1m",
        eager=True,
    )
    closes = [100.0 + index for index in range(rows)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": [price + 1.0 for price in closes],
            "low": [price - 1.0 for price in closes],
            "close": closes,
        }
    )


def test_atr_labels_accept_lazy_input() -> None:
    result = atr_triple_barrier_labels(
        _ohlc_data().lazy(),
        atr_period=1,
        max_holding_bars=1,
    )
    assert len(result) == 5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"atr_tp_multiple": True}, "atr_tp_multiple"),
        ({"atr_sl_multiple": "bad"}, "atr_sl_multiple"),
        ({"atr_tp_multiple": np.inf}, "atr_tp_multiple"),
        ({"atr_sl_multiple": 0.0}, "atr_sl_multiple"),
        ({"atr_period": True}, "atr_period"),
        ({"atr_period": 0}, "atr_period"),
    ],
)
def test_atr_labels_reject_invalid_options(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        atr_triple_barrier_labels(_ohlc_data(), **kwargs)  # type: ignore[arg-type]


def test_atr_labels_warn_for_unbounded_large_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _ohlc_data(5001)
    monkeypatch.setattr(
        "ml4t.engineer.labeling.atr_barriers.triple_barrier_labels",
        lambda frame, **_kwargs: frame,
    )
    with pytest.warns(UserWarning, match="O\\(N\\*5,001\\)"):
        result = atr_triple_barrier_labels(data, atr_period=1)
    assert len(result) == 5001


@pytest.mark.parametrize("value", [True, "1", np.nan, np.inf])
def test_meta_numeric_option_rejects_nonfinite_or_nonnumeric_values(value: object) -> None:
    with pytest.raises(ValueError, match="finite number"):
        _validate_finite_number(value, "value")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("minimum_inclusive", "expected"),
    [(True, "at least"), (False, "greater than")],
)
def test_meta_numeric_option_enforces_minimum(
    minimum_inclusive: bool,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        _validate_finite_number(
            0.0 if not minimum_inclusive else -0.1,
            "value",
            minimum=0.0,
            minimum_inclusive=minimum_inclusive,
        )


def test_meta_numeric_option_enforces_maximum() -> None:
    with pytest.raises(ValueError, match="at most"):
        _validate_finite_number(1.1, "value", maximum=1.0)


@pytest.mark.parametrize(
    ("data", "column", "error", "message"),
    [
        (pl.DataFrame({"other": [1]}), "missing", ValueError, "Missing required"),
        (pl.DataFrame({"value": ["bad"]}), "value", TypeError, "must be numeric"),
        (pl.DataFrame({"value": [np.inf]}), "value", ValueError, "only finite"),
    ],
)
def test_meta_numeric_column_validation(
    data: pl.DataFrame,
    column: str,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _validate_numeric_column(data, column, "test")


def test_meta_public_functions_reject_non_dataframes() -> None:
    with pytest.raises(TypeError, match="Polars DataFrame"):
        meta_labels([], "signal", "return")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Polars DataFrame"):
        apply_meta_model([], "signal", "probability")  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, 0, -1, "not-a-duration", 1.5])
def test_percentile_horizon_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="horizon"):
        _canonical_horizon(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, 0, -1, "not-a-duration", 1.5])
def test_percentile_lookback_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="lookback_window"):
        _validate_lookback_window(value)  # type: ignore[arg-type]


def test_percentile_horizon_and_lookback_accept_duration() -> None:
    assert _canonical_horizon(" 1H ") == ("1h", True)
    assert _validate_lookback_window("1h") is True


def test_percentile_labels_reject_missing_session_column() -> None:
    data = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 4), "1m", eager=True
            ),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )
    with pytest.raises(ValueError, match="Session column"):
        rolling_percentile_binary_labels(
            data,
            percentile=50,
            horizon=1,
            lookback_window=2,
            session_col="session",
        )


@pytest.mark.parametrize("values", [["bad"], [[1.0]], [np.nan], [0.0]])
def test_finite_float_array_rejects_invalid_values(values: list[object]) -> None:
    with pytest.raises(DataValidationError):
        _finite_float_array(values, "values", positive=True)


def test_ohlc_validation_supports_close_only() -> None:
    data = pl.DataFrame({"close": [100.0, 101.0]})
    closes, opens, highs, lows = _validate_ohlc_arrays(data, "close", None, None, None)
    np.testing.assert_array_equal(closes, highs)
    np.testing.assert_array_equal(closes, lows)
    assert np.isnan(opens).all()


@pytest.mark.parametrize(
    ("open_col", "high_col", "low_col", "message"),
    [
        (None, "missing", "low", "High column"),
        (None, "high", "missing", "Low column"),
        (None, "high", None, "provided together"),
        ("open", None, None, "open_col requires"),
        ("missing", "high", "low", "Open column"),
    ],
)
def test_ohlc_validation_rejects_missing_or_unpaired_columns(
    open_col: str | None,
    high_col: str | None,
    low_col: str | None,
    message: str,
) -> None:
    data = pl.DataFrame({"close": [100.0], "high": [101.0], "low": [99.0]})
    with pytest.raises(DataValidationError, match=message):
        _validate_ohlc_arrays(data, "close", open_col, high_col, low_col)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"close": 100.0, "open": 100.0, "high": 99.0, "low": 101.0}, "high prices"),
        ({"close": 102.0, "open": 100.0, "high": 101.0, "low": 99.0}, "close prices"),
        ({"close": 100.0, "open": 102.0, "high": 101.0, "low": 99.0}, "open prices"),
    ],
)
def test_ohlc_validation_rejects_inconsistent_prices(row: dict[str, float], message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        _validate_ohlc_arrays(pl.DataFrame(row), "close", "open", "high", "low")


def _barrier_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 2), "1m", eager=True
            ),
            "close": [100.0, 101.0, 102.0],
            "holding": [1.0, 1.5, 2.0],
        }
    )


def test_barrier_preparation_requires_timestamp_for_duration() -> None:
    data = _barrier_data()
    indices = np.array([0], dtype=np.intp)
    for holding in (timedelta(minutes=1), "1m"):
        config = LabelingConfig.triple_barrier(max_holding_period=holding)
        with pytest.raises(DataValidationError, match="timestamp_col required"):
            _prepare_barrier_arrays(data, config, indices)


def test_barrier_preparation_rejects_bad_holding_column() -> None:
    data = _barrier_data()
    indices = np.array([1], dtype=np.intp)
    missing = LabelingConfig.triple_barrier(max_holding_period="missing")
    with pytest.raises(DataValidationError, match="not found"):
        _prepare_barrier_arrays(data, missing, indices)

    fractional = LabelingConfig.triple_barrier(max_holding_period="holding")
    with pytest.raises(DataValidationError, match="integral"):
        _prepare_barrier_arrays(data, fractional, indices)


def test_barrier_preparation_supports_unsided_events() -> None:
    config = LabelingConfig.triple_barrier().model_copy(update={"side": None})
    arrays = _prepare_barrier_arrays(_barrier_data(), config, np.array([0], dtype=np.intp))
    np.testing.assert_array_equal(arrays[3], [0])


def test_triple_barrier_rejects_wrong_config_type_and_method() -> None:
    data = _barrier_data()
    with pytest.raises(TypeError, match="expects LabelingConfig"):
        triple_barrier_labels(data, object())  # type: ignore[arg-type]
    config = LabelingConfig.fixed_horizon(horizon=1)
    with pytest.raises(DataValidationError, match="method='triple_barrier'"):
        triple_barrier_labels(data, config)


def test_uniqueness_uses_inferred_bar_count_and_invalid_range_default() -> None:
    starts = np.array([0, 3], dtype=np.int64)
    ends = np.array([1, 2], dtype=np.int64)
    np.testing.assert_array_equal(build_concurrency(starts, ends), [1, 1, 0])
    np.testing.assert_array_equal(calculate_label_uniqueness(starts, ends), [1.0, 1.0])


@pytest.mark.parametrize("scheme", ["uniqueness_only", "returns_only", "equal"])
def test_sample_weight_schemes_are_normalized(scheme: str) -> None:
    weights = calculate_sample_weights(
        np.array([0.5, 1.0]),
        np.array([-2.0, 1.0]),
        scheme,  # type: ignore[arg-type]
    )
    assert weights.sum() == pytest.approx(2.0)


def test_zero_sample_weights_fall_back_to_equal() -> None:
    weights = calculate_sample_weights(np.zeros(2), np.zeros(2))
    np.testing.assert_array_equal(weights, [1.0, 1.0])


def test_expected_uniqueness_rejects_reversed_interval_by_zero_weight() -> None:
    result = _expected_uniqueness_for_candidate(
        np.array([2]), np.array([1]), np.zeros(3, dtype=np.int64), 0
    )
    assert result == 0.0


def test_sequential_bootstrap_accepts_generator_and_invalid_intervals() -> None:
    generator = np.random.default_rng(7)
    order = sequential_bootstrap(
        np.array([2, 0]),
        np.array([1, 0]),
        n_draws=2,
        random_state=generator,
    )
    assert len(order) == 2


@pytest.mark.parametrize(
    ("side", "expected"),
    [(-1, (98.0, 101.0)), (0, (102.0, 99.0))],
)
def test_barrier_prices_cover_short_and_unsided(side: int, expected: tuple[float, float]) -> None:
    assert _calculate_barrier_prices(100.0, 0.02, 0.01, side) == pytest.approx(expected)


def test_trailing_stop_kernels_cover_short_and_disabled_paths() -> None:
    assert _initialize_trailing_stop(100.0, 0.01, -1) == pytest.approx(101.0)
    assert _initialize_trailing_stop(100.0, 0.0, -1) == np.inf
    assert _initialize_trailing_stop(100.0, 0.0, 0) == -np.inf
    assert _update_trailing_stop(99.0, 101.0, 0.01, -1) == pytest.approx(99.99)
    assert _update_trailing_stop(99.0, 101.0, 0.0, -1) == 101.0


@pytest.mark.parametrize(
    ("prices", "side", "expected"),
    [((102.0, 100.0), 0, 1), ((100.0, 98.0), 0, -1), ((100.0, 100.0), 1, 0)],
)
def test_barrier_touch_covers_unsided_and_no_hit(
    prices: tuple[float, float], side: int, expected: int
) -> None:
    assert _check_barrier_touch(prices[0], prices[1], 101.0, 99.0, -np.inf, side) == expected


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((97.0, 101.0, 96.0, 98.0, 101.0, np.inf, -1), (1, 97.0)),
        ((102.0, 102.0, 99.0, 98.0, 101.0, np.inf, -1), (-1, 102.0)),
        ((np.nan, 100.0, 97.0, 98.0, 101.0, np.inf, -1), (1, 98.0)),
        ((102.0, 103.0, 99.0, 102.0, 99.0, -np.inf, 1), (1, 102.0)),
        ((98.0, 101.0, 97.0, 102.0, 99.0, -np.inf, 1), (-1, 98.0)),
        ((np.nan, 100.5, 99.5, 102.0, 99.0, -np.inf, 1), (0, np.nan)),
    ],
)
def test_barrier_exit_resolves_gaps_and_intrabar_hits(
    args: tuple[float, float, float, float, float, float, int],
    expected: tuple[int, float],
) -> None:
    label, price = _resolve_barrier_exit(*args)
    assert label == expected[0]
    if np.isnan(expected[1]):
        assert np.isnan(price)
    else:
        assert price == pytest.approx(expected[1])


def test_label_return_handles_zero_entry_and_short_position() -> None:
    assert _calculate_label_return(0.0, 1.0, 1) == 0.0
    assert _calculate_label_return(100.0, 90.0, -1) == pytest.approx(0.1)


def test_single_event_handles_out_of_range_event() -> None:
    values = np.array([100.0])
    result = _process_single_event(values, values, values, values, 1, 0.1, 0.1, 1, 1, 0.0, 1)
    assert result == (0, 1, 100.0, 0.0, 0)


def test_simple_calendar_validation_and_session_boundaries() -> None:
    with pytest.raises(TypeError, match="integer"):
        SimpleTradingCalendar(True)
    with pytest.raises(ValueError, match="positive"):
        SimpleTradingCalendar(0)
    calendar = SimpleTradingCalendar(30)
    assert calendar.next_session_break(datetime(2024, 1, 1)) is None


def test_explicit_calendar_requires_runtime_protocol() -> None:
    with pytest.raises(TypeError, match="is_trading_time"):
        _validate_explicit_calendar(object())  # type: ignore[arg-type]


class _TestCalendar:
    def __init__(self, trading: bool = True, boundary: object = None):
        self.trading = trading
        self.boundary = boundary

    def is_trading_time(self, timestamp: datetime) -> bool:
        return self.trading

    def next_session_break(self, timestamp: datetime) -> object:
        return self.boundary


def test_calendar_session_ids_reject_nontrading_timestamp() -> None:
    data = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)]})
    with pytest.raises(DataValidationError, match="outside"):
        _calendar_session_ids(data, _TestCalendar(trading=False), "timestamp", [])  # type: ignore[arg-type]


def test_calendar_session_ids_validate_returned_boundary() -> None:
    data = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)]})
    with pytest.raises(TypeError, match="datetime or None"):
        _calendar_session_ids(data, _TestCalendar(boundary="bad"), "timestamp", [])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="past boundary"):
        _calendar_session_ids(
            data,
            _TestCalendar(boundary=datetime(2023, 12, 31)),  # type: ignore[arg-type]
            "timestamp",
            [],
        )


def test_calendar_session_ids_reject_non_datetime_column() -> None:
    data = pl.DataFrame({"timestamp": [datetime(2024, 1, 1).date()]})
    with pytest.raises(TypeError, match="must contain datetime"):
        _calendar_session_ids(data, _TestCalendar(), "timestamp", [])  # type: ignore[arg-type]


def test_calendar_session_ids_reject_incompatible_timezones() -> None:
    data = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)]})
    with pytest.raises(TypeError, match="compatible timezone"):
        _calendar_session_ids(
            data,
            _TestCalendar(boundary=datetime(2024, 1, 1, 1, tzinfo=UTC)),  # type: ignore[arg-type]
            "timestamp",
            [],
        )


def test_market_calendar_normalization_and_future_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(TypeError, match="datetime values"):
        PandasMarketCalendar._normalize_timestamp("bad")  # type: ignore[arg-type]
    naive = datetime(2024, 1, 1)
    assert PandasMarketCalendar._normalize_timestamp(naive).tzinfo is UTC

    calendar = PandasMarketCalendar.__new__(PandasMarketCalendar)
    boundary = datetime(2024, 1, 1, 10, tzinfo=UTC)
    monkeypatch.setattr(
        calendar,
        "_trading_intervals_around",
        lambda _day: [(datetime(2024, 1, 1, 9, tzinfo=UTC), boundary, boundary, True)],
    )
    assert calendar.next_session_break(datetime(2024, 1, 1, 8, tzinfo=UTC)) == boundary


def test_calendar_labels_reject_wrong_labeling_method() -> None:
    config = LabelingConfig.fixed_horizon(horizon=1)
    with pytest.raises(ValueError, match="method='triple_barrier'"):
        calendar_aware_labels(_ohlc_data(), config, "auto")


def test_fixed_horizon_rejects_config_conflicts() -> None:
    data = _ohlc_data()
    wrong = LabelingConfig.triple_barrier()
    with pytest.raises(DataValidationError, match="method='fixed_horizon'"):
        fixed_time_horizon_labels(data, config=wrong)

    config = LabelingConfig.fixed_horizon(horizon=2, return_method="binary", threshold=0.1)
    for kwargs in (
        {"horizon": 1},
        {"method": "returns"},
        {"threshold": 0.2},
    ):
        with pytest.raises(DataValidationError, match="conflict"):
            fixed_time_horizon_labels(data, config=config, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [True, -1.0, np.inf])
def test_fixed_horizon_rejects_invalid_threshold(threshold: object) -> None:
    with pytest.raises(ValueError, match="finite nonnegative"):
        fixed_time_horizon_labels(
            _ohlc_data(),
            horizon=1,
            method="binary",
            threshold=threshold,  # type: ignore[arg-type]
        )


def test_fixed_horizon_rejects_threshold_for_returns() -> None:
    with pytest.raises(ValueError, match="only when method='binary'"):
        fixed_time_horizon_labels(_ohlc_data(), horizon=1, method="returns", threshold=0.1)


def test_time_based_horizon_helper_requires_timestamp() -> None:
    with pytest.raises(ValueError, match="requires a timestamp"):
        _time_based_horizon_labels(
            _ohlc_data(),
            "1m",
            "returns",
            "close",
            [],
            None,
            None,
            0.0,
        )


def test_trend_scanning_rejects_wrong_config_and_invalid_threshold() -> None:
    wrong = LabelingConfig.fixed_horizon(horizon=1)
    with pytest.raises(DataValidationError, match="method='trend_scanning'"):
        trend_scanning_labels(_ohlc_data(), config=wrong)
    with pytest.raises(ValueError, match="finite nonnegative"):
        trend_scanning_labels(_ohlc_data(), min_window=2, max_window=3, t_value_threshold=np.inf)


def test_grouped_trend_scanning_processes_each_group() -> None:
    data = pl.concat(
        [
            _ohlc_data().with_columns(pl.lit("A").alias("symbol")),
            _ohlc_data().with_columns(pl.lit("B").alias("symbol")),
        ]
    )
    result = trend_scanning_labels(data, min_window=2, max_window=3, group_col="symbol")
    assert result.group_by("symbol").len().sort("symbol")["len"].to_list() == [5, 5]


def test_trend_scanning_tolerates_regression_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scipy.stats.linregress",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("singular")),
    )
    result = _trend_scanning_single_group(_ohlc_data(), 2, 3, 1, "close", "timestamp", 0.0)
    assert result["label"].null_count() == len(result)


def test_duration_expression_accepts_timedelta() -> None:
    value = pl.select(duration_to_polars_expr(timedelta(seconds=1)).alias("duration")).item()
    assert value == timedelta(seconds=1)


def test_future_price_lookup_requires_timestamp() -> None:
    with pytest.raises(ValueError, match="Timestamp column not found"):
        _get_future_price_lookup(pl.DataFrame({"close": [1.0]}), "1m", "close", None, None, None)


def test_timestamp_resolution_warns_for_missing_and_ambiguous_columns() -> None:
    data = pl.DataFrame(
        {
            "first": [datetime(2024, 1, 1)],
            "second": [datetime(2024, 1, 2)],
        }
    )
    with (
        pytest.warns(UserWarning, match="Specified timestamp_col"),
        pytest.warns(UserWarning, match="Multiple datetime columns"),
    ):
        assert resolve_timestamp_col(data, "missing") == "first"


def test_group_resolution_rejects_missing_columns_and_detects_position() -> None:
    data = pl.DataFrame({"symbol": ["A"], "position": [1]})
    with pytest.raises(DataValidationError, match="group_col 'missing'"):
        resolve_group_cols(data, "missing")
    with pytest.raises(DataValidationError, match="group_col values"):
        resolve_group_cols(data, ["symbol", "missing"])
    assert resolve_group_cols(data, None) == ["symbol", "position"]
