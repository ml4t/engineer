"""Moving-average dispatcher contracts."""

import numpy as np
import pytest

import ml4t.engineer.features.utils.ma_types as ma_types


@pytest.mark.parametrize(
    ("matype", "function_name"),
    [
        (0, "sma_numba"),
        (1, "ema_numba"),
        (2, "wma_numba"),
        (3, "dema_numba"),
        (4, "tema_numba"),
        (5, "trima_numba"),
        (6, "kama_numba"),
        (8, "t3_numba"),
    ],
)
def test_apply_ma_dispatches_every_supported_type(
    monkeypatch: pytest.MonkeyPatch,
    matype: int,
    function_name: str,
) -> None:
    close = np.array([1.0, 2.0, 3.0])
    expected = np.array([4.0, 5.0, 6.0])
    received: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_ma(*args: object, **kwargs: object) -> np.ndarray:
        received.append((args, kwargs))
        return expected

    monkeypatch.setattr(ma_types, function_name, fake_ma)

    assert ma_types.apply_ma(close, period=3, matype=matype) is expected
    if matype == 6:
        assert received == [((close,), {"timeperiod": 3})]
    else:
        assert received == [((close, 3), {})]


@pytest.mark.parametrize(
    ("matype", "match"),
    [
        (7, "MAMA"),
        (-1, "Invalid matype=-1"),
        (9, "Invalid matype=9"),
    ],
)
def test_apply_ma_rejects_unsupported_types(matype: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ma_types.apply_ma(np.array([1.0, 2.0]), period=2, matype=matype)
