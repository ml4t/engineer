import warnings

import numpy as np
import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="fourier_features",
    category="ml",
    description="Fourier Features - a deterministic seasonal basis on row position",
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def fourier_features(
    close: pl.Expr | str | None = None,
    n_components: int = 10,
    period: int | None = None,
) -> dict[str, pl.Expr]:
    """Build a deterministic seasonal basis: sin and cos of row position.

    These are the Fourier *seasonality terms* used to let a model represent a cycle of a
    known length, not a spectral transform of a series. **No price or other data enters
    them.** Feature ``k`` is ``sin(2 pi k t / period)`` and ``cos(2 pi k t / period)``,
    where ``t`` is the row index, so the output depends only on how many rows there are
    and on ``period``.

    Two consequences worth stating, because both have been misread:

    - The basis is keyed on **row position, not on a timestamp.** It describes the cycle
      you asked for only where rows are sorted and regularly spaced with no gaps. Across a
      weekend, a holiday, or a filtered panel, position and time part company.
    - ``period`` is measured in **rows**, not in minutes. The 390 default is one US equity
      session counted in one-minute bars; on daily bars it describes a 390-session cycle,
      which is almost certainly not what the caller meant. Pass the period explicitly.

    Parameters
    ----------
    close : pl.Expr | str, optional
        **Deprecated and unused.** Accepted so existing callers keep working. The
        function never read it: the previous implementation evaluated
        ``pl.col(close) if isinstance(close, str) else close`` as a bare statement and
        discarded the result, so every version of this function has returned a basis
        independent of the series it was handed. Passing it now warns.
    n_components : int, default 10
        Number of harmonics. Component ``k`` completes ``k`` cycles per ``period`` rows.
    period : int, optional
        Cycle length **in rows**. Defaults to 390 with a warning; see above.

    Returns
    -------
    dict[str, pl.Expr]
        ``fourier_sin_k`` and ``fourier_cos_k`` for k in 1..n_components.

    Raises
    ------
    ValueError
        If n_components or period are not positive
    TypeError
        If n_components or period are not integers
    """
    # Validate inputs
    validate_window(n_components, min_window=1, name="n_components")
    if period is not None:
        validate_window(period, min_window=1, name="period")

    if close is not None:
        warnings.warn(
            "fourier_features() does not read `close` and never has: it returns a "
            "deterministic seasonal basis on row position. Drop the argument, and if you "
            "wanted spectral content of the series, this is not the function for it.",
            DeprecationWarning,
            stacklevel=2,
        )

    if period is None:
        period = 390  # one US equity session in one-minute bars
        warnings.warn(
            "fourier_features() period defaulted to 390 rows, which is one US equity "
            "session counted in one-minute bars. On any other bar size that is a cycle "
            "nobody asked for - on daily bars it is 390 sessions. Pass `period` explicitly.",
            UserWarning,
            stacklevel=2,
        )

    features = {}

    # Row position. Regular spacing is the caller's to guarantee; see the note above.
    t = pl.int_range(pl.len()).cast(pl.Float64)

    for k in range(1, n_components + 1):
        # Fourier basis functions
        features[f"fourier_sin_{k}"] = (2 * np.pi * k * t / period).sin()
        features[f"fourier_cos_{k}"] = (2 * np.pi * k * t / period).cos()

    return features
