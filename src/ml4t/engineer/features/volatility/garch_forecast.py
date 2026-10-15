import numpy as np
import numpy.typing as npt
import polars as pl

from ml4t.engineer._numba import jit
from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_positive,
    validate_threshold,
    validate_window,
)

# Helper functions


@jit(nopython=True, cache=True)  # type: ignore[misc]
def garch_volatility_forecast_nb(
    returns: npt.NDArray[np.float64],
    omega: float = 0.00001,
    alpha: float = 0.1,
    beta: float = 0.85,
    horizon: int = 1,
) -> npt.NDArray[np.float64]:
    """GARCH(1,1) volatility forecast (Numba optimized).

    Simple GARCH implementation for volatility forecasting.
    """
    n = len(returns)
    result = np.full(n, np.nan)

    # Filter NaN close to get valid returns
    valid_mask = ~np.isnan(returns)
    if not valid_mask.any():
        return result  # All NaN input, return all NaN

    # Work with valid returns only
    valid_returns = returns[valid_mask]
    n_valid = len(valid_returns)
    sigma2 = np.zeros(n_valid)

    # Initialize from the first observed return. A sample variance over the
    # following returns would leak those future shocks into earlier forecasts.
    sigma2[0] = valid_returns[0] * valid_returns[0]

    # sigma2[t] is the conditional variance of return t given information through t-1.
    for t in range(1, n_valid):
        sigma2[t] = omega + alpha * valid_returns[t - 1] ** 2 + beta * sigma2[t - 1]

    # Forecast from information available at t. The first step uses return t;
    # later steps replace the unknown shock with its conditional expectation.
    forecast = np.zeros(n_valid)
    persistence = alpha + beta
    for t in range(n_valid):
        h_ahead = omega + alpha * valid_returns[t] * valid_returns[t] + beta * sigma2[t]
        for _step in range(1, horizon):
            h_ahead = omega + persistence * h_ahead
        forecast[t] = h_ahead

    # Map back to original positions
    valid_idx = 0
    for i in range(n):
        if valid_mask[i]:
            result[i] = np.sqrt(forecast[valid_idx])
            valid_idx += 1

    return result


# Main feature function


@feature(
    name="garch_forecast",
    category="volatility",
    description="GARCH Volatility Forecast - conditional volatility model",
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def garch_forecast(
    returns: pl.Expr | str,
    horizon: int = 1,
    omega: float = 0.00001,
    alpha: float = 0.1,
    beta: float = 0.85,
) -> pl.Expr:
    """GARCH(1,1) volatility forecast.

    Provides volatility forecasts using a simple GARCH model, useful
    for predicting future volatility regimes.

    Parameters
    ----------
    returns : pl.Expr | str
        Returns column
    horizon : int, default 1
        Forecast horizon
    omega : float, default 0.00001
        GARCH constant term
    alpha : float, default 0.1
        ARCH coefficient
    beta : float, default 0.85
        GARCH coefficient

    Returns
    -------
    pl.Expr
        GARCH volatility forecast

    Raises
    ------
    ValueError
        If horizon is not positive, GARCH parameters are negative, or alpha + beta >= 1
    TypeError
        If horizon is not an integer or GARCH parameters are not numeric
    """
    # Validate inputs
    validate_window(horizon, min_window=1, name="horizon")
    validate_positive(omega, name="omega")
    validate_threshold(alpha, 0.0, 1.0, name="alpha")
    validate_threshold(beta, 0.0, 1.0, name="beta")

    if alpha + beta >= 1.0:
        raise ValueError(
            f"GARCH model requires alpha + beta < 1 for stationarity, got {alpha + beta:.6f}",
        )

    returns = pl.col(returns) if isinstance(returns, str) else returns

    # Apply GARCH forecast
    return returns.map_batches(
        lambda x: pl.Series(
            garch_volatility_forecast_nb(x.to_numpy(), omega, alpha, beta, horizon),
        ),
    )
