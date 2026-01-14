# statistics/ - 8 Statistical Features

Rolling statistical calculations. All TA-Lib compatible.

## Basic Statistics

### stddev(data, period=20, nbdev=1, price_col="close") -> DataFrame
Rolling standard deviation.

### var(data, period=20, nbdev=1, price_col="close") -> DataFrame
Rolling variance.

### avgdev(data, period=20, price_col="close") -> DataFrame
Rolling average deviation from mean.

## Linear Regression

### linearreg(data, period=14, price_col="close") -> DataFrame
Linear regression value at end of period.

### linearreg_slope(data, period=14, price_col="close") -> DataFrame
Slope of linear regression line.

### linearreg_angle(data, period=14, price_col="close") -> DataFrame
Angle of linear regression in degrees.

### linearreg_intercept(data, period=14, price_col="close") -> DataFrame
Y-intercept of regression line.

### tsf(data, period=14, price_col="close") -> DataFrame
Time Series Forecast. Regression projected forward.

## Structural Break

### structural_break(data, period=50, min_window=10) -> DataFrame
Detects regime changes using CUSUM test.
Note: File is 646 lines - complex implementation.
