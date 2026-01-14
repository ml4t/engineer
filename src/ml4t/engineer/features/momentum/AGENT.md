# momentum/ - 31 Momentum Indicators

All TA-Lib compatible unless noted. Returns DataFrame with indicator column(s).

## Oscillators (0-100 or bounded range)

### rsi(data, period=14, price_col="close") -> DataFrame
Relative Strength Index. Range: 0-100. Overbought >70, oversold <30.

### stoch(data, fastk=5, slowk=3, slowd=3) -> DataFrame
Stochastic Oscillator. Returns: stoch_k, stoch_d columns. Range: 0-100.

### stochf(data, fastk=5, fastd=3) -> DataFrame
Fast Stochastic. Returns: fastk, fastd columns. Range: 0-100.

### stochrsi(data, period=14, fastk=5, fastd=3) -> DataFrame
Stochastic RSI. Returns: stochrsi_k, stochrsi_d. Range: 0-100.

### willr(data, period=14) -> DataFrame
Williams %R. Range: -100 to 0. Overbought >-20, oversold <-80.

### mfi(data, period=14) -> DataFrame
Money Flow Index. Volume-weighted RSI. Range: 0-100.

### cci(data, period=20) -> DataFrame
Commodity Channel Index. Typically -100 to +100, unbounded.

### cmo(data, period=14) -> DataFrame
Chande Momentum Oscillator. Range: -100 to +100.

### ultosc(data, period1=7, period2=14, period3=28) -> DataFrame
Ultimate Oscillator. Multi-timeframe. Range: 0-100.

### bop(data) -> DataFrame
Balance of Power. Range: -1 to +1.

### imi(data, period=14) -> DataFrame
Intraday Momentum Index. Range: 0-100.

## Trend-Following

### adx(data, period=14) -> DataFrame
Average Directional Index. Trend strength 0-100. >25 trending.

### adxr(data, period=14) -> DataFrame
ADX Rating. Smoothed ADX.

### aroon(data, period=25) -> DataFrame
Aroon Oscillator. Returns: aroon_up, aroon_down, aroon_osc. Range: 0-100.

### plus_dm(data, period=14) -> DataFrame
Plus Directional Movement.

### minus_dm(data, period=14) -> DataFrame
Minus Directional Movement.

### directional(data, period=14) -> DataFrame
Directional indicators. Returns: plus_di, minus_di.

## MACD Family

### macd(data, fast=12, slow=26, signal=9) -> DataFrame
MACD. Returns: macd, macd_signal, macd_hist columns.

### macdfix(data, signal=9) -> DataFrame
MACD with fixed 12/26. Returns: macd, macd_signal, macd_hist.

### apo(data, fast=12, slow=26, matype=0) -> DataFrame
Absolute Price Oscillator.

### ppo(data, fast=12, slow=26, matype=0) -> DataFrame
Percentage Price Oscillator.

## Rate of Change

### mom(data, period=10) -> DataFrame
Momentum. Simple price difference.

### roc(data, period=10) -> DataFrame
Rate of Change (percentage).

### rocp(data, period=10) -> DataFrame
ROC Percentage.

### rocr(data, period=10) -> DataFrame
ROC Ratio.

### rocr100(data, period=10) -> DataFrame
ROC Ratio * 100.

### trix(data, period=30) -> DataFrame
Triple Exponential Average ROC.

## Special

### sar(data, acceleration=0.02, maximum=0.2) -> DataFrame
Parabolic SAR. Trend reversal indicator.

## Common Parameters

- `data`: pl.DataFrame with OHLCV columns
- `period`: Lookback window (int)
- `price_col`: Column to use (default "close")
- All return DataFrame with original data + indicator column(s)
