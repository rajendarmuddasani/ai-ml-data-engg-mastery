# 06 - Time Series

**Purpose:** Master temporal forecasting for production planning and trend analysis

## Notebooks

- **031_Time_Series_Fundamentals.ipynb** - ARIMA/SARIMA, stationarity (ADF/KPSS), ACF/PACF, differencing
- **032_Exponential_Smoothing.ipynb** - SES/Holt/Holt-Winters, α/β/γ parameters, additive/multiplicative seasonality
- **033_Prophet_Modern_TS.ipynb** - Facebook Prophet, automatic changepoints, multiple seasonality, holiday effects
- **034_VAR_Multivariate_TS.ipynb** - Vector Autoregression, Granger causality, impulse response, joint forecasting

## Key Learning Outcomes

- Forecast yield/test time 4-26 weeks ahead (±2-5% accuracy)
- Handle trend + seasonality automatically (Prophet)
- Model cross-dependencies (yield ↔ test time via VAR)
- Apply to fab capacity planning ($5-10M+ savings)
- Compare ARIMA vs Exponential Smoothing vs Prophet

## Prerequisites

- **02_Regression_Models** (linear models)
- Time series concepts (autocorrelation, stationarity)
- Basic statistics (moving averages, variance)

## Next Steps

Continue to **07_Deep_Learning** for neural network-based forecasting (LSTM, Transformers).

## Future Notebooks (035+)

- **035_LSTM_Time_Series** - Long Short-Term Memory networks for non-linear temporal patterns
- **036+** - Advanced topics (state space models, Kalman filters)
