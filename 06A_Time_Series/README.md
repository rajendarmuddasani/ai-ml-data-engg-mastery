# 06 - Time Series

**Purpose:** Master temporal forecasting for production planning and trend analysis

Time series analysis is critical for semiconductor manufacturing, where forecasting yield trends, test time patterns, and equipment failure rates can save millions in capacity planning and proactive maintenance. This section covers classical statistical methods (ARIMA), smoothing techniques (Exponential Smoothing), modern automated forecasting (Prophet), and multivariate dependencies (VAR).

## 📊 Learning Path Statistics

- **Total Notebooks:** 4
- **Completion Status:** ✅ All complete
- **Topics Covered:** Classical forecasting, exponential smoothing, automated forecasting, multivariate time series
- **Applications:** Production yield forecasting, test time optimization, fab capacity planning, equipment maintenance scheduling

---

## 📚 Notebooks

### [031_Time_Series_Fundamentals.ipynb](031_Time_Series_Fundamentals.ipynb)
**Classical Time Series Analysis with ARIMA/SARIMA**

Master the mathematical foundations of time series forecasting using Autoregressive Integrated Moving Average (ARIMA) models and their seasonal extensions (SARIMA).

**Topics Covered:**
- **Stationarity Testing:** Augmented Dickey-Fuller (ADF), KPSS tests
- **Time Series Components:** Trend, seasonality, residual decomposition
- **ACF/PACF Analysis:** Autocorrelation and partial autocorrelation functions
- **ARIMA Models:** AR(p), MA(q), I(d) components and (p,d,q) order selection
- **SARIMA Models:** Seasonal ARIMA with (P,D,Q,s) notation
- **Differencing:** Making non-stationary series stationary
- **Box-Jenkins Methodology:** Identification → Estimation → Diagnostic Checking

**Real-World Applications:**
- **Yield Forecasting:** Predict weekly wafer yield 4-26 weeks ahead (±2-5% MAPE)
- **Test Time Trends:** Forecast parametric test time changes for capacity planning
- **Equipment MTBF:** Model mean time between failures for maintenance scheduling
- **Throughput Prediction:** Forecast fab output for supply chain coordination

**Mathematical Foundations:**
```
ARIMA(p,d,q):
  AR(p): y_t = φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p} + ε_t
  MA(q): y_t = ε_t + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θₚε_{t-q}
  I(d): Apply differencing d times to achieve stationarity

SARIMA(p,d,q)(P,D,Q,s):
  Adds seasonal components with period s (e.g., s=52 for weekly data)
```

**Learning Outcomes:**
- Test time series for stationarity using ADF/KPSS tests
- Interpret ACF/PACF plots to determine ARIMA orders
- Build ARIMA models from scratch using MLE
- Diagnose model fit using residual analysis
- Apply seasonal differencing for quarterly/monthly patterns
- Compare ARIMA vs naive/seasonal naive benchmarks

---

### [032_Exponential_Smoothing.ipynb](032_Exponential_Smoothing.ipynb)
**Exponential Smoothing Methods (SES, Holt, Holt-Winters)**

Learn weighted averaging techniques that give more weight to recent observations, ideal for short-term forecasting in production environments.

**Topics Covered:**
- **Simple Exponential Smoothing (SES):** α parameter, level component only
- **Holt's Linear Trend Method:** α (level) + β (trend) parameters
- **Holt-Winters Method:** α (level) + β (trend) + γ (seasonality)
- **Additive vs Multiplicative Seasonality:** When to use each model
- **Parameter Optimization:** Grid search vs analytical solutions
- **State Space Models:** ETS framework (Error, Trend, Seasonality)
- **Forecasting Intervals:** Prediction uncertainty quantification

**Real-World Applications:**
- **Daily Yield Tracking:** React quickly to process shifts (α=0.3-0.7)
- **Test Time Smoothing:** Filter noise in parametric test execution times
- **Equipment Utilization:** Forecast next-week tool availability
- **Inventory Management:** Short-term component demand forecasting

**Mathematical Foundations:**
```
Simple Exponential Smoothing (SES):
  ŷ_{t+1} = α·y_t + (1-α)·ŷ_t
  
Holt's Method (Level + Trend):
  Level: l_t = α·y_t + (1-α)(l_{t-1} + b_{t-1})
  Trend: b_t = β(l_t - l_{t-1}) + (1-β)b_{t-1}
  Forecast: ŷ_{t+h} = l_t + h·b_t
  
Holt-Winters (Level + Trend + Seasonality):
  Additive: ŷ_{t+h} = l_t + h·b_t + s_{t+h-m}
  Multiplicative: ŷ_{t+h} = (l_t + h·b_t) × s_{t+h-m}
```

**Learning Outcomes:**
- Select α, β, γ parameters based on volatility characteristics
- Distinguish additive vs multiplicative seasonality patterns
- Implement exponential smoothing from scratch
- Apply Holt-Winters to quarterly production data
- Validate forecasts using rolling window cross-validation
- Compare smoothing methods to ARIMA for short horizons

---

### [033_Prophet_Modern_TS.ipynb](033_Prophet_Modern_TS.ipynb)
**Facebook Prophet for Automated Forecasting**

Master Facebook's Prophet library, designed for business time series with strong seasonal patterns, holiday effects, and automatic changepoint detection.

**Topics Covered:**
- **Prophet Components:** Trend (piecewise linear/logistic), seasonality (Fourier), holidays
- **Automatic Changepoint Detection:** Identify structural breaks in trends
- **Multiple Seasonality:** Daily, weekly, yearly patterns simultaneously
- **Holiday Effects:** Custom events (fab shutdowns, product launches)
- **Uncertainty Intervals:** Trend + seasonal uncertainty quantification
- **Additive Model:** y(t) = g(t) + s(t) + h(t) + ε_t
- **Saturating Growth:** Logistic growth curves for capacity-limited systems

**Real-World Applications:**
- **Fab Yield Forecasting:** Automatic detection of process excursions
- **Test Program Updates:** Model impact of software releases on test time
- **Capacity Planning:** Forecast wafer starts 6-12 months ahead
- **Holiday Effects:** Model fab shutdowns, Chinese New Year, summer slowdowns

**Mathematical Foundations:**
```
Prophet Model:
  y(t) = g(t) + s(t) + h(t) + ε_t
  
  g(t): Piecewise linear or logistic growth trend
  s(t): Periodic seasonality (Fourier series)
        s(t) = Σ[aₙcos(2πnt/P) + bₙsin(2πnt/P)]
  h(t): Holiday effects (indicator variables)
  ε_t: Error term (normal or Laplace distribution)

Changepoint Detection:
  - Prophet detects n_changepoints (default: 25) evenly spaced
  - Regularization prevents overfitting (changepoint_prior_scale)
```

**Learning Outcomes:**
- Deploy Prophet with minimal hyperparameter tuning
- Add custom seasonality (weekly fab cycles, quarterly reviews)
- Model holiday effects (fab maintenance windows, product transitions)
- Interpret changepoint dates for root cause analysis
- Compare Prophet to ARIMA/exponential smoothing
- Scale to forecasting 100+ time series (parallel execution)

---

### [034_VAR_Multivariate_TS.ipynb](034_VAR_Multivariate_TS.ipynb)
**Vector Autoregression (VAR) for Multivariate Time Series**

Learn to model joint dynamics between multiple time series variables, capturing cross-dependencies and feedback loops in complex systems.

**Topics Covered:**
- **VAR Models:** Multivariate extension of AR models
- **Granger Causality:** Testing if X helps predict Y
- **Impulse Response Functions (IRF):** Shock propagation analysis
- **Forecast Error Variance Decomposition (FEVD):** Contribution of shocks
- **Cointegration:** Long-run equilibrium relationships
- **VAR Order Selection:** AIC, BIC, likelihood ratio tests
- **Structural VAR (SVAR):** Identifying contemporaneous relationships

**Real-World Applications:**
- **Yield ↔ Test Time:** Model bidirectional causality (faster tests → higher yield? vice versa?)
- **Equipment ↔ Throughput:** How tool availability affects fab output
- **Multi-Site Coordination:** Forecast demand across wafer fab + test sites
- **Process Control:** Model interactions between temperature, pressure, flow rate

**Mathematical Foundations:**
```
VAR(p) Model for k variables:
  y_t = A₁y_{t-1} + A₂y_{t-2} + ... + Aₚy_{t-p} + ε_t
  
  where:
    y_t = [y₁,t, y₂,t, ..., yₖ,t]ᵀ  (k×1 vector)
    Aᵢ = k×k coefficient matrices
    ε_t = white noise vector
    
Example (k=2, p=1):
  yield_t = α₁₁yield_{t-1} + α₁₂test_time_{t-1} + ε₁,t
  test_time_t = α₂₁yield_{t-1} + α₂₂test_time_{t-1} + ε₂,t

Granger Causality:
  X Granger-causes Y if:
    Past values of X improve prediction of Y beyond past values of Y alone
```

**Learning Outcomes:**
- Build VAR models for 2-5 interdependent time series
- Test Granger causality to identify directional influences
- Compute impulse response functions for shock analysis
- Interpret FEVD to quantify variable importance
- Apply cointegration tests (Johansen, Engle-Granger)
- Compare VAR to univariate ARIMA for multivariate forecasting

---

## 🔗 Prerequisites

**Required Knowledge:**
- **02_Regression_Models:** Linear regression, residual analysis, optimization (MLE)
- **Basic Statistics:** Mean, variance, covariance, correlation, hypothesis testing
- **Time Series Concepts:** Autocorrelation, stationarity, trend, seasonality

**Recommended Background:**
- **Python Libraries:** pandas (time series manipulation), matplotlib (plotting)
- **Statistical Tests:** t-tests, F-tests, chi-square tests
- **Linear Algebra:** Matrix operations for VAR models

---

## 🎯 Key Learning Outcomes

By completing this section, you will:

✅ **Master Classical Forecasting:** Build ARIMA/SARIMA models from scratch  
✅ **Apply Exponential Smoothing:** Select optimal α, β, γ for production data  
✅ **Deploy Prophet at Scale:** Automate forecasting for 100+ time series  
✅ **Model Cross-Dependencies:** Use VAR for multivariate forecasting  
✅ **Forecast with Confidence:** Quantify prediction intervals (±2-5% MAPE)  
✅ **Business Impact:** Apply to fab capacity planning ($5-10M+ annual savings)  
✅ **Method Selection:** Compare ARIMA vs Exponential Smoothing vs Prophet vs VAR  
✅ **Diagnostic Skills:** Validate models using residual analysis, cross-validation  

---

## 📈 Technique Comparison Table

| Method | Best For | Horizons | Seasonality | Multivariate | Automation |
|--------|----------|----------|-------------|--------------|------------|
| **ARIMA** | Stationary series | Medium (4-26 weeks) | Manual (SARIMA) | No | Low |
| **Exp. Smoothing** | Short-term forecasts | Short (1-8 weeks) | Holt-Winters | No | Medium |
| **Prophet** | Business time series | Long (6-12 months) | Automatic | No | High |
| **VAR** | Cross-dependencies | Medium (4-26 weeks) | Manual | Yes | Low |

**When to Use:**
- **ARIMA:** Interpretable models, need statistical rigor (academic papers, regulatory)
- **Exponential Smoothing:** Real-time dashboards, operational forecasts (daily/weekly)
- **Prophet:** Production deployment at scale, minimal tuning, automatic anomalies
- **VAR:** System dynamics, causal analysis, multi-site coordination

---

## 🏭 Post-Silicon Validation Applications

### 1. **Weekly Yield Forecasting (ARIMA/Prophet)**
- **Input:** 52+ weeks of historical yield data (wafer sort, final test)
- **Output:** 4-26 week ahead yield predictions (±2-5% MAPE)
- **Business Value:** $5-10M annual savings via proactive capacity adjustments

### 2. **Test Time Optimization (Exponential Smoothing)**
- **Input:** Daily parametric test execution times per device
- **Output:** Next-week test time forecasts for capacity planning
- **Business Value:** Reduce test costs 10-15% by optimizing parallel test slots

### 3. **Fab Capacity Planning (Prophet + Holidays)**
- **Input:** Monthly wafer starts, historical shutdowns, product transitions
- **Output:** 6-12 month ahead capacity forecasts with uncertainty bands
- **Business Value:** Align supply chain, reduce expedite costs $2-5M/year

### 4. **Cross-Site Coordination (VAR)**
- **Input:** Wafer fab output + test site throughput (weekly data)
- **Output:** Joint forecasts capturing fab→test dependencies
- **Business Value:** Synchronize multi-site operations, reduce WIP buffers 20-30%

---

## 🔄 Next Steps

After mastering time series fundamentals:

1. **07_Deep_Learning:** Apply LSTM networks for non-linear temporal patterns
2. **06_ML_Engineering:** Feature engineering for time series (lag features, rolling statistics)
3. **09_Data_Engineering:** Build production pipelines for real-time forecasting
4. **13_MLOps_Production_ML:** Deploy Prophet/ARIMA models in Airflow/cloud environments

**Advanced Topics (Future Notebooks):**
- **035_LSTM_Time_Series:** Recurrent neural networks for complex patterns
- **State Space Models:** Kalman filters for real-time tracking
- **Transfer Learning:** Pre-trained models for small time series datasets

---

## 📝 Project Ideas

### Post-Silicon Validation Projects

1. **Device Yield Predictor**
   - Forecast weekly yield 13 weeks ahead using SARIMA (52-week seasonality)
   - Detect changepoints with Prophet for process excursions
   - Target: ±3% MAPE on test dataset

2. **Test Time Dashboard**
   - Real-time exponential smoothing for parametric test times
   - Holt-Winters for daily forecasts with weekly seasonality
   - Alert when test time exceeds forecast + 2σ

3. **Equipment MTBF Forecaster**
   - ARIMA models for mean time between failures (MTBF) per tool
   - Prophet for maintenance scheduling with holiday effects (fab shutdowns)
   - Reduce unplanned downtime 20-30%

4. **Multi-Site VAR Coordinator**
   - VAR(2) model for wafer fab output + test site throughput
   - Granger causality tests to identify bottlenecks
   - Optimize WIP buffers to reduce inventory costs

### General AI/ML Projects

5. **Stock Price Forecaster**
   - ARIMA/Prophet for daily closing prices (S&P 500, NASDAQ)
   - Compare model performance on training vs test periods
   - Backtest trading strategy (buy/sell signals)

6. **Energy Demand Predictor**
   - Holt-Winters for hourly electricity consumption (daily + weekly seasonality)
   - Prophet for long-term planning (6-12 months ahead)
   - Optimize power plant scheduling

7. **Sales Forecasting Engine**
   - Prophet for retail sales with holiday effects (Black Friday, Christmas)
   - Automatic changepoint detection for marketing campaign impact
   - Deploy to production for inventory management

8. **Multi-Product VAR System**
   - VAR model for cross-product sales dependencies
   - Impulse response analysis for promotional campaigns
   - FEVD to quantify product substitution effects

---

**Total Notebooks in Section:** 4  
**Estimated Completion Time:** 8-12 hours  
**Difficulty Level:** Intermediate  
**Prerequisites:** Linear regression, basic statistics, pandas

*Last Updated: December 2025*
