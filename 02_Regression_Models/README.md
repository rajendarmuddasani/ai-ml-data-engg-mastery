# 02 - Regression Models

**Purpose:** Master supervised learning for continuous target prediction and classification

**Why Regression?** Regression models are the foundation of machine learning. They teach you fundamental concepts like loss functions, gradient descent, regularization, and model evaluation - all critical for understanding advanced ML. Start here to build a solid ML foundation.

---

## 📚 Notebooks (010-015)

### **Linear Regression Foundation**

#### **010_Linear_Regression.ipynb** ✅ **GOLD STANDARD** (34 cells)
**The foundation of all machine learning - master this first!**

**Topics Covered:**
- Ordinary Least Squares (OLS) mathematics
- Simple vs Multiple Linear Regression
- Gradient descent optimization
- Model assumptions (linearity, homoscedasticity, independence, normality)
- Diagnostic tools (VIF, residual analysis, Q-Q plots)
- Feature engineering for regression

**Implementation Layers:**
- **From Scratch**: NumPy-only implementation showing all math
- **Production**: Scikit-learn with complete pipeline
- **Comparison**: Validate scratch implementation against sklearn

**8 Real-World Projects:**
1. **Device Power Consumption Predictor** (post-silicon) - Predict Idd from Vdd, frequency, temperature
2. **Test Time Estimator** (ATE optimization) - Optimize test flow for throughput
3. **Parametric Yield Prediction** (manufacturing) - Predict yield% from parametric test data
4. **Voltage-Frequency Characterization** (V-F curves) - Model device performance across conditions
5. **Sales Forecasting** (general) - Predict revenue from historical data
6. **Real Estate Price Prediction** - Housing price models
7. **Customer Lifetime Value (CLV)** - Predict customer value
8. **Energy Consumption Forecasting** - Smart grid applications

**Learning Outcomes:**
- Derive regression equations mathematically
- Implement gradient descent from scratch
- Diagnose model assumptions and violations
- Apply to STDF semiconductor test data
- Build production-ready ML pipelines

**Visual Learning:** Complete workflow Mermaid diagrams showing data → model → deployment

---

#### **011_Polynomial_Regression.ipynb** ✅ (37 cells)
**Model non-linear relationships while staying interpretable**

**Topics Covered:**
- Polynomial feature transformation
- Degree selection (cross-validation)
- Bias-variance tradeoff visualization
- Overfitting prevention techniques
- Interaction terms and feature crosses

**Key Concepts:**
- **Polynomial Features**: Transform x → [x, x², x³, ...] to capture non-linearity
- **Degree Selection**: Use validation curves to find optimal complexity
- **Regularization**: Combine with Ridge/Lasso to prevent overfitting
- **Interpretability**: Maintain coefficient interpretation despite complexity

**Real-World Applications:**
- **Post-Silicon**: Temperature-performance curves (device speed vs temperature is non-linear)
- **Post-Silicon**: Power consumption models (quadratic relationship with voltage)
- **General**: Growth modeling (S-curves, exponential trends)
- **General**: Demand forecasting with seasonal patterns

**Learning Outcomes:**
- Model non-linear relationships without losing interpretability
- Select polynomial degree systematically
- Visualize bias-variance tradeoff
- Apply to wafer test thermal data

---

### **Regularization & Feature Selection**

#### **012_Ridge_Lasso_ElasticNet.ipynb** ✅ (33 cells)
**Handle high-dimensional data and prevent overfitting**

**Topics Covered:**
- **Ridge Regression (L2)**: Shrinks coefficients, handles multicollinearity
- **Lasso Regression (L1)**: Feature selection via coefficient zeroing
- **ElasticNet**: Combines L1 + L2 benefits
- Regularization path visualization
- Cross-validation for hyperparameter tuning
- Multicollinearity detection (VIF)

**Mathematical Foundation:**
- L1 penalty: Σ|βᵢ| → sparse solutions (feature selection)
- L2 penalty: Σβᵢ² → smooth solutions (shrinkage)
- ElasticNet: α(L1) + (1-α)(L2) → best of both worlds

**Real-World Applications:**
- **Post-Silicon**: High-dimensional STDF parameter reduction (1000+ test parameters → 20 key features)
- **Post-Silicon**: Multicollinear device parameters (voltage/current/power are correlated)
- **General**: Gene expression analysis (thousands of features, few samples)
- **General**: Text classification with TF-IDF (sparse, high-dimensional)

**Learning Outcomes:**
- Choose between Ridge/Lasso/ElasticNet based on problem characteristics
- Perform automatic feature selection with Lasso
- Handle correlated features effectively
- Tune regularization strength systematically

---

### **Classification with Regression**

#### **013_Logistic_Regression.ipynb** ✅ (39 cells)
**Binary and multi-class classification with probabilistic outputs**

**Topics Covered:**
- Sigmoid function and log-odds (logit)
- Maximum Likelihood Estimation (MLE)
- Binary vs Multi-class (One-vs-Rest, Softmax)
- Decision boundaries and classification thresholds
- Model evaluation: ROC curve, AUC, precision-recall, confusion matrix
- Probability calibration

**Mathematical Foundation:**
- Sigmoid: σ(z) = 1 / (1 + e⁻ᶻ)
- Log-odds: log(p/(1-p)) = β₀ + β₁x₁ + ...
- Cross-entropy loss minimization
- Softmax for multi-class: P(y=k) = e^(zₖ) / Σe^(zᵢ)

**Real-World Applications:**
- **Post-Silicon**: Pass/Fail prediction from parametric test data
- **Post-Silicon**: Bin classification (speed grading, power bins)
- **Post-Silicon**: Failure mode identification (open/short/parametric)
- **General**: Customer churn prediction
- **General**: Credit risk assessment (default/no-default)
- **General**: Medical diagnosis (disease/no-disease)

**Learning Outcomes:**
- Understand difference between regression and classification loss functions
- Build calibrated probability classifiers
- Interpret ROC curves and select optimal thresholds
- Handle class imbalance in manufacturing data

---

### **Advanced Regression Techniques**

#### **014_Support_Vector_Regression.ipynb** ✅ (17 cells)
**Robust regression for data with outliers**

**Topics Covered:**
- Epsilon-insensitive loss function
- Kernel trick (linear, polynomial, RBF, sigmoid)
- Support vectors and margin maximization
- Hyperparameter tuning (C, epsilon, gamma)
- Comparison with linear regression

**Key Concepts:**
- **Epsilon-tube**: Ignore errors within ε of prediction (robustness to noise)
- **Kernel Methods**: Transform data to higher dimensions for non-linearity
- **Sparsity**: Only support vectors matter (computational efficiency)
- **Robustness**: Less sensitive to outliers than OLS regression

**Real-World Applications:**
- **Post-Silicon**: Robust prediction with outlier test data (ATE measurement noise)
- **Post-Silicon**: Device characterization across PVT corners (outlier tolerance)
- **General**: Financial forecasting (resilient to market anomalies)
- **General**: Sensor data with measurement errors

**Learning Outcomes:**
- Choose between SVR and linear regression based on data characteristics
- Apply kernel methods for non-linear patterns
- Handle outliers without data cleaning
- Tune SVR hyperparameters effectively

---

#### **015_Quantile_Regression.ipynb** ✅ (34 cells)
**Predict conditional quantiles and build prediction intervals**

**Topics Covered:**
- Conditional quantile estimation (median, quartiles, percentiles)
- Prediction intervals vs point predictions
- Asymmetric loss functions
- Quantile crossing prevention
- Value at Risk (VaR) and Conditional VaR (CVaR)

**Mathematical Foundation:**
- Quantile loss: ρₜ(u) = u(τ - I(u<0))
- Minimizes pinball loss for τ-th quantile
- Multiple quantiles → full conditional distribution
- Interpretable uncertainty quantification

**Real-World Applications:**
- **Post-Silicon**: Test time prediction intervals (worst-case planning for throughput)
- **Post-Silicon**: Yield forecasting with confidence bounds (manufacturing risk assessment)
- **Post-Silicon**: Parametric limit validation (what's the 99th percentile?)
- **General**: Financial risk management (VaR for portfolio losses)
- **General**: Demand forecasting with uncertainty (inventory optimization)
- **General**: Energy load forecasting (capacity planning)

**Learning Outcomes:**
- Build prediction intervals instead of point predictions
- Model uncertainty explicitly
- Apply to risk-sensitive semiconductor manufacturing decisions
- Understand quantile loss vs MSE loss

---

## 🎯 Learning Path

**Recommended Order:**
1. **010 - Linear Regression** ⭐ **START HERE** - Foundation for everything
2. **011 - Polynomial Regression** - Add non-linearity while staying simple
3. **012 - Regularization** - Handle high-dimensional data
4. **013 - Logistic Regression** - Transition to classification
5. **014 - SVR** - Advanced technique for outliers
6. **015 - Quantile Regression** - Uncertainty quantification

**Time Estimate:** 2-3 weeks (intensive) | 4-6 weeks (moderate pace)

---

## 📊 Section Statistics

| Metric | Value |
|--------|-------|
| **Total Notebooks** | 6 |
| **Complete Notebooks** | 6 (100% ✅) |
| **Total Cells** | 194+ |
| **Real-World Projects** | 30+ |
| **Implementation Styles** | From scratch + Production |

---

## 🔑 Key Learning Outcomes

After completing this section, you will:

✅ **Mathematical Foundation**
- Derive regression equations from first principles
- Understand loss functions (MSE, MAE, Cross-Entropy, Quantile)
- Implement gradient descent optimization
- Apply matrix calculus to ML problems

✅ **Implementation Skills**
- Build models from scratch (NumPy only)
- Use scikit-learn production APIs
- Create end-to-end ML pipelines
- Handle real-world data issues

✅ **Model Evaluation**
- Diagnose model assumptions
- Detect overfitting and underfitting
- Calculate metrics (RMSE, R², AUC, Precision/Recall)
- Build calibrated probability models

✅ **Domain Applications**
- Apply to semiconductor test data (STDF format)
- Predict device yield, test time, parametric values
- Handle spatial correlations (wafer maps)
- Build production models for ATE systems

✅ **Advanced Techniques**
- Regularization for high-dimensional data
- Feature selection with Lasso
- Quantile regression for uncertainty
- Robust methods for outliers

---

## 🔗 Prerequisites

**Before starting this section, complete:**
- ✅ **[01_Foundations](../01_Foundations/)** - Python, NumPy, basic programming
- ✅ Linear algebra - Matrix operations, matrix inversion, eigenvalues
- ✅ Basic calculus - Derivatives, partial derivatives, chain rule
- ✅ Statistics - Mean, variance, distributions, hypothesis testing

**If rusty on math:** Review notebooks 004-007 in Foundations

---

## ➡️ Next Steps

After mastering regression models, continue to:

1. **[03_Tree_Based_Models](../03_Tree_Based_Models/)** - Non-linear, interpretable models (Decision Trees, Random Forest, XGBoost)
2. **[04_Distance_Based_Models](../04_Distance_Based_Models/)** - KNN, SVM for classification
3. **[06_ML_Engineering](../06_ML_Engineering/)** - Feature engineering, model evaluation, hyperparameter tuning

---

## 💡 Study Tips

1. **Master 010 First** - Linear Regression (010) is the gold standard template. Study it thoroughly.
2. **Implement from Scratch** - Don't skip the NumPy implementations - they teach you the math.
3. **Compare with Sklearn** - Validate your scratch implementations against production libraries.
4. **Work Through Projects** - Each notebook has 4-8 real projects - build at least 2 per notebook.
5. **Visualize Everything** - Plot residuals, learning curves, decision boundaries.
6. **Apply to Real Data** - Use STDF files if in semiconductor domain, or Kaggle datasets.

---

## 🛠️ Tools & Libraries

**Core Libraries:**
- NumPy (from-scratch implementations)
- Scikit-learn (production models)
- Pandas (data manipulation)
- Matplotlib/Seaborn (visualization)
- Statsmodels (statistical diagnostics)

**Domain-Specific:**
- STDF file parsers (for semiconductor data)
- Wafer map visualization libraries

---

## 📈 Progress Tracking

Mark notebooks as complete as you master them:

- [ ] 010_Linear_Regression ⭐ **START HERE**
- [ ] 011_Polynomial_Regression
- [ ] 012_Ridge_Lasso_ElasticNet
- [ ] 013_Logistic_Regression
- [ ] 014_Support_Vector_Regression
- [ ] 015_Quantile_Regression

---

## 🎓 Assessment Criteria

You've mastered this section when you can:

✅ Derive the normal equation for linear regression  
✅ Implement gradient descent from scratch  
✅ Explain L1 vs L2 regularization with examples  
✅ Interpret logistic regression coefficients as odds ratios  
✅ Build a complete ML pipeline (data → model → evaluation)  
✅ Apply models to real semiconductor test data  
✅ Choose the right regression technique for a given problem  
✅ Debug model failures (overfitting, poor assumptions, etc.)

---

## 🌟 Why This Section Matters

**Industry Relevance:**
- Regression is used in 80%+ of production ML systems
- Foundation for understanding neural networks (backprop is gradient descent!)
- Critical for semiconductor yield optimization and test cost reduction
- Interpretable models preferred in regulated industries (semiconductors, finance, healthcare)

**Career Impact:**
- Interview staple (derive gradient descent, explain regularization)
- Required for ML engineer roles at Qualcomm, AMD, NVIDIA, Intel
- Practical tool for data-driven decision making
- Gateway to advanced ML techniques

---

**Last Updated:** December 2025  
**Status:** All 6 notebooks complete (100% ✅)  
**Maintainer:** [@rajendarmuddasani](https://github.com/rajendarmuddasani)
