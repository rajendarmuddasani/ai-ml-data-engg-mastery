# 06 - ML Engineering

**Purpose:** Master the full ML lifecycle from feature engineering to production deployment

Machine Learning Engineering bridges the gap between data science experiments and production systems. This section covers the critical engineering skills needed to build robust, scalable ML systems: feature engineering, model evaluation, hyperparameter tuning, interpretability, and deployment strategies. Essential for transitioning from notebooks to production-grade ML applications.

## 📊 Learning Path Statistics

- **Total Notebooks:** 13
- **Completion Status:** ✅ All complete
- **Topics Covered:** Feature engineering, model evaluation, cross-validation, hyperparameter optimization, ensemble methods, interpretability, deployment, imbalanced data, AutoML
- **Applications:** Production ML pipelines, model deployment, feature stores, automated ML workflows

---

## 📚 Notebooks

### [041_Feature_Engineering_Masterclass.ipynb](041_Feature_Engineering_Masterclass.ipynb)
**Comprehensive Feature Engineering Techniques**

Master the art and science of creating predictive features from raw data. Feature engineering is often the difference between mediocre and exceptional model performance.

**Topics Covered:**
- **Numerical Features:** Scaling, normalization, binning, transformations (log, sqrt, Box-Cox)
- **Categorical Features:** One-hot encoding, label encoding, target encoding, frequency encoding
- **Temporal Features:** Date/time decomposition (year, month, day, hour, cyclical encoding)
- **Interaction Features:** Polynomial features, ratio features, product features
- **Domain-Specific Features:** Statistical aggregates, rolling windows, lag features
- **Text Features:** TF-IDF, word embeddings, n-grams
- **Missing Value Handling:** Imputation strategies (mean, median, KNN, iterative)

**Real-World Applications:**
- **Parametric Test Features:** Voltage ratios (Vdd/Idd), power efficiency metrics, frequency deviations
- **Spatial Features:** Die position on wafer (center vs edge), radial distance, quadrant indicators
- **Test Sequence Features:** Test order effects, cumulative test time, retry patterns
- **Process Features:** Lot-level aggregates, wafer-level statistics, historical yield trends

**Mathematical Foundations:**
```
Scaling Transformations:
  StandardScaler: z = (x - μ) / σ
  MinMaxScaler: x' = (x - x_min) / (x_max - x_min)
  RobustScaler: x' = (x - median) / IQR

Power Transformations:
  Box-Cox: y = (x^λ - 1) / λ  (λ ≠ 0)
           y = ln(x)           (λ = 0)
  
Cyclical Encoding (for time/angles):
  sin(2π × hour/24)
  cos(2π × hour/24)
```

**Learning Outcomes:**
- Design feature engineering pipelines for any domain
- Apply domain knowledge to create predictive features
- Handle high-cardinality categorical variables
- Create time-based features (lag, rolling statistics)
- Validate feature importance using SHAP/permutation
- Build feature stores for production ML

---

### [042_Model_Evaluation_Metrics.ipynb](042_Model_Evaluation_Metrics.ipynb)
**Comprehensive Guide to ML Evaluation Metrics**

Understand when and how to use different evaluation metrics for classification, regression, and ranking tasks. Choosing the right metric is critical for business alignment.

**Topics Covered:**
- **Classification Metrics:** Accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, confusion matrix
- **Regression Metrics:** MSE, RMSE, MAE, MAPE, R², adjusted R²
- **Ranking Metrics:** NDCG, MAP, MRR (Mean Reciprocal Rank)
- **Multi-Class Metrics:** Macro/micro/weighted averaging, per-class analysis
- **Business Metrics:** Cost-sensitive learning, custom loss functions
- **Probabilistic Metrics:** Log loss, Brier score, calibration curves
- **Threshold Selection:** ROC curves, precision-recall tradeoffs, optimal cutoffs

**Real-World Applications:**
- **Yield Classification:** Precision-recall for imbalanced failure detection (1-5% defect rate)
- **Test Time Regression:** MAPE for capacity planning forecasts (need ±5% accuracy)
- **Defect Detection:** Cost-sensitive metrics (false negatives cost $100K, false positives $1K)
- **Device Ranking:** NDCG for sorting dies by failure probability

**Mathematical Foundations:**
```
Classification Metrics:
  Precision = TP / (TP + FP)
  Recall = TP / (TP + FN)
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  ROC-AUC = ∫ TPR(FPR) dFPR
  
Regression Metrics:
  MSE = (1/n) Σ(y_i - ŷ_i)²
  RMSE = √MSE
  MAE = (1/n) Σ|y_i - ŷ_i|
  MAPE = (100/n) Σ|y_i - ŷ_i| / |y_i|
  R² = 1 - SS_res / SS_tot
```

**Learning Outcomes:**
- Select appropriate metrics for business objectives
- Interpret confusion matrices for multi-class problems
- Calculate ROC-AUC and PR-AUC for imbalanced datasets
- Design custom metrics for domain-specific costs
- Validate metric reliability using confidence intervals
- Report metrics with statistical significance tests

---

### [043_Cross_Validation_Strategies.ipynb](043_Cross_Validation_Strategies.ipynb)
**Advanced Cross-Validation Techniques**

Master validation strategies that provide reliable performance estimates while avoiding data leakage and overfitting.

**Topics Covered:**
- **K-Fold Cross-Validation:** Standard, stratified, repeated K-fold
- **Time Series CV:** Rolling window, expanding window, walk-forward validation
- **Group-Based CV:** GroupKFold for clustered/hierarchical data
- **Leave-One-Out (LOO):** When to use, computational considerations
- **Nested CV:** Hyperparameter tuning + model selection without bias
- **Custom Splitters:** Domain-specific validation strategies
- **Validation Set Strategy:** Train/validation/test splits, hold-out sets

**Real-World Applications:**
- **Wafer-Based CV:** Group by wafer_id to avoid data leakage (dies from same wafer correlated)
- **Time-Aware CV:** Walk-forward validation for yield forecasting (respect temporal ordering)
- **Lot-Based CV:** Group by lot_id for realistic generalization estimates
- **Stratified CV:** Maintain bin distribution across folds (HBin, fail codes)

**Mathematical Foundations:**
```
K-Fold Cross-Validation:
  - Split data into K folds (typically K=5 or 10)
  - Train on K-1 folds, validate on 1 fold
  - Repeat K times, average performance
  
  CV_score = (1/K) Σ score_k
  CV_std = √[(1/K) Σ(score_k - CV_score)²]

Nested Cross-Validation:
  Outer loop: Model selection (K1 folds)
  Inner loop: Hyperparameter tuning (K2 folds)
  Prevents overfitting on validation set
```

**Learning Outcomes:**
- Implement stratified K-fold for imbalanced datasets
- Apply time series CV for temporal data
- Design group-based CV for hierarchical structures
- Detect overfitting using train/validation curves
- Estimate confidence intervals for CV scores
- Avoid data leakage in feature engineering pipelines

---

### [044_Hyperparameter_Tuning.ipynb](044_Hyperparameter_Tuning.ipynb)
**Systematic Hyperparameter Optimization**

Learn efficient methods for finding optimal hyperparameters: from grid search to Bayesian optimization and modern neural architecture search.

**Topics Covered:**
- **Grid Search:** Exhaustive search over parameter grids
- **Random Search:** Sample random combinations (often outperforms grid)
- **Bayesian Optimization:** Gaussian processes, acquisition functions (EI, UCB)
- **Hyperband/Successive Halving:** Early stopping for efficient search
- **Optuna:** Modern hyperparameter framework with pruning
- **Search Space Design:** Continuous vs discrete, log-scale distributions
- **Multi-Fidelity Optimization:** Using subsets/epochs for faster evaluation

**Real-World Applications:**
- **XGBoost Tuning:** Optimize max_depth, learning_rate, n_estimators for yield prediction
- **SVM Tuning:** C and gamma for defect classification (5-20% accuracy gain)
- **Neural Network Tuning:** Learning rate schedules, batch size, dropout rates
- **Ensemble Tuning:** Optimize voting weights for stacked models

**Mathematical Foundations:**
```
Bayesian Optimization:
  1. Build surrogate model: P(y|x) ~ GP(μ(x), k(x,x'))
  2. Acquisition function: α(x) = E[I(x)] or UCB(x)
     EI(x) = E[max(0, f(x) - f(x_best))]
     UCB(x) = μ(x) + κσ(x)
  3. Select next x = argmax α(x)
  4. Evaluate f(x), update surrogate

Random Search vs Grid Search:
  - Random: Sample N random configurations
  - Grid: Evaluate all combinations (exponential in dimensions)
  - Random often finds better solutions in same time budget
```

**Learning Outcomes:**
- Implement grid search and random search from scratch
- Apply Bayesian optimization using Optuna or scikit-optimize
- Design effective search spaces (log-scale for learning rates)
- Use early stopping to accelerate hyperparameter search
- Visualize hyperparameter sensitivity with partial dependence plots
- Balance exploration vs exploitation in search strategies

---

### [045_Ensemble_Methods.ipynb](045_Ensemble_Methods.ipynb)
**Advanced Ensemble Techniques**

Master methods for combining multiple models to improve prediction accuracy, robustness, and generalization.

**Topics Covered:**
- **Voting Ensembles:** Hard voting, soft voting (probability averaging)
- **Bagging:** Bootstrap aggregating, random subspace method
- **Boosting:** AdaBoost, Gradient Boosting (covered in depth in 018-021)
- **Stacking:** Meta-learner trained on base model predictions
- **Blending:** Holdout-based stacking variant
- **Diversity-Based Ensembles:** Negative correlation learning, mixture of experts
- **Weighted Ensembles:** Optimizing ensemble weights

**Real-World Applications:**
- **Yield Prediction Ensemble:** Stack XGBoost + Random Forest + SVM (3-5% accuracy gain)
- **Defect Detection:** Voting ensemble for high-confidence predictions (reduce false alarms 30%)
- **Test Time Prediction:** Blend ARIMA + XGBoost for robust forecasts
- **Multi-Task Ensembles:** Separate models per device type, combined predictions

**Mathematical Foundations:**
```
Voting Ensemble:
  Hard Voting: ŷ = mode(h₁(x), h₂(x), ..., hₙ(x))
  Soft Voting: ŷ = argmax Σ P_i(y|x)
  
Stacking (2-Level):
  Level 0: Train base models h₁, h₂, ..., hₙ on D_train
  Level 1: Train meta-learner g on (h₁(x), ..., hₙ(x), y) using D_val
  Prediction: ŷ = g(h₁(x), h₂(x), ..., hₙ(x))

Weighted Ensemble:
  ŷ = Σ w_i h_i(x)  subject to Σ w_i = 1, w_i ≥ 0
```

**Learning Outcomes:**
- Build voting and stacking ensembles using scikit-learn
- Implement custom ensemble strategies (weighted averaging)
- Optimize ensemble diversity (correlation analysis)
- Apply out-of-fold predictions for stacking
- Diagnose when ensembles help vs hurt performance
- Deploy ensembles in production (serialize multiple models)

---

### [046_Model_Interpretation_Explainability.ipynb](046_Model_Interpretation_Explainability.ipynb)
**Making Black-Box Models Interpretable**

Learn techniques for explaining model predictions: essential for regulatory compliance, debugging, and building trust with stakeholders.

**Topics Covered:**
- **Feature Importance:** Gini importance, permutation importance, drop-column importance
- **SHAP (SHapley Additive exPlanations):** TreeSHAP, KernelSHAP, DeepSHAP
- **LIME (Local Interpretable Model-agnostic Explanations):** Local linear approximations
- **Partial Dependence Plots (PDP):** Marginal effect of features
- **Individual Conditional Expectation (ICE):** Per-sample feature effects
- **Counterfactual Explanations:** "What would need to change for different prediction?"
- **Global vs Local Explanations:** Model-level vs prediction-level interpretability

**Real-World Applications:**
- **Yield Prediction Explanations:** "Why did model predict 85% yield?" (SHAP waterfall plot)
- **Defect Root Cause:** "Which test parameters caused failure prediction?" (top 5 features)
- **Regulatory Compliance:** Explain decisions for FDA/ISO audits
- **Model Debugging:** Identify when model relies on spurious correlations

**Mathematical Foundations:**
```
SHAP Values (Shapley Values from Game Theory):
  φᵢ = Σ [|S|!(|F|-|S|-1)!] / |F|! × [f(S∪{i}) - f(S)]
       S⊆F\{i}
  
  where F = all features, S = subset of features
  φᵢ represents contribution of feature i to prediction

Permutation Importance:
  1. Baseline: score = evaluate_model(X_val, y_val)
  2. For each feature i:
     a. X_perm = X_val.copy()
     b. Shuffle X_perm[:, i]
     c. score_perm = evaluate_model(X_perm, y_val)
     d. importance_i = score - score_perm
```

**Learning Outcomes:**
- Compute SHAP values for tree-based and deep learning models
- Create LIME explanations for individual predictions
- Generate partial dependence plots for feature analysis
- Compare global feature importance across multiple methods
- Identify spurious correlations using explanation techniques
- Communicate model decisions to non-technical stakeholders

---

### [047_ML_Pipelines_Automation.ipynb](047_ML_Pipelines_Automation.ipynb)
**Building Production-Ready ML Pipelines**

Learn to automate the entire ML workflow: data preprocessing, feature engineering, model training, and prediction—all in a reproducible, maintainable pipeline.

**Topics Covered:**
- **Scikit-Learn Pipeline:** Sequential transformers + estimator
- **ColumnTransformer:** Apply different transformations to different columns
- **Custom Transformers:** Writing sklearn-compatible transformers
- **FeatureUnion:** Parallel feature extraction pipelines
- **Pipeline Serialization:** Saving/loading with joblib, pickle
- **Pipeline Hyperparameter Tuning:** GridSearchCV with nested pipelines
- **DAG-Based Pipelines:** Directed acyclic graphs for complex workflows

**Real-World Applications:**
- **Automated Feature Engineering:** Pipeline applies scaling, encoding, imputation automatically
- **Reproducible Training:** Same pipeline guarantees same preprocessing in train/test/production
- **A/B Testing:** Deploy competing pipelines, compare in production
- **Batch Prediction:** Process 1M+ devices per day with serialized pipeline

**Mathematical Foundations:**
```
Pipeline Structure:
  Pipeline([
    ('preprocessor', ColumnTransformer([...])),
    ('feature_eng', FeatureUnion([...])),
    ('model', XGBClassifier(...))
  ])

Execution Flow:
  1. X_train_transformed = preprocessor.fit_transform(X_train)
  2. X_train_features = feature_eng.fit_transform(X_train_transformed)
  3. model.fit(X_train_features, y_train)
  
  4. X_test_transformed = preprocessor.transform(X_test)
  5. X_test_features = feature_eng.transform(X_test_transformed)
  6. y_pred = model.predict(X_test_features)
```

**Learning Outcomes:**
- Build end-to-end pipelines with preprocessing + modeling
- Write custom sklearn-compatible transformers
- Apply different transformations to numerical vs categorical features
- Tune pipeline hyperparameters using GridSearchCV
- Serialize pipelines for production deployment
- Debug pipeline errors (transform vs fit_transform)

---

### [048_Model_Deployment.ipynb](048_Model_Deployment.ipynb)
**Deploying ML Models to Production**

Master the techniques and best practices for deploying ML models as production services: REST APIs, batch prediction, edge deployment.

**Topics Covered:**
- **Model Serialization:** Pickle, joblib, ONNX, TensorFlow SavedModel
- **REST API Deployment:** Flask, FastAPI, model serving
- **Batch Prediction:** Scheduled jobs, Spark integration
- **Model Monitoring:** Drift detection, performance tracking
- **A/B Testing:** Shadow mode, canary deployments, multi-armed bandits
- **Versioning:** Model registry, experiment tracking (MLflow)
- **Containerization:** Docker for reproducible environments

**Real-World Applications:**
- **Real-Time Yield Prediction API:** 10-50ms latency for fab decision support
- **Batch Test Time Forecasting:** Nightly Spark jobs processing 1M devices
- **Edge Deployment:** On-tester models for inline parametric screening
- **Multi-Model Serving:** Route devices to device-specific models

**Mathematical Foundations:**
```
REST API Request Flow:
  1. Client sends JSON: {"features": [1.2, 3.4, ...]}
  2. Server deserializes: X = parse_json(request)
  3. Preprocess: X_transformed = pipeline.transform(X)
  4. Predict: y_pred = model.predict(X_transformed)
  5. Respond: {"prediction": y_pred, "confidence": 0.87}

Drift Detection (KS Test):
  - Compare distribution of input features: P_train vs P_production
  - KS statistic: D = sup|F_train(x) - F_prod(x)|
  - Alert if D > threshold (e.g., 0.1) or p-value < 0.05
```

**Learning Outcomes:**
- Deploy scikit-learn models as Flask REST APIs
- Build FastAPI endpoints with Pydantic validation
- Implement batch prediction pipelines with Spark
- Monitor model performance in production (accuracy, latency)
- Detect data drift using statistical tests
- Version models using MLflow registry

---

### [049_Imbalanced_Data_Handling.ipynb](049_Imbalanced_Data_Handling.ipynb)
**Techniques for Imbalanced Classification**

Master strategies for handling class imbalance (1-99 ratio), common in defect detection, fraud detection, and rare event prediction.

**Topics Covered:**
- **Resampling:** SMOTE, ADASYN, random oversampling, random undersampling
- **Class Weighting:** Inverse frequency, balanced class weights
- **Ensemble Methods:** BalancedRandomForest, EasyEnsemble, RUSBoost
- **Cost-Sensitive Learning:** Custom loss functions, asymmetric costs
- **Evaluation Metrics:** Precision-recall curves, F1-score, Matthews correlation coefficient
- **Threshold Tuning:** Adjusting decision thresholds for optimal F1/precision/recall
- **Anomaly Detection Approaches:** One-class SVM, Isolation Forest (reframe as outlier detection)

**Real-World Applications:**
- **Defect Detection:** 1-5% failure rate in semiconductor testing
- **Equipment Failure:** Predict rare breakdowns (0.1-1% event rate)
- **Outlier Yield:** Identify wafers with anomalous parametric distributions
- **Test Escape Detection:** Flag devices likely to fail in field (99.9% pass rate)

**Mathematical Foundations:**
```
SMOTE (Synthetic Minority Oversampling):
  1. For each minority sample x_i:
  2. Find k nearest minority neighbors
  3. Select random neighbor x_j
  4. Generate synthetic sample:
     x_new = x_i + λ(x_j - x_i), λ ~ Uniform(0,1)

Class Weights:
  w_i = n_samples / (n_classes × n_samples_i)
  
  Modifies loss: L = Σ w_i × loss(y_i, ŷ_i)

F-Beta Score:
  F_β = (1 + β²) × (Precision × Recall) / (β²×Precision + Recall)
  β=0.5: Favor precision
  β=1: Balanced F1
  β=2: Favor recall
```

**Learning Outcomes:**
- Apply SMOTE and ADASYN for oversampling minority class
- Implement class weighting in scikit-learn models
- Tune decision thresholds for precision-recall tradeoffs
- Evaluate models using PR-AUC instead of ROC-AUC
- Compare resampling vs cost-sensitive approaches
- Design custom evaluation metrics for asymmetric costs

---

### [050_AutoML_Frameworks.ipynb](050_AutoML_Frameworks.ipynb)
**Automated Machine Learning (AutoML)**

Explore frameworks that automate model selection, hyperparameter tuning, and feature engineering: reducing ML expertise requirements.

**Topics Covered:**
- **Auto-Sklearn:** Automated sklearn pipeline construction
- **TPOT:** Genetic programming for pipeline optimization
- **H2O AutoML:** Distributed AutoML with leaderboard
- **AutoGluon:** Tabular, text, image AutoML (Amazon)
- **PyCaret:** Low-code ML library with compare_models()
- **FLAML:** Fast and lightweight AutoML (Microsoft)
- **Neural Architecture Search (NAS):** Automated deep learning design

**Real-World Applications:**
- **Rapid Prototyping:** Benchmark 20+ models in 30 minutes
- **Citizen Data Scientists:** Enable non-ML experts to build models
- **Production Baselines:** AutoML as baseline, then manual optimization
- **Feature Engineering Automation:** Discover interaction features automatically

**Mathematical Foundations:**
```
AutoML Search Space:
  - Algorithms: {LogReg, SVM, RF, XGBoost, Neural Nets}
  - Preprocessing: {StandardScaler, MinMaxScaler, RobustScaler}
  - Feature Selection: {SelectKBest, PCA, None}
  - Hyperparameters: {learning_rate ∈ [0.001, 0.1], ...}

Optimization Strategies:
  - Bayesian Optimization: Model P(score | config)
  - Genetic Algorithms: Evolve pipelines via crossover/mutation
  - Multi-Armed Bandits: Allocate compute to promising configs
```

**Learning Outcomes:**
- Run auto-sklearn for automated model selection
- Use TPOT for genetic programming-based pipeline optimization
- Deploy H2O AutoML for distributed training
- Compare AutoML results to manual modeling
- Understand when AutoML works well vs when manual tuning needed
- Extract best pipelines from AutoML for production deployment

---

### [103_Advanced_Feature_Engineering.ipynb](103_Advanced_Feature_Engineering.ipynb)
**Advanced Feature Engineering Techniques**

Deep dive into sophisticated feature engineering: embeddings, automated feature learning, feature stores, and domain-specific transformations.

**Topics Covered:**
- **Entity Embeddings:** Neural network embeddings for categorical variables
- **Feature Crosses:** Automated interaction feature generation
- **Target Encoding Variants:** CatBoost encoder, leave-one-out encoding
- **Feature Stores:** Feast, Tecton for production feature management
- **Time-Based Aggregations:** Rolling statistics, exponentially weighted means
- **Graph Features:** Node centrality, PageRank for relational data
- **Feature Selection at Scale:** Recursive feature elimination, Boruta

**Real-World Applications:**
- **Device Type Embeddings:** Learn 32-dimensional representations of 100+ device types
- **Wafer Map Features:** Spatial patterns (radial gradients, quadrant statistics)
- **Test Flow Features:** Sequence embeddings for multi-stage test programs
- **Feature Store:** Centralized repository for parametric test features across 10+ teams

**Learning Outcomes:**
- Implement entity embeddings using Keras/PyTorch
- Build feature stores for production ML systems
- Apply Boruta for feature selection at scale
- Design domain-specific features for complex data
- Benchmark feature engineering impact (lift curves)

---

### [104_Model_Interpretability_Explainability.ipynb](104_Model_Interpretability_Explainability.ipynb)
**Advanced Model Interpretability**

Advanced techniques for model interpretation: causal inference, counterfactual explanations, adversarial testing, and fairness analysis.

**Topics Covered:**
- **Causal Inference:** Intervention analysis, causal DAGs, do-calculus
- **Counterfactual Explanations:** DiCE, WhatIf tool, actionable recourse
- **Adversarial Robustness:** Testing model stability under perturbations
- **Fairness Analysis:** Demographic parity, equalized odds, bias detection
- **Concept Activation Vectors (CAV):** Interpret neural network concepts
- **Attention Mechanisms:** Visualizing attention weights in transformers
- **Model Cards:** Documenting model behavior, limitations, ethical considerations

**Real-World Applications:**
- **Root Cause Analysis:** "If Vdd increased by 10mV, would device pass?" (counterfactual)
- **Fairness Audits:** Ensure models don't discriminate by fab site, product line
- **Adversarial Testing:** Verify model robustness to sensor noise, measurement error
- **Regulatory Documentation:** Generate model cards for FDA/ISO submissions

**Learning Outcomes:**
- Generate counterfactual explanations using DiCE
- Test model fairness across demographic groups
- Create adversarial examples to probe model weaknesses
- Visualize attention weights for interpretability
- Document models using standardized model cards

---

### [105_AutoML_NAS.ipynb](105_AutoML_NAS.ipynb)
**Neural Architecture Search (NAS)**

Master automated neural network design: searching for optimal architectures, reducing manual design effort.

**Topics Covered:**
- **NAS Algorithms:** Reinforcement learning, evolutionary algorithms, gradient-based NAS
- **Search Spaces:** Cell-based, hierarchical, macro architectures
- **One-Shot NAS:** Weight sharing, supernet training (ENAS, DARTS)
- **Efficient NAS:** Neural architecture transfer, zero-cost proxies
- **Hardware-Aware NAS:** Optimize for latency, memory, energy consumption
- **AutoKeras:** Automated deep learning with Keras
- **NNI (Neural Network Intelligence):** Microsoft's AutoML toolkit

**Real-World Applications:**
- **Custom Architectures:** Design neural networks optimized for wafer map analysis
- **Edge Deployment:** NAS for resource-constrained tester hardware
- **Multi-Task NAS:** Find architectures that excel at both classification and regression
- **Transfer Learning:** Adapt architectures discovered on ImageNet to semiconductor data

**Mathematical Foundations:**
```
DARTS (Differentiable Architecture Search):
  - Relax discrete architecture search to continuous optimization
  - Architecture parameters: α = {α₁, α₂, ..., αₙ}
  - Operation weights: exp(α_i) / Σ exp(α_j)
  - Jointly optimize weights w and architecture α

NAS Objective:
  Find architecture A* that minimizes:
    A* = argmin L_val(w*(A), A)
  where w*(A) = argmin L_train(w, A)
```

**Learning Outcomes:**
- Implement simple NAS using random search
- Apply DARTS for differentiable architecture search
- Use AutoKeras for automated deep learning
- Benchmark NAS-discovered architectures vs manual designs
- Understand computational costs of NAS methods
- Deploy NAS architectures to production

---

## 🔗 Prerequisites

**Required Knowledge:**
- **02_Regression_Models:** Linear models, regularization
- **03_Tree_Based_Models:** Random Forest, XGBoost basics
- **Python Libraries:** scikit-learn, pandas, NumPy
- **Basic Statistics:** Hypothesis testing, confidence intervals

**Recommended Background:**
- **Software Engineering:** Version control (Git), testing, documentation
- **Production Systems:** APIs, databases, containerization (Docker)
- **Cloud Platforms:** AWS/GCP/Azure basics (for deployment)

---

## 🎯 Key Learning Outcomes

By completing this section, you will:

✅ **Master Feature Engineering:** Create predictive features from raw data (10-30% accuracy gains)  
✅ **Select Right Metrics:** Align evaluation metrics with business objectives  
✅ **Design Validation Strategies:** Avoid data leakage and overfitting  
✅ **Optimize Hyperparameters:** Apply Bayesian optimization for efficient tuning  
✅ **Build Ensembles:** Stack models for 3-5% accuracy improvements  
✅ **Explain Predictions:** Use SHAP/LIME for regulatory compliance and debugging  
✅ **Automate ML Pipelines:** Build reproducible, production-ready pipelines  
✅ **Deploy Models:** Serve models via REST APIs and batch systems  
✅ **Handle Imbalance:** Apply SMOTE, class weighting for rare event prediction  
✅ **Leverage AutoML:** Benchmark 20+ models automatically in minutes  
✅ **Advanced Techniques:** Entity embeddings, causal inference, NAS  

---

## 📈 Technique Comparison Table

| Technique | Best For | Complexity | Impact | Automation |
|-----------|----------|------------|--------|------------|
| **Feature Engineering** | Domain knowledge | Medium | High (10-30%) | Low |
| **Hyperparameter Tuning** | Model optimization | Low | Medium (3-10%) | High |
| **Ensemble Methods** | Accuracy boost | Medium | Medium (3-5%) | Medium |
| **Model Interpretation** | Trust, debugging | High | N/A | Low |
| **ML Pipelines** | Reproducibility | Medium | High (saves time) | High |
| **AutoML** | Rapid prototyping | Low | Medium (baseline) | Very High |

---

## 🏭 Post-Silicon Validation Applications

### 1. **Yield Prediction System**
- **Feature Engineering:** Voltage ratios, spatial features, test sequence patterns
- **Evaluation:** MAPE <3% for 13-week ahead forecasts
- **Deployment:** REST API for fab dashboard integration
- **Value:** $5-10M annual savings via proactive capacity adjustments

### 2. **Defect Detection Pipeline**
- **Imbalanced Data:** 1-5% defect rate, apply SMOTE + class weighting
- **Interpretability:** SHAP explanations for root cause analysis
- **Monitoring:** Drift detection for process shifts
- **Value:** Reduce test escapes 30-50% (avoid field failures)

### 3. **Test Time Optimization**
- **Ensemble:** Stack XGBoost + Random Forest for robust predictions
- **Cross-Validation:** Wafer-based GroupKFold to avoid leakage
- **Hyperparameter Tuning:** Bayesian optimization for XGBoost
- **Value:** Reduce test costs 10-15% ($2-3M annually)

### 4. **AutoML Baseline System**
- **AutoML:** Run H2O AutoML nightly on new product data
- **Human-in-Loop:** ML engineers refine top-3 models manually
- **Deployment:** Automated retraining pipeline (Airflow)
- **Value:** 10x faster model development for new products

---

## 🔄 Next Steps

After mastering ML Engineering:

1. **07_Deep_Learning:** Apply pipelines to neural networks (TensorFlow, PyTorch)
2. **09_Data_Engineering:** Build ETL pipelines for feature stores, data lakes
3. **10_MLOps:** Deploy models with CI/CD, monitoring, retraining automation
4. **13_MLOps_Production_ML:** Advanced production patterns (A/B testing, canary deployments)

**Advanced Topics:**
- **Feature Stores:** Feast, Tecton for centralized feature management
- **Model Registries:** MLflow, DVC for version control
- **Real-Time Serving:** TensorFlow Serving, TorchServe, ONNX Runtime

---

## 📝 Project Ideas

### Post-Silicon Validation Projects

1. **Parametric Test Feature Engineering**
   - Extract 50+ features from voltage, current, frequency measurements
   - Apply target encoding for categorical variables (device_type, lot_id)
   - Use Boruta for feature selection (retain top 20-30 features)
   - Target: 5-10% accuracy gain over raw features

2. **Yield Prediction Ensemble**
   - Stack XGBoost + LightGBM + Random Forest
   - Apply wafer-based GroupKFold for validation
   - SHAP analysis for top 10 feature importance
   - Deploy as REST API (50ms latency)

3. **Imbalanced Defect Detector**
   - Handle 1-5% defect rate using SMOTE + class weighting
   - Tune threshold for 95% recall, maximize precision
   - Implement drift detection (alert if feature distributions shift)
   - Cost-benefit analysis: $100K per false negative, $1K per false positive

4. **AutoML Production Baseline**
   - Run auto-sklearn on new product data weekly
   - Compare AutoML vs manual models (accuracy, training time)
   - Extract best pipeline for production deployment
   - Document findings in model card

### General AI/ML Projects

5. **Credit Card Fraud Detection**
   - Feature engineering: transaction velocity, amount patterns, merchant categories
   - Imbalanced classification: 0.1% fraud rate
   - Real-time scoring: <10ms latency requirement
   - Explainability: LIME for fraud investigation

6. **Customer Churn Prediction Pipeline**
   - End-to-end scikit-learn pipeline: preprocessing + feature engineering + XGBoost
   - Nested CV for unbiased hyperparameter tuning
   - Deploy Flask API with model versioning (MLflow)
   - A/B test against baseline model

7. **Sales Forecasting Ensemble**
   - Stack ARIMA + XGBoost + Random Forest
   - Feature engineering: lag features, rolling statistics, holiday indicators
   - Custom evaluation metric: weighted MAPE (penalize over-forecasting)
   - Batch prediction pipeline (Airflow)

8. **Image Classification with AutoML**
   - Apply AutoKeras/AutoGluon to custom image dataset
   - Compare NAS-discovered architecture vs ResNet50 transfer learning
   - Interpretability: Grad-CAM for visual explanations
   - Edge deployment: ONNX conversion for mobile

---

**Total Notebooks in Section:** 13  
**Estimated Completion Time:** 26-36 hours  
**Difficulty Level:** Intermediate to Advanced  
**Prerequisites:** ML fundamentals, Python proficiency, scikit-learn experience

*Last Updated: December 2025*
