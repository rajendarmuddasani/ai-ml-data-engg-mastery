# 03 - Tree-Based Models

**Purpose:** Master decision trees, random forests, and gradient boosting - the workhorses of production ML

**Why Tree-Based Models?** Tree models dominate real-world ML applications and Kaggle competitions. They handle non-linear relationships, mixed data types, missing values, and feature interactions automatically. No feature scaling required. Highly interpretable. Production-proven. Learn these to solve 70% of tabular data problems.

---

## 📚 Notebooks (016-022)

### **Foundation: Decision Trees**

#### **016_Decision_Trees.ipynb** ✅ (23 cells)
**The foundation of ensemble methods - understand splits and pruning**

**Topics Covered:**
- CART algorithm (Classification and Regression Trees)
- Splitting criteria: RSS (regression), Gini impurity, Entropy (classification)
- Tree pruning techniques (pre-pruning, post-pruning, cost-complexity)
- Feature importance via information gain
- Handling categorical variables
- Tree depth vs overfitting tradeoff

**Mathematical Foundation:**
- Gini Impurity: G = 1 - Σ(pᵢ²)
- Information Gain: IG = Entropy(parent) - Σ(weighted Entropy(children))
- Cost-Complexity: Cost(T) = RSS(T) + α|T|

**Real-World Applications:**
- **Post-Silicon**: Root cause analysis of test failures (interpretable decision rules)
- **Post-Silicon**: Device binning logic (speed/power grades)
- **Post-Silicon**: Parametric limit violation detection
- **General**: Credit approval (transparent decision rules for compliance)
- **General**: Medical diagnosis (interpretable to clinicians)
- **General**: Customer churn prediction with explanations

**Learning Outcomes:**
- Understand how trees partition feature space
- Implement CART algorithm from scratch
- Prune trees to prevent overfitting
- Interpret tree splits for business insights
- Apply to wafer test failure diagnosis

---

### **Ensemble Method: Bagging**

#### **017_Random_Forest.ipynb** ✅ (23 cells)
**Bootstrap aggregating for variance reduction**

**Topics Covered:**
- Bootstrap sampling and aggregation (bagging)
- Random feature subset selection
- Out-of-Bag (OOB) error estimation
- Feature importance (mean decrease impurity, permutation)
- Parallel training across trees
- Variance-bias tradeoff in ensembles

**Key Concepts:**
- **Bootstrap**: Sample N observations with replacement → create diverse trees
- **Random Features**: Select √p features per split → decorrelate trees
- **OOB Error**: ~37% observations not in each bootstrap → free validation set
- **Variance Reduction**: Ensemble variance ≈ (1/B) × single tree variance

**Real-World Applications:**
- **Post-Silicon**: Multi-failure mode classification (open/short/parametric)
- **Post-Silicon**: Yield prediction with uncertainty (OOB confidence intervals)
- **Post-Silicon**: Spatial wafer map pattern recognition
- **General**: Fraud detection (high accuracy, low false positives)
- **General**: Recommendation systems (feature interactions)
- **General**: Sensor data classification (noise robust)

**Learning Outcomes:**
- Build random forests from scratch
- Understand why randomness improves ensembles
- Calculate feature importance reliably
- Use OOB error for model selection
- Apply to semiconductor test data classification

**Performance:** Typically 2-5% better accuracy than single trees

---

### **Ensemble Method: Boosting**

#### **018_Gradient_Boosting.ipynb** ✅ (26 cells)
**Sequential learning to correct errors**

**Topics Covered:**
- Gradient boosting algorithm (GBM)
- Pseudo-residuals and gradient descent
- Learning rate (shrinkage) and number of trees
- Tree depth (weak learners) optimization
- Early stopping and regularization
- Subsample and column sampling

**Mathematical Foundation:**
- Loss function: L(y, F(x))
- Gradient: gᵢ = -∂L/∂F(xᵢ)
- Update: F_m(x) = F_{m-1}(x) + ν·h_m(x)
- Shrinkage parameter: ν (0.01-0.3)

**Real-World Applications:**
- **Post-Silicon**: Test time prediction (minimize ATE downtime)
- **Post-Silicon**: Parametric yield modeling (complex interactions)
- **Post-Silicon**: Adaptive test flow optimization
- **General**: Sales forecasting (seasonality + trends)
- **General**: Risk scoring (credit, insurance)
- **General**: Click-through rate prediction

**Learning Outcomes:**
- Understand boosting vs bagging conceptually
- Implement gradient boosting from scratch
- Tune learning rate vs number of trees
- Prevent overfitting with early stopping
- Apply to time-series test data

**Performance:** Often 5-15% better than Random Forest

---

### **Production-Grade Boosting**

#### **019_XGBoost.ipynb** ✅ (26 cells)
**Extreme Gradient Boosting - Kaggle winner's toolkit**

**Topics Covered:**
- Second-order gradient optimization
- L1/L2 regularization (prevent overfitting)
- Sparsity-aware algorithm (missing values)
- Weighted quantile sketch (approximate splits)
- GPU acceleration (100× speedup)
- Built-in cross-validation

**Advanced Features:**
- Tree pruning (max_depth, min_child_weight, gamma)
- Sampling (subsample, colsample_bytree)
- Learning rate scheduling
- Custom loss functions
- Early stopping with validation sets

**Real-World Applications:**
- **Post-Silicon**: High-dimensional STDF parameter screening (1000+ features)
- **Post-Silicon**: Real-time ATE yield prediction (latency <100ms)
- **Post-Silicon**: Multi-site correlation analysis
- **General**: CTR prediction (billions of records)
- **General**: Financial fraud detection (class imbalance)
- **General**: Energy consumption forecasting

**Hyperparameter Tuning:**
```python
params = {
    'max_depth': 6,          # Tree depth
    'learning_rate': 0.1,    # Shrinkage
    'subsample': 0.8,        # Row sampling
    'colsample_bytree': 0.8, # Column sampling
    'reg_alpha': 0,          # L1 regularization
    'reg_lambda': 1          # L2 regularization
}
```

**Learning Outcomes:**
- Master XGBoost API and parameters
- Tune models systematically (grid search, Bayesian optimization)
- Handle class imbalance with scale_pos_weight
- Deploy XGBoost models to production
- Benchmark against baseline models

**Performance:** Often wins Kaggle competitions (state-of-art accuracy)

---

#### **020_LightGBM.ipynb** ✅ (28 cells)
**Microsoft's ultra-fast gradient boosting (10-100× faster than XGBoost)**

**Topics Covered:**
- Histogram-based algorithm (binning)
- Leaf-wise (best-first) tree growth
- Gradient-based One-Side Sampling (GOSS)
- Exclusive Feature Bundling (EFB)
- Categorical feature support (no encoding needed)
- Distributed training

**Key Innovations:**
- **Histogram**: Bin continuous features → reduce memory + 10× speedup
- **Leaf-wise**: Grow deepest leaf first → better accuracy with fewer trees
- **GOSS**: Keep large gradients, sample small gradients → 20× speedup
- **EFB**: Bundle mutually exclusive features → reduce dimensionality

**Real-World Applications:**
- **Post-Silicon**: Large-scale wafer map analysis (millions of dies)
- **Post-Silicon**: Real-time production line monitoring
- **Post-Silicon**: Multi-fab benchmark analysis
- **General**: High-frequency trading (latency critical)
- **General**: Ad targeting (billions of users)
- **General**: IoT sensor analytics (streaming data)

**Performance Comparison:**
- **Training Speed**: 10-100× faster than XGBoost
- **Memory Usage**: 3-5× lower than XGBoost
- **Accuracy**: Comparable or slightly better

**Learning Outcomes:**
- Understand histogram-based boosting
- Use LightGBM for large datasets (>10M rows)
- Handle categorical features natively
- Optimize for speed vs accuracy
- Deploy to high-throughput systems

---

#### **021_CatBoost.ipynb** ✅ (25 cells)
**Yandex's boosting for categorical features (no preprocessing needed)**

**Topics Covered:**
- Ordered boosting (prevent target leakage)
- Ordered target statistics (categorical encoding)
- Oblivious trees (symmetric splits)
- High-cardinality categorical handling
- Robust to overfitting
- Built-in GPU support

**Key Innovations:**
- **Ordered Boosting**: Use different permutations → prevent overfitting
- **Target Statistics**: Encode categories using target mean (no leakage)
- **Oblivious Trees**: Same split criteria at each level → faster inference

**Real-World Applications:**
- **Post-Silicon**: Device ID categorization (high cardinality)
- **Post-Silicon**: Tester equipment grouping (categorical features)
- **Post-Silicon**: Lot/Wafer/Die hierarchy modeling
- **General**: E-commerce (product categories, user IDs)
- **General**: Natural language (text categorization)
- **General**: Web analytics (URL paths, user agents)

**When to Use CatBoost:**
- Many categorical features (>5)
- High-cardinality categoricals (>100 unique values)
- Prefer minimal preprocessing
- Need out-of-the-box performance

**Learning Outcomes:**
- Handle categorical features without encoding
- Understand ordered boosting vs standard boosting
- Use CatBoost for text and categorical data
- Compare CatBoost vs XGBoost vs LightGBM
- Deploy to production with minimal preprocessing

---

### **Advanced Ensembles**

#### **022_Voting_Stacking_Ensembles.ipynb** ✅ (25 cells)
**Combine multiple models for production robustness**

**Topics Covered:**
- Voting ensembles (hard voting, soft voting)
- Stacking (meta-learning)
- Blending (holdout set predictions)
- Model diversity importance
- Production deployment strategies
- A/B testing ensembles

**Ensemble Strategies:**
- **Voting**: Average predictions from multiple models
- **Stacking**: Train meta-model on base model predictions
- **Blending**: Similar to stacking but simpler (single holdout set)

**Real-World Applications:**
- **Post-Silicon**: Critical yield prediction (99.95% uptime required)
- **Post-Silicon**: Multi-stage test flow optimization
- **Post-Silicon**: Redundant failure detection systems
- **General**: Financial risk models (regulatory compliance)
- **General**: Medical diagnosis (multiple expert systems)
- **General**: Autonomous vehicles (safety-critical)

**Stacking Example:**
```python
# Level 0: Base models
models = [RandomForest(), XGBoost(), LightGBM()]

# Level 1: Meta-model
meta_model = LogisticRegression()

# Stacking with cross-validation
stacker = StackingClassifier(estimators=models, 
                              final_estimator=meta_model,
                              cv=5)
```

**Learning Outcomes:**
- Build voting and stacking ensembles
- Understand when stacking helps (model diversity)
- Deploy ensembles to production
- Monitor ensemble performance over time
- Debug ensemble failures

---

## 🎯 Learning Path

**Recommended Order:**
1. **016 - Decision Trees** ⭐ **START HERE** - Foundation for all ensembles
2. **017 - Random Forest** - Learn bagging and parallelization
3. **018 - Gradient Boosting** - Understand sequential learning
4. **019 - XGBoost** - Master the most popular algorithm
5. **020 - LightGBM** - Learn speed optimization techniques
6. **021 - CatBoost** - Handle categorical features
7. **022 - Voting & Stacking** - Combine models for production

**Time Estimate:** 2-3 weeks (intensive) | 4-6 weeks (moderate pace)

---

## 📊 Section Statistics

| Metric | Value |
|--------|-------|
| **Total Notebooks** | 7 |
| **Complete Notebooks** | 7 (100% ✅) |
| **Total Cells** | 176+ |
| **Real-World Projects** | 40+ |
| **Algorithms Covered** | 7 |

---

## 🔑 Key Learning Outcomes

After completing this section, you will:

✅ **Tree Fundamentals**
- Understand recursive partitioning
- Implement CART from scratch
- Prune trees to prevent overfitting
- Interpret tree splits

✅ **Ensemble Mastery**
- Master bagging (Random Forest)
- Master boosting (GBM, XGBoost, LightGBM, CatBoost)
- Understand bias-variance tradeoff
- Build stacking ensembles

✅ **Production Skills**
- Choose right algorithm for problem
- Tune hyperparameters systematically
- Handle large datasets efficiently
- Deploy models to production

✅ **Domain Applications**
- Predict semiconductor yield (90%+ accuracy)
- Classify test failures by root cause
- Optimize ATE test time
- Handle STDF parametric data

---

## 🔗 Prerequisites

**Before starting this section, complete:**
- ✅ **[02_Regression_Models](../02_Regression_Models/)** - Supervised learning basics
- ✅ Understanding of overfitting and cross-validation
- ✅ Basic statistics (mean, variance, distributions)

---

## ➡️ Next Steps

After mastering tree-based models, continue to:

1. **[04_Distance_Based_Models](../04_Distance_Based_Models/)** - KNN, SVM for different problem types
2. **[05_Clustering](../05_Clustering/)** - Unsupervised learning with trees (Isolation Forest)
3. **[06_ML_Engineering](../06_ML_Engineering/)** - Feature engineering, hyperparameter tuning at scale

---

## 💡 Study Tips

1. **Implement Trees First** - Understand 016 before ensemble methods
2. **Visualize Trees** - Plot small trees to understand splits
3. **Benchmark Everything** - Always compare XGBoost vs LightGBM vs CatBoost on your data
4. **Monitor Training** - Use validation curves to detect overfitting early
5. **GPU Acceleration** - Use GPU for XGBoost/LightGBM on large datasets (100× speedup)
6. **Tune Systematically** - Use Optuna or Hyperopt for hyperparameter search

---

## 🛠️ Tools & Libraries

**Core Libraries:**
- Scikit-learn (Decision Trees, Random Forest)
- XGBoost (most popular)
- LightGBM (fastest)
- CatBoost (best for categoricals)

**Hyperparameter Tuning:**
- Optuna (Bayesian optimization)
- Hyperopt (TPE algorithm)
- Scikit-Optimize (GP-based)

**Visualization:**
- Graphviz (tree visualization)
- SHAP (feature importance)
- Plotly (interactive plots)

---

## 📈 Algorithm Comparison

| Algorithm | Speed | Accuracy | Categoricals | Best For |
|-----------|-------|----------|--------------|----------|
| **Decision Tree** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ✅ | Interpretability |
| **Random Forest** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | Baseline |
| **Gradient Boosting** | ⭐⭐ | ⭐⭐⭐⭐ | ⚠️ | Accuracy |
| **XGBoost** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚠️ | Kaggle |
| **LightGBM** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | Large datasets |
| **CatBoost** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅✅ | High-card cats |

---

## 📈 Progress Tracking

Mark notebooks as complete as you master them:

- [ ] 016_Decision_Trees ⭐ **START HERE**
- [ ] 017_Random_Forest
- [ ] 018_Gradient_Boosting
- [ ] 019_XGBoost ⭐ **MOST USED**
- [ ] 020_LightGBM
- [ ] 021_CatBoost
- [ ] 022_Voting_Stacking_Ensembles

---

## 🌟 Why This Section Matters

**Industry Relevance:**
- XGBoost/LightGBM used in 70%+ of production ML systems
- Kaggle competition winners use tree ensembles 90% of time
- Interpretable (important for regulated industries like semiconductors)
- Handles messy real-world data (missing values, mixed types)

**Career Impact:**
- Most requested skill in ML job postings
- Essential for data science interviews
- Direct impact on business metrics (yield, revenue, cost)
- Foundation for understanding AutoML systems

---

**Last Updated:** December 2025  
**Status:** All 7 notebooks complete (100% ✅)  
**Maintainer:** [@rajendarmuddasani](https://github.com/rajendarmuddasani)
