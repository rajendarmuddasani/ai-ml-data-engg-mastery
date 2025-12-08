# 📊 Master Notebook Tracker

**Purpose:** Track all notebooks with details for quick reference and progress monitoring.

**Status Legend:** ✅ Complete | 🚧 In Progress | 📝 Planned

---

## Existing Notebooks

| # | Notebook | Category | Models/Techniques | Skills | General Projects | Post-Silicon Projects | Status |
|---|----------|----------|-------------------|--------|------------------|----------------------|--------|
| 001 | [001_DSA_Python_Mastery](./001_DSA_Python_Mastery.ipynb) | Foundations | Data Structures, Algorithms | Python, DSA, Problem Solving | Algorithm optimization, System design | Test flow optimization, Data structure for STDF parsing | ✅ |
| 002 | [002_Python_Advanced_Concepts](./002_Python_Advanced_Concepts.ipynb) | Foundations | Decorators, Generators, Context Managers | Advanced Python, Metaprogramming | Custom frameworks, Pipeline builders | ATE control scripts, Test sequencer design | ✅ |
| 010 | [010_Linear_Regression](./010_Linear_Regression.ipynb) | ML - Regression | Linear Regression, OLS | Scikit-learn, NumPy, Statistics | Sales forecasting, Real estate pricing, CLV prediction, Energy forecasting | Device power prediction, Test time estimation, Parametric yield, V-F characterization | ✅ |
| 079 | [079_RAG_Fundamentals](./079_RAG_Fundamentals.ipynb) | Modern AI | RAG, Vector DBs, Embeddings | LangChain, ChromaDB, OpenAI | Document Q&A, Knowledge base, Support bot, Research assistant | Test documentation search, Failure analysis KB, Design spec Q&A, Debug assistant | ✅ |

---

## Planned Notebooks - Regression Models (011-015)

| # | Notebook | Category | Models/Techniques | Skills | General Projects | Post-Silicon Projects | Status |
|---|----------|----------|-------------------|--------|------------------|----------------------|--------|
| 011 | [011_Polynomial_Regression](./011_Polynomial_Regression.ipynb) | ML - Regression | Polynomial Regression, Degree Selection, Bias-Variance Tradeoff | Feature engineering, Overfitting prevention, Cross-validation | Marketing response curves, Growth forecasting (S-curves), Price elasticity modeling, Environmental trend analysis | Temperature-performance characterization, V-F curve modeling, Device aging prediction, Non-linear test correlation | ✅ |
| 012 | [012_Ridge_Lasso_ElasticNet](./012_Ridge_Lasso_ElasticNet.ipynb) | ML - Regression | Ridge, Lasso, ElasticNet, L1/L2 Regularization | Regularization, Feature selection, Multicollinearity handling, CV tuning | Genomic biomarker discovery, Text classification (large vocab), Financial risk modeling, Image compression | High-dimensional STDF reduction (1000→50 params), Correlated test elimination, Sparse yield modeling, Robust power prediction | ✅ |
| 013 | [013_Logistic_Regression](./013_Logistic_Regression.ipynb) | ML - Classification | Logistic Regression, Sigmoid, Softmax, Multi-class | Classification, Probability estimation, ROC/AUC, Confusion matrix | Customer churn prediction, Fraud detection, Email spam filter, Medical diagnosis | Parametric test pass/fail, Multi-bin speed classification, Wafer defect detection, Test flow optimization | ✅ |
| 014 | [014_Support_Vector_Regression](./014_Support_Vector_Regression.ipynb) | ML - Regression | SVR (Linear/RBF/Poly), Epsilon-insensitive loss, Kernel trick | Robust regression, Kernel methods, Hyperparameter tuning (C/epsilon/gamma), Outlier handling | Financial market forecasting, Sensor anomaly-resistant modeling, Medical cost prediction, Energy demand forecasting | Outlier-robust yield prediction, Noise-resistant test time prediction, Robust V-F characterization, Extreme condition performance | ✅ |
| 015 | [015_Quantile_Regression](./015_Quantile_Regression.ipynb) | ML - Regression | Quantile Regression, Check loss, τ-quantiles (1st-99th) | Conditional quantiles, Prediction intervals, VaR/CVaR, Heteroscedastic modeling | Financial VaR/CVaR, Extreme weather forecasting, Healthcare cost tails, Supply chain SLA | Process capability bounds prediction, Worst-case yield prediction, Guard-band optimization, Specification limit setting | ✅ |

---

## Planned Notebooks - Tree-Based Models (016-022)

| # | Notebook | Category | Models/Techniques | Skills | General Projects | Post-Silicon Projects | Status |
|---|----------|----------|-------------------|--------|------------------|----------------------|--------|
| 016 | [016_Decision_Trees](./016_Decision_Trees.ipynb) | ML - Tree Models | Decision Trees, CART (Classification And Regression Trees), RSS/Gini splitting, Pruning, Feature importance | Tree algorithms, Recursive partitioning, Interpretability, Overfitting prevention | Medical diagnosis, Customer churn prediction, Loan approval, Predictive maintenance | Automatic speed binning, Test flow optimization, Non-linear power prediction, Wafer failure mode classification | ✅ |
| 017 | [017_Random_Forest](./017_Random_Forest.ipynb) | ML - Ensemble | Random Forest, Bagging, OOB error, Bootstrap aggregating, Feature importance (MDI/permutation) | Ensemble methods, Variance reduction, Parallel training, Hyperparameter tuning (n_estimators, max_features, max_depth) | Credit scoring, Medical diagnosis, Customer churn prediction, Kaggle baseline model | Multi-failure mode detection, Test importance ranking (25% time reduction), Wafer spatial pattern classification, Robust yield with missing data | ✅ |
| 018 | [018_Gradient_Boosting_GBM](./018_Gradient_Boosting.ipynb) | ML - Ensemble | Gradient Boosting, Sequential learning, Forward stagewise additive modeling, Pseudo-residuals, Early stopping | Boosting, Learning rate tuning, Validation curves, Bias reduction, Hyperparameter sensitivity | Customer lifetime value prediction, Credit risk scoring, Demand forecasting, Fraud detection | Predictive test time optimizer (15% reduction), Adaptive binning engine, Multi-site test correlation, Yield drift detection | ✅ |
| 019 | [019_XGBoost](./019_XGBoost.ipynb) | ML - Ensemble | XGBoost, L1/L2 regularization, 2nd-order gradients (Hessian), DMatrix, Native API vs sklearn, Early stopping | XGBoost API, Systematic hyperparameter tuning, Sparsity handling, GPU acceleration, Production deployment | Credit scoring (FICO replacement), E-commerce CTR prediction, Healthcare readmission risk, Kaggle competition framework | Real-time adaptive test system (30-40% time reduction), Multi-site test optimization, Parametric outlier detection at scale, Wafer map pattern clustering | ✅ |
| 020 | [020_LightGBM](./020_LightGBM.ipynb) | ML - Ensemble | LightGBM, Histogram-based, Leaf-wise, GOSS, EFB, Native categorical handling | Histogram binning (255 bins), Gradient-based sampling, Feature bundling, 10-100x speedup, lgb.Dataset API | E-commerce CTR prediction (<5ms), Financial fraud detection (10M transactions), Healthcare readmission, Kaggle framework | Streaming pipeline (500K devices/hour), Memory-efficient 10M analysis, Multi-site correlation, GPU wafer pattern detection | ✅ |
| 021 | [021_CatBoost](./021_CatBoost.ipynb) | ML - Ensemble | CatBoost, Ordered boosting, Ordered target statistics, Oblivious trees | High-cardinality categoricals (1000s-millions), Ordered encoding prevents leakage, Symmetric trees, Robust defaults | E-commerce user/product IDs, Credit risk merchant IDs, Healthcare ICD codes (70K), Marketing campaign attribution | Equipment-specific yield (500+ testers), Lot-level quality prediction, Multi-site categorical correlation, Supplier quality scoring | ✅ |
| 022 | [022_Voting_Stacking_Ensembles](./022_Voting_Stacking_Ensembles.ipynb) | ML - Meta-Ensemble | Voting (hard/soft/weighted), Stacking, Out-of-fold predictions, Meta-learning | Model combination, VotingClassifier, StackingClassifier, Ensemble diversity, Production robustness | Healthcare multi-model readmission ($8-15M), Financial fraud detection ensemble ($20-50M), E-commerce CTR meta-ensemble ($10-30M/quarter), Kaggle competition framework | Multi-model test flow optimizer (20-30% time reduction), Robust failure mode classification (60% debug reduction), Production drift-resistant yield predictor (99.95% uptime), Kaggle-style semiconductor yield competition | ✅ |

---

## Planned Notebooks - Distance & Instance-Based (023-025)

| # | Notebook | Category | Models/Techniques | Skills | General Projects | Post-Silicon Projects | Status |
|---|----------|----------|-------------------|--------|------------------|----------------------|--------|
| 023 | [023_K_Nearest_Neighbors](./023_K_Nearest_Neighbors.ipynb) | ML - Instance-Based | KNN Classification/Regression, Euclidean/Manhattan/Cosine distances, K selection via CV, Weighted voting, Curse of dimensionality | Instance-based learning, Distance metrics, Feature scaling (critical), KD-tree/Ball tree, FAISS for scale | Content-based recommendation ($5-20M), Medical case-based diagnosis (20-30% error reduction), Anomaly detection ($10-50M fraud blocked), Image similarity search (1B+ images) | Similar failure detection for root cause ($500K-2M), Reference die matching (15-25% test time reduction, $2-5M savings), Wafer map spatial clustering ($10-30M early detection), Multi-site test correlation ($5-15M cost reduction) | ✅ |
| 024 | [024_Support_Vector_Machines](./024_Support_Vector_Machines.ipynb) | ML - Margin-Based | SVM (SVC/SVR), Linear/RBF/Polynomial/Sigmoid kernels, Kernel trick, C and gamma tuning, Soft margin, Hinge loss | Maximum margin classification, Kernel methods, High-dimensional data (p>>n), Hyperparameter tuning (GridSearchCV), Margin-based confidence | Medical diagnosis support (92%+ accuracy, $20-50M), Financial fraud detection (<10ms, $50-200M), Text sentiment classification (10K vocab, $10-30M), MNIST digit recognition (98%+ accuracy, $5-15M) | Multi-class defect root cause classifier (90%+ accuracy, $500K-2M), Wafer spatial defect detection (95%+ systematic vs random, $2-5M), Margin-based reliability predictor (±15% 10-year failures, $10-30M), Multi-site test correlation engine (30% final test reduction, $5-15M) | ✅ |
| 024 | [024_Support_Vector_Machines](./024_Support_Vector_Machines.ipynb) | ML - SVM | SVM, Kernel methods, Margin optimization | Classification boundaries, Kernel tricks, Multi-class SVM | Binary classification, Text classification, Image classification, Imbalanced data | Pass/fail classification, Defect detection, Binary test outcomes, Margin-based binning | 📝 |
| 025 | [025_Naive_Bayes](./025_Naive_Bayes.ipynb) | ML - Probabilistic | Naive Bayes, Gaussian/Multinomial/Bernoulli | Probability theory, Conditional independence, Text classification | Spam detection, Document classification, Sentiment analysis, Real-time classification | Fast test classification, Probabilistic binning, Quick failure detection, Real-time sorting | 📝 |

---

## Planned Notebooks - Clustering (026-030)

| # | Notebook | Category | Models/Techniques | Skills | General Projects | Post-Silicon Projects | Status |
|---|----------|----------|-------------------|--------|------------------|----------------------|--------|
| 026 | [026_K_Means_Clustering](./026_K_Means_Clustering.ipynb) | ML - Clustering | K-Means, Elbow method, Centroid-based | Unsupervised learning, Cluster evaluation, K selection | Customer segmentation, Market analysis, Data compression, Pattern discovery | Wafer pattern clustering, Test group segmentation, Bin clustering, Process lot grouping | 📝 |
| 027 | [027_Hierarchical_Clustering](./027_Hierarchical_Clustering.ipynb) | ML - Clustering | Hierarchical, Dendrogram, Linkage methods | Tree-based clustering, Distance metrics, Cut-off selection | Taxonomy creation, Product categorization, Organizational structure, Gene clustering | Test hierarchy, Failure taxonomy, Die similarity tree, Parameter grouping | 📝 |
| 028 | [028_DBSCAN](./028_DBSCAN.ipynb) | ML - Clustering | DBSCAN, Density-based, Noise detection | Density clustering, Outlier detection, Non-spherical clusters | Geospatial analysis, Anomaly detection, Network analysis, User behavior | Spatial wafer defects, Density-based binning, Cluster outlier detection, Process anomalies | 📝 |
| 029 | [029_Gaussian_Mixture_Models](./029_Gaussian_Mixture_Models.ipynb) | ML - Clustering | GMM, EM algorithm, Soft clustering | Probabilistic clustering, EM optimization, Model selection | Soft segmentation, Probability assignment, Mixed populations, Density estimation | Multi-process detection, Soft bin assignment, Yield distribution modeling, Mixed lot analysis | 📝 |
| 030 | [030_Dimensionality_Reduction](./030_Dimensionality_Reduction.ipynb) | ML - Dim Reduction | PCA, t-SNE, UMAP, LDA | Feature extraction, Visualization, Manifold learning | Data visualization, Feature compression, Noise reduction, Exploratory analysis | STDF parameter reduction, Test visualization, High-dim wafer analysis, Feature compression | 📝 |

---

## Usage Instructions

### For AI Agents:
1. **Before creating a notebook**: Check this table to see what's planned
2. **After creating a notebook**: Update the row with ✅ status
3. **Track progress**: Use this table to prioritize next notebooks
4. **Reference patterns**: Look at completed notebooks for structure examples

### Table Update Workflow:
1. Change status from 📝 → 🚧 when starting work
2. Add actual project titles as they're created
3. Update skills/models if scope changes during development
4. Change status to ✅ when notebook is complete and tested
5. Keep this table synchronized with MASTER_LEARNING_ROADMAP.md

### Column Definitions:
- **#**: Notebook number in sequence
- **Notebook**: Hyperlinked filename
- **Category**: High-level grouping (Foundations, ML, DL, etc.)
- **Models/Techniques**: Specific algorithms/methods covered
- **Skills**: Technical competencies learned
- **General Projects**: 4 non-semiconductor project ideas
- **Post-Silicon Projects**: 4 semiconductor/test-focused project ideas
- **Status**: ✅ Complete | 🚧 In Progress | 📝 Planned

---

**Last Updated:** 2025-12-07  
**Maintained By:** Automated notebook generation process
