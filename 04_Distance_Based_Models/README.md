# 04 - Distance-Based Models

**Purpose:** Master instance-based and margin-based learning algorithms

**Why Distance-Based Models?** These algorithms work fundamentally differently from tree-based models. KNN uses similarity between instances (lazy learning), SVM finds optimal decision boundaries (margin maximization), and Naive Bayes leverages probability theory. Each has unique strengths: KNN for irregular decision boundaries, SVM for high-dimensional data, Naive Bayes for real-time inference.

---

## 📚 Notebooks (023-025)

### **Instance-Based Learning**

#### **023_K_Nearest_Neighbors.ipynb** ✅ (35 cells)
**Lazy learning - classify based on similarity to training examples**

**Topics Covered:**
- KNN algorithm for classification and regression
- Distance metrics (Euclidean, Manhattan, Minkowski, Cosine, Hamming)
- K selection (cross-validation, elbow method)
- Weighted KNN (distance weighting)
- Curse of dimensionality and feature scaling
- Efficient search: KD-tree, Ball tree, Locality-Sensitive Hashing
- Handling imbalanced data

**Mathematical Foundation:**
- Euclidean distance: d(x,y) = √(Σ(xᵢ-yᵢ)²)
- Manhattan distance: d(x,y) = Σ|xᵢ-yᵢ|
- Minkowski distance: d(x,y) = (Σ|xᵢ-yᵢ|ᵖ)^(1/p)
- Weighted voting: w(x) = 1/d(x,xᵢ)²

**Key Concepts:**
- **Lazy Learning**: No training phase - stores all data and computes at prediction time
- **K Selection**: Small K → complex boundaries (overfitting), Large K → smooth boundaries (underfitting)
- **Distance Weighting**: Closer neighbors have more influence
- **Curse of Dimensionality**: In high dimensions, all points become equidistant

**Real-World Applications:**
- **Post-Silicon**: Similar die failure pattern matching on wafer maps
- **Post-Silicon**: Device parameter clustering (find similar test results)
- **Post-Silicon**: Nearest-neighbor outlier detection (parametric anomalies)
- **General**: Recommendation systems (find similar users/items)
- **General**: Image recognition (find similar images)
- **General**: Anomaly detection (distance to normal behavior)

**Advantages:**
- Simple and intuitive
- No training phase (online learning)
- Naturally handles multi-class problems
- Non-parametric (makes no assumptions about data distribution)

**Disadvantages:**
- Slow prediction (O(N) for each prediction)
- Memory intensive (stores entire dataset)
- Sensitive to irrelevant features
- Requires feature scaling

**Learning Outcomes:**
- Implement KNN from scratch
- Choose appropriate distance metrics
- Optimize K using cross-validation
- Apply efficient search structures (KD-tree)
- Handle curse of dimensionality with feature selection
- Scale features properly before KNN

**Performance Tips:**
- Use KD-tree or Ball tree for large datasets (100× speedup)
- Apply PCA for dimensionality reduction
- Use feature selection to remove irrelevant features
- Scale features to [0,1] range

---

### **Margin-Based Learning**

#### **024_Support_Vector_Machines.ipynb** ✅ (27 cells)
**Maximum margin classifiers - find optimal decision boundaries**

**Topics Covered:**
- SVM for classification (SVC) and regression (SVR)
- Linear SVM and maximum margin hyperplane
- Kernel trick (RBF, polynomial, sigmoid)
- Soft margin (C parameter) for non-separable data
- Hyperparameter tuning (C, gamma, kernel)
- Support vectors and decision function
- Multi-class strategies (one-vs-one, one-vs-rest)

**Mathematical Foundation:**
- Decision boundary: w·x + b = 0
- Margin: 2/||w||
- Optimization: min(1/2)||w||² + C·Σξᵢ
- Kernel functions:
  - Linear: K(x,y) = x·y
  - RBF: K(x,y) = exp(-γ||x-y||²)
  - Polynomial: K(x,y) = (γx·y + r)ᵈ

**Key Concepts:**
- **Maximum Margin**: Find hyperplane that maximizes distance to nearest points
- **Support Vectors**: Only points near boundary matter (sparsity)
- **Kernel Trick**: Map data to higher dimensions without explicit computation
- **C Parameter**: Controls trade-off between margin width and misclassifications
- **Gamma Parameter**: Controls influence of single training example (RBF kernel)

**Real-World Applications:**
- **Post-Silicon**: Binary pass/fail classification (high accuracy needed)
- **Post-Silicon**: Wafer bin classification (speed vs power categories)
- **Post-Silicon**: High-dimensional parametric space classification (100+ features)
- **General**: Text classification (high-dimensional sparse data)
- **General**: Image classification (kernel methods)
- **General**: Bioinformatics (gene expression classification)

**Hyperparameter Tuning Guide:**
```python
# Small C → Wide margin (underfitting risk)
# Large C → Narrow margin (overfitting risk)
C = [0.1, 1, 10, 100]

# Small gamma → Far influence (smooth boundary)
# Large gamma → Close influence (complex boundary)
gamma = [0.001, 0.01, 0.1, 1]

# Kernel selection
kernel = ['linear', 'rbf', 'poly']
```

**When to Use SVM:**
- ✅ High-dimensional data (text, genomics)
- ✅ Clear margin of separation exists
- ✅ More features than samples
- ✅ Need kernel methods for non-linearity
- ❌ Very large datasets (slow training O(N²))
- ❌ Noisy data with overlapping classes
- ❌ Need probability estimates (use Platt scaling)

**Learning Outcomes:**
- Understand maximum margin concept
- Apply kernel trick for non-linear problems
- Tune C and gamma systematically
- Choose appropriate kernel for problem
- Interpret support vectors
- Scale SVM to large datasets

**Performance Comparison:**
- **Linear SVM**: Fast, good for linearly separable data
- **RBF SVM**: Most popular, handles non-linearity well
- **Polynomial SVM**: Good for image data, needs careful tuning
- **Training Time**: O(N²) to O(N³) - slow for large N

---

### **Probabilistic Learning**

#### **025_Naive_Bayes.ipynb** ✅ (22 cells)
**Fast probabilistic classifiers based on Bayes' theorem**

**Topics Covered:**
- Bayes' theorem and conditional probability
- Naive independence assumption
- Gaussian Naive Bayes (continuous features)
- Multinomial Naive Bayes (discrete counts, text)
- Bernoulli Naive Bayes (binary features)
- Laplace smoothing (additive smoothing)
- Probability calibration
- Real-time inference (< 1ms)

**Mathematical Foundation:**
- Bayes' theorem: P(y|X) = P(X|y)·P(y) / P(X)
- Naive assumption: P(X|y) = P(x₁|y)·P(x₂|y)·...·P(xₙ|y)
- Gaussian NB: P(xᵢ|y) = (1/√(2πσ²))·exp(-(xᵢ-μ)²/(2σ²))
- Multinomial NB: P(xᵢ|y) = (count(xᵢ,y) + α) / (count(y) + α·n_features)
- Laplace smoothing: Add α (typically 1) to avoid zero probabilities

**Key Concepts:**
- **Naive Independence**: Assume features are conditionally independent (rarely true, but works well)
- **Generative Model**: Models P(X|y) and P(y), then uses Bayes' theorem
- **Extremely Fast**: Training and prediction both very fast
- **Works with Small Data**: Effective even with limited training samples

**Naive Bayes Variants:**

**1. Gaussian NB:**
- For continuous features with normal distribution
- Use case: Sensor data, measurements, parametric test values

**2. Multinomial NB:**
- For discrete count features
- Use case: Text classification (word counts), document categorization

**3. Bernoulli NB:**
- For binary features (present/absent)
- Use case: Text classification (word presence), feature presence detection

**Real-World Applications:**
- **Post-Silicon**: Real-time test failure classification (< 10ms inference)
- **Post-Silicon**: Lot-level yield prediction (fast screening)
- **Post-Silicon**: Email-style log classification (Multinomial NB)
- **General**: Spam filtering (classic application, 99%+ accuracy)
- **General**: Sentiment analysis (text classification)
- **General**: Medical diagnosis (symptom → disease probability)
- **General**: Real-time fraud detection (millisecond latency)

**Advantages:**
- ⚡ Extremely fast training and prediction
- 📊 Works well with small datasets
- 📈 Handles high-dimensional data naturally
- 🔢 Provides probability estimates
- 💾 Low memory footprint
- 🚀 Easy to implement and interpret

**Disadvantages:**
- Naive independence assumption often violated
- Cannot learn feature interactions
- Sensitive to irrelevant features
- Probability estimates not always well-calibrated

**When to Use Naive Bayes:**
- ✅ Text classification (spam, sentiment, categorization)
- ✅ Real-time inference required (< 10ms)
- ✅ Limited training data
- ✅ High-dimensional data (text, sparse features)
- ✅ Baseline model (fast to try)
- ❌ Features are highly correlated
- ❌ Need to model feature interactions
- ❌ Need perfectly calibrated probabilities

**Learning Outcomes:**
- Understand Bayes' theorem intuitively
- Implement Naive Bayes from scratch
- Choose appropriate variant (Gaussian/Multinomial/Bernoulli)
- Apply Laplace smoothing correctly
- Build real-time classifiers
- Calibrate probability estimates

**Performance:**
- Training: O(N·D) - very fast
- Prediction: O(C·D) - very fast (C = # classes, D = # features)
- Typically 70-85% accuracy (baseline, but very fast)

---

## 🎯 Learning Path

**Recommended Order:**
1. **023 - K-Nearest Neighbors** ⭐ **START HERE** - Simplest algorithm, builds intuition
2. **024 - Support Vector Machines** - More sophisticated boundary finding
3. **025 - Naive Bayes** - Probabilistic approach, fastest inference

**Time Estimate:** 1-2 weeks (intensive) | 2-3 weeks (moderate pace)

---

## 📊 Section Statistics

| Metric | Value |
|--------|-------|
| **Total Notebooks** | 3 |
| **Complete Notebooks** | 3 (100% ✅) |
| **Total Cells** | 84+ |
| **Real-World Projects** | 18+ |
| **Algorithms Covered** | 3 |

---

## 🔑 Key Learning Outcomes

After completing this section, you will:

✅ **Algorithm Understanding**
- Master instance-based learning (KNN)
- Understand margin-based learning (SVM)
- Apply probabilistic learning (Naive Bayes)

✅ **Distance Metrics**
- Choose appropriate metrics for data type
- Handle curse of dimensionality
- Scale features properly

✅ **Kernel Methods**
- Apply kernel trick for non-linearity
- Tune kernel hyperparameters
- Understand computational tradeoffs

✅ **Production Skills**
- Build real-time classifiers (< 10ms)
- Handle high-dimensional data
- Choose algorithm based on requirements

✅ **Domain Applications**
- Classify semiconductor test failures
- Match similar die failure patterns
- Real-time yield prediction

---

## 🔗 Prerequisites

**Before starting this section, complete:**
- ✅ **[02_Regression_Models](../02_Regression_Models/)** - Classification basics
- ✅ Distance metrics (Euclidean, Manhattan, Cosine)
- ✅ Probability theory (Bayes' theorem, conditional probability)
- ✅ Linear algebra (dot products, norms)

---

## ➡️ Next Steps

After mastering distance-based models, continue to:

1. **[05_Clustering](../05_Clustering/)** - Unsupervised learning, finding patterns without labels
2. **[06_ML_Engineering](../06_ML_Engineering/)** - Feature engineering, model evaluation
3. **[03_Tree_Based_Models](../03_Tree_Based_Models/)** - Compare with ensemble methods

---

## 💡 Study Tips

1. **Visualize Decision Boundaries** - Plot 2D examples to understand how each algorithm works
2. **Feature Scaling is Critical** - Always scale features for KNN and SVM (not needed for Naive Bayes)
3. **Start Simple** - Try linear models before kernel methods
4. **Use Grid Search** - Systematically tune C, gamma, K
5. **Benchmark Speed** - Compare training and inference times
6. **Understand Tradeoffs** - KNN (slow prediction), SVM (slow training), NB (fast everything)

---

## 🛠️ Tools & Libraries

**Core Libraries:**
- Scikit-learn (all three algorithms)
- Faiss (fast similarity search for KNN)
- LIBSVM (underlying SVM implementation)

**Distance Metrics:**
- Scipy.spatial.distance (comprehensive metrics)
- Scikit-learn.metrics.pairwise (pairwise distances)

**Visualization:**
- Matplotlib (decision boundaries)
- Seaborn (confusion matrices)

---

## 📈 Algorithm Comparison

| Algorithm | Training Speed | Prediction Speed | Accuracy | Interpretability | Best For |
|-----------|----------------|------------------|----------|------------------|----------|
| **KNN** | ⚡ Instant (lazy) | 🐢 Slow O(N) | ⭐⭐⭐ | ⭐⭐⭐⭐ | Irregular boundaries, small data |
| **SVM** | 🐢 Slow O(N²-N³) | ⚡ Fast O(SV) | ⭐⭐⭐⭐⭐ | ⭐⭐ | High-dim, clear margins |
| **Naive Bayes** | ⚡⚡ Very fast | ⚡⚡ Very fast | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Text, real-time, baseline |

**When to Use Each:**
- **KNN**: Small datasets (<10K), irregular decision boundaries, need explainability
- **SVM**: High-dimensional data, clear separation, accuracy critical
- **Naive Bayes**: Text classification, real-time inference, limited training data

---

## 📈 Progress Tracking

Mark notebooks as complete as you master them:

- [ ] 023_K_Nearest_Neighbors ⭐ **START HERE**
- [ ] 024_Support_Vector_Machines
- [ ] 025_Naive_Bayes

---

## 🌟 Why This Section Matters

**Industry Relevance:**
- KNN: Used in recommendation systems, anomaly detection
- SVM: Gold standard for text classification, bioinformatics
- Naive Bayes: Deployed in spam filters (Gmail uses it), real-time systems

**Career Impact:**
- Core ML algorithms in technical interviews
- Understand when NOT to use deep learning
- Foundation for understanding kernel methods
- Critical for building fast inference systems

**Unique Strengths:**
- Fundamentally different from tree-based methods
- Teaches distance metrics and similarity
- Introduces kernel methods (foundation for deep learning)
- Shows probabilistic reasoning approach

---

**Last Updated:** December 2025  
**Status:** All 3 notebooks complete (100% ✅)  
**Maintainer:** [@rajendarmuddasani](https://github.com/rajendarmuddasani)
