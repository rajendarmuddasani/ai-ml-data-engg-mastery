# 05 - Clustering & Unsupervised Learning

**Purpose:** Master unsupervised learning - pattern discovery, dimensionality reduction, anomaly detection, and recommendations

**Why Unsupervised Learning?** Most real-world data is unlabeled. Clustering finds hidden patterns, dimensionality reduction speeds up ML pipelines, anomaly detection catches defects, and recommender systems drive engagement. These techniques are essential for exploratory data analysis and production systems.

---

## 📚 Notebooks (026-031, 035-040) - 12 Total

### **🎯 Clustering Algorithms (026-029)**

#### **026_K-Means_Clustering.ipynb** ✅ (27 cells)
**Centroid-based clustering - fast and scalable**

**Topics:** K-Means, Lloyd's algorithm, K-Means++, elbow method, silhouette score, mini-batch K-Means  
**Applications:** Wafer map defect patterns, device binning, customer segmentation  
**Speed:** Very fast - O(n·k·i) where i is typically small  

---

#### **027_Hierarchical_Clustering.ipynb** ✅ (26 cells)
**Tree-based clustering - creates hierarchy of clusters**

**Topics:** Agglomerative/divisive, linkage methods (single/complete/average/Ward), dendrograms  
**Applications:** Test program structure analysis, failure mode taxonomy  
**Advantage:** No need to pre-specify K, provides cluster hierarchy  

---

#### **028_DBSCAN.ipynb** ✅ (20 cells)
**Density-based clustering - handles arbitrary shapes**

**Topics:** Core/border/noise points, eps-neighborhood, MinPts, arbitrary cluster shapes  
**Applications:** Wafer map spatial clustering (irregular defect patterns), outlier detection  
**Strength:** Finds clusters of any shape, identifies outliers automatically  

---

#### **029_Gaussian_Mixture_Models.ipynb** ✅ (19 cells)
**Probabilistic soft clustering - uncertainty quantification**

**Topics:** EM algorithm, soft assignments, BIC/AIC model selection, covariance types  
**Applications:** Probabilistic device classification, overlapping failure modes  
**Output:** Cluster membership probabilities (not hard assignments)  

---

### **📉 Dimensionality Reduction (030-031)**

#### **030_Dimensionality_Reduction.ipynb** ✅ (20 cells)
**Reduce high-dimensional data for visualization and speedup**

**Topics:** PCA, t-SNE, UMAP, explained variance, manifold learning  
**Applications:** 1000D STDF → 50D (20× ML speedup), 2D/3D visualization  
**Impact:** Reduce computation time, remove noise, visualize patterns  

---

#### **031_Feature_Selection.ipynb** ✅ (15 cells)
**Select most important features systematically**

**Topics:** Filter methods, wrapper methods, embedded methods, mutual information  
**Applications:** Reduce 500 test parameters to 20 critical ones  
**Benefit:** Faster training, better generalization, interpretability  

---

### **🔍 Anomaly Detection (035-038)**

#### **035_Autoencoders.ipynb** ✅ (20 cells)
**Neural network-based dimensionality reduction**

**Topics:** Encoder-decoder architecture, reconstruction loss, latent space  
**Applications:** Nonlinear dimensionality reduction, feature learning  

---

#### **036_Isolation_Forest.ipynb** ✅ (16 cells)
**Tree-based anomaly detection - fast and effective**

**Topics:** Isolation trees, anomaly score, contamination parameter  
**Applications:** Parametric test outliers, yield excursions  
**Speed:** Very fast - O(n log n), handles high dimensions  

---

#### **037_One-Class_SVM.ipynb** ✅ (17 cells)
**SVM for novelty detection**

**Topics:** One-class SVM, nu parameter, kernel methods for anomalies  
**Applications:** Normal behavior modeling, rare failure detection  

---

#### **038_AutoEncoders_Anomalies.ipynb** ✅ (20 cells)
**Deep learning for anomaly detection**

**Topics:** Autoencoder reconstruction error, threshold selection  
**Applications:** Complex pattern anomalies in wafer maps  
**Strength:** Captures nonlinear patterns trees miss  

---

### **🎲 Pattern Mining & Recommendations (039-040)**

#### **039_Association_Rules_Apriori.ipynb** ✅ (19 cells)
**Market basket analysis - find frequent patterns**

**Topics:** Apriori algorithm, support, confidence, lift, frequent itemsets  
**Applications:** Test failure co-occurrence, correlated parameter failures  
**Output:** "If test A fails, test B fails 80% of time"  

---

#### **040_Recommender_Systems.ipynb** ✅ (25 cells)
**Build recommendation engines**

**Topics:** Collaborative filtering, content-based, matrix factorization, SVD  
**Applications:** Test flow optimization, similar device recommendations  
**Techniques:** User-based CF, item-based CF, hybrid systems  

---

## 🎯 Learning Path

**Recommended Order:**
1. **026 K-Means** ⭐ START - Simplest clustering
2. **027 Hierarchical** - Understand dendrograms
3. **028 DBSCAN** - Handle arbitrary shapes
4. **029 GMM** - Probabilistic approach
5. **030 Dimensionality Reduction** - PCA, t-SNE, UMAP
6. **031 Feature Selection** - Reduce feature space
7. **036 Isolation Forest** - Fast anomaly detection
8. **037-038 Advanced Anomaly** - SVM and autoencoders
9. **039 Association Rules** - Pattern mining
10. **040 Recommender Systems** - Applied recommendations

**Time:** 3-4 weeks (intensive) | 6-8 weeks (moderate)

---

## 📊 Statistics

| Metric | Value |
|--------|-------|
| **Notebooks** | 12 |
| **Complete** | 12 (100% ✅) |
| **Cells** | 244+ |
| **Projects** | 60+ |

---

## 🔑 Key Outcomes

✅ **Clustering Mastery** - K-Means, Hierarchical, DBSCAN, GMM  
✅ **Dimensionality Reduction** - PCA, t-SNE, UMAP (1000D→50D)  
✅ **Anomaly Detection** - Isolation Forest, One-Class SVM, Autoencoders  
✅ **Pattern Mining** - Association rules, recommendations  
✅ **Production Skills** - Deploy unsupervised systems  

---

## 🔗 Prerequisites

- ✅ **[03_Tree_Based_Models](../03_Tree_Based_Models/)** - Feature importance  
- ✅ Linear algebra (eigenvalues, SVD)  
- ✅ Distance metrics  

---

## ➡️ Next Steps

1. **[06_Time_Series](../06_Time_Series/)** - Temporal patterns  
2. **[06_ML_Engineering](../06_ML_Engineering/)** - Feature engineering  
3. **[07_Deep_Learning](../07_Deep_Learning/)** - Neural networks  

---

**Last Updated:** December 2025  
**Status:** 12/12 notebooks complete (100% ✅)
