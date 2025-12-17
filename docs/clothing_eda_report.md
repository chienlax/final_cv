# EDA Report: Clothing, Shoes and Jewelry

**Generated:** 2025-12-17T16:34:15.529452  
**Sampling Strategy:** Dense (K-Core k=5 + 12 months)

---

## 1. Data Overview

### Loading Statistics

| Metric | Interactions | Metadata |
|--------|-------------|----------|
| Total Records | 66,033,346 | 7,218,481 |
| Sampled Records | 852,544 | 61,078 |
| Memory (MB) | 532.12 | 141.09 |

### Interaction Statistics

| Metric | Value |
|--------|-------|
| Users | 65,116 |
| Items | 61,078 |
| Interactions | 852,544 |
| Avg Rating | 4.46 |
| Rating Std | 0.94 |
| Sparsity | 99.97856397% |

---

## 2. Rating Distribution

![Rating Distribution](figures/clothing/rating_distribution_clothing,_shoes_and_jewelry.png)

| Rating | Count | Percentage |
|--------|-------|------------|
| 1.0 | 22,205 | 2.6% |
| 2.0 | 24,584 | 2.9% |
| 3.0 | 66,431 | 7.8% |
| 4.0 | 161,835 | 19.0% |
| 5.0 | 577,489 | 67.7% |

---

## 3. User and Item Analysis

### Power-Law Distribution

![Interaction Frequency](figures/clothing/interaction_frequency_clothing,_shoes_and_jewelry.png)

**User Patterns:**
- Mean interactions/user: 13.09
- Median interactions/user: 7.0
- Cold-start users (<5 interactions): 0.0%
- Power-law exponent α: 2.50

**Item Patterns:**
- Mean interactions/item: 13.96
- Median interactions/item: 11.0
- Cold-start items (<5 interactions): 0.0%
- Power-law exponent α: 2.09

### Pareto Analysis (Interaction Concentration)

Top users account for a disproportionate share of interactions:

| User Tier | % of Total Interactions |
|-----------|------------------------|
| Top 1% | 11.7% |
| Top 5% | 32.2% |
| Top 10% | 45.2% |
| Top 20% | 59.1% |
| Top 50% | 79.4% |
| Top 100% | 100.0% |

---

## 4. Temporal Analysis

![Temporal Patterns](figures/clothing/temporal_patterns_clothing,_shoes_and_jewelry.png)

**Date Range:** 2022-09-18 to 2023-09-12  
**Duration:** 359 days

---

## 5. Text Analysis

![Text Length Distribution](figures/clothing/text_length_clothing,_shoes_and_jewelry.png)

| Metric | Value |
|--------|-------|
| Avg Review Length | 278 chars |
| Avg Title Length | 21 chars |
| Reviews with Text | 100.0% |

---

## 6. Multimodal Analysis


![Multimodal Coverage](figures/clothing/multimodal_coverage_clothing,_shoes_and_jewelry.png)

### Feature Coverage

| Feature | Coverage |
|---------|----------|
| Title | 100.0% |
| Description | 21.9% |
| Features | 99.5% |
| Images | 100.0% |
| **Complete (Text + Image)** | 100.0% |

### Image Statistics

| Metric | Value |
|--------|-------|
| Items with Images | 61,077 |
| Avg Images/Item | 6.16 |

---

## 7. Sparsity and K-Core Analysis

![Sparsity](figures/clothing/sparsity_clothing,_shoes_and_jewelry.png)

**Matrix Sparsity:** 99.97856397%  
**Density:** 0.02143603%

### K-Core Filtering Impact

| k | Users Retained | Items Retained | Interactions Retained |
|---|----------------|----------------|----------------------|
| 2 | 100.0% | 100.0% | 100.0% |
| 3 | 100.0% | 100.0% | 100.0% |
| 5 | 100.0% | 100.0% | 100.0% |
| 10 | 17.6% | 35.8% | 43.6% |
| 20 | 0.0% | 0.0% | 0.0% |

---

## 8. Category Distribution

![Categories](figures/clothing/category_distribution_clothing,_shoes_and_jewelry.png)

Top categories in the dataset:

| Category | Count |
|----------|-------|
| Clothing, Shoes & Jewelry | 61,030 |
| Shoe, Jewelry & Watch Accessories | 48 |

---

## 9. Key Insights and Recommendations

### Data Quality
1. **High Sparsity:** The dataset exhibits extreme sparsity typical of recommendation datasets
2. **Power-Law Distribution:** Both users and items follow power-law distributions (long-tail)
3. **Cold-Start Challenge:** Significant portion of users/items have few interactions

### Preprocessing Recommendations
1. **K-Core Filtering:** Use k=5 as baseline (balances data quality vs. coverage)
2. **Multimodal Features:** Leverage text/image to address cold-start problem
3. **Negative Sampling:** Use popularity-based hard negative sampling for BPR


---

## 10. Multimodal Recommendation Readiness (Academic Analysis)


### 10.1 Modality-Interaction Alignment (Liu et al., 2024)

![Modality Alignment](figures/clothing/modality_alignment_clothing,_shoes_and_jewelry.png)

Tests the **Homophily Hypothesis**: Do visually similar items share similar interaction patterns?

| Metric | Value |
|--------|-------|
| Pairs Analyzed | 1,000 |
| Pearson r | 0.0465 |
| p-value | 0.1421 |
| Spearman ρ | 0.0483 |

**Interpretation:** No significant correlation - visual features may not align with user preferences


### 10.2 Visual Manifold Structure (Xu et al., 2025)

![Visual Manifold](figures/clothing/visual_manifold_clothing,_shoes_and_jewelry.png)

Analyzes whether CLIP embeddings form meaningful clusters by category.

| Metric | Value |
|--------|-------|
| Items Projected | 5,000 |
| Projection Method | UMAP |
| Silhouette Score | -0.0646 |
| Unique Categories | 2 |

**Interpretation:** No meaningful visual clustering - visual features may not align with categories


### 10.3 BPR Negative Sampling Hardness (Xu et al., 2025)

![BPR Hardness](figures/clothing/bpr_hardness_clothing,_shoes_and_jewelry.png)

Evaluates whether random negative sampling produces informative training signal.

| Metric | Value |
|--------|-------|
| Users Analyzed | 100 |
| Pairs Analyzed | 1,000 |
| Mean Visual Distance | 0.3360 |
| Easy Negatives (>0.8) | 0.0% |
| Medium Negatives | 65.4% |
| Hard Negatives (<0.3) | 34.6% |

**Interpretation:** Good distribution of hard negatives - random sampling may suffice

**Recommendation:** Standard BPR with random negatives should work


---

## 11. LATTICE Feasibility Assessment

> [!CAUTION]
> ⛔ **STOP** - LATTICE feasibility checks failed. Revisit Feature Extraction.

### 11.1 Graph Connectivity (k-NN, k=5)

| Metric | Value | Status |
|--------|-------|--------|
| Connected Components | 1 | - |
| Giant Component Size | 5,000 | - |
| Giant Component Coverage | 100.0% | ✅ PASS |
| Threshold | >50.0% | - |

**Interpretation:** PASS: Giant component covers 100.0% of items (threshold: 50.0%). Graph is sufficiently connected for LATTICE.

### 11.2 Feature Collapse Detection (White Wall Test)

| Metric | Value | Status |
|--------|-------|--------|
| Pairs Sampled | 10,000 | - |
| Avg Cosine Similarity | 0.6672 | ⚠️ WARNING |
| Std Cosine Similarity | 0.0885 | - |
| High Similarity Pairs (>0.9) | 0.2% | - |
| Pass Threshold | <0.5 | - |

**Interpretation:** WARNING: Avg cosine similarity = 0.667 (pass: <0.5, collapse: >0.9). Features show moderate similarity. May work but suboptimal. Consider testing with alternative visual encoder.

### Summary

| Check | Value | Status |
|-------|-------|--------|
| Alignment (Pearson r) | 0.0465 | ✅ |
| Connectivity (Giant %) | 100.0% | ✅ |
| No Collapse (Avg Cosine) | 0.6672 | ❌ |

**Decision:** STOP

---

*Report generated by EDA Pipeline for Multimodal Recommendation System*
