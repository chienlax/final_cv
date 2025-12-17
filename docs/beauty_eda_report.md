# EDA Report: Beauty and Personal Care

**Generated:** 2025-12-17T19:08:17.407319  
**Sampling Strategy:** Dense (K-Core k=5 + 36 months)

---

## 1. Data Overview

### Loading Statistics

| Metric | Interactions | Metadata |
|--------|-------------|----------|
| Total Records | 23,911,390 | 1,028,914 |
| Sampled Records | 1,778,010 | 86,436 |
| Memory (MB) | 1084.68 | 208.44 |

### Interaction Statistics

| Metric | Value |
|--------|-------|
| Users | 178,621 |
| Items | 86,436 |
| Interactions | 1,778,010 |
| Avg Rating | 4.26 |
| Rating Std | 1.22 |
| Sparsity | 99.98848386% |

---

## 2. Rating Distribution

![Rating Distribution](figures/beauty/rating_distribution_beauty_and_personal_care.png)

| Rating | Count | Percentage |
|--------|-------|------------|
| 1.0 | 128,382 | 7.2% |
| 2.0 | 79,346 | 4.5% |
| 3.0 | 147,686 | 8.3% |
| 4.0 | 262,524 | 14.8% |
| 5.0 | 1,160,072 | 65.2% |

---

## 3. User and Item Analysis

### Power-Law Distribution

![Interaction Frequency](figures/beauty/interaction_frequency_beauty_and_personal_care.png)

**User Patterns:**
- Mean interactions/user: 9.95
- Median interactions/user: 6.0
- Cold-start users (<5 interactions): 0.0%
- Power-law exponent α: 3.00

**Item Patterns:**
- Mean interactions/item: 20.57
- Median interactions/item: 11.0
- Cold-start items (<5 interactions): 0.0%
- Power-law exponent α: 1.95

### Pareto Analysis (Interaction Concentration)

Top users account for a disproportionate share of interactions:

| User Tier | % of Total Interactions |
|-----------|------------------------|
| Top 1% | 18.3% |
| Top 5% | 31.4% |
| Top 10% | 39.3% |
| Top 20% | 50.5% |
| Top 50% | 73.2% |
| Top 100% | 100.0% |

---

## 4. Temporal Analysis

![Temporal Patterns](figures/beauty/temporal_patterns_beauty_and_personal_care.png)

**Date Range:** 2020-09-28 to 2023-09-12  
**Duration:** 1,078 days

---

## 5. Text Analysis

![Text Length Distribution](figures/beauty/text_length_beauty_and_personal_care.png)

| Metric | Value |
|--------|-------|
| Avg Review Length | 250 chars |
| Avg Title Length | 21 chars |
| Reviews with Text | 100.0% |

---

## 6. Multimodal Analysis


![Multimodal Coverage](figures/beauty/multimodal_coverage_beauty_and_personal_care.png)

### Feature Coverage

| Feature | Coverage |
|---------|----------|
| Title | 100.0% |
| Description | 34.2% |
| Features | 88.9% |
| Images | 100.0% |
| **Complete (Text + Image)** | 100.0% |

### Image Statistics

| Metric | Value |
|--------|-------|
| Items with Images | 86,434 |
| Avg Images/Item | 6.52 |

---

## 7. Sparsity and K-Core Analysis

![Sparsity](figures/beauty/sparsity_beauty_and_personal_care.png)

**Matrix Sparsity:** 99.98848386%  
**Density:** 0.01151614%

### K-Core Filtering Impact

| k | Users Retained | Items Retained | Interactions Retained |
|---|----------------|----------------|----------------------|
| 2 | 100.0% | 100.0% | 100.0% |
| 3 | 100.0% | 100.0% | 100.0% |
| 5 | 100.0% | 100.0% | 100.0% |
| 10 | 6.1% | 26.4% | 24.2% |
| 20 | 0.0% | 0.1% | 0.2% |

---

## 8. Category Distribution

![Categories](figures/beauty/category_distribution_beauty_and_personal_care.png)

Top categories in the dataset:

| Category | Count |
|----------|-------|
| Beauty & Personal Care | 86,436 |

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

![Modality Alignment](figures/beauty/modality_alignment_beauty_and_personal_care.png)

Tests the **Homophily Hypothesis**: Do visually similar items share similar interaction patterns?

| Metric | Value |
|--------|-------|
| Pairs Analyzed | 5,000 |
| Pearson r | 0.0347 |
| p-value | 0.0141 |
| Spearman ρ | 0.0280 |

**Interpretation:** Very weak correlation - visual signal exists but is minimal


### 10.2 Visual Manifold Structure (Xu et al., 2025)

![Visual Manifold](figures/beauty/visual_manifold_beauty_and_personal_care.png)

Analyzes whether CLIP embeddings form meaningful clusters by category.

| Metric | Value |
|--------|-------|
| Items Projected | 5,000 |
| Projection Method | UMAP |
| Silhouette Score | 0.0000 |
| Unique Categories | 1 |

**Interpretation:** No meaningful visual clustering - visual features may not align with categories


### 10.3 BPR Negative Sampling Hardness (Xu et al., 2025)

![BPR Hardness](figures/beauty/bpr_hardness_beauty_and_personal_care.png)

Evaluates whether random negative sampling produces informative training signal.

| Metric | Value |
|--------|-------|
| Users Analyzed | 1,000 |
| Pairs Analyzed | 10,000 |
| Mean Visual Distance | 0.5063 |
| Easy Negatives (>0.8) | 0.1% |
| Medium Negatives | 98.8% |
| Hard Negatives (<0.3) | 1.2% |

**Interpretation:** Moderate negative difficulty - room for improvement

**Recommendation:** Consider mixing random and hard negative sampling


---

## 11. LATTICE Feasibility Assessment

> [!TIP]
> ✅ **PROCEED** with LATTICE architecture - All feasibility checks passed.

### 11.1 Graph Connectivity (k-NN, k=5)

| Metric | Value | Status |
|--------|-------|--------|
| Connected Components | 2 | - |
| Giant Component Size | 9,990 | - |
| Giant Component Coverage | 99.9% | ✅ PASS |
| Threshold | >50.0% | - |

**Interpretation:** PASS: Giant component covers 99.9% of items (threshold: 50.0%). Graph is sufficiently connected for LATTICE.

### 11.2 Feature Collapse Detection (White Wall Test)

| Metric | Value | Status |
|--------|-------|--------|
| Pairs Sampled | 20,000 | - |
| Avg Cosine Similarity | 0.4932 | ✅ PASS |
| Std Cosine Similarity | 0.0904 | - |
| High Similarity Pairs (>0.9) | 0.0% | - |
| Pass Threshold | <0.5 | - |

**Interpretation:** PASS: Avg cosine similarity = 0.493 (threshold: 0.5). Features show good variance. Visual encoder is producing discriminative embeddings.

### Summary

| Check | Value | Status |
|-------|-------|--------|
| Alignment (Pearson r) | 0.0347 | ✅ |
| Connectivity (Giant %) | 99.9% | ✅ |
| No Collapse (Avg Cosine) | 0.4932 | ✅ |

**Decision:** PROCEED

---

*Report generated by EDA Pipeline for Multimodal Recommendation System*
