# EDA Report: Electronics

**Generated:** 2025-12-17T21:41:26.690508  
**Sampling Strategy:** Dense (K-Core k=5 + 60 months)
**Sampling Ratio**: 0.01

---

## 1. Data Overview

### Loading Statistics

| Metric | Interactions | Metadata |
|--------|-------------|----------|
| Total Records | 43,886,944 | 1,610,012 |
| Sampled Records | 5,071,539 | 167,692 |
| Memory (MB) | 3052.9 | 420.01 |

### Interaction Statistics

| Metric | Value |
|--------|-------|
| Users | 604,284 |
| Items | 167,692 |
| Interactions | 5,071,539 |
| Avg Rating | 4.30 |
| Rating Std | 1.25 |
| Sparsity | 99.99499520% |

---

## 2. Rating Distribution

![Rating Distribution](figures/electronics/rating_distribution_electronics.png)

| Rating | Count | Percentage |
|--------|-------|------------|
| 1.0 | 412,102 | 8.1% |
| 2.0 | 206,042 | 4.1% |
| 3.0 | 321,119 | 6.3% |
| 4.0 | 615,763 | 12.1% |
| 5.0 | 3,516,513 | 69.3% |

---

## 3. User and Item Analysis

### Power-Law Distribution

![Interaction Frequency](figures/electronics/interaction_frequency_electronics.png)

**User Patterns:**
- Mean interactions/user: 8.39
- Median interactions/user: 6.0
- Cold-start users (<5 interactions): 0.0%
- Power-law exponent α: 3.05

**Item Patterns:**
- Mean interactions/item: 30.24
- Median interactions/item: 11.0
- Cold-start items (<5 interactions): 0.0%
- Power-law exponent α: 1.88

### Pareto Analysis (Interaction Concentration)

Top users account for a disproportionate share of interactions:

| User Tier | % of Total Interactions |
|-----------|------------------------|
| Top 1% | 6.9% |
| Top 5% | 18.0% |
| Top 10% | 27.0% |
| Top 20% | 40.4% |
| Top 50% | 68.1% |
| Top 100% | 100.0% |

---

## 4. Temporal Analysis

![Temporal Patterns](figures/electronics/temporal_patterns_electronics.png)

**Date Range:** 2018-10-09 to 2023-09-12  
**Duration:** 1,798 days

---

## 5. Text Analysis

![Text Length Distribution](figures/electronics/text_length_electronics.png)

| Metric | Value |
|--------|-------|
| Avg Review Length | 282 chars |
| Avg Title Length | 24 chars |
| Reviews with Text | 100.0% |

---

## 6. Multimodal Analysis


![Multimodal Coverage](figures/electronics/multimodal_coverage_electronics.png)

### Feature Coverage

| Feature | Coverage |
|---------|----------|
| Title | 100.0% |
| Description | 41.7% |
| Features | 90.2% |
| Images | 100.0% |
| **Complete (Text + Image)** | 100.0% |

### Image Statistics

| Metric | Value |
|--------|-------|
| Items with Images | 167,681 |
| Avg Images/Item | 6.47 |

---

## 7. Sparsity and K-Core Analysis

![Sparsity](figures/electronics/sparsity_electronics.png)

**Matrix Sparsity:** 99.99499520%  
**Density:** 0.00500480%

### K-Core Filtering Impact

| k | Users Retained | Items Retained | Interactions Retained |
|---|----------------|----------------|----------------------|
| 2 | 100.0% | 100.0% | 100.0% |
| 3 | 100.0% | 100.0% | 100.0% |
| 5 | 100.0% | 100.0% | 100.0% |
| 10 | 10.5% | 20.6% | 20.2% |
| 20 | 0.0% | 0.0% | 0.0% |

---

## 8. Category Distribution

![Categories](figures/electronics/category_distribution_electronics.png)

Top categories in the dataset:

| Category | Count |
|----------|-------|
| Electronics | 159,913 |
| All Electronics | 3,373 |
| Computers | 1,609 |
| Amazon Devices | 977 |
| Apple Products | 477 |
| Home Audio & Theater | 469 |
| Camera & Photo | 338 |
| Car & Vehicle Electronics | 290 |
| Amazon Devices & Accessories | 98 |
| Car Electronics | 92 |

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

![Modality Alignment](figures/electronics/modality_alignment_electronics.png)

Tests the **Homophily Hypothesis**: Do visually similar items share similar interaction patterns?

| Metric | Value |
|--------|-------|
| Pairs Analyzed | 20,000 |
| Pearson r | 0.0106 |
| p-value | 0.1350 |
| Spearman ρ | 0.0042 |

**Interpretation:** No significant correlation - visual features may not align with user preferences


### 10.2 Visual Manifold Structure (Xu et al., 2025)

![Visual Manifold](figures/electronics/visual_manifold_electronics.png)

Analyzes whether CLIP embeddings form meaningful clusters by category.

| Metric | Value |
|--------|-------|
| Items Projected | 10,000 |
| Projection Method | UMAP |
| Silhouette Score | -0.5550 |
| Unique Categories | 13 |

**Interpretation:** No meaningful visual clustering - visual features may not align with categories


### 10.3 BPR Negative Sampling Hardness (Xu et al., 2025)

![BPR Hardness](figures/electronics/bpr_hardness_electronics.png)

Evaluates whether random negative sampling produces informative training signal.

| Metric | Value |
|--------|-------|
| Users Analyzed | 2,000 |
| Pairs Analyzed | 40,000 |
| Mean Visual Distance | 0.4377 |
| Easy Negatives (>0.8) | 0.0% |
| Medium Negatives | 95.6% |
| Hard Negatives (<0.3) | 4.4% |

**Interpretation:** Moderate negative difficulty - room for improvement

**Recommendation:** Consider mixing random and hard negative sampling

### 10.4 Text Embedding Extraction (Sentence-BERT)

| Metric | Value |
|--------|-------|
| Model | `sentence-transformers/all-mpnet-base-v2` |
| Items Processed | 25,000 |
| Success Rate | 100.0% |
| Embedding Dimension | 768 |
| Processing Time | 280.5s |
| Throughput | 89.1 items/sec |
| Avg Text Length | 467 chars |

### 10.5 Semantic-Interaction Alignment (Text)

![Semantic Alignment](figures/electronics/semantic_alignment_electronics.png)

Tests whether items with similar text descriptions have similar buyers.

| Metric | Value |
|--------|-------|
| Pairs Analyzed | 7,500 |
| Pearson r | 0.0234 |
| p-value | 0.0429 |
| Mean Text Similarity | 0.2036 |
| Mean Interaction Similarity | 0.0000 |
| **Signal Strength** | 🔴 NOISE |

**Interpretation:** Very weak correlation (r=0.0234): Text descriptions do NOT predict user behavior. Users likely buy based on visual appeal, brand, or price rather than descriptions.

**Recommendation:** Deprioritize text encoder in final model, or use text only as filter/fallback.

### 10.6 Cross-Modal Consistency (Text vs Image)

![Cross-Modal Consistency](figures/electronics/cross_modal_consistency_electronics.png)

Measures whether text and image embeddings agree for the same items.

| Metric | Value |
|--------|-------|
| Items with Both Modalities | 2,930 |
| Projection Method | linear |
| Text Dim → Projected | 768 → 768 |
| Image Dim → Projected | 768 → 768 |
| **Mean Similarity** | -0.0063 |
| Std Similarity | 0.0324 |
| Low Agreement (<0.3) | 100.0% |
| Moderate (0.3-0.6) | 0.0% |
| High Agreement (>0.6) | 0.0% |
| **Status** | 🔴 DISAGREE |

**Interpretation:** LOW cross-modal agreement (avg=-0.006): Text and image embeddings point in different directions. This indicates a fundamental mismatch - either descriptions don't match images, or encoders have domain shift.

**Recommendation:** Investigate: (1) Check if product images match descriptions, (2) Fine-tune encoders on domain, (3) Use separate modality branches.


---

## 11. LATTICE Feasibility Assessment

> [!CAUTION]
> ⛔ **STOP** - LATTICE feasibility checks failed. Revisit Feature Extraction.

### 11.1 Graph Connectivity (k-NN, k=5)

| Metric | Value | Status |
|--------|-------|--------|
| Connected Components | 4 | - |
| Giant Component Size | 9,946 | - |
| Giant Component Coverage | 99.5% | ✅ PASS |
| Threshold | >50.0% | - |

**Interpretation:** PASS: Giant component covers 99.5% of items (threshold: 50.0%). Graph is sufficiently connected for LATTICE.

### 11.2 Feature Collapse Detection (White Wall Test)

| Metric | Value | Status |
|--------|-------|--------|
| Pairs Sampled | 50,000 | - |
| Avg Cosine Similarity | 0.5645 | ⚠️ WARNING |
| Std Cosine Similarity | 0.0841 | - |
| High Similarity Pairs (>0.9) | 0.0% | - |
| Pass Threshold | <0.5 | - |

**Interpretation:** WARNING: Avg cosine similarity = 0.564 (pass: <0.5, collapse: >0.9). Features show moderate similarity. May work but suboptimal. Consider testing with alternative visual encoder.

### Summary

| Check | Value | Status |
|-------|-------|--------|
| Alignment (Pearson r) | 0.0106 | ✅ |
| Connectivity (Giant %) | 99.5% | ✅ |
| No Collapse (Avg Cosine) | 0.5645 | ❌ |

**Decision:** STOP

---

*Report generated by EDA Pipeline for Multimodal Recommendation System*
