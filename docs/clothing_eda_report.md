# EDA Report: Clothing, Shoes and Jewelry

**Generated:** 2025-12-18T13:52:55.870148  
**Sampling Strategy:** random

---

## 1. Data Overview

### Loading Statistics

| Metric | Interactions | Metadata |
|--------|-------------|----------|
| Total Records | 23,102,537 | 7,218,481 |
| Sampled Records | 23,102,537 | 714,957 |
| Memory (MB) | 5706.39 | 1378.84 |

### Interaction Statistics

| Metric | Value |
|--------|-------|
| Users | 2,524,981 |
| Items | 714,957 |
| Interactions | 23,102,537 |
| Avg Rating | 4.29 |
| Rating Std | 1.19 |
| Sparsity | 99.99872026% |

### 5-Core Validation

This dataset uses **5-core filtering**, ensuring every user and item has at least 5 interactions.
This removes cold-start users/items and creates a denser, more connected graph for recommendation algorithms.


| Property | Value | Status |
|----------|-------|--------|
| Min interactions/user | 5 | ✅ VALID |
| Min interactions/item | 5 | ✅ VALID |
| Avg interactions/user | 9.15 | - |
| Avg interactions/item | 32.31 | - |
| Median interactions/user | 7 | - |
| Median interactions/item | 10 | - |


> [!NOTE]
> **Density Analysis:** At 0.00127974% density, this dataset is extremely sparse - typical for raw e-commerce data before filtering.
> 
> **Benchmark Reference:**
> - Raw Amazon data: ~0.001-0.01% density (99.99%+ sparsity)
> - 5-core filtered: ~0.01-0.1% density (typical range)
> - MovieLens 20M: ~0.5% density (research benchmark)
> - Netflix Prize: ~1.2% density (denser due to explicit ratings)

---

## 2. Rating Distribution

![Rating Distribution](figures/clothing/rating_distribution_clothing,_shoes_and_jewelry.png)

| Rating | Count | Percentage |
|--------|-------|------------|
| 1.0 | 1,415,199 | 6.1% |
| 2.0 | 1,139,555 | 4.9% |
| 3.0 | 2,001,386 | 8.7% |
| 4.0 | 3,359,236 | 14.5% |
| 5.0 | 15,187,161 | 65.7% |

### Rating Skewness Analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean Rating | 4.29 | strongly left-skewed (positive bias) |
| % Positive (4-5 ⭐) | 80.3% | Unusual distribution |
| Rating Std | 1.19 | Moderate variance |

> [!TIP]
> **Insight:** Users preferentially leave reviews for products they like. Amazon reviews typically show 60-80% positive ratings due to self-selection bias.

---

## 3. User and Item Analysis

### Power-Law Distribution

![Interaction Frequency](figures/clothing/interaction_frequency_clothing,_shoes_and_jewelry.png)

**User Patterns:**
- Mean interactions/user: 9.15
- Median interactions/user: 7.0
- Cold-start users (<5 interactions): 0.0%
- Power-law exponent α: 2.80

**Item Patterns:**
- Mean interactions/item: 32.31
- Median interactions/item: 10.0
- Cold-start items (<5 interactions): 0.0%
- Power-law exponent α: 1.91

### Power-Law Fit Quality Interpretation


| Entity | α Value | Interpretation | Implication |
|--------|---------|----------------|-------------|
| Users | 2.80 | Good fit - typical power-law for recommendation data | Standard sampling OK |
| Items | 1.91 | Flatter distribution - popularity more evenly spread | Uniform negative sampling acceptable |

> [!NOTE]
> **Power-Law Reference:** α ≈ 2.0-3.0 indicates a well-behaved power-law typical for recommendation systems.
> - α < 2.0: Distribution is relatively flat (many active users/popular items)
> - α > 3.0: Extreme concentration (Pareto principle strongly applies)

### Pareto Analysis (Interaction Concentration)

Top users account for a disproportionate share of interactions:

| User Tier | % of Total Interactions |
|-----------|------------------------|
| Top 1% | 7.1% |
| Top 5% | 18.4% |
| Top 10% | 27.8% |
| Top 20% | 41.8% |
| Top 50% | 69.7% |
| Top 100% | 100.0% |

---

## 4. Temporal Analysis

![Temporal Patterns](figures/clothing/temporal_patterns_clothing,_shoes_and_jewelry.png)

**Date Range:** 1999-12-01 to 2023-09-12  
**Duration:** 8,686 days

---

## 5. Text Analysis

![Text Length Distribution](figures/clothing/text_length_clothing,_shoes_and_jewelry.png)

| Metric | Value |
|--------|-------|
| Avg Review Length | 0 chars |
| Avg Title Length | 0 chars |
| Reviews with Text | 0.0% |

---

## 6. Multimodal Analysis


![Multimodal Coverage](figures/clothing/multimodal_coverage_clothing,_shoes_and_jewelry.png)

### Feature Coverage

| Feature | Coverage |
|---------|----------|
| Title | 100.0% |
| Description | 49.4% |
| Features | 95.4% |
| Images | 100.0% |
| **Complete (Text + Image)** | 100.0% |

### Image Statistics

| Metric | Value |
|--------|-------|
| Items with Images | 714,606 |
| Avg Images/Item | 5.22 |

---

## 7. Sparsity and K-Core Analysis

![Sparsity](figures/clothing/sparsity_clothing,_shoes_and_jewelry.png)

**Matrix Sparsity:** 99.99872026%  
**Density:** 0.00127974%

### K-Core Filtering Impact

| k | Users Retained | Items Retained | Interactions Retained |
|---|----------------|----------------|----------------------|
| 5 | 100.0% | 100.0% | 100.0% |
| 10 | 16.9% | 24.2% | 30.0% |
| 15 | 1.4% | 2.7% | 3.3% |
| 20 | 0.0% | 0.0% | 0.0% |

---

## 8. Category Distribution

![Categories](figures/clothing/category_distribution_clothing,_shoes_and_jewelry.png)

Top categories in the dataset:

| Category | Count |
|----------|-------|
| Clothing, Shoes & Jewelry | 714,466 |
| Shoe, Jewelry & Watch Accessories | 491 |

---

## 9. Data-Driven Insights and Recommendations

### Dataset Quality Assessment

| Aspect | Finding | Implication |
|--------|---------|-------------|
| **5-Core Status** | ✅ 5-core validation passed - dataset is properly filtered | No additional k-core filtering needed; proceed with model training |
| **Sparsity** | Extreme sparsity (<0.01% density) - collaborative filtering alone may struggle | Strongly recommend hybrid approach with content-based features (text/image embeddings) |
| **Long-Tail** | User α=2.80, Item α=1.91 | Standard random sampling is acceptable for training |
| **Cold-Start** | Cold-start is minimal (0.0% users, 0.0% items) thanks to 5-core filtering | Focus on recommendation quality rather than cold-start mitigation |

### Actionable Recommendations

1. **Multimodal Features:** At 0.00127974% density, leverage text and image embeddings to enrich item representations. CLIP or Sentence-BERT embeddings can bridge sparse interaction signals.

2. **Rating Bias Correction:** Mean rating of 4.29 indicates strong positive bias. Consider using implicit feedback (interactions) rather than explicit ratings for training.

3. **Ready for Training:** 5-core filtering ensures sufficient interaction density. Proceed with LATTICE, LightGCN, or other graph-based methods.

4. **Visual Feature Extraction:** With 100% image coverage, consider CLIP-based visual embeddings for multimodal recommendation.


---

## 10. Multimodal Recommendation Readiness (Academic Analysis)


### 10.1 Modality-Interaction Alignment (Liu et al., 2024)

![Modality Alignment](figures/clothing/modality_alignment_clothing,_shoes_and_jewelry.png)

Tests the **Homophily Hypothesis**: Do visually similar items share similar interaction patterns?

| Metric | Value |
|--------|-------|
| Pairs Analyzed | 20,000 |
| Pearson r | 0.0190 |
| p-value | 0.0072 |
| Spearman ρ | 0.0205 |

**Interpretation:** Very weak correlation - visual signal exists but is minimal


### 10.2 Visual Manifold Structure (Xu et al., 2025)

![Visual Manifold](figures/clothing/visual_manifold_clothing,_shoes_and_jewelry.png)

Analyzes whether CLIP embeddings form meaningful clusters by category.

| Metric | Value |
|--------|-------|
| Items Projected | 10,000 |
| Projection Method | UMAP |
| Silhouette Score | -0.1409 |
| Unique Categories | 2 |

**Interpretation:** No meaningful visual clustering - visual features may not align with categories


### 10.3 BPR Negative Sampling Hardness (Xu et al., 2025)

![BPR Hardness](figures/clothing/bpr_hardness_clothing,_shoes_and_jewelry.png)

Evaluates whether random negative sampling produces informative training signal.

| Metric | Value |
|--------|-------|
| Users Analyzed | 2,000 |
| Pairs Analyzed | 40,000 |
| Mean Visual Distance | 0.3722 |
| Easy Negatives (>0.8) | 0.0% |
| Medium Negatives | 80.0% |
| Hard Negatives (<0.3) | 20.0% |

**Interpretation:** Moderate negative difficulty - room for improvement

**Recommendation:** Consider mixing random and hard negative sampling

### 10.4 Text Embedding Extraction (Sentence-BERT)

| Metric | Value |
|--------|-------|
| Model | `sentence-transformers/all-mpnet-base-v2` |
| Items Processed | 25,000 |
| Success Rate | 100.0% |
| Embedding Dimension | 768 |
| Processing Time | 184.6s |
| Throughput | 135.4 items/sec |
| Avg Text Length | 429 chars |

### 10.5 Semantic-Interaction Alignment (Text)

![Semantic Alignment](figures/clothing/semantic_alignment_clothing,_shoes_and_jewelry.png)

Tests whether items with similar text descriptions have similar buyers.

| Metric | Value |
|--------|-------|
| Pairs Analyzed | 7,500 |
| Pearson r | -0.0058 |
| p-value | 0.6163 |
| Mean Text Similarity | 0.2948 |
| Mean Interaction Similarity | 0.0000 |
| **Signal Strength** | 🔴 NOISE |

**Interpretation:** Very weak correlation (r=-0.0058): Text descriptions do NOT predict user behavior. Users likely buy based on visual appeal, brand, or price rather than descriptions.

**Recommendation:** Deprioritize text encoder in final model, or use text only as filter/fallback.

### 10.6 Cross-Modal Consistency (Text vs Image)

![Cross-Modal Consistency](figures/clothing/cross_modal_consistency_clothing,_shoes_and_jewelry.png)

Measures whether text and image embeddings agree for the same items.

| Metric | Value |
|--------|-------|
| Items with Both Modalities | 438 |
| Projection Method | linear |
| Text Dim → Projected | 768 → 768 |
| Image Dim → Projected | 768 → 768 |
| **Mean Similarity** | -0.0204 |
| Std Similarity | 0.0370 |
| Low Agreement (<0.3) | 100.0% |
| Moderate (0.3-0.6) | 0.0% |
| High Agreement (>0.6) | 0.0% |
| **Status** | 🔴 DISAGREE |

**Interpretation:** LOW cross-modal agreement (avg=-0.020): Text and image embeddings point in different directions. This indicates a fundamental mismatch - either descriptions don't match images, or encoders have domain shift.

**Recommendation:** Investigate: (1) Check if product images match descriptions, (2) Fine-tune encoders on domain, (3) Use separate modality branches.

### 10.7 CCA Cross-Modal Analysis

Canonical Correlation Analysis measures linear relationship capacity between modalities.

| Metric | Value |
|--------|-------|
| Items Analyzed | 438 |
| CCA Components | 10 |
| Mean CCA Correlation | 1.0000 |
| Top-5 Correlations | 1.000, 1.000, 1.000, 1.000, 1.000 |

**Interpretation:** STRONG CCA correlation (1.000). Good linear relationship between modalities.

**Recommendation:** MICRO contrastive loss should converge well.

### 10.8 Anisotropy Check (Signal Crisis Fix)

![Anisotropy Comparison](figures/clothing/anisotropy_comparison_clothing,_shoes_and_jewelry.png)

Detects "Cone Effect" in embeddings and tests if mean centering helps.

| Metric | Before Centering | After Centering |
|--------|------------------|-----------------|
| Avg Cosine Similarity | 0.6195 | 0.0007 |
| Std Cosine Similarity | 0.0881 | 0.1422 |
| Pairs Sampled | 20,000 | - |
| Improvement Ratio | 99.9% | - |
| **Status** | ⚠️ ANISOTROPIC | - |

**Interpretation:** ANISOTROPIC: Avg cosine = 0.620 (>0.4). Centering FIXED the issue: after centering = 0.001. Embeddings were in a narrow cone but centering spread them out.

**Recommendation:** Apply mean centering to all embeddings before using in MRS models. This should significantly improve LATTICE/MICRO performance.

### 10.9 User Consistency (Interaction Homophily)

![User Consistency](figures/clothing/user_consistency_clothing,_shoes_and_jewelry.png)

Measures whether users buy visually similar items (validates visual MRS approach).

| Metric | Value |
|--------|-------|
| Users Analyzed | 1,500 |
| Users with ≥5 Items | 2,524,981 |
| Mean Local Distance | 0.3496 |
| Mean Global Distance | 0.3801 |
| **Consistency Ratio** | 0.9199 |
| Users with Visual Coherence | 54.5% |
| **Status** | ✅ CONSISTENT |

**Interpretation:** MODERATE CONSISTENCY: Some visual preference signal exists. Local (0.350) < Global (0.380). Ratio = 0.920.

**Recommendation:** Visual features have some predictive power. Consider combining with text features for better performance.


---

## 11. LATTICE Feasibility Assessment

> [!CAUTION]
> ⛔ **STOP** - LATTICE feasibility checks failed. Revisit Feature Extraction.

### 11.1 Graph Connectivity (k-NN, k=5)

| Metric | Value | Status |
|--------|-------|--------|
| Connected Components | 1 | - |
| Giant Component Size | 10,000 | - |
| Giant Component Coverage | 100.0% | ✅ PASS |
| Threshold | >50.0% | - |

**Interpretation:** PASS: Giant component covers 100.0% of items (threshold: 50.0%). Graph is sufficiently connected for LATTICE.

### 11.2 Feature Collapse Detection (White Wall Test)

| Metric | Value | Status |
|--------|-------|--------|
| Pairs Sampled | 50,000 | - |
| Avg Cosine Similarity | 0.6199 | ⚠️ WARNING |
| Std Cosine Similarity | 0.0881 | - |
| High Similarity Pairs (>0.9) | 0.0% | - |
| Pass Threshold | <0.5 | - |

**Interpretation:** WARNING: Avg cosine similarity = 0.620 (pass: <0.5, collapse: >0.9). Features show moderate similarity. May work but suboptimal. Consider testing with alternative visual encoder.

### Summary

| Check | Value | Status |
|-------|-------|--------|
| Alignment (Pearson r) | 0.019 | ✅ |
| Connectivity (Giant %) | 100.0% | ✅ |
| No Collapse (Avg Cosine) | 0.6199 | ❌ |

**Decision:** STOP

---

*Report generated by EDA Pipeline for Multimodal Recommendation System*
