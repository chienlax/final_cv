# Evaluation Framework

Comprehensive documentation for Research Questions, Metrics, and Experiment Setup for the Multimodal Recommendation System (MRS) project.

---

## Table of Contents

1. [Research Questions](#1-research-questions)
2. [Evaluation Metrics](#2-evaluation-metrics)
3. [Experiment Setup](#3-experiment-setup)
4. [Experiment Results](#4-experiment-results)
5. [In-Depth Analysis](#5-in-depth-analysis)


---

## 1. Research Questions

### 1.1 Primary Research Questions

| ID | Research Question | Hypothesis | Validation Method |
|----|-------------------|------------|-------------------|
| **RQ1** | How do different modalities (visual vs. text) affect recommendation quality across domains? | Visual modality has stronger predictive power for aesthetic-driven domains (Beauty, Clothing), while text modality contributes more equally for functional domains (Electronics). | Ablation study: Compare full model vs. visual-only vs. text-only |
| **RQ2** | Can graph-based multimodal methods (LATTICE, MICRO) effectively address the cold-start problem? | Inductive cold-start evaluation should show >50% of warm performance when using modal features only. | Three-track evaluation: Track 3 (cold-start) vs. Track 1 (warm) |
| **RQ3** | How do LATTICE, MICRO, and DiffMM compare in terms of recommendation quality and cold-start handling? | DiffMM's generative approach provides the best cold-start performance; LATTICE provides the best warm performance. | Cross-model comparison on all three tracks |
| **RQ4** | Does the nature of item modality alignment affect model selection? | Datasets with strong CCA correlation (>0.9) favor MICRO's contrastive approach; datasets with weak direct alignment favor LATTICE's structure learning. | Correlation of EDA alignment metrics with model performance |

### 1.2 Secondary Research Questions

| ID | Research Question | Motivation |
|----|-------------------|------------|
| **RQ5** | How does user sparsity affect multimodal recommendation performance? | Sparse users rely more on content features; active users have sufficient collaborative signal. |
| **RQ6** | What is the impact of anisotropy correction (mean centering) on embedding quality? | EDA revealed all datasets exhibit anisotropic embeddings (avg cosine ~0.5). |
| **RQ7** | Do hard negative sampling strategies improve BPR training over uniform sampling? | EDA shows 95-99% of negatives fall in "medium" difficulty range. |

### 1.3 Hypotheses Based on EDA Findings

Based on the exploratory data analysis, we formulate the following domain-specific hypotheses:

| Dataset | Observation from EDA | Hypothesis |
|---------|---------------------|------------|
| **Beauty** | Visual alignment r = -0.0009, Text alignment r = 0.025 | Text features marginally outperform visual for Beauty products |
| **Clothing** | Visual alignment r = 0.019, Text alignment r = -0.006 | Visual features are the primary signal for Clothing recommendations |
| **Electronics** | Visual alignment r = 0.016, Text alignment r = 0.018 | Both modalities contribute equally for Electronics |

> **Note:** All datasets show weak direct alignment (|r| < 0.05), indicating that simple cosine similarity does not capture user preference patterns. This motivates the use of learned representations (LATTICE/MICRO/DiffMM) over raw feature similarity.

---

## 2. Evaluation Metrics

### 2.1 Ranking Metrics

The evaluation framework implements **all-ranking evaluation**, computing metrics over the entire item catalog (not sampled negatives).

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Recall@K** | $\frac{\|\{i : i \in \text{TopK} \cap i \in \text{GT}\}\|}{\|\text{GT}\|}$ | Fraction of relevant items retrieved in top-K |
| **NDCG@K** | $\frac{\text{DCG}@K}{\text{IDCG}@K}$ where $\text{DCG} = \sum_{i=1}^{K} \frac{\mathbb{1}[i \in \text{GT}]}{\log_2(i+1)}$ | Position-aware ranking quality |
| **Precision@K** | $\frac{\|\{i : i \in \text{TopK} \cap i \in \text{GT}\}\|}{K}$ | Fraction of top-K items that are relevant |

### 2.2 K Values

Based on typical recommendation system evaluation practices:

| K | Use Case | Rationale |
|---|----------|-----------|
| 10 | Short list | Typical mobile/widget display |
| **20** | **Primary metric** | Standard benchmark comparison |
| 50 | Long list | Web/catalog browsing |

> **Primary Metric:** `Recall@20` is used for early stopping and model selection, following conventions in LATTICE and MICRO papers.

### 2.3 Metric Computation Details

```python
# Implementation from src/evaluator.py
def compute_metrics(scores, ground_truth, train_positive, k_list=[10, 20, 50]):
    """
    All-ranking evaluation with training item masking.
    
    Args:
        scores: (n_users, n_items) prediction scores
        ground_truth: {user_idx: set(item_idx)} test positive items
        train_positive: {user_idx: set(item_idx)} training items to mask
        k_list: List of K values [10, 20, 50]
    """
    # Mask training items (set to -inf)
    for user, items in train_positive.items():
        scores[user, items] = -inf
    
    # Get top-K items and compute metrics
    topk_indices = torch.topk(scores, max(k_list), dim=1)[1]
    # ... compute Recall, NDCG, Precision
```

---

## 3. Experiment Setup

### 3.1 Three-Track Evaluation Protocol

The evaluation framework implements a comprehensive three-track protocol to assess both warm performance and cold-start capability.

| Track | Split | Filter | Purpose | Metric Focus |
|-------|-------|--------|---------|--------------|
| **Track 1: Warm** | test_warm.txt | All users | Standard warm-start performance | Recall@K, NDCG@K |
| **Track 2a: Sparse Users** | test_warm.txt | Users with ≤5 interactions | User robustness (cold-ish users) | Recall@20 |
| **Track 2b: Active Users** | test_warm.txt | Users with ≥20 interactions | Active user performance | Recall@20 |
| **Track 3: Cold-Start** | test_cold.txt | All users, inductive mode | True cold-start (unseen items) | Recall@K |

### 3.2 Data Split Strategy

```
Raw 5-core CSV → Seed Sampling → Recursive k-Core → Warm/Cold Item Split → Temporal Train/Val/Test
```

**Configuration Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `seed_users` | 10,000 | Initial user sample for tractable experiments |
| `k_core` | 5 | Minimum interactions per user/item |
| `cold_item_ratio` | 0.20 | 20% of items held out as cold |
| `train_ratio` | 0.80 | Training set from warm interactions |
| `val_ratio` | 0.10 | Validation set for early stopping |
| `test_warm_ratio` | 0.10 | Test set for warm evaluation |

**Item ID Structure:**
- Warm Items: IDs `[0, N_warm - 1]`
- Cold Items: IDs `[N_warm, N_total - 1]`

> **Critical:** Cold items NEVER appear in training data. Their embeddings come purely from multimodal features (inductive inference).

### 3.3 Inductive Mode (Cold-Start Evaluation)

For Track 3, each model uses a different strategy to handle cold items:

| Model | Cold-Start Strategy |
|-------|---------------------|
| **LATTICE** | `modal_emb = project(visual) + project(text)` — no ID embedding |
| **MICRO** | `modal_emb = fused(visual, text)` via contrastive projection heads |
| **DiffMM** | `item_emb = sample_from_noise(modal_condition)` — generative forward |

### 3.4 Dataset Configuration

**Target Datasets:**

| Dataset | Full Size | Experiment Subset | Sparsity |
|---------|-----------|-------------------|----------|
| **Beauty** | 729K users, 207K items, 6.6M interactions | ~12-15K users, ~8-10K items | 99.996% |
| **Electronics** | 1.6M users, 368K items, 15.5M interactions | ~12-15K users, ~8-10K items | 99.997% |

> **Note:** Clothing dataset (2.5M users, 715K items) may be included as a stretch goal but requires careful memory management.

### 3.5 Model Configuration

**Shared Hyperparameters (Fair Comparison):**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `BATCH_SIZE` | 1024 | Gradient noise = implicit regularization |
| `EPOCHS` | 300 | With early stopping |
| `PATIENCE` | 50 | Generous for generative models |
| `LR` | 5e-4 | Lower LR for deeper models |
| `L2_REG` | 1e-3 | Strong weight decay |
| `EMBED_DIM` | 384 | Divisible by attention heads |
| `N_LAYERS` | 3 | Sweet spot (4 → oversmoothing) |
| `N_NEGATIVES` | 64 | Statistically sufficient |

**Modality Projection (MLP Bridge):**

```
768 (CLIP/SBERT) → 1024 (hidden) → LeakyReLU → Dropout(0.5) → 384 (embed_dim)
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `PROJECTION_HIDDEN_DIM` | 1024 | Non-linear visual→preference mapping |
| `PROJECTION_DROPOUT` | 0.5 | Prevent feature memorization |

**Model-Specific Parameters:**

| Model | Parameter | Value | Description |
|-------|-----------|-------|-------------|
| LATTICE | `LATTICE_K` | 40 | k-NN neighbors for item graph |
| LATTICE | `LATTICE_LAMBDA` | 0.5 | Balance original vs learned graph |
| MICRO | `MICRO_TAU` | 0.2 | InfoNCE temperature |
| MICRO | `MICRO_ALPHA` | 0.1 | Contrastive loss weight |
| DiffMM | `DIFFMM_STEPS` | 100 | Diffusion steps |
| DiffMM | `DIFFMM_NOISE_SCALE` | 0.1 | Base noise level |
| DiffMM | `DIFFMM_LAMBDA_MSI` | 1e-2 | MSI loss weight |

### 3.6 Feature Extraction

**Visual Features (CLIP):**
- Model: `openai/clip-vit-large-patch14`
- Output: 768-dimensional embeddings
- Anisotropy correction: Mean centering on warm items

**Text Features (Sentence-BERT):**
- Model: `sentence-transformers/all-mpnet-base-v2`
- Input: `title + description + features` (concatenated)
- Output: 768-dimensional embeddings
- Anisotropy correction: Mean centering on warm items

**Preprocessing Pipeline:**
```python
# Mean centering (critical for LATTICE/MICRO performance)
x = x / norm(x)              # L2 normalize
mu = x[:n_warm].mean(axis=0) # Mean from WARM items only
x = x - mu                    # Center
x = x / norm(x)              # Re-normalize
```

### 3.7 Loss Functions

**Primary Loss (BPR):**

$$L_{\text{BPR}} = -\sum_{(u,i,j) \in D} \log \sigma(\hat{y}_{ui} - \hat{y}_{uj})$$

Where:
- $(u, i, j)$: user $u$, positive item $i$, negative item $j$
- $\hat{y}_{ui} = \mathbf{e}_u^\top \mathbf{e}_i$: predicted score
- $\sigma$: sigmoid function

**Auxiliary Losses:**

| Model | Auxiliary Loss | Weight |
|-------|----------------|--------|
| LATTICE | L2 regularization only | — |
| MICRO | InfoNCE contrastive | $\alpha = 0.1$ |
| DiffMM | Diffusion denoising MSE | $\lambda_{\text{MSI}} = 0.01$ |

### 3.8 Training Infrastructure

**Hardware Configuration:**
- GPU: RTX 3060 (12GB VRAM)
- CPU: i5-13500 (6P+8E = 20 threads)
- RAM: 64GB

**Optimizations:**
- Mixed Precision Training (AMP): ~25% VRAM savings, ~15% speedup
- Cosine Annealing LR Scheduler
- Parallel data loading (6 workers)
- Gradient clipping (max_norm=1.0)

---

## 4. Experiment Results

This section presents the experimental results from training LATTICE, MICRO, and DiffMM on three Amazon 2023 datasets (Beauty, Clothing, Electronics).

> Generated tables and figures are saved in `experiment_result/`.

### 4.1 Main Results (Track 1: Warm Performance)

The following table reports performance on **warm items** (items seen during training) across all datasets and models.

| Dataset     | Model   |   recall@10 |   recall@20 |   recall@50 |   ndcg@10 |   ndcg@20 |   ndcg@50 |   precision@10 |   precision@20 |   precision@50 |
|:------------|:--------|------------:|------------:|------------:|----------:|----------:|----------:|---------------:|---------------:|---------------:|
| Beauty      | DiffMM  |      0.0442 |      0.0696 |      0.1193 |    0.0263 |    0.0332 |    0.0438 |         0.0061 |         0.0048 |         0.0033 |
| Beauty      | LATTICE |      0.0440 |      0.0657 |      0.1126 |    0.0258 |    0.0317 |    0.0417 |         0.0060 |         0.0046 |         0.0031 |
| Beauty      | **MICRO**   |      **0.0484** |      **0.0737** |      **0.1212** |    **0.0282** |    **0.0350** |    **0.0453** |         **0.0066** |         **0.0050** |         **0.0034** |
| Clothing    | DiffMM  |      0.0305 |      0.0470 |      0.0800 |    0.0164 |    0.0208 |    0.0279 |         0.0041 |         0.0031 |         0.0021 |
| Clothing    | LATTICE |      0.0295 |      0.0468 |      0.0842 |    0.0162 |    0.0208 |    0.0285 |         0.0039 |         0.0031 |         0.0022 |
| Clothing    | **MICRO**   |      **0.0350** |      **0.0504** |      **0.0859** |    **0.0189** |    **0.0230** |    **0.0305** |         **0.0047** |         **0.0033** |         **0.0023** |
| Electronics | **DiffMM**  |      **0.0540** |      **0.0809** |      **0.1306** |    **0.0308** |    **0.0381** |    **0.0487** |         **0.0075** |         **0.0056** |         **0.0036** |
| Electronics | LATTICE |      0.0445 |      0.0674 |      0.1099 |    0.0257 |    0.0319 |    0.0409 |         0.0062 |         0.0046 |         0.0030 |
| Electronics | MICRO   |      0.0478 |      0.0740 |      0.1221 |    0.0281 |    0.0352 |    0.0454 |         0.0067 |         0.0052 |         0.0033 |

**Key Findings (Track 1):**
- **MICRO** achieves the best Recall@20 on Beauty (0.0737) and Clothing (0.0504).
- **DiffMM** outperforms on Electronics (0.0809), suggesting the diffusion approach benefits functional product domains.
- **LATTICE** consistently ranks third, indicating that pure structure learning may be less effective than contrastive or generative approaches.

### 4.2 User Robustness (Track 2: Sparse vs. Active Users)

This table compares model performance across user activity levels to assess **user robustness**—whether models disproportionately favor users with rich interaction history.

| Dataset     | Model   | User Type    |   recall@10 |   recall@20 |   recall@50 |   ndcg@10 |   ndcg@20 |   ndcg@50 |   precision@10 |   precision@20 |   precision@50 |
|:------------|:--------|:-------------|------------:|------------:|------------:|----------:|----------:|----------:|---------------:|---------------:|---------------:|
| Beauty      | DiffMM  | Active (≥20) |      0.0380 |      0.0787 |      0.1472 |    0.0319 |    0.0455 |    0.0635 |         0.0140 |         0.0131 |         0.0092 |
| Beauty      | DiffMM  | Sparse (≤5)  |      0.0430 |      0.0661 |      0.1139 |    0.0245 |    0.0306 |    0.0405 |         0.0054 |         0.0041 |         0.0029 |
| Beauty      | LATTICE | Active (≥20) |      0.0658 |      0.0900 |      0.1255 |    0.0420 |    0.0508 |    0.0621 |         0.0159 |         0.0126 |         0.0084 |
| Beauty      | LATTICE | Sparse (≤5)  |      0.0420 |      0.0629 |      0.1071 |    0.0239 |    0.0296 |    0.0387 |         0.0051 |         0.0040 |         0.0027 |
| Beauty      | MICRO   | Active (≥20) |      0.0616 |      0.0872 |      0.1412 |    0.0471 |    0.0554 |    0.0708 |         0.0187 |         0.0131 |         0.0090 |
| Beauty      | MICRO   | Sparse (≤5)  |      0.0472 |      0.0714 |      0.1156 |    0.0265 |    0.0329 |    0.0423 |         0.0058 |         0.0044 |         0.0029 |
| Clothing    | DiffMM  | Active (≥20) |      0.0415 |      0.0515 |      0.0815 |    0.0290 |    0.0335 |    0.0425 |         0.0176 |         0.0118 |         0.0071 |
| Clothing    | DiffMM  | Sparse (≤5)  |      0.0304 |      0.0471 |      0.0789 |    0.0167 |    0.0211 |    0.0278 |         0.0039 |         0.0029 |         0.0020 |
| Clothing    | LATTICE | Active (≥20) |      0.0356 |      0.0562 |      0.0662 |    0.0258 |    0.0329 |    0.0366 |         0.0147 |         0.0103 |         0.0053 |
| Clothing    | LATTICE | Sparse (≤5)  |      0.0309 |      0.0474 |      0.0832 |    0.0172 |    0.0216 |    0.0290 |         0.0039 |         0.0030 |         0.0021 |
| Clothing    | MICRO   | Active (≥20) |      0.0586 |      0.0586 |      0.0827 |    0.0340 |    0.0340 |    0.0421 |         0.0206 |         0.0103 |         0.0065 |
| Clothing    | MICRO   | Sparse (≤5)  |      0.0337 |      0.0485 |      0.0836 |    0.0183 |    0.0222 |    0.0295 |         0.0043 |         0.0030 |         0.0021 |
| Electronics | DiffMM  | Active (≥20) |      0.0581 |      0.0844 |      0.1449 |    0.0483 |    0.0582 |    0.0754 |         0.0209 |         0.0154 |         0.0103 |
| Electronics | DiffMM  | Sparse (≤5)  |      0.0551 |      0.0829 |      0.1312 |    0.0304 |    0.0378 |    0.0480 |         0.0068 |         0.0052 |         0.0033 |
| Electronics | LATTICE | Active (≥20) |      0.0440 |      0.0690 |      0.1212 |    0.0323 |    0.0408 |    0.0552 |         0.0165 |         0.0121 |         0.0084 |
| Electronics | LATTICE | Sparse (≤5)  |      0.0454 |      0.0681 |      0.1084 |    0.0252 |    0.0311 |    0.0394 |         0.0055 |         0.0041 |         0.0026 |
| Electronics | MICRO   | Active (≥20) |      0.0566 |      0.0771 |      0.1456 |    0.0407 |    0.0488 |    0.0673 |         0.0198 |         0.0143 |         0.0099 |
| Electronics | MICRO   | Sparse (≤5)  |      0.0491 |      0.0757 |      0.1219 |    0.0277 |    0.0347 |    0.0442 |         0.0061 |         0.0047 |         0.0030 |

#### Active/Sparse Performance Ratio (Recall@20)

To quantify the performance gap, we compute the **Active/Sparse Ratio** = `Recall@20(Active) / Recall@20(Sparse)`:

| Dataset     | Model   | Sparse R@20 | Active R@20 | Active/Sparse Ratio |
|:------------|:--------|------------:|------------:|--------------------:|
| Beauty      | DiffMM  |      0.0661 |      0.0787 |        **1.19×**    |
| Beauty      | LATTICE |      0.0629 |      0.0900 |        **1.43×**    |
| Beauty      | MICRO   |      0.0714 |      0.0872 |        **1.22×**    |
| Clothing    | DiffMM  |      0.0471 |      0.0515 |        **1.09×**    |
| Clothing    | LATTICE |      0.0474 |      0.0562 |        **1.19×**    |
| Clothing    | MICRO   |      0.0485 |      0.0586 |        **1.21×**    |
| Electronics | DiffMM  |      0.0829 |      0.0844 |        **1.02×**    |
| Electronics | LATTICE |      0.0681 |      0.0690 |        **1.01×**    |
| Electronics | MICRO   |      0.0757 |      0.0771 |        **1.02×**    |

**Key Findings (Track 2 - User Robustness):**

1. **Electronics shows remarkable user fairness** (ratio ~1.01-1.02×):
   - All three models perform nearly identically on sparse vs. active users.
   - This suggests that **functional product domains** benefit heavily from multimodal features, reducing reliance on collaborative history.
   - Implication: Multimodal approaches effectively bridge the user cold-start gap for technical products.

2. **Beauty shows the largest gap** (ratio 1.19-1.43×):
   - LATTICE exhibits the highest disparity (1.43×), indicating its structure-learning approach favors users with richer interaction graphs.
   - DiffMM and MICRO show more balanced performance (~1.2×).
   - Hypothesis: Aesthetic preferences are more subjective and harder to infer from limited interactions.

3. **Model-specific observations:**
   - **DiffMM** is the most user-fair across domains (1.02-1.19×), likely due to its generative approach sampling from a learned distribution rather than relying heavily on user embeddings.
   - **LATTICE** shows the highest variance (1.01-1.43×), suggesting its k-NN graph structure amplifies cold-user disadvantages when interaction density is low.
   - **MICRO** maintains consistent middle-ground performance (1.02-1.22×).

4. **Precision gap is larger than Recall gap:**
   - Active users have 2-4× higher precision than sparse users (e.g., Beauty MICRO: 0.0131 vs 0.0044).
   - This indicates models successfully retrieve relevant items for sparse users but rank them less precisely due to weaker preference signals.


### 4.3 Cold-Start Performance (Track 3: Inductive Evaluation)

The following table shows performance on **cold items** (items not seen during training, using only modal features).

| Dataset     | Model   |   recall@10 |   recall@20 |   recall@50 |   Cold/Warm@10 |   Cold/Warm@20 |   Cold/Warm@50 |
|:------------|:--------|------------:|------------:|------------:|---------------:|---------------:|---------------:|
| Beauty      | DiffMM  |      0.0055 |      0.0108 |      0.0282 |         0.1244 |         0.1553 |         0.2365 |
| Beauty      | **LATTICE** |      **0.0599** |      **0.0875** |      **0.1407** |         **1.3605** |         **1.3305** |         **1.2494** |
| Beauty      | **MICRO**   |      **0.0663** |      **0.0975** |      **0.1597** |         **1.3690** |         **1.3231** |         **1.3172** |
| Clothing    | DiffMM  |      0.0061 |      0.0113 |      0.0275 |         0.2015 |         0.2413 |         0.3435 |
| Clothing    | **LATTICE** |      **0.0493** |      **0.0750** |      **0.1226** |         **1.6719** |         **1.6014** |         **1.4554** |
| Clothing    | **MICRO**   |      **0.0482** |      **0.0770** |      **0.1376** |         **1.3796** |         **1.5274** |         **1.6023** |
| Electronics | DiffMM  |      0.0051 |      0.0099 |      0.0252 |         0.0952 |         0.1222 |         0.1931 |
| Electronics | LATTICE |      0.0397 |      0.0658 |      0.1280 |         0.8912 |         0.9752 |         1.1648 |
| Electronics | MICRO   |      0.0408 |      0.0682 |      0.1312 |         0.8528 |         0.9222 |         1.0751 |

> **Cold/Warm Ratio** = `track3_recall / track1_recall`. Values >1.0 indicate the model performs *better* on cold items than warm items.

**Key Findings (Track 3 - Cold-Start):**

1. **LATTICE and MICRO achieve Cold/Warm ratios >1.0** on Beauty and Clothing, meaning they actually perform *better* on cold items! This counter-intuitive result suggests:
   - Modal features (visual/text) provide stronger signal than ID embeddings for these aesthetic domains.
   - The inductive MLP projections generalize well to unseen items.

2. **DiffMM fails catastrophically on cold-start** (Cold/Warm ratio ~0.12-0.24), indicating:
   - The generative sampling approach does not effectively leverage modal conditioning.
   - The diffusion process may be overfitting to ID-based patterns.

3. **Electronics shows weaker cold-start transfer** (ratios ~0.9-1.0), suggesting:
   - Functional products benefit more from collaborative signals than content features.

### 4.4 Training Dynamics

The following figures show training loss and validation metrics across epochs.

**Training Loss:**

![Beauty Training Loss](../experiment_result/figures/training_loss_beauty.png)

![Clothing Training Loss](../experiment_result/figures/training_loss_clothing.png)

![Electronics Training Loss](../experiment_result/figures/training_loss_electronics.png)

**Validation Recall@20:**

![Beauty Val Recall](../experiment_result/figures/val_recall_beauty.png)

![Clothing Val Recall](../experiment_result/figures/val_recall_clothing.png)

![Electronics Val Recall](../experiment_result/figures/val_recall_electronics.png)

---

## 5. In-Depth Analysis

### 5.1 Ablation Studies

To answer **RQ1** (modality contribution per domain), we conduct three-way ablation:

| Condition | Visual Features | Text Features |
|-----------|-----------------|---------------|
| **Full** | ✓ | ✓ |
| **Visual-Only** | ✓ | Zeroed |
| **Text-Only** | Zeroed | ✓ |

**Implementation:**
```python
# From src/dataset.py
if ablation_mode == "no_visual":
    self.feat_visual = torch.zeros_like(self.feat_visual)
elif ablation_mode == "no_text":
    self.feat_text = torch.zeros_like(self.feat_text)
```

> Ablation results are stored in `checkpoints_ablation/`. Analysis to be added in Section 5.1.

### 5.2 Sensitivity Analysis

> Sensitivity analysis results to be populated from `scripts/run_sensitivity.py` outputs.

---

## References

1. **Zhang et al. (2021)** - LATTICE: Mining Latent Structures for Multimodal Recommendation
2. **Zhang et al. (2022)** - MICRO: Contrastive Multimodal Recommendation
3. **Jiang et al. (2023)** - DiffMM: Diffusion Model for Multimodal Recommendation
4. **Xu et al. (2025)** - Multimodal Recommender Systems: A Survey

---

*Documentation generated for the Multimodal Recommendation System Evaluation Framework*

