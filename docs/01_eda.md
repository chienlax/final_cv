# Exploratory Data Analysis (EDA) Documentation

This document provides technical documentation for the EDA pipeline used to analyze the Amazon Review 2023 dataset for multimodal recommendation research.

---

## Table of Contents

1. [Data Sources](#1-data-sources)
2. [Sampling Strategies](#2-sampling-strategies)
3. [Basic Statistics](#3-basic-statistics)
4. [Sparsity & K-Core Analysis](#4-sparsity--k-core-analysis)
5. [User & Item Behavior Analysis](#5-user--item-behavior-analysis)
6. [Multimodal Feature Analysis](#6-multimodal-feature-analysis)
7. [Visualizations](#7-visualizations)
8. [Multimodal Feature Quality Analysis](#8-multimodal-feature-quality-analysis)
9. [Text Analysis (Semantic Gap Validation)](#9-text-analysis-semantic-gap-validation)
10. [Model Feasibility Decision](#10-model-feasibility-decision)
11. [Command-Line Interface](#11-command-line-interface)

---

## 1. Data Sources

### Amazon Review 2023 Dataset

The EDA pipeline processes the [Amazon Review 2023 dataset](https://amazon-reviews-2023.github.io/), a large-scale dataset of product reviews and metadata.

#### Interaction Files
| Dataset | File | Description |
|---------|------|-------------|
| Beauty | `Beauty_and_Personal_Care.jsonl.gz` | ~24M reviews |
| Clothing | `Clothing_Shoes_and_Jewelry.jsonl.gz` | ~32M reviews |
| Electronics | `Electronics.jsonl.gz` | ~20M reviews |

#### Metadata Files
| Dataset | File | Description |
|---------|------|-------------|
| Beauty | `meta_Beauty_and_Personal_Care.jsonl.gz` | ~1M products |
| Clothing | `meta_Clothing_Shoes_and_Jewelry.jsonl.gz` | ~1.5M products |
| Electronics | `meta_Electronics.jsonl.gz` | ~1M products |

#### Schema

**Interactions:**
| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | Anonymized user identifier |
| `parent_asin` | string | Product identifier |
| `rating` | float | Rating 1.0-5.0 |
| `timestamp` | int | Unix timestamp |
| `text` | string | Review text |
| `title` | string | Review title |
| `verified_purchase` | bool | Verified purchase flag |
| `helpful_vote` | int | Helpful votes count |

**Metadata:**
| Field | Type | Description |
|-------|------|-------------|
| `parent_asin` | string | Product identifier |
| `title` | string | Product title |
| `description` | list[str] | Product descriptions |
| `features` | list[str] | Product features |
| `images` | list[dict] | Image URLs (hi_res, large, thumb) |
| `main_category` | string | Primary category |
| `average_rating` | float | Product average rating |
| `rating_number` | int | Total rating count |

---

## 2. Sampling Strategies

The data loader in [`src/eda/data_loader.py`](../src/eda/data_loader.py) implements 5 sampling strategies, each preserving different data properties.

### Strategy Comparison

| Strategy | Function | Description | When to Use |
|----------|----------|-------------|-------------|
| **Random** | `load_interactions_sample()` | Hash-based user sampling | Quick exploration, memory-efficient |
| **K-Core** | `load_interactions_kcore()` | Iterative k-core filtering | Structural analysis, graph models |
| **Temporal** | `load_interactions_temporal()` | Last N months only | Recent trend analysis |
| **Dense** | `load_interactions_dense_subgraph()` | K-Core + Temporal combined | LATTICE/GNN feasibility |
| **Full** | `load_interactions_all()` | No sampling | Production preprocessing |

### Method Details

#### Random Sampling
```python
load_interactions_sample(file_path, sample_ratio=0.1, seed=42)
```
- Uses **deterministic hash-based sampling** on `user_id` for reproducibility
- Formula: `MD5(user_id + seed) % 100 < sample_ratio * 100`
- **Preserves**: Original distribution of ratings, categories
- **Use case**: Quick dataset exploration, memory-constrained environments

#### K-Core Filtering
```python
load_interactions_kcore(file_path, k=5, max_iterations=50)
```
- **Iteratively removes** users and items with fewer than `k` interactions until convergence
- **Preserves**: Network density, graph structure for GNN models
- **Trade-off**: Higher `k` → denser graph but fewer entities
- **Recommended**: `k=5` balances density vs. coverage

#### Temporal Filtering
```python
load_interactions_temporal(file_path, months=6, end_date=None)
```
- Selects interactions from **last N months** only
- **Preserves**: Temporal density, recent trends, seasonal patterns
- **Use case**: Time-sensitive analysis, avoiding distribution shift from 20-year span

#### Dense Subgraph (Combined)
```python
load_interactions_dense_subgraph(file_path, strategy="dense", k=5, months=12)
```
- Applies **temporal filtering first**, then **k-core filtering**
- Produces the **densest possible subgraph** for academic analysis
- **Critical for**: LATTICE feasibility checks, graph connectivity analysis

---

## 3. Basic Statistics

Computed by [`src/eda/basic_stats.py`](../src/eda/basic_stats.py).

### Interaction Statistics

| Metric | Description | Formula/Method |
|--------|-------------|----------------|
| `n_interactions` | Total review count | `len(df)` |
| `n_users` | Unique users | `df['user_id'].nunique()` |
| `n_items` | Unique items | `df['item_id'].nunique()` |
| `rating_mean` | Average rating | `df['rating'].mean()` |
| `rating_std` | Rating standard deviation | `df['rating'].std()` |
| `rating_median` | Median rating | `df['rating'].median()` |
| `avg_review_length` | Mean review text length | `df['text'].str.len().mean()` |
| `avg_title_length` | Mean title length | `df['title'].str.len().mean()` |
| `verified_purchase_pct` | % verified purchases | `df['verified'].mean() * 100` |

### Temporal Statistics

| Metric | Description |
|--------|-------------|
| `date_min` | Earliest interaction date |
| `date_max` | Latest interaction date |
| `date_range_days` | Total time span in days |
| Monthly aggregates | Interaction counts per month |

### Why These Metrics?
- **Scale understanding**: Users/items/interactions define matrix size
- **Rating distribution**: J-shaped distributions common in reviews (positive skew)
- **Temporal span**: Important for train/val/test temporal splits

---

## 4. Sparsity & K-Core Analysis

Computed by [`src/eda/sparsity_analysis.py`](../src/eda/sparsity_analysis.py).

### Sparsity Metrics

| Metric | Formula | Typical Value |
|--------|---------|---------------|
| **Density** | `Interactions / (Users × Items)` | 0.0001% - 0.01% |
| **Sparsity** | `1 - Density` | 99.99%+ |

> [!NOTE]
> Recommendation datasets are **extremely sparse**. A density of 0.001% means only 1 in 100,000 possible user-item pairs have an interaction.

### K-Core Filtering Simulation

The pipeline simulates k-core filtering for multiple thresholds:

| k | Effect |
|---|--------|
| 2 | Light filtering, removes one-time users/items |
| 3 | Moderate filtering |
| **5** | Recommended baseline for most analyses |
| 10 | Aggressive filtering, dense core only |
| 20 | Very aggressive, may lose significant data |

**Output metrics per k:**
- User retention %
- Item retention %
- Interaction retention %
- Post-filter sparsity

### Why K-Core?
- **Graph models (LATTICE, LightGCN)** require dense subgraphs for message passing
- **Cold-start entities** (users/items with 1-2 interactions) provide weak signal
- **Trade-off**: Higher k → better signal quality but smaller dataset

---

## 5. User & Item Behavior Analysis

Computed by [`src/eda/user_item_analysis.py`](../src/eda/user_item_analysis.py).

### Power-Law Distribution

Recommendation datasets follow **power-law distributions** (Zipf's Law):
- Few users have many interactions (power users)
- Most users have 1-2 interactions (casual users)

**Estimation Method**: Maximum Likelihood Estimation (Clauset et al., 2009)

| Metric | Description |
|--------|-------------|
| `power_law_alpha_users` | Power-law exponent for user activity |
| `power_law_alpha_items` | Power-law exponent for item popularity |

> Typical α values: 1.5-2.5 (higher → steeper long-tail)

### Cold-Start Analysis

| Metric | Definition |
|--------|------------|
| `cold_start_users_pct` | % users with < 5 interactions |
| `cold_start_items_pct` | % items with < 5 interactions |

> [!WARNING]
> Cold-start percentages of 80-95% are common, making multimodal features critical for new entity representation.

### Pareto Analysis (Interaction Concentration)

Measures how interactions are concentrated among top entities:

| Tier | Typical Contribution |
|------|---------------------|
| Top 1% users | 10-15% of interactions |
| Top 5% users | 25-35% of interactions |
| Top 10% users | 35-45% of interactions |
| Top 20% users | 50-60% of interactions |

**Why This Matters:**
- Popularity bias in training
- Top users/items dominate evaluation metrics
- Negative sampling needs to account for popularity

---

## 6. Multimodal Feature Analysis

Computed by [`src/eda/multimodal_analysis.py`](../src/eda/multimodal_analysis.py).

### Feature Coverage

| Feature | What It Measures |
|---------|------------------|
| `title_coverage` | % items with non-empty title |
| `description_coverage` | % items with description |
| `features_coverage` | % items with product features |
| `image_coverage` | % items with at least 1 image URL |
| `complete_coverage` | % items with ALL modalities |

### Image Statistics

| Metric | Description |
|--------|-------------|
| `items_with_images` | Count of items with image URLs |
| `avg_images_per_item` | Mean images per product |
| `url_validation_rate` | % of sampled URLs returning HTTP 200 |

### Why Multimodal Analysis?
- **LATTICE/MMGCN** require image features for all items
- Missing modalities require fallback strategies (zero padding, text-only)
- URL validation ensures images are still accessible

---

## 7. Visualizations

Generated by [`src/eda/visualizations.py`](../src/eda/visualizations.py).

### Plot Catalog

| Plot | File Pattern | Insight |
|------|--------------|---------|
| Rating Distribution | `rating_distribution_{dataset}.png` | J-shaped curve, positive skew typical |
| Interaction Frequency | `interaction_frequency_{dataset}.png` | Power-law verification (log-log linear) |
| Temporal Patterns | `temporal_patterns_{dataset}.png` | Seasonality, growth trends |
| Text Length | `text_length_{dataset}.png` | Preprocessing needs (truncation, padding) |
| Sparsity | `sparsity_{dataset}.png` | Visual scale of cold-start problem |
| Category Distribution | `category_distribution_{dataset}.png` | Dataset composition |
| Multimodal Coverage | `multimodal_coverage_{dataset}.png` | Feature completeness by modality |
| Modality Alignment | `modality_alignment_{dataset}.png` | Visual-interaction correlation |
| Visual Manifold | `visual_manifold_{dataset}.png` | UMAP projection of embeddings |
| BPR Hardness | `bpr_hardness_{dataset}.png` | Negative sampling difficulty |
| Semantic Alignment | `semantic_alignment_{dataset}.png` | Text-interaction correlation |
| Cross-Modal | `cross_modal_consistency_{dataset}.png` | Text vs image agreement |

---

## 8. Multimodal Feature Quality Analysis

These analyses assess whether the dataset is suitable for multimodal recommendation models (LATTICE, MICRO, DiffMM) based on recent research.

### 8.1 Image Embedding Extraction (CLIP)

**Source**: [`src/eda/embedding_extractor.py`](../src/eda/embedding_extractor.py)

**Model**: `openai/clip-vit-large-patch14` (768-dim, 75.4% ImageNet accuracy)

**Optimizations**:
- **Parallel downloads**: ThreadPoolExecutor with 16 workers
- **Batch prefetching**: Downloads next batch while GPU processes current
- **Batch size**: 128 for GPU efficiency

| Metric | Description |
|--------|-------------|
| `items_per_second` | Processing throughput |
| `success_rate` | % images successfully downloaded |
| `embedding_dim` | 768 (CLIP ViT-L/14) |

### 8.2 Modality-Interaction Alignment

**Source**: [`src/eda/modality_alignment.py`](../src/eda/modality_alignment.py)  
**Reference**: Liu et al. (2024)

**Hypothesis (Homophily)**: Visually similar items should share similar user interaction patterns.

**Method**:
1. Sample random item pairs
2. Compute **visual similarity** (CLIP embedding cosine similarity)
3. Compute **interaction similarity** (Jaccard similarity of user sets)
4. Calculate **Pearson/Spearman correlation**

| Metric | Interpretation |
|--------|----------------|
| Pearson r > 0.3 | Strong alignment, visual features are predictive |
| Pearson r ~ 0 | No alignment, visual features may not help |
| Pearson r < 0 | Negative alignment (unusual) |

### 8.3 Visual Manifold Structure

**Source**: [`src/eda/visual_manifold.py`](../src/eda/visual_manifold.py)  
**Reference**: Xu et al. (2025)

**Question**: Do CLIP embeddings form meaningful clusters by category?

**Method**:
1. Extract CLIP embeddings for items
2. Project to 2D using **UMAP** or **t-SNE**
3. Compute **Silhouette Score** by category labels

| Silhouette Score | Interpretation |
|------------------|----------------|
| > 0.5 | Strong clustering, visual features align with categories |
| 0.2-0.5 | Moderate clustering |
| < 0.2 | Weak clustering, categories may not be visually distinct |

### 8.4 BPR Hardness Assessment

**Source**: [`src/eda/bpr_hardness.py`](../src/eda/bpr_hardness.py)  
**Reference**: Xu et al. (2025)

**Question**: Is random negative sampling sufficient for BPR training?

**Method**:
1. For each user, identify positive items
2. Sample random negative items
3. Compute **visual distance** between positive and negative

| Category | Visual Distance | Meaning |
|----------|-----------------|---------|
| Easy | > 0.8 | Clearly different, trivial to distinguish |
| Medium | 0.3-0.8 | Somewhat similar, informative signal |
| Hard | < 0.3 | Very similar, challenging negatives |

> [!TIP]
> If >80% negatives are "Easy", random sampling is likely sufficient.  
> If many are "Hard", consider hard negative mining strategies.

### 8.5 Graph Connectivity Check

**Source**: [`src/eda/graph_connectivity.py`](../src/eda/graph_connectivity.py)  
**Reference**: Liu et al. (2024)

**Question**: Is the visual k-NN graph connected enough for LATTICE message passing?

**Method**:
1. Build k-NN graph from visual embeddings (k=5)
2. Find connected components
3. Measure **giant component coverage**

| Giant Component % | Verdict |
|-------------------|---------|
| > 50% | **PASS** - Graph is connected enough |
| < 50% | **FAIL** - Graph is fragmented |

### 8.6 Feature Collapse Detection

**Source**: [`src/eda/feature_collapse.py`](../src/eda/feature_collapse.py)  
**Reference**: Xu et al. (2025) - "White Wall Test"

**Question**: Has the encoder (e.g., ImageNet ResNet) experienced domain shift?

**Problem**: Product photos on white backgrounds may all map to similar embeddings if the encoder isn't domain-adapted.

**Method**:
1. Sample random pairs of embeddings
2. Compute average **cosine similarity**

| Avg Cosine Similarity | Verdict |
|-----------------------|---------|
| < 0.5 | **PASS** - Embeddings are diverse |
| 0.5-0.9 | **WARNING** - Some collapse detected |
| > 0.9 | **COLLAPSED** - All embeddings nearly identical |

---

## 9. Text Analysis (Semantic Gap Validation)

These analyses validate whether text descriptions predict user behavior.

### 9.1 Text Embedding Extraction (Sentence-BERT)

**Source**: [`src/eda/text_embedding_extractor.py`](../src/eda/text_embedding_extractor.py)

**Model**: `sentence-transformers/all-mpnet-base-v2` (768-dim, best quality general-purpose)

**Text Sources** (concatenated):
- Product title
- Description
- Features list

| Metric | Description |
|--------|-------------|
| `items_per_second` | Processing throughput |
| `avg_text_length` | Mean character length per item |
| `embedding_dim` | 768 (MPNet) |

### 9.2 Semantic-Interaction Alignment

**Source**: [`src/eda/semantic_alignment.py`](../src/eda/semantic_alignment.py)  
**Reference**: Liu et al. (2024)

**Hypothesis**: Items with similar text descriptions should have similar buyers.

**Method**:
1. Sample random item pairs
2. Compute **text similarity** (SBERT cosine similarity)
3. Compute **interaction similarity** (Jaccard of user sets)
4. Calculate **Pearson correlation**

| Pearson r | Signal Strength | Interpretation |
|-----------|-----------------|----------------|
| < 0.05 | **NOISE** | Text is noise/spammy, users don't read |
| 0.05-0.15 | **WEAK** | Some signal, consider fine-tuning |
| > 0.15 | **STRONG** | Text predicts purchases, proceed with text features |

### 9.3 Cross-Modal Consistency

**Source**: [`src/eda/cross_modal_consistency.py`](../src/eda/cross_modal_consistency.py)  
**Reference**: Liu et al. (2024) - "Modality Alignment" challenge

**Question**: Do text and image modalities agree?

**Problem**: If text says "Red Dress" but image looks like "Blue Shoe", multimodal models will have conflicting signals.

**Method**:
1. Project both embeddings to same dimension
2. Compute per-item cosine similarity
3. Report mean and distribution

| Mean Similarity | Status | Interpretation |
|-----------------|--------|----------------|
| < 0.3 | **DISAGREE** | Modalities conflict, use separate branches |
| 0.3-0.6 | **MODERATE** | Complementary info, late fusion recommended |
| > 0.6 | **AGREE** | Well-aligned, early fusion recommended |

---

## 10. Model Feasibility Decision

The EDA pipeline produces a final feasibility assessment for multimodal recommendations:

### Decision Criteria

| Check | Source | Pass Condition |
|-------|--------|----------------|
| **Alignment** | Modality-Interaction | Pearson r is not NaN and measurable |
| **Connectivity** | Graph k-NN | Giant component > 50% |
| **Collapse** | Feature Collapse | Avg cosine similarity < 0.5 |

### Decision Logic
```
IF all checks PASS → PROCEED with LATTICE
ELSE → STOP and investigate:
  - Connectivity fail → Use different encoder or denser k
  - Collapse fail → Use domain-adapted encoder (e.g., CLIP-Fashion)
  - Alignment fail → Visual features may not help for this domain
```

### Additional Signals from Text Analysis

| Signal | Recommendation |
|--------|----------------|
| Semantic alignment r < 0.05 | Deprioritize text encoder |
| Cross-modal mean < 0.3 | Use separate modality branches |
| Both strong | Full multimodal approach recommended |

---

## 11. Command-Line Interface

The main entry point is [`src/run_eda.py`](../src/run_eda.py).

### Basic Usage

```bash
# Single dataset, random sampling
python src/run_eda.py --dataset beauty --sample-ratio 0.01 --output docs/

# Single dataset with image download
python src/run_eda.py --dataset clothing --sample-ratio 0.01 --output docs/ --download-images

# All datasets, dense sampling with academic analysis
python src/run_eda.py --dataset all --sample-ratio 0.05 --sampling-strategy dense --kcore-k 5 --temporal-months 48 --academic-analysis
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | `beauty`, `clothing`, `electronics`, `both`, or `all` |
| `--sample-ratio` | 0.1 | Fraction for random sampling |
| `--sampling-strategy` | random | `random`, `kcore`, `temporal`, or `dense` |
| `--kcore-k` | 5 | K-core threshold |
| `--temporal-months` | 6 | Time window for temporal sampling |
| `--output` | docs/ | Output directory |
| `--download-images` | false | Download sample product images |
| `--image-sample-size` | 500 | Number of images to download |
| `--academic-analysis` | false | Run LATTICE feasibility checks + text analysis |
| `--seed` | 42 | Random seed |

### Output Files

| File | Description |
|------|-------------|
| `{dataset}_eda_results.json` | All computed statistics |
| `{dataset}_eda_report.md` | Human-readable markdown report |
| `figures/{dataset}/*.png` | Generated visualizations |
| `images/{dataset}/*.jpg` | Downloaded sample images (if enabled) |

---

## References

1. **Liu et al. (2024)** - LATTICE: Multimodal Graph Collaborative Filtering
2. **Xu et al. (2025)** - Feature Quality Analysis for Multimodal Recommendations
3. **Clauset et al. (2009)** - Power-law distributions in empirical data

---

*Documentation generated for the Amazon Review 2023 EDA Pipeline*
