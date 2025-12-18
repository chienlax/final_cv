# Model Setup and Configuration

Comprehensive documentation for the LATTICE/MICRO/DiffMM multimodal recommendation system.

## Overview

This project implements three state-of-the-art multimodal recommendation models with **inductive cold-start evaluation support**:

| Model | Key Technique | Cold-Start Strategy |
|-------|--------------|---------------------|
| **LATTICE** | k-NN graph learning from modal features | Modal embeddings → GCN propagation |
| **MICRO** | InfoNCE contrastive modal alignment | Fused visual+text embeddings |
| **DiffMM** | Diffusion process with MSI | Noise sampling + modal conditioning |

---

## 1. Data Preprocessing

### 1.1 Pipeline Overview

```
Raw CSV (5-core) → Seed Sampling → Recursive k-Core → Warm/Cold Split → ID Remapping → Feature Extraction
```

### 1.2 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed_users` | 10,000 | Initial user sample size |
| `min_total_nodes` | 8,000 | Increase seed if below (overfitting risk) |
| `max_total_nodes` | 20,000 | Decrease seed if above (OOM risk) |
| `k_core` | 5 | Min interactions per user/item |
| `cold_item_ratio` | 0.20 | Fraction of items held as cold |
| `train_ratio` | 0.80 | Warm interactions for training |
| `val_ratio` | 0.10 | Warm interactions for validation |
| `test_warm_ratio` | 0.10 | Warm interactions for warm test |

### 1.3 Warm/Cold Item Split

**Block ID Structure:**
- Warm Items: IDs `[0, N_warm - 1]`
- Cold Items: IDs `[N_warm, N_total - 1]`

> **Critical**: Cold items NEVER appear in training data. Their embeddings must come purely from multimodal features.

### 1.4 Feature Extraction (Optimized)

**Visual Features (CLIP):**
- Model: `openai/clip-vit-large-patch14`
- Output: 768-dimensional embeddings
- **Parallel downloads:** 16 workers (ThreadPoolExecutor)
- **Batch size:** 64 (uses ~10GB VRAM)
- **Prefetch queue:** Background thread downloads while GPU processes

**Text Features (SBERT):**
- Model: `sentence-transformers/all-mpnet-base-v2`
- Input: `title + description + features`
- Output: 768-dimensional embeddings
- **Batch size:** 64

**Anisotropy Correction:**
```python
x = x / norm(x)              # L2 normalize
mu = x[:n_warm].mean(axis=0) # Mean from WARM items only
x = x - mu                    # Center
x = x / norm(x)              # Re-normalize
```

### 1.5 Output Files

```
data/processed/{dataset}/
├── train.txt          # user_idx item_idx (warm items only)
├── val.txt            # user_idx item_idx (warm items only)
├── test_warm.txt      # user_idx item_idx (warm items only)
├── test_cold.txt      # user_idx item_idx (cold items only)
├── feat_visual.npy    # (N_items, 768) float32
├── feat_text.npy      # (N_items, 768) float32
├── maps.json          # ID mappings + metadata
└── stat.txt           # Statistics for paper reporting
```

---

## 2. Model Architecture

### 2.1 Common Configuration

All models share these **hardware-optimized** hyperparameters (the "Ferrari" upgrade 🏎️):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `BATCH_SIZE` | 1024 | Smaller batches = gradient noise = implicit regularization |
| `EPOCHS` | 100 | With early stopping |
| `PATIENCE` | 100 | Generous for generative models |
| `LR` | 5e-4 | Lower LR for deeper model |
| `L2_REG` | 1e-3 | Strong weight decay for high param-to-data ratio |
| `EMBED_DIM` | 384 | Divisible by attention heads (6, 12) |
| `N_LAYERS` | 3 | Sweet spot (4 → oversmoothing) |
| `N_NEGATIVES` | 64 | Statistically sufficient, VRAM-friendly |

### 2.2 Modality Projection (MLP Bridge)

> **Why MLP instead of Linear?** The mapping from "Visual Space" to "Preference Space" is NOT linear! A red dress (visual) correlates with "party wear" (preference) through complex patterns.

**Architecture:**
```
768 (CLIP/SBERT) → 1024 (hidden) → LeakyReLU → Dropout(0.5) → 384 (embed_dim)
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `PROJECTION_HIDDEN_DIM` | 1024 | Wide hidden layer for non-linear mapping |
| `PROJECTION_DROPOUT` | 0.5 | Aggressive regularization to prevent feature memorization |

These parameters are **shared across all items** = better generalization to cold items.

### 2.3 LATTICE Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LATTICE_K` | 40 | k-NN neighbors (broader semantic neighborhoods) |
| `LATTICE_LAMBDA` | 0.5 | Balance: original vs learned graph |

### 2.4 MICRO Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MICRO_TAU` | 0.2 | InfoNCE temperature (lower = sharper) |
| `MICRO_ALPHA` | 0.1 | Contrastive aux loss weight |

### 2.5 DiffMM Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DIFFMM_STEPS` | 100 | Diffusion steps (high precision) |
| `DIFFMM_NOISE_SCALE` | 0.1 | Base noise level |
| `DIFFMM_LAMBDA_MSI` | 1e-2 | MSI loss weight |
| `DIFFMM_MLP_WIDTH` | 512 | Denoising network width (compute sink) |

---

## 3. Training Pipeline

### 3.1 Loss Functions

**BPR Loss (all models):**
```python
loss = -log(sigmoid(score_pos - score_neg)).mean()
```

**Auxiliary Losses:**
- MICRO: InfoNCE for visual-text-ID alignment
- DiffMM: Diffusion denoising MSE

### 3.2 Negative Sampling

- Strategy: **Uniform random**
- Negatives per positive: 1

### 3.3 Early Stopping

- Monitor: Recall@20 on validation set
- Patience: 20 epochs

---

## 4. Evaluation Protocol

### 4.1 Three-Track Evaluation

| Track | Split | Filter | Purpose |
|-------|-------|--------|---------|
| 1 | test_warm | All users | Standard performance |
| 2a | test_warm | Sparse (≤5 interactions) | User robustness |
| 2b | test_warm | Active (≥20 interactions) | User robustness |
| 3 | test_cold | All users (inductive mode) | Cold-start ability |

### 4.2 Metrics

- **Recall@K** (K=10, 20, 50)
- **NDCG@K** (K=10, 20, 50)
- **Precision@K** (K=10, 20, 50)

### 4.3 Inductive Mode

For Track 3 (cold items), models use:
- **LATTICE/MICRO**: Modal embeddings only (no ID lookup)
- **DiffMM**: Sample from noise conditioned on modal features

---

## 5. Project Structure

```
src/
├── common/
│   ├── config.py          # Unified hyperparameters
│   └── utils.py           # Seed, device, early stopping
├── preprocessing/
│   ├── inductive_config.py
│   ├── inductive_pipeline.py
│   └── run_preprocessing.py
├── models/
│   ├── base.py            # Abstract base class
│   ├── lattice.py         # LATTICE implementation
│   ├── micro.py           # MICRO implementation
│   └── diffmm.py          # DiffMM implementation
├── dataset.py             # DataLoader, adjacency matrix
├── trainer.py             # Training loop
├── evaluator.py           # Metrics computation
└── main.py                # CLI entry point
```
