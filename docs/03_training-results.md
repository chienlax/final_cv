# Training Results Report

Comprehensive evaluation of LATTICE, MICRO, and DiffMM across three Amazon domains.

## Configuration Summary

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 384 |
| GCN Layers | 3 |
| Batch Size | 1024 |
| Learning Rate | 5e-4 |
| Projection MLP | 768 → 1024 → 384 |
| Projection Dropout | 0.5 |

---

## Track 1: Warm Performance

Standard test set with items seen during training.

### Recall@K

| Dataset | Metric | LATTICE | MICRO | DiffMM | Best |
|---------|--------|---------|-------|--------|------|
| **Beauty** | R@10 | 3.81% | **4.80%** | 4.54% | MICRO |
| | R@20 | 6.07% | **7.10%** | 6.87% | MICRO |
| | R@50 | 10.71% | **11.96%** | 12.02% | DiffMM |
| **Clothing** | R@10 | 2.92% | **3.20%** | 3.13% | MICRO |
| | R@20 | 4.53% | **4.74%** | 4.72% | MICRO |
| | R@50 | 7.86% | **8.58%** | 8.29% | MICRO |
| **Electronics** | R@10 | 4.18% | 4.82% | **5.10%** | DiffMM |
| | R@20 | 6.55% | 7.43% | **7.88%** | DiffMM |
| | R@50 | 10.90% | 12.15% | **12.41%** | DiffMM |

### NDCG@K

| Dataset | Metric | LATTICE | MICRO | DiffMM |
|---------|--------|---------|-------|--------|
| **Beauty** | N@10 | 2.25% | **2.78%** | 2.64% |
| | N@20 | 2.86% | **3.41%** | 3.27% |
| **Clothing** | N@10 | 1.58% | **1.80%** | 1.68% |
| | N@20 | 2.02% | **2.22%** | 2.11% |
| **Electronics** | N@10 | 2.47% | 2.82% | **2.88%** |
| | N@20 | 3.11% | 3.53% | **3.62%** |

---

## Track 2: User Robustness

Performance segmented by user activity level.

### Sparse Users (≤5 interactions)

| Dataset | n_users | LATTICE R@20 | MICRO R@20 | DiffMM R@20 |
|---------|---------|--------------|------------|-------------|
| Beauty | 4,145 | 5.48% | **6.74%** | 6.61% |
| Clothing | 4,253 | 4.73% | 4.67% | **4.66%** |
| Electronics | 3,859 | 6.57% | **7.62%** | 8.08% |

### Active Users (≥20 interactions)

| Dataset | n_users | LATTICE R@20 | MICRO R@20 | DiffMM R@20 |
|---------|---------|--------------|------------|-------------|
| Beauty | 107 | **8.30%** | 7.77% | 6.98% |
| Clothing | 34 | 5.13% | **6.45%** | 5.71% |
| Electronics | 91 | 6.48% | **7.44%** | 8.13% |

> **Insight**: LATTICE performs better on active users in Beauty, suggesting k-NN graph benefits from rich user histories.

---

## Track 3: Cold-Start Performance (Inductive Mode)

Critical test: items **never seen** during training, embeddings from modal features only.

### Recall@K

| Dataset | Metric | LATTICE | MICRO | DiffMM |
|---------|--------|---------|-------|--------|
| **Beauty** | R@10 | 4.99% | **6.07%** | 0.67% ⚠️ |
| | R@20 | 8.11% | **9.25%** | 1.12% ⚠️ |
| | R@50 | 14.15% | **15.73%** | 2.85% ⚠️ |
| **Clothing** | R@10 | 4.49% | **4.56%** | 0.52% ⚠️ |
| | R@20 | 7.10% | **7.22%** | 1.13% ⚠️ |
| | R@50 | 12.35% | **13.22%** | 2.68% ⚠️ |
| **Electronics** | R@10 | 3.75% | **4.11%** | 0.64% ⚠️ |
| | R@20 | 6.59% | **6.84%** | 1.14% ⚠️ |
| | R@50 | 12.94% | **13.13%** | 2.78% ⚠️ |

### NDCG@K (Cold)

| Dataset | LATTICE N@20 | MICRO N@20 | DiffMM N@20 |
|---------|--------------|------------|-------------|
| Beauty | 4.05% | **4.80%** | 0.49% |
| Clothing | 3.55% | **3.69%** | 0.46% |
| Electronics | 3.05% | **3.31%** | 0.50% |

---

## Critical Finding: DiffMM Cold-Start Collapse

> [!CAUTION]
> DiffMM shows **~1% Recall@20** on cold items compared to **7-9%** for LATTICE/MICRO.

### Root Cause Hypothesis

The diffusion process in `sample_from_noise()` is not properly conditioning on modal features:

1. **MSI Network Failure**: The Modality Signal Injection may not be effectively guiding the denoising process
2. **Generation vs Retrieval Mismatch**: Generated embeddings may land in "empty" regions of the preference space
3. **Noise Schedule**: The linear beta schedule may not be appropriate for modal conditioning

### Recommended Investigation

```python
# Debug check: Compare generated vs projected embeddings
with torch.no_grad():
    modal_emb = model.get_modal_embeddings(cold_items)  # Direct projection
    gen_emb = model.sample_from_noise(modal_emb)        # Diffusion output
    
    # Cosine similarity should be high if conditioning works
    cos_sim = F.cosine_similarity(modal_emb, gen_emb).mean()
    print(f"Modal-Generated similarity: {cos_sim:.4f}")  # Should be > 0.5
```

---

## Model Comparison Summary

| Criterion | Winner | Notes |
|-----------|--------|-------|
| **Warm Recall** | DiffMM/MICRO | DiffMM wins on Electronics |
| **Cold Recall** | **MICRO** | Clear winner across all datasets |
| **Sparse Users** | MICRO | Better generalization |
| **Active Users** | Mixed | LATTICE competitive |
| **Training Speed** | LATTICE | Fewer auxiliary losses |
| **Cold-Start Ready** | **MICRO** | Only reliable choice |

---

## Recommendations

1. **Production Deployment**: Use **MICRO** for cold-start critical applications
2. **Further Investigation**: Debug DiffMM's MSI mechanism before cold-start deployment
3. **Ensemble Potential**: LATTICE + MICRO ensemble may outperform either alone
4. **Ablation Needed**: Test visual-only vs text-only to understand modality contribution
