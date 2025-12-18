"""
Unified configuration for LATTICE/MICRO/DiffMM training.

Hardware-optimized for RTX 3060 (12GB VRAM).
"""


class Config:
    """Unified training configuration for fair model comparison."""
    
    # =========================================================================
    # HARDWARE & PATHS
    # =========================================================================
    DEVICE = "cuda"
    SEED = 2024
    DATA_PATH = "data/processed/"
    
    # =========================================================================
    # UNIVERSAL TRAINING PARAMS (Fair Comparison)
    # =========================================================================
    BATCH_SIZE = 1024      # Safe for 3060, agreed by both papers
    EPOCHS = 100
    PATIENCE = 20          # Early stopping (generative models have noisy loss)
    LR = 1e-3              # Industry standard
    L2_REG = 1e-4          # Balance: 1e-3 hurts cold-start, 1e-5 overfits
    
    # =========================================================================
    # MODEL ARCHITECTURE (Hardware-Optimized)
    # =========================================================================
    EMBED_DIM = 64         # Both papers agree
    N_LAYERS = 2           # Sweet spot for VRAM (3 layers → ~11GB with kNN)
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    TOP_K = [10, 20, 50]   # Recall@K, NDCG@K, Precision@K
    EVAL_BATCH_SIZE = 256
    
    # =========================================================================
    # NEGATIVE SAMPLING
    # =========================================================================
    N_NEGATIVES = 1        # 1 hard negative per positive (BPR)
    NEGATIVE_STRATEGY = "uniform"  # Not popularity-weighted
    
    # =========================================================================
    # LATTICE SPECIFICS
    # =========================================================================
    LATTICE_K = 10         # k-NN neighbors for graph learning
    LATTICE_LAMBDA = 0.5   # Balance original vs learned graph
    
    # =========================================================================
    # MICRO SPECIFICS
    # =========================================================================
    MICRO_TAU = 0.2        # InfoNCE temperature (0.2 > 0.5 for graphs)
    MICRO_ALPHA = 0.1      # Auxiliary loss weight (don't overpower BPR)
    
    # =========================================================================
    # DiffMM SPECIFICS
    # =========================================================================
    DIFFMM_STEPS = 5       # Fast diffusion (100 unnecessary for graphs)
    DIFFMM_NOISE_SCALE = 0.1
    DIFFMM_LAMBDA_MSI = 1e-2  # Modality Signal Injection weight
