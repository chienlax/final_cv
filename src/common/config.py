"""
Unified configuration for LATTICE/MICRO/DiffMM training.

Hardware-optimized for:
- CPU: i5-13500 (6P+8E = 20 threads)
- RAM: 64GB
- GPU: RTX 3060 (12GB VRAM)

Target VRAM usage: 8-10GB (leave headroom for peaks)
"""


class Config:
    """Unified training configuration for fair model comparison."""
    
    # =========================================================================
    # HARDWARE & PATHS
    # =========================================================================
    DEVICE = "cuda"
    SEED = 42
    DATA_PATH = "data/processed/"
    
    # =========================================================================
    # DATALOADER (Optimized for i5-13500)
    # =========================================================================
    NUM_WORKERS = 6          # P-cores for parallel data loading
    PIN_MEMORY = True        # Faster CPU→GPU transfer
    PREFETCH_FACTOR = 4      # Batches to prefetch per worker
    PERSISTENT_WORKERS = True  # Avoid worker restart overhead
    
    # =========================================================================
    # MIXED PRECISION (AMP)
    # =========================================================================
    USE_AMP = True           # ~25% VRAM savings, ~15% speedup
    
    # =========================================================================
    # UNIVERSAL TRAINING PARAMS
    # =========================================================================
    # Aggressive GPU utilization for 12GB VRAM
    BATCH_SIZE = 4096        # Doubled - more GPU parallelism
    EPOCHS = 200
    PATIENCE = 30            # Early stopping
    LR = 1e-3                # Industry standard
    L2_REG = 1e-4            # Regularization
    LR_SCHEDULER = "cosine"  # Cosine annealing
    
    # =========================================================================
    # MODEL ARCHITECTURE (Increased for GPU utilization)
    # =========================================================================
    EMBED_DIM = 128          # Doubled from 64 → 4x more computation
    N_LAYERS = 4             # Increased from 3 → more GCN passes
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    TOP_K = [10, 20, 50]     # Recall@K, NDCG@K, Precision@K
    EVAL_BATCH_SIZE = 4096   # Large (no gradients during eval)
    
    # =========================================================================
    # NEGATIVE SAMPLING
    # =========================================================================
    N_NEGATIVES = 8          # Increased from 4 → more negative pairs
    NEGATIVE_STRATEGY = "uniform"
    
    # =========================================================================
    # LATTICE SPECIFICS
    # =========================================================================
    LATTICE_K = 20           # Increased from 10 → denser k-NN graph
    LATTICE_LAMBDA = 0.5     # Balance original vs learned graph
    
    # =========================================================================
    # MICRO SPECIFICS
    # =========================================================================
    MICRO_TAU = 0.2          # InfoNCE temperature
    MICRO_ALPHA = 0.1        # Contrastive loss weight
    
    # =========================================================================
    # DiffMM SPECIFICS
    # =========================================================================
    DIFFMM_STEPS = 10        # Increased from 5 → better generation quality
    DIFFMM_NOISE_SCALE = 0.1
    DIFFMM_LAMBDA_MSI = 1e-2
