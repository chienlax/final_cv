"""
Preprocessing Module for Multimodal Recommendation System.

Provides preprocessing utilities including:
- K-core filtering
- ID encoding
- Train/val/test splits
- Negative sampling for BPR
"""

from .config import PreprocessConfig
from .pipeline import (
    apply_kcore_filter,
    create_id_mappings,
    create_train_val_test_split,
    save_processed_data,
    load_processed_data,
)
from .negative_sampling import (
    sample_negatives_uniform,
    sample_negatives_popularity,
    create_bpr_triplets,
)

__all__ = [
    "PreprocessConfig",
    "apply_kcore_filter",
    "create_id_mappings",
    "create_train_val_test_split",
    "save_processed_data",
    "load_processed_data",
    "sample_negatives_uniform",
    "sample_negatives_popularity",
    "create_bpr_triplets",
]
