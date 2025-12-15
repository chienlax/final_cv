"""
Preprocessing configuration for multimodal recommendation system.

Defines configuration parameters for:
- K-core filtering thresholds
- Train/val/test split ratios
- Sampling parameters
- Output paths
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline."""
    
    # K-core filtering
    k_core: int = 5  # Minimum interactions per user/item
    
    # Data splitting
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_by_time: bool = True  # If True, split by timestamp; else random
    
    # Sampling for development
    sample_ratio: float = 1.0  # 1.0 = use full dataset
    
    # Negative sampling
    n_negatives: int = 4  # Negatives per positive for BPR
    negative_strategy: str = "uniform"  # "uniform" or "popularity"
    
    # Random seed
    random_seed: int = 42
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("data/processed"))
    
    # Dataset selection
    dataset: str = "beauty"  # "beauty" or "clothing"
    
    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.k_core <= 100, "k_core must be between 1 and 100"
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        assert 0 < self.sample_ratio <= 1.0, "sample_ratio must be in (0, 1]"
        assert self.negative_strategy in ["uniform", "popularity"], \
            "negative_strategy must be 'uniform' or 'popularity'"
        
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
    
    @property
    def interaction_file(self) -> Path:
        """Get path to interaction data file."""
        if self.dataset == "beauty":
            return self.data_dir / "Beauty_and_Personal_Care.jsonl.gz"
        elif self.dataset == "clothing":
            return self.data_dir / "Clothing_Shoes_and_Jewelry.jsonl.gz"
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
    
    @property
    def metadata_file(self) -> Path:
        """Get path to metadata file."""
        if self.dataset == "beauty":
            return self.data_dir / "meta_Beauty_and_Personal_Care.jsonl.gz"
        elif self.dataset == "clothing":
            return self.data_dir / "meta_Clothing_Shoes_and_Jewelry.jsonl.gz"
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "k_core": self.k_core,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "split_by_time": self.split_by_time,
            "sample_ratio": self.sample_ratio,
            "n_negatives": self.n_negatives,
            "negative_strategy": self.negative_strategy,
            "random_seed": self.random_seed,
            "dataset": self.dataset,
        }
