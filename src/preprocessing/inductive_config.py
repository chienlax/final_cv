"""
Inductive preprocessing configuration for cold-start multimodal recommendation.

Optimized for RTX 3060 (12GB VRAM) and 40GB RAM.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class InductivePreprocessConfig:
    """Configuration for inductive cold-start preprocessing."""
    
    # Topological Subsampling
    seed_users: int = 10000       # Target: ~12k-15k users, ~8k-10k items
    min_total_nodes: int = 8000   # Below = overfitting risk, increase seed
    max_total_nodes: int = 20000  # Above = OOM risk, decrease seed
    k_core: int = 5               # Minimum interactions per user/item
    
    # Inductive Split (Cold Items)
    cold_item_ratio: float = 0.20  # 20% items held out as cold
    
    # Warm Interaction Split
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_warm_ratio: float = 0.10
    
    # Feature Extraction (optimized for RTX 3060)
    feature_batch_size: int = 64    # GPU batch size (uses ~3GB extra VRAM)
    n_download_workers: int = 16    # Parallel image downloads
    download_timeout: int = 10      # Per-image timeout in seconds
    clip_model: str = "openai/clip-vit-large-patch14"
    sbert_model: str = "sentence-transformers/all-mpnet-base-v2"
    text_columns: list = field(default_factory=lambda: ["title", "description", "features"])
    max_text_length: int = 512
    
    # Hardware
    num_workers: int = 6  # P-cores only on i5-13500
    
    # Reproducibility
    seed: int = 2024
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data/raw"))
    output_dir: Path = field(default_factory=lambda: Path("data/processed"))
    
    # Dataset selection
    dataset: str = "electronics"
    
    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.cold_item_ratio < 1, "cold_item_ratio must be in (0, 1)"
        assert abs(self.train_ratio + self.val_ratio + self.test_warm_ratio - 1.0) < 1e-6
        assert self.min_total_nodes < self.max_total_nodes
        
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
    
    @property
    def interaction_file(self) -> Path:
        """Get path to 5-core filtered interaction CSV."""
        files = {
            "electronics": "Electronics.csv",
            "beauty": "Beauty_and_Personal_Care.csv",
            "clothing": "Clothing_Shoes_and_Jewelry.csv",
        }
        if self.dataset not in files:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        return self.data_dir / files[self.dataset]
    
    @property
    def metadata_file(self) -> Path:
        """Get path to metadata JSONL.gz file."""
        files = {
            "electronics": "meta_Electronics.jsonl.gz",
            "beauty": "meta_Beauty_and_Personal_Care.jsonl.gz",
            "clothing": "meta_Clothing_Shoes_and_Jewelry.jsonl.gz",
        }
        if self.dataset not in files:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        return self.data_dir / files[self.dataset]
    
    @property
    def dataset_output_dir(self) -> Path:
        """Get output directory for this dataset."""
        return self.output_dir / self.dataset
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "seed_users": self.seed_users,
            "min_total_nodes": self.min_total_nodes,
            "max_total_nodes": self.max_total_nodes,
            "k_core": self.k_core,
            "cold_item_ratio": self.cold_item_ratio,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_warm_ratio": self.test_warm_ratio,
            "seed": self.seed,
            "dataset": self.dataset,
            "clip_model": self.clip_model,
            "sbert_model": self.sbert_model,
            "text_columns": self.text_columns,
        }
