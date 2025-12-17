"""
Feature Collapse Detection for Visual Embeddings.

Implements the "White Wall Test" from Xu et al. (2025):
- Measure average cosine similarity of random embedding pairs
- Detect if encoder has mapped all items to narrow cone (Feature Collapse)

Pass Criterion: Average cosine similarity < 0.5
Collapse Signal: Average cosine similarity > 0.9
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CollapseResult:
    """Results from feature collapse detection."""
    
    n_items: int = 0
    n_pairs_sampled: int = 0
    
    # Cosine similarity statistics
    avg_cosine_similarity: float = 0.0
    std_cosine_similarity: float = 0.0
    min_cosine_similarity: float = 0.0
    max_cosine_similarity: float = 0.0
    median_cosine_similarity: float = 0.0
    
    # Distribution buckets
    pct_very_low: float = 0.0    # < 0.1 (orthogonal)
    pct_low: float = 0.0         # 0.1 - 0.3
    pct_medium: float = 0.0      # 0.3 - 0.5
    pct_high: float = 0.0        # 0.5 - 0.9
    pct_very_high: float = 0.0   # > 0.9 (near-identical)
    
    # Pass/Fail
    pass_threshold: float = 0.5
    collapse_threshold: float = 0.9
    is_pass: bool = False
    is_collapsed: bool = False
    interpretation: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_items": self.n_items,
            "n_pairs_sampled": self.n_pairs_sampled,
            "statistics": {
                "mean": round(self.avg_cosine_similarity, 4),
                "std": round(self.std_cosine_similarity, 4),
                "min": round(self.min_cosine_similarity, 4),
                "max": round(self.max_cosine_similarity, 4),
                "median": round(self.median_cosine_similarity, 4),
            },
            "distribution": {
                "very_low_pct": round(self.pct_very_low, 2),
                "low_pct": round(self.pct_low, 2),
                "medium_pct": round(self.pct_medium, 2),
                "high_pct": round(self.pct_high, 2),
                "very_high_pct": round(self.pct_very_high, 2),
            },
            "thresholds": {
                "pass": self.pass_threshold,
                "collapse": self.collapse_threshold,
            },
            "is_pass": self.is_pass,
            "is_collapsed": self.is_collapsed,
            "interpretation": self.interpretation,
        }


def analyze_feature_collapse(
    embeddings: np.ndarray,
    n_pairs: int = 10000,
    pass_threshold: float = 0.5,
    collapse_threshold: float = 0.9,
    seed: int = 42,
) -> CollapseResult:
    """
    Analyze feature collapse via random pair cosine similarity.
    
    The "White Wall Test": If an encoder (e.g., ImageNet ResNet on white-background
    product photos) experiences domain shift, it may map all items to a narrow
    cone in embedding space. This causes cosine similarity to be dominated by
    floating-point noise rather than semantic differences.
    
    Args:
        embeddings: Visual embeddings matrix (n_items, embedding_dim).
        n_pairs: Number of random pairs to sample.
        pass_threshold: Maximum avg similarity to pass (default 0.5).
        collapse_threshold: Avg similarity indicating collapse (default 0.9).
        seed: Random seed for reproducibility.
        
    Returns:
        CollapseResult with analysis results.
    """
    logger.info(f"Analyzing feature collapse (sampling {n_pairs} pairs)...")
    
    result = CollapseResult(
        n_items=len(embeddings),
        pass_threshold=pass_threshold,
        collapse_threshold=collapse_threshold,
    )
    
    if len(embeddings) < 2:
        result.interpretation = "Insufficient items for collapse analysis"
        return result
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = embeddings / norms
    
    # Sample random pairs
    rng = np.random.default_rng(seed)
    n_items = len(embeddings)
    max_pairs = n_items * (n_items - 1) // 2
    actual_pairs = min(n_pairs, max_pairs)
    
    # Generate unique random pairs
    if actual_pairs < max_pairs // 10:
        # Sparse sampling: generate random indices
        idx1 = rng.integers(0, n_items, size=actual_pairs)
        idx2 = rng.integers(0, n_items, size=actual_pairs)
        # Ensure different indices
        same_mask = idx1 == idx2
        idx2[same_mask] = (idx2[same_mask] + 1) % n_items
    else:
        # Dense sampling: use all pairs (up to limit)
        pairs = []
        for i in range(n_items):
            for j in range(i + 1, n_items):
                pairs.append((i, j))
                if len(pairs) >= actual_pairs:
                    break
            if len(pairs) >= actual_pairs:
                break
        pairs = np.array(pairs)
        idx1, idx2 = pairs[:, 0], pairs[:, 1]
    
    result.n_pairs_sampled = len(idx1)
    
    # Compute cosine similarities (dot product of normalized vectors)
    similarities = np.sum(normalized[idx1] * normalized[idx2], axis=1)
    
    # Compute statistics
    result.avg_cosine_similarity = float(np.mean(similarities))
    result.std_cosine_similarity = float(np.std(similarities))
    result.min_cosine_similarity = float(np.min(similarities))
    result.max_cosine_similarity = float(np.max(similarities))
    result.median_cosine_similarity = float(np.median(similarities))
    
    # Compute distribution buckets
    n = len(similarities)
    result.pct_very_low = float(np.sum(similarities < 0.1) / n * 100)
    result.pct_low = float(np.sum((similarities >= 0.1) & (similarities < 0.3)) / n * 100)
    result.pct_medium = float(np.sum((similarities >= 0.3) & (similarities < 0.5)) / n * 100)
    result.pct_high = float(np.sum((similarities >= 0.5) & (similarities < 0.9)) / n * 100)
    result.pct_very_high = float(np.sum(similarities >= 0.9) / n * 100)
    
    # Pass/Fail determination
    result.is_pass = result.avg_cosine_similarity < pass_threshold
    result.is_collapsed = result.avg_cosine_similarity >= collapse_threshold
    
    # Generate interpretation
    if result.is_collapsed:
        result.interpretation = (
            f"FAIL (COLLAPSED): Avg cosine similarity = {result.avg_cosine_similarity:.3f} "
            f"(threshold: {collapse_threshold}). Features have collapsed to a narrow cone. "
            "Visual encoder likely unsuitable (domain shift). "
            "Recommendation: Switch to CLIP-Fashion or domain-specific encoder."
        )
    elif result.is_pass:
        result.interpretation = (
            f"PASS: Avg cosine similarity = {result.avg_cosine_similarity:.3f} "
            f"(threshold: {pass_threshold}). Features show good variance. "
            "Visual encoder is producing discriminative embeddings."
        )
    else:
        result.interpretation = (
            f"WARNING: Avg cosine similarity = {result.avg_cosine_similarity:.3f} "
            f"(pass: <{pass_threshold}, collapse: >{collapse_threshold}). "
            "Features show moderate similarity. May work but suboptimal. "
            "Consider testing with alternative visual encoder."
        )
    
    logger.info(f"  Avg cosine similarity: {result.avg_cosine_similarity:.4f}")
    logger.info(f"  Distribution: very_low={result.pct_very_low:.1f}%, very_high={result.pct_very_high:.1f}%")
    logger.info(f"  Result: {'PASS' if result.is_pass else ('COLLAPSED' if result.is_collapsed else 'WARNING')}")
    
    return result


def compute_embedding_variance(embeddings: np.ndarray) -> dict:
    """
    Compute embedding variance metrics.
    
    Low variance indicates potential feature collapse.
    
    Args:
        embeddings: Visual embeddings matrix.
        
    Returns:
        Dictionary with variance metrics.
    """
    # Per-dimension variance
    dim_variances = np.var(embeddings, axis=0)
    
    # Overall metrics
    avg_dim_variance = float(np.mean(dim_variances))
    min_dim_variance = float(np.min(dim_variances))
    max_dim_variance = float(np.max(dim_variances))
    
    # Count "dead" dimensions (near-zero variance)
    dead_dims = int(np.sum(dim_variances < 1e-6))
    
    # Effective dimensionality (PCA-based estimate)
    # Using ratio of sum of variances squared to sum of squared variances
    sum_var = np.sum(dim_variances)
    sum_var_sq = np.sum(dim_variances ** 2)
    effective_dim = (sum_var ** 2) / sum_var_sq if sum_var_sq > 0 else 0
    
    return {
        "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
        "avg_dim_variance": round(avg_dim_variance, 6),
        "min_dim_variance": round(min_dim_variance, 6),
        "max_dim_variance": round(max_dim_variance, 6),
        "dead_dimensions": dead_dims,
        "effective_dimensionality": round(effective_dim, 2),
    }
