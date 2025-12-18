"""
Anisotropy Analysis Module.

Implements the "Signal Crisis" fix by detecting and correcting the "Cone Effect"
in pre-trained embeddings (CLIP/BERT). High average cosine similarity (~0.5) 
indicates embeddings occupy a narrow cone, which breaks distance-based models.

References:
- Ethayarajh, 2019: "How Contextual are Contextualized Word Representations?"
- Timkey & van Schijndel, 2021: "All Bark and No Bite: Rogue Dimensions in Transformers"
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnisotropyResult:
    """Result of anisotropy analysis."""
    
    # Before centering
    avg_cosine_before: float = 0.0
    std_cosine_before: float = 0.0
    
    # After centering
    avg_cosine_after: float = 0.0
    std_cosine_after: float = 0.0
    
    # Analysis
    n_pairs_sampled: int = 0
    is_anisotropic: bool = False  # True if avg_before > 0.4
    improvement_ratio: float = 0.0  # (before - after) / before
    
    # Interpretation
    interpretation: str = ""
    recommendation: str = ""
    
    def to_dict(self) -> dict:
        return {
            "before_centering": {
                "avg_cosine": self.avg_cosine_before,
                "std_cosine": self.std_cosine_before,
            },
            "after_centering": {
                "avg_cosine": self.avg_cosine_after,
                "std_cosine": self.std_cosine_after,
            },
            "n_pairs_sampled": self.n_pairs_sampled,
            "is_anisotropic": self.is_anisotropic,
            "improvement_ratio": self.improvement_ratio,
            "interpretation": self.interpretation,
            "recommendation": self.recommendation,
        }


def center_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Apply isotropic post-processing by subtracting mean vector.
    
    This simple technique can significantly improve embedding quality
    by centering the distribution around the origin.
    
    Args:
        embeddings: Array of shape [n_items, embedding_dim].
        
    Returns:
        Centered embeddings of same shape.
    """
    mean_vec = embeddings.mean(axis=0)
    return embeddings - mean_vec


def compute_pairwise_cosine_sample(
    embeddings: np.ndarray,
    n_pairs: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute cosine similarity for a random sample of pairs.
    
    Avoids O(n²) computation for large datasets.
    
    Args:
        embeddings: Array of shape [n_items, embedding_dim].
        n_pairs: Number of random pairs to sample.
        seed: Random seed.
        
    Returns:
        Array of cosine similarities.
    """
    rng = np.random.default_rng(seed)
    n_items = len(embeddings)
    
    if n_items < 2:
        return np.array([])
    
    # Generate random pairs (without replacement if possible)
    max_pairs = n_items * (n_items - 1) // 2
    n_pairs = min(n_pairs, max_pairs)
    
    # Sample random indices
    idx_a = rng.integers(0, n_items, size=n_pairs)
    idx_b = rng.integers(0, n_items, size=n_pairs)
    
    # Ensure pairs are different
    same_mask = idx_a == idx_b
    idx_b[same_mask] = (idx_b[same_mask] + 1) % n_items
    
    # Compute cosine similarities
    emb_a = embeddings[idx_a]
    emb_b = embeddings[idx_b]
    
    # Normalize
    norm_a = np.linalg.norm(emb_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(emb_b, axis=1, keepdims=True)
    
    # Avoid division by zero
    norm_a = np.maximum(norm_a, 1e-8)
    norm_b = np.maximum(norm_b, 1e-8)
    
    # Cosine similarity
    cos_sim = np.sum((emb_a / norm_a) * (emb_b / norm_b), axis=1)
    
    return cos_sim


def analyze_anisotropy(
    embeddings: np.ndarray,
    n_pairs: int = 10000,
    anisotropy_threshold: float = 0.4,
    seed: int = 42,
) -> AnisotropyResult:
    """
    Analyze feature anisotropy and test if centering helps.
    
    The "Cone Effect": Pre-trained encoders (CLIP, BERT) often map inputs
    to a narrow cone in embedding space. This causes:
    - High average cosine similarity (~0.5)
    - Distance metrics dominated by this shared direction
    - Poor performance for distance-based models (LATTICE, MICRO)
    
    Fix: Subtract the mean vector to center embeddings around origin.
    
    Args:
        embeddings: Array of shape [n_items, embedding_dim].
        n_pairs: Number of random pairs to sample for analysis.
        anisotropy_threshold: Avg cosine above this indicates anisotropy.
        seed: Random seed.
        
    Returns:
        AnisotropyResult with before/after statistics.
    """
    result = AnisotropyResult()
    result.n_pairs_sampled = n_pairs
    
    logger.info("Analyzing embedding anisotropy...")
    
    if len(embeddings) < 2:
        result.interpretation = "Not enough embeddings for analysis."
        return result
    
    # Before centering
    logger.info("  Computing cosine similarities (before centering)...")
    cos_before = compute_pairwise_cosine_sample(embeddings, n_pairs, seed)
    result.avg_cosine_before = float(np.mean(cos_before))
    result.std_cosine_before = float(np.std(cos_before))
    
    # After centering
    logger.info("  Centering embeddings (subtracting mean)...")
    centered_emb = center_embeddings(embeddings)
    
    logger.info("  Computing cosine similarities (after centering)...")
    cos_after = compute_pairwise_cosine_sample(centered_emb, n_pairs, seed)
    result.avg_cosine_after = float(np.mean(cos_after))
    result.std_cosine_after = float(np.std(cos_after))
    
    # Analysis
    result.is_anisotropic = result.avg_cosine_before > anisotropy_threshold
    
    if result.avg_cosine_before > 0:
        result.improvement_ratio = (
            (result.avg_cosine_before - result.avg_cosine_after) 
            / result.avg_cosine_before
        )
    
    # Interpretation
    if result.is_anisotropic:
        if result.avg_cosine_after < 0.1:
            result.interpretation = (
                f"ANISOTROPIC: Avg cosine = {result.avg_cosine_before:.3f} (>{anisotropy_threshold}). "
                f"Centering FIXED the issue: after centering = {result.avg_cosine_after:.3f}. "
                "Embeddings were in a narrow cone but centering spread them out."
            )
            result.recommendation = (
                "Apply mean centering to all embeddings before using in MRS models. "
                "This should significantly improve LATTICE/MICRO performance."
            )
        else:
            result.interpretation = (
                f"ANISOTROPIC: Avg cosine = {result.avg_cosine_before:.3f} (>{anisotropy_threshold}). "
                f"Centering PARTIALLY helped: after = {result.avg_cosine_after:.3f}. "
                "Embeddings have some intrinsic similarity that persists after centering."
            )
            result.recommendation = (
                "Consider PCA/ZCA whitening or using a different encoder "
                "(e.g., CLIP fine-tuned on fashion data). "
                "Current embeddings may have limited discriminative power."
            )
    else:
        result.interpretation = (
            f"ISOTROPIC: Avg cosine = {result.avg_cosine_before:.3f} (<{anisotropy_threshold}). "
            "Embeddings have good spread in the vector space. "
            "No centering needed."
        )
        result.recommendation = (
            "Embeddings look healthy. Proceed with LATTICE/MICRO training."
        )
    
    logger.info(f"  Before centering: avg={result.avg_cosine_before:.4f}, std={result.std_cosine_before:.4f}")
    logger.info(f"  After centering:  avg={result.avg_cosine_after:.4f}, std={result.std_cosine_after:.4f}")
    logger.info(f"  Is anisotropic: {result.is_anisotropic}")
    
    return result


def get_centered_embeddings(
    embeddings: np.ndarray,
    item_indices: dict[str, int],
) -> Tuple[np.ndarray, dict[str, int]]:
    """
    Convenience function to get centered embeddings with indices.
    
    Use this to pre-process embeddings before other analyses.
    
    Args:
        embeddings: Original embeddings.
        item_indices: Mapping from item_id to index.
        
    Returns:
        Tuple of (centered embeddings, same item_indices).
    """
    return center_embeddings(embeddings), item_indices
