"""
Cross-Modal Consistency Analysis.

Checks whether Text and Image modalities agree for the same items.

Reference: Liu et al. (2024) - "Modality Alignment" challenge in MRS.

If text says "Red Dress" but image embedding looks like "Blue Shoe",
the multimodal model will have conflicting signals.

Method:
1. Project both modalities to same dimension
2. Compute per-item cosine similarity
3. Report statistics and flag misaligned items

Interpretation:
- Avg sim < 0.3: Modalities DISAGREE (red flag)
- Avg sim 0.3-0.6: Moderate agreement
- Avg sim > 0.6: Strong agreement
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class CrossModalResult:
    """Results from cross-modal consistency analysis."""
    
    n_items_analyzed: int = 0
    n_items_with_both: int = 0
    
    # Similarity statistics
    mean_similarity: float = 0.0
    std_similarity: float = 0.0
    median_similarity: float = 0.0
    min_similarity: float = 0.0
    max_similarity: float = 0.0
    
    # Distribution buckets
    pct_low_agreement: float = 0.0    # < 0.3
    pct_moderate_agreement: float = 0.0  # 0.3-0.6
    pct_high_agreement: float = 0.0   # > 0.6
    
    # Projection info
    projection_method: str = ""
    text_dim: int = 0
    image_dim: int = 0
    projected_dim: int = 0
    
    # Interpretation
    alignment_status: str = ""  # "disagree", "moderate", "agree"
    interpretation: str = ""
    recommendation: str = ""
    
    # Per-item similarities for plotting
    similarities: list[float] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_items_analyzed": self.n_items_analyzed,
            "n_items_with_both": self.n_items_with_both,
            "statistics": {
                "mean": round(self.mean_similarity, 4),
                "std": round(self.std_similarity, 4),
                "median": round(self.median_similarity, 4),
                "min": round(self.min_similarity, 4),
                "max": round(self.max_similarity, 4),
            },
            "distribution": {
                "low_agreement_pct": round(self.pct_low_agreement, 1),
                "moderate_agreement_pct": round(self.pct_moderate_agreement, 1),
                "high_agreement_pct": round(self.pct_high_agreement, 1),
            },
            "projection": {
                "method": self.projection_method,
                "text_dim": self.text_dim,
                "image_dim": self.image_dim,
                "projected_dim": self.projected_dim,
            },
            "alignment_status": self.alignment_status,
            "interpretation": self.interpretation,
            "recommendation": self.recommendation,
        }


def project_to_common_dim(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    method: str = "linear",
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Project two embedding matrices to the same dimension.
    
    Args:
        embeddings_a: First embedding matrix (n_items, dim_a).
        embeddings_b: Second embedding matrix (n_items, dim_b).
        method: Projection method ('linear', 'truncate', or 'pad').
        
    Returns:
        Tuple of (projected_a, projected_b, common_dim).
    """
    dim_a = embeddings_a.shape[1]
    dim_b = embeddings_b.shape[1]
    
    if dim_a == dim_b:
        return embeddings_a, embeddings_b, dim_a
    
    if method == "truncate":
        # Truncate to smaller dimension
        common_dim = min(dim_a, dim_b)
        return embeddings_a[:, :common_dim], embeddings_b[:, :common_dim], common_dim
    
    elif method == "pad":
        # Pad smaller to larger dimension
        common_dim = max(dim_a, dim_b)
        if dim_a < common_dim:
            padding = np.zeros((embeddings_a.shape[0], common_dim - dim_a))
            embeddings_a = np.hstack([embeddings_a, padding])
        if dim_b < common_dim:
            padding = np.zeros((embeddings_b.shape[0], common_dim - dim_b))
            embeddings_b = np.hstack([embeddings_b, padding])
        return embeddings_a, embeddings_b, common_dim
    
    elif method == "linear":
        # Learn linear projection from larger to smaller
        common_dim = min(dim_a, dim_b)
        
        if dim_a > dim_b:
            # Project A to B's dimension using random projection
            np.random.seed(42)
            projection = np.random.randn(dim_a, common_dim).astype(np.float32)
            projection = projection / np.linalg.norm(projection, axis=0)
            embeddings_a = embeddings_a @ projection
        else:
            np.random.seed(42)
            projection = np.random.randn(dim_b, common_dim).astype(np.float32)
            projection = projection / np.linalg.norm(projection, axis=0)
            embeddings_b = embeddings_b @ projection
        
        # Re-normalize after projection
        embeddings_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
        embeddings_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)
        
        return embeddings_a, embeddings_b, common_dim
    
    else:
        raise ValueError(f"Unknown projection method: {method}")


def interpret_cross_modal(mean_sim: float) -> tuple[str, str, str]:
    """
    Interpret cross-modal consistency score.
    
    Args:
        mean_sim: Mean cosine similarity between modalities.
        
    Returns:
        Tuple of (status, interpretation, recommendation).
    """
    if np.isnan(mean_sim):
        return (
            "unknown",
            "Unable to compute cross-modal similarity",
            "Check data quality",
        )
    
    if mean_sim < 0.3:
        return (
            "disagree",
            f"LOW cross-modal agreement (avg={mean_sim:.3f}): Text and image embeddings "
            "point in different directions. This indicates a fundamental mismatch - "
            "either descriptions don't match images, or encoders have domain shift.",
            "Investigate: (1) Check if product images match descriptions, "
            "(2) Fine-tune encoders on domain, (3) Use separate modality branches.",
        )
    elif mean_sim < 0.6:
        return (
            "moderate",
            f"MODERATE cross-modal agreement (avg={mean_sim:.3f}): Some alignment exists "
            "but modalities capture different aspects. This is common for complementary "
            "information (text = features, image = style).",
            "Proceed with multimodal fusion. Late fusion may work better than joint embedding.",
        )
    else:
        return (
            "agree",
            f"HIGH cross-modal agreement (avg={mean_sim:.3f}): Text and image embeddings "
            "are well-aligned! Multimodal model will have consistent signal.",
            "Full confidence in joint multimodal approach. Early fusion recommended.",
        )


def analyze_cross_modal_consistency(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    item_indices_text: dict[str, int],
    item_indices_image: dict[str, int],
    projection_method: str = "linear",
) -> CrossModalResult:
    """
    Analyze consistency between text and image modalities.
    
    Computes per-item similarity between text and image embeddings
    to detect modality misalignment.
    
    Args:
        text_embeddings: Text embeddings (n_text, text_dim).
        image_embeddings: Image embeddings (n_image, image_dim).
        item_indices_text: Mapping item_id -> text embedding index.
        item_indices_image: Mapping item_id -> image embedding index.
        projection_method: How to handle different dimensions.
        
    Returns:
        CrossModalResult with consistency statistics.
    """
    logger.info("Analyzing cross-modal consistency (text vs image)...")
    
    result = CrossModalResult()
    result.projection_method = projection_method
    result.text_dim = text_embeddings.shape[1] if len(text_embeddings) > 0 else 0
    result.image_dim = image_embeddings.shape[1] if len(image_embeddings) > 0 else 0
    
    if len(text_embeddings) == 0 or len(image_embeddings) == 0:
        result.interpretation = "Missing embeddings for one or both modalities"
        return result
    
    # Find common items
    common_items = set(item_indices_text.keys()) & set(item_indices_image.keys())
    result.n_items_analyzed = len(set(item_indices_text.keys()) | set(item_indices_image.keys()))
    result.n_items_with_both = len(common_items)
    
    logger.info(f"  Items with text: {len(item_indices_text)}")
    logger.info(f"  Items with image: {len(item_indices_image)}")
    logger.info(f"  Items with BOTH: {result.n_items_with_both}")
    
    if result.n_items_with_both < 100:
        result.interpretation = f"Insufficient common items ({result.n_items_with_both}). Need ≥100."
        return result
    
    # Extract embeddings for common items
    common_items = sorted(common_items)
    text_emb_common = np.array([text_embeddings[item_indices_text[item]] for item in common_items])
    image_emb_common = np.array([image_embeddings[item_indices_image[item]] for item in common_items])
    
    # Project to common dimension
    logger.info(f"  Projecting to common dimension (method={projection_method})...")
    text_proj, image_proj, common_dim = project_to_common_dim(
        text_emb_common, image_emb_common, method=projection_method
    )
    result.projected_dim = common_dim
    logger.info(f"  Projected: text={result.text_dim} → {common_dim}, image={result.image_dim} → {common_dim}")
    
    # Compute per-item cosine similarity
    logger.info("  Computing per-item similarities...")
    similarities = np.sum(text_proj * image_proj, axis=1)  # Dot product of normalized vectors
    
    # Store for plotting
    result.similarities = similarities.tolist()
    
    # Statistics
    result.mean_similarity = float(np.mean(similarities))
    result.std_similarity = float(np.std(similarities))
    result.median_similarity = float(np.median(similarities))
    result.min_similarity = float(np.min(similarities))
    result.max_similarity = float(np.max(similarities))
    
    # Distribution buckets
    result.pct_low_agreement = float(np.mean(similarities < 0.3) * 100)
    result.pct_moderate_agreement = float(np.mean((similarities >= 0.3) & (similarities < 0.6)) * 100)
    result.pct_high_agreement = float(np.mean(similarities >= 0.6) * 100)
    
    # Interpretation
    result.alignment_status, result.interpretation, result.recommendation = \
        interpret_cross_modal(result.mean_similarity)
    
    logger.info(f"  Mean similarity: {result.mean_similarity:.4f}")
    logger.info(f"  Status: {result.alignment_status.upper()}")
    
    return result
