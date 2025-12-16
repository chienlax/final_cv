"""
Visual Manifold Structure Analysis.

Implements visual embedding analysis from Xu et al. (2025):
- UMAP/t-SNE projection of image embeddings
- Cluster visualization by category and rating

This helps determine if image features form meaningful clusters
that could benefit multimodal recommendation.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("umap-learn not available, will use t-SNE only")

try:
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available for manifold analysis")


@dataclass
class VisualManifoldResult:
    """Results from visual manifold structure analysis."""
    
    n_items: int = 0
    method: str = "umap"  # "umap" or "tsne"
    
    # Projection coordinates
    projection_x: list[float] = field(default_factory=list)
    projection_y: list[float] = field(default_factory=list)
    
    # Labels for coloring
    categories: list[str] = field(default_factory=list)
    ratings: list[float] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)
    
    # Quality metrics
    silhouette_score_category: float = 0.0
    n_unique_categories: int = 0
    
    # Interpretation
    interpretation: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_items": self.n_items,
            "method": self.method,
            "quality": {
                "silhouette_score_category": round(self.silhouette_score_category, 4),
                "n_unique_categories": self.n_unique_categories,
            },
            "interpretation": self.interpretation,
            # Note: projection coordinates are large, only include summary
            "projection_summary": {
                "x_range": [min(self.projection_x), max(self.projection_x)] if self.projection_x else [0, 0],
                "y_range": [min(self.projection_y), max(self.projection_y)] if self.projection_y else [0, 0],
            }
        }


def project_embeddings_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "cosine",
    seed: int = 42,
) -> np.ndarray:
    """
    Project high-dimensional embeddings to 2D using UMAP.
    
    Args:
        embeddings: Array of shape [n_items, embedding_dim].
        n_neighbors: Number of neighbors for UMAP.
        min_dist: Minimum distance between points.
        n_components: Number of output dimensions.
        metric: Distance metric.
        seed: Random seed.
        
    Returns:
        Array of shape [n_items, 2] with 2D coordinates.
    """
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn is required for UMAP projection")
    
    logger.info(f"  Running UMAP projection on {len(embeddings)} embeddings...")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=seed,
    )
    
    projection = reducer.fit_transform(embeddings)
    
    return projection


def project_embeddings_tsne(
    embeddings: np.ndarray,
    perplexity: float = 30.0,
    n_components: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """
    Project high-dimensional embeddings to 2D using t-SNE.
    
    Args:
        embeddings: Array of shape [n_items, embedding_dim].
        perplexity: Perplexity parameter for t-SNE.
        n_components: Number of output dimensions.
        seed: Random seed.
        
    Returns:
        Array of shape [n_items, 2] with 2D coordinates.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for t-SNE projection")
    
    logger.info(f"  Running t-SNE projection on {len(embeddings)} embeddings...")
    
    # Adjust perplexity if necessary
    n_samples = len(embeddings)
    if perplexity >= n_samples:
        perplexity = max(5.0, n_samples / 5)
        logger.warning(f"  Adjusted perplexity to {perplexity} due to small sample size")
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    
    projection = tsne.fit_transform(embeddings)
    
    return projection


def analyze_visual_manifold(
    metadata_df: pd.DataFrame,
    embeddings: np.ndarray,
    item_indices: dict[str, int],
    method: Literal["umap", "tsne"] = "umap",
    max_items: int = 5000,
    seed: int = 42,
) -> VisualManifoldResult:
    """
    Analyze visual manifold structure of item embeddings.
    
    Projects embeddings to 2D and analyzes cluster quality by category.
    
    Args:
        metadata_df: DataFrame with 'item_id', 'main_category', 'average_rating' columns.
        embeddings: Array of shape [n_items, embedding_dim] with CLIP embeddings.
        item_indices: Dictionary mapping item_id to embedding index.
        method: Projection method ("umap" or "tsne").
        max_items: Maximum items to project (for performance).
        seed: Random seed.
        
    Returns:
        VisualManifoldResult with projection and quality metrics.
    """
    logger.info(f"Analyzing visual manifold structure using {method.upper()}...")
    
    result = VisualManifoldResult(method=method)
    
    if not SKLEARN_AVAILABLE:
        result.interpretation = "scikit-learn not available for manifold analysis"
        return result
    
    # Get items with both embeddings and metadata
    items_with_embeddings = set(item_indices.keys())
    valid_items = metadata_df[metadata_df["item_id"].isin(items_with_embeddings)].copy()
    
    if len(valid_items) < 10:
        result.interpretation = "Not enough items with embeddings"
        return result
    
    # Sample if too many items
    if len(valid_items) > max_items:
        logger.info(f"  Sampling {max_items} items from {len(valid_items)}")
        valid_items = valid_items.sample(n=max_items, random_state=seed)
    
    result.n_items = len(valid_items)
    
    # Get embeddings for valid items
    valid_item_ids = valid_items["item_id"].tolist()
    valid_indices = [item_indices[item_id] for item_id in valid_item_ids]
    valid_embeddings = embeddings[valid_indices]
    
    # Project to 2D
    try:
        if method == "umap" and UMAP_AVAILABLE:
            projection = project_embeddings_umap(valid_embeddings, seed=seed)
        else:
            if method == "umap":
                logger.warning("  UMAP not available, falling back to t-SNE")
            projection = project_embeddings_tsne(valid_embeddings, seed=seed)
            result.method = "tsne"
    except Exception as e:
        logger.error(f"  Projection failed: {e}")
        result.interpretation = f"Projection failed: {str(e)}"
        return result
    
    # Store results
    result.projection_x = projection[:, 0].tolist()
    result.projection_y = projection[:, 1].tolist()
    result.item_ids = valid_item_ids
    result.categories = valid_items["main_category"].fillna("Unknown").tolist()
    result.ratings = valid_items["average_rating"].fillna(0.0).tolist()
    
    # Compute silhouette score for category clustering
    unique_categories = list(set(result.categories))
    result.n_unique_categories = len(unique_categories)
    
    if result.n_unique_categories > 1 and result.n_unique_categories < len(result.categories):
        try:
            # Encode categories to integers
            le = LabelEncoder()
            category_labels = le.fit_transform(result.categories)
            
            # Silhouette score measures cluster quality
            result.silhouette_score_category = float(silhouette_score(
                projection, category_labels, metric="euclidean"
            ))
            
            logger.info(f"  Silhouette score (category): {result.silhouette_score_category:.4f}")
        except Exception as e:
            logger.warning(f"  Could not compute silhouette score: {e}")
    
    # Interpretation
    if result.silhouette_score_category > 0.5:
        result.interpretation = "Strong visual clustering by category - good for multimodal recommendation"
    elif result.silhouette_score_category > 0.25:
        result.interpretation = "Moderate visual clustering - visual features capture some category structure"
    elif result.silhouette_score_category > 0:
        result.interpretation = "Weak visual clustering - pre-trained features may need fine-tuning"
    else:
        result.interpretation = "No meaningful visual clustering - visual features may not align with categories"
    
    logger.info(f"  Interpretation: {result.interpretation}")
    
    return result
