"""
Modality-Interaction Alignment Analysis.

Implements the "Homophily Hypothesis" check from Liu et al. (2024):
- Do visually similar items share similar interaction patterns?

This module computes:
1. Visual Similarity: Cosine similarity of CLIP embeddings
2. Interaction Similarity: Jaccard similarity of user interaction sets
3. Correlation analysis between visual and interaction similarities
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ModalityAlignmentResult:
    """Results from modality-interaction alignment analysis."""
    
    n_pairs_sampled: int = 0
    pearson_correlation: float = 0.0
    pearson_pvalue: float = 1.0
    spearman_correlation: float = 0.0
    spearman_pvalue: float = 1.0
    
    # Statistics
    visual_sim_mean: float = 0.0
    visual_sim_std: float = 0.0
    interaction_sim_mean: float = 0.0
    interaction_sim_std: float = 0.0
    
    # Interpretation
    interpretation: str = ""
    
    # Raw data for plotting
    visual_similarities: list[float] = field(default_factory=list)
    interaction_similarities: list[float] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_pairs_sampled": self.n_pairs_sampled,
            "pearson": {
                "correlation": round(self.pearson_correlation, 4),
                "pvalue": self.pearson_pvalue,
            },
            "spearman": {
                "correlation": round(self.spearman_correlation, 4),
                "pvalue": self.spearman_pvalue,
            },
            "visual_similarity": {
                "mean": round(self.visual_sim_mean, 4),
                "std": round(self.visual_sim_std, 4),
            },
            "interaction_similarity": {
                "mean": round(self.interaction_sim_mean, 4),
                "std": round(self.interaction_sim_std, 4),
            },
            "interpretation": self.interpretation,
        }


def compute_interaction_similarity_jaccard(
    item_users: dict[str, set[str]],
    item_pairs: list[tuple[str, str]],
) -> np.ndarray:
    """
    Compute Jaccard similarity of user interaction sets for item pairs.
    
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    
    Args:
        item_users: Dictionary mapping item_id to set of user_ids who interacted.
        item_pairs: List of (item_id_1, item_id_2) pairs.
        
    Returns:
        Array of Jaccard similarities for each pair.
    """
    similarities = []
    
    for item_a, item_b in item_pairs:
        users_a = item_users.get(item_a, set())
        users_b = item_users.get(item_b, set())
        
        if len(users_a) == 0 and len(users_b) == 0:
            # Both have no interactions
            similarities.append(0.0)
        else:
            intersection = len(users_a & users_b)
            union = len(users_a | users_b)
            jaccard = intersection / union if union > 0 else 0.0
            similarities.append(jaccard)
    
    return np.array(similarities)


def compute_visual_similarity_cosine(
    embeddings: np.ndarray,
    item_indices: dict[str, int],
    item_pairs: list[tuple[str, str]],
) -> np.ndarray:
    """
    Compute cosine similarity between item embeddings for given pairs.
    
    Args:
        embeddings: Array of shape [n_items, embedding_dim].
        item_indices: Dictionary mapping item_id to embedding index.
        item_pairs: List of (item_id_1, item_id_2) pairs.
        
    Returns:
        Array of cosine similarities for each pair.
    """
    similarities = []
    
    for item_a, item_b in item_pairs:
        idx_a = item_indices.get(item_a)
        idx_b = item_indices.get(item_b)
        
        if idx_a is None or idx_b is None:
            similarities.append(np.nan)
            continue
        
        emb_a = embeddings[idx_a]
        emb_b = embeddings[idx_b]
        
        # Cosine similarity
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        
        if norm_a == 0 or norm_b == 0:
            similarities.append(0.0)
        else:
            cos_sim = np.dot(emb_a, emb_b) / (norm_a * norm_b)
            similarities.append(float(cos_sim))
    
    return np.array(similarities)


def sample_item_pairs(
    item_ids: list[str],
    n_pairs: int = 1000,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """
    Sample random item pairs for similarity analysis.
    
    Args:
        item_ids: List of item IDs.
        n_pairs: Number of pairs to sample.
        seed: Random seed.
        
    Returns:
        List of (item_id_1, item_id_2) tuples.
    """
    np.random.seed(seed)
    
    n_items = len(item_ids)
    if n_items < 2:
        return []
    
    # Sample random pairs (without replacement for indices)
    pairs = []
    max_attempts = n_pairs * 10
    attempts = 0
    
    while len(pairs) < n_pairs and attempts < max_attempts:
        i = np.random.randint(0, n_items)
        j = np.random.randint(0, n_items)
        
        if i != j:
            pair = (item_ids[i], item_ids[j])
            # Normalize pair order to avoid duplicates
            normalized = (min(pair), max(pair))
            if normalized not in pairs:
                pairs.append(normalized)
        
        attempts += 1
    
    return pairs


def analyze_modality_alignment(
    interactions_df: pd.DataFrame,
    embeddings: np.ndarray,
    item_indices: dict[str, int],
    n_pairs: int = 1000,
    seed: int = 42,
) -> ModalityAlignmentResult:
    """
    Analyze alignment between visual modality and interaction patterns.
    
    Implements the Homophily Hypothesis check from Liu et al. (2024):
    Do visually similar items share similar interaction patterns?
    
    Args:
        interactions_df: DataFrame with 'user_id' and 'item_id' columns.
        embeddings: Array of shape [n_items, embedding_dim] with CLIP embeddings.
        item_indices: Dictionary mapping item_id to embedding index.
        n_pairs: Number of item pairs to sample for analysis.
        seed: Random seed.
        
    Returns:
        ModalityAlignmentResult with correlation statistics.
    """
    logger.info(f"Analyzing modality-interaction alignment with {n_pairs} pairs...")
    
    result = ModalityAlignmentResult()
    
    # Build item -> users mapping
    item_users = interactions_df.groupby("item_id")["user_id"].apply(set).to_dict()
    
    # Get items that have both embeddings and interactions
    items_with_embeddings = set(item_indices.keys())
    items_with_interactions = set(item_users.keys())
    valid_items = list(items_with_embeddings & items_with_interactions)
    
    logger.info(f"  Items with both embeddings and interactions: {len(valid_items)}")
    
    if len(valid_items) < 10:
        logger.warning("  Not enough items with both embeddings and interactions")
        result.interpretation = "Insufficient data for analysis"
        return result
    
    # Sample item pairs
    item_pairs = sample_item_pairs(valid_items, n_pairs=n_pairs, seed=seed)
    result.n_pairs_sampled = len(item_pairs)
    
    logger.info(f"  Sampled {len(item_pairs)} item pairs")
    
    # Compute visual similarities
    visual_sims = compute_visual_similarity_cosine(embeddings, item_indices, item_pairs)
    
    # Compute interaction similarities
    interaction_sims = compute_interaction_similarity_jaccard(item_users, item_pairs)
    
    # Remove pairs with NaN values
    valid_mask = ~(np.isnan(visual_sims) | np.isnan(interaction_sims))
    visual_sims = visual_sims[valid_mask]
    interaction_sims = interaction_sims[valid_mask]
    
    if len(visual_sims) < 10:
        logger.warning("  Not enough valid pairs after filtering")
        result.interpretation = "Insufficient valid pairs for analysis"
        return result
    
    # Store for plotting
    result.visual_similarities = visual_sims.tolist()
    result.interaction_similarities = interaction_sims.tolist()
    
    # Compute statistics
    result.visual_sim_mean = float(np.mean(visual_sims))
    result.visual_sim_std = float(np.std(visual_sims))
    result.interaction_sim_mean = float(np.mean(interaction_sims))
    result.interaction_sim_std = float(np.std(interaction_sims))
    
    # Correlation analysis
    pearson_r, pearson_p = stats.pearsonr(visual_sims, interaction_sims)
    spearman_r, spearman_p = stats.spearmanr(visual_sims, interaction_sims)
    
    result.pearson_correlation = float(pearson_r)
    result.pearson_pvalue = float(pearson_p)
    result.spearman_correlation = float(spearman_r)
    result.spearman_pvalue = float(spearman_p)
    
    # Interpretation
    if pearson_p > 0.05:
        result.interpretation = "No significant correlation - visual features may not align with user preferences"
    elif abs(pearson_r) < 0.1:
        result.interpretation = "Very weak correlation - visual signal exists but is minimal"
    elif abs(pearson_r) < 0.3:
        result.interpretation = "Weak positive correlation - some visual-interaction alignment exists"
    elif abs(pearson_r) < 0.5:
        result.interpretation = "Moderate correlation - visual features have meaningful signal"
    else:
        result.interpretation = "Strong correlation - visual features strongly predict interaction patterns"
    
    logger.info(f"  Pearson r = {pearson_r:.4f} (p = {pearson_p:.4f})")
    logger.info(f"  Interpretation: {result.interpretation}")
    
    return result
