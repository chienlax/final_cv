"""
User Consistency Analysis Module.

Implements Interaction Homophily check to measure whether users buy 
visually similar items. This tests if visual features have predictive
power for user preferences.

If users buy random-looking items (local_dist ≈ global_dist), 
then visual MRS models like DiffMM will fail.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)


@dataclass
class UserConsistencyResult:
    """Result of user consistency analysis."""
    
    n_users_analyzed: int = 0
    n_users_with_enough_items: int = 0
    min_items_threshold: int = 5
    
    # Distance metrics
    mean_local_distance: float = 0.0   # Avg pairwise distance within user
    std_local_distance: float = 0.0
    mean_global_distance: float = 0.0  # Avg pairwise distance overall
    std_global_distance: float = 0.0
    
    # Consistency ratio
    consistency_ratio: float = 0.0  # local / global (< 1 means consistent)
    is_consistent: bool = False
    
    # Distribution
    users_with_visual_coherence_pct: float = 0.0  # % users with local < global
    
    # Interpretation
    interpretation: str = ""
    recommendation: str = ""
    
    def to_dict(self) -> dict:
        return {
            "n_users_analyzed": self.n_users_analyzed,
            "n_users_with_enough_items": self.n_users_with_enough_items,
            "distances": {
                "mean_local": self.mean_local_distance,
                "std_local": self.std_local_distance,
                "mean_global": self.mean_global_distance,
                "std_global": self.std_global_distance,
            },
            "consistency_ratio": self.consistency_ratio,
            "is_consistent": self.is_consistent,
            "users_with_visual_coherence_pct": self.users_with_visual_coherence_pct,
            "interpretation": self.interpretation,
            "recommendation": self.recommendation,
        }


def compute_user_local_distance(
    user_items: list[str],
    embeddings: np.ndarray,
    item_indices: dict[str, int],
) -> Optional[float]:
    """
    Compute average pairwise visual distance for items bought by a user.
    
    Args:
        user_items: List of item IDs bought by the user.
        embeddings: Embedding matrix.
        item_indices: Mapping from item_id to index.
        
    Returns:
        Mean pairwise distance, or None if not enough items.
    """
    # Get indices for items with embeddings
    valid_indices = [item_indices[item] for item in user_items if item in item_indices]
    
    if len(valid_indices) < 2:
        return None
    
    # Get embeddings
    user_embeddings = embeddings[valid_indices]
    
    # Compute pairwise distances (cosine distance = 1 - cosine_similarity)
    # Using 'cosine' metric from pdist
    distances = pdist(user_embeddings, metric='cosine')
    
    return float(np.mean(distances))


def calculate_user_consistency(
    interactions_df: pd.DataFrame,
    embeddings: np.ndarray,
    item_indices: dict[str, int],
    n_users: int = 1000,
    min_items_per_user: int = 5,
    global_sample_size: int = 5000,
    seed: int = 42,
) -> UserConsistencyResult:
    """
    Measure if users buy visually similar items (Interaction Homophily).
    
    Logic:
    - For each sampled user, get their interacted items
    - Compute average pairwise visual distance (using pdist)
    - Compare to global random baseline
    
    If local_dist < global_dist, visual features have predictive power
    for user preferences. This validates visual MRS approaches.
    
    Args:
        interactions_df: DataFrame with 'user_id' and 'item_id' columns.
        embeddings: Visual embeddings matrix (n_items, embedding_dim).
        item_indices: Dictionary mapping item_id to embedding index.
        n_users: Number of users to sample for analysis.
        min_items_per_user: Minimum items required per user.
        global_sample_size: Number of random pairs for global baseline.
        seed: Random seed.
        
    Returns:
        UserConsistencyResult with local vs global comparison.
    """
    rng = np.random.default_rng(seed)
    result = UserConsistencyResult()
    result.min_items_threshold = min_items_per_user
    
    logger.info("Analyzing user consistency (interaction homophily)...")
    
    # Get user interaction counts
    user_items_map = interactions_df.groupby('user_id')['item_id'].apply(list).to_dict()
    
    # Filter users with enough items
    eligible_users = [
        uid for uid, items in user_items_map.items()
        if len(items) >= min_items_per_user
    ]
    result.n_users_with_enough_items = len(eligible_users)
    
    if len(eligible_users) == 0:
        result.interpretation = f"No users with >= {min_items_per_user} items found."
        return result
    
    # Sample users
    sample_size = min(n_users, len(eligible_users))
    sampled_users = rng.choice(eligible_users, size=sample_size, replace=False)
    result.n_users_analyzed = sample_size
    
    logger.info(f"  Sampling {sample_size} users (from {len(eligible_users)} eligible)...")
    
    # Compute local distances for each user
    local_distances = []
    for user_id in sampled_users:
        user_items = user_items_map[user_id]
        dist = compute_user_local_distance(user_items, embeddings, item_indices)
        if dist is not None:
            local_distances.append(dist)
    
    if len(local_distances) == 0:
        result.interpretation = "No users have enough items with embeddings."
        return result
    
    result.mean_local_distance = float(np.mean(local_distances))
    result.std_local_distance = float(np.std(local_distances))
    
    logger.info(f"  Computed local distances for {len(local_distances)} users")
    
    # Compute global baseline (random pairs)
    logger.info(f"  Computing global baseline ({global_sample_size} random pairs)...")
    
    all_items = list(item_indices.keys())
    n_items = len(all_items)
    
    global_distances = []
    pairs_computed = 0
    
    while pairs_computed < global_sample_size:
        idx_a = rng.integers(0, n_items)
        idx_b = rng.integers(0, n_items)
        
        if idx_a == idx_b:
            continue
        
        emb_a = embeddings[idx_a]
        emb_b = embeddings[idx_b]
        
        # Cosine distance
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        
        if norm_a > 0 and norm_b > 0:
            cos_sim = np.dot(emb_a, emb_b) / (norm_a * norm_b)
            dist = 1 - cos_sim
            global_distances.append(dist)
            pairs_computed += 1
    
    result.mean_global_distance = float(np.mean(global_distances))
    result.std_global_distance = float(np.std(global_distances))
    
    # Compute consistency ratio
    if result.mean_global_distance > 0:
        result.consistency_ratio = result.mean_local_distance / result.mean_global_distance
    
    result.is_consistent = result.consistency_ratio < 1.0
    
    # Compute percentage of users with visual coherence
    users_coherent = sum(1 for d in local_distances if d < result.mean_global_distance)
    result.users_with_visual_coherence_pct = users_coherent / len(local_distances) * 100
    
    # Interpretation
    if result.is_consistent:
        if result.consistency_ratio < 0.8:
            result.interpretation = (
                f"STRONG CONSISTENCY: Users buy visually similar items. "
                f"Local distance ({result.mean_local_distance:.3f}) << "
                f"Global distance ({result.mean_global_distance:.3f}). "
                f"Ratio = {result.consistency_ratio:.3f}. "
                f"{result.users_with_visual_coherence_pct:.1f}% of users show coherent visual preferences."
            )
            result.recommendation = (
                "Visual features have strong predictive power. "
                "DiffMM and LATTICE should work well on this dataset."
            )
        else:
            result.interpretation = (
                f"MODERATE CONSISTENCY: Some visual preference signal exists. "
                f"Local ({result.mean_local_distance:.3f}) < "
                f"Global ({result.mean_global_distance:.3f}). "
                f"Ratio = {result.consistency_ratio:.3f}."
            )
            result.recommendation = (
                "Visual features have some predictive power. "
                "Consider combining with text features for better performance."
            )
    else:
        result.interpretation = (
            f"NO CONSISTENCY: Users buy visually random items. "
            f"Local ({result.mean_local_distance:.3f}) >= "
            f"Global ({result.mean_global_distance:.3f}). "
            f"Ratio = {result.consistency_ratio:.3f}."
        )
        result.recommendation = (
            "Visual features do NOT predict user preferences. "
            "DiffMM/LATTICE will struggle. Consider text-only models or "
            "fine-tuning visual encoder on this domain."
        )
    
    logger.info(f"  Local distance: {result.mean_local_distance:.4f} ± {result.std_local_distance:.4f}")
    logger.info(f"  Global distance: {result.mean_global_distance:.4f} ± {result.std_global_distance:.4f}")
    logger.info(f"  Consistency ratio: {result.consistency_ratio:.4f}")
    logger.info(f"  Is consistent: {result.is_consistent}")
    
    return result
