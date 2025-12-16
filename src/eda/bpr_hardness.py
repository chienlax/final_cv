"""
BPR Hardness Assessment.

Implements negative sampling analysis from Xu et al. (2025):
- Analyze visual distance between positive and negative items
- Identify "easy" vs "hard" negatives for BPR training

A key insight is that random negatives are often visually distinct
from positives, making BPR learning trivial and ineffective for
capturing style/visual preferences.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BPRHardnessResult:
    """Results from BPR hardness analysis."""
    
    n_users_analyzed: int = 0
    n_pairs_analyzed: int = 0
    
    # Visual distance statistics (positive vs negative)
    mean_visual_distance: float = 0.0
    median_visual_distance: float = 0.0
    std_visual_distance: float = 0.0
    
    # Hardness categorization
    # Easy: distance > 0.8 (very dissimilar - trivial to distinguish)
    # Medium: 0.3 <= distance <= 0.8
    # Hard: distance < 0.3 (visually similar - challenging)
    pct_easy_negatives: float = 0.0
    pct_medium_negatives: float = 0.0
    pct_hard_negatives: float = 0.0
    
    # Distribution for plotting
    distance_distribution: list[float] = field(default_factory=list)
    
    # Hard negative examples (item pairs)
    hard_negative_examples: list[dict] = field(default_factory=list)
    
    # Interpretation
    interpretation: str = ""
    recommendation: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_users_analyzed": self.n_users_analyzed,
            "n_pairs_analyzed": self.n_pairs_analyzed,
            "visual_distance": {
                "mean": round(self.mean_visual_distance, 4),
                "median": round(self.median_visual_distance, 4),
                "std": round(self.std_visual_distance, 4),
            },
            "hardness_distribution": {
                "easy_pct": round(self.pct_easy_negatives, 2),
                "medium_pct": round(self.pct_medium_negatives, 2),
                "hard_pct": round(self.pct_hard_negatives, 2),
            },
            "n_hard_negative_examples": len(self.hard_negative_examples),
            "interpretation": self.interpretation,
            "recommendation": self.recommendation,
        }


def compute_visual_distance(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
) -> float:
    """
    Compute visual distance (1 - cosine similarity) between embeddings.
    
    Returns a value in [0, 2] where:
    - 0 = identical
    - 1 = orthogonal
    - 2 = opposite
    
    Args:
        embedding_a: First embedding vector.
        embedding_b: Second embedding vector.
        
    Returns:
        Visual distance.
    """
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)
    
    if norm_a == 0 or norm_b == 0:
        return 1.0  # Default to orthogonal if zero vector
    
    cos_sim = np.dot(embedding_a, embedding_b) / (norm_a * norm_b)
    # Clamp to valid range
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
    # Convert to distance [0, 2]
    distance = 1.0 - cos_sim
    
    return float(distance)


def sample_positive_negative_pairs(
    interactions_df: pd.DataFrame,
    item_indices: dict[str, int],
    n_users: int = 100,
    n_negatives_per_user: int = 10,
    seed: int = 42,
) -> list[tuple[str, str, str]]:
    """
    Sample (user, positive_item, negative_item) triplets.
    
    Args:
        interactions_df: DataFrame with 'user_id' and 'item_id' columns.
        item_indices: Dictionary mapping item_id to embedding index.
        n_users: Number of users to sample.
        n_negatives_per_user: Negatives per positive item.
        seed: Random seed.
        
    Returns:
        List of (user_id, positive_item_id, negative_item_id) tuples.
    """
    np.random.seed(seed)
    
    # Build user -> items mapping
    user_items = interactions_df.groupby("user_id")["item_id"].apply(set).to_dict()
    
    # Filter to items with embeddings
    items_with_embeddings = set(item_indices.keys())
    
    # Get users with at least one positive item that has embeddings
    valid_users = []
    for user_id, items in user_items.items():
        valid_items = items & items_with_embeddings
        if len(valid_items) > 0:
            valid_users.append((user_id, valid_items))
    
    if len(valid_users) < 10:
        logger.warning("Not enough valid users for BPR hardness analysis")
        return []
    
    # Sample users
    sampled_users = np.random.choice(
        len(valid_users),
        size=min(n_users, len(valid_users)),
        replace=False
    )
    
    # Build negative pool (all items with embeddings)
    all_items = list(items_with_embeddings)
    
    triplets = []
    
    for idx in sampled_users:
        user_id, positive_items = valid_users[idx]
        
        # Sample one positive item
        pos_item = np.random.choice(list(positive_items))
        
        # Sample random negatives (items user hasn't interacted with)
        negative_pool = list(items_with_embeddings - user_items[user_id])
        
        if len(negative_pool) == 0:
            continue
        
        neg_items = np.random.choice(
            negative_pool,
            size=min(n_negatives_per_user, len(negative_pool)),
            replace=False
        )
        
        for neg_item in neg_items:
            triplets.append((user_id, pos_item, neg_item))
    
    return triplets


def analyze_bpr_hardness(
    interactions_df: pd.DataFrame,
    embeddings: np.ndarray,
    item_indices: dict[str, int],
    n_users: int = 100,
    n_negatives_per_user: int = 10,
    easy_threshold: float = 0.8,
    hard_threshold: float = 0.3,
    seed: int = 42,
) -> BPRHardnessResult:
    """
    Analyze BPR negative sampling hardness.
    
    Computes visual distances between positive and randomly-sampled negative
    items to assess how "easy" standard BPR training would be.
    
    Args:
        interactions_df: DataFrame with 'user_id' and 'item_id' columns.
        embeddings: Array of shape [n_items, embedding_dim] with CLIP embeddings.
        item_indices: Dictionary mapping item_id to embedding index.
        n_users: Number of users to sample.
        n_negatives_per_user: Negatives per positive item.
        easy_threshold: Distance above which negatives are "easy".
        hard_threshold: Distance below which negatives are "hard".
        seed: Random seed.
        
    Returns:
        BPRHardnessResult with distance statistics and recommendations.
    """
    logger.info("Analyzing BPR negative sampling hardness...")
    
    result = BPRHardnessResult()
    
    # Sample triplets
    triplets = sample_positive_negative_pairs(
        interactions_df,
        item_indices,
        n_users=n_users,
        n_negatives_per_user=n_negatives_per_user,
        seed=seed,
    )
    
    if len(triplets) < 10:
        result.interpretation = "Insufficient triplets for analysis"
        return result
    
    logger.info(f"  Sampled {len(triplets)} triplets from {n_users} users")
    
    # Compute distances
    distances = []
    hard_examples = []
    users_seen = set()
    
    for user_id, pos_item, neg_item in triplets:
        pos_idx = item_indices[pos_item]
        neg_idx = item_indices[neg_item]
        
        distance = compute_visual_distance(
            embeddings[pos_idx],
            embeddings[neg_idx]
        )
        
        distances.append(distance)
        users_seen.add(user_id)
        
        # Track hard negatives (visually similar but not purchased)
        if distance < hard_threshold:
            hard_examples.append({
                "user_id": str(user_id),
                "positive_item": str(pos_item),
                "negative_item": str(neg_item),
                "visual_distance": round(distance, 4),
            })
    
    result.n_users_analyzed = len(users_seen)
    result.n_pairs_analyzed = len(distances)
    result.distance_distribution = distances
    
    # Statistics
    distances_arr = np.array(distances)
    result.mean_visual_distance = float(np.mean(distances_arr))
    result.median_visual_distance = float(np.median(distances_arr))
    result.std_visual_distance = float(np.std(distances_arr))
    
    # Categorize
    n_easy = np.sum(distances_arr > easy_threshold)
    n_hard = np.sum(distances_arr < hard_threshold)
    n_medium = len(distances_arr) - n_easy - n_hard
    
    result.pct_easy_negatives = 100.0 * n_easy / len(distances_arr)
    result.pct_medium_negatives = 100.0 * n_medium / len(distances_arr)
    result.pct_hard_negatives = 100.0 * n_hard / len(distances_arr)
    
    # Store top hard negative examples
    result.hard_negative_examples = sorted(
        hard_examples, key=lambda x: x["visual_distance"]
    )[:20]
    
    logger.info(f"  Mean visual distance: {result.mean_visual_distance:.4f}")
    logger.info(f"  Easy: {result.pct_easy_negatives:.1f}%, Medium: {result.pct_medium_negatives:.1f}%, Hard: {result.pct_hard_negatives:.1f}%")
    
    # Interpretation
    if result.pct_easy_negatives > 80:
        result.interpretation = "Most random negatives are trivially easy - model won't learn visual preferences"
        result.recommendation = "Use hard negative sampling based on visual similarity"
    elif result.pct_easy_negatives > 50:
        result.interpretation = "Majority of negatives are easy - limited visual learning signal"
        result.recommendation = "Consider popularity-weighted or visual-similarity-based negative sampling"
    elif result.pct_hard_negatives > 30:
        result.interpretation = "Good distribution of hard negatives - random sampling may suffice"
        result.recommendation = "Standard BPR with random negatives should work"
    else:
        result.interpretation = "Moderate negative difficulty - room for improvement"
        result.recommendation = "Consider mixing random and hard negative sampling"
    
    logger.info(f"  Interpretation: {result.interpretation}")
    
    return result


def find_hard_negatives_for_user(
    user_id: str,
    positive_items: list[str],
    all_items: list[str],
    embeddings: np.ndarray,
    item_indices: dict[str, int],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Find hard negatives for a user based on visual similarity.
    
    Hard negatives are items that:
    1. The user hasn't interacted with
    2. Are visually similar to items the user has interacted with
    
    Args:
        user_id: User ID.
        positive_items: Items the user has interacted with.
        all_items: All candidate items.
        embeddings: Array of embeddings.
        item_indices: Mapping from item_id to embedding index.
        top_k: Number of hard negatives to return.
        
    Returns:
        List of (item_id, similarity_score) tuples.
    """
    positive_set = set(positive_items)
    negative_candidates = [item for item in all_items if item not in positive_set]
    
    if len(negative_candidates) == 0:
        return []
    
    # Compute average positive embedding
    pos_indices = [item_indices[item] for item in positive_items if item in item_indices]
    if len(pos_indices) == 0:
        return []
    
    avg_pos_embedding = np.mean(embeddings[pos_indices], axis=0)
    
    # Score all negative candidates
    similarities = []
    for neg_item in negative_candidates:
        if neg_item not in item_indices:
            continue
        
        neg_idx = item_indices[neg_item]
        neg_emb = embeddings[neg_idx]
        
        # Cosine similarity
        sim = np.dot(avg_pos_embedding, neg_emb) / (
            np.linalg.norm(avg_pos_embedding) * np.linalg.norm(neg_emb) + 1e-8
        )
        similarities.append((neg_item, float(sim)))
    
    # Sort by similarity (highest = hardest negatives)
    similarities.sort(key=lambda x: -x[1])
    
    return similarities[:top_k]
