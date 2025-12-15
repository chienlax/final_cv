"""
Negative sampling strategies for BPR training.

Implements:
- Uniform random negative sampling
- Popularity-based negative sampling
- BPR triplet creation
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def sample_negatives_uniform(
    user_items: dict[int, set[int]],
    n_items: int,
    n_negatives: int = 4,
    seed: int = 42,
) -> dict[int, list[int]]:
    """
    Sample negative items uniformly for each user.
    
    Args:
        user_items: Dictionary mapping user_idx to set of positive item_idx.
        n_items: Total number of items.
        n_negatives: Number of negatives to sample per user.
        seed: Random seed.
        
    Returns:
        Dictionary mapping user_idx to list of negative item_idx.
    """
    np.random.seed(seed)
    
    all_items = set(range(n_items))
    user_negatives = {}
    
    for user_idx, positive_items in user_items.items():
        negative_pool = list(all_items - positive_items)
        
        if len(negative_pool) >= n_negatives:
            negatives = np.random.choice(negative_pool, size=n_negatives, replace=False)
        else:
            negatives = np.random.choice(negative_pool, size=n_negatives, replace=True)
        
        user_negatives[user_idx] = negatives.tolist()
    
    return user_negatives


def sample_negatives_popularity(
    user_items: dict[int, set[int]],
    item_popularity: np.ndarray,
    n_negatives: int = 4,
    seed: int = 42,
) -> dict[int, list[int]]:
    """
    Sample negative items based on popularity (hard negatives).
    
    Popular items that the user hasn't interacted with are more likely
    to be sampled as negatives.
    
    Args:
        user_items: Dictionary mapping user_idx to set of positive item_idx.
        item_popularity: Array of popularity scores per item. Shape: [n_items]
        n_negatives: Number of negatives to sample per user.
        seed: Random seed.
        
    Returns:
        Dictionary mapping user_idx to list of negative item_idx.
    """
    np.random.seed(seed)
    
    n_items = len(item_popularity)
    user_negatives = {}
    
    for user_idx, positive_items in user_items.items():
        # Create mask for negative items
        mask = np.ones(n_items, dtype=bool)
        mask[list(positive_items)] = False
        
        negative_items = np.where(mask)[0]
        negative_probs = item_popularity[negative_items]
        
        # Normalize probabilities
        negative_probs = negative_probs / negative_probs.sum()
        
        # Sample based on popularity
        if len(negative_items) >= n_negatives:
            negatives = np.random.choice(
                negative_items,
                size=n_negatives,
                replace=False,
                p=negative_probs,
            )
        else:
            negatives = np.random.choice(
                negative_items,
                size=n_negatives,
                replace=True,
                p=negative_probs,
            )
        
        user_negatives[user_idx] = negatives.tolist()
    
    return user_negatives


def create_bpr_triplets(
    train_df: pd.DataFrame,
    n_items: int,
    n_negatives: int = 1,
    strategy: str = "uniform",
    seed: int = 42,
) -> np.ndarray:
    """
    Create BPR triplets (user, positive_item, negative_item) for training.
    
    For each positive interaction, sample n_negatives negative items.
    
    Args:
        train_df: Training DataFrame with 'user_idx' and 'item_idx' columns.
        n_items: Total number of items.
        n_negatives: Number of negatives per positive.
        strategy: "uniform" or "popularity" sampling.
        seed: Random seed.
        
    Returns:
        Array of shape [n_triplets, 3] with (user_idx, pos_item_idx, neg_item_idx).
    """
    logger.info(f"Creating BPR triplets with {strategy} sampling, {n_negatives} negatives per positive...")
    
    np.random.seed(seed)
    
    # Build user -> items mapping
    user_items = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    
    # Calculate item popularity for popularity-based sampling
    item_counts = train_df["item_idx"].value_counts()
    item_popularity = np.zeros(n_items)
    item_popularity[item_counts.index] = item_counts.values
    item_popularity = item_popularity ** 0.75  # Smooth popularity (common practice)
    
    triplets = []
    
    for _, row in train_df.iterrows():
        user_idx = row["user_idx"]
        pos_item_idx = row["item_idx"]
        
        # Get items user has interacted with
        positive_items = user_items[user_idx]
        
        # Sample negatives
        negative_pool = list(set(range(n_items)) - positive_items)
        
        if len(negative_pool) == 0:
            continue
        
        if strategy == "popularity":
            # Weight by popularity
            neg_probs = item_popularity[negative_pool]
            neg_probs = neg_probs / neg_probs.sum()
            
            neg_items = np.random.choice(
                negative_pool,
                size=min(n_negatives, len(negative_pool)),
                replace=False,
                p=neg_probs,
            )
        else:
            # Uniform sampling
            neg_items = np.random.choice(
                negative_pool,
                size=min(n_negatives, len(negative_pool)),
                replace=False,
            )
        
        for neg_item_idx in neg_items:
            triplets.append([user_idx, pos_item_idx, neg_item_idx])
    
    triplets = np.array(triplets, dtype=np.int64)
    
    logger.info(f"Created {len(triplets):,} BPR triplets")
    
    return triplets


def build_user_item_sets(df: pd.DataFrame) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """
    Build user-to-items and item-to-users mappings.
    
    Args:
        df: DataFrame with 'user_idx' and 'item_idx' columns.
        
    Returns:
        Tuple of (user_to_items, item_to_users) dictionaries.
    """
    user_to_items = df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    item_to_users = df.groupby("item_idx")["user_idx"].apply(set).to_dict()
    
    return user_to_items, item_to_users


def get_item_popularity(df: pd.DataFrame, n_items: int) -> np.ndarray:
    """
    Calculate item popularity (interaction count).
    
    Args:
        df: DataFrame with 'item_idx' column.
        n_items: Total number of items.
        
    Returns:
        Array of shape [n_items] with popularity counts.
    """
    counts = df["item_idx"].value_counts()
    popularity = np.zeros(n_items, dtype=np.float32)
    popularity[counts.index] = counts.values
    
    return popularity
