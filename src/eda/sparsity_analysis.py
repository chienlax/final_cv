"""
Sparsity and k-core analysis for recommendation datasets.

Provides analysis of:
- Interaction matrix sparsity
- K-core filtering simulation
- Data retention analysis
- User/item survival curves
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SparsityStats:
    """Sparsity analysis statistics."""
    
    n_users: int = 0
    n_items: int = 0
    n_interactions: int = 0
    
    possible_interactions: int = 0
    density: float = 0.0
    sparsity: float = 0.0
    
    # After k-core filtering
    kcore_results: dict[int, dict] = field(default_factory=dict)  # k -> stats
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original": {
                "n_users": self.n_users,
                "n_items": self.n_items,
                "n_interactions": self.n_interactions,
                "possible_interactions": self.possible_interactions,
                "density": f"{self.density:.6%}",
                "sparsity": f"{self.sparsity:.6%}",
            },
            "kcore_analysis": self.kcore_results,
        }


def analyze_sparsity(df: pd.DataFrame) -> SparsityStats:
    """
    Analyze interaction matrix sparsity.
    
    Args:
        df: DataFrame with 'user_id' and 'item_id' columns.
        
    Returns:
        SparsityStats object.
    """
    logger.info("Analyzing sparsity...")
    
    stats = SparsityStats()
    
    if len(df) == 0:
        return stats
    
    stats.n_users = df["user_id"].nunique()
    stats.n_items = df["item_id"].nunique()
    stats.n_interactions = len(df)
    
    stats.possible_interactions = stats.n_users * stats.n_items
    
    if stats.possible_interactions > 0:
        stats.density = stats.n_interactions / stats.possible_interactions
        stats.sparsity = 1 - stats.density
    
    logger.info(f"Sparsity: {stats.sparsity:.6%} ({stats.n_interactions:,} / {stats.possible_interactions:,})")
    
    return stats


def simulate_kcore_filtering(
    df: pd.DataFrame,
    k_values: list[int] = None,
    max_iterations: int = 100,
) -> dict[int, dict]:
    """
    Simulate k-core filtering for different k values.
    
    K-core filtering iteratively removes users/items with fewer than k interactions
    until convergence.
    
    Args:
        df: DataFrame with 'user_id' and 'item_id' columns.
        k_values: List of k values to test (default: [2, 3, 5, 10, 20]).
        max_iterations: Maximum iterations for convergence.
        
    Returns:
        Dictionary mapping k -> statistics after filtering.
    """
    if k_values is None:
        k_values = [2, 3, 5, 10, 20]
    
    logger.info(f"Simulating k-core filtering for k={k_values}...")
    
    results = {}
    
    for k in k_values:
        filtered_df = _apply_kcore_filter(df.copy(), k=k, max_iterations=max_iterations)
        
        n_users_after = filtered_df["user_id"].nunique()
        n_items_after = filtered_df["item_id"].nunique()
        n_interactions_after = len(filtered_df)
        
        user_retention = n_users_after / df["user_id"].nunique() * 100 if df["user_id"].nunique() > 0 else 0
        item_retention = n_items_after / df["item_id"].nunique() * 100 if df["item_id"].nunique() > 0 else 0
        interaction_retention = n_interactions_after / len(df) * 100 if len(df) > 0 else 0
        
        # Recalculate sparsity
        possible = n_users_after * n_items_after
        density = n_interactions_after / possible if possible > 0 else 0
        
        results[k] = {
            "k": k,
            "n_users": n_users_after,
            "n_items": n_items_after,
            "n_interactions": n_interactions_after,
            "user_retention_pct": round(user_retention, 2),
            "item_retention_pct": round(item_retention, 2),
            "interaction_retention_pct": round(interaction_retention, 2),
            "density_after": f"{density:.6%}",
            "avg_interactions_per_user": round(n_interactions_after / n_users_after, 2) if n_users_after > 0 else 0,
            "avg_interactions_per_item": round(n_interactions_after / n_items_after, 2) if n_items_after > 0 else 0,
        }
        
        logger.info(f"  k={k}: {user_retention:.1f}% users, {item_retention:.1f}% items, {interaction_retention:.1f}% interactions retained")
    
    return results


def _apply_kcore_filter(
    df: pd.DataFrame,
    k: int,
    max_iterations: int = 100,
) -> pd.DataFrame:
    """
    Apply iterative k-core filtering.
    
    Args:
        df: DataFrame to filter.
        k: Minimum interaction threshold.
        max_iterations: Maximum convergence iterations.
        
    Returns:
        Filtered DataFrame.
    """
    for iteration in range(max_iterations):
        n_before = len(df)
        
        # Filter users with < k interactions
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= k].index
        df = df[df["user_id"].isin(valid_users)]
        
        # Filter items with < k interactions
        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= k].index
        df = df[df["item_id"].isin(valid_items)]
        
        n_after = len(df)
        
        if n_after == n_before:
            break  # Converged
    
    return df


def compute_retention_curve(
    df: pd.DataFrame,
    k_range: range = None,
) -> pd.DataFrame:
    """
    Compute data retention curve for a range of k values.
    
    Useful for choosing the optimal k-core threshold.
    
    Args:
        df: DataFrame with 'user_id' and 'item_id' columns.
        k_range: Range of k values to test.
        
    Returns:
        DataFrame with retention statistics per k.
    """
    if k_range is None:
        k_range = range(1, 21)
    
    results = []
    
    original_users = df["user_id"].nunique()
    original_items = df["item_id"].nunique()
    original_interactions = len(df)
    
    for k in k_range:
        filtered = _apply_kcore_filter(df.copy(), k=k)
        
        results.append({
            "k": k,
            "n_users": filtered["user_id"].nunique(),
            "n_items": filtered["item_id"].nunique(),
            "n_interactions": len(filtered),
            "user_retention": filtered["user_id"].nunique() / original_users * 100,
            "item_retention": filtered["item_id"].nunique() / original_items * 100,
            "interaction_retention": len(filtered) / original_interactions * 100,
        })
    
    return pd.DataFrame(results)


def analyze_interaction_distribution_tiers(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Analyze the distribution of interactions across different entity tiers.
    
    Shows how interactions are concentrated among top users/items.
    
    Args:
        df: DataFrame with 'user_id' and 'item_id' columns.
        
    Returns:
        Dictionary with 'users' and 'items' DataFrames.
    """
    results = {}
    
    # User tiers
    user_counts = df["user_id"].value_counts().sort_values(ascending=False)
    user_cumsum = user_counts.cumsum()
    
    tiers = [0.01, 0.05, 0.10, 0.20, 0.50, 1.0]
    user_tier_stats = []
    
    for tier in tiers:
        n_users_in_tier = int(len(user_counts) * tier)
        if n_users_in_tier == 0:
            n_users_in_tier = 1
        
        interactions_from_tier = user_cumsum.iloc[n_users_in_tier - 1]
        pct_interactions = interactions_from_tier / len(df) * 100
        
        user_tier_stats.append({
            "tier_pct": f"Top {tier*100:.0f}%",
            "n_users": n_users_in_tier,
            "interactions": interactions_from_tier,
            "pct_of_total_interactions": round(pct_interactions, 2),
        })
    
    results["users"] = pd.DataFrame(user_tier_stats)
    
    # Item tiers
    item_counts = df["item_id"].value_counts().sort_values(ascending=False)
    item_cumsum = item_counts.cumsum()
    
    item_tier_stats = []
    
    for tier in tiers:
        n_items_in_tier = int(len(item_counts) * tier)
        if n_items_in_tier == 0:
            n_items_in_tier = 1
        
        interactions_from_tier = item_cumsum.iloc[n_items_in_tier - 1]
        pct_interactions = interactions_from_tier / len(df) * 100
        
        item_tier_stats.append({
            "tier_pct": f"Top {tier*100:.0f}%",
            "n_items": n_items_in_tier,
            "interactions": interactions_from_tier,
            "pct_of_total_interactions": round(pct_interactions, 2),
        })
    
    results["items"] = pd.DataFrame(item_tier_stats)
    
    return results
