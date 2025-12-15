"""
User and item behavior analysis for recommendation datasets.

Provides in-depth analysis of:
- Power-law distribution fitting
- Cold-start identification
- User engagement patterns
- Category-wise distributions
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class UserItemPatterns:
    """User and item behavior pattern statistics."""
    
    # User patterns
    n_users: int = 0
    user_interaction_stats: dict[str, float] = None  # mean, median, std, max, min
    power_law_alpha_users: float = 0.0  # Power-law exponent
    cold_start_users_pct: float = 0.0  # Users with <5 interactions
    
    # Item patterns
    n_items: int = 0
    item_interaction_stats: dict[str, float] = None
    power_law_alpha_items: float = 0.0
    cold_start_items_pct: float = 0.0  # Items with <5 interactions
    
    # Engagement
    avg_rating_per_user: float = 0.0
    rating_variance_per_user: float = 0.0
    
    def __post_init__(self):
        if self.user_interaction_stats is None:
            self.user_interaction_stats = {}
        if self.item_interaction_stats is None:
            self.item_interaction_stats = {}
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "users": {
                "n_users": self.n_users,
                "interaction_stats": self.user_interaction_stats,
                "power_law_alpha": round(self.power_law_alpha_users, 3),
                "cold_start_pct": round(self.cold_start_users_pct, 2),
            },
            "items": {
                "n_items": self.n_items,
                "interaction_stats": self.item_interaction_stats,
                "power_law_alpha": round(self.power_law_alpha_items, 3),
                "cold_start_pct": round(self.cold_start_items_pct, 2),
            },
            "engagement": {
                "avg_rating_per_user": round(self.avg_rating_per_user, 3),
                "rating_variance_per_user": round(self.rating_variance_per_user, 3),
            },
        }


def analyze_user_item_patterns(
    df: pd.DataFrame,
    cold_start_threshold: int = 5,
) -> UserItemPatterns:
    """
    Analyze user and item interaction patterns.
    
    Args:
        df: DataFrame with columns [user_id, item_id, rating].
        cold_start_threshold: Minimum interactions to not be cold-start.
        
    Returns:
        UserItemPatterns object with analysis results.
    """
    logger.info("Analyzing user and item patterns...")
    
    patterns = UserItemPatterns()
    
    if len(df) == 0:
        return patterns
    
    # -------------------------------------------------------------------------
    # User analysis
    # -------------------------------------------------------------------------
    user_counts = df["user_id"].value_counts()
    patterns.n_users = len(user_counts)
    
    patterns.user_interaction_stats = {
        "mean": float(user_counts.mean()),
        "median": float(user_counts.median()),
        "std": float(user_counts.std()),
        "min": int(user_counts.min()),
        "max": int(user_counts.max()),
        "p25": float(user_counts.quantile(0.25)),
        "p75": float(user_counts.quantile(0.75)),
        "p95": float(user_counts.quantile(0.95)),
    }
    
    # Cold-start users
    cold_users = (user_counts < cold_start_threshold).sum()
    patterns.cold_start_users_pct = float(cold_users / len(user_counts) * 100)
    
    # Power-law fit for users
    patterns.power_law_alpha_users = _estimate_power_law_alpha(user_counts.values)
    
    # -------------------------------------------------------------------------
    # Item analysis
    # -------------------------------------------------------------------------
    item_counts = df["item_id"].value_counts()
    patterns.n_items = len(item_counts)
    
    patterns.item_interaction_stats = {
        "mean": float(item_counts.mean()),
        "median": float(item_counts.median()),
        "std": float(item_counts.std()),
        "min": int(item_counts.min()),
        "max": int(item_counts.max()),
        "p25": float(item_counts.quantile(0.25)),
        "p75": float(item_counts.quantile(0.75)),
        "p95": float(item_counts.quantile(0.95)),
    }
    
    # Cold-start items
    cold_items = (item_counts < cold_start_threshold).sum()
    patterns.cold_start_items_pct = float(cold_items / len(item_counts) * 100)
    
    # Power-law fit for items
    patterns.power_law_alpha_items = _estimate_power_law_alpha(item_counts.values)
    
    # -------------------------------------------------------------------------
    # Engagement analysis
    # -------------------------------------------------------------------------
    if "rating" in df.columns:
        user_ratings = df.groupby("user_id")["rating"].agg(["mean", "var"])
        patterns.avg_rating_per_user = float(user_ratings["mean"].mean())
        patterns.rating_variance_per_user = float(user_ratings["var"].mean())
    
    logger.info(f"User patterns: {patterns.cold_start_users_pct:.1f}% cold-start, α={patterns.power_law_alpha_users:.2f}")
    logger.info(f"Item patterns: {patterns.cold_start_items_pct:.1f}% cold-start, α={patterns.power_law_alpha_items:.2f}")
    
    return patterns


def _estimate_power_law_alpha(values: np.ndarray) -> float:
    """
    Estimate power-law exponent using MLE for discrete power-law.
    
    Uses the Clauset et al. (2009) method approximation.
    
    Args:
        values: Array of positive integer values (e.g., interaction counts).
        
    Returns:
        Estimated power-law exponent alpha.
    """
    values = values[values > 0]  # Filter zeros
    if len(values) < 10:
        return 0.0
    
    # Simple MLE estimator: alpha = 1 + n / sum(log(x / x_min))
    x_min = values.min()
    n = len(values)
    
    log_sum = np.sum(np.log(values / (x_min - 0.5)))
    
    if log_sum <= 0:
        return 0.0
    
    alpha = 1 + n / log_sum
    
    return float(alpha)


def compute_user_activity_segments(
    df: pd.DataFrame,
    n_segments: int = 5,
) -> pd.DataFrame:
    """
    Segment users by activity level (quintiles).
    
    Args:
        df: DataFrame with 'user_id' column.
        n_segments: Number of segments to create.
        
    Returns:
        DataFrame with segment statistics.
    """
    user_counts = df["user_id"].value_counts().reset_index()
    user_counts.columns = ["user_id", "n_interactions"]
    
    # Create quantile-based segments - don't use labels due to potential duplicate drops
    try:
        user_counts["segment"] = pd.qcut(
            user_counts["n_interactions"],
            q=n_segments,
            duplicates="drop",
        )
    except ValueError:
        # Fallback to fewer segments if values are too concentrated
        user_counts["segment"] = pd.cut(
            user_counts["n_interactions"],
            bins=3,
            labels=["Low", "Medium", "High"],
        )
    
    # Aggregate by segment
    segment_stats = user_counts.groupby("segment", observed=True).agg(
        n_users=("user_id", "count"),
        min_interactions=("n_interactions", "min"),
        max_interactions=("n_interactions", "max"),
        mean_interactions=("n_interactions", "mean"),
    ).reset_index()
    
    segment_stats["segment"] = segment_stats["segment"].astype(str)
    segment_stats["pct_of_users"] = segment_stats["n_users"] / len(user_counts) * 100
    
    return segment_stats


def compute_item_popularity_segments(
    df: pd.DataFrame,
    n_segments: int = 5,
) -> pd.DataFrame:
    """
    Segment items by popularity (quintiles).
    
    Args:
        df: DataFrame with 'item_id' column.
        n_segments: Number of segments.
        
    Returns:
        DataFrame with segment statistics.
    """
    item_counts = df["item_id"].value_counts().reset_index()
    item_counts.columns = ["item_id", "n_interactions"]
    
    # Create quantile-based segments - don't use labels due to potential duplicate drops
    try:
        item_counts["segment"] = pd.qcut(
            item_counts["n_interactions"],
            q=n_segments,
            duplicates="drop",
        )
    except ValueError:
        # Fallback to fewer segments if values are too concentrated
        item_counts["segment"] = pd.cut(
            item_counts["n_interactions"],
            bins=3,
            labels=["Low", "Medium", "High"],
        )
    
    segment_stats = item_counts.groupby("segment", observed=True).agg(
        n_items=("item_id", "count"),
        min_interactions=("n_interactions", "min"),
        max_interactions=("n_interactions", "max"),
        mean_interactions=("n_interactions", "mean"),
    ).reset_index()
    
    segment_stats["segment"] = segment_stats["segment"].astype(str)
    segment_stats["pct_of_items"] = segment_stats["n_items"] / len(item_counts) * 100
    
    return segment_stats


def analyze_rating_behavior_by_activity(
    df: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Analyze how rating behavior varies with user activity level.
    
    Args:
        df: DataFrame with user_id and rating columns.
        n_bins: Number of activity bins.
        
    Returns:
        DataFrame showing rating patterns by activity level.
    """
    user_activity = df.groupby("user_id").agg(
        n_interactions=("item_id", "count"),
        avg_rating=("rating", "mean"),
        rating_std=("rating", "std"),
    ).reset_index()
    
    # Bin by activity
    user_activity["activity_bin"] = pd.cut(
        user_activity["n_interactions"],
        bins=n_bins,
        labels=[f"Bin_{i+1}" for i in range(n_bins)],
    )
    
    # Aggregate by bin
    bin_stats = user_activity.groupby("activity_bin").agg(
        n_users=("user_id", "count"),
        mean_activity=("n_interactions", "mean"),
        avg_rating=("avg_rating", "mean"),
        rating_std=("rating_std", "mean"),
    ).reset_index()
    
    return bin_stats


def identify_super_users_and_items(
    df: pd.DataFrame,
    top_percentile: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify super users and super items (top percentile by interactions).
    
    Args:
        df: DataFrame with user_id and item_id.
        top_percentile: Percentile threshold (e.g., 0.01 = top 1%).
        
    Returns:
        Tuple of (super_users_df, super_items_df).
    """
    # Super users
    user_counts = df["user_id"].value_counts()
    threshold_users = user_counts.quantile(1 - top_percentile)
    super_users = user_counts[user_counts >= threshold_users].reset_index()
    super_users.columns = ["user_id", "n_interactions"]
    
    # Super items
    item_counts = df["item_id"].value_counts()
    threshold_items = item_counts.quantile(1 - top_percentile)
    super_items = item_counts[item_counts >= threshold_items].reset_index()
    super_items.columns = ["item_id", "n_interactions"]
    
    logger.info(f"Super users (top {top_percentile*100:.1f}%): {len(super_users)} users")
    logger.info(f"Super items (top {top_percentile*100:.1f}%): {len(super_items)} items")
    
    return super_users, super_items


def analyze_interaction_distribution_tiers(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Analyze the distribution of interactions across different entity tiers.
    
    Shows how interactions are concentrated among top users/items (Pareto analysis).
    
    Args:
        df: DataFrame with 'user_id' and 'item_id' columns.
        
    Returns:
        Dictionary with 'users' and 'items' DataFrames showing tier concentration.
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
