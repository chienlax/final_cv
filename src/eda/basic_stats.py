"""
Basic statistics computation for Amazon Review 2023 dataset.

Computes core statistics for interactions and metadata including:
- User/item counts and distributions
- Rating distributions
- Temporal patterns
- Text length statistics
- Missing value analysis
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class InteractionStats:
    """Statistics for interaction data."""
    
    # Counts
    n_interactions: int = 0
    n_users: int = 0
    n_items: int = 0
    
    # Rating stats
    rating_mean: float = 0.0
    rating_std: float = 0.0
    rating_median: float = 0.0
    rating_distribution: dict[float, int] = field(default_factory=dict)
    
    # Density
    sparsity: float = 0.0  # 1 - density
    avg_interactions_per_user: float = 0.0
    avg_interactions_per_item: float = 0.0
    
    # Temporal
    date_min: str = ""
    date_max: str = ""
    date_range_days: int = 0
    
    # Text
    avg_review_length: float = 0.0
    avg_title_length: float = 0.0
    reviews_with_text_pct: float = 0.0
    
    # Verification
    verified_purchase_pct: float = 0.0
    avg_helpful_votes: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "counts": {
                "n_interactions": self.n_interactions,
                "n_users": self.n_users,
                "n_items": self.n_items,
            },
            "ratings": {
                "mean": round(self.rating_mean, 3),
                "std": round(self.rating_std, 3),
                "median": self.rating_median,
                "distribution": self.rating_distribution,
            },
            "density": {
                "sparsity": round(self.sparsity, 6),
                "avg_per_user": round(self.avg_interactions_per_user, 2),
                "avg_per_item": round(self.avg_interactions_per_item, 2),
            },
            "temporal": {
                "date_min": self.date_min,
                "date_max": self.date_max,
                "date_range_days": self.date_range_days,
            },
            "text": {
                "avg_review_length": round(self.avg_review_length, 1),
                "avg_title_length": round(self.avg_title_length, 1),
                "reviews_with_text_pct": round(self.reviews_with_text_pct, 2),
            },
            "verification": {
                "verified_purchase_pct": round(self.verified_purchase_pct, 2),
                "avg_helpful_votes": round(self.avg_helpful_votes, 2),
            },
        }


@dataclass
class MetadataStats:
    """Statistics for item metadata."""
    
    n_items: int = 0
    
    # Text coverage
    items_with_title_pct: float = 0.0
    items_with_description_pct: float = 0.0
    items_with_features_pct: float = 0.0
    
    # Text lengths
    avg_title_length: float = 0.0
    avg_description_length: float = 0.0
    
    # Images
    items_with_images_pct: float = 0.0
    avg_image_count: float = 0.0
    
    # Categories
    n_categories: int = 0
    top_categories: dict[str, int] = field(default_factory=dict)
    
    # Ratings (from metadata)
    avg_rating_mean: float = 0.0
    avg_rating_count: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "counts": {"n_items": self.n_items},
            "text_coverage": {
                "with_title_pct": round(self.items_with_title_pct, 2),
                "with_description_pct": round(self.items_with_description_pct, 2),
                "with_features_pct": round(self.items_with_features_pct, 2),
            },
            "text_lengths": {
                "avg_title": round(self.avg_title_length, 1),
                "avg_description": round(self.avg_description_length, 1),
            },
            "images": {
                "with_images_pct": round(self.items_with_images_pct, 2),
                "avg_count": round(self.avg_image_count, 2),
            },
            "categories": {
                "n_unique": self.n_categories,
                "top_10": dict(list(self.top_categories.items())[:10]),
            },
            "ratings": {
                "avg_rating": round(self.avg_rating_mean, 2),
                "avg_rating_count": round(self.avg_rating_count, 1),
            },
        }


def compute_basic_statistics(
    interactions_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None = None,
) -> tuple[InteractionStats, MetadataStats | None]:
    """
    Compute comprehensive statistics for interaction and metadata DataFrames.
    
    Args:
        interactions_df: DataFrame with columns [user_id, item_id, rating, timestamp, ...].
        metadata_df: Optional DataFrame with item metadata.
        
    Returns:
        Tuple of (InteractionStats, MetadataStats or None).
    """
    logger.info("Computing interaction statistics...")
    int_stats = _compute_interaction_stats(interactions_df)
    
    meta_stats = None
    if metadata_df is not None and len(metadata_df) > 0:
        logger.info("Computing metadata statistics...")
        meta_stats = _compute_metadata_stats(metadata_df)
    
    return int_stats, meta_stats


def _compute_interaction_stats(df: pd.DataFrame) -> InteractionStats:
    """Compute statistics for interaction DataFrame."""
    stats = InteractionStats()
    
    if len(df) == 0:
        return stats
    
    # Basic counts
    stats.n_interactions = len(df)
    stats.n_users = df["user_id"].nunique()
    stats.n_items = df["item_id"].nunique()
    
    # Rating statistics
    if "rating" in df.columns:
        ratings = df["rating"].dropna()
        if len(ratings) > 0:
            stats.rating_mean = float(ratings.mean())
            stats.rating_std = float(ratings.std())
            stats.rating_median = float(ratings.median())
            stats.rating_distribution = ratings.value_counts().sort_index().to_dict()
    
    # Sparsity: 1 - (interactions / (users * items))
    possible_interactions = stats.n_users * stats.n_items
    if possible_interactions > 0:
        stats.sparsity = 1.0 - (stats.n_interactions / possible_interactions)
    
    stats.avg_interactions_per_user = stats.n_interactions / stats.n_users if stats.n_users > 0 else 0
    stats.avg_interactions_per_item = stats.n_interactions / stats.n_items if stats.n_items > 0 else 0
    
    # Temporal statistics
    if "timestamp" in df.columns:
        timestamps = df["timestamp"].dropna()
        if len(timestamps) > 0:
            stats.date_min = str(timestamps.min().date()) if hasattr(timestamps.min(), "date") else str(timestamps.min())
            stats.date_max = str(timestamps.max().date()) if hasattr(timestamps.max(), "date") else str(timestamps.max())
            try:
                stats.date_range_days = (timestamps.max() - timestamps.min()).days
            except Exception:
                stats.date_range_days = 0
    
    # Text statistics
    if "review_text" in df.columns:
        text_lengths = df["review_text"].fillna("").str.len()
        stats.avg_review_length = float(text_lengths.mean())
        stats.reviews_with_text_pct = float((text_lengths > 0).mean() * 100)
    
    if "review_title" in df.columns:
        title_lengths = df["review_title"].fillna("").str.len()
        stats.avg_title_length = float(title_lengths.mean())
    
    # Verification stats
    if "verified_purchase" in df.columns:
        stats.verified_purchase_pct = float(df["verified_purchase"].mean() * 100)
    
    if "helpful_vote" in df.columns:
        stats.avg_helpful_votes = float(df["helpful_vote"].mean())
    
    return stats


def _compute_metadata_stats(df: pd.DataFrame) -> MetadataStats:
    """Compute statistics for metadata DataFrame."""
    stats = MetadataStats()
    
    if len(df) == 0:
        return stats
    
    stats.n_items = len(df)
    
    # Text coverage
    if "title" in df.columns:
        has_title = df["title"].fillna("").str.len() > 0
        stats.items_with_title_pct = float(has_title.mean() * 100)
        stats.avg_title_length = float(df["title"].fillna("").str.len().mean())
    
    if "description" in df.columns:
        has_desc = df["description"].fillna("").str.len() > 0
        stats.items_with_description_pct = float(has_desc.mean() * 100)
        stats.avg_description_length = float(df["description"].fillna("").str.len().mean())
    
    if "features" in df.columns:
        has_features = df["features"].fillna("").str.len() > 0
        stats.items_with_features_pct = float(has_features.mean() * 100)
    
    # Image coverage
    if "image_count" in df.columns:
        stats.items_with_images_pct = float((df["image_count"] > 0).mean() * 100)
        stats.avg_image_count = float(df["image_count"].mean())
    
    # Categories
    if "main_category" in df.columns:
        categories = df["main_category"].dropna()
        categories = categories[categories.str.len() > 0]
        stats.n_categories = categories.nunique()
        stats.top_categories = categories.value_counts().head(20).to_dict()
    
    # Ratings from metadata
    if "average_rating" in df.columns:
        ratings = df["average_rating"].dropna()
        if len(ratings) > 0:
            stats.avg_rating_mean = float(ratings.mean())
    
    if "rating_number" in df.columns:
        counts = df["rating_number"].dropna()
        if len(counts) > 0:
            stats.avg_rating_count = float(counts.mean())
    
    return stats


def compute_rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute detailed rating distribution with percentages.
    
    Args:
        df: DataFrame with 'rating' column.
        
    Returns:
        DataFrame with rating value counts and percentages.
    """
    if "rating" not in df.columns:
        return pd.DataFrame()
    
    dist = df["rating"].value_counts().sort_index()
    dist_df = pd.DataFrame({
        "rating": dist.index,
        "count": dist.values,
        "percentage": (dist.values / len(df) * 100).round(2),
    })
    
    return dist_df


def compute_user_item_frequency(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Compute interaction frequency per user and per item.
    
    Args:
        df: DataFrame with 'user_id' and 'item_id' columns.
        
    Returns:
        Tuple of (user_freq Series, item_freq Series).
    """
    user_freq = df["user_id"].value_counts()
    item_freq = df["item_id"].value_counts()
    
    return user_freq, item_freq


def compute_temporal_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute temporal statistics by month/year.
    
    Args:
        df: DataFrame with 'timestamp' column (datetime).
        
    Returns:
        DataFrame with monthly interaction counts.
    """
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    
    df_temp = df.copy()
    df_temp["year_month"] = df_temp["timestamp"].dt.to_period("M")
    
    monthly = df_temp.groupby("year_month").agg(
        n_interactions=("user_id", "count"),
        n_users=("user_id", "nunique"),
        n_items=("item_id", "nunique"),
        avg_rating=("rating", "mean"),
    ).reset_index()
    
    monthly["year_month"] = monthly["year_month"].astype(str)
    
    return monthly
