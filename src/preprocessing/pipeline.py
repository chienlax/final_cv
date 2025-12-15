"""
Main preprocessing pipeline for multimodal recommendation.

Implements:
- K-core filtering (iterative)
- User/Item ID encoding
- Train/val/test splitting (temporal or random)
- Data serialization
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import PreprocessConfig

logger = logging.getLogger(__name__)


def apply_kcore_filter(
    df: pd.DataFrame,
    k: int = 5,
    max_iterations: int = 100,
) -> pd.DataFrame:
    """
    Apply iterative k-core filtering to ensure all users/items have >= k interactions.
    
    Args:
        df: DataFrame with 'user_id' and 'item_id' columns.
        k: Minimum interaction threshold.
        max_iterations: Maximum convergence iterations.
        
    Returns:
        Filtered DataFrame.
    """
    logger.info(f"Applying k-core filtering with k={k}...")
    
    n_original = len(df)
    n_users_original = df["user_id"].nunique()
    n_items_original = df["item_id"].nunique()
    
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
            logger.info(f"K-core converged after {iteration + 1} iterations")
            break
    
    n_users_after = df["user_id"].nunique()
    n_items_after = df["item_id"].nunique()
    
    logger.info(
        f"K-core filtering: {n_original:,} -> {len(df):,} interactions "
        f"({len(df)/n_original*100:.1f}% retained)"
    )
    logger.info(
        f"  Users: {n_users_original:,} -> {n_users_after:,} "
        f"({n_users_after/n_users_original*100:.1f}%)"
    )
    logger.info(
        f"  Items: {n_items_original:,} -> {n_items_after:,} "
        f"({n_items_after/n_items_original*100:.1f}%)"
    )
    
    return df.reset_index(drop=True)


def create_id_mappings(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """
    Create integer mappings for user_id and item_id.
    
    Args:
        df: DataFrame with 'user_id' and 'item_id' columns.
        
    Returns:
        Tuple of (df_with_ids, user_to_idx, item_to_idx).
    """
    logger.info("Creating ID mappings...")
    
    # Create mappings
    unique_users = df["user_id"].unique()
    unique_items = df["item_id"].unique()
    
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
    
    # Apply mappings
    df = df.copy()
    df["user_idx"] = df["user_id"].map(user_to_idx)
    df["item_idx"] = df["item_id"].map(item_to_idx)
    
    logger.info(f"Created mappings: {len(user_to_idx):,} users, {len(item_to_idx):,} items")
    
    return df, user_to_idx, item_to_idx


def create_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    by_time: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split interactions into train/val/test sets.
    
    Args:
        df: DataFrame with interactions.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        by_time: If True, split by timestamp (more realistic). If False, random split.
        seed: Random seed.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    logger.info(f"Splitting data: train={train_ratio}, val={val_ratio}, test={test_ratio}, by_time={by_time}")
    
    if by_time and "timestamp" in df.columns:
        # Sort by timestamp
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        
        n = len(df_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]
    else:
        # Random split
        np.random.seed(seed)
        shuffled_idx = np.random.permutation(len(df))
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[shuffled_idx[:train_end]]
        val_df = df.iloc[shuffled_idx[train_end:val_end]]
        test_df = df.iloc[shuffled_idx[val_end:]]
    
    logger.info(f"Split sizes: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    
    return train_df, val_df, test_df


def save_processed_data(
    output_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    config: PreprocessConfig,
    metadata_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Save processed data to disk.
    
    Args:
        output_dir: Directory to save files.
        train_df, val_df, test_df: Split DataFrames.
        user_to_idx, item_to_idx: ID mappings.
        config: Preprocessing configuration.
        metadata_df: Optional filtered metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving processed data to {output_dir}")
    
    # Save splits as parquet (efficient)
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    
    # Save mappings as JSON
    with open(output_dir / "user_to_idx.json", "w") as f:
        json.dump(user_to_idx, f)
    
    with open(output_dir / "item_to_idx.json", "w") as f:
        json.dump(item_to_idx, f)
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Save metadata if provided
    if metadata_df is not None:
        metadata_df.to_parquet(output_dir / "metadata.parquet", index=False)
    
    # Save summary statistics
    stats = {
        "n_users": len(user_to_idx),
        "n_items": len(item_to_idx),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_total": len(train_df) + len(val_df) + len(test_df),
    }
    
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved: {len(train_df):,} train, {len(val_df):,} val, {len(test_df):,} test")


def load_processed_data(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Load processed data from disk.
    
    Args:
        data_dir: Directory containing processed files.
        
    Returns:
        Tuple of (train_df, val_df, test_df, user_to_idx, item_to_idx).
    """
    data_dir = Path(data_dir)
    
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    
    with open(data_dir / "user_to_idx.json") as f:
        user_to_idx = json.load(f)
    
    with open(data_dir / "item_to_idx.json") as f:
        item_to_idx = json.load(f)
    
    logger.info(f"Loaded: {len(train_df):,} train, {len(val_df):,} val, {len(test_df):,} test")
    
    return train_df, val_df, test_df, user_to_idx, item_to_idx


def filter_metadata_by_items(
    metadata_df: pd.DataFrame,
    valid_items: set[str],
) -> pd.DataFrame:
    """
    Filter metadata to only include items present in interactions.
    
    Args:
        metadata_df: Full metadata DataFrame.
        valid_items: Set of item IDs to keep.
        
    Returns:
        Filtered metadata DataFrame.
    """
    filtered = metadata_df[metadata_df["item_id"].isin(valid_items)].copy()
    logger.info(f"Filtered metadata: {len(metadata_df):,} -> {len(filtered):,} items")
    return filtered.reset_index(drop=True)
