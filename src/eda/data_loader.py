"""
Data loader for Amazon Review 2023 dataset.

Provides memory-efficient streaming and sampling utilities for large JSONL.gz files.
Implements hash-based sampling for reproducibility.
"""

import gzip
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DataStats:
    """Summary statistics for loaded data."""
    
    total_records: int = 0
    sampled_records: int = 0
    sample_ratio: float = 1.0
    columns: list[str] = field(default_factory=list)
    memory_mb: float = 0.0
    sampling_method: str = "random"  # random, kcore, temporal, dense
    sampling_params: dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (
            f"DataStats(total={self.total_records:,}, sampled={self.sampled_records:,}, "
            f"ratio={self.sample_ratio:.2%}, method={self.sampling_method}, memory={self.memory_mb:.1f}MB)"
        )


def stream_jsonl(file_path: Path) -> Generator[dict[str, Any], None, None]:
    """
    Generator that streams records from a JSONL file (compressed or uncompressed).
    
    Automatically handles .jsonl.gz (gzip) and .jsonl (plain) files.
    Prefers uncompressed version if both exist for faster loading.
    
    Args:
        file_path: Path to the .jsonl or .jsonl.gz file.
        
    Yields:
        Dictionary record for each line in the file.
    """
    file_path = Path(file_path)
    
    # Check for uncompressed version first (faster)
    if file_path.suffix == ".gz":
        uncompressed_path = file_path.with_suffix("")  # Remove .gz
        if uncompressed_path.exists():
            logger.info(f"  Using uncompressed file: {uncompressed_path.name}")
            file_path = uncompressed_path
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Choose appropriate opener
    is_gzipped = file_path.suffix == ".gz"
    opener = gzip.open if is_gzipped else open
    
    with opener(file_path, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                yield json.loads(line.strip())
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                continue


# Keep old name for backward compatibility
stream_jsonl_gz = stream_jsonl


def _hash_sample(key: str, mod: int, seed: int = 42) -> bool:
    """
    Deterministic hash-based sampling.
    
    Uses MD5 hash of key + seed for reproducible sampling.
    
    Args:
        key: String key to hash (e.g., user_id or item_id).
        mod: Modulo divisor (e.g., 10 for 10% sample).
        seed: Random seed for reproducibility.
        
    Returns:
        True if the key should be included in the sample.
    """
    hash_input = f"{key}_{seed}".encode("utf-8")
    hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
    return (hash_val % mod) == 0


def load_interactions_sample(
    file_path: Path,
    sample_ratio: float = 0.1,
    seed: int = 42,
    max_records: Optional[int] = None,
) -> tuple[pd.DataFrame, DataStats]:
    """
    Load interaction data with hash-based sampling.
    
    Samples based on user_id for consistent user sampling.
    Expected schema: user_id, parent_asin, rating, timestamp, text, title, ...
    
    Args:
        file_path: Path to the interaction .jsonl.gz file.
        sample_ratio: Fraction of data to sample (0.0 to 1.0).
        seed: Random seed for reproducibility.
        max_records: Optional limit on total records to process.
        
    Returns:
        Tuple of (DataFrame, DataStats).
    """
    file_path = Path(file_path)
    mod = int(1 / sample_ratio) if sample_ratio < 1.0 else 1
    
    records: list[dict[str, Any]] = []
    total_count = 0
    
    logger.info(f"Loading interactions from {file_path.name} with {sample_ratio:.0%} sampling...")
    
    for record in stream_jsonl_gz(file_path):
        total_count += 1
        
        if max_records and total_count > max_records:
            break
        
        # Sample by user_id for consistent user sampling
        user_id = record.get("user_id", "")
        if sample_ratio >= 1.0 or _hash_sample(user_id, mod, seed):
            # Extract relevant fields with defaults
            processed = {
                "user_id": user_id,
                "item_id": record.get("parent_asin", record.get("asin", "")),
                "rating": record.get("rating", np.nan),
                "timestamp": record.get("timestamp", 0),
                "review_text": record.get("text", ""),
                "review_title": record.get("title", ""),
                "verified_purchase": record.get("verified_purchase", False),
                "helpful_vote": record.get("helpful_vote", 0),
            }
            records.append(processed)
        
        if total_count % 500_000 == 0:
            logger.info(f"  Processed {total_count:,} records, sampled {len(records):,}")
    
    df = pd.DataFrame(records)
    
    # Convert timestamp to datetime
    if "timestamp" in df.columns and len(df) > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    
    stats = DataStats(
        total_records=total_count,
        sampled_records=len(df),
        sample_ratio=len(df) / total_count if total_count > 0 else 0.0,
        columns=list(df.columns),
        memory_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
    )
    
    logger.info(f"Loaded {stats}")
    return df, stats


def load_metadata_sample(
    file_path: Path,
    item_ids: Optional[set[str]] = None,
    sample_ratio: float = 0.1,
    seed: int = 42,
    max_records: Optional[int] = None,
) -> tuple[pd.DataFrame, DataStats]:
    """
    Load item metadata with optional filtering by item_ids.
    
    Expected schema: parent_asin, title, description, features, images, ...
    
    Args:
        file_path: Path to the metadata .jsonl.gz file.
        item_ids: Optional set of item IDs to filter (for alignment with interactions).
        sample_ratio: Fraction to sample if item_ids not provided.
        seed: Random seed for reproducibility.
        max_records: Optional limit on total records to process.
        
    Returns:
        Tuple of (DataFrame, DataStats).
    """
    file_path = Path(file_path)
    mod = int(1 / sample_ratio) if sample_ratio < 1.0 else 1
    
    records: list[dict[str, Any]] = []
    total_count = 0
    
    logger.info(f"Loading metadata from {file_path.name}...")
    
    for record in stream_jsonl_gz(file_path):
        total_count += 1
        
        if max_records and total_count > max_records:
            break
        
        item_id = record.get("parent_asin", record.get("asin", ""))
        
        # Filter logic: either by item_ids set or by sampling
        include = False
        if item_ids is not None:
            include = item_id in item_ids
        elif sample_ratio >= 1.0:
            include = True
        else:
            include = _hash_sample(item_id, mod, seed)
        
        if include:
            # Extract images - handle nested structure
            images = record.get("images", [])
            image_urls = []
            if isinstance(images, list):
                for img in images:
                    if isinstance(img, dict):
                        # Try different image size keys
                        for key in ["large", "hi_res", "thumb"]:
                            if key in img and img[key]:
                                image_urls.append(img[key])
                                break
                    elif isinstance(img, str):
                        image_urls.append(img)
            
            # Extract features as text
            features = record.get("features", [])
            features_text = " | ".join(features) if isinstance(features, list) else str(features)
            
            # Extract categories
            categories = record.get("categories", [])
            if isinstance(categories, list) and len(categories) > 0:
                main_category = categories[0] if isinstance(categories[0], str) else str(categories[0])
            else:
                main_category = record.get("main_category", "")
            
            processed = {
                "item_id": item_id,
                "title": record.get("title", ""),
                "description": " ".join(record.get("description", [])) if isinstance(record.get("description"), list) else str(record.get("description", "")),
                "features": features_text,
                "main_category": main_category,
                "price": record.get("price", ""),
                "average_rating": record.get("average_rating", np.nan),
                "rating_number": record.get("rating_number", 0),
                "image_count": len(image_urls),
                "image_urls": image_urls[:5],  # Limit to 5 images
                "store": record.get("store", ""),
            }
            records.append(processed)
        
        if total_count % 200_000 == 0:
            logger.info(f"  Processed {total_count:,} records, sampled {len(records):,}")
    
    df = pd.DataFrame(records)
    
    stats = DataStats(
        total_records=total_count,
        sampled_records=len(df),
        sample_ratio=len(df) / total_count if total_count > 0 else 0.0,
        columns=list(df.columns),
        memory_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
    )
    
    logger.info(f"Loaded {stats}")
    return df, stats


def count_total_records(file_path: Path) -> int:
    """
    Count total records in a JSONL.gz file without loading into memory.
    
    Args:
        file_path: Path to the .jsonl.gz file.
        
    Returns:
        Total number of records.
    """
    count = 0
    for _ in stream_jsonl_gz(file_path):
        count += 1
        if count % 1_000_000 == 0:
            logger.info(f"  Counted {count:,} records...")
    return count


# =============================================================================
# Dense Subgraph Sampling Functions
# =============================================================================

def _apply_kcore_filter(
    df: pd.DataFrame,
    k: int,
    max_iterations: int = 100,
) -> pd.DataFrame:
    """
    Apply iterative k-core filtering.
    
    K-core filtering removes users/items with < k interactions until convergence.
    This preserves network density critical for graph-based models like LATTICE.
    
    Args:
        df: DataFrame with 'user_id' and 'item_id' columns.
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
            logger.info(f"  K-Core converged at iteration {iteration + 1}")
            break
    
    return df


def load_interactions_all(
    file_path: Path,
    max_records: Optional[int] = None,
) -> tuple[pd.DataFrame, DataStats]:
    """
    Load ALL interactions without any sampling.
    
    Use with caution on large datasets - may run out of memory.
    
    Args:
        file_path: Path to the interaction .jsonl.gz file.
        max_records: Optional limit on total records to process.
        
    Returns:
        Tuple of (DataFrame, DataStats).
    """
    file_path = Path(file_path)
    records: list[dict[str, Any]] = []
    total_count = 0
    
    logger.info(f"Loading ALL interactions from {file_path.name}...")
    
    for record in stream_jsonl_gz(file_path):
        total_count += 1
        
        if max_records and total_count > max_records:
            break
        
        processed = {
            "user_id": record.get("user_id", ""),
            "item_id": record.get("parent_asin", record.get("asin", "")),
            "rating": record.get("rating", np.nan),
            "timestamp": record.get("timestamp", 0),
            "review_text": record.get("text", ""),
            "review_title": record.get("title", ""),
            "verified_purchase": record.get("verified_purchase", False),
            "helpful_vote": record.get("helpful_vote", 0),
        }
        records.append(processed)
        
        if total_count % 1_000_000 == 0:
            logger.info(f"  Processed {total_count:,} records")
    
    df = pd.DataFrame(records)
    
    if "timestamp" in df.columns and len(df) > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    
    stats = DataStats(
        total_records=total_count,
        sampled_records=len(df),
        sample_ratio=1.0,
        columns=list(df.columns),
        memory_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
        sampling_method="all",
    )
    
    logger.info(f"Loaded {stats}")
    return df, stats


def load_interactions_kcore(
    file_path: Path,
    k: int = 5,
    max_iterations: int = 50,
    max_records: Optional[int] = None,
) -> tuple[pd.DataFrame, DataStats]:
    """
    Load ALL interactions, then apply iterative K-Core filtering.
    
    Preserves network density for graph-based analysis (LATTICE, GCN).
    This is the recommended sampling for structural analysis.
    
    Args:
        file_path: Path to the interaction .jsonl.gz file.
        k: Minimum interactions per user AND per item.
        max_iterations: Maximum k-core convergence iterations.
        max_records: Optional limit on records to load before filtering.
        
    Returns:
        Tuple of (Filtered DataFrame, DataStats).
    """
    logger.info(f"Loading interactions with K-Core filtering (k={k})...")
    
    # Load all data first
    df, load_stats = load_interactions_all(file_path, max_records=max_records)
    
    original_size = len(df)
    original_users = df["user_id"].nunique()
    original_items = df["item_id"].nunique()
    
    # Apply k-core filtering
    logger.info(f"  Applying k-core filter (k={k})...")
    df = _apply_kcore_filter(df, k=k, max_iterations=max_iterations)
    
    final_size = len(df)
    final_users = df["user_id"].nunique()
    final_items = df["item_id"].nunique()
    
    retention_pct = final_size / original_size * 100 if original_size > 0 else 0
    
    logger.info(f"  K-Core filtering: {original_size:,} -> {final_size:,} ({retention_pct:.1f}% retained)")
    logger.info(f"  Users: {original_users:,} -> {final_users:,}, Items: {original_items:,} -> {final_items:,}")
    
    stats = DataStats(
        total_records=load_stats.total_records,
        sampled_records=len(df),
        sample_ratio=len(df) / load_stats.total_records if load_stats.total_records > 0 else 0.0,
        columns=list(df.columns),
        memory_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
        sampling_method="kcore",
        sampling_params={
            "k": k,
            "original_interactions": original_size,
            "original_users": original_users,
            "original_items": original_items,
            "retention_pct": round(retention_pct, 2),
        },
    )
    
    logger.info(f"Loaded {stats}")
    return df, stats


def load_interactions_temporal(
    file_path: Path,
    months: int = 6,
    end_date: Optional[str] = None,
    max_records: Optional[int] = None,
) -> tuple[pd.DataFrame, DataStats]:
    """
    Load interactions from the last N months only.
    
    Time-window sampling preserves temporal density and recent trends.
    More representative than random sampling from a 20-year span.
    
    Args:
        file_path: Path to the interaction .jsonl.gz file.
        months: Number of months to include (from end_date backwards).
        end_date: Optional end date (YYYY-MM-DD). Defaults to latest in data.
        max_records: Optional limit on total records to process.
        
    Returns:
        Tuple of (Filtered DataFrame, DataStats).
    """
    from datetime import datetime, timedelta
    
    logger.info(f"Loading interactions with temporal filtering (last {months} months)...")
    
    # First pass: find date range if end_date not specified
    if end_date is None:
        logger.info("  First pass: finding date range...")
        max_timestamp = 0
        count = 0
        for record in stream_jsonl_gz(file_path):
            ts = record.get("timestamp", 0)
            if ts > max_timestamp:
                max_timestamp = ts
            count += 1
            if max_records and count > max_records:
                break
        
        if max_timestamp > 0:
            end_dt = datetime.fromtimestamp(max_timestamp / 1000)
        else:
            end_dt = datetime.now()
        logger.info(f"  Found max date: {end_dt.strftime('%Y-%m-%d')}")
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Calculate cutoff
    cutoff_dt = end_dt - timedelta(days=months * 30)
    cutoff_ts = int(cutoff_dt.timestamp() * 1000)
    
    logger.info(f"  Filtering: {cutoff_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
    
    # Second pass: load data within time window
    records: list[dict[str, Any]] = []
    total_count = 0
    filtered_count = 0
    
    for record in stream_jsonl_gz(file_path):
        total_count += 1
        
        if max_records and total_count > max_records:
            break
        
        ts = record.get("timestamp", 0)
        if ts >= cutoff_ts:
            processed = {
                "user_id": record.get("user_id", ""),
                "item_id": record.get("parent_asin", record.get("asin", "")),
                "rating": record.get("rating", np.nan),
                "timestamp": ts,
                "review_text": record.get("text", ""),
                "review_title": record.get("title", ""),
                "verified_purchase": record.get("verified_purchase", False),
                "helpful_vote": record.get("helpful_vote", 0),
            }
            records.append(processed)
            filtered_count += 1
        
        if total_count % 1_000_000 == 0:
            logger.info(f"  Processed {total_count:,}, kept {filtered_count:,}")
    
    df = pd.DataFrame(records)
    
    if "timestamp" in df.columns and len(df) > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    
    stats = DataStats(
        total_records=total_count,
        sampled_records=len(df),
        sample_ratio=len(df) / total_count if total_count > 0 else 0.0,
        columns=list(df.columns),
        memory_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
        sampling_method="temporal",
        sampling_params={
            "months": months,
            "start_date": cutoff_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
        },
    )
    
    logger.info(f"Loaded {stats}")
    return df, stats


def load_interactions_dense_subgraph(
    file_path: Path,
    strategy: str = "kcore",
    k: int = 5,
    months: int = 6,
    max_records: Optional[int] = None,
) -> tuple[pd.DataFrame, DataStats]:
    """
    Main entry point for dense subgraph sampling.
    
    Supports multiple strategies:
    - "kcore": K-Core filtering (recommended for structural analysis)
    - "temporal": Time-window filtering (last N months)
    - "dense": Combined approach (temporal first, then k-core)
    
    Args:
        file_path: Path to the interaction .jsonl.gz file.
        strategy: Sampling strategy ("kcore", "temporal", or "dense").
        k: K-Core threshold (for "kcore" and "dense").
        months: Time window in months (for "temporal" and "dense").
        max_records: Optional limit on records.
        
    Returns:
        Tuple of (Filtered DataFrame, DataStats).
    """
    if strategy == "kcore":
        return load_interactions_kcore(file_path, k=k, max_records=max_records)
    elif strategy == "temporal":
        return load_interactions_temporal(file_path, months=months, max_records=max_records)
    elif strategy == "dense":
        # Combined: temporal first, then k-core
        logger.info(f"Loading with dense subgraph (temporal {months}mo + k-core k={k})...")
        
        # Load temporal subset
        df, temp_stats = load_interactions_temporal(file_path, months=months, max_records=max_records)
        
        if len(df) == 0:
            return df, temp_stats
        
        # Apply k-core on temporal subset
        original_size = len(df)
        df = _apply_kcore_filter(df, k=k)
        
        stats = DataStats(
            total_records=temp_stats.total_records,
            sampled_records=len(df),
            sample_ratio=len(df) / temp_stats.total_records if temp_stats.total_records > 0 else 0.0,
            columns=list(df.columns),
            memory_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
            sampling_method="dense",
            sampling_params={
                "months": months,
                "k": k,
                "temporal_size": original_size,
                "kcore_size": len(df),
            },
        )
        
        logger.info(f"Loaded {stats}")
        return df, stats
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}. Use 'kcore', 'temporal', or 'dense'.")

