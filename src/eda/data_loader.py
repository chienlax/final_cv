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
    
    def __repr__(self) -> str:
        return (
            f"DataStats(total={self.total_records:,}, sampled={self.sampled_records:,}, "
            f"ratio={self.sample_ratio:.2%}, memory={self.memory_mb:.1f}MB)"
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
