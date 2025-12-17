"""
Sentence-BERT Text Embedding Extractor.

Extracts text embeddings using Sentence-Transformers for semantic analysis.
Optimized with batch processing and GPU acceleration.

Reference: all-mpnet-base-v2 - best quality general-purpose sentence embeddings
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional imports with availability flags
TORCH_AVAILABLE = False
SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class TextEmbeddingResult:
    """Results from text embedding extraction."""
    
    n_items_attempted: int = 0
    n_items_successful: int = 0
    n_items_failed: int = 0
    embedding_dim: int = 0
    model_name: str = ""
    device_used: str = "cpu"
    processing_time_sec: float = 0.0
    items_per_second: float = 0.0
    
    # Text statistics
    avg_text_length: float = 0.0
    text_columns_used: list[str] = field(default_factory=list)
    
    # Mapping
    item_ids: list[str] = field(default_factory=list)
    item_indices: dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_items_attempted": self.n_items_attempted,
            "n_items_successful": self.n_items_successful,
            "n_items_failed": self.n_items_failed,
            "success_rate": round(self.n_items_successful / max(1, self.n_items_attempted) * 100, 2),
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "device_used": self.device_used,
            "processing_time_sec": round(self.processing_time_sec, 2),
            "items_per_second": round(self.items_per_second, 2),
            "avg_text_length": round(self.avg_text_length, 1),
            "text_columns_used": self.text_columns_used,
        }


def get_device() -> str:
    """Get the best available device (CUDA > CPU)."""
    if not TORCH_AVAILABLE:
        return "cpu"
    
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"  Using GPU: {gpu_name}")
    else:
        device = "cpu"
        logger.info("  Using CPU (no CUDA available)")
    
    return device


def prepare_item_text(
    row: pd.Series,
    text_columns: list[str],
    max_length: int = 512,
) -> str:
    """
    Prepare concatenated text from multiple columns for an item.
    
    Args:
        row: DataFrame row with text columns.
        text_columns: List of column names to concatenate.
        max_length: Maximum character length (truncate if longer).
        
    Returns:
        Concatenated text string.
    """
    parts = []
    
    for col in text_columns:
        if col not in row.index:
            continue
            
        value = row[col]
        
        if value is None:
            continue
        elif isinstance(value, list):
            # Handle list columns like 'features' or 'description'
            text = " ".join(str(v) for v in value if v)
        elif isinstance(value, str):
            text = value
        else:
            text = str(value)
        
        if text.strip():
            parts.append(text.strip())
    
    combined = " | ".join(parts)
    
    # Truncate if too long (SBERT has max token limit)
    if len(combined) > max_length:
        combined = combined[:max_length]
    
    return combined


def extract_text_embeddings(
    metadata_df: pd.DataFrame,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 128,
    text_columns: list[str] = None,
    max_items: int = 10000,
    max_text_length: int = 512,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, int], TextEmbeddingResult]:
    """
    Extract Sentence-BERT embeddings for item text.
    
    Concatenates title + description + features into single text and encodes
    with a pretrained Sentence-Transformer model.
    
    Args:
        metadata_df: DataFrame with 'item_id' and text columns.
        model_name: Sentence-Transformer model to use.
        batch_size: Batch size for encoding.
        text_columns: List of text columns to use (default: title, description, features).
        max_items: Maximum items to process.
        max_text_length: Max characters per item.
        seed: Random seed for sampling.
        
    Returns:
        Tuple of (embeddings array, item_indices dict, result stats).
    """
    start_time = time.time()
    
    if text_columns is None:
        text_columns = ["title", "description", "features"]
    
    logger.info("Extracting Sentence-BERT text embeddings...")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Text columns: {text_columns}")
    logger.info(f"  Batch size: {batch_size}")
    
    result = TextEmbeddingResult()
    result.model_name = model_name
    result.text_columns_used = text_columns
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        return np.array([]), {}, result
    
    # Get device
    device = get_device()
    result.device_used = device
    
    # Load model
    logger.info(f"  Loading Sentence-BERT model...")
    try:
        model = SentenceTransformer(model_name, device=device)
        result.embedding_dim = model.get_sentence_embedding_dimension()
        logger.info(f"  Model loaded (dim={result.embedding_dim})")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return np.array([]), {}, result
    
    # Sample if necessary
    if len(metadata_df) > max_items:
        logger.info(f"  Sampling {max_items} items from {len(metadata_df)}")
        metadata_df = metadata_df.sample(n=max_items, random_state=seed)
    
    result.n_items_attempted = len(metadata_df)
    
    # Prepare texts for all items
    logger.info(f"  Preparing text for {result.n_items_attempted} items...")
    texts = []
    item_ids = []
    text_lengths = []
    
    for _, row in metadata_df.iterrows():
        item_id = row.get("item_id", row.get("parent_asin", ""))
        text = prepare_item_text(row, text_columns, max_text_length)
        
        if text.strip():
            texts.append(text)
            item_ids.append(item_id)
            text_lengths.append(len(text))
        else:
            result.n_items_failed += 1
    
    if len(texts) == 0:
        logger.warning("  No texts to encode")
        return np.array([]), {}, result
    
    result.avg_text_length = np.mean(text_lengths)
    logger.info(f"  Avg text length: {result.avg_text_length:.0f} chars")
    
    # Encode all texts in batches
    logger.info(f"  Encoding {len(texts)} texts...")
    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        return np.array([]), {}, result
    
    # Build results
    result.n_items_successful = len(item_ids)
    result.item_ids = item_ids
    result.item_indices = {item_id: idx for idx, item_id in enumerate(item_ids)}
    
    # Calculate timing stats
    result.processing_time_sec = time.time() - start_time
    result.items_per_second = result.n_items_successful / max(0.1, result.processing_time_sec)
    
    logger.info(f"  Extracted {result.n_items_successful} embeddings (dim={result.embedding_dim})")
    logger.info(f"  Total time: {result.processing_time_sec:.1f}s ({result.items_per_second:.1f} items/sec)")
    
    return embeddings, result.item_indices, result


def create_dummy_text_embeddings(
    metadata_df: pd.DataFrame,
    embedding_dim: int = 768,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Create dummy random text embeddings for testing.
    
    Args:
        metadata_df: DataFrame with 'item_id' column.
        embedding_dim: Embedding dimension.
        seed: Random seed.
        
    Returns:
        Tuple of (embeddings array, item_indices dict).
    """
    np.random.seed(seed)
    
    item_ids = metadata_df.get("item_id", metadata_df.get("parent_asin", pd.Series())).tolist()
    n_items = len(item_ids)
    
    # Generate random unit vectors
    embeddings = np.random.randn(n_items, embedding_dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    item_indices = {item_id: idx for idx, item_id in enumerate(item_ids)}
    
    logger.info(f"  Created {n_items} dummy text embeddings (dim={embedding_dim})")
    
    return embeddings, item_indices
