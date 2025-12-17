"""
CLIP Embedding Extractor for Visual Features.

Extracts CLIP image embeddings for multimodal recommendation analysis.
Supports GPU acceleration when available.

Optimized with:
- Parallel image downloads (ThreadPoolExecutor)
- Background prefetching of next batch
- Increased batch size for GPU efficiency
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional, Any
import io
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional imports with availability flags
TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
PIL_AVAILABLE = False
REQUESTS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import CLIPProcessor, CLIPModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    pass

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class EmbeddingExtractionResult:
    """Results from embedding extraction."""
    
    n_items_attempted: int = 0
    n_items_successful: int = 0
    n_items_failed: int = 0
    embedding_dim: int = 0
    device_used: str = "cpu"
    processing_time_sec: float = 0.0
    items_per_second: float = 0.0
    
    # Mapping and embeddings
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
            "device_used": self.device_used,
            "processing_time_sec": round(self.processing_time_sec, 2),
            "items_per_second": round(self.items_per_second, 2),
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


def load_image_from_url(url: str, timeout: int = 10) -> Optional["Image.Image"]:
    """
    Load an image from a URL.
    
    Args:
        url: Image URL.
        timeout: Request timeout in seconds.
        
    Returns:
        PIL Image or None if failed.
    """
    if not PIL_AVAILABLE or not REQUESTS_AVAILABLE:
        return None
    
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content))
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
    except Exception as e:
        logger.debug(f"Failed to load image from {url}: {e}")
        return None


def download_item_image(row: pd.Series, timeout: int = 10) -> tuple[str, Optional["Image.Image"]]:
    """
    Download image for a single item, trying multiple URLs.
    
    Args:
        row: DataFrame row with 'item_id' and 'image_urls'.
        timeout: Request timeout in seconds.
        
    Returns:
        Tuple of (item_id, image or None).
    """
    item_id = row["item_id"]
    image_urls = row["image_urls"]
    
    # Try first 3 URLs
    for url in image_urls[:3]:
        image = load_image_from_url(url, timeout=timeout)
        if image is not None:
            return (item_id, image)
    
    return (item_id, None)


def download_batch_images_parallel(
    batch_df: pd.DataFrame,
    max_workers: int = 16,
    timeout: int = 10,
) -> list[tuple[str, Optional["Image.Image"]]]:
    """
    Download images for a batch in parallel using ThreadPoolExecutor.
    
    Args:
        batch_df: DataFrame batch with 'item_id' and 'image_urls'.
        max_workers: Number of parallel download threads.
        timeout: Request timeout per image.
        
    Returns:
        List of (item_id, image or None) tuples.
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_item_image, row, timeout): row["item_id"]
            for _, row in batch_df.iterrows()
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                item_id = futures[future]
                logger.debug(f"Download failed for {item_id}: {e}")
                results.append((item_id, None))
    
    return results


def prefetch_worker(
    batch_queue: Queue,
    image_queue: Queue,
    max_workers: int = 16,
    timeout: int = 10,
):
    """
    Background worker to prefetch images for next batch.
    
    Runs in a separate thread, continuously downloading batches
    while the main thread processes them on GPU.
    
    Args:
        batch_queue: Input queue of batch DataFrames (None = stop signal).
        image_queue: Output queue of downloaded images.
        max_workers: Parallel download threads.
        timeout: Request timeout per image.
    """
    while True:
        batch_df = batch_queue.get()
        
        if batch_df is None:  # Poison pill - stop signal
            image_queue.put(None)
            break
        
        # Download images in parallel
        images = download_batch_images_parallel(batch_df, max_workers, timeout)
        image_queue.put(images)


def extract_clip_embeddings(
    metadata_df: pd.DataFrame,
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 128,
    max_items: int = 5000,
    download_workers: int = 16,
    download_timeout: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, int], EmbeddingExtractionResult]:
    """
    Extract CLIP embeddings for items from image URLs.
    
    Optimized with parallel downloads and batch prefetching for maximum throughput.
    
    Args:
        metadata_df: DataFrame with 'item_id' and 'image_urls' columns.
        model_name: CLIP model to use.
        batch_size: Batch size for GPU inference (default: 128).
        max_items: Maximum items to process.
        download_workers: Number of parallel download threads (default: 16).
        download_timeout: Timeout per image download in seconds.
        seed: Random seed for sampling.
        
    Returns:
        Tuple of (embeddings array, item_indices dict, result stats).
    """
    start_time = time.time()
    logger.info("Extracting CLIP embeddings (optimized parallel mode)...")
    logger.info(f"  Settings: batch_size={batch_size}, download_workers={download_workers}")
    
    result = EmbeddingExtractionResult()
    
    if not all([TORCH_AVAILABLE, TRANSFORMERS_AVAILABLE, PIL_AVAILABLE, REQUESTS_AVAILABLE]):
        missing = []
        if not TORCH_AVAILABLE:
            missing.append("torch")
        if not TRANSFORMERS_AVAILABLE:
            missing.append("transformers")
        if not PIL_AVAILABLE:
            missing.append("Pillow")
        if not REQUESTS_AVAILABLE:
            missing.append("requests")
        
        logger.error(f"Missing dependencies for CLIP: {', '.join(missing)}")
        return np.array([]), {}, result
    
    # Get device
    device = get_device()
    result.device_used = device
    
    # Load model
    logger.info(f"  Loading CLIP model: {model_name}")
    try:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {e}")
        return np.array([]), {}, result
    
    # Filter items with images
    items_with_images = metadata_df[
        metadata_df["image_urls"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    ].copy()
    
    # Sample if necessary
    if len(items_with_images) > max_items:
        logger.info(f"  Sampling {max_items} items from {len(items_with_images)}")
        items_with_images = items_with_images.sample(n=max_items, random_state=seed)
    
    result.n_items_attempted = len(items_with_images)
    logger.info(f"  Processing {result.n_items_attempted} items with prefetching...")
    
    embeddings_list = []
    item_ids_list = []
    
    # Create batch list
    batches = []
    for batch_start in range(0, len(items_with_images), batch_size):
        batch_end = min(batch_start + batch_size, len(items_with_images))
        batch_df = items_with_images.iloc[batch_start:batch_end]
        batches.append(batch_df)
    
    # Setup prefetch queues
    batch_queue = Queue(maxsize=2)  # Max 2 batches prefetched
    image_queue = Queue(maxsize=2)
    
    # Start prefetch worker thread
    prefetch_thread = Thread(
        target=prefetch_worker,
        args=(batch_queue, image_queue, download_workers, download_timeout),
        daemon=True,
    )
    prefetch_thread.start()
    
    # Enqueue first batch for prefetching
    if len(batches) > 0:
        batch_queue.put(batches[0])
    
    # Process batches with prefetching
    for batch_idx, batch_df in enumerate(batches):
        # Enqueue next batch for prefetching (while we process current)
        if batch_idx + 1 < len(batches):
            batch_queue.put(batches[batch_idx + 1])
        else:
            # No more batches, send stop signal
            batch_queue.put(None)
        
        # Get prefetched images for current batch
        downloaded = image_queue.get()
        
        if downloaded is None:
            break
        
        # Separate successful downloads
        batch_images = []
        batch_item_ids = []
        
        for item_id, image in downloaded:
            if image is not None:
                batch_images.append(image)
                batch_item_ids.append(item_id)
            else:
                result.n_items_failed += 1
        
        if len(batch_images) == 0:
            continue
        
        # Extract embeddings on GPU
        try:
            with torch.no_grad():
                inputs = processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                image_features = model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                embeddings_list.append(image_features.cpu().numpy())
                item_ids_list.extend(batch_item_ids)
        except Exception as e:
            logger.warning(f"  Batch {batch_idx} GPU inference failed: {e}")
            result.n_items_failed += len(batch_images)
            continue
        
        # Progress logging
        processed = (batch_idx + 1) * batch_size
        elapsed = time.time() - start_time
        rate = len(item_ids_list) / elapsed if elapsed > 0 else 0
        
        if batch_idx % 5 == 0 or batch_idx == len(batches) - 1:
            logger.info(f"  Batch {batch_idx+1}/{len(batches)}: {len(item_ids_list)} embeddings ({rate:.1f} items/sec)")
    
    # Wait for prefetch thread to finish
    prefetch_thread.join(timeout=5)
    
    if len(embeddings_list) == 0:
        logger.warning("  No embeddings extracted")
        return np.array([]), {}, result
    
    # Combine all embeddings
    embeddings = np.vstack(embeddings_list)
    result.n_items_successful = len(item_ids_list)
    result.embedding_dim = embeddings.shape[1]
    result.item_ids = item_ids_list
    result.item_indices = {item_id: idx for idx, item_id in enumerate(item_ids_list)}
    
    # Calculate timing stats
    result.processing_time_sec = time.time() - start_time
    result.items_per_second = result.n_items_successful / max(0.1, result.processing_time_sec)
    
    logger.info(f"  Extracted {result.n_items_successful} embeddings (dim={result.embedding_dim})")
    logger.info(f"  Total time: {result.processing_time_sec:.1f}s ({result.items_per_second:.1f} items/sec)")
    
    return embeddings, result.item_indices, result


def create_dummy_embeddings(
    metadata_df: pd.DataFrame,
    embedding_dim: int = 512,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Create dummy random embeddings for testing.
    
    Args:
        metadata_df: DataFrame with 'item_id' column.
        embedding_dim: Embedding dimension.
        seed: Random seed.
        
    Returns:
        Tuple of (embeddings array, item_indices dict).
    """
    np.random.seed(seed)
    
    item_ids = metadata_df["item_id"].tolist()
    n_items = len(item_ids)
    
    # Generate random unit vectors
    embeddings = np.random.randn(n_items, embedding_dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    item_indices = {item_id: idx for idx, item_id in enumerate(item_ids)}
    
    logger.info(f"  Created {n_items} dummy embeddings (dim={embedding_dim})")
    
    return embeddings, item_indices
