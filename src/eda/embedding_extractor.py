"""
CLIP Embedding Extractor for Visual Features.

Extracts CLIP image embeddings for multimodal recommendation analysis.
Supports GPU acceleration when available.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import io

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


def extract_clip_embeddings(
    metadata_df: pd.DataFrame,
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
    max_items: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, int], EmbeddingExtractionResult]:
    """
    Extract CLIP embeddings for items from image URLs.
    
    Args:
        metadata_df: DataFrame with 'item_id' and 'image_urls' columns.
        model_name: CLIP model to use.
        batch_size: Batch size for inference.
        max_items: Maximum items to process.
        seed: Random seed for sampling.
        
    Returns:
        Tuple of (embeddings array, item_indices dict, result stats).
    """
    logger.info("Extracting CLIP embeddings...")
    
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
    logger.info(f"  Processing {result.n_items_attempted} items...")
    
    embeddings_list = []
    item_ids_list = []
    
    # Process in batches
    for batch_start in range(0, len(items_with_images), batch_size):
        batch_end = min(batch_start + batch_size, len(items_with_images))
        batch_df = items_with_images.iloc[batch_start:batch_end]
        
        batch_images = []
        batch_item_ids = []
        
        for _, row in batch_df.iterrows():
            item_id = row["item_id"]
            image_urls = row["image_urls"]
            
            # Try to load first available image
            image = None
            for url in image_urls[:3]:  # Try first 3 URLs
                image = load_image_from_url(url)
                if image is not None:
                    break
            
            if image is not None:
                batch_images.append(image)
                batch_item_ids.append(item_id)
            else:
                result.n_items_failed += 1
        
        if len(batch_images) == 0:
            continue
        
        # Extract embeddings
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
            logger.warning(f"  Batch {batch_start}-{batch_end} failed: {e}")
            result.n_items_failed += len(batch_images)
            continue
        
        if (batch_start // batch_size) % 10 == 0:
            logger.info(f"  Processed {batch_end}/{result.n_items_attempted} items...")
    
    if len(embeddings_list) == 0:
        logger.warning("  No embeddings extracted")
        return np.array([]), {}, result
    
    # Combine all embeddings
    embeddings = np.vstack(embeddings_list)
    result.n_items_successful = len(item_ids_list)
    result.embedding_dim = embeddings.shape[1]
    result.item_ids = item_ids_list
    result.item_indices = {item_id: idx for idx, item_id in enumerate(item_ids_list)}
    
    logger.info(f"  Extracted {result.n_items_successful} embeddings (dim={result.embedding_dim})")
    
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
