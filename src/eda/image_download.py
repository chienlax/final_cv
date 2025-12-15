"""
Image download utilities for multimodal analysis.

Provides utilities to:
- Download sample images from Amazon product URLs
- Validate and check image quality
- Save images for visual feature extraction
"""

import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)


@dataclass
class ImageDownloadStats:
    """Statistics for image download operation."""
    
    total_urls: int = 0
    downloaded: int = 0
    failed: int = 0
    skipped_existing: int = 0
    total_bytes: int = 0
    elapsed_seconds: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_urls == 0:
            return 0.0
        return self.downloaded / self.total_urls * 100
    
    def __repr__(self) -> str:
        return (
            f"ImageDownloadStats(downloaded={self.downloaded}/{self.total_urls} "
            f"({self.success_rate:.1f}%), failed={self.failed}, "
            f"size={self.total_bytes / 1024 / 1024:.1f}MB, time={self.elapsed_seconds:.1f}s)"
        )


def download_images_sample(
    metadata_df: pd.DataFrame,
    output_dir: Path,
    sample_size: int = 1000,
    max_images_per_item: int = 1,
    max_workers: int = 8,
    timeout: int = 10,
    seed: int = 42,
    skip_existing: bool = True,
) -> ImageDownloadStats:
    """
    Download a sample of product images for analysis.
    
    Args:
        metadata_df: DataFrame with 'item_id' and 'image_urls' columns.
        output_dir: Directory to save images.
        sample_size: Number of items to sample.
        max_images_per_item: Maximum images to download per item.
        max_workers: Number of concurrent download threads.
        timeout: Request timeout in seconds.
        seed: Random seed for sampling.
        skip_existing: Skip downloading if file already exists.
        
    Returns:
        ImageDownloadStats with download statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading sample of {sample_size} items to {output_dir}")
    
    # Filter items with images
    df_with_images = metadata_df[
        metadata_df["image_urls"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    ].copy()
    
    if len(df_with_images) == 0:
        logger.warning("No items with images found")
        return ImageDownloadStats()
    
    # Sample items
    sample_df = df_with_images.sample(
        n=min(sample_size, len(df_with_images)),
        random_state=seed,
    )
    
    # Prepare download tasks
    tasks = []
    for _, row in sample_df.iterrows():
        item_id = row["item_id"]
        urls = row["image_urls"]
        
        for i, url in enumerate(urls[:max_images_per_item]):
            if url and isinstance(url, str) and url.startswith("http"):
                filename = f"{item_id}_{i}.jpg"
                filepath = output_dir / filename
                tasks.append((item_id, url, filepath))
    
    logger.info(f"Prepared {len(tasks)} download tasks")
    
    # Download with threading
    stats = ImageDownloadStats(total_urls=len(tasks))
    start_time = time.time()
    
    def download_single(task: tuple) -> tuple[bool, int]:
        item_id, url, filepath = task
        
        if skip_existing and filepath.exists():
            return "skipped", 0
        
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Validate it's an image
            content_type = response.headers.get("content-type", "")
            if "image" not in content_type.lower():
                return "failed", 0
            
            # Download and save
            image_data = response.content
            
            # Verify with PIL
            img = Image.open(BytesIO(image_data))
            img.verify()
            
            # Re-open and save (verify() invalidates the image)
            img = Image.open(BytesIO(image_data))
            img = img.convert("RGB")  # Ensure RGB for consistency
            img.save(filepath, "JPEG", quality=85)
            
            return "success", len(image_data)
            
        except Exception as e:
            logger.debug(f"Failed to download {url}: {e}")
            return "failed", 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single, task): task for task in tasks}
        
        for i, future in enumerate(as_completed(futures)):
            result, size = future.result()
            
            if result == "success":
                stats.downloaded += 1
                stats.total_bytes += size
            elif result == "skipped":
                stats.skipped_existing += 1
            else:
                stats.failed += 1
            
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{len(tasks)} ({stats.downloaded} downloaded, {stats.failed} failed)")
    
    stats.elapsed_seconds = time.time() - start_time
    
    logger.info(f"Download complete: {stats}")
    
    return stats


def validate_downloaded_images(
    image_dir: Path,
    min_size_bytes: int = 1000,
    min_dimension: int = 50,
) -> pd.DataFrame:
    """
    Validate downloaded images and collect statistics.
    
    Args:
        image_dir: Directory containing downloaded images.
        min_size_bytes: Minimum file size to be considered valid.
        min_dimension: Minimum width/height in pixels.
        
    Returns:
        DataFrame with validation results per image.
    """
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        logger.warning(f"Image directory not found: {image_dir}")
        return pd.DataFrame()
    
    results = []
    
    for filepath in image_dir.glob("*.jpg"):
        try:
            file_size = filepath.stat().st_size
            
            with Image.open(filepath) as img:
                width, height = img.size
                mode = img.mode
            
            is_valid = (
                file_size >= min_size_bytes and
                width >= min_dimension and
                height >= min_dimension
            )
            
            results.append({
                "filename": filepath.name,
                "item_id": filepath.stem.rsplit("_", 1)[0],
                "file_size_bytes": file_size,
                "width": width,
                "height": height,
                "mode": mode,
                "is_valid": is_valid,
            })
            
        except Exception as e:
            results.append({
                "filename": filepath.name,
                "item_id": filepath.stem.rsplit("_", 1)[0],
                "file_size_bytes": filepath.stat().st_size if filepath.exists() else 0,
                "width": 0,
                "height": 0,
                "mode": "error",
                "is_valid": False,
                "error": str(e),
            })
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        n_valid = df["is_valid"].sum()
        logger.info(f"Validated {len(df)} images: {n_valid} valid ({n_valid/len(df)*100:.1f}%)")
    
    return df


def compute_image_statistics(
    image_dir: Path,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """
    Compute statistics about downloaded images.
    
    Args:
        image_dir: Directory containing images.
        sample_size: Optional sample size for analysis.
        seed: Random seed.
        
    Returns:
        Dictionary with image statistics.
    """
    image_dir = Path(image_dir)
    
    image_files = list(image_dir.glob("*.jpg"))
    
    if len(image_files) == 0:
        return {"error": "No images found"}
    
    if sample_size and sample_size < len(image_files):
        import random
        random.seed(seed)
        image_files = random.sample(image_files, sample_size)
    
    widths = []
    heights = []
    file_sizes = []
    aspect_ratios = []
    
    for filepath in image_files:
        try:
            file_sizes.append(filepath.stat().st_size)
            
            with Image.open(filepath) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h if h > 0 else 0)
        except Exception:
            continue
    
    if len(widths) == 0:
        return {"error": "Could not read any images"}
    
    import numpy as np
    
    return {
        "n_images": len(image_files),
        "n_analyzed": len(widths),
        "width": {
            "mean": float(np.mean(widths)),
            "median": float(np.median(widths)),
            "min": int(np.min(widths)),
            "max": int(np.max(widths)),
        },
        "height": {
            "mean": float(np.mean(heights)),
            "median": float(np.median(heights)),
            "min": int(np.min(heights)),
            "max": int(np.max(heights)),
        },
        "file_size_kb": {
            "mean": float(np.mean(file_sizes) / 1024),
            "median": float(np.median(file_sizes) / 1024),
            "total_mb": float(sum(file_sizes) / 1024 / 1024),
        },
        "aspect_ratio": {
            "mean": float(np.mean(aspect_ratios)),
            "median": float(np.median(aspect_ratios)),
        },
    }
