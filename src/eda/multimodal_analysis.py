"""
Multimodal feature analysis for Amazon Review 2023 dataset.

Analyzes coverage and quality of multimodal features:
- Image availability and URL validation
- Text field coverage (title, description, features)
- Feature completeness matrix
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class MultimodalCoverage:
    """Multimodal feature coverage statistics."""
    
    n_items: int = 0
    
    # Text modality
    text_coverage: dict[str, float] = field(default_factory=dict)  # field -> % with content
    text_avg_lengths: dict[str, float] = field(default_factory=dict)  # field -> avg chars
    
    # Visual modality
    items_with_images: int = 0
    items_with_images_pct: float = 0.0
    image_count_distribution: dict[int, int] = field(default_factory=dict)
    avg_images_per_item: float = 0.0
    
    # Image URL validation (sample)
    sample_urls_checked: int = 0
    sample_urls_valid: int = 0
    sample_urls_valid_pct: float = 0.0
    
    # Completeness
    items_with_all_modalities: int = 0  # has text + image
    items_with_all_modalities_pct: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_items": self.n_items,
            "text": {
                "coverage": {k: round(v, 2) for k, v in self.text_coverage.items()},
                "avg_lengths": {k: round(v, 1) for k, v in self.text_avg_lengths.items()},
            },
            "visual": {
                "items_with_images": self.items_with_images,
                "items_with_images_pct": round(self.items_with_images_pct, 2),
                "avg_images_per_item": round(self.avg_images_per_item, 2),
                "image_count_distribution": dict(list(self.image_count_distribution.items())[:10]),
            },
            "url_validation": {
                "sample_checked": self.sample_urls_checked,
                "sample_valid": self.sample_urls_valid,
                "valid_pct": round(self.sample_urls_valid_pct, 2),
            },
            "completeness": {
                "items_with_all_modalities": self.items_with_all_modalities,
                "items_with_all_modalities_pct": round(self.items_with_all_modalities_pct, 2),
            },
        }


def analyze_multimodal_coverage(
    metadata_df: pd.DataFrame,
    validate_urls: bool = True,
    url_sample_size: int = 100,
    seed: int = 42,
) -> MultimodalCoverage:
    """
    Analyze multimodal feature coverage in metadata.
    
    Args:
        metadata_df: DataFrame with columns [title, description, features, image_count, image_urls].
        validate_urls: Whether to validate a sample of image URLs.
        url_sample_size: Number of URLs to validate.
        seed: Random seed for URL sampling.
        
    Returns:
        MultimodalCoverage object with statistics.
    """
    logger.info("Analyzing multimodal feature coverage...")
    
    coverage = MultimodalCoverage(n_items=len(metadata_df))
    
    if len(metadata_df) == 0:
        return coverage
    
    # Text modality analysis
    text_fields = ["title", "description", "features"]
    for field in text_fields:
        if field in metadata_df.columns:
            has_content = metadata_df[field].fillna("").str.len() > 0
            coverage.text_coverage[field] = float(has_content.mean() * 100)
            coverage.text_avg_lengths[field] = float(
                metadata_df[field].fillna("").str.len().mean()
            )
    
    # Visual modality analysis
    if "image_count" in metadata_df.columns:
        has_images = metadata_df["image_count"] > 0
        coverage.items_with_images = int(has_images.sum())
        coverage.items_with_images_pct = float(has_images.mean() * 100)
        coverage.avg_images_per_item = float(metadata_df["image_count"].mean())
        
        # Image count distribution
        img_dist = metadata_df["image_count"].value_counts().sort_index()
        coverage.image_count_distribution = img_dist.to_dict()
    
    # URL validation (sample check)
    if validate_urls and "image_urls" in metadata_df.columns:
        coverage = _validate_image_urls_sample(
            metadata_df, coverage, sample_size=url_sample_size, seed=seed
        )
    
    # Completeness: items with both text AND image
    has_text = (
        metadata_df.get("title", pd.Series(dtype=str)).fillna("").str.len() > 0
    ) | (
        metadata_df.get("description", pd.Series(dtype=str)).fillna("").str.len() > 0
    )
    has_image = metadata_df.get("image_count", pd.Series(dtype=int)) > 0
    
    complete = has_text & has_image
    coverage.items_with_all_modalities = int(complete.sum())
    coverage.items_with_all_modalities_pct = float(complete.mean() * 100)
    
    logger.info(f"Multimodal coverage: {coverage.items_with_all_modalities_pct:.1f}% items have text+image")
    
    return coverage


def _validate_image_urls_sample(
    metadata_df: pd.DataFrame,
    coverage: MultimodalCoverage,
    sample_size: int = 100,
    seed: int = 42,
) -> MultimodalCoverage:
    """
    Validate a sample of image URLs by checking HTTP HEAD requests.
    
    Args:
        metadata_df: DataFrame with 'image_urls' column (list of URLs).
        coverage: MultimodalCoverage object to update.
        sample_size: Number of URLs to check.
        seed: Random seed.
        
    Returns:
        Updated MultimodalCoverage object.
    """
    logger.info(f"Validating {sample_size} image URLs...")
    
    # Collect all URLs
    all_urls = []
    for urls in metadata_df["image_urls"].dropna():
        if isinstance(urls, list):
            all_urls.extend(urls)
    
    if len(all_urls) == 0:
        logger.warning("No image URLs found to validate")
        return coverage
    
    # Sample URLs
    random.seed(seed)
    sample_urls = random.sample(all_urls, min(sample_size, len(all_urls)))
    
    coverage.sample_urls_checked = len(sample_urls)
    
    # Validate URLs with threading
    valid_count = 0
    
    def check_url(url: str) -> bool:
        try:
            response = requests.head(url, timeout=5, allow_redirects=True)
            return response.status_code == 200
        except Exception:
            return False
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_url, url): url for url in sample_urls}
        for future in as_completed(futures):
            if future.result():
                valid_count += 1
    
    coverage.sample_urls_valid = valid_count
    coverage.sample_urls_valid_pct = (valid_count / len(sample_urls) * 100) if sample_urls else 0.0
    
    logger.info(f"URL validation: {valid_count}/{len(sample_urls)} ({coverage.sample_urls_valid_pct:.1f}%) valid")
    
    return coverage


def compute_text_token_statistics(
    df: pd.DataFrame,
    text_column: str = "review_text",
    sample_size: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Compute approximate token statistics for text fields.
    
    Uses whitespace tokenization as a proxy (actual BERT tokens will be different).
    
    Args:
        df: DataFrame with text column.
        text_column: Name of the text column.
        sample_size: Number of records to sample for analysis.
        seed: Random seed.
        
    Returns:
        Dictionary with token statistics.
    """
    if text_column not in df.columns:
        return {}
    
    # Sample for efficiency
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=seed)
    texts = sample_df[text_column].fillna("")
    
    # Whitespace tokenization (rough approximation)
    word_counts = texts.str.split().str.len()
    
    return {
        "column": text_column,
        "sample_size": len(sample_df),
        "avg_words": float(word_counts.mean()),
        "median_words": float(word_counts.median()),
        "max_words": int(word_counts.max()),
        "percentile_95": float(word_counts.quantile(0.95)),
        "empty_pct": float((word_counts == 0).mean() * 100),
    }


def analyze_feature_completeness_matrix(
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a feature completeness matrix showing which items have which features.
    
    Args:
        metadata_df: DataFrame with metadata columns.
        
    Returns:
        DataFrame with completeness summary per item (or aggregated).
    """
    completeness = {}
    
    # Check each relevant column
    columns_to_check = ["title", "description", "features", "image_count", "main_category", "price"]
    
    for col in columns_to_check:
        if col in metadata_df.columns:
            if col == "image_count":
                completeness[col] = (metadata_df[col] > 0).astype(int)
            elif col == "price":
                completeness[col] = (
                    metadata_df[col].fillna("").astype(str).str.len() > 0
                ).astype(int)
            else:
                completeness[col] = (
                    metadata_df[col].fillna("").str.len() > 0
                ).astype(int)
    
    completeness_df = pd.DataFrame(completeness)
    completeness_df["total_features"] = completeness_df.sum(axis=1)
    
    # Summary statistics
    summary = {
        "feature": list(completeness.keys()) + ["all_features"],
        "coverage_pct": [
            completeness_df[col].mean() * 100 for col in completeness.keys()
        ] + [(completeness_df["total_features"] == len(completeness)).mean() * 100],
    }
    
    return pd.DataFrame(summary)
