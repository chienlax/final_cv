"""
EDA Module for Multimodal Recommendation System.

This package provides exploratory data analysis tools for the Amazon Review 2023 dataset.
"""

from .data_loader import (
    stream_jsonl_gz,
    load_interactions_sample,
    load_metadata_sample,
    DataStats,
)
from .basic_stats import compute_basic_statistics, compute_rating_distribution
from .visualizations import (
    plot_rating_distribution,
    plot_interaction_frequency,
    plot_temporal_patterns,
    plot_text_length_distribution,
)
from .multimodal_analysis import analyze_multimodal_coverage
from .user_item_analysis import analyze_user_item_patterns
from .sparsity_analysis import analyze_sparsity, simulate_kcore_filtering
from .image_download import download_images_sample, validate_downloaded_images

__all__ = [
    "stream_jsonl_gz",
    "load_interactions_sample",
    "load_metadata_sample",
    "DataStats",
    "compute_basic_statistics",
    "compute_rating_distribution",
    "plot_rating_distribution",
    "plot_interaction_frequency",
    "plot_temporal_patterns",
    "plot_text_length_distribution",
    "analyze_multimodal_coverage",
    "analyze_user_item_patterns",
    "analyze_sparsity",
    "simulate_kcore_filtering",
    "download_images_sample",
    "validate_downloaded_images",
]
