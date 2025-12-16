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
    plot_modality_alignment,
    plot_visual_manifold,
    plot_bpr_hardness_distribution,
)
from .multimodal_analysis import analyze_multimodal_coverage
from .user_item_analysis import analyze_user_item_patterns
from .sparsity_analysis import analyze_sparsity, simulate_kcore_filtering
from .image_download import download_images_sample, validate_downloaded_images

# Academic Analysis Modules
from .modality_alignment import analyze_modality_alignment, ModalityAlignmentResult
from .visual_manifold import analyze_visual_manifold, VisualManifoldResult
from .bpr_hardness import analyze_bpr_hardness, BPRHardnessResult
from .embedding_extractor import extract_clip_embeddings, create_dummy_embeddings

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
    "plot_modality_alignment",
    "plot_visual_manifold",
    "plot_bpr_hardness_distribution",
    "analyze_multimodal_coverage",
    "analyze_user_item_patterns",
    "analyze_sparsity",
    "simulate_kcore_filtering",
    "download_images_sample",
    "validate_downloaded_images",
    # Academic Analysis
    "analyze_modality_alignment",
    "ModalityAlignmentResult",
    "analyze_visual_manifold",
    "VisualManifoldResult",
    "analyze_bpr_hardness",
    "BPRHardnessResult",
    "extract_clip_embeddings",
    "create_dummy_embeddings",
]

