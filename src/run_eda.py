"""
Main EDA execution script for Amazon Review 2023 dataset.

Performs comprehensive exploratory data analysis including:
- Data loading with sampling
- Basic statistics computation
- Visualization generation
- Multimodal coverage analysis
- User/item pattern analysis
- Sparsity and k-core analysis
- Image downloading (optional)

Usage:
    python src/run_eda.py --dataset beauty --sample-ratio 0.01 --output docs/ --download-images
    python src/run_eda.py --dataset clothing --sample-ratio 0.01 --output docs/ --download-images
    python src/run_eda.py --dataset both --sample-ratio 0.01 --output docs/ --download-images
    python src/run_eda.py --dataset beauty --sample-ratio 0.01 --output docs/ --download-images --academic-analysis
    python src/run_eda.py --dataset clothing electronics --sample-ratio 0.01 --sampling-strategy dense --kcore-k 5 --temporal-months 60 --academic-analysis
    python src/run_eda.py --dataset all --academic-analysis
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path for imports
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from eda.data_loader import (
    load_interactions_sample,
    load_metadata_sample,
    load_interactions_dense_subgraph,
    load_interactions_from_csv,
    load_metadata_for_items,
    detect_file_format,
    unzip_gzip_file,
    DataStats,
)
import pandas as pd
from eda.basic_stats import (
    compute_basic_statistics,
    compute_rating_distribution,
    compute_user_item_frequency,
    compute_temporal_statistics,
)
from eda.visualizations import (
    plot_rating_distribution,
    plot_interaction_frequency,
    plot_temporal_patterns,
    plot_text_length_distribution,
    plot_sparsity_visualization,
    plot_category_distribution,
    plot_multimodal_coverage,
)
from eda.multimodal_analysis import (
    analyze_multimodal_coverage,
    analyze_feature_completeness_matrix,
)
from eda.user_item_analysis import (
    analyze_user_item_patterns,
    compute_user_activity_segments,
    compute_item_popularity_segments,
    analyze_interaction_distribution_tiers,
)
from eda.sparsity_analysis import (
    analyze_sparsity,
    simulate_kcore_filtering,
    compute_retention_curve,
)
from eda.image_download import download_images_sample, validate_downloaded_images, compute_image_statistics

# Academic Analysis Modules
from eda.modality_alignment import analyze_modality_alignment
from eda.visual_manifold import analyze_visual_manifold
from eda.bpr_hardness import analyze_bpr_hardness

# LATTICE Feasibility Modules
from eda.graph_connectivity import analyze_graph_connectivity
from eda.feature_collapse import analyze_feature_collapse
from eda.embedding_extractor import extract_clip_embeddings, create_dummy_embeddings

# Text Embedding Modules
from eda.text_embedding_extractor import extract_text_embeddings, create_dummy_text_embeddings
from eda.semantic_alignment import analyze_semantic_alignment
from eda.cross_modal_consistency import analyze_cross_modal_consistency, compute_cca_alignment
from eda.anisotropy_analysis import analyze_anisotropy, center_embeddings, compute_pairwise_cosine_sample
from eda.user_consistency import calculate_user_consistency

from eda.visualizations import (
    plot_modality_alignment,
    plot_visual_manifold,
    plot_bpr_hardness_distribution,
    plot_semantic_alignment,
    plot_cross_modal_consistency,
    plot_anisotropy_comparison,
    plot_user_consistency,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Dataset configurations
# CSV files are required for interactions (5-core filtered)
# JSONL.gz files are used for metadata only
DATASETS = {
    "beauty": {
        "interactions": "Beauty_and_Personal_Care.csv",
        "metadata": "meta_Beauty_and_Personal_Care.jsonl.gz",
        "display_name": "Beauty and Personal Care",
    },
    "clothing": {
        "interactions": "Clothing_Shoes_and_Jewelry.csv",
        "metadata": "meta_Clothing_Shoes_and_Jewelry.jsonl.gz",
        "display_name": "Clothing, Shoes and Jewelry",
    },
    "electronics": {
        "interactions": "Electronics.csv",
        "metadata": "meta_Electronics.jsonl.gz",
        "display_name": "Electronics",
    },
}


def run_eda_for_dataset(
    dataset_name: str,
    data_dir: Path,
    output_dir: Path,
    download_images: bool = False,
    image_sample_size: int = 500,
    academic_analysis: bool = False,
    seed: int = 42,
) -> dict:
    """
    Run complete EDA pipeline for a single dataset.
    
    Args:
        dataset_name: "beauty", "clothing", or "electronics".
        data_dir: Path to data directory.
        output_dir: Path to output directory.
        download_images: Whether to download sample images.
        image_sample_size: Number of images to download.
        academic_analysis: Run academic analysis (alignment, anisotropy, etc.).
        seed: Random seed.
        
    Returns:
        Dictionary with all EDA results.
    """
    config = DATASETS[dataset_name]
    display_name = config["display_name"]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting EDA for: {display_name}")
    logger.info(f"Data format: 5-core filtered CSV")
    logger.info(f"{'='*60}\n")
    
    # Create output directories
    figures_dir = output_dir / "figures" / dataset_name
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "dataset": dataset_name,
        "display_name": display_name,
        "data_format": "5-core CSV",
        "timestamp": datetime.now().isoformat(),
    }
    
    # =========================================================================
    # Phase 1: Load Data
    # =========================================================================
    logger.info("Phase 1: Loading data...")
    
    # CSV for interactions, JSONL.gz for metadata
    interactions_path = data_dir / config["interactions"]
    metadata_path = data_dir / config["metadata"]
    
    if not interactions_path.exists():
        raise FileNotFoundError(f"Interaction CSV not found: {interactions_path}")
    
    # Load interactions from CSV
    logger.info(f"Loading interactions from: {interactions_path.name}")
    interactions_df, int_load_stats = load_interactions_from_csv(interactions_path)
    
    # Get unique item IDs for metadata filtering
    item_ids = set(interactions_df["item_id"].unique())
    
    # Load metadata for items in interactions (with auto-unzip)
    unzipped_meta = metadata_path.with_suffix("").with_suffix(".jsonl") if metadata_path.suffix == ".gz" else metadata_path
    if metadata_path.exists() or unzipped_meta.exists():
        metadata_df, meta_load_stats = load_metadata_for_items(
            metadata_path, item_ids=item_ids, unzip_first=True
        )
    else:
        logger.warning(f"Metadata file not found: {metadata_path}")
        metadata_df = pd.DataFrame()
        meta_load_stats = DataStats()
    
    results["interactions_load_stats"] = {
        "total_records": int_load_stats.total_records,
        "sampled_records": int_load_stats.sampled_records,
        "memory_mb": round(int_load_stats.memory_mb, 2),
        "n_users": int_load_stats.sampling_params.get("n_users", 0),
        "n_items": int_load_stats.sampling_params.get("n_items", 0),
    }
    results["metadata_load_stats"] = {
        "total_records": meta_load_stats.total_records,
        "sampled_records": meta_load_stats.sampled_records,
        "memory_mb": round(meta_load_stats.memory_mb, 2),
    }
    
    # =========================================================================
    # Phase 2: Basic Statistics
    # =========================================================================
    logger.info("\nPhase 2: Computing basic statistics...")
    
    int_stats, meta_stats = compute_basic_statistics(interactions_df, metadata_df)
    results["interaction_stats"] = int_stats.to_dict()
    results["metadata_stats"] = meta_stats.to_dict() if meta_stats else None
    
    # Rating distribution
    rating_dist = compute_rating_distribution(interactions_df)
    results["rating_distribution"] = rating_dist.to_dict("records")
    
    # User/item frequency
    user_freq, item_freq = compute_user_item_frequency(interactions_df)
    
    # Temporal statistics
    temporal_stats = compute_temporal_statistics(interactions_df)
    results["temporal_stats_monthly"] = temporal_stats.to_dict("records") if len(temporal_stats) > 0 else []
    
    # =========================================================================
    # Phase 3: Generate Visualizations
    # =========================================================================
    logger.info("\nPhase 3: Generating visualizations...")
    
    # Rating distribution
    plot_rating_distribution(interactions_df, figures_dir, display_name)
    
    # Interaction frequency (power-law)
    plot_interaction_frequency(user_freq, item_freq, figures_dir, display_name)
    
    # Temporal patterns
    plot_temporal_patterns(interactions_df, figures_dir, display_name)
    
    # Text length distribution
    plot_text_length_distribution(interactions_df, figures_dir, display_name)
    
    # Sparsity visualization
    plot_sparsity_visualization(
        int_stats.n_users, int_stats.n_items, int_stats.n_interactions,
        figures_dir, display_name
    )
    
    # Category distribution (from metadata)
    if len(metadata_df) > 0:
        plot_category_distribution(metadata_df, figures_dir, display_name)
        plot_multimodal_coverage(metadata_df, figures_dir, display_name)
    
    # =========================================================================
    # Phase 4: Advanced Analysis
    # =========================================================================
    logger.info("\nPhase 4: Running advanced analysis...")
    
    # User/item patterns
    patterns = analyze_user_item_patterns(interactions_df)
    results["user_item_patterns"] = patterns.to_dict()
    
    # User activity segments
    user_segments = compute_user_activity_segments(interactions_df)
    results["user_segments"] = user_segments.to_dict("records")
    
    # Item popularity segments
    item_segments = compute_item_popularity_segments(interactions_df)
    results["item_segments"] = item_segments.to_dict("records")
    
    # Interaction distribution tiers (Pareto analysis)
    tier_analysis = analyze_interaction_distribution_tiers(interactions_df)
    results["tier_analysis"] = {
        "users": tier_analysis["users"].to_dict("records"),
        "items": tier_analysis["items"].to_dict("records"),
    }
    
    # Multimodal coverage (from metadata)
    if len(metadata_df) > 0:
        mm_coverage = analyze_multimodal_coverage(
            metadata_df, validate_urls=False  # Skip URL validation for speed
        )
        results["multimodal_coverage"] = mm_coverage.to_dict()
        
        # Feature completeness
        completeness = analyze_feature_completeness_matrix(metadata_df)
        results["feature_completeness"] = completeness.to_dict("records")
    
    # =========================================================================
    # Phase 5: Sparsity and K-Core Analysis
    # =========================================================================
    logger.info("\nPhase 5: Sparsity and k-core analysis...")
    
    sparsity_stats = analyze_sparsity(interactions_df)
    results["sparsity"] = {
        "n_users": sparsity_stats.n_users,
        "n_items": sparsity_stats.n_items,
        "n_interactions": sparsity_stats.n_interactions,
        "possible_interactions": sparsity_stats.possible_interactions,
        "density": f"{sparsity_stats.density:.8%}",
        "sparsity": f"{sparsity_stats.sparsity:.8%}",
    }
    
    # K-core filtering simulation
    kcore_results = simulate_kcore_filtering(interactions_df, k_values=[2, 3, 5, 10, 20])
    results["kcore_analysis"] = kcore_results
    
    # =========================================================================
    # Phase 6: Image Download (Optional)
    # =========================================================================
    if download_images and len(metadata_df) > 0:
        logger.info(f"\nPhase 6: Downloading {image_sample_size} sample images...")
        
        images_dir = output_dir / "images" / dataset_name
        download_stats = download_images_sample(
            metadata_df, images_dir, sample_size=image_sample_size, seed=seed
        )
        
        results["image_download"] = {
            "downloaded": download_stats.downloaded,
            "failed": download_stats.failed,
            "success_rate": round(download_stats.success_rate, 2),
            "total_mb": round(download_stats.total_bytes / 1024 / 1024, 2),
        }
        
        # Validate and get image statistics
        if download_stats.downloaded > 0:
            img_stats = compute_image_statistics(images_dir)
            results["image_statistics"] = img_stats
    
    # =========================================================================
    # Phase 7: Academic Analysis (Optional - requires CLIP)
    # Implements Liu et al. (2024) and Xu et al. (2025) analysis
    # =========================================================================
    if academic_analysis and len(metadata_df) > 0:
        logger.info("\nPhase 7: Running academic analysis...")
        
        # Extract CLIP embeddings (or use dummy for testing)
        try:
            embeddings, item_indices, emb_stats = extract_clip_embeddings(
                metadata_df,
                batch_size=256,
                max_items=min(10000, len(metadata_df)),
                seed=seed,
            )
            results["embedding_extraction"] = emb_stats.to_dict()
        except Exception as e:
            logger.warning(f"CLIP extraction failed: {e}, using dummy embeddings")
            embeddings, item_indices = create_dummy_embeddings(metadata_df, seed=seed)
            results["embedding_extraction"] = {"method": "dummy", "n_items": len(item_indices)}
        
        if len(item_indices) > 0:
            # 7.1 Modality-Interaction Alignment (Liu et al., 2024)
            logger.info("  7.1 Modality-Interaction Alignment...")
            try:
                alignment_result = analyze_modality_alignment(
                    interactions_df, embeddings, item_indices,
                    n_pairs=min(10000*2, len(item_indices) * (len(item_indices) - 1) // 2),
                    seed=seed,
                )
                results["modality_alignment"] = alignment_result.to_dict()
                
                # Generate visualization
                if len(alignment_result.visual_similarities) > 0:
                    plot_modality_alignment(
                        alignment_result.visual_similarities,
                        alignment_result.interaction_similarities,
                        alignment_result.pearson_correlation,
                        alignment_result.pearson_pvalue,
                        figures_dir, display_name,
                    )
            except Exception as e:
                logger.warning(f"  Modality alignment failed: {e}")
            
            # 7.2 Visual Manifold Structure (Xu et al., 2025)
            logger.info("  7.2 Visual Manifold Structure...")
            try:
                manifold_result = analyze_visual_manifold(
                    metadata_df, embeddings, item_indices,
                    method="umap",
                    max_items=min(10000*2, len(item_indices)),
                    seed=seed,
                )
                results["visual_manifold"] = manifold_result.to_dict()
                
                # Generate visualization
                if len(manifold_result.projection_x) > 0:
                    plot_visual_manifold(
                        manifold_result.projection_x,
                        manifold_result.projection_y,
                        manifold_result.categories,
                        manifold_result.ratings,
                        manifold_result.silhouette_score_category,
                        manifold_result.method,
                        figures_dir, display_name,
                    )
            except Exception as e:
                logger.warning(f"  Visual manifold failed: {e}")
            
            # 7.3 BPR Hardness Assessment (Xu et al., 2025)
            logger.info("  7.3 BPR Hardness Assessment...")
            try:
                hardness_result = analyze_bpr_hardness(
                    interactions_df, embeddings, item_indices,
                    n_users=min(2000, len(interactions_df["user_id"].unique())),
                    n_negatives_per_user=20,
                    seed=seed,
                )
                results["bpr_hardness"] = hardness_result.to_dict()
                
                # Generate visualization
                if len(hardness_result.distance_distribution) > 0:
                    plot_bpr_hardness_distribution(
                        hardness_result.distance_distribution,
                        hardness_result.pct_easy_negatives,
                        hardness_result.pct_medium_negatives,
                        hardness_result.pct_hard_negatives,
                        hardness_result.mean_visual_distance,
                        figures_dir, display_name,
                    )
            except Exception as e:
                logger.warning(f"  BPR hardness failed: {e}")
            
            # 7.4 Graph Connectivity Check (LATTICE Feasibility)
            logger.info("  7.4 Graph Connectivity Check (LATTICE Feasibility)...")
            try:
                connectivity_result = analyze_graph_connectivity(
                    embeddings, item_indices,
                    k=5,
                    pass_threshold=50.0,
                )
                results["graph_connectivity"] = connectivity_result.to_dict()
                logger.info(f"    {'PASS' if connectivity_result.is_pass else 'FAIL'}: Giant component {connectivity_result.giant_component_coverage_pct:.1f}%")
            except Exception as e:
                logger.warning(f"  Graph connectivity failed: {e}")
            
            # 7.5 Feature Collapse Check (White Wall Test)
            logger.info("  7.5 Feature Collapse Check (White Wall Test)...")
            try:
                collapse_result = analyze_feature_collapse(
                    embeddings,
                    n_pairs=min(50000, len(item_indices) * (len(item_indices) - 1) // 2),
                    pass_threshold=0.5,
                    collapse_threshold=0.9,
                    seed=seed,
                )
                results["feature_collapse"] = collapse_result.to_dict()
                status = 'PASS' if collapse_result.is_pass else ('COLLAPSED' if collapse_result.is_collapsed else 'WARNING')
                logger.info(f"    {status}: Avg cosine similarity {collapse_result.avg_cosine_similarity:.4f}")
            except Exception as e:
                logger.warning(f"  Feature collapse check failed: {e}")
            
            # 7.6 Text Embedding Extraction (Sentence-BERT)
            logger.info("  7.6 Text Embedding Extraction (Sentence-BERT)...")
            text_embeddings = None
            text_item_indices = {}
            try:
                text_embeddings, text_item_indices, text_stats = extract_text_embeddings(
                    metadata_df,
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    batch_size=256,
                    max_items=min(25000, len(metadata_df)),
                    seed=seed,
                )
                results["text_embedding_extraction"] = text_stats.to_dict()
                logger.info(f"    Extracted {text_stats.n_items_successful} text embeddings ({text_stats.items_per_second:.1f} items/sec)")
            except Exception as e:
                logger.warning(f"  Text embedding extraction failed: {e}")
            
            # 7.7 Semantic-Interaction Alignment (Text)
            if text_embeddings is not None and len(text_item_indices) > 0:
                logger.info("  7.7 Semantic-Interaction Alignment (Text)...")
                try:
                    semantic_result = analyze_semantic_alignment(
                        interactions_df, text_embeddings, text_item_indices,
                        n_pairs=min(7500, len(text_item_indices) * (len(text_item_indices) - 1) // 2),
                        seed=seed,
                    )
                    results["semantic_alignment"] = semantic_result.to_dict()
                    
                    # Generate visualization
                    if len(semantic_result.text_similarities) > 0:
                        plot_semantic_alignment(
                            semantic_result.text_similarities,
                            semantic_result.interaction_similarities,
                            semantic_result.pearson_correlation,
                            semantic_result.pearson_pvalue,
                            semantic_result.signal_strength,
                            figures_dir, display_name,
                        )
                    
                    logger.info(f"    Signal Strength: {semantic_result.signal_strength.upper()} (r={semantic_result.pearson_correlation:.4f})")
                except Exception as e:
                    logger.warning(f"  Semantic alignment failed: {e}")
                
                # 7.8 Cross-Modal Consistency (Text vs Image)
                logger.info("  7.8 Cross-Modal Consistency (Text vs Image)...")
                try:
                    crossmodal_result = analyze_cross_modal_consistency(
                        text_embeddings, embeddings,
                        text_item_indices, item_indices,
                        projection_method="linear",
                    )
                    results["cross_modal_consistency"] = crossmodal_result.to_dict()
                    
                    # Generate visualization
                    if len(crossmodal_result.similarities) > 0:
                        plot_cross_modal_consistency(
                            crossmodal_result.similarities,
                            crossmodal_result.mean_similarity,
                            crossmodal_result.alignment_status,
                            figures_dir, display_name,
                        )
                    
                    logger.info(f"    Cross-Modal Status: {crossmodal_result.alignment_status.upper()} (mean={crossmodal_result.mean_similarity:.4f})")
                except Exception as e:
                    logger.warning(f"  Cross-modal consistency failed: {e}")
                
                # 7.9 CCA Cross-Modal Analysis
                logger.info("  7.9 CCA Cross-Modal Analysis...")
                try:
                    cca_result = compute_cca_alignment(
                        text_embeddings, embeddings,
                        text_item_indices, item_indices,
                        n_components=10,
                        max_items=5000,
                        seed=seed,
                    )
                    results["cca_alignment"] = cca_result
                    
                    if "mean_correlation" in cca_result:
                        logger.info(f"    CCA Mean Correlation: {cca_result['mean_correlation']:.4f}")
                except Exception as e:
                    logger.warning(f"  CCA analysis failed: {e}")
            
            # 7.10 Anisotropy Check (Signal Crisis Fix)
            logger.info("  7.10 Anisotropy Check (Signal Crisis Fix)...")
            try:
                anisotropy_result = analyze_anisotropy(
                    embeddings,
                    n_pairs=min(20000, len(item_indices) * (len(item_indices) - 1) // 2),
                    anisotropy_threshold=0.4,
                    seed=seed,
                )
                results["anisotropy"] = anisotropy_result.to_dict()
                
                # Generate visualization
                cos_before = compute_pairwise_cosine_sample(embeddings, n_pairs=5000, seed=seed).tolist()
                centered_emb = center_embeddings(embeddings)
                cos_after = compute_pairwise_cosine_sample(centered_emb, n_pairs=5000, seed=seed).tolist()
                
                plot_anisotropy_comparison(
                    anisotropy_result.avg_cosine_before,
                    anisotropy_result.std_cosine_before,
                    anisotropy_result.avg_cosine_after,
                    anisotropy_result.std_cosine_after,
                    cos_before,
                    cos_after,
                    figures_dir, display_name,
                )
                
                status = "ANISOTROPIC" if anisotropy_result.is_anisotropic else "ISOTROPIC"
                logger.info(f"    Status: {status} (before={anisotropy_result.avg_cosine_before:.3f}, after={anisotropy_result.avg_cosine_after:.3f})")
            except Exception as e:
                logger.warning(f"  Anisotropy check failed: {e}")
            
            # 7.11 User Consistency Score (Interaction Homophily)
            logger.info("  7.11 User Consistency Score (Interaction Homophily)...")
            try:
                consistency_result = calculate_user_consistency(
                    interactions_df, embeddings, item_indices,
                    n_users=min(1500, interactions_df["user_id"].nunique()),
                    min_items_per_user=5,
                    global_sample_size=10000,
                    seed=seed,
                )
                results["user_consistency"] = consistency_result.to_dict()
                
                # Generate visualization
                plot_user_consistency(
                    consistency_result.mean_local_distance,
                    consistency_result.std_local_distance,
                    consistency_result.mean_global_distance,
                    consistency_result.std_global_distance,
                    consistency_result.consistency_ratio,
                    consistency_result.users_with_visual_coherence_pct,
                    figures_dir, display_name,
                )
                
                status = "CONSISTENT" if consistency_result.is_consistent else "INCONSISTENT"
                logger.info(f"    Status: {status} (ratio={consistency_result.consistency_ratio:.3f})")
            except Exception as e:
                logger.warning(f"  User consistency check failed: {e}")
            
            # Generate LATTICE Go/No-Go Summary
            logger.info("\n  === LATTICE FEASIBILITY SUMMARY ===")
            go_nogo = {"checks": {}, "decision": "UNKNOWN"}
            
            # Check 1: Alignment (non-NaN)
            if results.get("modality_alignment"):
                ma = results["modality_alignment"]
                pearson = ma.get("pearson", {}).get("correlation")
                alignment_pass = pearson is not None and not (isinstance(pearson, float) and (pearson != pearson))  # NaN check
                go_nogo["checks"]["alignment"] = {"pass": alignment_pass, "value": pearson}
                logger.info(f"    Alignment Check: {'PASS' if alignment_pass else 'FAIL'} (Pearson r = {pearson})")
            
            # Check 2: Connectivity (>50%)
            if results.get("graph_connectivity"):
                gc = results["graph_connectivity"]
                go_nogo["checks"]["connectivity"] = {"pass": gc["is_pass"], "value": gc["giant_component_coverage_pct"]}
                logger.info(f"    Connectivity Check: {'PASS' if gc['is_pass'] else 'FAIL'} ({gc['giant_component_coverage_pct']:.1f}%)")
            
            # Check 3: No Collapse (<0.5 avg sim)
            if results.get("feature_collapse"):
                fc = results["feature_collapse"]
                go_nogo["checks"]["collapse"] = {"pass": fc["is_pass"], "value": fc["statistics"]["mean"]}
                logger.info(f"    Collapse Check: {'PASS' if fc['is_pass'] else 'FAIL'} (Avg cosine = {fc['statistics']['mean']:.4f})")
            
            # Overall decision
            all_pass = all(c.get("pass", False) for c in go_nogo["checks"].values())
            go_nogo["decision"] = "PROCEED" if all_pass else "STOP"
            results["lattice_feasibility"] = go_nogo
            
            logger.info(f"\n    >>> LATTICE Decision: {go_nogo['decision']} <<<")
            if not all_pass:
                logger.warning("    Recommendation: Revisit Feature Extraction (e.g., use CLIP-Fashion)")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    results_path = output_dir / f"{dataset_name}_eda_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {results_path}")
    logger.info(f"Figures saved to: {figures_dir}")
    
    return results


def generate_markdown_report(
    results: dict,
    output_dir: Path,
    dataset_name: str,
) -> Path:
    """
    Generate a markdown report from EDA results.
    
    Args:
        results: Dictionary with EDA results.
        output_dir: Output directory.
        dataset_name: Dataset identifier.
        
    Returns:
        Path to generated markdown file.
    """
    display_name = results["display_name"]
    # Create figure suffix matching visualization functions: display_name.lower().replace(' ', '_')
    figure_suffix = display_name.lower().replace(' ', '_')
    
    # Build sampling info string
    sampling_info = results.get('sampling_strategy', 'random')
    if sampling_info == 'kcore':
        sampling_info = f"K-Core (k={results.get('kcore_k', 5)})"
    elif sampling_info == 'temporal':
        sampling_info = f"Temporal ({results.get('temporal_months', 6)} months)"
    elif sampling_info == 'dense':
        sampling_info = f"Dense (K-Core k={results.get('kcore_k', 5)} + {results.get('temporal_months', 6)} months)"
    elif results.get('sample_ratio'):
        sampling_info = f"Random {results['sample_ratio']:.0%}"
    
    md_content = f"""# EDA Report: {display_name}

**Generated:** {results['timestamp']}  
**Sampling Strategy:** {sampling_info}

---

## 1. Data Overview

### Loading Statistics

| Metric | Interactions | Metadata |
|--------|-------------|----------|
| Total Records | {results['interactions_load_stats']['total_records']:,} | {results['metadata_load_stats']['total_records']:,} |
| Sampled Records | {results['interactions_load_stats']['sampled_records']:,} | {results['metadata_load_stats']['sampled_records']:,} |
| Memory (MB) | {results['interactions_load_stats']['memory_mb']} | {results['metadata_load_stats']['memory_mb']} |

### Interaction Statistics

| Metric | Value |
|--------|-------|
| Users | {results['interaction_stats']['counts']['n_users']:,} |
| Items | {results['interaction_stats']['counts']['n_items']:,} |
| Interactions | {results['interaction_stats']['counts']['n_interactions']:,} |
| Avg Rating | {results['interaction_stats']['ratings']['mean']:.2f} |
| Rating Std | {results['interaction_stats']['ratings']['std']:.2f} |
| Sparsity | {results['sparsity']['sparsity']} |

---

## 2. Rating Distribution

![Rating Distribution](figures/{dataset_name}/rating_distribution_{figure_suffix}.png)

| Rating | Count | Percentage |
|--------|-------|------------|
"""
    
    for row in results.get('rating_distribution', []):
        md_content += f"| {row.get('rating', 'N/A')} | {row.get('count', 0):,} | {row.get('percentage', 0):.1f}% |\n"
    
    md_content += f"""
---

## 3. User and Item Analysis

### Power-Law Distribution

![Interaction Frequency](figures/{dataset_name}/interaction_frequency_{figure_suffix}.png)

**User Patterns:**
- Mean interactions/user: {results['user_item_patterns']['users']['interaction_stats'].get('mean', 0):.2f}
- Median interactions/user: {results['user_item_patterns']['users']['interaction_stats'].get('median', 0):.1f}
- Cold-start users (<5 interactions): {results['user_item_patterns']['users']['cold_start_pct']:.1f}%
- Power-law exponent α: {results['user_item_patterns']['users']['power_law_alpha']:.2f}

**Item Patterns:**
- Mean interactions/item: {results['user_item_patterns']['items']['interaction_stats'].get('mean', 0):.2f}
- Median interactions/item: {results['user_item_patterns']['items']['interaction_stats'].get('median', 0):.1f}
- Cold-start items (<5 interactions): {results['user_item_patterns']['items']['cold_start_pct']:.1f}%
- Power-law exponent α: {results['user_item_patterns']['items']['power_law_alpha']:.2f}

### Pareto Analysis (Interaction Concentration)

Top users account for a disproportionate share of interactions:

| User Tier | % of Total Interactions |
|-----------|------------------------|
"""
    
    for row in results.get('tier_analysis', {}).get('users', []):
        md_content += f"| {row.get('tier_pct', 'N/A')} | {row.get('pct_of_total_interactions', 0):.1f}% |\n"
    
    md_content += f"""
---

## 4. Temporal Analysis

![Temporal Patterns](figures/{dataset_name}/temporal_patterns_{figure_suffix}.png)

**Date Range:** {results['interaction_stats']['temporal']['date_min']} to {results['interaction_stats']['temporal']['date_max']}  
**Duration:** {results['interaction_stats']['temporal']['date_range_days']:,} days

---

## 5. Text Analysis

![Text Length Distribution](figures/{dataset_name}/text_length_{figure_suffix}.png)

| Metric | Value |
|--------|-------|
| Avg Review Length | {results['interaction_stats']['text']['avg_review_length']:.0f} chars |
| Avg Title Length | {results['interaction_stats']['text']['avg_title_length']:.0f} chars |
| Reviews with Text | {results['interaction_stats']['text']['reviews_with_text_pct']:.1f}% |

---

## 6. Multimodal Analysis

"""
    if results.get('multimodal_coverage'):
        mc = results['multimodal_coverage']
        md_content += f"""
![Multimodal Coverage](figures/{dataset_name}/multimodal_coverage_{figure_suffix}.png)

### Feature Coverage

| Feature | Coverage |
|---------|----------|
| Title | {mc['text']['coverage'].get('title', 0):.1f}% |
| Description | {mc['text']['coverage'].get('description', 0):.1f}% |
| Features | {mc['text']['coverage'].get('features', 0):.1f}% |
| Images | {mc['visual']['items_with_images_pct']:.1f}% |
| **Complete (Text + Image)** | {mc['completeness']['items_with_all_modalities_pct']:.1f}% |

### Image Statistics

| Metric | Value |
|--------|-------|
| Items with Images | {mc['visual']['items_with_images']:,} |
| Avg Images/Item | {mc['visual']['avg_images_per_item']:.2f} |
"""
    
    md_content += f"""
---

## 7. Sparsity and K-Core Analysis

![Sparsity](figures/{dataset_name}/sparsity_{figure_suffix}.png)

**Matrix Sparsity:** {results['sparsity']['sparsity']}  
**Density:** {results['sparsity']['density']}

### K-Core Filtering Impact

| k | Users Retained | Items Retained | Interactions Retained |
|---|----------------|----------------|----------------------|
"""
    
    for k, stats in results.get('kcore_analysis', {}).items():
        md_content += f"| {k} | {stats['user_retention_pct']:.1f}% | {stats['item_retention_pct']:.1f}% | {stats['interaction_retention_pct']:.1f}% |\n"
    
    md_content += f"""
---

## 8. Category Distribution

![Categories](figures/{dataset_name}/category_distribution_{figure_suffix}.png)

Top categories in the dataset:

| Category | Count |
|----------|-------|
"""
    
    if results.get('metadata_stats') and results['metadata_stats'].get('categories'):
        for cat, count in list(results['metadata_stats']['categories']['top_10'].items())[:10]:
            md_content += f"| {cat} | {count:,} |\n"
    
    md_content += """
---

## 9. Key Insights and Recommendations

### Data Quality
1. **High Sparsity:** The dataset exhibits extreme sparsity typical of recommendation datasets
2. **Power-Law Distribution:** Both users and items follow power-law distributions (long-tail)
3. **Cold-Start Challenge:** Significant portion of users/items have few interactions

### Preprocessing Recommendations
1. **K-Core Filtering:** Use k=5 as baseline (balances data quality vs. coverage)
2. **Multimodal Features:** Leverage text/image to address cold-start problem
3. **Negative Sampling:** Use popularity-based hard negative sampling for BPR

"""
    
    # Section 10: Academic Analysis (if available)
    if results.get('modality_alignment') or results.get('visual_manifold') or results.get('bpr_hardness'):
        md_content += """
---

## 10. Multimodal Recommendation Readiness (Academic Analysis)

"""
        # 10.1 Modality Alignment
        if results.get('modality_alignment'):
            ma = results['modality_alignment']
            md_content += f"""
### 10.1 Modality-Interaction Alignment (Liu et al., 2024)

![Modality Alignment](figures/{dataset_name}/modality_alignment_{figure_suffix}.png)

Tests the **Homophily Hypothesis**: Do visually similar items share similar interaction patterns?

| Metric | Value |
|--------|-------|
| Pairs Analyzed | {ma.get('n_pairs_sampled', 0):,} |
| Pearson r | {ma['pearson']['correlation']:.4f} |
| p-value | {ma['pearson']['pvalue']:.4f} |
| Spearman ρ | {ma['spearman']['correlation']:.4f} |

**Interpretation:** {ma.get('interpretation', 'N/A')}

"""
        
        # 10.2 Visual Manifold
        if results.get('visual_manifold'):
            vm = results['visual_manifold']
            md_content += f"""
### 10.2 Visual Manifold Structure (Xu et al., 2025)

![Visual Manifold](figures/{dataset_name}/visual_manifold_{figure_suffix}.png)

Analyzes whether CLIP embeddings form meaningful clusters by category.

| Metric | Value |
|--------|-------|
| Items Projected | {vm.get('n_items', 0):,} |
| Projection Method | {vm.get('method', 'umap').upper()} |
| Silhouette Score | {vm['quality']['silhouette_score_category']:.4f} |
| Unique Categories | {vm['quality']['n_unique_categories']} |

**Interpretation:** {vm.get('interpretation', 'N/A')}

"""
        
        # 10.3 BPR Hardness
        if results.get('bpr_hardness'):
            bh = results['bpr_hardness']
            md_content += f"""
### 10.3 BPR Negative Sampling Hardness (Xu et al., 2025)

![BPR Hardness](figures/{dataset_name}/bpr_hardness_{figure_suffix}.png)

Evaluates whether random negative sampling produces informative training signal.

| Metric | Value |
|--------|-------|
| Users Analyzed | {bh.get('n_users_analyzed', 0):,} |
| Pairs Analyzed | {bh.get('n_pairs_analyzed', 0):,} |
| Mean Visual Distance | {bh['visual_distance']['mean']:.4f} |
| Easy Negatives (>0.8) | {bh['hardness_distribution']['easy_pct']:.1f}% |
| Medium Negatives | {bh['hardness_distribution']['medium_pct']:.1f}% |
| Hard Negatives (<0.3) | {bh['hardness_distribution']['hard_pct']:.1f}% |

**Interpretation:** {bh.get('interpretation', 'N/A')}

**Recommendation:** {bh.get('recommendation', 'N/A')}

"""
        
        # 10.4 Text Embedding Extraction
        if results.get('text_embedding_extraction'):
            te = results['text_embedding_extraction']
            md_content += f"""### 10.4 Text Embedding Extraction (Sentence-BERT)

| Metric | Value |
|--------|-------|
| Model | `{te.get('model_name', 'N/A')}` |
| Items Processed | {te.get('n_items_successful', 0):,} |
| Success Rate | {te.get('success_rate', 0):.1f}% |
| Embedding Dimension | {te.get('embedding_dim', 0)} |
| Processing Time | {te.get('processing_time_sec', 0):.1f}s |
| Throughput | {te.get('items_per_second', 0):.1f} items/sec |
| Avg Text Length | {te.get('avg_text_length', 0):.0f} chars |

"""
        
        # 10.5 Semantic-Interaction Alignment
        if results.get('semantic_alignment'):
            sa = results['semantic_alignment']
            signal = sa.get('signal_strength', 'unknown').upper()
            
            # Signal badge
            if signal == "STRONG":
                signal_badge = "🟢 STRONG"
            elif signal == "MODERATE":
                signal_badge = "🟡 MODERATE"
            elif signal == "WEAK":
                signal_badge = "🟠 WEAK"
            else:
                signal_badge = "🔴 NOISE"
            
            pearson = sa.get('pearson', {})
            stats = sa.get('statistics', {})
            
            md_content += f"""### 10.5 Semantic-Interaction Alignment (Text)

![Semantic Alignment](figures/{dataset_name}/semantic_alignment_{figure_suffix}.png)

Tests whether items with similar text descriptions have similar buyers.

| Metric | Value |
|--------|-------|
| Pairs Analyzed | {sa.get('n_pairs_sampled', 0):,} |
| Pearson r | {pearson.get('correlation', 'N/A')} |
| p-value | {pearson.get('pvalue', 'N/A')} |
| Mean Text Similarity | {stats.get('mean_text_similarity', 0):.4f} |
| Mean Interaction Similarity | {stats.get('mean_interaction_similarity', 0):.4f} |
| **Signal Strength** | {signal_badge} |

**Interpretation:** {sa.get('interpretation', 'N/A')}

**Recommendation:** {sa.get('recommendation', 'N/A')}

"""
        
        # 10.6 Cross-Modal Consistency
        if results.get('cross_modal_consistency'):
            cm = results['cross_modal_consistency']
            status = cm.get('alignment_status', 'unknown').upper()
            
            # Status badge
            if status == "AGREE":
                status_badge = "🟢 AGREE"
            elif status == "MODERATE":
                status_badge = "🟡 MODERATE"
            else:
                status_badge = "🔴 DISAGREE"
            
            stats = cm.get('statistics', {})
            dist = cm.get('distribution', {})
            proj = cm.get('projection', {})
            
            md_content += f"""### 10.6 Cross-Modal Consistency (Text vs Image)

![Cross-Modal Consistency](figures/{dataset_name}/cross_modal_consistency_{figure_suffix}.png)

Measures whether text and image embeddings agree for the same items.

| Metric | Value |
|--------|-------|
| Items with Both Modalities | {cm.get('n_items_with_both', 0):,} |
| Projection Method | {proj.get('method', 'N/A')} |
| Text Dim → Projected | {proj.get('text_dim', 0)} → {proj.get('projected_dim', 0)} |
| Image Dim → Projected | {proj.get('image_dim', 0)} → {proj.get('projected_dim', 0)} |
| **Mean Similarity** | {stats.get('mean', 0):.4f} |
| Std Similarity | {stats.get('std', 0):.4f} |
| Low Agreement (<0.3) | {dist.get('low_agreement_pct', 0):.1f}% |
| Moderate (0.3-0.6) | {dist.get('moderate_agreement_pct', 0):.1f}% |
| High Agreement (>0.6) | {dist.get('high_agreement_pct', 0):.1f}% |
| **Status** | {status_badge} |

**Interpretation:** {cm.get('interpretation', 'N/A')}

**Recommendation:** {cm.get('recommendation', 'N/A')}

"""
    
    # Section 10.7: CCA Analysis (if available)
    if results.get('cca_alignment'):
        cca = results['cca_alignment']
        correlations = cca.get('canonical_correlations', [])
        top_corrs = correlations[:5] if correlations else []
        
        md_content += f"""### 10.7 CCA Cross-Modal Analysis

Canonical Correlation Analysis measures linear relationship capacity between modalities.

| Metric | Value |
|--------|-------|
| Items Analyzed | {cca.get('n_items', 0):,} |
| CCA Components | {cca.get('n_components', 0)} |
| Mean CCA Correlation | {cca.get('mean_correlation', 0):.4f} |
| Top-5 Correlations | {', '.join(f'{c:.3f}' for c in top_corrs)} |

**Interpretation:** {cca.get('interpretation', 'N/A')}

**Recommendation:** {cca.get('recommendation', 'N/A')}

"""
    
    # Section 10.8: Anisotropy Check (if available)
    if results.get('anisotropy'):
        aniso = results['anisotropy']
        before = aniso.get('before_centering', {})
        after = aniso.get('after_centering', {})
        is_aniso = aniso.get('is_anisotropic', False)
        status = "⚠️ ANISOTROPIC" if is_aniso else "✅ ISOTROPIC"
        
        md_content += f"""### 10.8 Anisotropy Check (Signal Crisis Fix)

![Anisotropy Comparison](figures/{dataset_name}/anisotropy_comparison_{figure_suffix}.png)

Detects "Cone Effect" in embeddings and tests if mean centering helps.

| Metric | Before Centering | After Centering |
|--------|------------------|-----------------|
| Avg Cosine Similarity | {before.get('avg_cosine', 0):.4f} | {after.get('avg_cosine', 0):.4f} |
| Std Cosine Similarity | {before.get('std_cosine', 0):.4f} | {after.get('std_cosine', 0):.4f} |
| Pairs Sampled | {aniso.get('n_pairs_sampled', 0):,} | - |
| Improvement Ratio | {aniso.get('improvement_ratio', 0):.1%} | - |
| **Status** | {status} | - |

**Interpretation:** {aniso.get('interpretation', 'N/A')}

**Recommendation:** {aniso.get('recommendation', 'N/A')}

"""
    
    # Section 10.9: User Consistency (if available)
    if results.get('user_consistency'):
        uc = results['user_consistency']
        dists = uc.get('distances', {})
        is_cons = uc.get('is_consistent', False)
        status = "✅ CONSISTENT" if is_cons else "❌ INCONSISTENT"
        
        md_content += f"""### 10.9 User Consistency (Interaction Homophily)

![User Consistency](figures/{dataset_name}/user_consistency_{figure_suffix}.png)

Measures whether users buy visually similar items (validates visual MRS approach).

| Metric | Value |
|--------|-------|
| Users Analyzed | {uc.get('n_users_analyzed', 0):,} |
| Users with ≥{uc.get('min_items_threshold', 5)} Items | {uc.get('n_users_with_enough_items', 0):,} |
| Mean Local Distance | {dists.get('mean_local', 0):.4f} |
| Mean Global Distance | {dists.get('mean_global', 0):.4f} |
| **Consistency Ratio** | {uc.get('consistency_ratio', 0):.4f} |
| Users with Visual Coherence | {uc.get('users_with_visual_coherence_pct', 0):.1f}% |
| **Status** | {status} |

**Interpretation:** {uc.get('interpretation', 'N/A')}

**Recommendation:** {uc.get('recommendation', 'N/A')}

"""
    
    # Section 11: LATTICE Feasibility (if available)
    if results.get('graph_connectivity') or results.get('feature_collapse') or results.get('lattice_feasibility'):
        md_content += """
---

## 11. LATTICE Feasibility Assessment

"""
        # Go/No-Go Decision Banner
        if results.get('lattice_feasibility'):
            lf = results['lattice_feasibility']
            decision = lf.get('decision', 'UNKNOWN')
            if decision == "PROCEED":
                md_content += """> [!TIP]
> ✅ **PROCEED** with LATTICE architecture - All feasibility checks passed.

"""
            else:
                md_content += """> [!CAUTION]
> ⛔ **STOP** - LATTICE feasibility checks failed. Revisit Feature Extraction.

"""
        
        # 11.1 Graph Connectivity
        if results.get('graph_connectivity'):
            gc = results['graph_connectivity']
            status = "✅ PASS" if gc.get('is_pass', False) else "❌ FAIL"
            md_content += f"""### 11.1 Graph Connectivity (k-NN, k={gc.get('k_neighbors', 5)})

| Metric | Value | Status |
|--------|-------|--------|
| Connected Components | {gc.get('n_components', 0):,} | - |
| Giant Component Size | {gc.get('giant_component_size', 0):,} | - |
| Giant Component Coverage | {gc.get('giant_component_coverage_pct', 0):.1f}% | {status} |
| Threshold | >{gc.get('pass_threshold', 50)}% | - |

**Interpretation:** {gc.get('interpretation', 'N/A')}

"""
        
        # 11.2 Feature Collapse
        if results.get('feature_collapse'):
            fc = results['feature_collapse']
            stats = fc.get('statistics', {})
            is_pass = fc.get('is_pass', False)
            is_collapsed = fc.get('is_collapsed', False)
            
            if is_collapsed:
                status = "❌ COLLAPSED"
            elif is_pass:
                status = "✅ PASS"
            else:
                status = "⚠️ WARNING"
            
            md_content += f"""### 11.2 Feature Collapse Detection (White Wall Test)

| Metric | Value | Status |
|--------|-------|--------|
| Pairs Sampled | {fc.get('n_pairs_sampled', 0):,} | - |
| Avg Cosine Similarity | {stats.get('mean', 0):.4f} | {status} |
| Std Cosine Similarity | {stats.get('std', 0):.4f} | - |
| High Similarity Pairs (>0.9) | {fc.get('distribution', {}).get('very_high_pct', 0):.1f}% | - |
| Pass Threshold | <{fc.get('thresholds', {}).get('pass', 0.5)} | - |

**Interpretation:** {fc.get('interpretation', 'N/A')}

"""
        
        # Summary Table
        if results.get('lattice_feasibility'):
            lf = results['lattice_feasibility']
            checks = lf.get('checks', {})
            md_content += """### Summary

| Check | Value | Status |
|-------|-------|--------|
"""
            if 'alignment' in checks:
                status = "✅" if checks['alignment'].get('pass') else "❌"
                md_content += f"| Alignment (Pearson r) | {checks['alignment'].get('value', 'N/A')} | {status} |\n"
            if 'connectivity' in checks:
                status = "✅" if checks['connectivity'].get('pass') else "❌"
                md_content += f"| Connectivity (Giant %) | {checks['connectivity'].get('value', 0):.1f}% | {status} |\n"
            if 'collapse' in checks:
                status = "✅" if checks['collapse'].get('pass') else "❌"
                md_content += f"| No Collapse (Avg Cosine) | {checks['collapse'].get('value', 0):.4f} | {status} |\n"
            
            md_content += f"\n**Decision:** {lf.get('decision', 'UNKNOWN')}\n"
    
    md_content += """
---

*Report generated by EDA Pipeline for Multimodal Recommendation System*
"""
    
    report_path = output_dir / f"{dataset_name}_eda_report.md"
    with open(report_path, "w+", encoding="utf-8") as f:
        f.write(md_content)
    
    logger.info(f"Markdown report saved to: {report_path}")
    
    return report_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run EDA for Amazon Review 2023 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=["beauty", "clothing", "electronics", "all"],
        default=["beauty"],
        help="Dataset(s) to analyze (default: beauty). Can specify multiple: --dataset beauty clothing",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory (default: data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs",
        help="Output directory for results and figures (default: docs)",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Download sample images for visual analysis",
    )
    parser.add_argument(
        "--image-sample",
        type=int,
        default=500,
        help="Number of images to download (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--academic-analysis",
        action="store_true",
        help="Run academic analysis (alignment, anisotropy, user consistency, CCA)",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which datasets to process
    if "all" in args.dataset:
        datasets = ["beauty", "clothing", "electronics"]
    else:
        datasets = args.dataset
    
    all_results = {}
    
    for dataset in datasets:
        # Check if files exist
        config = DATASETS[dataset]
        int_path = data_dir / config["interactions"]
        meta_path = data_dir / config["metadata"]
        
        if not int_path.exists():
            logger.error(f"CSV file not found: {int_path}")
            continue
        if not meta_path.exists():
            # Check for unzipped version
            unzipped_meta = meta_path.with_suffix("").with_suffix(".jsonl") if meta_path.suffix == ".gz" else meta_path.with_suffix("")
            if not unzipped_meta.exists():
                logger.warning(f"Metadata file not found: {meta_path}")
        
        # Run EDA
        results = run_eda_for_dataset(
            dataset_name=dataset,
            data_dir=data_dir,
            output_dir=output_dir,
            download_images=args.download_images,
            image_sample_size=args.image_sample,
            academic_analysis=args.academic_analysis,
            seed=args.seed,
        )
        
        all_results[dataset] = results
        
        # Generate markdown report
        generate_markdown_report(results, output_dir, dataset)
    
    logger.info("\n" + "=" * 60)
    logger.info("EDA COMPLETE")
    logger.info("=" * 60)
    
    for dataset, results in all_results.items():
        logger.info(f"\n{results['display_name']}:")
        logger.info(f"  - Users: {results['interaction_stats']['counts']['n_users']:,}")
        logger.info(f"  - Items: {results['interaction_stats']['counts']['n_items']:,}")
        logger.info(f"  - Interactions: {results['interaction_stats']['counts']['n_interactions']:,}")
        logger.info(f"  - Sparsity: {results['sparsity']['sparsity']}")


if __name__ == "__main__":
    main()
