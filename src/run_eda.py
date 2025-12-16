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
    python src/run_eda.py --dataset beauty --sample-ratio 0.1 --output docs/
    python src/run_eda.py --dataset clothing --sample-ratio 0.1 --output docs/
    python src/run_eda.py --dataset both --sample-ratio 0.1 --output docs/ --download-images
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

from eda.data_loader import load_interactions_sample, load_metadata_sample
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
from eda.embedding_extractor import extract_clip_embeddings, create_dummy_embeddings
from eda.visualizations import (
    plot_modality_alignment,
    plot_visual_manifold,
    plot_bpr_hardness_distribution,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Dataset configurations
DATASETS = {
    "beauty": {
        "interactions": "Beauty_and_Personal_Care.jsonl.gz",
        "metadata": "meta_Beauty_and_Personal_Care.jsonl.gz",
        "display_name": "Beauty and Personal Care",
    },
    "clothing": {
        "interactions": "Clothing_Shoes_and_Jewelry.jsonl.gz",
        "metadata": "meta_Clothing_Shoes_and_Jewelry.jsonl.gz",
        "display_name": "Clothing, Shoes and Jewelry",
    },
}


def run_eda_for_dataset(
    dataset_name: str,
    data_dir: Path,
    output_dir: Path,
    sample_ratio: float = 0.1,
    download_images: bool = False,
    image_sample_size: int = 500,
    academic_analysis: bool = False,
    seed: int = 42,
) -> dict:
    """
    Run complete EDA pipeline for a single dataset.
    
    Args:
        dataset_name: "beauty" or "clothing".
        data_dir: Path to data directory.
        output_dir: Path to output directory.
        sample_ratio: Fraction of data to sample.
        download_images: Whether to download sample images.
        image_sample_size: Number of images to download.
        seed: Random seed.
        
    Returns:
        Dictionary with all EDA results.
    """
    config = DATASETS[dataset_name]
    display_name = config["display_name"]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting EDA for: {display_name}")
    logger.info(f"Sample ratio: {sample_ratio:.5%}")
    logger.info(f"{'='*60}\n")
    
    # Create output directories
    figures_dir = output_dir / "figures" / dataset_name
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "dataset": dataset_name,
        "display_name": display_name,
        "sample_ratio": sample_ratio,
        "timestamp": datetime.now().isoformat(),
    }
    
    # =========================================================================
    # Phase 1: Load Data
    # =========================================================================
    logger.info("Phase 1: Loading data...")
    
    interactions_path = data_dir / config["interactions"]
    metadata_path = data_dir / config["metadata"]
    
    # Load interactions
    interactions_df, int_load_stats = load_interactions_sample(
        interactions_path, sample_ratio=sample_ratio, seed=seed
    )
    results["interactions_load_stats"] = {
        "total_records": int_load_stats.total_records,
        "sampled_records": int_load_stats.sampled_records,
        "memory_mb": round(int_load_stats.memory_mb, 2),
    }
    
    # Get unique item IDs for metadata filtering
    item_ids = set(interactions_df["item_id"].unique())
    
    # Load metadata (filtered to items in interactions)
    metadata_df, meta_load_stats = load_metadata_sample(
        metadata_path, item_ids=item_ids, seed=seed
    )
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
                batch_size=32,
                max_items=min(5000, len(metadata_df)),
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
                    n_pairs=min(1000, len(item_indices) * (len(item_indices) - 1) // 2),
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
                    max_items=min(5000, len(item_indices)),
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
                    n_users=min(100, len(interactions_df["user_id"].unique())),
                    n_negatives_per_user=10,
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
    
    md_content = f"""# EDA Report: {display_name}

**Generated:** {results['timestamp']}  
**Sample Ratio:** {results['sample_ratio']:.0%}

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

![Rating Distribution](figures/{dataset_name}/rating_distribution_{dataset_name.replace(' ', '_')}.png)

| Rating | Count | Percentage |
|--------|-------|------------|
"""
    
    for row in results.get('rating_distribution', []):
        md_content += f"| {row.get('rating', 'N/A')} | {row.get('count', 0):,} | {row.get('percentage', 0):.1f}% |\n"
    
    md_content += f"""
---

## 3. User and Item Analysis

### Power-Law Distribution

![Interaction Frequency](figures/{dataset_name}/interaction_frequency_{dataset_name.replace(' ', '_')}.png)

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

![Temporal Patterns](figures/{dataset_name}/temporal_patterns_{dataset_name.replace(' ', '_')}.png)

**Date Range:** {results['interaction_stats']['temporal']['date_min']} to {results['interaction_stats']['temporal']['date_max']}  
**Duration:** {results['interaction_stats']['temporal']['date_range_days']:,} days

---

## 5. Text Analysis

![Text Length Distribution](figures/{dataset_name}/text_length_{dataset_name.replace(' ', '_')}.png)

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
![Multimodal Coverage](figures/{dataset_name}/multimodal_coverage_{dataset_name.replace(' ', '_')}.png)

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

![Sparsity](figures/{dataset_name}/sparsity_{dataset_name.replace(' ', '_')}.png)

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

![Categories](figures/{dataset_name}/category_distribution_{dataset_name.replace(' ', '_')}.png)

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

![Modality Alignment](figures/{dataset_name}/modality_alignment_{dataset_name.replace(' ', '_')}.png)

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

![Visual Manifold](figures/{dataset_name}/visual_manifold_{dataset_name.replace(' ', '_')}.png)

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

![BPR Hardness](figures/{dataset_name}/bpr_hardness_{dataset_name.replace(' ', '_')}.png)

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
    
    md_content += """
---

*Report generated by EDA Pipeline for Multimodal Recommendation System*
"""
    
    report_path = output_dir / f"{dataset_name}_eda_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
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
        type=str,
        choices=["beauty", "clothing", "both"],
        default="both",
        help="Dataset to analyze (default: both)",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.1,
        help="Fraction of data to sample (default: 0.1 = 10%%)",
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
        help="Run academic analysis (modality alignment, visual manifold, BPR hardness)",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which datasets to process
    datasets = ["beauty", "clothing"] if args.dataset == "both" else [args.dataset]
    
    all_results = {}
    
    for dataset in datasets:
        # Check if files exist
        config = DATASETS[dataset]
        int_path = data_dir / config["interactions"]
        meta_path = data_dir / config["metadata"]
        
        if not int_path.exists():
            logger.error(f"Interaction file not found: {int_path}")
            continue
        if not meta_path.exists():
            logger.warning(f"Metadata file not found: {meta_path}")
        
        # Run EDA
        results = run_eda_for_dataset(
            dataset_name=dataset,
            data_dir=data_dir,
            output_dir=output_dir,
            sample_ratio=args.sample_ratio,
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
