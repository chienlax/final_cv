"""
Extract experiment results from checkpoints and generate comparison tables/plots.

Outputs:
- experiment_result/tables/*.csv, *.md  (comparison tables)
- experiment_result/figures/*.png        (training curves)

Usage:
    python scripts/extract_results.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASETS = ["beauty", "clothing", "electronics"]
MODELS = ["lattice", "micro", "diffmm"]
TRACKS = ["track1_warm", "track2_sparse", "track2_active", "track3_cold"]
METRICS = [
    "recall@10", "recall@20", "recall@50",
    "ndcg@10", "ndcg@20", "ndcg@50",
    "precision@10", "precision@20", "precision@50"
]

# Plot styling - distinct colors per model
MODEL_COLORS = {
    "lattice": "#2E86AB",   # Blue
    "micro": "#A23B72",     # Magenta/Pink
    "diffmm": "#F18F01",    # Orange
}
MODEL_MARKERS = {
    "lattice": "o",
    "micro": "s",
    "diffmm": "^",
}
MODEL_DISPLAY_NAMES = {
    "lattice": "LATTICE",
    "micro": "MICRO",
    "diffmm": "DiffMM",
}
DATASET_DISPLAY_NAMES = {
    "beauty": "Beauty",
    "clothing": "Clothing",
    "electronics": "Electronics",
}

# Figure settings
FIGURE_DPI = 300
FIGURE_SIZE = (10, 6)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_eval_results(checkpoint_dir: Path) -> pd.DataFrame:
    """
    Load eval_results.json from all dataset/model folders.
    
    Args:
        checkpoint_dir: Root checkpoint directory.
        
    Returns:
        DataFrame with columns: [dataset, model, track, metric, value]
    """
    records = []
    
    for dataset in DATASETS:
        for model in MODELS:
            eval_file = checkpoint_dir / dataset / model / "eval_results.json"
            
            if not eval_file.exists():
                logger.warning(f"Missing: {eval_file}")
                continue
            
            with open(eval_file) as f:
                data = json.load(f)
            
            for track, track_data in data.items():
                for metric in METRICS:
                    if metric in track_data:
                        records.append({
                            "dataset": dataset,
                            "model": model,
                            "track": track,
                            "metric": metric,
                            "value": track_data[metric],
                            "n_users": track_data.get("n_users", None),
                        })
    
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} metric records from {checkpoint_dir}")
    return df


def load_training_history(checkpoint_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """
    Load training_history.json from all dataset/model folders.
    
    Returns:
        Nested dict: {dataset: {model: {history_dict}}}
    """
    history = {}
    
    for dataset in DATASETS:
        history[dataset] = {}
        for model in MODELS:
            history_file = checkpoint_dir / dataset / model / "training_history.json"
            
            if not history_file.exists():
                logger.warning(f"Missing: {history_file}")
                continue
            
            with open(history_file) as f:
                history[dataset][model] = json.load(f)
    
    return history


# =============================================================================
# TABLE GENERATION
# =============================================================================

def create_main_results_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Table 1: Main Results (Track 1 - Warm).
    
    Returns:
        DataFrame with columns: [Dataset, Model, recall@10, ..., precision@50]
    """
    # Filter to Track 1 only
    df = results_df[results_df["track"] == "track1_warm"].copy()
    
    # Pivot to wide format
    pivot = df.pivot_table(
        index=["dataset", "model"],
        columns="metric",
        values="value",
        aggfunc="first"
    ).reset_index()
    
    # Reorder columns
    cols = ["dataset", "model"] + METRICS
    pivot = pivot[[c for c in cols if c in pivot.columns]]
    
    # Apply display names
    pivot["dataset"] = pivot["dataset"].map(DATASET_DISPLAY_NAMES)
    pivot["model"] = pivot["model"].map(MODEL_DISPLAY_NAMES)
    
    # Rename columns for display
    pivot.columns = [c.replace("@", "@") for c in pivot.columns]
    pivot = pivot.rename(columns={"dataset": "Dataset", "model": "Model"})
    
    return pivot


def create_user_robustness_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Table 2: User Robustness (Track 2 - Sparse vs Active).
    
    Returns:
        DataFrame comparing sparse and active user performance.
    """
    # Filter to Track 2 only
    df = results_df[results_df["track"].isin(["track2_sparse", "track2_active"])].copy()
    
    # Pivot to wide format
    pivot = df.pivot_table(
        index=["dataset", "model", "track"],
        columns="metric",
        values="value",
        aggfunc="first"
    ).reset_index()
    
    # Add user type column
    pivot["user_type"] = pivot["track"].map({
        "track2_sparse": "Sparse (≤5)",
        "track2_active": "Active (≥20)"
    })
    
    # Reorder columns
    cols = ["dataset", "model", "user_type"] + METRICS
    pivot = pivot[[c for c in cols if c in pivot.columns]]
    
    # Apply display names
    pivot["dataset"] = pivot["dataset"].map(DATASET_DISPLAY_NAMES)
    pivot["model"] = pivot["model"].map(MODEL_DISPLAY_NAMES)
    
    pivot = pivot.rename(columns={
        "dataset": "Dataset",
        "model": "Model",
        "user_type": "User Type"
    })
    
    return pivot


def create_cold_start_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Table 3: Cold-Start Performance (Track 3) with Cold/Warm Ratio.
    
    Returns:
        DataFrame with cold-start metrics and relative performance.
    """
    # Get Track 1 (warm) and Track 3 (cold) data
    warm_df = results_df[results_df["track"] == "track1_warm"].copy()
    cold_df = results_df[results_df["track"] == "track3_cold"].copy()
    
    # Pivot cold data
    cold_pivot = cold_df.pivot_table(
        index=["dataset", "model"],
        columns="metric",
        values="value",
        aggfunc="first"
    ).reset_index()
    
    # Pivot warm data for ratio calculation
    warm_pivot = warm_df.pivot_table(
        index=["dataset", "model"],
        columns="metric",
        values="value",
        aggfunc="first"
    ).reset_index()
    
    # Merge to calculate ratios
    merged = cold_pivot.merge(
        warm_pivot,
        on=["dataset", "model"],
        suffixes=("_cold", "_warm")
    )
    
    # Calculate Cold/Warm ratio for each recall metric
    result_data = []
    for _, row in merged.iterrows():
        record = {
            "dataset": row["dataset"],
            "model": row["model"],
        }
        
        # Add cold metrics
        for metric in METRICS:
            cold_col = f"{metric}_cold"
            if cold_col in row:
                record[metric] = row[cold_col]
        
        # Calculate Cold/Warm ratios for recall metrics
        for k in [10, 20, 50]:
            cold_key = f"recall@{k}_cold"
            warm_key = f"recall@{k}_warm"
            if cold_key in row and warm_key in row and row[warm_key] > 0:
                ratio = row[cold_key] / row[warm_key]
                record[f"ratio@{k}"] = ratio
        
        result_data.append(record)
    
    result_df = pd.DataFrame(result_data)
    
    # Reorder columns
    ratio_cols = ["ratio@10", "ratio@20", "ratio@50"]
    cols = ["dataset", "model"] + METRICS + ratio_cols
    result_df = result_df[[c for c in cols if c in result_df.columns]]
    
    # Apply display names
    result_df["dataset"] = result_df["dataset"].map(DATASET_DISPLAY_NAMES)
    result_df["model"] = result_df["model"].map(MODEL_DISPLAY_NAMES)
    
    result_df = result_df.rename(columns={
        "dataset": "Dataset",
        "model": "Model",
        "ratio@10": "Cold/Warm@10",
        "ratio@20": "Cold/Warm@20",
        "ratio@50": "Cold/Warm@50",
    })
    
    return result_df


def save_table(df: pd.DataFrame, output_dir: Path, name: str):
    """Save table as both CSV and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = output_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"Saved: {csv_path}")
    
    # Save Markdown
    md_path = output_dir / f"{name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
    logger.info(f"Saved: {md_path}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_curves(
    history: Dict[str, Dict[str, Dict]],
    output_dir: Path,
    metric_key: str,
    ylabel: str,
    filename_prefix: str,
):
    """
    Plot training curves for a specific metric across all datasets.
    
    Args:
        history: Nested dict {dataset: {model: {history}}}
        output_dir: Directory to save plots
        metric_key: Key in history dict (e.g., "train_loss", "val_recall")
        ylabel: Y-axis label
        filename_prefix: Prefix for output filenames
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset in DATASETS:
        if dataset not in history:
            continue
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        max_epochs = 0
        for model in MODELS:
            if model not in history[dataset]:
                continue
            
            model_history = history[dataset][model]
            if metric_key not in model_history:
                continue
            
            values = model_history[metric_key]
            epochs = range(1, len(values) + 1)
            max_epochs = max(max_epochs, len(values))
            
            ax.plot(
                epochs,
                values,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                markevery=max(1, len(values) // 10),  # Show ~10 markers
                markersize=6,
                linewidth=1.5,
                label=MODEL_DISPLAY_NAMES[model],
                alpha=0.9,
            )
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{DATASET_DISPLAY_NAMES[dataset]} - {ylabel}", fontsize=14)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, max_epochs)
        
        plt.tight_layout()
        
        output_path = output_dir / f"{filename_prefix}_{dataset}.png"
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved: {output_path}")


def plot_combined_training_curves(
    history: Dict[str, Dict[str, Dict]],
    output_dir: Path,
):
    """
    Plot combined training curves (Loss + Validation Recall) in a 2-panel figure.
    
    Args:
        history: Nested dict {dataset: {model: {history}}}
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset in DATASETS:
        if dataset not in history:
            continue
        
        # Create 2-panel figure (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        max_epochs = 0
        
        # Panel 1: Training Loss
        for model in MODELS:
            if model not in history[dataset]:
                continue
            
            model_history = history[dataset][model]
            if "train_loss" not in model_history:
                continue
            
            values = model_history["train_loss"]
            epochs = range(1, len(values) + 1)
            max_epochs = max(max_epochs, len(values))
            
            ax1.plot(
                epochs,
                values,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                markevery=max(1, len(values) // 10),
                markersize=5,
                linewidth=1.5,
                label=MODEL_DISPLAY_NAMES[model],
                alpha=0.9,
            )
        
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Training Loss (BPR)", fontsize=11)
        ax1.set_title("(a) Training Loss", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, max_epochs)
        
        # Panel 2: Validation Recall@20
        for model in MODELS:
            if model not in history[dataset]:
                continue
            
            model_history = history[dataset][model]
            if "val_recall" not in model_history:
                continue
            
            values = model_history["val_recall"]
            epochs = range(1, len(values) + 1)
            
            ax2.plot(
                epochs,
                values,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                markevery=max(1, len(values) // 10),
                markersize=5,
                linewidth=1.5,
                label=MODEL_DISPLAY_NAMES[model],
                alpha=0.9,
            )
        
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Validation Recall@20", fontsize=11)
        ax2.set_title("(b) Validation Recall@20", fontsize=12, fontweight="bold")
        ax2.legend(loc="lower right", fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, max_epochs)
        
        # Overall title
        fig.suptitle(
            f"{DATASET_DISPLAY_NAMES[dataset]} - Training Dynamics",
            fontsize=14,
            fontweight="bold",
            y=1.02
        )
        
        plt.tight_layout()
        
        output_path = output_dir / f"training_combined_{dataset}.png"
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved: {output_path}")


def plot_all_training_curves(history: Dict, output_dir: Path):
    """Generate all training curve plots."""
    
    # Combined plots (Loss + Val Recall in one figure)
    plot_combined_training_curves(history, output_dir)
    
    # Individual plots (kept for reference)
    # 1. Training Loss
    plot_training_curves(
        history, output_dir,
        metric_key="train_loss",
        ylabel="Training Loss (BPR)",
        filename_prefix="training_loss",
    )
    
    # 2. Validation Recall@20
    plot_training_curves(
        history, output_dir,
        metric_key="val_recall",
        ylabel="Validation Recall@20",
        filename_prefix="val_recall",
    )
    
    # 3. Validation NDCG@20
    plot_training_curves(
        history, output_dir,
        metric_key="val_ndcg",
        ylabel="Validation NDCG@20",
        filename_prefix="val_ndcg",
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    # Paths
    project_root = Path(__file__).parent.parent
    checkpoint_dir = project_root / "checkpoints"
    output_dir = project_root / "experiment_result"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT RESULTS EXTRACTION")
    logger.info("=" * 60)
    
    # -----------------------------------------------------------------
    # 1. Load evaluation results
    # -----------------------------------------------------------------
    logger.info("\n[1/4] Loading evaluation results...")
    results_df = load_eval_results(checkpoint_dir)
    
    if results_df.empty:
        logger.error("No evaluation results found!")
        return
    
    # -----------------------------------------------------------------
    # 2. Create comparison tables
    # -----------------------------------------------------------------
    logger.info("\n[2/4] Creating comparison tables...")
    
    # Table 1: Main Results (Track 1 - Warm)
    main_results = create_main_results_table(results_df)
    save_table(main_results, tables_dir, "main_results")
    
    # Table 2: User Robustness (Track 2)
    user_robustness = create_user_robustness_table(results_df)
    save_table(user_robustness, tables_dir, "user_robustness")
    
    # Table 3: Cold-Start Performance (Track 3)
    cold_start = create_cold_start_table(results_df)
    save_table(cold_start, tables_dir, "cold_start")
    
    # -----------------------------------------------------------------
    # 3. Load training history
    # -----------------------------------------------------------------
    logger.info("\n[3/4] Loading training histories...")
    history = load_training_history(checkpoint_dir)
    
    # -----------------------------------------------------------------
    # 4. Generate training curve plots
    # -----------------------------------------------------------------
    logger.info("\n[4/4] Generating training curve plots...")
    plot_all_training_curves(history, figures_dir)
    
    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"  - Tables: {tables_dir}")
    logger.info(f"  - Figures: {figures_dir}")
    
    # Print main results summary
    logger.info("\n--- Main Results Preview (Recall@20) ---")
    if "recall@20" in main_results.columns:
        preview = main_results[["Dataset", "Model", "recall@20"]].copy()
        preview = preview.rename(columns={"recall@20": "Recall@20"})
        print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
