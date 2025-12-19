"""
Extract ablation results from checkpoints_ablation and generate comparison tables/plots.

Outputs:
- ablation_result/tables/*.csv, *.md  (comparison tables)
- ablation_result/figures/*.png        (training curves)

Usage:
    python scripts/extract_ablation.py
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
ABLATION_MODES = ["no_visual", "no_text"]
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
# Ablation line styles
ABLATION_STYLES = {
    "full": "-",        # Solid line
    "no_visual": "--",  # Dashed
    "no_text": ":",     # Dotted
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
ABLATION_DISPLAY_NAMES = {
    "full": "Full",
    "no_visual": "No-Visual",
    "no_text": "No-Text",
}
DATASET_DISPLAY_NAMES = {
    "beauty": "Beauty",
    "clothing": "Clothing",
    "electronics": "Electronics",
}

# Figure settings
FIGURE_DPI = 300

# =============================================================================
# DATA LOADING
# =============================================================================

def load_ablation_results(ablation_dir: Path) -> pd.DataFrame:
    """
    Load eval_results.json from all ablation folders.
    
    Returns:
        DataFrame with columns: [dataset, model, ablation, track, metric, value]
    """
    records = []
    
    for dataset in DATASETS:
        for model in MODELS:
            for ablation in ABLATION_MODES:
                folder_name = f"{model}_{ablation}"
                eval_file = ablation_dir / dataset / folder_name / "eval_results.json"
                
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
                                "ablation": ablation,
                                "track": track,
                                "metric": metric,
                                "value": track_data[metric],
                            })
    
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} ablation metric records")
    return df


def load_main_results(checkpoint_dir: Path) -> pd.DataFrame:
    """
    Load baseline (full multimodal) results for comparison.
    """
    records = []
    
    for dataset in DATASETS:
        for model in MODELS:
            eval_file = checkpoint_dir / dataset / model / "eval_results.json"
            
            if not eval_file.exists():
                logger.warning(f"Missing baseline: {eval_file}")
                continue
            
            with open(eval_file) as f:
                data = json.load(f)
            
            for track, track_data in data.items():
                for metric in METRICS:
                    if metric in track_data:
                        records.append({
                            "dataset": dataset,
                            "model": model,
                            "ablation": "full",
                            "track": track,
                            "metric": metric,
                            "value": track_data[metric],
                        })
    
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} baseline metric records")
    return df


def load_ablation_history(ablation_dir: Path) -> Dict:
    """
    Load training_history.json from all ablation folders.
    
    Returns:
        Nested dict: {dataset: {model: {ablation: {history}}}}
    """
    history = {}
    
    for dataset in DATASETS:
        history[dataset] = {}
        for model in MODELS:
            history[dataset][model] = {}
            for ablation in ABLATION_MODES:
                folder_name = f"{model}_{ablation}"
                history_file = ablation_dir / dataset / folder_name / "training_history.json"
                
                if not history_file.exists():
                    logger.warning(f"Missing: {history_file}")
                    continue
                
                with open(history_file) as f:
                    history[dataset][model][ablation] = json.load(f)
    
    return history


def load_main_history(checkpoint_dir: Path) -> Dict:
    """Load baseline training history."""
    history = {}
    
    for dataset in DATASETS:
        history[dataset] = {}
        for model in MODELS:
            history_file = checkpoint_dir / dataset / model / "training_history.json"
            
            if not history_file.exists():
                continue
            
            with open(history_file) as f:
                history[dataset][model] = {"full": json.load(f)}
    
    return history


# =============================================================================
# TABLE GENERATION
# =============================================================================

def create_ablation_comparison_table(combined_df: pd.DataFrame, track: str = "track1_warm") -> pd.DataFrame:
    """
    Create table comparing Full vs No-Visual vs No-Text for a specific track.
    """
    df = combined_df[combined_df["track"] == track].copy()
    
    # Pivot to wide format
    pivot = df.pivot_table(
        index=["dataset", "model", "ablation"],
        columns="metric",
        values="value",
        aggfunc="first"
    ).reset_index()
    
    # Reorder columns
    cols = ["dataset", "model", "ablation"] + METRICS
    pivot = pivot[[c for c in cols if c in pivot.columns]]
    
    # Apply display names
    pivot["dataset"] = pivot["dataset"].map(DATASET_DISPLAY_NAMES)
    pivot["model"] = pivot["model"].map(MODEL_DISPLAY_NAMES)
    pivot["ablation"] = pivot["ablation"].map(ABLATION_DISPLAY_NAMES)
    
    pivot = pivot.rename(columns={
        "dataset": "Dataset",
        "model": "Model",
        "ablation": "Condition"
    })
    
    # Sort for readability
    pivot = pivot.sort_values(["Dataset", "Model", "Condition"])
    
    return pivot


def create_modality_contribution_table(combined_df: pd.DataFrame, track: str = "track1_warm") -> pd.DataFrame:
    """
    Calculate modality contribution: % drop when modality is removed.
    
    Visual Drop = (Full - No-Visual) / Full * 100
    Text Drop = (Full - No-Text) / Full * 100
    """
    df = combined_df[combined_df["track"] == track].copy()
    
    # Focus on primary metric: recall@20
    df_recall = df[df["metric"] == "recall@20"]
    
    result_data = []
    
    for dataset in DATASETS:
        for model in MODELS:
            ds_model_df = df_recall[(df_recall["dataset"] == dataset) & (df_recall["model"] == model)]
            
            full_val = ds_model_df[ds_model_df["ablation"] == "full"]["value"].values
            no_visual_val = ds_model_df[ds_model_df["ablation"] == "no_visual"]["value"].values
            no_text_val = ds_model_df[ds_model_df["ablation"] == "no_text"]["value"].values
            
            if len(full_val) == 0 or len(no_visual_val) == 0 or len(no_text_val) == 0:
                continue
            
            full_val = full_val[0]
            no_visual_val = no_visual_val[0]
            no_text_val = no_text_val[0]
            
            visual_drop = (full_val - no_visual_val) / full_val * 100 if full_val > 0 else 0
            text_drop = (full_val - no_text_val) / full_val * 100 if full_val > 0 else 0
            
            # Determine dominant modality
            if abs(visual_drop) > abs(text_drop):
                dominant = "Visual" if visual_drop > 0 else "Neither"
            elif abs(text_drop) > abs(visual_drop):
                dominant = "Text" if text_drop > 0 else "Neither"
            else:
                dominant = "Equal"
            
            result_data.append({
                "Dataset": DATASET_DISPLAY_NAMES[dataset],
                "Model": MODEL_DISPLAY_NAMES[model],
                "Full R@20": full_val,
                "No-Visual R@20": no_visual_val,
                "No-Text R@20": no_text_val,
                "Visual Drop (%)": visual_drop,
                "Text Drop (%)": text_drop,
                "Dominant": dominant,
            })
    
    return pd.DataFrame(result_data)


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

def plot_ablation_overview(
    ablation_history: Dict,
    main_history: Dict,
    output_dir: Path,
):
    """
    Plot all 6 ablation conditions per dataset in one figure.
    Layout: 1 row × 2 cols (Loss | Val Recall)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset in DATASETS:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        max_epochs = 0
        
        # Plot each model × ablation combination
        for model in MODELS:
            color = MODEL_COLORS[model]
            
            # Full (baseline) - from main_history
            if model in main_history.get(dataset, {}) and "full" in main_history[dataset][model]:
                hist = main_history[dataset][model]["full"]
                if "train_loss" in hist:
                    epochs = range(1, len(hist["train_loss"]) + 1)
                    max_epochs = max(max_epochs, len(hist["train_loss"]))
                    ax1.plot(epochs, hist["train_loss"], color=color, linestyle="-",
                             linewidth=1.5, alpha=0.9, label=f"{MODEL_DISPLAY_NAMES[model]} (Full)")
                if "val_recall" in hist:
                    epochs = range(1, len(hist["val_recall"]) + 1)
                    ax2.plot(epochs, hist["val_recall"], color=color, linestyle="-",
                             linewidth=1.5, alpha=0.9, label=f"{MODEL_DISPLAY_NAMES[model]} (Full)")
            
            # Ablation runs
            for ablation in ABLATION_MODES:
                if ablation in ablation_history.get(dataset, {}).get(model, {}):
                    hist = ablation_history[dataset][model][ablation]
                    linestyle = ABLATION_STYLES[ablation]
                    label_suffix = ABLATION_DISPLAY_NAMES[ablation]
                    
                    if "train_loss" in hist:
                        epochs = range(1, len(hist["train_loss"]) + 1)
                        max_epochs = max(max_epochs, len(hist["train_loss"]))
                        ax1.plot(epochs, hist["train_loss"], color=color, linestyle=linestyle,
                                 linewidth=1.2, alpha=0.7, label=f"{MODEL_DISPLAY_NAMES[model]} ({label_suffix})")
                    
                    if "val_recall" in hist:
                        epochs = range(1, len(hist["val_recall"]) + 1)
                        ax2.plot(epochs, hist["val_recall"], color=color, linestyle=linestyle,
                                 linewidth=1.2, alpha=0.7, label=f"{MODEL_DISPLAY_NAMES[model]} ({label_suffix})")
        
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Training Loss (BPR)", fontsize=11)
        ax1.set_title("(a) Training Loss", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, max_epochs)
        
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Validation Recall@20", fontsize=11)
        ax2.set_title("(b) Validation Recall@20", fontsize=12, fontweight="bold")
        ax2.legend(loc="lower right", fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, max_epochs)
        
        fig.suptitle(f"{DATASET_DISPLAY_NAMES[dataset]} - Ablation Analysis (All Conditions)",
                     fontsize=14, fontweight="bold", y=1.02)
        
        plt.tight_layout()
        output_path = output_dir / f"ablation_overview_{dataset}.png"
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {output_path}")


def plot_ablation_per_model(
    ablation_history: Dict,
    main_history: Dict,
    output_dir: Path,
):
    """
    Plot Full vs No-Visual vs No-Text per model (model-centric view).
    Creates 3 figures per dataset (one per model).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset in DATASETS:
        for model in MODELS:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
            
            color = MODEL_COLORS[model]
            max_epochs = 0
            
            # Full (baseline)
            if model in main_history.get(dataset, {}) and "full" in main_history[dataset][model]:
                hist = main_history[dataset][model]["full"]
                if "train_loss" in hist:
                    epochs = range(1, len(hist["train_loss"]) + 1)
                    max_epochs = max(max_epochs, len(hist["train_loss"]))
                    ax1.plot(epochs, hist["train_loss"], color=color, linestyle="-",
                             linewidth=2, marker="o", markevery=max(1, len(hist["train_loss"])//8),
                             markersize=5, alpha=0.9, label="Full")
                if "val_recall" in hist:
                    epochs = range(1, len(hist["val_recall"]) + 1)
                    ax2.plot(epochs, hist["val_recall"], color=color, linestyle="-",
                             linewidth=2, marker="o", markevery=max(1, len(hist["val_recall"])//8),
                             markersize=5, alpha=0.9, label="Full")
            
            # Ablation runs
            markers = {"no_visual": "s", "no_text": "^"}
            for ablation in ABLATION_MODES:
                if ablation in ablation_history.get(dataset, {}).get(model, {}):
                    hist = ablation_history[dataset][model][ablation]
                    linestyle = ABLATION_STYLES[ablation]
                    marker = markers[ablation]
                    label = ABLATION_DISPLAY_NAMES[ablation]
                    
                    if "train_loss" in hist:
                        epochs = range(1, len(hist["train_loss"]) + 1)
                        max_epochs = max(max_epochs, len(hist["train_loss"]))
                        ax1.plot(epochs, hist["train_loss"], color=color, linestyle=linestyle,
                                 linewidth=1.5, marker=marker, markevery=max(1, len(hist["train_loss"])//8),
                                 markersize=5, alpha=0.8, label=label)
                    
                    if "val_recall" in hist:
                        epochs = range(1, len(hist["val_recall"]) + 1)
                        ax2.plot(epochs, hist["val_recall"], color=color, linestyle=linestyle,
                                 linewidth=1.5, marker=marker, markevery=max(1, len(hist["val_recall"])//8),
                                 markersize=5, alpha=0.8, label=label)
            
            ax1.set_xlabel("Epoch", fontsize=11)
            ax1.set_ylabel("Training Loss (BPR)", fontsize=11)
            ax1.set_title("(a) Training Loss", fontsize=12, fontweight="bold")
            ax1.legend(loc="upper right", fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(1, max_epochs)
            
            ax2.set_xlabel("Epoch", fontsize=11)
            ax2.set_ylabel("Validation Recall@20", fontsize=11)
            ax2.set_title("(b) Validation Recall@20", fontsize=12, fontweight="bold")
            ax2.legend(loc="lower right", fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(1, max_epochs)
            
            fig.suptitle(f"{DATASET_DISPLAY_NAMES[dataset]} - {MODEL_DISPLAY_NAMES[model]} Ablation",
                         fontsize=14, fontweight="bold", y=1.02)
            
            plt.tight_layout()
            output_path = output_dir / f"ablation_{model}_{dataset}.png"
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    ablation_dir = project_root / "checkpoints_ablation"
    checkpoint_dir = project_root / "checkpoints"
    output_dir = project_root / "ablation_result"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    
    logger.info("=" * 60)
    logger.info("ABLATION RESULTS EXTRACTION")
    logger.info("=" * 60)
    
    # -----------------------------------------------------------------
    # 1. Load results
    # -----------------------------------------------------------------
    logger.info("\n[1/5] Loading ablation results...")
    ablation_df = load_ablation_results(ablation_dir)
    main_df = load_main_results(checkpoint_dir)
    
    if ablation_df.empty:
        logger.error("No ablation results found!")
        return
    
    # Combine for analysis
    combined_df = pd.concat([main_df, ablation_df], ignore_index=True)
    logger.info(f"Combined: {len(combined_df)} total metric records")
    
    # -----------------------------------------------------------------
    # 2. Create comparison tables
    # -----------------------------------------------------------------
    logger.info("\n[2/5] Creating comparison tables...")
    
    # Table 1: Warm ablation comparison
    warm_comparison = create_ablation_comparison_table(combined_df, "track1_warm")
    save_table(warm_comparison, tables_dir, "ablation_warm")
    
    # Table 2: Cold-start ablation comparison
    cold_comparison = create_ablation_comparison_table(combined_df, "track3_cold")
    save_table(cold_comparison, tables_dir, "ablation_cold")
    
    # Table 3: Modality contribution (warm)
    contribution_warm = create_modality_contribution_table(combined_df, "track1_warm")
    save_table(contribution_warm, tables_dir, "modality_contribution_warm")
    
    # Table 4: Modality contribution (cold)
    contribution_cold = create_modality_contribution_table(combined_df, "track3_cold")
    save_table(contribution_cold, tables_dir, "modality_contribution_cold")
    
    # -----------------------------------------------------------------
    # 3. Load training histories
    # -----------------------------------------------------------------
    logger.info("\n[3/5] Loading training histories...")
    ablation_history = load_ablation_history(ablation_dir)
    main_history = load_main_history(checkpoint_dir)
    
    # -----------------------------------------------------------------
    # 4. Generate overview plots (all conditions)
    # -----------------------------------------------------------------
    logger.info("\n[4/5] Generating overview plots...")
    plot_ablation_overview(ablation_history, main_history, figures_dir)
    
    # -----------------------------------------------------------------
    # 5. Generate per-model plots
    # -----------------------------------------------------------------
    logger.info("\n[5/5] Generating per-model plots...")
    plot_ablation_per_model(ablation_history, main_history, figures_dir)
    
    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"  - Tables: {tables_dir}")
    logger.info(f"  - Figures: {figures_dir}")
    
    # Print modality contribution summary
    logger.info("\n--- Modality Contribution Summary (Warm) ---")
    print(contribution_warm[["Dataset", "Model", "Visual Drop (%)", "Text Drop (%)", "Dominant"]].to_string(index=False))


if __name__ == "__main__":
    main()
