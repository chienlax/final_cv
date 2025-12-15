"""
Visualization utilities for EDA.

Provides matplotlib/seaborn-based plotting functions for:
- Rating distributions
- User/item interaction frequency (log-scale)
- Temporal patterns (heatmaps, time series)
- Text length distributions
- Sparsity visualizations
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style defaults
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Default figure settings
DEFAULT_FIG_SIZE = (10, 6)
DEFAULT_DPI = 150


def save_figure(fig: plt.Figure, output_path: Path, name: str) -> Path:
    """Save figure to file and return path."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig_path = output_path / f"{name}.png"
    fig.savefig(fig_path, dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    logger.info(f"Saved figure: {fig_path}")
    return fig_path


def plot_rating_distribution(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    dataset_name: str = "Dataset",
) -> plt.Figure:
    """
    Plot rating distribution as bar chart with percentages.
    
    Args:
        df: DataFrame with 'rating' column.
        output_path: Optional path to save figure.
        dataset_name: Name for plot title.
        
    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    
    rating_counts = df["rating"].value_counts().sort_index()
    percentages = (rating_counts / len(df) * 100)
    
    colors = sns.color_palette("RdYlGn", n_colors=5)
    bars = ax.bar(rating_counts.index, rating_counts.values, color=colors, edgecolor="black", linewidth=0.5)
    
    # Add percentage labels
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(
            f"{pct:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )
    
    ax.set_xlabel("Rating", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Rating Distribution - {dataset_name}", fontsize=14, fontweight="bold")
    ax.set_xticks([1, 2, 3, 4, 5])
    
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path, f"rating_distribution_{dataset_name.lower().replace(' ', '_')}")
    
    return fig


def plot_interaction_frequency(
    user_freq: pd.Series,
    item_freq: pd.Series,
    output_path: Optional[Path] = None,
    dataset_name: str = "Dataset",
) -> plt.Figure:
    """
    Plot user and item interaction frequency distributions (log-log scale).
    
    Demonstrates power-law behavior typical in recommendation datasets.
    
    Args:
        user_freq: Series with user interaction counts.
        item_freq: Series with item interaction counts.
        output_path: Optional path to save figure.
        dataset_name: Name for plot title.
        
    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # User frequency (log-log)
    ax1 = axes[0]
    user_counts = user_freq.value_counts().sort_index()
    ax1.scatter(user_counts.index, user_counts.values, alpha=0.6, s=20, c="steelblue")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of Interactions (per user)", fontsize=11)
    ax1.set_ylabel("Number of Users", fontsize=11)
    ax1.set_title("User Interaction Frequency", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    # Add statistics annotation
    ax1.annotate(
        f"Median: {user_freq.median():.0f}\nMean: {user_freq.mean():.1f}\nMax: {user_freq.max():,}",
        xy=(0.95, 0.95), xycoords="axes fraction",
        fontsize=9, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    
    # Item frequency (log-log)
    ax2 = axes[1]
    item_counts = item_freq.value_counts().sort_index()
    ax2.scatter(item_counts.index, item_counts.values, alpha=0.6, s=20, c="darkorange")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of Interactions (per item)", fontsize=11)
    ax2.set_ylabel("Number of Items", fontsize=11)
    ax2.set_title("Item Interaction Frequency", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    ax2.annotate(
        f"Median: {item_freq.median():.0f}\nMean: {item_freq.mean():.1f}\nMax: {item_freq.max():,}",
        xy=(0.95, 0.95), xycoords="axes fraction",
        fontsize=9, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    
    fig.suptitle(f"Interaction Frequency Analysis - {dataset_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path, f"interaction_frequency_{dataset_name.lower().replace(' ', '_')}")
    
    return fig


def plot_temporal_patterns(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    dataset_name: str = "Dataset",
) -> plt.Figure:
    """
    Plot temporal patterns including monthly trend and weekday/hour heatmap.
    
    Args:
        df: DataFrame with 'timestamp' column (datetime).
        output_path: Optional path to save figure.
        dataset_name: Name for plot title.
        
    Returns:
        matplotlib Figure object.
    """
    if "timestamp" not in df.columns:
        logger.warning("No timestamp column found for temporal analysis")
        fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
        ax.text(0.5, 0.5, "No timestamp data available", ha="center", va="center", fontsize=14)
        return fig
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Monthly trend
    ax1 = axes[0]
    df_temp = df.copy()
    df_temp["year_month"] = df_temp["timestamp"].dt.to_period("M")
    monthly = df_temp.groupby("year_month").size()
    
    # Convert to proper datetime for plotting
    monthly_dates = monthly.index.to_timestamp()
    ax1.fill_between(monthly_dates, monthly.values, alpha=0.3, color="steelblue")
    ax1.plot(monthly_dates, monthly.values, color="steelblue", linewidth=2)
    
    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("Number of Reviews", fontsize=11)
    ax1.set_title("Monthly Review Trend", fontsize=12, fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))
    
    # Weekday x Hour heatmap
    ax2 = axes[1]
    df_temp["weekday"] = df_temp["timestamp"].dt.dayofweek
    df_temp["hour"] = df_temp["timestamp"].dt.hour
    
    heatmap_data = df_temp.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    
    sns.heatmap(
        heatmap_data,
        ax=ax2,
        cmap="YlOrRd",
        cbar_kws={"label": "Number of Reviews"},
    )
    
    ax2.set_xlabel("Hour of Day", fontsize=11)
    ax2.set_ylabel("Day of Week", fontsize=11)
    ax2.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)
    ax2.set_title("Review Activity by Day and Hour", fontsize=12, fontweight="bold")
    
    fig.suptitle(f"Temporal Analysis - {dataset_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path, f"temporal_patterns_{dataset_name.lower().replace(' ', '_')}")
    
    return fig


def plot_text_length_distribution(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    dataset_name: str = "Dataset",
) -> plt.Figure:
    """
    Plot review text and title length distributions.
    
    Args:
        df: DataFrame with 'review_text' and 'review_title' columns.
        output_path: Optional path to save figure.
        dataset_name: Name for plot title.
        
    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Review text length
    if "review_text" in df.columns:
        ax1 = axes[0]
        text_lengths = df["review_text"].fillna("").str.len()
        
        # Filter to reasonable range for visualization
        text_lengths_filtered = text_lengths[text_lengths <= text_lengths.quantile(0.99)]
        
        ax1.hist(text_lengths_filtered, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        ax1.axvline(text_lengths.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {text_lengths.mean():.0f}")
        ax1.axvline(text_lengths.median(), color="green", linestyle="--", linewidth=2, label=f"Median: {text_lengths.median():.0f}")
        
        ax1.set_xlabel("Review Text Length (characters)", fontsize=11)
        ax1.set_ylabel("Frequency", fontsize=11)
        ax1.set_title("Review Text Length Distribution", fontsize=12, fontweight="bold")
        ax1.legend()
    
    # Review title length
    if "review_title" in df.columns:
        ax2 = axes[1]
        title_lengths = df["review_title"].fillna("").str.len()
        title_lengths_filtered = title_lengths[title_lengths <= title_lengths.quantile(0.99)]
        
        ax2.hist(title_lengths_filtered, bins=50, color="darkorange", edgecolor="black", alpha=0.7)
        ax2.axvline(title_lengths.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {title_lengths.mean():.0f}")
        ax2.axvline(title_lengths.median(), color="green", linestyle="--", linewidth=2, label=f"Median: {title_lengths.median():.0f}")
        
        ax2.set_xlabel("Review Title Length (characters)", fontsize=11)
        ax2.set_ylabel("Frequency", fontsize=11)
        ax2.set_title("Review Title Length Distribution", fontsize=12, fontweight="bold")
        ax2.legend()
    
    fig.suptitle(f"Text Length Analysis - {dataset_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path, f"text_length_{dataset_name.lower().replace(' ', '_')}")
    
    return fig


def plot_sparsity_visualization(
    n_users: int,
    n_items: int,
    n_interactions: int,
    output_path: Optional[Path] = None,
    dataset_name: str = "Dataset",
) -> plt.Figure:
    """
    Visualize dataset sparsity with comparison to dense matrix.
    
    Args:
        n_users: Number of unique users.
        n_items: Number of unique items.
        n_interactions: Number of interactions.
        output_path: Optional path to save figure.
        dataset_name: Name for plot title.
        
    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    total_possible = n_users * n_items
    density = n_interactions / total_possible
    sparsity = 1 - density
    
    # Create pie-like bar visualization
    categories = ["Observed\nInteractions", "Missing\nInteractions"]
    values = [n_interactions, total_possible - n_interactions]
    colors = ["#2ecc71", "#e74c3c"]
    
    # Use log scale for comparison (since numbers are vastly different)
    log_values = [np.log10(max(v, 1)) for v in values]
    
    bars = ax.barh(categories, log_values, color=colors, edgecolor="black", height=0.5)
    
    # Add text annotations
    for bar, val, log_val in zip(bars, values, log_values):
        ax.text(
            log_val + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}\n({val/total_possible*100:.4f}%)",
            va="center", ha="left", fontsize=10,
        )
    
    ax.set_xlabel("Log10(Count)", fontsize=11)
    ax.set_title(f"Sparsity Visualization - {dataset_name}\nSparsity: {sparsity:.6%}", fontsize=12, fontweight="bold")
    
    # Add summary text
    summary_text = (
        f"Users: {n_users:,}\n"
        f"Items: {n_items:,}\n"
        f"Interactions: {n_interactions:,}\n"
        f"Possible: {total_possible:,.0f}\n"
        f"Density: {density:.6%}"
    )
    ax.text(
        0.98, 0.02, summary_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path, f"sparsity_{dataset_name.lower().replace(' ', '_')}")
    
    return fig


def plot_category_distribution(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    dataset_name: str = "Dataset",
    top_n: int = 15,
) -> plt.Figure:
    """
    Plot distribution of item categories.
    
    Args:
        df: DataFrame with 'main_category' column.
        output_path: Optional path to save figure.
        dataset_name: Name for plot title.
        top_n: Number of top categories to show.
        
    Returns:
        matplotlib Figure object.
    """
    if "main_category" not in df.columns:
        fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
        ax.text(0.5, 0.5, "No category data available", ha="center", va="center", fontsize=14)
        return fig
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    category_counts = df["main_category"].value_counts().head(top_n)
    
    colors = sns.color_palette("viridis", n_colors=len(category_counts))
    bars = ax.barh(category_counts.index[::-1], category_counts.values[::-1], color=colors[::-1], edgecolor="black")
    
    # Add count labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + ax.get_xlim()[1] * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width):,}",
            ha="left", va="center", fontsize=9,
        )
    
    ax.set_xlabel("Number of Items", fontsize=11)
    ax.set_ylabel("Category", fontsize=11)
    ax.set_title(f"Top {top_n} Categories - {dataset_name}", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path, f"category_distribution_{dataset_name.lower().replace(' ', '_')}")
    
    return fig


def plot_multimodal_coverage(
    metadata_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    dataset_name: str = "Dataset",
) -> plt.Figure:
    """
    Visualize multimodal feature coverage (text, images).
    
    Args:
        metadata_df: DataFrame with title, description, features, image_count columns.
        output_path: Optional path to save figure.
        dataset_name: Name for plot title.
        
    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Feature coverage
    ax1 = axes[0]
    
    coverage = {}
    if "title" in metadata_df.columns:
        coverage["Title"] = (metadata_df["title"].fillna("").str.len() > 0).mean() * 100
    if "description" in metadata_df.columns:
        coverage["Description"] = (metadata_df["description"].fillna("").str.len() > 0).mean() * 100
    if "features" in metadata_df.columns:
        coverage["Features"] = (metadata_df["features"].fillna("").str.len() > 0).mean() * 100
    if "image_count" in metadata_df.columns:
        coverage["Images"] = (metadata_df["image_count"] > 0).mean() * 100
    
    if coverage:
        colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"][:len(coverage)]
        bars = ax1.bar(coverage.keys(), coverage.values(), color=colors, edgecolor="black")
        
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
            )
        
        ax1.set_ylabel("Coverage (%)", fontsize=11)
        ax1.set_title("Feature Coverage", fontsize=12, fontweight="bold")
        ax1.set_ylim(0, 105)
    
    # Image count distribution
    ax2 = axes[1]
    if "image_count" in metadata_df.columns:
        img_counts = metadata_df["image_count"].value_counts().sort_index()
        img_counts = img_counts[img_counts.index <= 10]  # Limit to 0-10 images
        
        ax2.bar(img_counts.index, img_counts.values, color="steelblue", edgecolor="black")
        ax2.set_xlabel("Number of Images", fontsize=11)
        ax2.set_ylabel("Number of Items", fontsize=11)
        ax2.set_title("Image Count Distribution", fontsize=12, fontweight="bold")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))
    
    fig.suptitle(f"Multimodal Coverage - {dataset_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path, f"multimodal_coverage_{dataset_name.lower().replace(' ', '_')}")
    
    return fig
