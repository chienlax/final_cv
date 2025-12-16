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


# =============================================================================
# Academic Analysis Visualizations
# =============================================================================

def plot_modality_alignment(
    visual_similarities: list[float],
    interaction_similarities: list[float],
    pearson_r: float,
    pearson_p: float,
    output_path: Optional[Path] = None,
    dataset_name: str = "Dataset",
) -> plt.Figure:
    """
    Plot modality-interaction alignment scatter plot.
    
    Visualizes the correlation between visual and interaction similarities
    to assess the Homophily Hypothesis.
    
    Args:
        visual_similarities: List of visual similarity scores.
        interaction_similarities: List of interaction similarity scores.
        pearson_r: Pearson correlation coefficient.
        pearson_p: Pearson p-value.
        output_path: Optional path to save figure.
        dataset_name: Name for plot title.
        
    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(
        visual_similarities,
        interaction_similarities,
        alpha=0.4,
        s=20,
        c="steelblue",
        edgecolors="none",
    )
    
    # Add regression line
    if len(visual_similarities) > 2:
        z = np.polyfit(visual_similarities, interaction_similarities, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(visual_similarities), max(visual_similarities), 100)
        ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f"r = {pearson_r:.3f}")
    
    ax.set_xlabel("Visual Similarity (Cosine)", fontsize=12)
    ax.set_ylabel("Interaction Similarity (Jaccard)", fontsize=12)
    ax.set_title(
        f"Modality-Interaction Alignment - {dataset_name}\n"
        f"(Pearson r = {pearson_r:.3f}, p = {pearson_p:.4f})",
        fontsize=14, fontweight="bold"
    )
    
    # Add interpretation
    if pearson_p > 0.05:
        interpretation = "No significant correlation"
        color = "orange"
    elif abs(pearson_r) < 0.1:
        interpretation = "Very weak correlation"
        color = "orange"
    elif pearson_r > 0.3:
        interpretation = "Moderate-strong positive correlation"
        color = "green"
    else:
        interpretation = "Weak positive correlation"
        color = "lightgreen"
    
    ax.annotate(
        f"Interpretation: {interpretation}",
        xy=(0.05, 0.95), xycoords="axes fraction",
        fontsize=11, ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor=color, alpha=0.3),
    )
    
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path, f"modality_alignment_{dataset_name.lower().replace(' ', '_')}")
    
    return fig


def plot_visual_manifold(
    projection_x: list[float],
    projection_y: list[float],
    categories: list[str],
    ratings: list[float],
    silhouette_score: float,
    method: str = "umap",
    output_path: Optional[Path] = None,
    dataset_name: str = "Dataset",
) -> plt.Figure:
    """
    Plot visual manifold UMAP/t-SNE projection.
    
    Args:
        projection_x: X coordinates from projection.
        projection_y: Y coordinates from projection.
        categories: Category labels for coloring.
        ratings: Rating values for secondary coloring.
        silhouette_score: Cluster quality score.
        method: Projection method name.
        output_path: Optional path to save figure.
        dataset_name: Name for plot title.
        
    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Color by category
    ax1 = axes[0]
    unique_categories = list(set(categories))
    n_categories = len(unique_categories)
    
    # Limit colors if too many categories
    if n_categories <= 20:
        palette = sns.color_palette("husl", n_colors=n_categories)
        category_to_color = {cat: palette[i] for i, cat in enumerate(unique_categories)}
        colors = [category_to_color[cat] for cat in categories]
        
        scatter = ax1.scatter(
            projection_x, projection_y,
            c=colors, alpha=0.6, s=15, edgecolors="none"
        )
        
        # Add legend for top categories
        from matplotlib.patches import Patch
        top_categories = sorted(set(categories), key=categories.count, reverse=True)[:10]
        legend_elements = [
            Patch(facecolor=category_to_color[cat], label=cat[:30])
            for cat in top_categories
        ]
        ax1.legend(handles=legend_elements, loc="upper right", fontsize=8, title="Categories")
    else:
        # Too many categories, use a simple color
        ax1.scatter(projection_x, projection_y, alpha=0.5, s=10, c="steelblue")
        ax1.annotate(
            f"{n_categories} categories (too many to color)",
            xy=(0.05, 0.95), xycoords="axes fraction",
            fontsize=10, ha="left", va="top",
        )
    
    ax1.set_xlabel(f"{method.upper()} Dimension 1", fontsize=11)
    ax1.set_ylabel(f"{method.upper()} Dimension 2", fontsize=11)
    ax1.set_title(f"Colored by Category\n(Silhouette: {silhouette_score:.3f})", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.2)
    
    # Right: Color by rating
    ax2 = axes[1]
    valid_ratings = [r if r > 0 else np.nan for r in ratings]
    
    scatter = ax2.scatter(
        projection_x, projection_y,
        c=valid_ratings, cmap="RdYlGn", alpha=0.6, s=15, edgecolors="none",
        vmin=1, vmax=5,
    )
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
    cbar.set_label("Average Rating", fontsize=10)
    
    ax2.set_xlabel(f"{method.upper()} Dimension 1", fontsize=11)
    ax2.set_ylabel(f"{method.upper()} Dimension 2", fontsize=11)
    ax2.set_title("Colored by Average Rating", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.2)
    
    fig.suptitle(
        f"Visual Manifold Structure ({method.upper()}) - {dataset_name}\n"
        f"({len(projection_x):,} items projected)",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path, f"visual_manifold_{dataset_name.lower().replace(' ', '_')}")
    
    return fig


def plot_bpr_hardness_distribution(
    distance_distribution: list[float],
    pct_easy: float,
    pct_medium: float,
    pct_hard: float,
    mean_distance: float,
    output_path: Optional[Path] = None,
    dataset_name: str = "Dataset",
) -> plt.Figure:
    """
    Plot BPR negative sampling hardness distribution.
    
    Args:
        distance_distribution: Visual distances for positive-negative pairs.
        pct_easy: Percentage of easy negatives.
        pct_medium: Percentage of medium negatives.
        pct_hard: Percentage of hard negatives.
        mean_distance: Mean visual distance.
        output_path: Optional path to save figure.
        dataset_name: Name for plot title.
        
    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Histogram of visual distances
    ax1 = axes[0]
    
    ax1.hist(
        distance_distribution,
        bins=50,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )
    
    # Add threshold lines
    ax1.axvline(0.3, color="green", linestyle="--", linewidth=2, label="Hard threshold (0.3)")
    ax1.axvline(0.8, color="red", linestyle="--", linewidth=2, label="Easy threshold (0.8)")
    ax1.axvline(mean_distance, color="orange", linestyle="-", linewidth=2, label=f"Mean: {mean_distance:.3f}")
    
    ax1.set_xlabel("Visual Distance (1 - Cosine Similarity)", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Distribution of Positive-Negative Visual Distances", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right: Stacked bar for hardness categories
    ax2 = axes[1]
    
    categories = ["Hard\n(<0.3)", "Medium\n(0.3-0.8)", "Easy\n(>0.8)"]
    percentages = [pct_hard, pct_medium, pct_easy]
    colors = ["#27ae60", "#f39c12", "#e74c3c"]  # Green, Orange, Red
    
    bars = ax2.bar(categories, percentages, color=colors, edgecolor="black")
    
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax2.annotate(
            f"{pct:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold",
        )
    
    ax2.set_ylabel("Percentage of Negative Samples", fontsize=11)
    ax2.set_title("Negative Sample Hardness Categories", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(percentages) * 1.15)
    
    # Add recommendation
    if pct_easy > 80:
        recommendation = "⚠️ Most negatives are trivially easy!\nRecommend: Hard negative sampling"
        color = "#ffcccb"
    elif pct_easy > 50:
        recommendation = "⚡ Majority negatives are easy\nConsider: Mixed sampling strategy"
        color = "#fff3cd"
    else:
        recommendation = "✓ Good hardness distribution\nRandom sampling may suffice"
        color = "#d4edda"
    
    ax2.annotate(
        recommendation,
        xy=(0.5, 0.02), xycoords="axes fraction",
        fontsize=10, ha="center", va="bottom",
        bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
    )
    
    fig.suptitle(
        f"BPR Negative Sampling Hardness - {dataset_name}\n"
        f"({len(distance_distribution):,} pairs analyzed)",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path, f"bpr_hardness_{dataset_name.lower().replace(' ', '_')}")
    
    return fig
