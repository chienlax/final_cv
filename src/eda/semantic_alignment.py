"""
Semantic-Interaction Alignment Analysis.

Tests the hypothesis: Do items with similar text descriptions have similar buyers?

This implements the "Semantic Gap" validation from Liu et al. (2024):
- Sim_text(i,j): Cosine similarity of SBERT embeddings
- Sim_interaction(i,j): Jaccard similarity of user sets
- Metric: Pearson correlation between the two

Interpretation:
- r < 0.05: Text is noise/spammy
- r 0.05-0.15: Weak signal
- r > 0.15: Strong predictor (proceed with text-based model)
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SemanticAlignmentResult:
    """Results from semantic-interaction alignment analysis."""
    
    n_pairs_sampled: int = 0
    
    # Correlation metrics
    pearson_correlation: float = 0.0
    pearson_pvalue: float = 1.0
    spearman_correlation: float = 0.0
    spearman_pvalue: float = 1.0
    
    # Summary statistics
    mean_text_similarity: float = 0.0
    std_text_similarity: float = 0.0
    mean_interaction_similarity: float = 0.0
    std_interaction_similarity: float = 0.0
    
    # Signal strength interpretation
    signal_strength: str = ""  # "noise", "weak", "moderate", "strong"
    interpretation: str = ""
    recommendation: str = ""
    
    # Raw data for plotting
    text_similarities: list[float] = field(default_factory=list)
    interaction_similarities: list[float] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_pairs_sampled": self.n_pairs_sampled,
            "pearson": {
                "correlation": round(self.pearson_correlation, 4) if not np.isnan(self.pearson_correlation) else None,
                "pvalue": round(self.pearson_pvalue, 4) if not np.isnan(self.pearson_pvalue) else None,
            },
            "spearman": {
                "correlation": round(self.spearman_correlation, 4) if not np.isnan(self.spearman_correlation) else None,
                "pvalue": round(self.spearman_pvalue, 4) if not np.isnan(self.spearman_pvalue) else None,
            },
            "statistics": {
                "mean_text_similarity": round(self.mean_text_similarity, 4),
                "std_text_similarity": round(self.std_text_similarity, 4),
                "mean_interaction_similarity": round(self.mean_interaction_similarity, 4),
                "std_interaction_similarity": round(self.std_interaction_similarity, 4),
            },
            "signal_strength": self.signal_strength,
            "interpretation": self.interpretation,
            "recommendation": self.recommendation,
        }


def compute_text_similarity_cosine(
    embeddings: np.ndarray,
    item_indices: dict[str, int],
    item_pairs: list[tuple[str, str]],
) -> np.ndarray:
    """
    Compute cosine similarity between text embeddings for given pairs.
    
    Args:
        embeddings: Array of shape [n_items, embedding_dim].
        item_indices: Dictionary mapping item_id to embedding index.
        item_pairs: List of (item_id_1, item_id_2) pairs.
        
    Returns:
        Array of cosine similarities for each pair.
    """
    similarities = []
    
    for item_a, item_b in item_pairs:
        if item_a not in item_indices or item_b not in item_indices:
            continue
            
        idx_a = item_indices[item_a]
        idx_b = item_indices[item_b]
        
        emb_a = embeddings[idx_a]
        emb_b = embeddings[idx_b]
        
        # Cosine similarity (embeddings assumed normalized)
        sim = np.dot(emb_a, emb_b)
        similarities.append(sim)
    
    return np.array(similarities)


def compute_interaction_similarity_jaccard(
    item_users: dict[str, set[str]],
    item_pairs: list[tuple[str, str]],
) -> np.ndarray:
    """
    Compute Jaccard similarity of user interaction sets for item pairs.
    
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    
    Args:
        item_users: Dictionary mapping item_id to set of user_ids who interacted.
        item_pairs: List of (item_id_1, item_id_2) pairs.
        
    Returns:
        Array of Jaccard similarities for each pair.
    """
    similarities = []
    
    for item_a, item_b in item_pairs:
        users_a = item_users.get(item_a, set())
        users_b = item_users.get(item_b, set())
        
        if len(users_a) == 0 and len(users_b) == 0:
            similarities.append(0.0)
            continue
        
        intersection = len(users_a & users_b)
        union = len(users_a | users_b)
        
        jaccard = intersection / union if union > 0 else 0.0
        similarities.append(jaccard)
    
    return np.array(similarities)


def sample_item_pairs(
    item_ids: list[str],
    n_pairs: int = 5000,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """
    Sample random item pairs for similarity analysis.
    
    Args:
        item_ids: List of item IDs.
        n_pairs: Number of pairs to sample.
        seed: Random seed.
        
    Returns:
        List of (item_id_1, item_id_2) tuples.
    """
    np.random.seed(seed)
    
    n_items = len(item_ids)
    max_pairs = n_items * (n_items - 1) // 2
    n_pairs = min(n_pairs, max_pairs)
    
    pairs = set()
    attempts = 0
    max_attempts = n_pairs * 10
    
    while len(pairs) < n_pairs and attempts < max_attempts:
        i, j = np.random.randint(0, n_items, size=2)
        if i != j:
            pair = (item_ids[min(i, j)], item_ids[max(i, j)])
            pairs.add(pair)
        attempts += 1
    
    return list(pairs)


def interpret_correlation(r: float) -> tuple[str, str, str]:
    """
    Interpret Pearson correlation for semantic-interaction alignment.
    
    Args:
        r: Pearson correlation coefficient.
        
    Returns:
        Tuple of (signal_strength, interpretation, recommendation).
    """
    if np.isnan(r):
        return (
            "unknown",
            "Unable to compute correlation (insufficient data or zero variance)",
            "Check data quality and ensure sufficient item overlap",
        )
    
    if r < 0.05:
        return (
            "noise",
            f"Very weak correlation (r={r:.4f}): Text descriptions do NOT predict user behavior. "
            "Users likely buy based on visual appeal, brand, or price rather than descriptions.",
            "Deprioritize text encoder in final model, or use text only as filter/fallback.",
        )
    elif r < 0.15:
        return (
            "weak",
            f"Weak correlation (r={r:.4f}): Some signal exists but text is not a strong predictor. "
            "Descriptions may be generic or users only skim them.",
            "Include text modality but with lower weight. Consider fine-tuning SBERT on domain.",
        )
    elif r < 0.30:
        return (
            "moderate",
            f"Moderate correlation (r={r:.4f}): Text provides meaningful signal for recommendations. "
            "Users with similar text preferences tend to overlap.",
            "Proceed with text-based features. LATTICE text-item graph will be useful.",
        )
    else:
        return (
            "strong",
            f"Strong correlation (r={r:.4f}): Text is a strong proxy for user preferences! "
            "Items with similar descriptions share buyers.",
            "Full confidence in text modality. Use SBERT embeddings as primary feature.",
        )


def analyze_semantic_alignment(
    interactions_df: pd.DataFrame,
    text_embeddings: np.ndarray,
    item_indices: dict[str, int],
    n_pairs: int = 5000,
    seed: int = 42,
) -> SemanticAlignmentResult:
    """
    Analyze alignment between text semantics and interaction patterns.
    
    Tests the hypothesis: Do items with similar descriptions have similar buyers?
    
    Args:
        interactions_df: DataFrame with 'user_id' and 'item_id' columns.
        text_embeddings: Text embeddings matrix (n_items, embedding_dim).
        item_indices: Dictionary mapping item_id to embedding index.
        n_pairs: Number of item pairs to sample.
        seed: Random seed.
        
    Returns:
        SemanticAlignmentResult with correlation statistics.
    """
    logger.info("Analyzing semantic-interaction alignment...")
    
    result = SemanticAlignmentResult()
    
    if len(text_embeddings) < 2:
        result.interpretation = "Insufficient items for alignment analysis"
        return result
    
    # Build item -> user set mapping
    logger.info("  Building item-user mappings...")
    item_users = {}
    for _, row in interactions_df.iterrows():
        item_id = row["item_id"]
        user_id = row["user_id"]
        
        if item_id not in item_users:
            item_users[item_id] = set()
        item_users[item_id].add(user_id)
    
    # Find items with BOTH embeddings AND interactions
    common_items = [
        item_id for item_id in item_indices.keys()
        if item_id in item_users and len(item_users[item_id]) >= 2
    ]
    
    logger.info(f"  Items with embeddings: {len(item_indices)}")
    logger.info(f"  Items with interactions: {len(item_users)}")
    logger.info(f"  Common items (≥2 users): {len(common_items)}")
    
    if len(common_items) < 100:
        result.interpretation = f"Insufficient common items ({len(common_items)}). Need at least 100."
        return result
    
    # Sample item pairs
    logger.info(f"  Sampling {n_pairs} item pairs...")
    item_pairs = sample_item_pairs(common_items, n_pairs=n_pairs, seed=seed)
    result.n_pairs_sampled = len(item_pairs)
    
    if len(item_pairs) < 100:
        result.interpretation = f"Insufficient pairs sampled ({len(item_pairs)})"
        return result
    
    # Compute similarities
    logger.info("  Computing text similarities (cosine)...")
    text_sims = compute_text_similarity_cosine(text_embeddings, item_indices, item_pairs)
    
    logger.info("  Computing interaction similarities (Jaccard)...")
    interaction_sims = compute_interaction_similarity_jaccard(item_users, item_pairs)
    
    # Filter out zero-interaction pairs
    valid_mask = interaction_sims > 0
    if valid_mask.sum() < 100:
        # If too few overlapping users, include zeros but note it
        logger.warning(f"  Only {valid_mask.sum()} pairs have overlapping users")
        valid_mask = np.ones(len(text_sims), dtype=bool)
    
    text_sims_valid = text_sims[valid_mask]
    interaction_sims_valid = interaction_sims[valid_mask]
    
    # Store for plotting
    result.text_similarities = text_sims_valid.tolist()
    result.interaction_similarities = interaction_sims_valid.tolist()
    
    # Statistics
    result.mean_text_similarity = float(np.mean(text_sims_valid))
    result.std_text_similarity = float(np.std(text_sims_valid))
    result.mean_interaction_similarity = float(np.mean(interaction_sims_valid))
    result.std_interaction_similarity = float(np.std(interaction_sims_valid))
    
    # Correlation analysis
    logger.info("  Computing correlations...")
    
    # Check for zero variance
    if np.std(text_sims_valid) < 1e-10 or np.std(interaction_sims_valid) < 1e-10:
        logger.warning("  Zero variance in similarities - cannot compute correlation")
        result.pearson_correlation = np.nan
        result.spearman_correlation = np.nan
    else:
        # Pearson correlation
        try:
            pearson_r, pearson_p = stats.pearsonr(text_sims_valid, interaction_sims_valid)
            result.pearson_correlation = float(pearson_r)
            result.pearson_pvalue = float(pearson_p)
        except Exception as e:
            logger.warning(f"  Pearson failed: {e}")
            result.pearson_correlation = np.nan
        
        # Spearman correlation
        try:
            spearman_r, spearman_p = stats.spearmanr(text_sims_valid, interaction_sims_valid)
            result.spearman_correlation = float(spearman_r)
            result.spearman_pvalue = float(spearman_p)
        except Exception as e:
            logger.warning(f"  Spearman failed: {e}")
            result.spearman_correlation = np.nan
    
    # Interpretation
    result.signal_strength, result.interpretation, result.recommendation = \
        interpret_correlation(result.pearson_correlation)
    
    logger.info(f"  Pearson r: {result.pearson_correlation:.4f} (p={result.pearson_pvalue:.4f})")
    logger.info(f"  Signal: {result.signal_strength.upper()}")
    
    return result
