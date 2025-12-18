"""
Inductive preprocessing pipeline for cold-start multimodal recommendation.

Implements:
- Seed-based user sampling with size validation
- Recursive k-core pruning
- Warm/Cold item separation (80/20)
- Block-structured ID remapping
- Feature extraction with anisotropy correction
"""

import gzip
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .inductive_config import InductivePreprocessConfig

# Set OMP threads for P-cores only
os.environ["OMP_NUM_THREADS"] = "6"

logger = logging.getLogger(__name__)


def load_interactions(config: InductivePreprocessConfig) -> pd.DataFrame:
    """
    Load 5-core filtered interactions from CSV.
    
    Only loads required columns with strict dtypes to minimize RAM.
    
    Args:
        config: Preprocessing configuration.
        
    Returns:
        DataFrame with user_id, item_id, rating columns.
    """
    logger.info(f"Loading interactions from {config.interaction_file}")
    
    df = pd.read_csv(
        config.interaction_file,
        usecols=["user_id", "parent_asin", "rating"],
        dtype={
            "user_id": "str",
            "parent_asin": "str",
            "rating": "float32",
        },
    )
    
    # Rename to standard column names
    df = df.rename(columns={"parent_asin": "item_id"})
    
    logger.info(f"Loaded {len(df):,} interactions, {df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return df


def recursive_kcore_filter(
    df: pd.DataFrame,
    k: int = 5,
    max_iterations: int = 200,
) -> pd.DataFrame:
    """
    Apply recursive k-core filtering until convergence.
    
    Args:
        df: DataFrame with user_id, item_id columns.
        k: Minimum interactions per user/item.
        max_iterations: Safety limit on iterations.
        
    Returns:
        Filtered DataFrame.
    """
    logger.info(f"Applying recursive {k}-core filtering...")
    
    n_original = len(df)
    
    for iteration in range(max_iterations):
        n_before = len(df)
        
        # Filter items with < k interactions
        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= k].index
        df = df[df["item_id"].isin(valid_items)]
        
        # Filter users with < k interactions
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= k].index
        df = df[df["user_id"].isin(valid_users)]
        
        if len(df) == n_before:
            logger.info(f"K-core converged after {iteration + 1} iterations")
            break
    
    logger.info(f"K-core: {n_original:,} → {len(df):,} interactions ({len(df)/n_original*100:.1f}%)")
    
    return df.reset_index(drop=True)


def sample_and_prune(
    df: pd.DataFrame,
    config: InductivePreprocessConfig,
) -> pd.DataFrame:
    """
    Sample seed users and apply recursive k-core filtering.
    
    Automatically adjusts seed count if resulting graph is too small/large.
    
    Args:
        df: Full interaction DataFrame.
        config: Preprocessing configuration.
        
    Returns:
        Filtered DataFrame within target size range.
    """
    np.random.seed(config.seed)
    
    seed_users = config.seed_users
    max_attempts = 5
    
    for attempt in range(max_attempts):
        logger.info(f"Attempt {attempt + 1}: Sampling {seed_users:,} seed users...")
        
        # Sample users
        all_users = df["user_id"].unique()
        if seed_users >= len(all_users):
            sampled_users = all_users
        else:
            sampled_users = np.random.choice(all_users, size=seed_users, replace=False)
        
        # Filter to sampled users
        df_sub = df[df["user_id"].isin(sampled_users)].copy()
        
        # Apply recursive k-core
        df_sub = recursive_kcore_filter(df_sub, k=config.k_core)
        
        # Validate size
        n_users = df_sub["user_id"].nunique()
        n_items = df_sub["item_id"].nunique()
        n_total = n_users + n_items
        
        logger.info(f"Result: {n_users:,} users + {n_items:,} items = {n_total:,} total nodes")
        
        if n_total < config.min_total_nodes:
            # Too small - increase seed
            old_seed = seed_users
            seed_users = int(seed_users * 1.3)
            logger.warning(f"Graph too small ({n_total} < {config.min_total_nodes}). "
                          f"Increasing seed: {old_seed:,} → {seed_users:,}")
        elif n_total > config.max_total_nodes:
            # Too large - decrease seed
            old_seed = seed_users
            seed_users = int(seed_users * 0.7)
            logger.warning(f"Graph too large ({n_total} > {config.max_total_nodes}). "
                          f"Decreasing seed: {old_seed:,} → {seed_users:,}")
        else:
            logger.info(f"✓ Graph size {n_total:,} within target range [{config.min_total_nodes}, {config.max_total_nodes}]")
            return df_sub
    
    logger.warning(f"Could not achieve target size after {max_attempts} attempts. Using last result.")
    return df_sub


def split_items(
    df: pd.DataFrame,
    cold_ratio: float,
    seed: int,
) -> tuple[set, set]:
    """
    Split items into warm (training) and cold (test-only) sets.
    
    Args:
        df: Interaction DataFrame.
        cold_ratio: Fraction of items to hold out as cold.
        seed: Random seed.
        
    Returns:
        Tuple of (warm_item_ids, cold_item_ids).
    """
    np.random.seed(seed)
    
    all_items = df["item_id"].unique()
    n_cold = int(len(all_items) * cold_ratio)
    
    # Shuffle and split
    shuffled = np.random.permutation(all_items)
    cold_items = set(shuffled[:n_cold])
    warm_items = set(shuffled[n_cold:])
    
    logger.info(f"Item split: {len(warm_items):,} warm ({100-cold_ratio*100:.0f}%), "
                f"{len(cold_items):,} cold ({cold_ratio*100:.0f}%)")
    
    return warm_items, cold_items


def create_block_mappings(
    df: pd.DataFrame,
    warm_items: set,
    cold_items: set,
) -> dict:
    """
    Create block-structured ID mappings.
    
    Block structure:
    - Warm Items: [0, N_warm - 1]
    - Cold Items: [N_warm, N_total - 1]
    - Users: [0, N_users - 1] (separate namespace)
    
    Args:
        df: Interaction DataFrame.
        warm_items: Set of warm item IDs.
        cold_items: Set of cold item IDs.
        
    Returns:
        Dictionary with all mappings and metadata.
    """
    # Sort for reproducibility
    warm_list = sorted(warm_items)
    cold_list = sorted(cold_items)
    user_list = sorted(df["user_id"].unique())
    
    n_warm = len(warm_list)
    n_cold = len(cold_list)
    n_users = len(user_list)
    
    # Create mappings - warm items first, then cold
    item_to_idx = {}
    for idx, item in enumerate(warm_list):
        item_to_idx[item] = idx
    for idx, item in enumerate(cold_list):
        item_to_idx[item] = n_warm + idx
    
    user_to_idx = {user: idx for idx, user in enumerate(user_list)}
    
    # Reverse mappings
    idx_to_item = {v: k for k, v in item_to_idx.items()}
    idx_to_user = {v: k for k, v in user_to_idx.items()}
    
    maps = {
        "item_to_idx": item_to_idx,
        "idx_to_item": {str(k): v for k, v in idx_to_item.items()},  # JSON keys must be strings
        "user_to_idx": user_to_idx,
        "idx_to_user": {str(k): v for k, v in idx_to_user.items()},
        "n_warm_items": n_warm,
        "n_cold_items": n_cold,
        "n_items_total": n_warm + n_cold,
        "n_users": n_users,
    }
    
    logger.info(f"Created ID mappings: {n_users:,} users, {n_warm:,} warm items, {n_cold:,} cold items")
    
    return maps


def generate_split_files(
    df: pd.DataFrame,
    cold_items: set,
    maps: dict,
    output_dir: Path,
    config: InductivePreprocessConfig,
) -> dict:
    """
    Generate train/val/test split files.
    
    Files:
    - train.txt: 80% of warm interactions
    - val.txt: 10% of warm interactions
    - test_warm.txt: 10% of warm interactions
    - test_cold.txt: All cold item interactions
    
    Args:
        df: Interaction DataFrame.
        cold_items: Set of cold item IDs.
        maps: ID mappings dictionary.
        output_dir: Output directory.
        config: Preprocessing configuration.
        
    Returns:
        Dictionary with split statistics.
    """
    np.random.seed(config.seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    item_to_idx = maps["item_to_idx"]
    user_to_idx = maps["user_to_idx"]
    
    # Separate warm and cold interactions
    is_cold = df["item_id"].isin(cold_items)
    df_cold = df[is_cold].copy()
    df_warm = df[~is_cold].copy()
    
    logger.info(f"Warm interactions: {len(df_warm):,}, Cold interactions: {len(df_cold):,}")
    
    # Shuffle warm interactions
    df_warm = df_warm.sample(frac=1, random_state=config.seed).reset_index(drop=True)
    
    # Split warm interactions
    n_warm = len(df_warm)
    train_end = int(n_warm * config.train_ratio)
    val_end = int(n_warm * (config.train_ratio + config.val_ratio))
    
    df_train = df_warm.iloc[:train_end]
    df_val = df_warm.iloc[train_end:val_end]
    df_test_warm = df_warm.iloc[val_end:]
    
    def save_interactions(df_split: pd.DataFrame, filename: str):
        """Save interactions as space-separated user_idx item_idx."""
        filepath = output_dir / filename
        with open(filepath, "w") as f:
            for _, row in df_split.iterrows():
                user_idx = user_to_idx[row["user_id"]]
                item_idx = item_to_idx[row["item_id"]]
                f.write(f"{user_idx} {item_idx}\n")
        logger.info(f"Saved {len(df_split):,} interactions to {filename}")
    
    save_interactions(df_train, "train.txt")
    save_interactions(df_val, "val.txt")
    save_interactions(df_test_warm, "test_warm.txt")
    save_interactions(df_cold, "test_cold.txt")
    
    # Verify constraint: max item in train < n_warm
    train_items = df_train["item_id"].map(item_to_idx).values
    max_train_item = train_items.max()
    n_warm_items = maps["n_warm_items"]
    
    if max_train_item >= n_warm_items:
        raise ValueError(f"CRITICAL: Train contains cold items! max={max_train_item}, n_warm={n_warm_items}")
    
    logger.info(f"✓ Verified: max train item ({max_train_item}) < n_warm ({n_warm_items})")
    
    stats = {
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_test_warm": len(df_test_warm),
        "n_test_cold": len(df_cold),
    }
    
    return stats


def load_metadata_for_items(
    metadata_path: Path,
    item_ids: set,
) -> pd.DataFrame:
    """
    Stream metadata and filter to only needed items.
    
    Args:
        metadata_path: Path to metadata JSONL.gz file.
        item_ids: Set of item IDs to load.
        
    Returns:
        DataFrame with metadata for specified items.
    """
    logger.info(f"Loading metadata for {len(item_ids):,} items from {metadata_path}")
    
    records = []
    n_processed = 0
    
    # Check for uncompressed version first
    uncompressed_path = metadata_path.with_suffix("").with_suffix(".jsonl")
    if uncompressed_path.exists():
        open_func = open
        path = uncompressed_path
        logger.info(f"Using uncompressed metadata: {path}")
    else:
        open_func = gzip.open
        path = metadata_path
    
    with open_func(path, "rt", encoding="utf-8") as f:
        for line in f:
            n_processed += 1
            if n_processed % 100000 == 0:
                logger.info(f"Processed {n_processed:,} lines, found {len(records):,} items...")
            
            try:
                record = json.loads(line)
                item_id = record.get("parent_asin")
                if item_id in item_ids:
                    records.append({
                        "item_id": item_id,
                        "title": record.get("title", ""),
                        "description": " ".join(record.get("description", [])) if isinstance(record.get("description"), list) else record.get("description", ""),
                        "features": " ".join(record.get("features", [])) if isinstance(record.get("features"), list) else record.get("features", ""),
                        "images": record.get("images", []),
                    })
            except json.JSONDecodeError:
                continue
            
            # Early exit if we found all items
            if len(records) == len(item_ids):
                break
    
    df = pd.DataFrame(records)
    logger.info(f"Loaded metadata for {len(df):,}/{len(item_ids):,} items")
    
    return df


def extract_features(
    maps: dict,
    metadata_df: pd.DataFrame,
    output_dir: Path,
    config: InductivePreprocessConfig,
) -> dict:
    """
    Extract CLIP and SBERT features with anisotropy correction.
    
    OPTIMIZED VERSION:
    - Parallel image downloads (ThreadPoolExecutor)
    - Batched GPU inference
    - Prefetch queue for pipelining
    
    Args:
        maps: ID mappings dictionary.
        metadata_df: Metadata DataFrame.
        output_dir: Output directory.
        config: Preprocessing configuration.
        
    Returns:
        Dictionary with extraction statistics.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from queue import Queue
    from threading import Thread
    import time
    
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        from sentence_transformers import SentenceTransformer
        from PIL import Image
        import requests
        from io import BytesIO
    except ImportError as e:
        logger.error(f"Missing dependencies for feature extraction: {e}")
        raise
    
    output_dir = Path(output_dir)
    n_items = maps["n_items_total"]
    n_warm = maps["n_warm_items"]
    item_to_idx = maps["item_to_idx"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Extracting features on device: {device}")
    
    # Create item_id to metadata lookup
    metadata_lookup = {row["item_id"]: row for _, row in metadata_df.iterrows()}
    
    # =========================================================================
    # Text Features (SBERT) - Already batched, just run it
    # =========================================================================
    logger.info(f"Loading SBERT model: {config.sbert_model}")
    sbert = SentenceTransformer(config.sbert_model, device=device)
    
    # Prepare texts in ID order
    texts = []
    for idx in range(n_items):
        item_id = maps["idx_to_item"][str(idx)]
        if item_id in metadata_lookup:
            meta = metadata_lookup[item_id]
            text_parts = []
            for col in config.text_columns:
                if col in meta and meta[col]:
                    text_parts.append(str(meta[col]))
            text = " ".join(text_parts)[:config.max_text_length]
        else:
            text = ""
        texts.append(text if text else "unknown item")
    
    logger.info(f"Encoding {len(texts):,} texts with SBERT (batch_size={config.feature_batch_size})...")
    text_embeddings = sbert.encode(
        texts,
        batch_size=config.feature_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    
    # Anisotropy correction for text
    text_embeddings = apply_anisotropy_correction(text_embeddings, n_warm)
    
    np.save(output_dir / "feat_text.npy", text_embeddings.astype(np.float32))
    logger.info(f"Saved text features: {text_embeddings.shape}")
    
    # Free SBERT memory
    del sbert
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # =========================================================================
    # Visual Features (CLIP) - OPTIMIZED with parallel downloads + batched inference
    # =========================================================================
    logger.info(f"Loading CLIP model: {config.clip_model}")
    clip_model = CLIPModel.from_pretrained(config.clip_model).to(device)
    clip_processor = CLIPProcessor.from_pretrained(config.clip_model, use_fast=True)
    clip_model.eval()
    
    visual_dim = clip_model.config.projection_dim
    visual_embeddings = np.zeros((n_items, visual_dim), dtype=np.float32)
    
    # Collect all image URLs
    def get_image_url(meta):
        """Extract best image URL from metadata."""
        if meta is None:
            return None
        images = meta.get("images", [])
        if isinstance(images, list) and len(images) > 0:
            if isinstance(images[0], dict):
                return images[0].get("hi_res") or images[0].get("large")
            elif isinstance(images[0], str):
                return images[0]
        return None
    
    urls = []
    for idx in range(n_items):
        item_id = maps["idx_to_item"][str(idx)]
        url = get_image_url(metadata_lookup.get(item_id))
        urls.append(url)
    
    # --- Parallel Download Function ---
    def download_image(args):
        """Download single image with timeout."""
        idx, url = args
        if url is None:
            return idx, None
        try:
            response = requests.get(url, timeout=config.download_timeout)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return idx, img
        except Exception:
            return idx, None
    
    # --- Prefetch Queue with Background Thread ---
    batch_size = config.feature_batch_size
    n_workers = config.n_download_workers
    prefetch_queue = Queue(maxsize=2)  # Buffer 2 batches
    
    def prefetch_worker():
        """Background thread that downloads batches ahead of time."""
        for batch_start in range(0, n_items, batch_size):
            batch_end = min(batch_start + batch_size, n_items)
            batch_args = [(i, urls[i]) for i in range(batch_start, batch_end)]
            
            # Download batch in parallel
            results = {}
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(download_image, arg) for arg in batch_args]
                for future in as_completed(futures):
                    idx, img = future.result()
                    if img is not None:
                        results[idx] = img
            
            prefetch_queue.put((batch_start, batch_end, results))
        
        prefetch_queue.put(None)  # Sentinel
    
    # Start prefetching
    prefetch_thread = Thread(target=prefetch_worker, daemon=True)
    prefetch_thread.start()
    
    logger.info(f"Extracting visual features for {n_items:,} items...")
    logger.info(f"  Batch size: {batch_size}, Download workers: {n_workers}")
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    # Process batches from prefetch queue
    while True:
        item = prefetch_queue.get()
        if item is None:
            break
        
        batch_start, batch_end, downloaded = item
        batch_size_actual = batch_end - batch_start
        
        if not downloaded:
            failed += batch_size_actual
            continue
        
        # Prepare batch for CLIP
        indices = sorted(downloaded.keys())
        images = [downloaded[i] for i in indices]
        
        if len(images) == 0:
            failed += batch_size_actual
            continue
        
        try:
            # Batch process through CLIP
            inputs = clip_processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                features = clip_model.get_image_features(**inputs)
            
            # Store results
            features_np = features.cpu().numpy()
            for i, idx in enumerate(indices):
                visual_embeddings[idx] = features_np[i]
            
            successful += len(images)
            failed += batch_size_actual - len(images)
            
        except Exception as e:
            logger.warning(f"Batch {batch_start}-{batch_end} failed: {e}")
            failed += batch_size_actual
            continue
        
        # Progress logging
        processed = batch_end
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (n_items - processed) / rate if rate > 0 else 0
        
        if processed % (batch_size * 5) == 0 or processed == n_items:
            logger.info(f"Progress: {processed}/{n_items} ({successful} ok, {failed} fail) "
                       f"| {rate:.1f} items/s | ETA: {eta:.0f}s")
    
    elapsed_total = time.time() - start_time
    logger.info(f"Visual extraction complete: {successful:,} successful, {failed:,} failed "
               f"in {elapsed_total:.1f}s ({n_items/elapsed_total:.1f} items/s)")
    
    # Anisotropy correction for visual
    visual_embeddings = apply_anisotropy_correction(visual_embeddings, n_warm)
    
    np.save(output_dir / "feat_visual.npy", visual_embeddings.astype(np.float32))
    logger.info(f"Saved visual features: {visual_embeddings.shape}")
    
    return {
        "text_dim": text_embeddings.shape[1],
        "visual_dim": visual_embeddings.shape[1],
        "visual_successful": successful,
        "visual_failed": failed,
    }


def apply_anisotropy_correction(
    embeddings: np.ndarray,
    n_warm: int,
) -> np.ndarray:
    """
    Apply anisotropy correction (whitening) to embeddings.
    
    Uses only warm items for mean computation to prevent data leakage.
    
    Args:
        embeddings: (N, D) embedding matrix.
        n_warm: Number of warm items (indices 0 to n_warm-1).
        
    Returns:
        Corrected embeddings.
    """
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    embeddings = embeddings / norms
    
    # Compute mean from warm items only
    mu = embeddings[:n_warm].mean(axis=0)
    
    # Center all items
    embeddings = embeddings - mu
    
    # Re-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings = embeddings / norms
    
    return embeddings


def generate_statistics(
    maps: dict,
    split_stats: dict,
    feature_stats: dict,
    output_dir: Path,
    config: InductivePreprocessConfig,
) -> None:
    """
    Generate stat.txt for paper reporting.
    
    Args:
        maps: ID mappings dictionary.
        split_stats: Split file statistics.
        feature_stats: Feature extraction statistics.
        output_dir: Output directory.
        config: Preprocessing configuration.
    """
    output_dir = Path(output_dir)
    
    n_users = maps["n_users"]
    n_warm = maps["n_warm_items"]
    n_cold = maps["n_cold_items"]
    n_items = maps["n_items_total"]
    
    # Calculate density
    n_train = split_stats["n_train"]
    density = n_train / (n_users * n_warm) * 100
    
    stat_content = f"""Dataset: {config.dataset.title()}
=====================================

Graph Statistics
----------------
N_users: {n_users:,}
N_items_total: {n_items:,}
N_items_warm: {n_warm:,} ({n_warm/n_items*100:.1f}%)
N_items_cold: {n_cold:,} ({n_cold/n_items*100:.1f}%)

Interaction Statistics
----------------------
N_train: {split_stats['n_train']:,}
N_val: {split_stats['n_val']:,}
N_test_warm: {split_stats['n_test_warm']:,}
N_test_cold: {split_stats['n_test_cold']:,}
Density (train): {density:.4f}%

Feature Statistics
------------------
Text Feature Dim: {feature_stats.get('text_dim', 'N/A')}
Visual Feature Dim: {feature_stats.get('visual_dim', 'N/A')}
Visual Extraction Success: {feature_stats.get('visual_successful', 0):,}
Visual Extraction Failed: {feature_stats.get('visual_failed', 0):,}

Configuration
-------------
Seed Users: {config.seed_users:,}
K-Core: {config.k_core}
Cold Item Ratio: {config.cold_item_ratio}
Random Seed: {config.seed}
"""
    
    with open(output_dir / "stat.txt", "w") as f:
        f.write(stat_content)
    
    logger.info(f"Saved statistics to stat.txt")


def run_preprocessing(config: InductivePreprocessConfig) -> None:
    """
    Run the complete preprocessing pipeline.
    
    Args:
        config: Preprocessing configuration.
    """
    logger.info("=" * 60)
    logger.info("Starting Inductive Preprocessing Pipeline")
    logger.info("=" * 60)
    
    output_dir = config.dataset_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load interactions
    df = load_interactions(config)
    
    # Step 2: Sample and prune
    df = sample_and_prune(df, config)
    
    # Step 3: Split items into warm/cold
    warm_items, cold_items = split_items(df, config.cold_item_ratio, config.seed)
    
    # Step 4: Create block ID mappings
    maps = create_block_mappings(df, warm_items, cold_items)
    
    # Save maps immediately
    with open(output_dir / "maps.json", "w") as f:
        json.dump(maps, f, indent=2)
    logger.info(f"Saved maps.json")
    
    # Step 5: Generate split files
    split_stats = generate_split_files(df, cold_items, maps, output_dir, config)
    
    # Step 6: Extract features
    all_items = set(maps["item_to_idx"].keys())
    metadata_df = load_metadata_for_items(config.metadata_file, all_items)
    feature_stats = extract_features(maps, metadata_df, output_dir, config)
    
    # Step 7: Generate statistics
    generate_statistics(maps, split_stats, feature_stats, output_dir, config)
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Preprocessing Complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)
