"""
CLI entry point for inductive preprocessing.

Usage:
    python src/preprocessing/run_preprocessing.py --dataset electronics --seed-users 10000
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.inductive_config import InductivePreprocessConfig
from src.preprocessing.inductive_pipeline import run_preprocessing


def setup_logging():
    """Configure logging for preprocessing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Inductive preprocessing for multimodal recommendation"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="electronics",
        choices=["electronics", "beauty", "clothing"],
        help="Dataset to preprocess",
    )
    parser.add_argument(
        "--seed-users",
        type=int,
        default=10000,
        help="Number of seed users for sampling (default: 10000)",
    )
    parser.add_argument(
        "--k-core",
        type=int,
        default=5,
        help="Minimum interactions per user/item (default: 5)",
    )
    parser.add_argument(
        "--cold-ratio",
        type=float,
        default=0.2,
        help="Fraction of items to hold as cold (default: 0.2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Base output directory (default: data/processed)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Input data directory (default: data/raw)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed (default: 2024)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature extraction (for debugging)",
    )
    
    args = parser.parse_args()
    
    # Set OMP threads for P-cores
    os.environ["OMP_NUM_THREADS"] = "6"
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Inductive Preprocessing for Multimodal Recommendation")
    logger.info("=" * 60)
    
    # Create config
    config = InductivePreprocessConfig(
        dataset=args.dataset,
        seed_users=args.seed_users,
        k_core=args.k_core,
        cold_item_ratio=args.cold_ratio,
        output_dir=Path(args.output_dir),
        data_dir=Path(args.data_dir),
        seed=args.seed,
    )
    
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Seed Users: {config.seed_users:,}")
    logger.info(f"K-Core: {config.k_core}")
    logger.info(f"Cold Item Ratio: {config.cold_item_ratio}")
    logger.info(f"Output: {config.dataset_output_dir}")
    
    # Run preprocessing
    run_preprocessing(config)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
