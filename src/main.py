"""
Main entry point for training and evaluating multimodal recommendation models.

Usage:
    python src/main.py --model lattice --dataset electronics
    python src/main.py --model micro --dataset electronics
    python src/main.py --model diffmm --dataset electronics
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.common import Config, set_seed, setup_logging, get_device
from src.dataset import RecDataset
from src.models import LATTICEModel, MICROModel, DiffMM
from src.trainer import Trainer, run_three_track_evaluation


def create_model(
    model_name: str,
    dataset: RecDataset,
    config: Config,
) -> torch.nn.Module:
    """
    Create model instance.
    
    Args:
        model_name: One of "lattice", "micro", "diffmm".
        dataset: RecDataset instance.
        config: Config object.
        
    Returns:
        Model instance.
    """
    common_args = dict(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        n_warm=dataset.n_warm,
        embed_dim=config.EMBED_DIM,
        n_layers=config.N_LAYERS,
        feat_visual=dataset.feat_visual,
        feat_text=dataset.feat_text,
        device=config.DEVICE,
    )
    
    if model_name == "lattice":
        model = LATTICEModel(
            **common_args,
            k=config.LATTICE_K,
            graph_lambda=config.LATTICE_LAMBDA,
        )
    elif model_name == "micro":
        model = MICROModel(
            **common_args,
            tau=config.MICRO_TAU,
            alpha=config.MICRO_ALPHA,
        )
    elif model_name == "diffmm":
        model = DiffMM(
            **common_args,
            n_steps=config.DIFFMM_STEPS,
            noise_scale=config.DIFFMM_NOISE_SCALE,
            lambda_msi=config.DIFFMM_LAMBDA_MSI,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Log model info
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Created {model_name.upper()} model with {n_params:,} parameters")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate multimodal recommendation models"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lattice", "micro", "diffmm"],
        help="Model to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="electronics",
        help="Dataset name (must be preprocessed)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Base directory for processed data",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (default: Config.EPOCHS)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: Config.BATCH_SIZE)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: Config.LR)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: Config.SEED)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, only evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints",
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Override config if args provided
    config = Config()
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.LR = args.lr
    if args.seed is not None:
        config.SEED = args.seed
    
    # Set seed FIRST (critical for reproducibility)
    set_seed(config.SEED)
    
    logger.info("=" * 60)
    logger.info(f"Multimodal Recommendation: {args.model.upper()}")
    logger.info("=" * 60)
    
    # Device
    device = get_device()
    config.DEVICE = str(device)
    
    # Load dataset
    data_path = Path(args.data_dir) / args.dataset
    logger.info(f"Loading dataset from {data_path}")
    
    dataset = RecDataset(str(data_path), device=config.DEVICE)
    
    # Create model
    model = create_model(args.model, dataset, config)
    
    # Output directory
    output_dir = Path(args.output_dir) / args.dataset / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.eval_only:
        # Load checkpoint and evaluate
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        else:
            checkpoint_path = output_dir / "best.pt"
        
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
        
        # Run three-track evaluation
        results = run_three_track_evaluation(model, dataset, config)
        
        # Save results
        results_path = output_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results to {results_path}")
        
    else:
        # Train
        trainer = Trainer(
            model=model,
            dataset=dataset,
            config=config,
            output_dir=str(output_dir),
        )
        
        history = trainer.train(n_epochs=config.EPOCHS)
        
        # Save training history
        history_path = output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        
        # Run three-track evaluation on best model
        logger.info("\nFinal Evaluation on Best Model:")
        results = run_three_track_evaluation(model, dataset, config)
        
        # Save results
        results_path = output_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results to {results_path}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
