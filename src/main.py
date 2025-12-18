"""
Main entry point for training and evaluating multimodal recommendation models.

Usage:
    python src/main.py --model lattice --dataset electronics
    python src/main.py --model micro --dataset electronics
    python src/main.py --model diffmm --dataset electronics
    
Logs are saved to: logs/training/{dataset}_{model}_{timestamp}/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.common import (
    Config, 
    set_seed, 
    get_device,
    create_run_logger,
    log_system_info,
    log_config,
)
from src.dataset import RecDataset
from src.models import LATTICEModel, MICROModel, DiffMM
from src.trainer import Trainer, run_three_track_evaluation


def create_model(
    model_name: str,
    dataset: RecDataset,
    config: Config,
    logger: logging.Logger,
) -> torch.nn.Module:
    """
    Create model instance.
    
    This is where the magic happens. Or at least where we hope it does.
    
    Args:
        model_name: One of "lattice", "micro", "diffmm".
        dataset: RecDataset instance.
        config: Config object.
        logger: Logger instance.
        
    Returns:
        Model instance ready to disappoint or delight.
    """
    from src.common.config import QuirkyLogger
    
    common_args = dict(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        n_warm=dataset.n_warm,
        embed_dim=config.EMBED_DIM,
        n_layers=config.N_LAYERS,
        feat_visual=dataset.feat_visual,
        feat_text=dataset.feat_text,
        projection_hidden_dim=config.PROJECTION_HIDDEN_DIM,
        projection_dropout=config.PROJECTION_DROPOUT,
        device=config.DEVICE,
    )
    
    logger.info("")
    logger.info("=" * 50)
    logger.info("MODEL CREATION")
    logger.info("=" * 50)
    logger.info(QuirkyLogger.model_init(model_name.upper()))
    
    if model_name == "lattice":
        logger.info("Model: LATTICE (k-NN Graph Learning)")
        logger.info(f"  k_neighbors: {config.LATTICE_K}")
        logger.info(f"  graph_lambda: {config.LATTICE_LAMBDA}")
        model = LATTICEModel(
            **common_args,
            k=config.LATTICE_K,
            graph_lambda=config.LATTICE_LAMBDA,
        )
    elif model_name == "micro":
        logger.info("Model: MICRO (Contrastive Multimodal)")
        logger.info(f"  tau (temperature): {config.MICRO_TAU}")
        logger.info(f"  alpha (contrastive weight): {config.MICRO_ALPHA}")
        model = MICROModel(
            **common_args,
            tau=config.MICRO_TAU,
            alpha=config.MICRO_ALPHA,
        )
    elif model_name == "diffmm":
        logger.info("Model: DiffMM (Diffusion-based)")
        logger.info(f"  n_steps: {config.DIFFMM_STEPS}")
        logger.info(f"  noise_scale: {config.DIFFMM_NOISE_SCALE}")
        logger.info(f"  lambda_msi: {config.DIFFMM_LAMBDA_MSI}")
        logger.info(f"  mlp_width: {config.DIFFMM_MLP_WIDTH}")
        model = DiffMM(
            **common_args,
            n_steps=config.DIFFMM_STEPS,
            noise_scale=config.DIFFMM_NOISE_SCALE,
            lambda_msi=config.DIFFMM_LAMBDA_MSI,
            mlp_width=config.DIFFMM_MLP_WIDTH,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Log model architecture
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("")
    logger.info("Model Architecture:")
    logger.info(f"  Total parameters: {n_params:,}")
    logger.info(f"  Trainable parameters: {n_trainable:,}")
    logger.info(f"  Embed dimension: {config.EMBED_DIM}")
    logger.info(f"  GCN layers: {config.N_LAYERS}")
    logger.info(f"  Projection MLP: {768} → {config.PROJECTION_HIDDEN_DIM} → {config.EMBED_DIM}")
    logger.info(f"  Projection dropout: {config.PROJECTION_DROPOUT}")
    logger.info("")
    
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
    
    # Create timestamped logger
    run_type = "evaluation" if args.eval_only else "training"
    logger, run_dir = create_run_logger(
        run_type=run_type,
        dataset=args.dataset,
        model=args.model,
    )
    
    # Log system info
    log_system_info(logger)
    
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
    
    # Log full config
    logger.info("")
    logger.info("=" * 50)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Epochs: {config.EPOCHS}")
    logger.info(f"  Batch size: {config.BATCH_SIZE}")
    logger.info(f"  Learning rate: {config.LR}")
    logger.info(f"  L2 regularization: {config.L2_REG}")
    logger.info(f"  Patience: {config.PATIENCE}")
    logger.info(f"  Negative samples: {config.N_NEGATIVES}")
    logger.info(f"  Seed: {config.SEED}")
    
    # Set seed FIRST (critical for reproducibility)
    set_seed(config.SEED)
    logger.info(f"  Random seed set: {config.SEED}")
    
    # Device
    device = get_device()
    config.DEVICE = str(device)
    
    # Load dataset
    logger.info("")
    logger.info("=" * 50)
    logger.info("LOADING DATASET")
    logger.info("=" * 50)
    
    data_path = Path(args.data_dir) / args.dataset
    logger.info(f"Data path: {data_path}")
    
    dataset = RecDataset(str(data_path), device=config.DEVICE)
    
    logger.info(f"  Users: {dataset.n_users:,}")
    logger.info(f"  Items (total): {dataset.n_items:,}")
    logger.info(f"  Items (warm): {dataset.n_warm:,}")
    logger.info(f"  Items (cold): {dataset.n_items - dataset.n_warm:,}")
    logger.info(f"  Train interactions: {len(dataset.train_data):,}")
    logger.info(f"  Val interactions: {len(dataset.val_data):,}")
    logger.info(f"  Visual feature dim: {dataset.feat_visual.shape[1]}")
    logger.info(f"  Text feature dim: {dataset.feat_text.shape[1]}")
    
    # Create model
    model = create_model(args.model, dataset, config, logger)
    
    # Output directory
    output_dir = Path(args.output_dir) / args.dataset / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint directory: {output_dir}")
    
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
            if "epoch" in checkpoint:
                logger.info(f"  From epoch: {checkpoint['epoch']}")
            if "best_recall" in checkpoint:
                logger.info(f"  Best Recall@20: {checkpoint['best_recall']:.4f}")
        else:
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
        
        # Run three-track evaluation
        logger.info("")
        logger.info("=" * 50)
        logger.info("THREE-TRACK EVALUATION")
        logger.info("=" * 50)
        
        results = run_three_track_evaluation(model, dataset, config)
        
        # Save results
        results_path = run_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results to {results_path}")
        
    else:
        # Train
        logger.info("")
        logger.info("=" * 50)
        logger.info("TRAINING")
        logger.info("=" * 50)
        
        trainer = Trainer(
            model=model,
            dataset=dataset,
            config=config,
            output_dir=str(output_dir),
        )
        
        history = trainer.train(n_epochs=config.EPOCHS)
        
        # Save training history
        history_path = run_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")
        
        # Run three-track evaluation on best model
        logger.info("")
        logger.info("=" * 50)
        logger.info("FINAL EVALUATION (Best Model)")
        logger.info("=" * 50)
        
        results = run_three_track_evaluation(model, dataset, config)
        
        # Save results
        results_path = run_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved evaluation results to {results_path}")
        
        # Also copy to checkpoint dir
        import shutil
        shutil.copy(run_dir / "run.log", output_dir / "training.log")
        shutil.copy(run_dir / "eval_results.json", output_dir / "eval_results.json")
        shutil.copy(run_dir / "training_history.json", output_dir / "training_history.json")
        logger.info(f"Copied logs to {output_dir}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("RUN COMPLETE")
    logger.info(f"Log directory: {run_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
