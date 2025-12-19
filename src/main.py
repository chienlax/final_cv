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
        logger.info(f"  topk: {config.LATTICE_K}")
        logger.info(f"  lambda_coeff: {config.LATTICE_LAMBDA}")
        logger.info(f"  feat_embed_dim: {config.LATTICE_FEAT_EMBED_DIM}")
        logger.info(f"  n_layers (item graph): {config.LATTICE_N_ITEM_LAYERS}")
        logger.info(f"  n_ui_layers (LightGCN): {config.N_LAYERS}")
        model = LATTICEModel(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            n_warm=dataset.n_warm,
            embed_dim=config.EMBED_DIM,
            n_ui_layers=config.N_LAYERS,  # LightGCN layers
            feat_visual=dataset.feat_visual,
            feat_text=dataset.feat_text,
            topk=config.LATTICE_K,
            lambda_coeff=config.LATTICE_LAMBDA,
            feat_embed_dim=config.LATTICE_FEAT_EMBED_DIM,
            n_layers=config.LATTICE_N_ITEM_LAYERS,  # Item graph layers
            device=config.DEVICE,
        )
    elif model_name == "micro":
        logger.info("Model: MICRO (Contrastive Multimodal)")
        logger.info(f"  tau (temperature): {config.MICRO_TAU}")
        logger.info(f"  loss_ratio: {config.MICRO_LOSS_RATIO}")
        logger.info(f"  topk: {config.MICRO_TOPK}")
        logger.info(f"  lambda_coeff: {config.MICRO_LAMBDA}")
        logger.info(f"  item_layers: {config.MICRO_ITEM_LAYERS}")
        model = MICROModel(
            **common_args,
            topk=config.MICRO_TOPK,
            lambda_coeff=config.MICRO_LAMBDA,
            item_layers=config.MICRO_ITEM_LAYERS,
            tau=config.MICRO_TAU,
            loss_ratio=config.MICRO_LOSS_RATIO,
            sparse=config.MICRO_SPARSE,
            norm_type=config.MICRO_NORM_TYPE,
        )
    elif model_name == "diffmm":
        logger.info("Model: DiffMM (Diffusion-based + Cross-Modal Contrastive)")
        logger.info(f"  steps: {config.DIFFMM_STEPS}")
        logger.info(f"  noise_scale: {config.DIFFMM_NOISE_SCALE}")
        logger.info(f"  dims: {config.DIFFMM_DIMS}")
        logger.info(f"  ssl_reg (contrastive weight): {config.DIFFMM_SSL_REG}")
        logger.info(f"  temp (InfoNCE temperature): {config.DIFFMM_TEMP}")
        logger.info(f"  e_loss (GraphCL weight): {config.DIFFMM_E_LOSS}")
        logger.info(f"  rebuild_k: {config.DIFFMM_REBUILD_K}")
        model = DiffMM(
            **common_args,
            noise_scale=config.DIFFMM_NOISE_SCALE,
            noise_min=config.DIFFMM_NOISE_MIN,
            noise_max=config.DIFFMM_NOISE_MAX,
            steps=config.DIFFMM_STEPS,
            dims=config.DIFFMM_DIMS,
            d_emb_size=config.DIFFMM_D_EMB_SIZE,
            sampling_steps=config.DIFFMM_SAMPLING_STEPS,
            sampling_noise=config.DIFFMM_SAMPLING_NOISE,
            rebuild_k=config.DIFFMM_REBUILD_K,
            e_loss=config.DIFFMM_E_LOSS,
            ssl_reg=config.DIFFMM_SSL_REG,
            temp=config.DIFFMM_TEMP,
            keep_rate=config.DIFFMM_KEEP_RATE,
            ris_lambda=config.DIFFMM_RIS_LAMBDA,
            ris_adj_lambda=config.DIFFMM_RIS_ADJ_LAMBDA,
            trans=config.DIFFMM_TRANS,
            cl_method=config.DIFFMM_CL_METHOD,
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
    
    # === ABLATION ARGUMENTS ===
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=["none", "no_visual", "no_text"],
        help="Ablation mode: 'no_visual' zeros visual features, 'no_text' zeros text features",
    )
    
    # === SENSITIVITY ANALYSIS OVERRIDES ===
    parser.add_argument(
        "--lattice-k",
        type=int,
        default=None,
        help="Override LATTICE_K for sensitivity analysis",
    )
    parser.add_argument(
        "--micro-alpha",
        type=float,
        default=None,
        help="Override MICRO_ALPHA for sensitivity analysis",
    )
    parser.add_argument(
        "--diffmm-steps",
        type=int,
        default=None,
        help="Override DIFFMM_STEPS for sensitivity analysis",
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
    
    # Apply sensitivity analysis overrides
    if args.lattice_k is not None:
        config.LATTICE_K = args.lattice_k
    if args.micro_alpha is not None:
        config.MICRO_ALPHA = args.micro_alpha
    if args.diffmm_steps is not None:
        config.DIFFMM_STEPS = args.diffmm_steps
    
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
    if args.ablation != "none":
        logger.info(f"  ⚠️  ABLATION MODE: {args.ablation}")
    
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
    
    dataset = RecDataset(
        str(data_path), 
        device=config.DEVICE,
        ablation_mode=args.ablation,
    )
    
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
    
    # Output directory - SMART ROUTING to prevent overwrites
    # Ablation runs go to checkpoints_ablation/
    # Sensitivity runs go to checkpoints_sensitivity/
    base_output_dir = args.output_dir
    
    is_ablation = args.ablation != "none"
    is_sensitivity = any([
        args.lattice_k is not None,
        args.micro_alpha is not None,
        args.diffmm_steps is not None,
    ])
    
    if is_ablation:
        # Route to ablation directory with modality suffix
        base_output_dir = f"{args.output_dir}_ablation"
        output_dir = Path(base_output_dir) / args.dataset / f"{args.model}_{args.ablation}"
        logger.info(f"  🔀 Ablation run detected - routing to: {output_dir}")
    elif is_sensitivity:
        # Route to sensitivity directory with param suffix
        base_output_dir = f"{args.output_dir}_sensitivity"
        param_suffix = []
        if args.lattice_k is not None:
            param_suffix.append(f"k{args.lattice_k}")
        if args.micro_alpha is not None:
            param_suffix.append(f"alpha{args.micro_alpha}")
        if args.diffmm_steps is not None:
            param_suffix.append(f"steps{args.diffmm_steps}")
        suffix = "_".join(param_suffix)
        output_dir = Path(base_output_dir) / args.dataset / f"{args.model}_{suffix}"
        logger.info(f"  🔀 Sensitivity run detected - routing to: {output_dir}")
    else:
        # Normal run - use default path
        output_dir = Path(base_output_dir) / args.dataset / args.model
    
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
