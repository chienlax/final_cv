"""
Training loop for multimodal recommendation models.

Supports early stopping, checkpoint saving, and multi-loss logging.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common.utils import AverageMeter, EarlyStopping
from .evaluator import evaluate, format_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for multimodal recommendation models."""
    
    def __init__(
        self,
        model,
        dataset,
        config,
        output_dir: str = "checkpoints",
    ):
        """
        Args:
            model: Model to train.
            dataset: RecDataset instance.
            config: Config object with hyperparameters.
            output_dir: Directory for saving checkpoints.
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = Adam(
            model.parameters(),
            lr=config.LR,
            weight_decay=0,  # We use explicit L2 reg in loss
        )
        
        self.early_stopping = EarlyStopping(
            patience=config.PATIENCE,
            mode="max",
        )
        
        self.best_metric = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, dataloader: DataLoader) -> dict:
        """
        Train for one epoch.
        
        Args:
            dataloader: BPR training dataloader.
            
        Returns:
            Dict of average losses.
        """
        self.model.train()
        
        meters = {}
        adj = self.dataset.norm_adj
        
        pbar = tqdm(dataloader, desc="Training", leave=False)
        
        for users, pos_items, neg_items in pbar:
            users = users.to(self.model.device)
            pos_items = pos_items.to(self.model.device)
            neg_items = neg_items.to(self.model.device)
            
            self.optimizer.zero_grad()
            
            losses = self.model.compute_loss(
                adj, users, pos_items, neg_items,
                l2_reg=self.config.L2_REG,
            )
            
            loss = losses["loss"]
            loss.backward()
            
            self.optimizer.step()
            
            # Update meters
            for key, value in losses.items():
                if key not in meters:
                    meters[key] = AverageMeter()
                if isinstance(value, torch.Tensor):
                    value = value.item()
                meters[key].update(value)
            
            pbar.set_postfix({k: f"{v.avg:.4f}" for k, v in meters.items()})
        
        return {k: v.avg for k, v in meters.items()}
    
    def train(
        self,
        n_epochs: int = None,
        eval_every: int = 1,
    ) -> dict:
        """
        Full training loop.
        
        Args:
            n_epochs: Number of epochs (default: config.EPOCHS).
            eval_every: Evaluate every N epochs.
            
        Returns:
            Dict with training history.
        """
        from .dataset import create_bpr_dataloader
        
        n_epochs = n_epochs or self.config.EPOCHS
        
        dataloader = create_bpr_dataloader(
            self.dataset,
            batch_size=self.config.BATCH_SIZE,
            n_negatives=self.config.N_NEGATIVES,
        )
        
        history = {
            "train_loss": [],
            "val_recall": [],
            "val_ndcg": [],
        }
        
        logger.info(f"Starting training for {n_epochs} epochs...")
        logger.info(f"  Batch size: {self.config.BATCH_SIZE}")
        logger.info(f"  Learning rate: {self.config.LR}")
        logger.info(f"  L2 reg: {self.config.L2_REG}")
        
        start_time = time.time()
        
        for epoch in range(1, n_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_losses = self.train_epoch(dataloader)
            history["train_loss"].append(train_losses["loss"])
            
            epoch_time = time.time() - epoch_start
            
            logger.info(
                f"Epoch {epoch}/{n_epochs} | "
                f"Loss: {train_losses['loss']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Evaluate
            if epoch % eval_every == 0:
                val_metrics = evaluate(
                    self.model,
                    self.dataset,
                    split="val",
                    k_list=self.config.TOP_K,
                    batch_size=self.config.EVAL_BATCH_SIZE,
                )
                
                recall = val_metrics.get(f"recall@{self.config.TOP_K[1]}", 0)
                ndcg = val_metrics.get(f"ndcg@{self.config.TOP_K[1]}", 0)
                
                history["val_recall"].append(recall)
                history["val_ndcg"].append(ndcg)
                
                logger.info(
                    f"  Val Recall@{self.config.TOP_K[1]}: {recall:.4f} | "
                    f"NDCG@{self.config.TOP_K[1]}: {ndcg:.4f}"
                )
                
                # Save best model
                if recall > self.best_metric:
                    self.best_metric = recall
                    self.best_epoch = epoch
                    self._save_checkpoint("best.pt")
                    logger.info(f"  ★ New best model saved!")
                
                # Early stopping
                if self.early_stopping(recall):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time/60:.1f} minutes")
        logger.info(f"Best epoch: {self.best_epoch} (Recall@{self.config.TOP_K[1]}: {self.best_metric:.4f})")
        
        # Load best model
        self._load_checkpoint("best.pt")
        
        return history
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
        }, path)
    
    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.output_dir / filename
        if path.exists():
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from {path}")


def run_three_track_evaluation(
    model,
    dataset,
    config,
) -> dict:
    """
    Run the three-track evaluation protocol.
    
    Track 1: Warm performance (test_warm.txt)
    Track 2: User robustness (sparse vs active users)
    Track 3: Cold-start (test_cold.txt with inductive mode)
    
    Args:
        model: Trained model.
        dataset: RecDataset instance.
        config: Config object.
        
    Returns:
        Dict with all track results.
    """
    results = {}
    k_list = config.TOP_K
    
    logger.info("=" * 60)
    logger.info("THREE-TRACK EVALUATION")
    logger.info("=" * 60)
    
    # Track 1: Warm Performance
    logger.info("\n--- Track 1: Warm Performance ---")
    track1 = evaluate(model, dataset, split="test_warm", k_list=k_list)
    results["track1_warm"] = track1
    logger.info(format_metrics(track1, "Test Warm"))
    
    # Track 2: User Robustness
    logger.info("\n--- Track 2: User Robustness ---")
    
    sparse_users = dataset.get_sparse_users(max_degree=5)
    active_users = dataset.get_active_users(min_degree=20)
    
    if sparse_users:
        track2_sparse = evaluate(
            model, dataset, split="test_warm",
            k_list=k_list, filter_users=sparse_users
        )
        results["track2_sparse"] = track2_sparse
        logger.info(format_metrics(track2_sparse, f"Sparse Users (n={len(sparse_users)})"))
    
    if active_users:
        track2_active = evaluate(
            model, dataset, split="test_warm",
            k_list=k_list, filter_users=active_users
        )
        results["track2_active"] = track2_active
        logger.info(format_metrics(track2_active, f"Active Users (n={len(active_users)})"))
    
    # Track 3: Cold-Start
    logger.info("\n--- Track 3: Cold-Start (Inductive Mode) ---")
    track3 = evaluate(
        model, dataset, split="test_cold",
        k_list=k_list, inductive=True
    )
    results["track3_cold"] = track3
    logger.info(format_metrics(track3, "Test Cold"))
    
    logger.info("=" * 60)
    
    return results
