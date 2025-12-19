"""
Training loop for multimodal recommendation models.

Supports:
- Early stopping with patience
- Checkpoint saving/loading  
- Mixed precision training (AMP)
- LR scheduling (cosine annealing)
- Multi-loss logging
- Existential commentary on the nature of gradient descent 🎭
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common.config import QuirkyLogger
from .common.utils import AverageMeter, EarlyStopping
from .evaluator import evaluate, format_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for multimodal recommendation models.
    
    Features:
    - Mixed precision training (AMP) for VRAM savings and speedup
    - Cosine annealing LR scheduler for better convergence
    - Early stopping with configurable patience
    - Checkpoint saving with best model tracking
    - Questionable life advice via log messages
    
    "The training loop is a metaphor for life: iterate, mess up, adjust, repeat."
    """
    
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
        
        # Check if this is DiffMM for special handling
        self.is_diffmm = model.__class__.__name__ == "DiffMM"
        
        # Optimizer - DiffMM uses separate optimizers for denoisers and main model
        if self.is_diffmm:
            # Main optimizer (excludes denoisers)
            self.optimizer = Adam(
                model.get_main_parameters(),
                lr=config.LR,
                weight_decay=0,
            )
            # Denoiser optimizer - EXACT from Main.py line 76-81
            self.denoise_optimizer = Adam(
                model.get_denoiser_parameters(),
                lr=config.LR,
                weight_decay=0,
            )
            logger.info("DiffMM: Using separate optimizers for main model and denoisers")
        else:
            self.optimizer = Adam(
                model.parameters(),
                lr=config.LR,
                weight_decay=0,  # We use explicit L2 reg in loss
            )
            self.denoise_optimizer = None
        
        # LR Scheduler
        lr_scheduler_type = getattr(config, 'LR_SCHEDULER', None)
        if lr_scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=config.EPOCHS,
                eta_min=config.LR * 0.01,  # Minimum LR = 1% of initial
            )
            logger.info(f"Using Cosine Annealing LR scheduler (T_max={config.EPOCHS})")
        else:
            self.scheduler = None
        
        # Mixed Precision (AMP)
        self.use_amp = getattr(config, 'USE_AMP', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Using Automatic Mixed Precision (AMP)")
        else:
            self.scaler = None
        
        # Early stopping
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
            
        Batch shapes:
            users: (batch_size,) int64
            pos_items: (batch_size,) int64
            neg_items: (batch_size,) or (batch_size, n_neg) int64
            
        Note:
            Sparse matrix operations don't support FP16, so we run the forward
            pass (which contains LightGCN propagation) in FP32 and only use
            AMP for the loss computation and backward pass.
        """
        self.model.train()
        
        meters = {}
        adj = self.dataset.norm_adj
        
        # Track first batch for build_item_graph flag
        first_batch = True
        
        pbar = tqdm(dataloader, desc="Training", leave=False)
        
        for users, pos_items, neg_items in pbar:
            # Move to device - shapes: (batch,), (batch,), (batch,) or (batch, n_neg)
            users = users.to(self.model.device)
            pos_items = pos_items.to(self.model.device)
            neg_items = neg_items.to(self.model.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass in FP32 (sparse mm doesn't support FP16)
            # Then use AMP only for loss computation and backward
            if self.use_amp:
                # Determine if we need to rebuild item graphs (LATTICE/MICRO)
                # Build on first batch of epoch only
                build_graph = first_batch and hasattr(self.model, 'item_adj')
                
                # Check model type to determine forward signature
                model_type = self.model.__class__.__name__
                
                if model_type == "LATTICEModel":
                    # LATTICE: forward(adj, build_item_graph) -> (all_u, all_i)
                    all_u, all_i = self.model.forward(adj, build_item_graph=build_graph)
                    user_emb = all_u[users]
                    pos_emb = all_i[pos_items]
                    neg_emb = all_i[neg_items]
                    extra_kwargs = {}
                    
                elif model_type == "MICROModel":
                    # MICRO: forward(adj, build_item_graph) -> (all_u, all_i, image_emb, text_emb, fusion_emb)
                    all_u, all_i, image_emb, text_emb, fusion_emb = self.model.forward(adj, build_item_graph=build_graph)
                    user_emb = all_u[users]
                    pos_emb = all_i[pos_items]
                    neg_emb = all_i[neg_items]
                    extra_kwargs = {'image_emb': image_emb, 'text_emb': text_emb, 'fusion_emb': fusion_emb}
                    
                else:
                    # DiffMM and others: forward(adj, users, pos, neg) -> (u, pos, neg)
                    user_emb, pos_emb, neg_emb = self.model.forward(adj, users, pos_items, neg_items)
                    extra_kwargs = {}
                
                # For DiffMM: Compute contrastive loss BEFORE autocast
                # (it uses sparse ops internally via forward_visual/text_view)
                cl_loss_precomputed = None
                if hasattr(self.model, 'compute_contrastive_loss'):
                    cl_loss_precomputed = self.model.compute_contrastive_loss(adj, users, pos_items)
                
                # Loss computation WITH autocast (but cl_loss already computed in FP32)
                with torch.amp.autocast('cuda'):
                    losses = self.model._compute_loss_from_emb(
                        user_emb, pos_emb, neg_emb,
                        users, pos_items, neg_items,
                        adj=None,  # Don't pass adj - we precomputed cl_loss
                        l2_reg=self.config.L2_REG,
                        cl_loss_precomputed=cl_loss_precomputed,  # Pass precomputed CL loss
                        **extra_kwargs,
                    )
                
                # NaN/Inf safety check - critical for long training runs
                if torch.isnan(losses["loss"]) or torch.isinf(losses["loss"]):
                    logger.warning(f"⚠️ NaN/Inf detected in loss! Skipping this batch.")
                    first_batch = False  # Don't rebuild on next batch either
                    continue
                
                # Backward with gradient scaling
                self.scaler.scale(losses["loss"]).backward()
                
                # Gradient clipping for stability (before unscaling)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Determine if we need to rebuild item graphs (LATTICE/MICRO)
                build_graph = first_batch and hasattr(self.model, 'item_adj')
                
                # Check if model supports build_item_graph parameter
                if hasattr(self.model, 'item_adj'):
                    losses = self.model.compute_loss(
                        adj, users, pos_items, neg_items,
                        l2_reg=self.config.L2_REG,
                        build_item_graph=build_graph,
                    )
                else:
                    losses = self.model.compute_loss(
                        adj, users, pos_items, neg_items,
                        l2_reg=self.config.L2_REG,
                    )
                
                # NaN/Inf safety check
                if torch.isnan(losses["loss"]) or torch.isinf(losses["loss"]):
                    logger.warning(f"⚠️ NaN/Inf detected in loss! Skipping this batch.")
                    first_batch = False
                    continue
                
                losses["loss"].backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
            
            # Update meters
            for key, value in losses.items():
                if key not in meters:
                    meters[key] = AverageMeter()
                if isinstance(value, torch.Tensor):
                    value = value.item()
                meters[key].update(value)
            
            pbar.set_postfix({k: f"{v.avg:.4f}" for k, v in meters.items()})
            
            # Only first batch
            first_batch = False
        
        return {k: v.avg for k, v in meters.items()}
    
    def train_epoch_diffmm(self, bpr_dataloader: DataLoader, diffusion_data: torch.Tensor) -> dict:
        """
        Train DiffMM for one epoch with two-phase training.
        
        EXACT match to original Main.py trainEpoch() structure:
        1. Phase 1: Train denoisers on diffusion task
        2. Phase 2: Rebuild UI matrices using denoised samples
        3. Phase 3: Train main model with BPR + CL
        
        Args:
            bpr_dataloader: Standard BPR training dataloader
            diffusion_data: Training matrix as dense tensor [n_users, n_items]
            
        Returns:
            Dict of average losses
        """
        self.model.train()
        
        meters = {}
        adj = self.dataset.norm_adj
        batch_size = self.config.BATCH_SIZE
        n_users = diffusion_data.shape[0]
        
        # =================================================================
        # PHASE 1: Diffusion Training - EXACT from Main.py line 124-168
        # =================================================================
        logger.info("  Phase 1: Diffusion training...")
        
        diff_loss_image_total = 0.0
        diff_loss_text_total = 0.0
        n_diff_batches = 0
        
        for start_idx in range(0, n_users, batch_size):
            end_idx = min(start_idx + batch_size, n_users)
            batch_item = diffusion_data[start_idx:end_idx].to(self.model.device)
            batch_index = torch.arange(start_idx, end_idx).to(self.model.device)
            
            self.denoise_optimizer.zero_grad()
            
            loss_image, loss_text = self.model.train_diffusion_step(batch_item, batch_index)
            
            total_diff_loss = loss_image + loss_text
            total_diff_loss.backward()
            
            self.denoise_optimizer.step()
            
            diff_loss_image_total += loss_image.item()
            diff_loss_text_total += loss_text.item()
            n_diff_batches += 1
        
        meters["diff_loss_image"] = AverageMeter()
        meters["diff_loss_text"] = AverageMeter()
        meters["diff_loss_image"].update(diff_loss_image_total / max(n_diff_batches, 1))
        meters["diff_loss_text"].update(diff_loss_text_total / max(n_diff_batches, 1))
        
        # =================================================================
        # PHASE 2: Rebuild UI Matrices - EXACT from Main.py line 173-245
        # =================================================================
        logger.info("  Phase 2: Rebuilding UI matrices...")
        
        self.model.rebuild_ui_matrices(diffusion_data, batch_size=batch_size)
        
        logger.info("  Phase 3: BPR + CL training...")
        
        # =================================================================
        # PHASE 3: BPR + CL Training - EXACT from Main.py line 247-305
        # =================================================================
        pbar = tqdm(bpr_dataloader, desc="BPR+CL Training", leave=False)
        
        for users, pos_items, neg_items in pbar:
            users = users.to(self.model.device)
            pos_items = pos_items.to(self.model.device)
            neg_items = neg_items.to(self.model.device)
            
            self.optimizer.zero_grad()
            
            losses = self.model.compute_loss(
                adj, users, pos_items, neg_items,
                l2_reg=self.config.L2_REG,
            )
            
            if torch.isnan(losses["loss"]) or torch.isinf(losses["loss"]):
                logger.warning("⚠️ NaN/Inf detected in loss! Skipping this batch.")
                continue
            
            losses["loss"].backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
        
        # Create DataLoader with config params
        num_workers = getattr(self.config, 'NUM_WORKERS', 0)
        pin_memory = getattr(self.config, 'PIN_MEMORY', True)
        
        dataloader = create_bpr_dataloader(
            self.dataset,
            batch_size=self.config.BATCH_SIZE,
            n_negatives=self.config.N_NEGATIVES,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        
        # DiffMM needs diffusion_data: dense training matrix [n_users, n_items]
        # EXACT match to original DataHandler.py line 81-82: DiffusionData(torch.FloatTensor(self.trnMat.A))
        diffusion_data = None
        if self.is_diffmm:
            logger.info("DiffMM: Building diffusion training matrix (dense user-item matrix)...")
            from scipy import sparse
            n_users = self.dataset.n_users
            n_items = self.dataset.n_items
            
            # Build dense matrix from train_data
            row = self.dataset.train_data[:, 0]
            col = self.dataset.train_data[:, 1]
            data = np.ones(len(row), dtype=np.float32)
            trnMat = sparse.csr_matrix((data, (row, col)), shape=(n_users, n_items))
            diffusion_data = torch.FloatTensor(trnMat.toarray())
            logger.info(f"  Diffusion data shape: {diffusion_data.shape}")
        
        history = {
            "train_loss": [],
            "val_recall": [],
            "val_ndcg": [],
            "learning_rate": [],
        }
        
        logger.info(QuirkyLogger.train_start())
        logger.info(f"Starting training for {n_epochs} epochs...")
        logger.info(f"  Batch size: {self.config.BATCH_SIZE}")
        logger.info(f"  Learning rate: {self.config.LR}")
        logger.info(f"  L2 reg: {self.config.L2_REG}")
        logger.info(f"  N negatives: {self.config.N_NEGATIVES}")
        logger.info(f"  AMP enabled: {self.use_amp}")
        logger.info(f"  DataLoader workers: {num_workers}")
        if self.is_diffmm:
            logger.info(f"  DiffMM two-phase training: ENABLED")
        
        prev_loss = None  # Track loss for quirky message selection
        
        start_time = time.time()
        
        for epoch in range(1, n_epochs + 1):
            epoch_start = time.time()
            
            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            history["learning_rate"].append(current_lr)
            
            # Train - DiffMM uses specialized two-phase training
            if self.is_diffmm:
                train_losses = self.train_epoch_diffmm(dataloader, diffusion_data)
            else:
                train_losses = self.train_epoch(dataloader)
            history["train_loss"].append(train_losses["loss"])
            
            # Step LR scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Epoch summary with quirky flair
            loss_msg = QuirkyLogger.format_loss(train_losses['loss'], prev_loss)
            logger.info(QuirkyLogger.epoch_start(epoch))
            logger.info(
                f"  {loss_msg} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
            prev_loss = train_losses['loss']
            
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
                    logger.info(QuirkyLogger.early_stop())
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        total_time = time.time() - start_time
        logger.info(QuirkyLogger.training_done())
        logger.info(f"Training complete in {total_time/60:.1f} minutes")
        logger.info(f"Best epoch: {self.best_epoch} (Recall@{self.config.TOP_K[1]}: {self.best_metric:.4f})")
        
        # Load best model
        self._load_checkpoint("best.pt")
        
        return history
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
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
    logger.info(QuirkyLogger.cold_start())
    track3 = evaluate(
        model, dataset, split="test_cold",
        k_list=k_list, inductive=True
    )
    results["track3_cold"] = track3
    logger.info(format_metrics(track3, "Test Cold"))
    
    logger.info("=" * 60)
    
    return results
