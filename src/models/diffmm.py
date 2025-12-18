"""
DiffMM: Diffusion Model for Multimodal Recommendation.

Uses a diffusion process to generate user-item interactions conditioned
on multimodal features.

Key components:
- Diffusion process: Forward (add noise) and reverse (denoise)
- Modality Signal Injection (MSI): Conditions generation on visual features
- Generative forward: For cold items, generates interaction signal from noise
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMultimodalModel


class DiffMM(BaseMultimodalModel):
    """
    DiffMM: Diffusion-based Multimodal Recommendation.
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_warm: int,
        embed_dim: int,
        n_layers: int,
        feat_visual: torch.Tensor,
        feat_text: torch.Tensor,
        n_steps: int = 5,
        noise_scale: float = 0.1,
        lambda_msi: float = 1e-2,
        device: str = "cuda",
    ):
        """
        Args:
            n_users: Number of users.
            n_items: Total number of items.
            n_warm: Number of warm items.
            embed_dim: Embedding dimension.
            n_layers: Number of GCN layers.
            feat_visual: Visual features.
            feat_text: Text features.
            n_steps: Number of diffusion steps.
            noise_scale: Base noise scale.
            lambda_msi: Weight for MSI loss.
            device: torch device.
        """
        super().__init__(
            n_users, n_items, n_warm, embed_dim, n_layers,
            feat_visual, feat_text, device
        )
        
        self.n_steps = n_steps
        self.noise_scale = noise_scale
        self.lambda_msi = lambda_msi
        
        # Noise schedule (linear)
        self.register_buffer(
            "betas",
            torch.linspace(1e-4, 0.02, n_steps)
        )
        self.register_buffer(
            "alphas",
            1 - self.betas
        )
        self.register_buffer(
            "alphas_cumprod",
            torch.cumprod(self.alphas, dim=0)
        )
        
        # Denoising network
        self.denoise_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
        # Modality Signal Injection (MSI) network
        self.msi_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Time embedding
        self.time_embed = nn.Embedding(n_steps, embed_dim)
        
        self.to(device)
    
    def diffusion_forward(
        self,
        x: torch.Tensor,
        t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: add noise at step t.
        
        Args:
            x: (batch, dim) clean embeddings.
            t: Diffusion step.
            
        Returns:
            Tuple of (noisy_x, noise).
        """
        alpha_t = self.alphas_cumprod[t]
        
        noise = torch.randn_like(x) * self.noise_scale
        noisy_x = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        
        return noisy_x, noise
    
    def diffusion_reverse(
        self,
        noisy_x: torch.Tensor,
        t: int,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reverse diffusion: denoise at step t.
        
        Args:
            noisy_x: (batch, dim) noisy embeddings.
            t: Diffusion step.
            condition: (batch, dim) conditioning signal (modal features).
            
        Returns:
            Denoised embeddings.
        """
        # Time embedding
        t_emb = self.time_embed(torch.tensor([t], device=noisy_x.device))
        t_emb = t_emb.expand(noisy_x.shape[0], -1)
        
        # Condition with modal features
        conditioned = noisy_x + self.msi_net(condition)
        
        # Denoise
        input_feat = torch.cat([conditioned, t_emb], dim=1)
        pred_noise = self.denoise_net(input_feat)
        
        # Compute denoised x
        alpha_t = self.alphas_cumprod[t]
        denoised = (noisy_x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        
        return denoised
    
    def sample_from_noise(
        self,
        condition: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample embeddings from pure noise (for cold items).
        
        Args:
            condition: (batch, dim) modal features for conditioning.
            n_steps: Number of reverse steps (default: self.n_steps).
            
        Returns:
            Generated embeddings.
        """
        n_steps = n_steps or self.n_steps
        
        # Start from noise
        x = torch.randn(condition.shape[0], self.embed_dim, device=self.device)
        x = x * self.noise_scale
        
        # Reverse diffusion
        for t in reversed(range(n_steps)):
            x = self.diffusion_reverse(x, t, condition)
        
        return x
    
    def lightgcn_propagate(
        self,
        adj: torch.Tensor,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LightGCN message passing."""
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        embs = [all_emb]
        
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            embs.append(all_emb)
        
        all_emb = torch.stack(embs, dim=0).mean(dim=0)
        
        user_emb = all_emb[:self.n_users]
        item_emb = all_emb[self.n_users:]
        
        return user_emb, item_emb
    
    def forward(
        self,
        adj: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for BPR training."""
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # Fuse modal embeddings
        modal_emb = self.get_modal_embeddings()
        item_emb = item_emb + modal_emb
        
        # LightGCN propagation
        user_emb, item_emb = self.lightgcn_propagate(adj, user_emb, item_emb)
        
        # Lookup batch
        user_batch = user_emb[users]
        pos_batch = item_emb[pos_items]
        neg_batch = item_emb[neg_items]
        
        return user_batch, pos_batch, neg_batch
    
    def inductive_forward(
        self,
        adj: torch.Tensor,
        users: torch.Tensor,
        items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inductive forward for cold items.
        
        Cold items: Generate embedding from noise conditioned on modal features.
        """
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight.clone()
        
        cold_mask = items >= self.n_warm
        
        # Cold items: Generate from noise + modal conditioning
        if cold_mask.any():
            cold_items = items[cold_mask]
            modal_cond = self.get_modal_embeddings(cold_items)
            generated_emb = self.sample_from_noise(modal_cond)
            item_emb[cold_items] = generated_emb
        
        # Warm items: ID + modal
        warm_mask = items < self.n_warm
        if warm_mask.any():
            warm_items = items[warm_mask]
            item_emb[warm_items] = item_emb[warm_items] + self.get_modal_embeddings(warm_items)
        
        # Propagate
        user_emb, item_emb = self.lightgcn_propagate(adj, user_emb, item_emb[:self.n_items])
        
        return user_emb[users], item_emb[items]
    
    def generative_forward(
        self,
        adj: torch.Tensor,
        users: torch.Tensor,
        items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generative forward (alias for inductive_forward).
        
        This is the DiffMM-specific method for cold-start evaluation.
        """
        return self.inductive_forward(adj, users, items)
    
    def compute_diffusion_loss(
        self,
        item_emb: torch.Tensor,
        modal_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute diffusion denoising loss.
        
        Args:
            item_emb: (batch, dim) target item embeddings.
            modal_emb: (batch, dim) modal conditioning.
            
        Returns:
            MSE loss for noise prediction.
        """
        batch_size = item_emb.shape[0]
        
        # Random timestep
        t = torch.randint(0, self.n_steps, (1,)).item()
        
        # Forward diffusion
        noisy_emb, noise = self.diffusion_forward(item_emb, t)
        
        # Predict noise
        t_emb = self.time_embed(torch.tensor([t], device=self.device))
        t_emb = t_emb.expand(batch_size, -1)
        
        conditioned = noisy_emb + self.msi_net(modal_emb)
        input_feat = torch.cat([conditioned, t_emb], dim=1)
        pred_noise = self.denoise_net(input_feat)
        
        # MSE loss
        loss = F.mse_loss(pred_noise, noise)
        
        return loss
    
    def compute_loss(
        self,
        adj: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        l2_reg: float = 1e-4,
    ) -> dict:
        """Compute BPR + Diffusion + L2 loss."""
        user_emb, pos_emb, neg_emb = self.forward(adj, users, pos_items, neg_items)
        
        return self._compute_loss_from_emb(
            user_emb, pos_emb, neg_emb,
            users, pos_items, neg_items,
            l2_reg=l2_reg,
        )
    
    def _compute_loss_from_emb(
        self,
        user_emb: torch.Tensor,   # (batch, dim)
        pos_emb: torch.Tensor,    # (batch, dim)
        neg_emb: torch.Tensor,    # (batch, dim) or (batch, n_neg, dim)
        users: torch.Tensor,      # (batch,)
        pos_items: torch.Tensor,  # (batch,)
        neg_items: torch.Tensor,  # (batch,) or (batch, n_neg)
        l2_reg: float = 1e-4,
    ) -> dict:
        """
        Compute loss from pre-computed embeddings.
        
        This is separated from forward() to allow AMP to run loss in FP16
        while forward (with sparse ops) runs in FP32.
        """
        # BPR loss
        bpr = self.bpr_loss(user_emb, pos_emb, neg_emb)
        
        # Diffusion loss (MSI)
        pos_item_emb = self.item_embedding(pos_items)
        modal_emb = self.get_modal_embeddings(pos_items)
        diff_loss = self.compute_diffusion_loss(pos_item_emb, modal_emb)
        
        # L2 regularization - handle multi-negative
        if neg_items.dim() == 2:
            neg_items_flat = neg_items.flatten()
        else:
            neg_items_flat = neg_items
        
        reg = self.l2_reg_loss(
            self.user_embedding.weight[users],
            self.item_embedding.weight[pos_items],
            self.item_embedding.weight[neg_items_flat],
        )
        
        total = bpr + self.lambda_msi * diff_loss + l2_reg * reg
        
        return {
            "loss": total,
            "bpr_loss": bpr.item(),
            "diff_loss": diff_loss.item(),
            "reg_loss": reg.item(),
        }
    
    def _get_all_embeddings(
        self,
        adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get all embeddings after propagation."""
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        modal_emb = self.get_modal_embeddings()
        item_emb = item_emb + modal_emb
        
        user_emb, item_emb = self.lightgcn_propagate(adj, user_emb, item_emb)
        
        return user_emb, item_emb
