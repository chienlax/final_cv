"""
MICRO: Multimodal Item-wise Contrastive Recommendation.

Uses contrastive learning to align modal views with collaborative signals.

Key components:
- Contrastive views: Creates augmented views from visual/text modalities
- InfoNCE loss: Auxiliary contrastive loss for modal alignment
- Inductive mode: Uses fused modal embeddings for cold items
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMultimodalModel


class MICROModel(BaseMultimodalModel):
    """
    MICRO: Multimodal Item-wise Contrastive Recommendation.
    
    Uses the power of contrastive learning to teach embeddings that
    "things that go together, stay together" in vector space.
    It's like matchmaking, but for tensors. 💕
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
        tau: float = 0.2,
        alpha: float = 0.1,
        projection_hidden_dim: int = 1024,
        projection_dropout: float = 0.5,
        device: str = "cuda",
    ):
        """
        Args:
            n_users: Number of users.
            n_items: Total number of items.
            n_warm: Number of warm items.
            embed_dim: Embedding dimension.
            n_layers: Number of LightGCN layers.
            feat_visual: Visual features.
            feat_text: Text features.
            tau: InfoNCE temperature (lower = sharper).
            alpha: Weight for contrastive auxiliary loss.
            projection_hidden_dim: Hidden dim for modality MLP.
            projection_dropout: Dropout for modality MLP.
            device: torch device.
        """
        super().__init__(
            n_users, n_items, n_warm, embed_dim, n_layers,
            feat_visual, feat_text,
            projection_hidden_dim=projection_hidden_dim,
            projection_dropout=projection_dropout,
            device=device,
        )
        
        self.tau = tau
        self.alpha = alpha
        
        # Contrastive projection heads (the beauty of symmetry)
        self.visual_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.text_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.id_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.to(device)
    
    def get_contrastive_views(
        self,
        items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get contrastive views for items.
        
        Args:
            items: (batch,) item indices.
            
        Returns:
            Tuple of (visual_view, text_view, id_view) after projection heads.
        """
        # Project modal features
        visual_emb = self.visual_proj(self.feat_visual[items])
        text_emb = self.text_proj(self.feat_text[items])
        id_emb = self.item_embedding(items)
        
        # Apply contrastive heads
        visual_view = self.visual_head(visual_emb)
        text_view = self.text_head(text_emb)
        id_view = self.id_head(id_emb)
        
        # L2 normalize
        visual_view = F.normalize(visual_view, p=2, dim=1)
        text_view = F.normalize(text_view, p=2, dim=1)
        id_view = F.normalize(id_view, p=2, dim=1)
        
        return visual_view, text_view, id_view
    
    def infonce_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.
        
        Args:
            anchor: (batch, dim) anchor embeddings.
            positive: (batch, dim) positive embeddings.
            negatives: Optional (batch, n_neg, dim) negative embeddings.
                      If None, uses other batch items as negatives.
            
        Returns:
            Scalar InfoNCE loss.
        """
        batch_size = anchor.shape[0]
        
        # Positive similarity
        pos_sim = (anchor * positive).sum(dim=1) / self.tau
        
        if negatives is None:
            # Use other batch items as negatives
            sim_matrix = anchor @ positive.T / self.tau
            
            # Mask out positive pairs (diagonal)
            mask = torch.eye(batch_size, device=anchor.device).bool()
            sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
            
            # Log-softmax trick
            logits = torch.cat([pos_sim.unsqueeze(1), sim_matrix], dim=1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
            
            loss = F.cross_entropy(logits, labels)
        else:
            # Explicit negatives
            neg_sim = (anchor.unsqueeze(1) * negatives).sum(dim=2) / self.tau
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
            loss = F.cross_entropy(logits, labels)
        
        return loss
    
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
        # Get base embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # Fuse modal embeddings
        modal_emb = self.get_modal_embeddings()
        item_emb = item_emb + modal_emb
        
        # LightGCN propagation
        user_emb, item_emb = self.lightgcn_propagate(adj, user_emb, item_emb)
        
        # Lookup batch embeddings
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
        
        Cold items use fused modal embeddings (no ID).
        """
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight.clone()
        
        # Split warm/cold handling
        cold_mask = items >= self.n_warm
        
        # Cold items: pure modal embedding
        if cold_mask.any():
            cold_items = items[cold_mask]
            item_emb[cold_items] = self.get_modal_embeddings(cold_items)
        
        # Warm items: ID + modal
        warm_mask = items < self.n_warm
        if warm_mask.any():
            warm_items = items[warm_mask]
            item_emb[warm_items] = item_emb[warm_items] + self.get_modal_embeddings(warm_items)
        
        # Propagate
        user_emb, item_emb = self.lightgcn_propagate(adj, user_emb, item_emb[:self.n_items])
        
        return user_emb[users], item_emb[items]
    
    def compute_loss(
        self,
        adj: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        l2_reg: float = 1e-4,
    ) -> dict:
        """Compute BPR + InfoNCE + L2 loss."""
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
        adj: torch.Tensor = None, # Unused, for API compatibility with DiffMM
        l2_reg: float = 1e-4,
        cl_loss_precomputed: torch.Tensor = None,  # Unused, for API compatibility
    ) -> dict:
        """
        Compute loss from pre-computed embeddings.
        
        This is separated from forward() to allow AMP to run loss in FP16
        while forward (with sparse ops) runs in FP32.
        """
        # BPR loss
        bpr = self.bpr_loss(user_emb, pos_emb, neg_emb)
        
        # Contrastive loss on positive items
        visual_view, text_view, id_view = self.get_contrastive_views(pos_items)
        
        # Visual-ID alignment
        cl_vi = self.infonce_loss(visual_view, id_view)
        # Text-ID alignment
        cl_ti = self.infonce_loss(text_view, id_view)
        # Visual-Text alignment
        cl_vt = self.infonce_loss(visual_view, text_view)
        
        cl_loss = (cl_vi + cl_ti + cl_vt) / 3
        
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
        
        total = bpr + self.alpha * cl_loss + l2_reg * reg
        
        return {
            "loss": total,
            "bpr_loss": bpr.item(),
            "cl_loss": cl_loss.item(),
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
