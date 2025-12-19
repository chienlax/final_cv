"""
LATTICE: Mining Latent Structures for Multimodal Recommendation.

Faithful implementation matching the official LATTICE paper (MM'21).
Reference: https://github.com/CRIPAC-DIG/LATTICE

Key Architecture:
    1. Item-item graph learning: k-NN from modal features
    2. Dynamic graph rebuilding: `build_item_graph` flag per epoch
    3. LightGCN backbone for user-item propagation
    4. Modal fusion with learnable weights

Training:
    - First batch of each epoch: build_item_graph=True
    - Subsequent batches: build_item_graph=False (use cached)
"""

from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMultimodalModel


# =============================================================================
# Utility Functions (from original codes/Models.py)
# =============================================================================

def build_sim(context: torch.Tensor) -> torch.Tensor:
    """
    Build cosine similarity matrix.
    
    Args:
        context: (N, dim) feature matrix
        
    Returns:
        (N, N) similarity matrix
    """
    context_norm = context / (context.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    sim = torch.mm(context_norm, context_norm.T)
    return sim


def build_knn_neighbourhood(adj: torch.Tensor, topk: int) -> torch.Tensor:
    """
    Build k-NN adjacency by keeping top-k neighbors per node.
    
    Args:
        adj: (N, N) similarity matrix
        topk: Number of neighbors to keep
        
    Returns:
        (N, N) sparse k-NN adjacency
    """
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adj = torch.zeros_like(adj).scatter_(-1, knn_ind, knn_val)
    return weighted_adj


def compute_normalized_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """
    Compute symmetric normalized Laplacian: D^(-1/2) A D^(-1/2).
    
    Args:
        adj: (N, N) adjacency matrix
        
    Returns:
        (N, N) normalized adjacency
    """
    rowsum = torch.sum(adj, dim=-1)
    d_inv_sqrt = torch.pow(rowsum + 1e-8, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


# =============================================================================
# LATTICE Model
# =============================================================================

class LATTICEModel(BaseMultimodalModel):
    """
    LATTICE: Learning All-modality Structured Item Graph for Recommendation.
    
    Full implementation matching the official LATTICE paper.
    
    Architecture:
        1. Modal feature projection (linear)
        2. k-NN item graph from projected features
        3. Item graph message passing (n_layers hops)
        4. LightGCN on user-item bipartite graph
        5. Learnable modal fusion weights
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
        # LATTICE-specific params
        topk: int = 10,
        lambda_coeff: float = 0.9,
        feat_embed_dim: int = 64,
        n_item_layers: int = 1,
        # Base params
        projection_hidden_dim: int = 1024,
        projection_dropout: float = 0.5,
        device: str = "cuda",
    ):
        """
        Args:
            n_users: Number of users
            n_items: Number of items
            n_warm: Number of warm items
            embed_dim: Embedding dimension
            n_layers: Number of LightGCN layers
            feat_visual: (n_items, visual_dim) visual features
            feat_text: (n_items, text_dim) text features
            topk: k for k-NN graph
            lambda_coeff: Weight for original vs learned graph (higher = more original)
            feat_embed_dim: Modal feature projection dimension
            n_item_layers: Number of item graph conv layers
            projection_hidden_dim: Base projection hidden dim
            projection_dropout: Base projection dropout
            device: Torch device
        """
        super().__init__(
            n_users, n_items, n_warm, embed_dim, n_layers,
            feat_visual, feat_text,
            projection_hidden_dim=projection_hidden_dim,
            projection_dropout=projection_dropout,
            device=device,
        )
        
        self.topk = topk
        self.lambda_coeff = lambda_coeff
        self.feat_embed_dim = feat_embed_dim
        self.n_item_layers = n_item_layers
        
        # Modal embeddings (trainable from pretrained)
        self.image_embedding = nn.Embedding.from_pretrained(
            feat_visual.clone(), freeze=False
        )
        self.text_embedding = nn.Embedding.from_pretrained(
            feat_text.clone(), freeze=False
        )
        
        # Modal transformation layers
        self.image_trs = nn.Linear(feat_visual.shape[1], feat_embed_dim)
        self.text_trs = nn.Linear(feat_text.shape[1], feat_embed_dim)
        
        # Learnable modal fusion weights
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)
        
        # Build original adjacencies from raw features
        with torch.no_grad():
            image_adj = build_sim(feat_visual.to(device))
            image_adj = build_knn_neighbourhood(image_adj, topk=topk)
            image_adj = compute_normalized_laplacian(image_adj)
            
            text_adj = build_sim(feat_text.to(device))
            text_adj = build_knn_neighbourhood(text_adj, topk=topk)
            text_adj = compute_normalized_laplacian(text_adj)
        
        self.register_buffer("image_original_adj", image_adj)
        self.register_buffer("text_original_adj", text_adj)
        
        # Learned item adjacency (will be set during forward with build_item_graph=True)
        self.item_adj: Optional[torch.Tensor] = None
        
        self.to(device)
    
    def forward(
        self,
        adj: torch.Tensor,
        users: torch.Tensor = None,
        pos_items: torch.Tensor = None,
        neg_items: torch.Tensor = None,
        build_item_graph: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional item graph rebuilding.
        
        Args:
            adj: (N, N) user-item bipartite adjacency
            users: (batch,) user indices (optional for get_embeddings mode)
            pos_items: (batch,) positive item indices
            neg_items: (batch,) or (batch, n_neg) negative item indices
            build_item_graph: Whether to rebuild item graph (first batch of epoch)
            
        Returns:
            (user_emb, pos_item_emb, neg_item_emb) for batch if users provided
            (all_user_emb, all_item_emb, None) if users is None
        """
        # Project modal features
        image_feats = self.image_trs(self.image_embedding.weight)  # (n_items, feat_dim)
        text_feats = self.text_trs(self.text_embedding.weight)  # (n_items, feat_dim)
        
        if build_item_graph:
            # Rebuild learned adjacency from projected features
            weight = self.softmax(self.modal_weight)
            
            image_adj = build_sim(image_feats)
            image_adj = build_knn_neighbourhood(image_adj, topk=self.topk)
            
            text_adj = build_sim(text_feats)
            text_adj = build_knn_neighbourhood(text_adj, topk=self.topk)
            
            # Weighted fusion of learned adjacencies
            learned_adj = weight[0] * image_adj + weight[1] * text_adj
            learned_adj = compute_normalized_laplacian(learned_adj)
            
            # Combine with original adjacencies
            original_adj = weight[0] * self.image_original_adj + weight[1] * self.text_original_adj
            
            # Interpolate: learned vs original
            self.item_adj = (1 - self.lambda_coeff) * learned_adj + self.lambda_coeff * original_adj
        else:
            # Use cached adjacency (detach to avoid gradient flow through graph construction)
            if self.item_adj is not None:
                self.item_adj = self.item_adj.detach()
        
        # Item graph propagation (n_item_layers hops)
        h = self.item_embedding.weight  # (n_items, dim)
        if self.item_adj is not None:
            for _ in range(self.n_item_layers):
                h = torch.mm(self.item_adj, h)
        
        # LightGCN on user-item graph
        ego_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)  # (N, dim)
        
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings.append(ego_embeddings)
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)  # (N, dim)
        
        u_g_embeddings, i_g_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items], dim=0
        )
        
        # Add item graph enhanced embeddings
        i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
        
        if users is None:
            return u_g_embeddings, i_g_embeddings, None
        
        return (
            u_g_embeddings[users],
            i_g_embeddings[pos_items],
            i_g_embeddings[neg_items],
        )
    
    def compute_loss(
        self,
        adj: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        l2_reg: float = 1e-5,
        build_item_graph: bool = False,
    ) -> dict:
        """
        Compute BPR + L2 loss.
        
        Args:
            adj: User-item adjacency
            users: User indices
            pos_items: Positive item indices
            neg_items: Negative item indices
            l2_reg: L2 regularization weight
            build_item_graph: Whether to rebuild item graph
        """
        user_emb, pos_emb, neg_emb = self.forward(
            adj, users, pos_items, neg_items, build_item_graph=build_item_graph
        )
        
        return self._compute_loss_from_emb(
            user_emb, pos_emb, neg_emb,
            users, pos_items, neg_items,
            l2_reg=l2_reg,
        )
    
    def _compute_loss_from_emb(
        self,
        user_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        adj: torch.Tensor = None,
        l2_reg: float = 1e-5,
        cl_loss_precomputed: torch.Tensor = None,
    ) -> dict:
        """Compute loss from pre-computed embeddings."""
        # BPR loss
        pos_scores = (user_emb * pos_emb).sum(dim=-1)
        neg_scores = (user_emb * neg_emb).sum(dim=-1)
        
        mf_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # L2 regularization
        regularizer = (
            0.5 * (user_emb ** 2).sum() +
            0.5 * (pos_emb ** 2).sum() +
            0.5 * (neg_emb ** 2).sum()
        )
        regularizer = regularizer / user_emb.shape[0]
        emb_loss = l2_reg * regularizer
        
        total = mf_loss + emb_loss
        
        return {
            "loss": total,
            "bpr_loss": mf_loss.item(),
            "reg_loss": emb_loss.item(),
        }
    
    def inductive_forward(
        self,
        adj: torch.Tensor,
        users: torch.Tensor,
        items: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inductive forward for cold items.
        
        Cold items (idx >= n_warm) use modal features only.
        """
        # Get all embeddings
        u_emb, i_emb, _ = self.forward(adj, build_item_graph=False)
        
        # Cold items: Replace with modal features
        cold_mask = items >= self.n_warm
        if cold_mask.any():
            cold_items = items[cold_mask]
            img_feats = self.image_trs(self.image_embedding.weight[cold_items])
            txt_feats = self.text_trs(self.text_embedding.weight[cold_items])
            modal_emb = 0.5 * img_feats + 0.5 * txt_feats
            
            i_emb = i_emb.clone()
            i_emb[cold_items] = F.normalize(modal_emb, p=2, dim=1)
        
        return u_emb[users], i_emb[items]
    
    def _get_all_embeddings(
        self,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all embeddings after propagation."""
        u_emb, i_emb, _ = self.forward(adj, build_item_graph=False)
        return u_emb, i_emb
