"""
Base model class for multimodal recommendation.

Provides common functionality for LATTICE, MICRO, and DiffMM.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseMultimodalModel(ABC, nn.Module):
    """
    Abstract base class for multimodal recommendation models.
    
    All models must implement:
    - forward(): Standard forward pass for training
    - inductive_forward(): Forward pass for cold items (no ID embeddings)
    - compute_loss(): Compute training loss
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
        projection_hidden_dim: int = 1024,
        projection_dropout: float = 0.5,
        device: str = "cuda",
    ):
        """
        Args:
            n_users: Number of users.
            n_items: Total number of items (warm + cold).
            n_warm: Number of warm items (with training signal).
            embed_dim: Embedding dimension.
            n_layers: Number of GCN layers.
            feat_visual: (n_items, D_v) visual features.
            feat_text: (n_items, D_t) text features.
            projection_hidden_dim: Hidden dimension for modality MLP (default 1024).
            projection_dropout: Dropout rate for modality MLP (default 0.5).
            device: torch device.
        """
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_warm = n_warm
        self.n_cold = n_items - n_warm
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.device = device
        
        # ID Embeddings (the "bad" weight - capped to prevent overfitting)
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        
        # Feature projections (the "good" weight - shared across items)
        # Using 2-layer MLP instead of simple Linear for non-linear mapping
        # "Visual Space" → "Preference Space" is NOT a linear relationship!
        self.visual_dim = feat_visual.shape[1] if feat_visual is not None else 0
        self.text_dim = feat_text.shape[1] if feat_text is not None else 0
        
        if self.visual_dim > 0:
            self.visual_proj = nn.Sequential(
                nn.Linear(self.visual_dim, projection_hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(projection_dropout),
                nn.Linear(projection_hidden_dim, embed_dim),
            )
            self.register_buffer("feat_visual", feat_visual)
        
        if self.text_dim > 0:
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_dim, projection_hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(projection_dropout),
                nn.Linear(projection_hidden_dim, embed_dim),
            )
            self.register_buffer("feat_text", feat_text)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights.
        
        Fun fact: Xavier initialization is named after Xavier Glorot.
        We're basically asking Xavier for good luck with our gradients.
        """
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Initialize projection MLPs (iterate through Sequential layers)
        for proj_name in ['visual_proj', 'text_proj']:
            if hasattr(self, proj_name):
                proj = getattr(self, proj_name)
                for layer in proj:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
    
    def get_user_embeddings(self) -> torch.Tensor:
        """Get all user embeddings."""
        return self.user_embedding.weight
    
    def get_item_embeddings(self, include_cold: bool = True) -> torch.Tensor:
        """
        Get item embeddings.
        
        Args:
            include_cold: If True, include cold item embeddings.
            
        Returns:
            (n_items, embed_dim) or (n_warm, embed_dim) tensor.
        """
        if include_cold:
            return self.item_embedding.weight
        return self.item_embedding.weight[:self.n_warm]
    
    def get_modal_embeddings(self, items: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get fused modal embeddings for items.
        
        Args:
            items: Optional item indices. If None, returns all items.
            
        Returns:
            Fused visual + text embeddings.
        """
        if items is not None:
            visual = self.visual_proj(self.feat_visual[items])
            text = self.text_proj(self.feat_text[items])
        else:
            visual = self.visual_proj(self.feat_visual)
            text = self.text_proj(self.feat_text)
        
        # Simple average fusion (can be overridden)
        return (visual + text) / 2
    
    @abstractmethod
    def forward(
        self,
        adj: torch.Tensor,        # (N, N) sparse - N = n_users + n_items
        users: torch.Tensor,      # (batch,) int64 user indices
        pos_items: torch.Tensor,  # (batch,) int64 positive item indices
        neg_items: torch.Tensor,  # (batch,) or (batch, n_neg) int64 negative indices
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            adj: Normalized bipartite adjacency matrix (sparse).
                 Shape: (n_users + n_items, n_users + n_items)
            users: User indices for this batch.
                   Shape: (batch_size,)
            pos_items: Positive item indices.
                       Shape: (batch_size,)
            neg_items: Negative item indices.
                       Shape: (batch_size,) when n_negatives=1
                       Shape: (batch_size, n_negatives) when n_negatives>1
            
        Returns:
            user_emb:  User embeddings. Shape: (batch_size, embed_dim)
            pos_emb:   Positive item embeddings. Shape: (batch_size, embed_dim)
            neg_emb:   Negative item embeddings.
                       Shape: (batch_size, embed_dim) when n_negatives=1
                       Shape: (batch_size, n_negatives, embed_dim) when n_negatives>1
        """
        pass
    
    @abstractmethod
    def inductive_forward(
        self,
        adj: torch.Tensor,        # (N, N) sparse
        users: torch.Tensor,      # (batch,) int64
        items: torch.Tensor,      # (batch,) int64 - may include cold items
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inductive forward pass for cold items.
        
        Cold items (idx >= n_warm) use ONLY modal embeddings (no ID lookup).
        
        Args:
            adj: Normalized adjacency matrix.
                 Shape: (n_users + n_items, n_users + n_items)
            users: User indices.
                   Shape: (batch_size,)
            items: Item indices (may include cold items with idx >= n_warm).
                   Shape: (batch_size,)
            
        Returns:
            user_emb: User embeddings. Shape: (batch_size, embed_dim)
            item_emb: Item embeddings. Shape: (batch_size, embed_dim)
        """
        pass
    
    def bpr_loss(
        self,
        user_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BPR loss.
        
        Args:
            user_emb: (batch, dim) user embeddings.
            pos_emb: (batch, dim) positive item embeddings.
            neg_emb: (batch, dim) or (batch, n_neg, dim) negative item embeddings.
            
        Returns:
            Scalar BPR loss.
        """
        pos_scores = (user_emb * pos_emb).sum(dim=1)  # (batch,)
        
        # Handle multiple negatives
        if neg_emb.dim() == 3:
            # neg_emb: (batch, n_neg, dim)
            # user_emb needs to be (batch, 1, dim) for broadcasting
            user_expanded = user_emb.unsqueeze(1)  # (batch, 1, dim)
            neg_scores = (user_expanded * neg_emb).sum(dim=2)  # (batch, n_neg)
            # Average over negatives
            loss = -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()
        else:
            # Single negative: (batch, dim)
            neg_scores = (user_emb * neg_emb).sum(dim=1)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        return loss
    
    def l2_reg_loss(
        self,
        *embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute L2 regularization loss.
        
        Args:
            embeddings: Tensors to regularize.
            
        Returns:
            Scalar L2 loss.
        """
        reg_loss = 0.0
        for emb in embeddings:
            reg_loss += emb.norm(2).pow(2) / emb.shape[0]
        
        return reg_loss
    
    @abstractmethod
    def compute_loss(
        self,
        adj: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        l2_reg: float = 1e-4,
    ) -> dict:
        """
        Compute total training loss.
        
        Args:
            adj: Normalized adjacency matrix.
            users: (batch,) user indices.
            pos_items: (batch,) positive item indices.
            neg_items: (batch,) negative item indices.
            l2_reg: L2 regularization weight.
            
        Returns:
            Dict with 'loss' and optional aux losses.
        """
        pass
    
    def predict(
        self,
        adj: torch.Tensor,
        users: torch.Tensor,
        items: Optional[torch.Tensor] = None,
        inductive: bool = False,
    ) -> torch.Tensor:
        """
        Compute prediction scores.
        
        Args:
            adj: Normalized adjacency matrix.
            users: (batch,) user indices.
            items: Optional (batch,) item indices. If None, score all items.
            inductive: If True, use inductive mode for cold items.
            
        Returns:
            (batch,) scores if items given, else (batch, n_items) scores.
        """
        with torch.no_grad():
            if inductive and items is not None:
                user_emb, item_emb = self.inductive_forward(adj, users, items)
                scores = (user_emb * item_emb).sum(dim=1)
            else:
                # Get all embeddings via forward
                all_user_emb, all_item_emb = self._get_all_embeddings(adj)
                
                user_emb = all_user_emb[users]
                
                if items is not None:
                    item_emb = all_item_emb[items]
                    scores = (user_emb * item_emb).sum(dim=1)
                else:
                    scores = user_emb @ all_item_emb.T
        
        return scores
    
    @abstractmethod
    def _get_all_embeddings(
        self,
        adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get all user and item embeddings after GCN.
        
        Args:
            adj: Normalized adjacency matrix.
            
        Returns:
            Tuple of (all_user_emb, all_item_emb).
        """
        pass
