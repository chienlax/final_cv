"""
LATTICE: Mining Latent Structures for Multimodal Recommendation.

Learns a latent item-item graph from multimodal features, then uses
LightGCN-style message passing for recommendation.

Key components:
- k-NN graph learner: Builds item similarity graph from modal features
- Graph fusion: Combines learned graph with interaction graph
- Inductive mode: Uses only modal embeddings for cold items
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMultimodalModel


class LATTICEModel(BaseMultimodalModel):
    """
    LATTICE: Learning All-modality Structured Item Graph for Recommendation.
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
        k: int = 10,
        graph_lambda: float = 0.5,
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
            k: Number of k-NN neighbors for graph learning.
            graph_lambda: Weight to balance original vs learned graph.
            device: torch device.
        """
        super().__init__(
            n_users, n_items, n_warm, embed_dim, n_layers,
            feat_visual, feat_text, device
        )
        
        self.k = k
        self.graph_lambda = graph_lambda
        
        # Modality-specific attention weights
        self.modal_attention = nn.Parameter(torch.ones(2) / 2)
        
        # Build learned item-item graph
        self.item_graph = None  # Will be built lazily
        
        self.to(device)
    
    def _build_item_graph(self) -> torch.Tensor:
        """
        Build k-NN item-item graph from modal features.
        
        Returns:
            Sparse (n_items, n_items) adjacency matrix.
        """
        with torch.no_grad():
            # Project features
            visual_emb = self.visual_proj(self.feat_visual)
            text_emb = self.text_proj(self.feat_text)
            
            # Attention-weighted fusion
            attn = F.softmax(self.modal_attention, dim=0)
            fused = attn[0] * visual_emb + attn[1] * text_emb
            
            # Normalize for cosine similarity
            fused = F.normalize(fused, p=2, dim=1)
            
            # Compute similarity matrix (chunked to save memory)
            n_items = fused.shape[0]
            chunk_size = 1000
            
            indices_list = []
            values_list = []
            
            for i in range(0, n_items, chunk_size):
                end_i = min(i + chunk_size, n_items)
                chunk = fused[i:end_i]
                
                # Compute similarities for chunk
                sims = chunk @ fused.T  # (chunk_size, n_items)
                
                # Get top-k for each row (excluding self)
                sims[:, i:end_i] = -float('inf')  # Mask self-connections
                topk_vals, topk_idx = sims.topk(self.k, dim=1)
                
                # Build sparse indices
                for j in range(end_i - i):
                    row_idx = i + j
                    for kk in range(self.k):
                        col_idx = topk_idx[j, kk].item()
                        val = topk_vals[j, kk].item()
                        if val > 0:
                            indices_list.append([row_idx, col_idx])
                            values_list.append(val)
            
            # Create sparse tensor
            if indices_list:
                indices = torch.LongTensor(indices_list).T.to(self.device)
                values = torch.FloatTensor(values_list).to(self.device)
                graph = torch.sparse_coo_tensor(
                    indices, values, (n_items, n_items)
                ).coalesce()
            else:
                graph = torch.sparse_coo_tensor(
                    torch.empty(2, 0, dtype=torch.long, device=self.device),
                    torch.empty(0, device=self.device),
                    (n_items, n_items)
                )
            
            # Symmetric normalization
            graph = self._normalize_sparse(graph)
            
        return graph
    
    def _normalize_sparse(self, adj: torch.Tensor) -> torch.Tensor:
        """Normalize sparse adjacency matrix."""
        # Add self-loops
        n = adj.shape[0]
        eye_indices = torch.arange(n, device=self.device).unsqueeze(0).repeat(2, 1)
        eye_values = torch.ones(n, device=self.device)
        eye = torch.sparse_coo_tensor(eye_indices, eye_values, (n, n))
        
        adj = adj + eye
        adj = adj.coalesce()
        
        # Compute degrees
        degrees = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv_sqrt = degrees.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        
        # D^(-1/2) * A * D^(-1/2)
        indices = adj.indices()
        values = adj.values()
        values = d_inv_sqrt[indices[0]] * values * d_inv_sqrt[indices[1]]
        
        return torch.sparse_coo_tensor(indices, values, adj.shape).coalesce()
    
    def lightgcn_propagate(
        self,
        adj: torch.Tensor,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        LightGCN message passing.
        
        Args:
            adj: Normalized user-item bipartite adjacency.
            user_emb: (n_users, dim) user embeddings.
            item_emb: (n_items, dim) item embeddings.
            
        Returns:
            Tuple of aggregated (user_emb, item_emb).
        """
        # Stack user and item embeddings
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        embs = [all_emb]
        
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            embs.append(all_emb)
        
        # Mean aggregation
        all_emb = torch.stack(embs, dim=0).mean(dim=0)
        
        user_emb = all_emb[:self.n_users]
        item_emb = all_emb[self.n_users:]
        
        return user_emb, item_emb
    
    def item_graph_enhance(
        self,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enhance item embeddings with learned item graph.
        
        Args:
            item_emb: (n_items, dim) item embeddings.
            
        Returns:
            Enhanced item embeddings.
        """
        if self.item_graph is None:
            self.item_graph = self._build_item_graph()
        
        # Message passing on item graph
        enhanced = torch.sparse.mm(self.item_graph, item_emb)
        
        # Blend with original
        item_emb = self.graph_lambda * enhanced + (1 - self.graph_lambda) * item_emb
        
        return item_emb
    
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
        
        # Add modal information to items
        modal_emb = self.get_modal_embeddings()
        item_emb = item_emb + modal_emb
        
        # LightGCN propagation on user-item graph
        user_emb, item_emb = self.lightgcn_propagate(adj, user_emb, item_emb)
        
        # Enhance items with learned item graph
        item_emb = self.item_graph_enhance(item_emb)
        
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
        
        Cold items (idx >= n_warm) use ONLY modal embeddings.
        """
        # Get user embeddings (standard path)
        user_emb = self.user_embedding.weight
        
        # Get item embeddings - split warm/cold
        item_emb = self.item_embedding.weight.clone()
        
        # For cold items, REPLACE ID embedding with modal embedding
        cold_mask = items >= self.n_warm
        
        if cold_mask.any():
            cold_items = items[cold_mask]
            modal_emb = self.get_modal_embeddings(cold_items)
            item_emb[cold_items] = modal_emb
        
        # Warm items: ID + modal
        warm_mask = items < self.n_warm
        if warm_mask.any():
            warm_items = items[warm_mask]
            modal_emb = self.get_modal_embeddings(warm_items)
            item_emb[warm_items] = item_emb[warm_items] + modal_emb
        
        # LightGCN propagation
        user_emb, item_emb = self.lightgcn_propagate(adj, user_emb, item_emb[:self.n_items])
        
        # Item graph enhancement
        item_emb = self.item_graph_enhance(item_emb)
        
        return user_emb[users], item_emb[items]
    
    def compute_loss(
        self,
        adj: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        l2_reg: float = 1e-4,
    ) -> dict:
        """Compute BPR + L2 regularization loss."""
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
        bpr = self.bpr_loss(user_emb, pos_emb, neg_emb)
        
        # Handle multi-negative for L2 reg
        if neg_items.dim() == 2:
            neg_items_flat = neg_items.flatten()
        else:
            neg_items_flat = neg_items
        
        reg = self.l2_reg_loss(
            self.user_embedding.weight[users],
            self.item_embedding.weight[pos_items],
            self.item_embedding.weight[neg_items_flat],
        )
        
        total = bpr + l2_reg * reg
        
        return {
            "loss": total,
            "bpr_loss": bpr.item(),
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
        item_emb = self.item_graph_enhance(item_emb)
        
        return user_emb, item_emb
