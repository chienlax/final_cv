"""
LATTICE: Mining Latent Structures for Multimodal Recommendation.

Based on the official CRIPAC-DIG/LATTICE (MM'21).
Reference: https://github.com/CRIPAC-DIG/LATTICE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional


def build_knn_neighbourhood(adj: torch.Tensor, topk: int) -> torch.Tensor:
    """Build k-NN adjacency by keeping top-k neighbors per node."""
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix


def compute_normalized_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """Compute symmetric normalized Laplacian."""
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


def build_sim(context: torch.Tensor) -> torch.Tensor:
    """Build cosine similarity matrix."""
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim


class LATTICEModel(nn.Module):
    """
    LATTICE: Learning All-modality Structured Item Graph.
    
    Uses LightGCN backbone with modal-aware item graph.
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_warm: int,  # For cold-start compatibility
        embed_dim: int,
        n_ui_layers: int,  # Number of UI graph layers (weight_size length in original)
        feat_visual: torch.Tensor,
        feat_text: torch.Tensor,
        # LATTICE-specific params (from parser.py)
        topk: int = 10,
        lambda_coeff: float = 0.9,
        feat_embed_dim: int = 64,
        n_layers: int = 1,  # Number of item graph layers (args.n_layers in original)
        # Unused but kept for API compatibility
        projection_hidden_dim: int = 1024,
        projection_dropout: float = 0.5,
        device: str = "cuda",
    ):
        """
        Args:
            n_users, n_items: Dataset sizes
            embed_dim: embedding_dim in original
            n_ui_layers: Number of UI GCN layers
            feat_visual: Image features
            feat_text: Text features
            topk: K for KNN graph
            lambda_coeff: Interpolation coefficient
            feat_embed_dim: Feature embedding dimension
            n_layers: Number of item graph layers
        """
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_warm = n_warm
        self.embedding_dim = embed_dim
        self.n_ui_layers = n_ui_layers  # For user-item GCN
        self.n_layers = n_layers  # For item graph propagation
        self.topk = topk
        self.lambda_coeff = lambda_coeff
        self.device = device
        
        # ID Embeddings
        self.user_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # Modal Embeddings
        self.image_embedding = nn.Embedding.from_pretrained(
            feat_visual.clone() if isinstance(feat_visual, torch.Tensor) else torch.Tensor(feat_visual),
            freeze=False
        )
        self.text_embedding = nn.Embedding.from_pretrained(
            feat_text.clone() if isinstance(feat_text, torch.Tensor) else torch.Tensor(feat_text),
            freeze=False
        )
        
        # Build original adjacencies - EXACT match to original line 60-74
        # Note: Original loads from file if exists, we compute directly
        image_adj = build_sim(self.image_embedding.weight.detach())
        image_adj = build_knn_neighbourhood(image_adj, topk=topk)
        image_adj = compute_normalized_laplacian(image_adj)
        
        text_adj = build_sim(self.text_embedding.weight.detach())
        text_adj = build_knn_neighbourhood(text_adj, topk=topk)
        text_adj = compute_normalized_laplacian(text_adj)
        
        self.register_buffer("text_original_adj", text_adj)
        self.register_buffer("image_original_adj", image_adj)
        
        # Modal transformation
        self.image_trs = nn.Linear(feat_visual.shape[1], feat_embed_dim)
        self.text_trs = nn.Linear(feat_text.shape[1], feat_embed_dim)
        
        self.cold_proj = nn.Linear(feat_visual.shape[1], embed_dim)
        
        # Modal weight
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)
        
        # Item adjacency (set during forward)
        self.item_adj: Optional[torch.Tensor] = None
        
        self.to(device)
    
    def forward(
        self,
        adj: torch.Tensor,
        build_item_graph: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            adj: Normalized user-item bipartite adjacency (sparse)
            build_item_graph: Whether to rebuild item graph
            
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        # Project modal features
        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)
        
        # Build or reuse item adjacency
        if build_item_graph or self.item_adj is None:
            weight = self.softmax(self.modal_weight)
            
            self.image_adj = build_sim(image_feats)
            self.image_adj = build_knn_neighbourhood(self.image_adj, topk=self.topk)
            
            self.text_adj = build_sim(text_feats)
            self.text_adj = build_knn_neighbourhood(self.text_adj, topk=self.topk)
            
            learned_adj = weight[0] * self.image_adj + weight[1] * self.text_adj
            learned_adj = compute_normalized_laplacian(learned_adj)
            original_adj = weight[0] * self.image_original_adj + weight[1] * self.text_original_adj
            self.item_adj = (1 - self.lambda_coeff) * learned_adj + self.lambda_coeff * original_adj
        else:
            self.item_adj = self.item_adj.detach()
        
        # Item graph propagation
        h = self.item_id_embedding.weight
        for i in range(self.n_layers):
            h = torch.mm(self.item_adj, h)
        
        # LightGCN on user-item graph
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
        
        return u_g_embeddings, i_g_embeddings
    
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
        Compute BPR loss.
        """
        ua_embeddings, ia_embeddings = self.forward(adj, build_item_graph=build_item_graph)
        
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]
        
        pos_scores = torch.sum(torch.mul(u_g_embeddings, pos_i_g_embeddings), dim=1)
        neg_scores = torch.sum(torch.mul(u_g_embeddings, neg_i_g_embeddings), dim=1)
        
        regularizer = 1./2*(u_g_embeddings**2).sum() + 1./2*(pos_i_g_embeddings**2).sum() + 1./2*(neg_i_g_embeddings**2).sum()
        regularizer = regularizer / u_g_embeddings.shape[0]
        
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        
        emb_loss = l2_reg * regularizer
        
        batch_loss = mf_loss + emb_loss
        
        return {
            "loss": batch_loss,
            "bpr_loss": mf_loss.item(),
            "reg_loss": emb_loss.item(),
        }
    
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
        """Compute loss from pre-computed embeddings (for AMP compatibility)."""
        # BPR loss - same formula as compute_loss
        pos_scores = torch.sum(torch.mul(user_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(user_emb, neg_emb), dim=1)
        
        regularizer = 1./2*(user_emb**2).sum() + 1./2*(pos_emb**2).sum() + 1./2*(neg_emb**2).sum()
        regularizer = regularizer / user_emb.shape[0]
        
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        
        emb_loss = l2_reg * regularizer
        
        batch_loss = mf_loss + emb_loss
        
        return {
            "loss": batch_loss,
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
        u_emb, i_emb = self.forward(adj, build_item_graph=False)
        
        # Cold items: Replace with projected modal features
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
        return self.forward(adj, build_item_graph=False)
    
    def get_modal_embeddings(self, items: torch.Tensor = None) -> torch.Tensor:
        """
        Get fused modal embeddings for items (for cold-start evaluation).
        
        Uses cold_proj to map raw 768-dim features to embed_dim (384).
        
        Args:
            items: Item indices. If None, returns all items.
            
        Returns:
            Fused visual + text embeddings in embed_dim space.
        """
        if items is not None:
            img_feats = self.image_embedding.weight[items]
            txt_feats = self.text_embedding.weight[items]
        else:
            img_feats = self.image_embedding.weight
            txt_feats = self.text_embedding.weight
        
        # Project raw features to embed_dim and fuse
        fused = 0.5 * self.cold_proj(img_feats) + 0.5 * self.cold_proj(txt_feats)
        return F.normalize(fused, p=2, dim=1)
