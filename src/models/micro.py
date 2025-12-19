"""
MICRO: Multimodal Item-wise Contrastive Recommendation.

FAITHFUL implementation matching the official CRIPAC-DIG/MICRO (SIGIR'22).
Reference: https://github.com/CRIPAC-DIG/MICRO

This is a DIRECT PORT of the original code with minimal adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional


# =============================================================================
# Utility Functions - EXACT COPY from original utility/norm.py
# =============================================================================

def build_sim(context: torch.Tensor) -> torch.Tensor:
    """Build cosine similarity matrix.
    
    EXACT match to original MICRO/codes/utility/norm.py line 3-6.
    """
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim


def build_knn_normalized_graph(
    adj: torch.Tensor,
    topk: int,
    is_sparse: bool = True,
    norm_type: str = "sym",
) -> torch.Tensor:
    """Build k-NN normalized graph.
    
    EXACT match to original MICRO/codes/utility/norm.py line 8-21.
    NOTE: Original uses torch_scatter for sparse, we use dense fallback.
    """
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    
    if is_sparse:
        # Original uses torch_scatter for sparse - we use dense fallback
        # Then convert to sparse for compatibility
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        L_norm = get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)
        # Convert dense to sparse for memory efficiency
        indices = L_norm.nonzero(as_tuple=False).t()
        values = L_norm[indices[0], indices[1]]
        return torch.sparse_coo_tensor(indices, values, L_norm.shape).coalesce()
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)


def get_dense_laplacian(adj: torch.Tensor, normalization: str = 'sym') -> torch.Tensor:
    """Compute normalized Laplacian for dense matrix.
    
    EXACT match to original MICRO/codes/utility/norm.py line 39-54.
    """
    if normalization == 'sym':
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm


# =============================================================================
# MICRO Model - FAITHFUL TO ORIGINAL
# =============================================================================

class MICROModel(nn.Module):
    """
    MICRO: Multimodal Item-wise Contrastive Recommendation.
    
    DIRECT PORT from original MICRO/codes/Models.py with:
    - Same architecture
    - Same forward pass logic  
    - Same variable names where practical
    - LightGCN backbone only (per user request)
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_warm: int,  # For cold-start compatibility
        embed_dim: int,
        n_ui_layers: int,  # Number of UI graph layers (len(weight_size) in original)
        feat_visual: torch.Tensor,
        feat_text: torch.Tensor,
        # MICRO-specific params (from parser.py)
        topk: int = 10,
        lambda_coeff: float = 0.9,
        layers: int = 1,  # args.layers for item graph conv
        tau: float = 0.5,  # Contrastive temperature
        loss_ratio: float = 0.03,  # args.loss_ratio
        sparse: bool = True,  # args.sparse
        norm_type: str = "sym",  # args.norm_type
        # Unused but kept for API compatibility
        projection_hidden_dim: int = 1024,
        projection_dropout: float = 0.5,
        device: str = "cuda",
    ):
        """
        Args match original MICRO __init__ from Models.py line 14.
        
        n_users, n_items: Dataset sizes  
        embed_dim: embedding_dim in original (args.embed_size)
        n_ui_layers: len(weight_size) for UI GCN layers
        feat_visual: image_feats in original
        feat_text: text_feats in original
        topk: args.topk
        lambda_coeff: args.lambda_coeff
        layers: args.layers for item graph conv
        tau: Hardcoded 0.5 in original (line 62)
        loss_ratio: args.loss_ratio
        sparse: args.sparse
        norm_type: args.norm_type
        """
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_warm = n_warm
        self.embedding_dim = embed_dim
        self.n_ui_layers = n_ui_layers  # For user-item GCN
        self.layers = layers  # For item graph propagation (args.layers in original)
        self.topk = topk
        self.lambda_coeff = lambda_coeff
        self.tau = tau
        self.loss_ratio = loss_ratio
        self.sparse = sparse
        self.norm_type = norm_type
        self.device = device
        
        # ID Embeddings - EXACT match to original line 22-25
        self.user_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # Modal Embeddings - EXACT match to original line 37-38
        self.image_embedding = nn.Embedding.from_pretrained(
            feat_visual.clone() if isinstance(feat_visual, torch.Tensor) else torch.Tensor(feat_visual),
            freeze=False
        )
        self.text_embedding = nn.Embedding.from_pretrained(
            feat_text.clone() if isinstance(feat_text, torch.Tensor) else torch.Tensor(feat_text),
            freeze=False
        )
        
        # Build original adjacencies - EXACT match to original line 41-48
        image_adj = build_sim(self.image_embedding.weight.detach())
        image_adj = build_knn_normalized_graph(image_adj, topk=topk, is_sparse=sparse, norm_type=norm_type)
        
        text_adj = build_sim(self.text_embedding.weight.detach())
        text_adj = build_knn_normalized_graph(text_adj, topk=topk, is_sparse=sparse, norm_type=norm_type)
        
        self.text_original_adj = text_adj
        self.image_original_adj = image_adj
        
        # Modal transformation - EXACT match to original line 50-51
        self.image_trs = nn.Linear(feat_visual.shape[1], embed_dim)
        self.text_trs = nn.Linear(feat_text.shape[1], embed_dim)
        
        # Softmax - line 53
        self.softmax = nn.Softmax(dim=-1)
        
        # Query network - EXACT match to original line 56-60
        self.query = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )
        
        # Item adjacencies (set during forward)
        self.image_adj: Optional[torch.Tensor] = None
        self.text_adj: Optional[torch.Tensor] = None
        
        self.to(device)
    
    def mm(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Matrix multiply - EXACT match to original line 64-68."""
        if self.sparse:
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)
    
    def sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Cosine similarity - EXACT match to original line 69-72."""
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    
    def batched_contrastive_loss(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor, 
        batch_size: int = 4096,
    ) -> torch.Tensor:
        """Batched contrastive loss - EXACT match to original line 74-93."""
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]
            
            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        
        loss_vec = torch.cat(losses)
        return loss_vec.mean()
    
    def forward(
        self,
        adj: torch.Tensor,
        build_item_graph: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass - EXACT match to original Models.py line 95-157.
        
        Returns (u_g_embeddings, i_g_embeddings, image_item_embeds, text_item_embeds, h)
        as in original.
        """
        # Line 96-97: Project modal features
        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)
        
        # Line 98-109: Build or reuse item adjacencies
        if build_item_graph:
            self.image_adj = build_sim(image_feats)
            self.image_adj = build_knn_normalized_graph(self.image_adj, topk=self.topk, is_sparse=self.sparse, norm_type=self.norm_type)
            self.image_adj = (1 - self.lambda_coeff) * self.image_adj + self.lambda_coeff * self.image_original_adj
            
            self.text_adj = build_sim(text_feats)
            self.text_adj = build_knn_normalized_graph(self.text_adj, topk=self.topk, is_sparse=self.sparse, norm_type=self.norm_type)
            self.text_adj = (1 - self.lambda_coeff) * self.text_adj + self.lambda_coeff * self.text_original_adj
        else:
            self.image_adj = self.image_adj.detach()
            self.text_adj = self.text_adj.detach()
        
        # Line 111-118: Item graph propagation
        image_item_embeds = self.item_id_embedding.weight
        text_item_embeds = self.item_id_embedding.weight
        
        for i in range(self.layers):
            image_item_embeds = self.mm(self.image_adj, image_item_embeds)
        
        for i in range(self.layers):
            text_item_embeds = self.mm(self.text_adj, text_item_embeds)
        
        # Line 121-123: Attention-based fusion
        att = torch.cat([self.query(image_item_embeds), self.query(text_item_embeds)], dim=-1)
        weight = self.softmax(att)
        h = weight[:, 0].unsqueeze(dim=1) * image_item_embeds + weight[:, 1].unsqueeze(dim=1) * text_item_embeds
        
        # Line 146-157: LightGCN on user-item graph
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
        
        return u_g_embeddings, i_g_embeddings, image_item_embeds, text_item_embeds, h
    
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
        Compute loss - matches original main.py training loop (line 88-104).
        """
        # Get all embeddings
        ua_embeddings, ia_embeddings, image_item_embeds, text_item_embeds, fusion_embed = self.forward(
            adj, build_item_graph=build_item_graph
        )
        
        # Index into batch
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]
        
        # BPR loss - EXACT match to original main.py line 168-180
        pos_scores = torch.sum(torch.mul(u_g_embeddings, pos_i_g_embeddings), dim=1)
        neg_scores = torch.sum(torch.mul(u_g_embeddings, neg_i_g_embeddings), dim=1)
        
        regularizer = 1./2*(u_g_embeddings**2).sum() + 1./2*(pos_i_g_embeddings**2).sum() + 1./2*(neg_i_g_embeddings**2).sum()
        regularizer = regularizer / u_g_embeddings.shape[0]
        
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        
        emb_loss = l2_reg * regularizer
        
        # Contrastive loss - EXACT match to original main.py line 99-103
        batch_contrastive_loss = self.batched_contrastive_loss(image_item_embeds, fusion_embed)
        batch_contrastive_loss += self.batched_contrastive_loss(text_item_embeds, fusion_embed)
        batch_contrastive_loss *= self.loss_ratio
        
        batch_loss = mf_loss + emb_loss + batch_contrastive_loss
        
        return {
            "loss": batch_loss,
            "bpr_loss": mf_loss.item(),
            "reg_loss": emb_loss.item(),
            "cl_loss": batch_contrastive_loss.item(),
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
        image_emb: torch.Tensor = None,
        text_emb: torch.Tensor = None,
        fusion_emb: torch.Tensor = None,
    ) -> dict:
        """Compute loss from pre-computed embeddings (for AMP compatibility)."""
        # BPR loss
        pos_scores = torch.sum(torch.mul(user_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(user_emb, neg_emb), dim=1)
        
        regularizer = 1./2*(user_emb**2).sum() + 1./2*(pos_emb**2).sum() + 1./2*(neg_emb**2).sum()
        regularizer = regularizer / user_emb.shape[0]
        
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        
        emb_loss = l2_reg * regularizer
        
        # Contrastive loss
        cl_loss = torch.tensor(0.0, device=user_emb.device)
        if image_emb is not None and text_emb is not None and fusion_emb is not None:
            cl_loss = self.batched_contrastive_loss(image_emb, fusion_emb)
            cl_loss += self.batched_contrastive_loss(text_emb, fusion_emb)
            cl_loss *= self.loss_ratio
        
        batch_loss = mf_loss + emb_loss + cl_loss
        
        return {
            "loss": batch_loss,
            "bpr_loss": mf_loss.item(),
            "reg_loss": emb_loss.item(),
            "cl_loss": cl_loss.item() if isinstance(cl_loss, torch.Tensor) else 0.0,
        }
    
    def inductive_forward(
        self,
        adj: torch.Tensor,
        users: torch.Tensor,
        items: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inductive forward for cold items."""
        u_emb, i_emb, _, _, _ = self.forward(adj, build_item_graph=False)
        
        # Cold items: Use fused modal features
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
        u_emb, i_emb, _, _, _ = self.forward(adj, build_item_graph=False)
        return u_emb, i_emb
