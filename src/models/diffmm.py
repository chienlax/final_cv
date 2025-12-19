"""
DiffMM: Diffusion Model for Multimodal Recommendation.

Faithful implementation matching the official HKUDS/DiffMM (ACM MM'24) paper.
Reference: https://github.com/HKUDS/DiffMM

Key Architecture:
    1. GaussianDiffusion: Noise schedule, q_sample (forward), p_sample (reverse)
    2. Denoise: Per-modality denoising networks with time embeddings
    3. Model: Main GCN with modal adjacency matrices rebuilt via diffusion
    4. Loss: L_total = L_bpr + L_diff + λ_cl * L_cl

Training Loop (per epoch):
    1. Diffusion training: Train denoise models to predict noise
    2. UI Matrix Rebuild: Use p_sample to generate user-item predictions
    3. GCN Training: BPR loss on rebuilt adjacency + contrastive loss
"""

import math
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix

from .base import BaseMultimodalModel


# =============================================================================
# Utility Functions (from original Utils.py)
# =============================================================================

def pairwise_predict(
    user_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
) -> torch.Tensor:
    """
    Compute BPR score difference: score(u, pos) - score(u, neg).
    
    Args:
        user_emb: (batch, dim) user embeddings
        pos_emb: (batch, dim) positive item embeddings
        neg_emb: (batch, dim) negative item embeddings
        
    Returns:
        (batch,) score differences
    """
    pos_scores = (user_emb * pos_emb).sum(dim=-1)  # (batch,)
    neg_scores = (user_emb * neg_emb).sum(dim=-1)  # (batch,)
    return pos_scores - neg_scores


def contrastive_loss(
    embeds1: torch.Tensor,
    embeds2: torch.Tensor,
    nodes: torch.Tensor,
    temp: float = 0.5,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss between two embedding views.
    
    From original DiffMM Utils.py - contrastLoss function.
    
    Args:
        embeds1: (N, dim) first view embeddings
        embeds2: (N, dim) second view embeddings  
        nodes: (batch,) indices of nodes in this batch
        temp: Temperature for softmax
        
    Returns:
        Scalar contrastive loss
    """
    # Normalize embeddings
    embeds1 = F.normalize(embeds1, p=2, dim=1)  # (N, dim)
    embeds2 = F.normalize(embeds2, p=2, dim=1)  # (N, dim)
    
    # Get batch embeddings
    batch_emb1 = embeds1[nodes]  # (batch, dim)
    batch_emb2 = embeds2[nodes]  # (batch, dim)
    
    # Positive scores: diagonal elements
    pos_scores = (batch_emb1 * batch_emb2).sum(dim=-1) / temp  # (batch,)
    
    # All-pairs similarity
    all_scores = batch_emb1 @ embeds2.T / temp  # (batch, N)
    
    # InfoNCE: -log(exp(pos) / sum(exp(all)))
    loss = -pos_scores + torch.logsumexp(all_scores, dim=-1)
    
    return loss.mean()


# =============================================================================
# GCNLayer - Simple graph convolution
# =============================================================================

class GCNLayer(nn.Module):
    """Simple GCN layer: output = adj @ input."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, adj: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adj: (N, N) sparse adjacency matrix
            embeds: (N, dim) node embeddings
            
        Returns:
            (N, dim) propagated embeddings
        """
        return torch.spmm(adj, embeds)


# =============================================================================
# SpAdjDropEdge - Sparse adjacency edge dropout
# =============================================================================

class SpAdjDropEdge(nn.Module):
    """
    Edge dropout for sparse adjacency matrices.
    
    From original DiffMM Model.py.
    """
    
    def __init__(self, keep_rate: float):
        super().__init__()
        self.keep_rate = keep_rate
    
    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout to sparse adjacency.
        
        Args:
            adj: Sparse (N, N) adjacency
            
        Returns:
            Dropped sparse adjacency
        """
        if self.keep_rate >= 1.0:
            return adj
            
        vals = adj._values()
        idxs = adj._indices()
        edge_num = vals.size()[0]
        
        # Random mask
        mask = ((torch.rand(edge_num, device=vals.device) + self.keep_rate).floor()).bool()
        
        # Apply mask and scale
        new_vals = vals[mask] / self.keep_rate
        new_idxs = idxs[:, mask]
        
        return torch.sparse_coo_tensor(new_idxs, new_vals, adj.shape).coalesce()


# =============================================================================
# Denoise - Per-modality denoising network with time embeddings
# =============================================================================

class Denoise(nn.Module):
    """
    Denoising network for diffusion model.
    
    Takes noisy input + time embedding, predicts clean output.
    From original DiffMM Model.py.
    
    Architecture:
        input -> [concat time_emb] -> in_layers -> out_layers -> output
    """
    
    def __init__(
        self,
        in_dims: List[int],
        out_dims: List[int],
        emb_size: int,
        norm: bool = False,
        dropout: float = 0.5,
    ):
        """
        Args:
            in_dims: Input MLP dimensions (e.g., [n_items, 1000, latdim])
            out_dims: Output MLP dimensions (e.g., [latdim, 1000, n_items])
            emb_size: Time embedding dimension
            norm: Whether to normalize input
            dropout: Dropout probability
        """
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.norm = norm
        
        # Time embedding layer
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        # Input layers (first layer has time_emb concatenated)
        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        self.in_layers = nn.ModuleList([
            nn.Linear(d_in, d_out) 
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])
        ])
        
        # Output layers
        self.out_layers = nn.ModuleList([
            nn.Linear(d_in, d_out)
            for d_in, d_out in zip(out_dims[:-1], out_dims[1:])
        ])
        
        self.drop = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier-like scheme."""
        for layer in self.in_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        mess_dropout: bool = True,
    ) -> torch.Tensor:
        """
        Denoise forward pass.
        
        Args:
            x: (batch, in_dim) noisy input
            timesteps: (batch,) diffusion timestep indices
            mess_dropout: Whether to apply dropout
            
        Returns:
            (batch, out_dim) denoised output
        """
        device = x.device
        
        # Sinusoidal time embedding
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, self.time_emb_dim // 2, dtype=torch.float32, device=device)
            / (self.time_emb_dim // 2)
        )
        temp = timesteps[:, None].float() * freqs[None]  # (batch, time_dim/2)
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)  # (batch, time_dim)
        
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
        
        emb = self.emb_layer(time_emb)  # (batch, time_dim)
        
        # Normalize input if specified
        if self.norm:
            x = F.normalize(x, p=2, dim=1)
        
        # Dropout
        if mess_dropout:
            x = self.drop(x)
        
        # Concat input with time embedding
        h = torch.cat([x, emb], dim=-1)  # (batch, in_dim + time_dim)
        
        # Input layers
        for layer in self.in_layers:
            h = layer(h)
            h = torch.tanh(h)
        
        # Output layers
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        
        return h


# =============================================================================
# GaussianDiffusion - Full diffusion process
# =============================================================================

class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion process for recommendation.
    
    From original DiffMM Model.py.
    
    Key methods:
        - q_sample: Forward diffusion (add noise)
        - p_sample: Reverse diffusion (denoise)
        - training_losses: Compute MSE + GraphCL loss
    """
    
    def __init__(
        self,
        noise_scale: float,
        noise_min: float,
        noise_max: float,
        steps: int,
        beta_fixed: bool = True,
    ):
        """
        Args:
            noise_scale: Scaling factor for variance
            noise_min: Minimum noise level
            noise_max: Maximum noise level
            steps: Number of diffusion steps
            beta_fixed: Whether to fix first beta
        """
        super().__init__()
        
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        
        if noise_scale != 0:
            betas = self._get_betas()
            betas = torch.tensor(betas, dtype=torch.float64)
            
            if beta_fixed:
                betas[0] = 0.0001
            
            self.register_buffer("betas", betas)
            self._calculate_for_diffusion()
    
    def _get_betas(self) -> np.ndarray:
        """Compute beta schedule from linear variance."""
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, self.steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
        
        return np.array(betas)
    
    def _calculate_for_diffusion(self):
        """Pre-compute values for diffusion process."""
        alphas = 1.0 - self.betas
        
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        alphas_cumprod_next = torch.cat([alphas_cumprod[1:], torch.tensor([0.0])])
        
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("alphas_cumprod_next", alphas_cumprod_next)
        
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Posterior variance
        posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.cat([posterior_variance[1:2], posterior_variance[1:]]))
        )
        
        # Posterior mean coefficients
        self.register_buffer(
            "posterior_mean_coef1",
            self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise at timestep t.
        
        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        
        Args:
            x_start: (batch, dim) clean data
            t: (batch,) timestep indices
            noise: Optional pre-generated noise
            
        Returns:
            (batch, dim) noisy data
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def _extract(
        self,
        arr: torch.Tensor,
        timesteps: torch.Tensor,
        broadcast_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Extract values from array at timestep indices."""
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def p_mean_variance(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior mean and variance.
        
        Args:
            model: Denoise model
            x: (batch, dim) noisy input
            t: (batch,) timestep indices
            
        Returns:
            Tuple of (model_mean, model_log_variance)
        """
        model_output = model(x, t, mess_dropout=False)
        
        model_variance = self._extract(self.posterior_variance, t, x.shape)
        model_log_variance = self._extract(self.posterior_log_variance_clipped, t, x.shape)
        
        model_mean = (
            self._extract(self.posterior_mean_coef1, t, x.shape) * model_output +
            self._extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        
        return model_mean, model_log_variance
    
    def p_sample(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        steps: int,
        sampling_noise: bool = False,
    ) -> torch.Tensor:
        """
        Reverse diffusion: denoise from x_t to x_0.
        
        Args:
            model: Denoise model
            x_start: (batch, dim) input (can be noisy or clean)
            steps: Number of steps to denoise
            sampling_noise: Whether to add noise during sampling
            
        Returns:
            (batch, dim) denoised output
        """
        device = x_start.device
        
        if steps == 0:
            x_t = x_start
        else:
            t = torch.full((x_start.shape[0],), steps - 1, dtype=torch.long, device=device)
            x_t = self.q_sample(x_start, t)
        
        # Reverse diffusion loop
        indices = list(range(self.steps))[::-1]
        
        for i in indices:
            t = torch.full((x_t.shape[0],), i, dtype=torch.long, device=device)
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
            
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
            else:
                x_t = model_mean
        
        return x_t
    
    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        item_embeds: torch.Tensor,
        batch_index: torch.Tensor,
        modal_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute diffusion training losses.
        
        Args:
            model: Denoise model
            x_start: (batch, n_items) user-item interaction vectors
            item_embeds: (n_items, dim) item ID embeddings
            batch_index: (batch,) user indices
            modal_feats: (n_items, dim) modal feature embeddings
            
        Returns:
            Tuple of (diff_loss, gc_loss) per sample
        """
        batch_size = x_start.size(0)
        device = x_start.device
        
        # Random timesteps
        ts = torch.randint(0, self.steps, (batch_size,), dtype=torch.long, device=device)
        noise = torch.randn_like(x_start)
        
        # Forward diffusion
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start
        
        # Predict clean signal
        model_output = model(x_t, ts)
        
        # MSE loss
        mse = self._mean_flat((x_start - model_output) ** 2)
        
        # SNR weighting
        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where(ts == 0, torch.ones_like(weight), weight)
        
        diff_loss = weight * mse
        
        # GraphCL loss: align user's predicted items with modal features
        usr_model_embeds = torch.mm(model_output, modal_feats)  # (batch, dim)
        usr_id_embeds = torch.mm(x_start, item_embeds)  # (batch, dim)
        
        gc_loss = self._mean_flat((usr_model_embeds - usr_id_embeds) ** 2)
        
        return diff_loss, gc_loss
    
    def _mean_flat(self, tensor: torch.Tensor) -> torch.Tensor:
        """Take mean over all non-batch dimensions."""
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    
    def SNR(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio at timestep t."""
        t = t.clamp(min=0)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])


# =============================================================================
# DiffMM Model - Main model class
# =============================================================================

class DiffMM(BaseMultimodalModel):
    """
    DiffMM: Diffusion-based Multimodal Recommendation.
    
    Full implementation matching official HKUDS/DiffMM (ACM MM'24).
    
    Architecture:
        1. Per-modality Denoise networks for diffusion
        2. UI matrix rebuilt via diffusion sampling
        3. Multi-hop GCN on modal-specific adjacencies
        4. Cross-modal contrastive learning
    
    Loss: L = L_bpr + L_diff_img + L_diff_txt + λ_cl * L_cl
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
        # Diffusion params
        noise_scale: float = 0.1,
        noise_min: float = 0.0001,
        noise_max: float = 0.02,
        steps: int = 5,
        dims: str = "[1000]",
        d_emb_size: int = 10,
        sampling_steps: int = 0,
        sampling_noise: bool = False,
        rebuild_k: int = 1,
        # Loss weights
        e_loss: float = 0.1,
        ssl_reg: float = 1e-2,
        temp: float = 0.5,
        # Architecture params
        keep_rate: float = 0.5,
        ris_lambda: float = 0.5,
        ris_adj_lambda: float = 0.2,
        trans: int = 0,
        cl_method: int = 0,
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
            embed_dim: Embedding dimension (latdim in original)
            n_layers: Number of GCN layers (gnn_layer in original)
            feat_visual: (n_items, visual_dim) visual features
            feat_text: (n_items, text_dim) text features
            noise_scale: Diffusion noise scale
            noise_min: Min noise level
            noise_max: Max noise level
            steps: Diffusion steps
            dims: Denoise MLP dimensions as string
            d_emb_size: Time embedding size
            sampling_steps: Steps for p_sample
            sampling_noise: Add noise during sampling
            rebuild_k: Top-k for UI matrix rebuild
            e_loss: GraphCL loss weight
            ssl_reg: Contrastive loss weight
            temp: Contrastive temperature
            keep_rate: Edge dropout keep rate
            ris_lambda: Residual modal lambda
            ris_adj_lambda: Residual adjacency lambda
            trans: Transform type (0: param, 1: linear, 2: mixed)
            cl_method: 0=modal-modal, 1=modal-main
            projection_hidden_dim: Hidden dim for base projection
            projection_dropout: Dropout for base projection
            device: Torch device
        """
        # Initialize base class (creates user/item embeddings)
        super().__init__(
            n_users, n_items, n_warm, embed_dim, n_layers,
            feat_visual, feat_text,
            projection_hidden_dim=projection_hidden_dim,
            projection_dropout=projection_dropout,
            device=device,
        )
        
        # Store params
        self.noise_scale = noise_scale
        self.steps = steps
        self.dims = eval(dims) if isinstance(dims, str) else dims
        self.d_emb_size = d_emb_size
        self.sampling_steps = sampling_steps
        self.sampling_noise = sampling_noise
        self.rebuild_k = rebuild_k
        self.e_loss = e_loss
        self.ssl_reg = ssl_reg
        self.temp = temp
        self.keep_rate = keep_rate
        self.ris_lambda = ris_lambda
        self.ris_adj_lambda = ris_adj_lambda
        self.trans = trans
        self.cl_method = cl_method
        
        # Get feature dimensions
        image_feat_dim = feat_visual.shape[1]
        text_feat_dim = feat_text.shape[1]
        
        # Feature embeddings (store raw features)
        self.register_buffer("image_embedding", feat_visual)
        self.register_buffer("text_embedding", feat_text)
        
        # Modal transformation layers (based on trans param)
        if trans == 1:
            self.image_trans = nn.Linear(image_feat_dim, embed_dim)
            self.text_trans = nn.Linear(text_feat_dim, embed_dim)
        elif trans == 0:
            self.image_trans = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(image_feat_dim, embed_dim))
            )
            self.text_trans = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(text_feat_dim, embed_dim))
            )
        else:  # trans == 2
            self.image_trans = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(image_feat_dim, embed_dim))
            )
            self.text_trans = nn.Linear(text_feat_dim, embed_dim)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList([GCNLayer() for _ in range(n_layers)])
        
        # Edge dropout
        self.edge_dropper = SpAdjDropEdge(keep_rate)
        
        # Modal weight (learnable)
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)
        
        # Diffusion model
        self.diffusion_model = GaussianDiffusion(
            noise_scale, noise_min, noise_max, steps
        )
        
        # Denoise models (per modality)
        out_dims = self.dims + [n_items]
        in_dims = out_dims[::-1]
        
        self.denoise_model_image = Denoise(
            in_dims, out_dims, d_emb_size, norm=False, dropout=0.5
        )
        self.denoise_model_text = Denoise(
            in_dims, out_dims, d_emb_size, norm=False, dropout=0.5
        )
        
        # Misc
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)
        
        # UI matrices (will be built during training)
        self.image_ui_matrix: Optional[torch.Tensor] = None
        self.text_ui_matrix: Optional[torch.Tensor] = None
        
        self.to(device)
    
    # =========================================================================
    # Feature extraction
    # =========================================================================
    
    def get_image_feats(self) -> torch.Tensor:
        """Get projected image features."""
        if self.trans == 0 or self.trans == 2:
            return self.leaky_relu(torch.mm(self.image_embedding, self.image_trans))
        else:
            return self.image_trans(self.image_embedding)
    
    def get_text_feats(self) -> torch.Tensor:
        """Get projected text features."""
        if self.trans == 0:
            return self.leaky_relu(torch.mm(self.text_embedding, self.text_trans))
        else:
            return self.text_trans(self.text_embedding)
    
    # =========================================================================
    # Forward passes
    # =========================================================================
    
    def forward_mm(
        self,
        adj: torch.Tensor,
        image_adj: torch.Tensor,
        text_adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main multimodal forward pass.
        
        Uses rebuilt modal adjacency matrices for enhanced propagation.
        
        Args:
            adj: (N, N) user-item bipartite adjacency
            image_adj: (N, N) image modal adjacency (rebuilt)
            text_adj: (N, N) text modal adjacency (rebuilt)
            
        Returns:
            (user_emb, item_emb) final embeddings
        """
        image_feats = self.get_image_feats()  # (n_items, dim)
        text_feats = self.get_text_feats()  # (n_items, dim)
        
        weight = self.softmax(self.modal_weight)
        
        u_emb = self.user_embedding.weight  # (n_users, dim)
        i_emb = self.item_embedding.weight  # (n_items, dim)
        
        # === Image modality processing ===
        # Propagate on rebuilt image adjacency
        embeds_image_adj = torch.cat([u_emb, i_emb], dim=0)  # (N, dim)
        embeds_image_adj = torch.spmm(image_adj, embeds_image_adj)
        
        # Propagate features on original adjacency
        embeds_image = torch.cat([u_emb, F.normalize(image_feats)], dim=0)
        embeds_image = torch.spmm(adj, embeds_image)
        
        # Additional hop
        embeds_image_ = torch.cat([embeds_image[:self.n_users], i_emb], dim=0)
        embeds_image_ = torch.spmm(adj, embeds_image_)
        embeds_image = embeds_image + embeds_image_
        
        # === Text modality processing ===
        embeds_text_adj = torch.cat([u_emb, i_emb], dim=0)
        embeds_text_adj = torch.spmm(text_adj, embeds_text_adj)
        
        embeds_text = torch.cat([u_emb, F.normalize(text_feats)], dim=0)
        embeds_text = torch.spmm(adj, embeds_text)
        
        embeds_text_ = torch.cat([embeds_text[:self.n_users], i_emb], dim=0)
        embeds_text_ = torch.spmm(adj, embeds_text_)
        embeds_text = embeds_text + embeds_text_
        
        # Add residual from rebuilt adjacencies
        embeds_image = embeds_image + self.ris_adj_lambda * embeds_image_adj
        embeds_text = embeds_text + self.ris_adj_lambda * embeds_text_adj
        
        # Weighted fusion
        embeds_modal = weight[0] * embeds_image + weight[1] * embeds_text
        
        # GCN layers
        embeds = embeds_modal
        embeds_list = [embeds]
        for gcn in self.gcn_layers:
            embeds = gcn(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        
        # Final residual
        embeds = embeds + self.ris_lambda * F.normalize(embeds_modal)
        
        return embeds[:self.n_users], embeds[self.n_users:]
    
    def forward_cl_mm(
        self,
        adj: torch.Tensor,
        image_adj: torch.Tensor,
        text_adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for contrastive learning views.
        
        Returns separate embeddings for each modality.
        
        Args:
            adj: User-item bipartite adjacency
            image_adj: Image modal adjacency
            text_adj: Text modal adjacency
            
        Returns:
            (user_img, item_img, user_txt, item_txt)
        """
        image_feats = self.get_image_feats()
        text_feats = self.get_text_feats()
        
        u_emb = self.user_embedding.weight
        
        # Image view
        embeds_image = torch.cat([u_emb, F.normalize(image_feats)], dim=0)
        embeds_image = torch.spmm(image_adj, embeds_image)
        
        embeds1 = embeds_image
        embeds_list1 = [embeds1]
        for gcn in self.gcn_layers:
            embeds1 = gcn(adj, embeds_list1[-1])
            embeds_list1.append(embeds1)
        embeds1 = sum(embeds_list1)
        
        # Text view
        embeds_text = torch.cat([u_emb, F.normalize(text_feats)], dim=0)
        embeds_text = torch.spmm(text_adj, embeds_text)
        
        embeds2 = embeds_text
        embeds_list2 = [embeds2]
        for gcn in self.gcn_layers:
            embeds2 = gcn(adj, embeds_list2[-1])
            embeds_list2.append(embeds2)
        embeds2 = sum(embeds_list2)
        
        return (
            embeds1[:self.n_users], embeds1[self.n_users:],
            embeds2[:self.n_users], embeds2[self.n_users:],
        )
    
    def forward(
        self,
        adj: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for BPR training.
        
        Uses rebuilt UI matrices if available, else falls back to base.
        """
        if self.image_ui_matrix is not None and self.text_ui_matrix is not None:
            user_emb, item_emb = self.forward_mm(
                adj, self.image_ui_matrix, self.text_ui_matrix
            )
        else:
            # Fallback to simple forward
            user_emb, item_emb = self._simple_forward(adj)
        
        return user_emb[users], item_emb[pos_items], item_emb[neg_items]
    
    def _simple_forward(
        self,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple LightGCN forward without modal adjacencies."""
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # Add modal embeddings
        modal_emb = 0.5 * self.get_image_feats() + 0.5 * self.get_text_feats()
        item_emb = item_emb + modal_emb
        
        # LightGCN propagation
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.spmm(adj, all_emb)
            embs.append(all_emb)
        all_emb = torch.stack(embs, dim=0).mean(dim=0)
        
        return all_emb[:self.n_users], all_emb[self.n_users:]
    
    # =========================================================================
    # UI Matrix Operations
    # =========================================================================
    
    def normalize_adj(self, mat: sp.coo_matrix) -> sp.coo_matrix:
        """Symmetric normalization of adjacency matrix."""
        degree = np.array(mat.sum(axis=-1)).flatten()
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return mat.dot(d_mat_inv_sqrt).T.dot(d_mat_inv_sqrt).tocoo()
    
    def build_ui_matrix(
        self,
        u_list: np.ndarray,
        i_list: np.ndarray,
        edge_list: np.ndarray,
    ) -> torch.Tensor:
        """
        Build user-item sparse matrix from edge list.
        
        Args:
            u_list: User indices
            i_list: Item indices
            edge_list: Edge weights
            
        Returns:
            Sparse (N, N) adjacency tensor
        """
        mat = coo_matrix(
            (edge_list, (u_list, i_list)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )
        
        # Build bipartite adjacency
        a = sp.csr_matrix((self.n_users, self.n_users))
        b = sp.csr_matrix((self.n_items, self.n_items))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.T, b])])
        mat = (mat != 0).astype(np.float32)
        mat = mat + sp.eye(mat.shape[0])
        mat = self.normalize_adj(mat)
        
        # Convert to torch sparse
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        
        return torch.sparse_coo_tensor(idxs, vals, shape).to(self.device)
    
    def rebuild_ui_matrices(
        self,
        diffusion_loader,
    ) -> None:
        """
        Rebuild UI matrices using diffusion sampling.
        
        Called once per epoch after diffusion training.
        
        Args:
            diffusion_loader: DataLoader yielding (batch_item, batch_index)
        """
        u_list_image, i_list_image, edge_list_image = [], [], []
        u_list_text, i_list_text, edge_list_text = [], [], []
        
        item_embeds = self.item_embedding.weight.detach()
        
        with torch.no_grad():
            for batch_item, batch_index in diffusion_loader:
                batch_item = batch_item.to(self.device)
                batch_index = batch_index.to(self.device)
                
                # Image modality
                denoised_image = self.diffusion_model.p_sample(
                    self.denoise_model_image,
                    batch_item,
                    self.sampling_steps,
                    self.sampling_noise,
                )
                _, top_indices_img = torch.topk(denoised_image, k=self.rebuild_k)
                
                for i in range(batch_index.shape[0]):
                    for j in range(top_indices_img[i].shape[0]):
                        u_list_image.append(int(batch_index[i].cpu().numpy()))
                        i_list_image.append(int(top_indices_img[i][j].cpu().numpy()))
                        edge_list_image.append(1.0)
                
                # Text modality
                denoised_text = self.diffusion_model.p_sample(
                    self.denoise_model_text,
                    batch_item,
                    self.sampling_steps,
                    self.sampling_noise,
                )
                _, top_indices_txt = torch.topk(denoised_text, k=self.rebuild_k)
                
                for i in range(batch_index.shape[0]):
                    for j in range(top_indices_txt[i].shape[0]):
                        u_list_text.append(int(batch_index[i].cpu().numpy()))
                        i_list_text.append(int(top_indices_txt[i][j].cpu().numpy()))
                        edge_list_text.append(1.0)
        
        # Build matrices
        self.image_ui_matrix = self.build_ui_matrix(
            np.array(u_list_image),
            np.array(i_list_image),
            np.array(edge_list_image),
        )
        self.image_ui_matrix = self.edge_dropper(self.image_ui_matrix)
        
        self.text_ui_matrix = self.build_ui_matrix(
            np.array(u_list_text),
            np.array(i_list_text),
            np.array(edge_list_text),
        )
        self.text_ui_matrix = self.edge_dropper(self.text_ui_matrix)
    
    # =========================================================================
    # Loss computation
    # =========================================================================
    
    def reg_loss(self) -> torch.Tensor:
        """L2 regularization loss on embeddings."""
        ret = self.user_embedding.weight.norm(2).square()
        ret += self.item_embedding.weight.norm(2).square()
        return ret
    
    def compute_diffusion_loss(
        self,
        batch_item: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute diffusion training loss for one batch.
        
        Args:
            batch_item: (batch, n_items) user-item vectors
            batch_index: (batch,) user indices
            
        Returns:
            (loss_image, loss_text) diffusion losses
        """
        item_embeds = self.item_embedding.weight.detach()
        image_feats = self.get_image_feats().detach()
        text_feats = self.get_text_feats().detach()
        
        # Image modality
        diff_loss_img, gc_loss_img = self.diffusion_model.training_losses(
            self.denoise_model_image,
            batch_item,
            item_embeds,
            batch_index,
            image_feats,
        )
        loss_image = diff_loss_img.mean() + gc_loss_img.mean() * self.e_loss
        
        # Text modality
        diff_loss_txt, gc_loss_txt = self.diffusion_model.training_losses(
            self.denoise_model_text,
            batch_item,
            item_embeds,
            batch_index,
            text_feats,
        )
        loss_text = diff_loss_txt.mean() + gc_loss_txt.mean() * self.e_loss
        
        return loss_image, loss_text
    
    def compute_contrastive_loss(
        self,
        adj: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-modal contrastive loss.
        
        Following original DiffMM implementation.
        """
        if self.image_ui_matrix is None or self.text_ui_matrix is None:
            return torch.tensor(0.0, device=self.device)
        
        # Get main embeddings
        usr_embeds, itm_embeds = self.forward_mm(
            adj, self.image_ui_matrix, self.text_ui_matrix
        )
        
        # Get contrastive views
        usr_img, itm_img, usr_txt, itm_txt = self.forward_cl_mm(
            adj, self.image_ui_matrix, self.text_ui_matrix
        )
        
        # Cross-modal contrastive (image vs text)
        cl_loss = contrastive_loss(usr_img, usr_txt, users, self.temp)
        cl_loss += contrastive_loss(itm_img, itm_txt, pos_items, self.temp)
        
        # Main vs modal contrastive
        if self.cl_method == 1:
            cl_loss = contrastive_loss(usr_embeds, usr_img, users, self.temp)
            cl_loss += contrastive_loss(itm_embeds, itm_img, pos_items, self.temp)
            cl_loss += contrastive_loss(usr_embeds, usr_txt, users, self.temp)
            cl_loss += contrastive_loss(itm_embeds, itm_txt, pos_items, self.temp)
        else:
            # Add main-modal alignment on top
            cl_loss += contrastive_loss(usr_embeds, usr_img, users, self.temp)
            cl_loss += contrastive_loss(itm_embeds, itm_img, pos_items, self.temp)
            cl_loss += contrastive_loss(usr_embeds, usr_txt, users, self.temp)
            cl_loss += contrastive_loss(itm_embeds, itm_txt, pos_items, self.temp)
        
        return cl_loss * self.ssl_reg
    
    def compute_loss(
        self,
        adj: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        l2_reg: float = 1e-5,
    ) -> dict:
        """
        Compute full BPR + regularization loss.
        
        Note: Diffusion loss is computed separately during diffusion training phase.
        """
        user_emb, pos_emb, neg_emb = self.forward(adj, users, pos_items, neg_items)
        
        # BPR loss
        score_diff = pairwise_predict(user_emb, pos_emb, neg_emb)
        bpr_loss = -score_diff.sigmoid().log().mean()
        
        # Regularization
        reg_loss = self.reg_loss() * l2_reg
        
        # Contrastive loss
        cl_loss = self.compute_contrastive_loss(adj, users, pos_items)
        
        total = bpr_loss + reg_loss + cl_loss
        
        return {
            "loss": total,
            "bpr_loss": bpr_loss.item(),
            "reg_loss": reg_loss.item(),
            "cl_loss": cl_loss.item() if isinstance(cl_loss, torch.Tensor) else 0.0,
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
        # BPR loss
        score_diff = pairwise_predict(user_emb, pos_emb, neg_emb)
        bpr_loss = -score_diff.sigmoid().log().mean()
        
        # Regularization
        reg_loss = self.reg_loss() * l2_reg
        
        # Contrastive loss
        if cl_loss_precomputed is not None:
            cl_loss = cl_loss_precomputed
        elif adj is not None:
            cl_loss = self.compute_contrastive_loss(adj, users, pos_items)
        else:
            cl_loss = torch.tensor(0.0, device=self.device)
        
        total = bpr_loss + reg_loss + cl_loss
        
        return {
            "loss": total,
            "bpr_loss": bpr_loss.item(),
            "reg_loss": reg_loss.item(),
            "cl_loss": cl_loss.item() if isinstance(cl_loss, torch.Tensor) else 0.0,
        }
    
    # =========================================================================
    # Inductive mode for cold-start
    # =========================================================================
    
    def inductive_forward(
        self,
        adj: torch.Tensor,
        users: torch.Tensor,
        items: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inductive forward for cold items."""
        user_emb, item_emb = self._simple_forward(adj)
        
        # Cold items: Use modal features only
        cold_mask = items >= self.n_warm
        if cold_mask.any():
            cold_items = items[cold_mask]
            modal_emb = 0.5 * self.get_image_feats()[cold_items] + 0.5 * self.get_text_feats()[cold_items]
            item_emb = item_emb.clone()
            item_emb[cold_items] = modal_emb
        
        return user_emb[users], item_emb[items]
    
    def _get_all_embeddings(
        self,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all embeddings after propagation."""
        if self.image_ui_matrix is not None and self.text_ui_matrix is not None:
            return self.forward_mm(adj, self.image_ui_matrix, self.text_ui_matrix)
        return self._simple_forward(adj)
