"""
DiffMM: Diffusion Model for Multimodal Recommendation.

FAITHFUL implementation matching the official HKUDS/DiffMM (ACM MM'24).
Reference: https://github.com/HKUDS/DiffMM

This is a DIRECT PORT of the original code with minimal adaptation.
"""

import math
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix


# =============================================================================
# Utility Functions - EXACT COPY from original Utils/Utils.py
# =============================================================================

def innerProduct(usrEmbeds: torch.Tensor, itmEmbeds: torch.Tensor) -> torch.Tensor:
    """Dot product along last dimension."""
    return torch.sum(usrEmbeds * itmEmbeds, dim=-1)


def pairPredict(ancEmbeds: torch.Tensor, posEmbeds: torch.Tensor, negEmbeds: torch.Tensor) -> torch.Tensor:
    """BPR score difference.
    
    EXACT match to original Utils/Utils.py line 7-8.
    """
    return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)


def contrastLoss(embeds1: torch.Tensor, embeds2: torch.Tensor, nodes: torch.Tensor, temp: float) -> torch.Tensor:
    """InfoNCE contrastive loss.
    
    EXACT match to original Utils/Utils.py line 31-38.
    """
    embeds1 = F.normalize(embeds1, p=2)
    embeds2 = F.normalize(embeds2, p=2)
    pckEmbeds1 = embeds1[nodes]
    pckEmbeds2 = embeds2[nodes]
    nume = torch.exp(torch.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
    deno = torch.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1)
    return -torch.log(nume / deno).mean()


# =============================================================================
# GCNLayer - EXACT COPY
# =============================================================================

class GCNLayer(nn.Module):
    """Simple GCN layer.
    
    EXACT match to original Model.py line 215-220.
    """
    def __init__(self):
        super(GCNLayer, self).__init__()
    
    def forward(self, adj: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
        return torch.spmm(adj, embeds)


# =============================================================================
# SpAdjDropEdge - EXACT COPY
# =============================================================================

class SpAdjDropEdge(nn.Module):
    """Edge dropout for sparse adjacency.
    
    EXACT match to original Model.py line 222-236.
    """
    def __init__(self, keepRate: float):
        super(SpAdjDropEdge, self).__init__()
        self.keepRate = keepRate
    
    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)
        
        newVals = vals[mask] / self.keepRate
        newIdxs = idxs[:, mask]
        
        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)


# =============================================================================
# Denoise - EXACT COPY
# =============================================================================

class Denoise(nn.Module):
    """Denoising network for diffusion.
    
    EXACT match to original Model.py line 238-296.
    """
    def __init__(self, in_dims: List[int], out_dims: List[int], emb_size: int, norm: bool = False, dropout: float = 0.5):
        super(Denoise, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.norm = norm
        
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        out_dims_temp = self.out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
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
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, mess_dropout: bool = True) -> torch.Tensor:
        device = x.device
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).to(device)
        temp = timesteps[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        if mess_dropout:
            x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        return h


# =============================================================================
# GaussianDiffusion - EXACT COPY
# =============================================================================

class GaussianDiffusion(nn.Module):
    """Gaussian Diffusion process.
    
    EXACT match to original Model.py line 298-420.
    """
    def __init__(self, noise_scale: float, noise_min: float, noise_max: float, steps: int, beta_fixed: bool = True):
        super(GaussianDiffusion, self).__init__()
        
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        
        if noise_scale != 0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64)
            if beta_fixed:
                self.betas[0] = 0.0001
            
            self.calculate_for_diffusion()
    
    def get_betas(self) -> np.ndarray:
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, self.steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
        return np.array(betas)
    
    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0])])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))
    
    def p_sample(self, model: nn.Module, x_start: torch.Tensor, steps: int, sampling_noise: bool = False) -> torch.Tensor:
        device = x_start.device
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps-1] * x_start.shape[0]).to(device)
            x_t = self.q_sample(x_start, t)
        
        indices = list(range(self.steps))[::-1]
        
        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(device)
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = ((t!=0).float().view(-1, *([1]*(len(x_t.shape)-1))))
                x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
            else:
                x_t = model_mean
        return x_t
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = x_start.device
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract_into_tensor(self.sqrt_alphas_cumprod.to(device), t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod.to(device), t, x_start.shape) * noise
    
    def _extract_into_tensor(self, arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape) -> torch.Tensor:
        device = timesteps.device
        arr = arr.to(device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def p_mean_variance(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        model_output = model(x, t, False)
        
        model_variance = self._extract_into_tensor(self.posterior_variance.to(device), t, x.shape)
        model_log_variance = self._extract_into_tensor(self.posterior_log_variance_clipped.to(device), t, x.shape)
        
        model_mean = (self._extract_into_tensor(self.posterior_mean_coef1.to(device), t, x.shape) * model_output + 
                      self._extract_into_tensor(self.posterior_mean_coef2.to(device), t, x.shape) * x)
        
        return model_mean, model_log_variance
    
    def training_losses(self, model: nn.Module, x_start: torch.Tensor, itmEmbeds: torch.Tensor, batch_index: torch.Tensor, model_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x_start.device
        batch_size = x_start.size(0)
        
        ts = torch.randint(0, self.steps, (batch_size,)).long().to(device)
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start
        
        model_output = model(x_t, ts)
        
        mse = self.mean_flat((x_start - model_output) ** 2)
        
        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)
        
        diff_loss = weight * mse
        
        usr_model_embeds = torch.mm(model_output, model_feats)
        usr_id_embeds = torch.mm(x_start, itmEmbeds)
        
        gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)
        
        return diff_loss, gc_loss
    
    def mean_flat(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    
    def SNR(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])


# =============================================================================
# DiffMM Model - FAITHFUL PORT FROM ORIGINAL
# =============================================================================

class DiffMM(nn.Module):
    """
    DiffMM: Diffusion-based Multimodal Recommendation.
    
    DIRECT PORT from original HKUDS/DiffMM with:
    - Same architecture (Model.py)
    - Same training logic (Main.py)
    - Same variable names where practical
    - LightGCN backbone
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_warm: int,  # For API compatibility
        embed_dim: int,
        n_layers: int,
        feat_visual: torch.Tensor,
        feat_text: torch.Tensor,
        # Diffusion params (from Params.py)
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
        reg: float = 1e-5,
        # Architecture params
        keep_rate: float = 0.5,
        ris_lambda: float = 0.5,
        ris_adj_lambda: float = 0.2,
        trans: int = 0,
        cl_method: int = 0,
        # Unused but for API compatibility
        projection_hidden_dim: int = 1024,
        projection_dropout: float = 0.5,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_warm = n_warm
        self.embedding_dim = embed_dim
        self.n_layers = n_layers
        self.device = device
        
        # Params
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
        self.reg = reg
        self.keep_rate = keep_rate
        self.ris_lambda = ris_lambda
        self.ris_adj_lambda = ris_adj_lambda
        self.trans = trans
        self.cl_method = cl_method
        
        # === Model.py __init__ line 17-18 ===
        init = nn.init.xavier_uniform_
        self.uEmbeds = nn.Parameter(init(torch.empty(n_users, embed_dim)))
        self.iEmbeds = nn.Parameter(init(torch.empty(n_items, embed_dim)))
        
        # === GCN layers - line 19 ===
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(n_layers)])
        
        # === Edge dropper - line 21 ===
        self.edgeDropper = SpAdjDropEdge(keep_rate)
        
        # === Feature dimensions ===
        image_feat_dim = feat_visual.shape[1]
        text_feat_dim = feat_text.shape[1]
        
        # === Modal transformations - line 23-31 ===
        if trans == 1:
            self.image_trans = nn.Linear(image_feat_dim, embed_dim)
            self.text_trans = nn.Linear(text_feat_dim, embed_dim)
        elif trans == 0:
            self.image_trans = nn.Parameter(init(torch.empty(size=(image_feat_dim, embed_dim))))
            self.text_trans = nn.Parameter(init(torch.empty(size=(text_feat_dim, embed_dim))))
        else:  # trans == 2
            self.image_trans = nn.Parameter(init(torch.empty(size=(image_feat_dim, embed_dim))))
            self.text_trans = nn.Linear(text_feat_dim, embed_dim)
        
        # === Feature embeddings - line 38-39 ===
        self.register_buffer("image_embedding", feat_visual.clone())
        self.register_buffer("text_embedding", feat_text.clone())
        
        # === Modal weight - line 48 ===
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)
        
        # === Misc - line 51-53 ===
        self.dropout = nn.Dropout(p=0.1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        # === Diffusion model ===
        self.diffusion_model = GaussianDiffusion(noise_scale, noise_min, noise_max, steps)
        
        # === Denoise models (per modality) ===
        out_dims = self.dims + [n_items]
        in_dims = out_dims[::-1]
        self.denoise_model_image = Denoise(in_dims, out_dims, d_emb_size, norm=False)
        self.denoise_model_text = Denoise(in_dims, out_dims, d_emb_size, norm=False)
        
        # UI matrices (built during training)
        self.image_UI_matrix: Optional[torch.Tensor] = None
        self.text_UI_matrix: Optional[torch.Tensor] = None
        
        self.to(device)
    
    # =========================================================================
    # Feature extraction - EXACT from Model.py line 55-73
    # =========================================================================
    
    def getItemEmbeds(self) -> torch.Tensor:
        return self.iEmbeds
    
    def getUserEmbeds(self) -> torch.Tensor:
        return self.uEmbeds
    
    def getImageFeats(self) -> torch.Tensor:
        if self.trans == 0 or self.trans == 2:
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            return image_feats
        else:
            return self.image_trans(self.image_embedding)
    
    def getTextFeats(self) -> torch.Tensor:
        if self.trans == 0:
            text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
            return text_feats
        else:
            return self.text_trans(self.text_embedding)
    
    # =========================================================================
    # forward_MM - EXACT from Model.py line 85-153
    # =========================================================================
    
    def forward_MM(self, adj: torch.Tensor, image_adj: torch.Tensor, text_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Main multimodal forward pass."""
        if self.trans == 0:
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
        elif self.trans == 1:
            image_feats = self.image_trans(self.image_embedding)
            text_feats = self.text_trans(self.text_embedding)
        else:
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            text_feats = self.text_trans(self.text_embedding)
        
        weight = self.softmax(self.modal_weight)
        
        embedsImageAdj = torch.concat([self.uEmbeds, self.iEmbeds])
        embedsImageAdj = torch.spmm(image_adj, embedsImageAdj)
        
        embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
        embedsImage = torch.spmm(adj, embedsImage)
        
        embedsImage_ = torch.concat([embedsImage[:self.n_users], self.iEmbeds])
        embedsImage_ = torch.spmm(adj, embedsImage_)
        embedsImage += embedsImage_
        
        embedsTextAdj = torch.concat([self.uEmbeds, self.iEmbeds])
        embedsTextAdj = torch.spmm(text_adj, embedsTextAdj)
        
        embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
        embedsText = torch.spmm(adj, embedsText)
        
        embedsText_ = torch.concat([embedsText[:self.n_users], self.iEmbeds])
        embedsText_ = torch.spmm(adj, embedsText_)
        embedsText += embedsText_
        
        embedsImage += self.ris_adj_lambda * embedsImageAdj
        embedsText += self.ris_adj_lambda * embedsTextAdj
        
        embedsModal = weight[0] * embedsImage + weight[1] * embedsText
        
        embeds = embedsModal
        embedsLst = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        embeds = sum(embedsLst)
        
        embeds = embeds + self.ris_lambda * F.normalize(embedsModal)
        
        return embeds[:self.n_users], embeds[self.n_users:]
    
    # =========================================================================
    # forward_cl_MM - EXACT from Model.py line 155-207
    # =========================================================================
    
    def forward_cl_MM(self, adj: torch.Tensor, image_adj: torch.Tensor, text_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for contrastive views."""
        if self.trans == 0:
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
        elif self.trans == 1:
            image_feats = self.image_trans(self.image_embedding)
            text_feats = self.text_trans(self.text_embedding)
        else:
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            text_feats = self.text_trans(self.text_embedding)
        
        embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
        embedsImage = torch.spmm(image_adj, embedsImage)
        
        embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
        embedsText = torch.spmm(text_adj, embedsText)
        
        embeds1 = embedsImage
        embedsLst1 = [embeds1]
        for gcn in self.gcnLayers:
            embeds1 = gcn(adj, embedsLst1[-1])
            embedsLst1.append(embeds1)
        embeds1 = sum(embedsLst1)
        
        embeds2 = embedsText
        embedsLst2 = [embeds2]
        for gcn in self.gcnLayers:
            embeds2 = gcn(adj, embedsLst2[-1])
            embedsLst2.append(embeds2)
        embeds2 = sum(embedsLst2)
        
        return embeds1[:self.n_users], embeds1[self.n_users:], embeds2[:self.n_users], embeds2[self.n_users:]
    
    # =========================================================================
    # reg_loss - EXACT from Model.py line 209-213
    # =========================================================================
    
    def reg_loss(self) -> torch.Tensor:
        ret = 0
        ret += self.uEmbeds.norm(2).square()
        ret += self.iEmbeds.norm(2).square()
        return ret
    
    # =========================================================================
    # UI Matrix Operations - EXACT from Main.py line 89-110
    # =========================================================================
    
    def normalizeAdj(self, mat) -> sp.coo_matrix:
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()
    
    def buildUIMatrix(self, u_list: np.ndarray, i_list: np.ndarray, edge_list: np.ndarray) -> torch.Tensor:
        mat = coo_matrix((edge_list, (u_list, i_list)), shape=(self.n_users, self.n_items), dtype=np.float32)
        
        a = sp.csr_matrix((self.n_users, self.n_users))
        b = sp.csr_matrix((self.n_items, self.n_items))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)
        
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        
        return torch.sparse.FloatTensor(idxs, vals, shape).to(self.device)
    
    # =========================================================================
    # Forward for compatibility
    # =========================================================================
    
    def forward(self, adj: torch.Tensor, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for BPR training."""
        if self.image_UI_matrix is not None and self.text_UI_matrix is not None:
            usrEmbeds, itmEmbeds = self.forward_MM(adj, self.image_UI_matrix, self.text_UI_matrix)
        else:
            # Simple fallback
            usrEmbeds, itmEmbeds = self._simple_forward(adj)
        
        return usrEmbeds[users], itmEmbeds[pos_items], itmEmbeds[neg_items]
    
    def _simple_forward(self, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple LightGCN fallback."""
        all_emb = torch.concat([self.uEmbeds, self.iEmbeds])
        embs = [all_emb]
        for gcn in self.gcnLayers:
            all_emb = gcn(adj, all_emb)
            embs.append(all_emb)
        all_emb = sum(embs)
        return all_emb[:self.n_users], all_emb[self.n_users:]
    
    # =========================================================================
    # compute_loss - matches Main.py training logic
    # =========================================================================
    
    def compute_loss(
        self,
        adj: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        l2_reg: float = 1e-5,
    ) -> dict:
        """Compute BPR + CL loss (called after UI matrix rebuilt)."""
        if self.image_UI_matrix is None or self.text_UI_matrix is None:
            # Fallback - no CL
            user_emb, pos_emb, neg_emb = self.forward(adj, users, pos_items, neg_items)
            scoreDiff = pairPredict(user_emb, pos_emb, neg_emb)
            bprLoss = -(scoreDiff).sigmoid().log().sum() / users.shape[0]
            regLoss = self.reg_loss() * l2_reg
            return {"loss": bprLoss + regLoss, "bpr_loss": bprLoss.item(), "reg_loss": regLoss.item(), "cl_loss": 0.0}
        
        # Main training - EXACT from Main.py line 257-292
        usrEmbeds, itmEmbeds = self.forward_MM(adj, self.image_UI_matrix, self.text_UI_matrix)
        ancEmbeds = usrEmbeds[users]
        posEmbeds = itmEmbeds[pos_items]
        negEmbeds = itmEmbeds[neg_items]
        
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = -(scoreDiff).sigmoid().log().sum() / users.shape[0]
        regLoss = self.reg_loss() * l2_reg
        
        # CL loss
        usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.forward_cl_MM(adj, self.image_UI_matrix, self.text_UI_matrix)
        
        clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, users, self.temp) + 
                  contrastLoss(itmEmbeds1, itmEmbeds2, pos_items, self.temp)) * self.ssl_reg
        
        clLoss1 = (contrastLoss(usrEmbeds, usrEmbeds1, users, self.temp) + 
                   contrastLoss(itmEmbeds, itmEmbeds1, pos_items, self.temp)) * self.ssl_reg
        clLoss2 = (contrastLoss(usrEmbeds, usrEmbeds2, users, self.temp) + 
                   contrastLoss(itmEmbeds, itmEmbeds2, pos_items, self.temp)) * self.ssl_reg
        clLoss_ = clLoss1 + clLoss2
        
        if self.cl_method == 1:
            clLoss = clLoss_
        # else: clLoss stays as modal-modal contrastive (already computed above)
        
        total_loss = bprLoss + regLoss + clLoss
        
        return {
            "loss": total_loss,
            "bpr_loss": bprLoss.item(),
            "reg_loss": regLoss.item(),
            "cl_loss": clLoss.item(),
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
        """Compute loss from pre-computed embeddings (for trainer compatibility)."""
        scoreDiff = pairPredict(user_emb, pos_emb, neg_emb)
        bprLoss = -(scoreDiff).sigmoid().log().sum() / users.shape[0]
        regLoss = self.reg_loss() * l2_reg
        
        cl_loss = cl_loss_precomputed if cl_loss_precomputed is not None else torch.tensor(0.0, device=user_emb.device)
        
        total_loss = bprLoss + regLoss + cl_loss
        
        return {
            "loss": total_loss,
            "bpr_loss": bprLoss.item(),
            "reg_loss": regLoss.item(),
            "cl_loss": cl_loss.item() if isinstance(cl_loss, torch.Tensor) else 0.0,
        }
    
    def compute_contrastive_loss(self, adj: torch.Tensor, users: torch.Tensor, pos_items: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss separately for trainer."""
        if self.image_UI_matrix is None or self.text_UI_matrix is None:
            return torch.tensor(0.0, device=self.device)
        
        usrEmbeds, itmEmbeds = self.forward_MM(adj, self.image_UI_matrix, self.text_UI_matrix)
        usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.forward_cl_MM(adj, self.image_UI_matrix, self.text_UI_matrix)
        
        clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, users, self.temp) + 
                  contrastLoss(itmEmbeds1, itmEmbeds2, pos_items, self.temp)) * self.ssl_reg
        
        clLoss1 = (contrastLoss(usrEmbeds, usrEmbeds1, users, self.temp) + 
                   contrastLoss(itmEmbeds, itmEmbeds1, pos_items, self.temp)) * self.ssl_reg
        clLoss2 = (contrastLoss(usrEmbeds, usrEmbeds2, users, self.temp) + 
                   contrastLoss(itmEmbeds, itmEmbeds2, pos_items, self.temp)) * self.ssl_reg
        clLoss_ = clLoss1 + clLoss2
        
        if self.cl_method == 1:
            return clLoss_
        else:
            return clLoss  # modal-modal only when cl_method == 0
    
    # =========================================================================
    # For compatibility
    # =========================================================================
    
    def inductive_forward(self, adj: torch.Tensor, users: torch.Tensor, items: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inductive forward for cold items."""
        if self.image_UI_matrix is not None and self.text_UI_matrix is not None:
            u_emb, i_emb = self.forward_MM(adj, self.image_UI_matrix, self.text_UI_matrix)
        else:
            u_emb, i_emb = self._simple_forward(adj)
        
        # Cold items: Use modal features
        cold_mask = items >= self.n_warm
        if cold_mask.any():
            cold_items = items[cold_mask]
            img_feats = self.getImageFeats()[cold_items]
            txt_feats = self.getTextFeats()[cold_items]
            modal_emb = 0.5 * img_feats + 0.5 * txt_feats
            
            i_emb = i_emb.clone()
            i_emb[cold_items] = F.normalize(modal_emb, p=2, dim=1)
        
        return u_emb[users], i_emb[items]
    
    def _get_all_embeddings(self, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all embeddings after propagation."""
        if self.image_UI_matrix is not None and self.text_UI_matrix is not None:
            return self.forward_MM(adj, self.image_UI_matrix, self.text_UI_matrix)
        else:
            return self._simple_forward(adj)
    
    # =========================================================================
    # Diffusion Training - EXACT from Main.py line 124-168
    # =========================================================================
    
    def train_diffusion_step(
        self,
        batch_item: torch.Tensor,  # User interaction row from trnMat
        batch_index: torch.Tensor,  # User index
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single diffusion training step.
        
        EXACT match to original Main.py line 124-159.
        
        Args:
            batch_item: User's interaction vector [batch, n_items]
            batch_index: User indices [batch]
            
        Returns:
            Tuple of (image_loss, text_loss)
        """
        # Detach embeddings from main model (no gradients flow back)
        iEmbeds = self.iEmbeds.detach()
        
        image_feats = self.getImageFeats().detach()
        text_feats = self.getTextFeats().detach()
        
        # Training losses for image denoiser
        diff_loss_image, gc_loss_image = self.diffusion_model.training_losses(
            self.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats
        )
        loss_image = diff_loss_image.mean() + gc_loss_image.mean() * self.e_loss
        
        # Training losses for text denoiser
        diff_loss_text, gc_loss_text = self.diffusion_model.training_losses(
            self.denoise_model_text, batch_item, iEmbeds, batch_index, text_feats
        )
        loss_text = diff_loss_text.mean() + gc_loss_text.mean() * self.e_loss
        
        return loss_image, loss_text
    
    @torch.no_grad()
    def rebuild_ui_matrices(
        self,
        diffusion_data: torch.Tensor,  # Full trnMat as dense tensor [n_users, n_items]
        batch_size: int = 1024,
    ) -> None:
        """
        Rebuild UI matrices using denoised samples.
        
        EXACT match to original Main.py line 173-235.
        
        Args:
            diffusion_data: Training matrix as dense tensor [n_users, n_items]
            batch_size: Batch size for processing
        """
        u_list_image = []
        i_list_image = []
        edge_list_image = []
        
        u_list_text = []
        i_list_text = []
        edge_list_text = []
        
        n_users = diffusion_data.shape[0]
        
        for start_idx in range(0, n_users, batch_size):
            end_idx = min(start_idx + batch_size, n_users)
            batch_item = diffusion_data[start_idx:end_idx].to(self.device)
            batch_index = torch.arange(start_idx, end_idx).to(self.device)
            
            # Image denoising
            denoised_batch = self.diffusion_model.p_sample(
                self.denoise_model_image, batch_item, self.sampling_steps, self.sampling_noise
            )
            top_val, indices_ = torch.topk(denoised_batch, k=self.rebuild_k)
            
            for i in range(batch_index.shape[0]):
                for j in range(indices_[i].shape[0]):
                    u_list_image.append(int(batch_index[i].cpu().numpy()))
                    i_list_image.append(int(indices_[i][j].cpu().numpy()))
                    edge_list_image.append(1.0)
            
            # Text denoising
            denoised_batch = self.diffusion_model.p_sample(
                self.denoise_model_text, batch_item, self.sampling_steps, self.sampling_noise
            )
            top_val, indices_ = torch.topk(denoised_batch, k=self.rebuild_k)
            
            for i in range(batch_index.shape[0]):
                for j in range(indices_[i].shape[0]):
                    u_list_text.append(int(batch_index[i].cpu().numpy()))
                    i_list_text.append(int(indices_[i][j].cpu().numpy()))
                    edge_list_text.append(1.0)
        
        # Build image UI matrix
        u_list_image = np.array(u_list_image)
        i_list_image = np.array(i_list_image)
        edge_list_image = np.array(edge_list_image)
        self.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
        self.image_UI_matrix = self.edgeDropper(self.image_UI_matrix)
        
        # Build text UI matrix
        u_list_text = np.array(u_list_text)
        i_list_text = np.array(i_list_text)
        edge_list_text = np.array(edge_list_text)
        self.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
        self.text_UI_matrix = self.edgeDropper(self.text_UI_matrix)
    
    def get_denoiser_parameters(self) -> list:
        """Get parameters for denoise models only (for separate optimizer)."""
        return list(self.denoise_model_image.parameters()) + list(self.denoise_model_text.parameters())
    
    def get_main_parameters(self) -> list:
        """Get parameters for main model only (excluding denoisers)."""
        denoiser_params = set(self.get_denoiser_parameters())
        return [p for p in self.parameters() if p not in denoiser_params]
