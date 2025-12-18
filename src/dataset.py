"""
Shared dataset class for LATTICE/MICRO/DiffMM models.

Handles loading of preprocessed data and provides uniform interface for all models.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class RecDataset:
    """
    Recommendation dataset for multimodal models.
    
    Loads preprocessed interaction files and features, builds sparse
    adjacency matrix for graph-based models.
    """
    
    def __init__(
        self,
        data_dir: str,
        device: str = "cuda",
        load_features: bool = True,
    ):
        """
        Initialize dataset from preprocessed directory.
        
        Args:
            data_dir: Path to processed data directory.
            device: torch device for tensors.
            load_features: Whether to load feature files.
        """
        self.data_dir = Path(data_dir)
        self.device = device
        
        # Load mappings
        with open(self.data_dir / "maps.json") as f:
            self.maps = json.load(f)
        
        self.n_users = self.maps["n_users"]
        self.n_items = self.maps["n_items_total"]
        self.n_warm = self.maps["n_warm_items"]
        self.n_cold = self.maps["n_cold_items"]
        
        logger.info(f"Dataset: {self.n_users:,} users, {self.n_items:,} items "
                   f"({self.n_warm:,} warm, {self.n_cold:,} cold)")
        
        # Load interaction files
        self.train_data = self._load_interactions("train.txt")
        self.val_data = self._load_interactions("val.txt")
        self.test_warm_data = self._load_interactions("test_warm.txt")
        self.test_cold_data = self._load_interactions("test_cold.txt")
        
        logger.info(f"Interactions: train={len(self.train_data):,}, val={len(self.val_data):,}, "
                   f"test_warm={len(self.test_warm_data):,}, test_cold={len(self.test_cold_data):,}")
        
        # Build adjacency matrix from training data
        self.adj_matrix = self._build_adjacency()
        self.norm_adj = self._normalize_adjacency(self.adj_matrix)
        
        # Load features
        if load_features:
            self.feat_visual = self._load_features("feat_visual.npy")
            self.feat_text = self._load_features("feat_text.npy")
            logger.info(f"Features: visual={self.feat_visual.shape}, text={self.feat_text.shape}")
        else:
            self.feat_visual = None
            self.feat_text = None
        
        # Compute user degrees for Track 2 (robustness analysis)
        self.user_degrees = self._compute_user_degrees()
    
    def _load_interactions(self, filename: str) -> np.ndarray:
        """Load interaction file as numpy array."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return np.array([], dtype=np.int32).reshape(-1, 2)
        
        data = np.loadtxt(filepath, dtype=np.int32)
        if data.ndim == 1:
            data = data.reshape(-1, 2)
        return data
    
    def _load_features(self, filename: str) -> torch.Tensor:
        """Load feature file as torch tensor."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.warning(f"Feature file not found: {filepath}")
            return None
        
        arr = np.load(filepath)
        return torch.from_numpy(arr).float().to(self.device)
    
    def _build_adjacency(self) -> sparse.csr_matrix:
        """
        Build user-item bipartite adjacency matrix from training data.
        
        Matrix structure:
        [   0   , R ]
        [ R^T   , 0 ]
        
        Where R is the user-item interaction matrix.
        """
        n_total = self.n_users + self.n_items
        
        # Build user-item interaction matrix
        users = self.train_data[:, 0]
        items = self.train_data[:, 1] + self.n_users  # Shift item indices
        
        # Create bipartite adjacency
        row = np.concatenate([users, items])
        col = np.concatenate([items, users])
        data = np.ones(len(row), dtype=np.float32)
        
        adj = sparse.csr_matrix((data, (row, col)), shape=(n_total, n_total))
        
        logger.info(f"Adjacency matrix: {adj.shape}, nnz={adj.nnz:,}")
        
        return adj
    
    def _normalize_adjacency(self, adj: sparse.csr_matrix) -> torch.Tensor:
        """
        Normalize adjacency matrix for GCN: D^(-1/2) A D^(-1/2).
        
        Args:
            adj: Sparse adjacency matrix.
            
        Returns:
            Normalized adjacency as sparse torch tensor.
        """
        # Add self-loops
        adj = adj + sparse.eye(adj.shape[0], dtype=np.float32)
        
        # Compute D^(-1/2)
        degrees = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(degrees, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat = sparse.diags(d_inv_sqrt)
        
        # Symmetric normalization
        norm_adj = d_mat @ adj @ d_mat
        
        # Convert to sparse torch tensor (avoid slow list-to-tensor conversion)
        norm_adj = norm_adj.tocoo()
        indices = torch.LongTensor(np.array([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(np.array(norm_adj.data))
        
        return torch.sparse_coo_tensor(
            indices, values, norm_adj.shape
        ).to(self.device)
    
    def _compute_user_degrees(self) -> dict:
        """Compute degree (interaction count) for each user."""
        degree_count = {}
        for u, _ in self.train_data:
            degree_count[u] = degree_count.get(u, 0) + 1
        return degree_count
    
    def get_sparse_users(self, max_degree: int = 5) -> set:
        """Get users with <= max_degree interactions."""
        return {u for u, d in self.user_degrees.items() if d <= max_degree}
    
    def get_active_users(self, min_degree: int = 20) -> set:
        """Get users with >= min_degree interactions."""
        return {u for u, d in self.user_degrees.items() if d >= min_degree}
    
    def get_user_positive_items(self, split: str = "train") -> dict:
        """
        Get positive items for each user.
        
        Args:
            split: Which split to use ("train", "val", "test_warm", "test_cold").
            
        Returns:
            Dictionary mapping user_idx to set of positive item_idx.
        """
        data_map = {
            "train": self.train_data,
            "val": self.val_data,
            "test_warm": self.test_warm_data,
            "test_cold": self.test_cold_data,
        }
        
        data = data_map.get(split, self.train_data)
        
        user_items = {}
        for u, i in data:
            if u not in user_items:
                user_items[u] = set()
            user_items[u].add(i)
        
        return user_items


class BPRDataset(Dataset):
    """
    Dataset for BPR training with uniform negative sampling.
    
    Returns:
        - When n_negatives=1: (user, pos_item, neg_item) - all int64 scalars
        - When n_negatives>1: (user, pos_item, neg_items) where neg_items is (n_negatives,) array
    
    After batching by DataLoader:
        - users: (batch_size,) int64
        - pos_items: (batch_size,) int64
        - neg_items: (batch_size,) or (batch_size, n_negatives) int64
    """
    
    def __init__(
        self,
        interactions: np.ndarray,  # (N, 2) user-item pairs
        n_items: int,
        user_positive_items: dict,  # {user_idx: set(item_idx, ...)}
        n_negatives: int = 1,
    ):
        """
        Args:
            interactions: (N, 2) array of (user_idx, item_idx) pairs.
            n_items: Total number of items.
            user_positive_items: Dict mapping user to set of positive items.
            n_negatives: Number of negatives per positive (1-16 typical).
        """
        self.interactions = interactions
        self.n_items = n_items
        self.user_positive = user_positive_items
        self.n_negatives = n_negatives
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user, pos_item = self.interactions[idx]
        
        # Sample negative item(s) uniformly
        neg_items = []
        user_pos = self.user_positive.get(user, set())
        
        for _ in range(self.n_negatives):
            neg = np.random.randint(0, self.n_items)
            while neg in user_pos:
                neg = np.random.randint(0, self.n_items)
            neg_items.append(neg)
        
        # Always return numpy array for proper collation
        neg_items = np.array(neg_items, dtype=np.int64)
        
        if self.n_negatives == 1:
            return user, pos_item, neg_items[0]
        return user, pos_item, neg_items


def create_bpr_dataloader(
    dataset: RecDataset,
    batch_size: int = 1024,
    n_negatives: int = 1,
    num_workers: int = 8,
    shuffle: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> DataLoader:
    """
    Create DataLoader for BPR training.
    
    Optimized for CPU/GPU pipelining with parallel data loading.
    
    Args:
        dataset: RecDataset instance.
        batch_size: Batch size.
        n_negatives: Number of negatives per positive.
        num_workers: Number of worker processes (use P-cores).
        shuffle: Whether to shuffle data.
        pin_memory: Pin memory for faster CPU→GPU transfer.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Batches to prefetch per worker.
        
    Returns:
        DataLoader for training.
        
    Shapes:
        Each batch yields:
        - users: (batch_size,) int64
        - pos_items: (batch_size,) int64
        - neg_items: (batch_size,) or (batch_size, n_negatives) int64
    """
    user_positive = dataset.get_user_positive_items("train")
    
    bpr_dataset = BPRDataset(
        interactions=dataset.train_data,
        n_items=dataset.n_items,
        user_positive_items=user_positive,
        n_negatives=n_negatives,
    )
    
    # Only use persistent_workers and prefetch_factor if num_workers > 0
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    
    return DataLoader(bpr_dataset, **loader_kwargs)
