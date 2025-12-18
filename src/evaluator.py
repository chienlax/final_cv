"""
Evaluator for multimodal recommendation models.

Computes Recall@K, NDCG@K, Precision@K for all-ranking evaluation.
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_metrics(
    scores: torch.Tensor,
    ground_truth: dict,
    train_positive: dict,
    k_list: list = [10, 20, 50],
) -> dict:
    """
    Compute ranking metrics for a batch of users.
    
    Args:
        scores: (n_users, n_items) prediction scores.
        ground_truth: Dict mapping user_idx to set of ground truth items.
        train_positive: Dict mapping user_idx to set of training items (to exclude).
        k_list: List of K values for metrics.
        
    Returns:
        Dict of metric_name -> value.
    """
    device = scores.device
    n_users = scores.shape[0]
    n_items = scores.shape[1]
    
    # Mask training items (set to -inf)
    for i, user in enumerate(ground_truth.keys()):
        if user in train_positive:
            train_items = list(train_positive[user])
            scores[i, train_items] = -float('inf')
    
    # Get top-K items
    max_k = max(k_list)
    _, topk_indices = torch.topk(scores, max_k, dim=1)
    topk_indices = topk_indices.cpu().numpy()
    
    # Convert ground truth to list for indexing
    users = list(ground_truth.keys())
    
    # Initialize metrics
    metrics = {f"recall@{k}": 0.0 for k in k_list}
    metrics.update({f"ndcg@{k}": 0.0 for k in k_list})
    metrics.update({f"precision@{k}": 0.0 for k in k_list})
    
    n_valid = 0
    
    for i, user in enumerate(users):
        gt_items = ground_truth.get(user, set())
        if len(gt_items) == 0:
            continue
        
        n_valid += 1
        
        for k in k_list:
            topk = topk_indices[i, :k]
            
            # Hits
            hits = sum(1 for item in topk if item in gt_items)
            
            # Recall@K
            metrics[f"recall@{k}"] += hits / len(gt_items)
            
            # Precision@K
            metrics[f"precision@{k}"] += hits / k
            
            # NDCG@K
            dcg = 0.0
            for j, item in enumerate(topk):
                if item in gt_items:
                    dcg += 1.0 / np.log2(j + 2)
            
            idcg = sum(1.0 / np.log2(j + 2) for j in range(min(len(gt_items), k)))
            
            if idcg > 0:
                metrics[f"ndcg@{k}"] += dcg / idcg
    
    # Average
    if n_valid > 0:
        for key in metrics:
            metrics[key] /= n_valid
    
    metrics["n_users"] = n_valid
    
    return metrics


@torch.no_grad()
def evaluate(
    model,
    dataset,
    split: str = "test_warm",
    k_list: list = [10, 20, 50],
    batch_size: int = 256,
    filter_users: Optional[set] = None,
    inductive: bool = False,
) -> dict:
    """
    Evaluate model on a dataset split.
    
    Args:
        model: Trained model.
        dataset: RecDataset instance.
        split: Which split to evaluate ("val", "test_warm", "test_cold").
        k_list: List of K values for metrics.
        batch_size: Evaluation batch size.
        filter_users: Optional set of users to include (for Track 2).
        inductive: If True, use inductive mode (for cold items).
        
    Returns:
        Dict of metric_name -> value.
    """
    model.eval()
    
    # Get ground truth
    gt_data = dataset.get_user_positive_items(split)
    train_pos = dataset.get_user_positive_items("train")
    
    # Filter users if specified
    if filter_users is not None:
        gt_data = {u: items for u, items in gt_data.items() if u in filter_users}
    
    if len(gt_data) == 0:
        logger.warning(f"No users to evaluate for split={split}")
        return {}
    
    users = list(gt_data.keys())
    all_metrics = {f"recall@{k}": [] for k in k_list}
    all_metrics.update({f"ndcg@{k}": [] for k in k_list})
    all_metrics.update({f"precision@{k}": [] for k in k_list})
    
    # Determine item range for scoring
    if split == "test_cold" or inductive:
        # Score only cold items
        item_range = range(dataset.n_warm, dataset.n_items)
    else:
        # Score only warm items
        item_range = range(dataset.n_warm)
    
    item_indices = torch.tensor(list(item_range), device=model.device)
    n_items_to_score = len(item_indices)
    
    # Get all embeddings
    adj = dataset.norm_adj
    
    if inductive:
        # For cold items, we need to handle differently
        all_user_emb, all_item_emb = model._get_all_embeddings(adj)
        
        # Generate cold item embeddings
        cold_modal = model.get_modal_embeddings(item_indices)
        if hasattr(model, 'sample_from_noise'):
            # DiffMM
            cold_item_emb = model.sample_from_noise(cold_modal)
        else:
            # LATTICE/MICRO - use modal embeddings directly
            cold_item_emb = cold_modal
    else:
        all_user_emb, all_item_emb = model._get_all_embeddings(adj)
        cold_item_emb = all_item_emb[item_indices]
    
    # Batch evaluation
    total_metrics = {key: 0.0 for key in all_metrics}
    n_total = 0
    
    for batch_start in range(0, len(users), batch_size):
        batch_end = min(batch_start + batch_size, len(users))
        batch_users = users[batch_start:batch_end]
        
        user_tensor = torch.tensor(batch_users, device=model.device)
        user_emb = all_user_emb[user_tensor]
        
        # Compute scores
        scores = user_emb @ cold_item_emb.T  # (batch, n_items_to_score)
        
        # Adjust item indices for ground truth comparison
        batch_gt = {u: gt_data[u] for u in batch_users}
        batch_train = {u: train_pos.get(u, set()) for u in batch_users}
        
        # Remap ground truth to local indices (offset by start of item_range)
        item_offset = item_range.start if hasattr(item_range, 'start') else 0
        
        remapped_gt = {}
        for i, u in enumerate(batch_users):
            items = batch_gt[u]
            remapped = {item - item_offset for item in items if item_offset <= item < item_offset + n_items_to_score}
            if remapped:
                remapped_gt[i] = remapped
        
        remapped_train = {}
        for i, u in enumerate(batch_users):
            items = batch_train[u]
            remapped = {item - item_offset for item in items if item_offset <= item < item_offset + n_items_to_score}
            if remapped:
                remapped_train[i] = remapped
        
        if not remapped_gt:
            continue
        
        batch_metrics = compute_metrics(scores, remapped_gt, remapped_train, k_list)
        
        n_batch = batch_metrics.get("n_users", 0)
        if n_batch > 0:
            for key in total_metrics:
                total_metrics[key] += batch_metrics.get(key, 0) * n_batch
            n_total += n_batch
    
    # Average
    if n_total > 0:
        for key in total_metrics:
            total_metrics[key] /= n_total
    
    total_metrics["n_users"] = n_total
    total_metrics["split"] = split
    
    return total_metrics


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """Format metrics for logging."""
    lines = []
    
    if prefix:
        lines.append(f"=== {prefix} ===")
    
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)
