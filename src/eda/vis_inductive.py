"""
Inductive Gap Visualization.

Visualizes the alignment between Warm (ID + Modal) and Cold (Modal only) item embeddings
using t-SNE projection.

Usage:
    python -c "from src.eda.vis_inductive import plot_inductive_gap, load_and_plot; load_and_plot('beauty', 'micro')"
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


def plot_inductive_gap(
    model,
    dataset,
    save_path: str = "inductive_gap.png",
    max_items: int = 5000,
    perplexity: int = 30,
):
    """
    Visualizes the alignment between Warm (trained) and Cold (inductive) items.
    
    A good model should show Blue (cold) dots distributed ON TOP of Red (warm) dots.
    A bad model shows a distinct Blue island = "Modality Gap".
    
    Args:
        model: Trained recommendation model.
        dataset: RecDataset instance.
        save_path: Path to save the visualization.
        max_items: Maximum items to plot (for t-SNE speed).
        perplexity: t-SNE perplexity parameter.
    """
    model.eval()
    device = model.device
    
    logger.info(f"🎨 Generating inductive gap visualization...")
    logger.info(f"   Warm items: {model.n_warm}, Cold items: {model.n_cold}")
    
    with torch.no_grad():
        # 1. Get Warm Item Embeddings (ID + Modal)
        warm_indices = torch.arange(model.n_warm, device=device)
        warm_id_emb = model.item_embedding(warm_indices)
        warm_modal_emb = model.get_modal_embeddings(warm_indices)
        warm_emb = warm_id_emb + warm_modal_emb
        
        # 2. Get Cold Item Embeddings (Modal ONLY - no ID)
        cold_indices = torch.arange(model.n_warm, model.n_items, device=device)
        cold_emb = model.get_modal_embeddings(cold_indices)
        
        # 3. Compute alignment statistics
        # For warm items, how aligned are ID and modal embeddings?
        warm_alignment = F.cosine_similarity(warm_id_emb, warm_modal_emb).mean().item()
        logger.info(f"   Warm ID-Modal alignment (cosine): {warm_alignment:.4f}")
        
    # 4. Stack and create labels
    all_emb = torch.cat([warm_emb, cold_emb]).cpu().numpy()
    labels = np.array([0] * len(warm_indices) + [1] * len(cold_indices))
    
    # 5. Subsample if too many items
    n_total = len(all_emb)
    if n_total > max_items:
        logger.info(f"   Subsampling {n_total} → {max_items} items for t-SNE...")
        idx = np.random.choice(n_total, max_items, replace=False)
        all_emb = all_emb[idx]
        labels = labels[idx]
    
    # 6. Run t-SNE
    logger.info(f"   Running t-SNE (perplexity={perplexity})... this may take a minute")
    tsne = TSNE(
        n_components=2, 
        perplexity=perplexity, 
        n_iter=1000,
        random_state=42,
        init='pca',
    )
    embedded = tsne.fit_transform(all_emb)
    
    # 7. Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot warm items (red, smaller, more transparent)
    warm_mask = labels == 0
    ax.scatter(
        embedded[warm_mask, 0], embedded[warm_mask, 1],
        c='#FF6B6B', alpha=0.3, s=8, label=f'Warm (ID+Modal) n={warm_mask.sum()}'
    )
    
    # Plot cold items (blue, larger, more visible)
    cold_mask = labels == 1
    ax.scatter(
        embedded[cold_mask, 0], embedded[cold_mask, 1],
        c='#4ECDC4', alpha=0.6, s=20, label=f'Cold (Modal Only) n={cold_mask.sum()}'
    )
    
    # Calculate cluster overlap (simplified metric)
    warm_center = embedded[warm_mask].mean(axis=0)
    cold_center = embedded[cold_mask].mean(axis=0)
    center_dist = np.linalg.norm(warm_center - cold_center)
    
    ax.set_title(
        f"Inductive Gap Analysis\n"
        f"Warm-Cold Center Distance: {center_dist:.2f} (lower = better alignment)",
        fontsize=14
    )
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    # Add interpretation note
    if center_dist < 5:
        interpretation = "✅ Good: Cold items overlap with warm items"
        color = 'green'
    elif center_dist < 15:
        interpretation = "⚠️ Moderate: Some modality gap present"
        color = 'orange'
    else:
        interpretation = "❌ Bad: Significant modality gap detected"
        color = 'red'
    
    ax.text(
        0.02, 0.02, interpretation,
        transform=ax.transAxes, fontsize=11,
        color=color, fontweight='bold',
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"   ✓ Saved visualization to {save_path}")
    
    # Return statistics for programmatic use
    return {
        "warm_alignment": warm_alignment,
        "center_distance": center_dist,
        "n_warm_plotted": warm_mask.sum(),
        "n_cold_plotted": cold_mask.sum(),
    }


def load_and_plot(
    dataset_name: str,
    model_name: str,
    checkpoint_dir: str = "checkpoints",
    output_dir: str = "docs/images",
):
    """
    Load a trained model checkpoint and create visualization.
    
    Args:
        dataset_name: e.g., "beauty", "clothing", "electronics"
        model_name: e.g., "lattice", "micro", "diffmm"
        checkpoint_dir: Base checkpoint directory
        output_dir: Output directory for images
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.dataset import RecDataset
    from src.models import LATTICEModel, MICROModel, DiffMM
    from src.common.config import Config
    
    logging.basicConfig(level=logging.INFO)
    
    # Load dataset
    data_path = Path("data/processed") / dataset_name
    dataset = RecDataset(str(data_path), device="cuda")
    
    # Load model
    config = Config()
    
    model_classes = {
        "lattice": LATTICEModel,
        "micro": MICROModel,
        "diffmm": DiffMM,
    }
    
    ModelClass = model_classes[model_name]
    
    # Create model with same architecture
    common_args = dict(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        n_warm=dataset.n_warm,
        embed_dim=config.EMBED_DIM,
        n_layers=config.N_LAYERS,
        feat_visual=dataset.feat_visual,
        feat_text=dataset.feat_text,
        projection_hidden_dim=config.PROJECTION_HIDDEN_DIM,
        projection_dropout=config.PROJECTION_DROPOUT,
        device=config.DEVICE,
    )
    
    if model_name == "lattice":
        model = ModelClass(**common_args, k=config.LATTICE_K, graph_lambda=config.LATTICE_LAMBDA)
    elif model_name == "micro":
        model = ModelClass(**common_args, tau=config.MICRO_TAU, alpha=config.MICRO_ALPHA)
    else:
        model = ModelClass(
            **common_args,
            n_steps=config.DIFFMM_STEPS,
            noise_scale=config.DIFFMM_NOISE_SCALE,
            lambda_msi=config.DIFFMM_LAMBDA_MSI,
            mlp_width=config.DIFFMM_MLP_WIDTH,
        )
    
    # Load checkpoint
    checkpoint_path = Path(checkpoint_dir) / dataset_name / model_name / "best.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}, using random weights")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate visualization
    save_path = output_path / f"inductive_gap_{dataset_name}_{model_name}.png"
    stats = plot_inductive_gap(model, dataset, str(save_path))
    
    print(f"\n📊 Statistics:")
    print(f"   Warm-Modal Alignment: {stats['warm_alignment']:.4f}")
    print(f"   Center Distance: {stats['center_distance']:.2f}")
    
    return stats


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        load_and_plot(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python vis_inductive.py <dataset> <model>")
        print("Example: python vis_inductive.py beauty micro")
