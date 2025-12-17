"""
Graph Connectivity Analysis for LATTICE Feasibility.

Implements the "LATTICE Feasibility Check" from Liu et al. (2024):
- Build k-NN graph from visual embeddings
- Measure connected components
- Determine if giant component covers sufficient items

Pass Criterion: Giant component coverage > 50%
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class ConnectivityResult:
    """Results from graph connectivity analysis."""
    
    n_items: int = 0
    k_neighbors: int = 5
    
    # Connected components
    n_components: int = 0
    giant_component_size: int = 0
    giant_component_coverage_pct: float = 0.0
    
    # Component size distribution
    avg_component_size: float = 0.0
    median_component_size: float = 0.0
    component_sizes: list[int] = field(default_factory=list)
    
    # Pass/Fail
    pass_threshold: float = 50.0  # Percentage
    is_pass: bool = False
    interpretation: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_items": self.n_items,
            "k_neighbors": self.k_neighbors,
            "n_components": self.n_components,
            "giant_component_size": self.giant_component_size,
            "giant_component_coverage_pct": round(self.giant_component_coverage_pct, 2),
            "avg_component_size": round(self.avg_component_size, 2),
            "median_component_size": round(self.median_component_size, 2),
            "top_10_component_sizes": self.component_sizes[:10],
            "pass_threshold": self.pass_threshold,
            "is_pass": self.is_pass,
            "interpretation": self.interpretation,
        }


def analyze_graph_connectivity(
    embeddings: np.ndarray,
    item_indices: list[str],
    k: int = 5,
    similarity_threshold: Optional[float] = None,
    pass_threshold: float = 50.0,
) -> ConnectivityResult:
    """
    Analyze graph connectivity for LATTICE feasibility.
    
    Builds a k-NN graph from visual embeddings and measures connected components.
    LATTICE requires a well-connected graph for message passing to work.
    
    Args:
        embeddings: Visual embeddings matrix (n_items, embedding_dim).
        item_indices: List of item IDs corresponding to embeddings.
        k: Number of nearest neighbors for k-NN graph.
        similarity_threshold: Optional minimum cosine similarity for edges.
        pass_threshold: Minimum giant component coverage % to pass (default 50%).
        
    Returns:
        ConnectivityResult with analysis results.
    """
    logger.info(f"Analyzing graph connectivity (k={k})...")
    
    result = ConnectivityResult(
        n_items=len(item_indices),
        k_neighbors=k,
        pass_threshold=pass_threshold,
    )
    
    if len(embeddings) < 2:
        result.interpretation = "Insufficient items for connectivity analysis"
        return result
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized_embeddings = embeddings / norms
    
    # Build k-NN graph
    logger.info(f"  Building k-NN graph with k={k}...")
    
    # Use more neighbors than k, then potentially filter by similarity
    effective_k = min(k + 1, len(embeddings))  # +1 because item is its own neighbor
    
    nn = NearestNeighbors(n_neighbors=effective_k, metric='cosine', algorithm='brute')
    nn.fit(normalized_embeddings)
    distances, indices = nn.kneighbors(normalized_embeddings)
    
    # Build sparse adjacency matrix
    n_items = len(embeddings)
    row_indices = []
    col_indices = []
    
    for i in range(n_items):
        for j_idx in range(1, effective_k):  # Skip self (index 0)
            j = indices[i, j_idx]
            
            # Optional similarity threshold filtering
            if similarity_threshold is not None:
                # Distance is 1 - cosine_similarity for cosine metric
                similarity = 1 - distances[i, j_idx]
                if similarity < similarity_threshold:
                    continue
            
            row_indices.append(i)
            col_indices.append(j)
    
    # Create symmetric adjacency matrix
    data = np.ones(len(row_indices))
    adjacency = csr_matrix((data, (row_indices, col_indices)), shape=(n_items, n_items))
    adjacency = adjacency + adjacency.T  # Make symmetric
    adjacency.data = np.ones_like(adjacency.data)  # Binary adjacency
    
    # Find connected components
    logger.info("  Computing connected components...")
    n_components, labels = connected_components(adjacency, directed=False)
    
    # Analyze component sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    component_sizes = sorted(counts.tolist(), reverse=True)
    
    result.n_components = n_components
    result.component_sizes = component_sizes
    result.giant_component_size = component_sizes[0] if component_sizes else 0
    result.giant_component_coverage_pct = (result.giant_component_size / n_items * 100) if n_items > 0 else 0.0
    result.avg_component_size = np.mean(component_sizes) if component_sizes else 0.0
    result.median_component_size = np.median(component_sizes) if component_sizes else 0.0
    
    # Pass/Fail determination
    result.is_pass = result.giant_component_coverage_pct >= pass_threshold
    
    # Generate interpretation
    if result.is_pass:
        result.interpretation = (
            f"PASS: Giant component covers {result.giant_component_coverage_pct:.1f}% of items "
            f"(threshold: {pass_threshold}%). Graph is sufficiently connected for LATTICE."
        )
    else:
        result.interpretation = (
            f"FAIL: Giant component covers only {result.giant_component_coverage_pct:.1f}% of items "
            f"(threshold: {pass_threshold}%). Graph is fragmented into {n_components} components. "
            "LATTICE message passing will fail. Consider: (1) using different visual encoder, "
            "(2) lowering similarity threshold, or (3) using denser k-NN."
        )
    
    logger.info(f"  Components: {n_components}, Giant: {result.giant_component_size} ({result.giant_component_coverage_pct:.1f}%)")
    logger.info(f"  Result: {'PASS' if result.is_pass else 'FAIL'}")
    
    return result


def compute_graph_density(
    embeddings: np.ndarray,
    k: int = 5,
) -> dict:
    """
    Compute additional graph density metrics.
    
    Args:
        embeddings: Visual embeddings matrix.
        k: Number of neighbors.
        
    Returns:
        Dictionary with density metrics.
    """
    n_items = len(embeddings)
    
    # For k-NN graph: each node has exactly k edges (ignoring self-loops)
    n_edges = n_items * k
    max_edges = n_items * (n_items - 1)  # Directed graph max edges
    
    density = n_edges / max_edges if max_edges > 0 else 0
    avg_degree = k  # By definition of k-NN
    
    return {
        "n_nodes": n_items,
        "n_edges": n_edges,
        "density": round(density, 6),
        "avg_degree": avg_degree,
    }
