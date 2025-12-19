"""
Utility functions for EDA report generation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_json_file(file_path: Union[str, Path]) -> Dict:
    """Load a JSON file and return its contents as a dictionary."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_all_eda_results(docs_dir: Union[str, Path]) -> Dict[str, Dict]:
    """
    Load all EDA result JSON files from the docs directory.
    
    Args:
        docs_dir: Path to the docs directory containing EDA JSON files
        
    Returns:
        Dictionary mapping dataset names to their EDA results
    """
    docs_dir = Path(docs_dir)
    eda_files = list(docs_dir.glob("*_eda_results.json"))
    
    results = {}
    for file_path in eda_files:
        try:
            data = load_json_file(file_path)
            dataset_name = data.get('dataset', file_path.stem.replace('_eda_results', ''))
            results[dataset_name] = data
            logger.info(f"Loaded EDA results for: {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
    
    return results


def safe_get(data: Dict, *keys, default=None) -> Any:
    """
    Safely get nested dictionary values.
    
    Args:
        data: Dictionary to extract from
        *keys: Sequence of keys to traverse
        default: Default value if key not found
        
    Returns:
        Value at the nested key path or default
    """
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
            if current is None:
                return default
        else:
            return default
    return current


def format_number(num: Union[int, float], precision: int = 2) -> str:
    """Format a number with thousands separators and optional decimal places."""
    if isinstance(num, int) or (isinstance(num, float) and num.is_integer()):
        return f"{int(num):,}"
    return f"{num:,.{precision}f}"


def format_percentage(value: float, precision: int = 2) -> str:
    """Format a value as a percentage string."""
    return f"{value:.{precision}f}%"


def format_memory(mb: float) -> str:
    """Format memory size in appropriate units."""
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.2f} MB"


def get_color_palette() -> Dict[str, str]:
    """Return a consistent color palette for visualizations."""
    return {
        'beauty': '#FF6B6B',
        'clothing': '#4ECDC4',
        'electronics': '#45B7D1',
        'primary': '#2C3E50',
        'secondary': '#95A5A6',
        'success': '#27AE60',
        'warning': '#F39C12',
        'danger': '#E74C3C',
        'info': '#3498DB',
    }


def get_dataset_display_names() -> Dict[str, str]:
    """Return display names for datasets."""
    return {
        'beauty': 'Beauty & Personal Care',
        'clothing': 'Clothing, Shoes & Jewelry',
        'electronics': 'Electronics',
    }


def create_output_directories(output_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Create output directory structure for the report.
    
    Returns:
        Dictionary with paths to subdirectories
    """
    output_dir = Path(output_dir)
    
    subdirs = {
        'root': output_dir,
        'figures': output_dir / 'figures',
        'tables': output_dir / 'tables',
    }
    
    for name, path in subdirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
    
    return subdirs


def interpret_correlation(r: float) -> str:
    """Interpret correlation coefficient strength."""
    abs_r = abs(r)
    if abs_r < 0.1:
        return "negligible"
    elif abs_r < 0.3:
        return "weak"
    elif abs_r < 0.5:
        return "moderate"
    elif abs_r < 0.7:
        return "strong"
    else:
        return "very strong"


def interpret_sparsity(sparsity_pct: float) -> str:
    """Interpret sparsity level for recommendation systems."""
    if sparsity_pct > 99.99:
        return "Extremely sparse - challenging for collaborative filtering"
    elif sparsity_pct > 99.9:
        return "Very sparse - requires strong regularization"
    elif sparsity_pct > 99:
        return "Sparse - typical for recommendation datasets"
    else:
        return "Moderately sparse - good density for collaborative methods"


def get_status_indicator(status: str) -> Dict[str, str]:
    """Get HTML/CSS styling for status indicators."""
    status_styles = {
        'pass': {'color': '#27AE60', 'icon': '✓', 'text': 'PASS'},
        'fail': {'color': '#E74C3C', 'icon': '✗', 'text': 'FAIL'},
        'warning': {'color': '#F39C12', 'icon': '⚠', 'text': 'WARNING'},
        'info': {'color': '#3498DB', 'icon': 'ℹ', 'text': 'INFO'},
    }
    return status_styles.get(status.lower(), status_styles['info'])


def calculate_percentile_thresholds(values: List[float], percentiles: List[int] = None) -> Dict[str, float]:
    """Calculate percentile thresholds for a list of values."""
    import numpy as np
    
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90, 95, 99]
    
    values = np.array(values)
    return {f"p{p}": float(np.percentile(values, p)) for p in percentiles}


def timestamp_to_str(dt: datetime = None) -> str:
    """Convert datetime to formatted string."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def validate_eda_structure(data: Dict) -> Dict[str, bool]:
    """
    Validate that EDA results contain expected sections.
    
    Returns:
        Dictionary mapping section names to their presence
    """
    expected_sections = [
        'dataset',
        'display_name',
        'interactions_load_stats',
        'metadata_load_stats',
        'five_core_stats',
        'interaction_stats',
        'metadata_stats',
        'rating_distribution',
        'temporal_stats_monthly',
        'sparsity',
        'embedding_extraction',
        'modality_alignment',
        'visual_manifold',
        'bpr_hardness',
        'graph_connectivity',
        'feature_collapse',
        'text_embedding_extraction',
        'semantic_alignment',
        'cross_modal_consistency',
        'cca_alignment',
        'anisotropy',
        'user_consistency',
        'lattice_feasibility',
    ]
    
    return {section: section in data for section in expected_sections}
