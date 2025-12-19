"""
Metrics extraction module for EDA results analysis.

This module provides the MetricsExtractor class that extracts and organizes
metrics from EDA JSON results into structured DataFrames for analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .utils import safe_get, format_number, format_percentage, load_all_eda_results


class MetricsExtractor:
    """
    Extract and organize metrics from EDA results for analysis and visualization.
    """
    
    def __init__(self, eda_results: Dict[str, Dict]):
        """
        Initialize the MetricsExtractor with EDA results.
        
        Args:
            eda_results: Dictionary mapping dataset names to their EDA results
        """
        self.eda_results = eda_results
        self.dataset_names = list(eda_results.keys())
    
    @classmethod
    def from_directory(cls, docs_dir: str) -> 'MetricsExtractor':
        """Create MetricsExtractor by loading EDA results from a directory."""
        eda_results = load_all_eda_results(docs_dir)
        return cls(eda_results)
    
    # =========================================================================
    # Section 1: Dataset Overview & Scale
    # =========================================================================
    
    def get_dataset_overview(self) -> pd.DataFrame:
        """Extract basic dataset overview metrics."""
        rows = []
        for name, data in self.eda_results.items():
            rows.append({
                'Dataset': name,
                'Display Name': safe_get(data, 'display_name', default=name),
                'Data Format': safe_get(data, 'data_format', default='Unknown'),
                'Timestamp': safe_get(data, 'timestamp', default='N/A'),
                'Total Users': safe_get(data, 'interactions_load_stats', 'n_users', default=0),
                'Total Items': safe_get(data, 'interactions_load_stats', 'n_items', default=0),
                'Total Interactions': safe_get(data, 'interactions_load_stats', 'total_records', default=0),
                'Memory (MB)': safe_get(data, 'interactions_load_stats', 'memory_mb', default=0),
            })
        return pd.DataFrame(rows)
    
    def get_scale_comparison(self) -> pd.DataFrame:
        """Get scale comparison metrics for all datasets."""
        rows = []
        for name, data in self.eda_results.items():
            interaction_stats = safe_get(data, 'interaction_stats', default={})
            sparsity_data = safe_get(data, 'sparsity', default={})
            
            rows.append({
                'Dataset': name,
                'Users': safe_get(interaction_stats, 'counts', 'n_users', default=0),
                'Items': safe_get(interaction_stats, 'counts', 'n_items', default=0),
                'Interactions': safe_get(interaction_stats, 'counts', 'n_interactions', default=0),
                'Sparsity (%)': safe_get(sparsity_data, 'sparsity', default='N/A'),
                'Density (%)': safe_get(sparsity_data, 'density', default='N/A'),
                'Avg per User': safe_get(interaction_stats, 'density', 'avg_per_user', default=0),
                'Avg per Item': safe_get(interaction_stats, 'density', 'avg_per_item', default=0),
            })
        return pd.DataFrame(rows)
    
    def get_five_core_stats(self) -> pd.DataFrame:
        """Extract 5-core filtering statistics."""
        rows = []
        for name, data in self.eda_results.items():
            core_stats = safe_get(data, 'five_core_stats', default={})
            rows.append({
                'Dataset': name,
                'Min User Interactions': safe_get(core_stats, 'min_interactions_per_user', default=0),
                'Min Item Interactions': safe_get(core_stats, 'min_interactions_per_item', default=0),
                'Max User Interactions': safe_get(core_stats, 'max_interactions_per_user', default=0),
                'Max Item Interactions': safe_get(core_stats, 'max_interactions_per_item', default=0),
                'Avg User Interactions': safe_get(core_stats, 'avg_interactions_per_user', default=0),
                'Avg Item Interactions': safe_get(core_stats, 'avg_interactions_per_item', default=0),
                'Median User Interactions': safe_get(core_stats, 'median_interactions_per_user', default=0),
                'Median Item Interactions': safe_get(core_stats, 'median_interactions_per_item', default=0),
                'Is Valid': safe_get(core_stats, 'is_5_core_valid', default=False),
            })
        return pd.DataFrame(rows)
    
    # =========================================================================
    # Section 2: Rating Distribution
    # =========================================================================
    
    def get_rating_statistics(self) -> pd.DataFrame:
        """Extract rating statistics for all datasets."""
        rows = []
        for name, data in self.eda_results.items():
            ratings = safe_get(data, 'interaction_stats', 'ratings', default={})
            rows.append({
                'Dataset': name,
                'Mean Rating': safe_get(ratings, 'mean', default=0),
                'Std Rating': safe_get(ratings, 'std', default=0),
                'Median Rating': safe_get(ratings, 'median', default=0),
            })
        return pd.DataFrame(rows)
    
    def get_rating_distribution(self) -> pd.DataFrame:
        """Extract rating distribution for all datasets."""
        rows = []
        for name, data in self.eda_results.items():
            distribution = safe_get(data, 'rating_distribution', default=[])
            for entry in distribution:
                rows.append({
                    'Dataset': name,
                    'Rating': entry.get('rating', 0),
                    'Count': entry.get('count', 0),
                    'Percentage': entry.get('percentage', 0),
                })
        return pd.DataFrame(rows)
    
    def get_rating_distribution_wide(self) -> pd.DataFrame:
        """Get rating distribution in wide format for comparison."""
        df = self.get_rating_distribution()
        if df.empty:
            return df
        return df.pivot(index='Dataset', columns='Rating', values='Percentage').reset_index()
    
    # =========================================================================
    # Section 3: Temporal Analysis
    # =========================================================================
    
    def get_temporal_overview(self) -> pd.DataFrame:
        """Get temporal range overview for all datasets."""
        rows = []
        for name, data in self.eda_results.items():
            temporal = safe_get(data, 'interaction_stats', 'temporal', default={})
            rows.append({
                'Dataset': name,
                'Date Min': safe_get(temporal, 'date_min', default='N/A'),
                'Date Max': safe_get(temporal, 'date_max', default='N/A'),
                'Date Range (Days)': safe_get(temporal, 'date_range_days', default=0),
            })
        return pd.DataFrame(rows)
    
    def get_temporal_monthly(self, dataset: str) -> pd.DataFrame:
        """Get monthly temporal statistics for a specific dataset."""
        data = self.eda_results.get(dataset, {})
        monthly_stats = safe_get(data, 'temporal_stats_monthly', default=[])
        
        if not monthly_stats:
            return pd.DataFrame()
        
        df = pd.DataFrame(monthly_stats)
        df['Dataset'] = dataset
        return df
    
    def get_temporal_monthly_all(self) -> pd.DataFrame:
        """Get monthly temporal statistics for all datasets."""
        dfs = []
        for name in self.dataset_names:
            df = self.get_temporal_monthly(name)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)
    
    # =========================================================================
    # Section 4: User & Item Behavior
    # =========================================================================
    
    def get_user_item_patterns(self) -> pd.DataFrame:
        """Extract user and item behavior patterns."""
        rows = []
        for name, data in self.eda_results.items():
            patterns = safe_get(data, 'user_item_patterns', default={})
            rows.append({
                'Dataset': name,
                **patterns
            })
        return pd.DataFrame(rows)
    
    def get_tier_analysis(self) -> Dict[str, pd.DataFrame]:
        """Extract tier analysis for users and items."""
        result = {'users': [], 'items': []}
        
        for name, data in self.eda_results.items():
            tier_data = safe_get(data, 'tier_analysis', default={})
            
            for tier in safe_get(tier_data, 'users', default=[]):
                result['users'].append({'Dataset': name, **tier})
            
            for tier in safe_get(tier_data, 'items', default=[]):
                result['items'].append({'Dataset': name, **tier})
        
        return {
            'users': pd.DataFrame(result['users']),
            'items': pd.DataFrame(result['items'])
        }
    
    def get_user_segments(self) -> pd.DataFrame:
        """Extract user segmentation data."""
        rows = []
        for name, data in self.eda_results.items():
            segments = safe_get(data, 'user_segments', default=[])
            for segment in segments:
                rows.append({'Dataset': name, **segment})
        return pd.DataFrame(rows)
    
    def get_item_segments(self) -> pd.DataFrame:
        """Extract item segmentation data."""
        rows = []
        for name, data in self.eda_results.items():
            segments = safe_get(data, 'item_segments', default=[])
            for segment in segments:
                rows.append({'Dataset': name, **segment})
        return pd.DataFrame(rows)
    
    # =========================================================================
    # Section 5: Multimodal Coverage
    # =========================================================================
    
    def get_multimodal_coverage(self) -> pd.DataFrame:
        """Extract multimodal coverage statistics."""
        rows = []
        for name, data in self.eda_results.items():
            coverage = safe_get(data, 'multimodal_coverage', default={})
            text_coverage = safe_get(coverage, 'text', 'coverage', default={})
            visual_coverage = safe_get(coverage, 'visual', default={})
            
            rows.append({
                'Dataset': name,
                'Total Items': safe_get(coverage, 'n_items', default=0),
                'Title Coverage (%)': safe_get(text_coverage, 'title', default=0),
                'Description Coverage (%)': safe_get(text_coverage, 'description', default=0),
                'Features Coverage (%)': safe_get(text_coverage, 'features', default=0),
                'Image Coverage (%)': safe_get(visual_coverage, 'items_with_images_pct', default=0),
                'Avg Images per Item': safe_get(visual_coverage, 'avg_images_per_item', default=0),
            })
        return pd.DataFrame(rows)
    
    def get_metadata_coverage(self) -> pd.DataFrame:
        """Extract metadata coverage from metadata_stats."""
        rows = []
        for name, data in self.eda_results.items():
            meta = safe_get(data, 'metadata_stats', default={})
            text_cov = safe_get(meta, 'text_coverage', default={})
            images = safe_get(meta, 'images', default={})
            
            rows.append({
                'Dataset': name,
                'Items': safe_get(meta, 'counts', 'n_items', default=0),
                'With Title (%)': safe_get(text_cov, 'with_title_pct', default=0),
                'With Description (%)': safe_get(text_cov, 'with_description_pct', default=0),
                'With Features (%)': safe_get(text_cov, 'with_features_pct', default=0),
                'With Images (%)': safe_get(images, 'with_images_pct', default=0),
                'Avg Image Count': safe_get(images, 'avg_count', default=0),
            })
        return pd.DataFrame(rows)
    
    def get_text_lengths(self) -> pd.DataFrame:
        """Extract average text lengths for all datasets."""
        rows = []
        for name, data in self.eda_results.items():
            lengths = safe_get(data, 'metadata_stats', 'text_lengths', default={})
            rows.append({
                'Dataset': name,
                'Avg Title Length': safe_get(lengths, 'avg_title', default=0),
                'Avg Description Length': safe_get(lengths, 'avg_description', default=0),
            })
        return pd.DataFrame(rows)
    
    def get_feature_completeness(self) -> pd.DataFrame:
        """Extract feature completeness data."""
        rows = []
        for name, data in self.eda_results.items():
            features = safe_get(data, 'feature_completeness', default=[])
            for feature in features:
                rows.append({
                    'Dataset': name,
                    'Feature': feature.get('feature', 'unknown'),
                    'Coverage (%)': feature.get('coverage_pct', 0),
                })
        return pd.DataFrame(rows)
    
    # =========================================================================
    # Section 6: Embedding Quality
    # =========================================================================
    
    def get_visual_embedding_stats(self) -> pd.DataFrame:
        """Extract visual embedding extraction statistics."""
        rows = []
        for name, data in self.eda_results.items():
            emb = safe_get(data, 'embedding_extraction', default={})
            rows.append({
                'Dataset': name,
                'Items Attempted': safe_get(emb, 'n_items_attempted', default=0),
                'Items Successful': safe_get(emb, 'n_items_successful', default=0),
                'Success Rate (%)': safe_get(emb, 'success_rate', default=0),
                'Embedding Dim': safe_get(emb, 'embedding_dim', default=0),
                'Device': safe_get(emb, 'device_used', default='N/A'),
                'Processing Time (s)': safe_get(emb, 'processing_time_sec', default=0),
                'Items/Second': safe_get(emb, 'items_per_second', default=0),
            })
        return pd.DataFrame(rows)
    
    def get_text_embedding_stats(self) -> pd.DataFrame:
        """Extract text embedding extraction statistics."""
        rows = []
        for name, data in self.eda_results.items():
            emb = safe_get(data, 'text_embedding_extraction', default={})
            rows.append({
                'Dataset': name,
                'Items Attempted': safe_get(emb, 'n_items_attempted', default=0),
                'Items Successful': safe_get(emb, 'n_items_successful', default=0),
                'Success Rate (%)': safe_get(emb, 'success_rate', default=0),
                'Embedding Dim': safe_get(emb, 'embedding_dim', default=0),
                'Model': safe_get(emb, 'model_name', default='N/A'),
                'Device': safe_get(emb, 'device_used', default='N/A'),
                'Processing Time (s)': safe_get(emb, 'processing_time_sec', default=0),
                'Items/Second': safe_get(emb, 'items_per_second', default=0),
                'Avg Text Length': safe_get(emb, 'avg_text_length', default=0),
            })
        return pd.DataFrame(rows)
    
    def get_feature_collapse_analysis(self) -> pd.DataFrame:
        """Extract feature collapse analysis results."""
        rows = []
        for name, data in self.eda_results.items():
            collapse = safe_get(data, 'feature_collapse', default={})
            stats = safe_get(collapse, 'statistics', default={})
            dist = safe_get(collapse, 'distribution', default={})
            
            rows.append({
                'Dataset': name,
                'N Items': safe_get(collapse, 'n_items', default=0),
                'N Pairs Sampled': safe_get(collapse, 'n_pairs_sampled', default=0),
                'Mean Similarity': safe_get(stats, 'mean', default=0),
                'Std Similarity': safe_get(stats, 'std', default=0),
                'Min Similarity': safe_get(stats, 'min', default=0),
                'Max Similarity': safe_get(stats, 'max', default=0),
                'Very Low (%)': safe_get(dist, 'very_low_pct', default=0),
                'Low (%)': safe_get(dist, 'low_pct', default=0),
                'Medium (%)': safe_get(dist, 'medium_pct', default=0),
                'High (%)': safe_get(dist, 'high_pct', default=0),
                'Very High (%)': safe_get(dist, 'very_high_pct', default=0),
                'Is Pass': safe_get(collapse, 'is_pass', default=False),
                'Is Collapsed': safe_get(collapse, 'is_collapsed', default=False),
                'Interpretation': safe_get(collapse, 'interpretation', default='N/A'),
            })
        return pd.DataFrame(rows)
    
    def get_anisotropy_analysis(self) -> pd.DataFrame:
        """Extract anisotropy analysis results."""
        rows = []
        for name, data in self.eda_results.items():
            aniso = safe_get(data, 'anisotropy', default={})
            before = safe_get(aniso, 'before_centering', default={})
            after = safe_get(aniso, 'after_centering', default={})
            
            rows.append({
                'Dataset': name,
                'Before Centering (Avg Cosine)': safe_get(before, 'avg_cosine', default=0),
                'Before Centering (Std)': safe_get(before, 'std_cosine', default=0),
                'After Centering (Avg Cosine)': safe_get(after, 'avg_cosine', default=0),
                'After Centering (Std)': safe_get(after, 'std_cosine', default=0),
                'Is Anisotropic': safe_get(aniso, 'is_anisotropic', default=False),
                'Improvement Ratio': safe_get(aniso, 'improvement_ratio', default=0),
                'Interpretation': safe_get(aniso, 'interpretation', default='N/A'),
                'Recommendation': safe_get(aniso, 'recommendation', default='N/A'),
            })
        return pd.DataFrame(rows)
    
    # =========================================================================
    # Section 7: Modality Alignment
    # =========================================================================
    
    def get_visual_alignment(self) -> pd.DataFrame:
        """Extract visual-interaction alignment statistics."""
        rows = []
        for name, data in self.eda_results.items():
            align = safe_get(data, 'modality_alignment', default={})
            pearson = safe_get(align, 'pearson', default={})
            spearman = safe_get(align, 'spearman', default={})
            visual_sim = safe_get(align, 'visual_similarity', default={})
            interaction_sim = safe_get(align, 'interaction_similarity', default={})
            
            rows.append({
                'Dataset': name,
                'N Pairs Sampled': safe_get(align, 'n_pairs_sampled', default=0),
                'Pearson r': safe_get(pearson, 'correlation', default=0),
                'Pearson p-value': safe_get(pearson, 'pvalue', default=1),
                'Spearman r': safe_get(spearman, 'correlation', default=0),
                'Spearman p-value': safe_get(spearman, 'pvalue', default=1),
                'Visual Sim (Mean)': safe_get(visual_sim, 'mean', default=0),
                'Visual Sim (Std)': safe_get(visual_sim, 'std', default=0),
                'Interaction Sim (Mean)': safe_get(interaction_sim, 'mean', default=0),
                'Interaction Sim (Std)': safe_get(interaction_sim, 'std', default=0),
                'Interpretation': safe_get(align, 'interpretation', default='N/A'),
            })
        return pd.DataFrame(rows)
    
    def get_semantic_alignment(self) -> pd.DataFrame:
        """Extract text-interaction (semantic) alignment statistics."""
        rows = []
        for name, data in self.eda_results.items():
            align = safe_get(data, 'semantic_alignment', default={})
            pearson = safe_get(align, 'pearson', default={})
            spearman = safe_get(align, 'spearman', default={})
            stats = safe_get(align, 'statistics', default={})
            
            rows.append({
                'Dataset': name,
                'N Pairs Sampled': safe_get(align, 'n_pairs_sampled', default=0),
                'Pearson r': safe_get(pearson, 'correlation', default=0),
                'Pearson p-value': safe_get(pearson, 'pvalue', default=1),
                'Spearman r': safe_get(spearman, 'correlation', default=0),
                'Spearman p-value': safe_get(spearman, 'pvalue', default=1),
                'Text Sim (Mean)': safe_get(stats, 'mean_text_similarity', default=0),
                'Text Sim (Std)': safe_get(stats, 'std_text_similarity', default=0),
                'Signal Strength': safe_get(align, 'signal_strength', default='N/A'),
                'Interpretation': safe_get(align, 'interpretation', default='N/A'),
                'Recommendation': safe_get(align, 'recommendation', default='N/A'),
            })
        return pd.DataFrame(rows)
    
    def get_cross_modal_consistency(self) -> pd.DataFrame:
        """Extract cross-modal (text-image) consistency statistics."""
        rows = []
        for name, data in self.eda_results.items():
            cross = safe_get(data, 'cross_modal_consistency', default={})
            stats = safe_get(cross, 'statistics', default={})
            dist = safe_get(cross, 'distribution', default={})
            
            rows.append({
                'Dataset': name,
                'N Items Analyzed': safe_get(cross, 'n_items_analyzed', default=0),
                'N Items with Both': safe_get(cross, 'n_items_with_both', default=0),
                'Mean Agreement': safe_get(stats, 'mean', default=0),
                'Std Agreement': safe_get(stats, 'std', default=0),
                'Median Agreement': safe_get(stats, 'median', default=0),
                'Low Agreement (%)': safe_get(dist, 'low_agreement_pct', default=0),
                'Moderate Agreement (%)': safe_get(dist, 'moderate_agreement_pct', default=0),
                'High Agreement (%)': safe_get(dist, 'high_agreement_pct', default=0),
                'Alignment Status': safe_get(cross, 'alignment_status', default='N/A'),
                'Interpretation': safe_get(cross, 'interpretation', default='N/A'),
                'Recommendation': safe_get(cross, 'recommendation', default='N/A'),
            })
        return pd.DataFrame(rows)
    
    def get_cca_alignment(self) -> pd.DataFrame:
        """Extract CCA alignment statistics."""
        rows = []
        for name, data in self.eda_results.items():
            cca = safe_get(data, 'cca_alignment', default={})
            correlations = safe_get(cca, 'canonical_correlations', default=[])
            
            rows.append({
                'Dataset': name,
                'N Items': safe_get(cca, 'n_items', default=0),
                'N Components': safe_get(cca, 'n_components', default=0),
                'Mean Correlation': safe_get(cca, 'mean_correlation', default=0),
                'Top 3 Correlations': correlations[:3] if correlations else [],
                'Interpretation': safe_get(cca, 'interpretation', default='N/A'),
                'Recommendation': safe_get(cca, 'recommendation', default='N/A'),
            })
        return pd.DataFrame(rows)
    
    # =========================================================================
    # Section 8: Graph Analysis
    # =========================================================================
    
    def get_graph_connectivity(self) -> pd.DataFrame:
        """Extract graph connectivity statistics."""
        rows = []
        for name, data in self.eda_results.items():
            graph = safe_get(data, 'graph_connectivity', default={})
            
            rows.append({
                'Dataset': name,
                'N Items': safe_get(graph, 'n_items', default=0),
                'K Neighbors': safe_get(graph, 'k_neighbors', default=0),
                'N Components': safe_get(graph, 'n_components', default=0),
                'Giant Component Size': safe_get(graph, 'giant_component_size', default=0),
                'Giant Component (%)': safe_get(graph, 'giant_component_coverage_pct', default=0),
                'Avg Component Size': safe_get(graph, 'avg_component_size', default=0),
                'Pass Threshold': safe_get(graph, 'pass_threshold', default=50),
                'Is Pass': safe_get(graph, 'is_pass', default=False),
                'Interpretation': safe_get(graph, 'interpretation', default='N/A'),
            })
        return pd.DataFrame(rows)
    
    def get_kcore_analysis(self) -> pd.DataFrame:
        """Extract k-core analysis results."""
        rows = []
        for name, data in self.eda_results.items():
            kcore = safe_get(data, 'kcore_analysis', default={})
            
            for k, stats in kcore.items():
                if isinstance(stats, dict):
                    rows.append({
                        'Dataset': name,
                        'K': safe_get(stats, 'k', default=int(k)),
                        'N Users': safe_get(stats, 'n_users', default=0),
                        'N Items': safe_get(stats, 'n_items', default=0),
                        'N Interactions': safe_get(stats, 'n_interactions', default=0),
                        'User Retention (%)': safe_get(stats, 'user_retention_pct', default=0),
                        'Item Retention (%)': safe_get(stats, 'item_retention_pct', default=0),
                        'Interaction Retention (%)': safe_get(stats, 'interaction_retention_pct', default=0),
                        'Density After': safe_get(stats, 'density_after', default='N/A'),
                    })
        return pd.DataFrame(rows)
    
    # =========================================================================
    # Section 9: BPR Hardness
    # =========================================================================
    
    def get_bpr_hardness(self) -> pd.DataFrame:
        """Extract BPR negative sampling hardness statistics."""
        rows = []
        for name, data in self.eda_results.items():
            bpr = safe_get(data, 'bpr_hardness', default={})
            dist = safe_get(bpr, 'visual_distance', default={})
            hardness = safe_get(bpr, 'hardness_distribution', default={})
            
            rows.append({
                'Dataset': name,
                'N Users Analyzed': safe_get(bpr, 'n_users_analyzed', default=0),
                'N Pairs Analyzed': safe_get(bpr, 'n_pairs_analyzed', default=0),
                'Mean Visual Distance': safe_get(dist, 'mean', default=0),
                'Median Visual Distance': safe_get(dist, 'median', default=0),
                'Std Visual Distance': safe_get(dist, 'std', default=0),
                'Easy (%)': safe_get(hardness, 'easy_pct', default=0),
                'Medium (%)': safe_get(hardness, 'medium_pct', default=0),
                'Hard (%)': safe_get(hardness, 'hard_pct', default=0),
                'Interpretation': safe_get(bpr, 'interpretation', default='N/A'),
                'Recommendation': safe_get(bpr, 'recommendation', default='N/A'),
            })
        return pd.DataFrame(rows)
    
    # =========================================================================
    # Section 10: User Consistency
    # =========================================================================
    
    def get_user_consistency(self) -> pd.DataFrame:
        """Extract user visual consistency statistics."""
        rows = []
        for name, data in self.eda_results.items():
            uc = safe_get(data, 'user_consistency', default={})
            distances = safe_get(uc, 'distances', default={})
            
            rows.append({
                'Dataset': name,
                'N Users Analyzed': safe_get(uc, 'n_users_analyzed', default=0),
                'N Users with Enough Items': safe_get(uc, 'n_users_with_enough_items', default=0),
                'Mean Local Distance': safe_get(distances, 'mean_local', default=0),
                'Std Local Distance': safe_get(distances, 'std_local', default=0),
                'Mean Global Distance': safe_get(distances, 'mean_global', default=0),
                'Std Global Distance': safe_get(distances, 'std_global', default=0),
                'Consistency Ratio': safe_get(uc, 'consistency_ratio', default=0),
                'Is Consistent': safe_get(uc, 'is_consistent', default=False),
                'Users with Visual Coherence (%)': safe_get(uc, 'users_with_visual_coherence_pct', default=0),
                'Interpretation': safe_get(uc, 'interpretation', default='N/A'),
                'Recommendation': safe_get(uc, 'recommendation', default='N/A'),
            })
        return pd.DataFrame(rows)
    
    # =========================================================================
    # Section 11: Model Feasibility
    # =========================================================================
    
    def get_lattice_feasibility(self) -> pd.DataFrame:
        """Extract LATTICE model feasibility assessment."""
        rows = []
        for name, data in self.eda_results.items():
            feas = safe_get(data, 'lattice_feasibility', default={})
            checks = safe_get(feas, 'checks', default={})
            
            rows.append({
                'Dataset': name,
                'Alignment Pass': safe_get(checks, 'alignment', 'pass', default=False),
                'Alignment Value': safe_get(checks, 'alignment', 'value', default=0),
                'Connectivity Pass': safe_get(checks, 'connectivity', 'pass', default=False),
                'Connectivity Value': safe_get(checks, 'connectivity', 'value', default=0),
                'Collapse Pass': safe_get(checks, 'collapse', 'pass', default=False),
                'Collapse Value': safe_get(checks, 'collapse', 'value', default=0),
                'Decision': safe_get(feas, 'decision', default='N/A'),
            })
        return pd.DataFrame(rows)
    
    # =========================================================================
    # Sparsity Analysis
    # =========================================================================
    
    def get_sparsity_analysis(self) -> pd.DataFrame:
        """Extract sparsity analysis results."""
        rows = []
        for name, data in self.eda_results.items():
            sparsity = safe_get(data, 'sparsity', default={})
            
            rows.append({
                'Dataset': name,
                'N Users': safe_get(sparsity, 'n_users', default=0),
                'N Items': safe_get(sparsity, 'n_items', default=0),
                'N Interactions': safe_get(sparsity, 'n_interactions', default=0),
                'Possible Interactions': safe_get(sparsity, 'possible_interactions', default=0),
                'Density': safe_get(sparsity, 'density', default='N/A'),
                'Sparsity': safe_get(sparsity, 'sparsity', default='N/A'),
            })
        return pd.DataFrame(rows)
    
    # =========================================================================
    # Visual Manifold
    # =========================================================================
    
    def get_visual_manifold(self) -> pd.DataFrame:
        """Extract visual manifold analysis results."""
        rows = []
        for name, data in self.eda_results.items():
            manifold = safe_get(data, 'visual_manifold', default={})
            quality = safe_get(manifold, 'quality', default={})
            
            rows.append({
                'Dataset': name,
                'N Items': safe_get(manifold, 'n_items', default=0),
                'Method': safe_get(manifold, 'method', default='N/A'),
                'Silhouette Score': safe_get(quality, 'silhouette_score_category', default=0),
                'N Unique Categories': safe_get(quality, 'n_unique_categories', default=0),
                'Interpretation': safe_get(manifold, 'interpretation', default='N/A'),
            })
        return pd.DataFrame(rows)
    
    # =========================================================================
    # Summary & Recommendations
    # =========================================================================
    
    def get_all_recommendations(self) -> pd.DataFrame:
        """Collect all recommendations from various analyses."""
        rows = []
        
        for name, data in self.eda_results.items():
            # Collect recommendations from different sections
            sections = [
                ('BPR Hardness', safe_get(data, 'bpr_hardness', 'recommendation', default='')),
                ('Semantic Alignment', safe_get(data, 'semantic_alignment', 'recommendation', default='')),
                ('Cross-Modal', safe_get(data, 'cross_modal_consistency', 'recommendation', default='')),
                ('CCA Alignment', safe_get(data, 'cca_alignment', 'recommendation', default='')),
                ('Anisotropy', safe_get(data, 'anisotropy', 'recommendation', default='')),
                ('User Consistency', safe_get(data, 'user_consistency', 'recommendation', default='')),
            ]
            
            for section, recommendation in sections:
                if recommendation:
                    rows.append({
                        'Dataset': name,
                        'Section': section,
                        'Recommendation': recommendation,
                    })
        
        return pd.DataFrame(rows)
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Generate a comprehensive summary of all key statistics."""
        rows = []
        
        for name, data in self.eda_results.items():
            # Collect key metrics from various sections
            row = {
                'Dataset': name,
                'Users': safe_get(data, 'interactions_load_stats', 'n_users', default=0),
                'Items': safe_get(data, 'interactions_load_stats', 'n_items', default=0),
                'Interactions': safe_get(data, 'interactions_load_stats', 'total_records', default=0),
                'Sparsity': safe_get(data, 'sparsity', 'sparsity', default='N/A'),
                'Mean Rating': safe_get(data, 'interaction_stats', 'ratings', 'mean', default=0),
                'Visual Alignment (r)': safe_get(data, 'modality_alignment', 'pearson', 'correlation', default=0),
                'Text Alignment (r)': safe_get(data, 'semantic_alignment', 'pearson', 'correlation', default=0),
                'Cross-Modal Agreement': safe_get(data, 'cross_modal_consistency', 'statistics', 'mean', default=0),
                'CCA Correlation': safe_get(data, 'cca_alignment', 'mean_correlation', default=0),
                'Feature Collapse (Avg Sim)': safe_get(data, 'feature_collapse', 'statistics', 'mean', default=0),
                'Anisotropic': safe_get(data, 'anisotropy', 'is_anisotropic', default=False),
                'Graph Connected (%)': safe_get(data, 'graph_connectivity', 'giant_component_coverage_pct', default=0),
                'LATTICE Decision': safe_get(data, 'lattice_feasibility', 'decision', default='N/A'),
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
