"""
EDA Report Generator - Main Script

This module provides the EDAReportGenerator class that orchestrates the
extraction of metrics, creation of visualizations, and generation of
comprehensive HTML reports from EDA results.

Usage:
    python -m src.analysis.eda_report_generator --input-dir docs/ --output-dir outputs/eda_reports/
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from jinja2 import Environment, FileSystemLoader

from .utils import (
    load_all_eda_results, 
    create_output_directories, 
    format_number,
    get_dataset_display_names,
    validate_eda_structure,
    timestamp_to_str
)
from .metrics import MetricsExtractor
from .visualizations import EDAVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EDAReportGenerator:
    """
    Generate comprehensive HTML reports from EDA results.
    
    This class orchestrates:
    1. Loading EDA JSON results
    2. Extracting metrics using MetricsExtractor
    3. Creating visualizations using EDAVisualizer
    4. Rendering the HTML report using Jinja2 templates
    """
    
    def __init__(self, input_dir: str, output_dir: str = None):
        """
        Initialize the report generator.
        
        Args:
            input_dir: Directory containing EDA JSON files
            output_dir: Directory for output report and figures
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else Path('outputs/eda_reports')
        
        # Load EDA results
        logger.info(f"Loading EDA results from: {self.input_dir}")
        self.eda_results = load_all_eda_results(self.input_dir)
        
        if not self.eda_results:
            raise ValueError(f"No EDA results found in {self.input_dir}")
        
        logger.info(f"Loaded {len(self.eda_results)} datasets: {list(self.eda_results.keys())}")
        
        # Initialize extractors and visualizers
        self.metrics = MetricsExtractor(self.eda_results)
        self.visualizer = EDAVisualizer()
        
        # Setup output directories
        self.output_dirs = create_output_directories(self.output_dir)
        
        # Setup Jinja2 environment
        template_dir = Path(__file__).parent / 'templates'
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        self.jinja_env.filters['format_number'] = format_number
        
        # Store generated figures and tables
        self.figures: Dict[str, Any] = {}
        self.tables: Dict[str, str] = {}
    
    def validate_data(self) -> Dict[str, Dict[str, bool]]:
        """Validate EDA data structure for all datasets."""
        validation_results = {}
        for name, data in self.eda_results.items():
            validation_results[name] = validate_eda_structure(data)
        return validation_results
    
    def _df_to_html(self, df, **kwargs) -> str:
        """Convert DataFrame to styled HTML table."""
        default_classes = 'table table-striped'
        classes = kwargs.pop('classes', default_classes)
        return df.to_html(
            index=False,
            classes=classes,
            border=0,
            **kwargs
        )
    
    def _fig_to_html(self, fig, include_plotlyjs: bool = False) -> str:
        """Convert Plotly figure to HTML div."""
        return fig.to_html(
            full_html=False,
            include_plotlyjs='cdn' if include_plotlyjs else False,
            div_id=None
        )
    
    # =========================================================================
    # Section Generators
    # =========================================================================
    
    def generate_overview_section(self) -> Dict[str, str]:
        """Generate dataset overview section content."""
        logger.info("Generating overview section...")
        
        # Extract data
        df_overview = self.metrics.get_dataset_overview()
        df_scale = self.metrics.get_scale_comparison()
        df_five_core = self.metrics.get_five_core_stats()
        df_sparsity = self.metrics.get_sparsity_analysis()
        
        # Create figures
        scale_fig = self.visualizer.plot_dataset_scale_comparison(df_overview)
        sparsity_fig = self.visualizer.plot_sparsity_comparison(df_sparsity)
        memory_fig = self.visualizer.plot_memory_usage(df_overview)
        
        return {
            'scale_comparison_chart': self._fig_to_html(scale_fig, include_plotlyjs=True),
            'sparsity_chart': self._fig_to_html(sparsity_fig),
            'memory_chart': self._fig_to_html(memory_fig),
            'five_core_table': self._df_to_html(df_five_core),
        }
    
    def generate_rating_section(self) -> Dict[str, str]:
        """Generate rating distribution section content."""
        logger.info("Generating rating section...")
        
        # Extract data
        df_stats = self.metrics.get_rating_statistics()
        df_dist = self.metrics.get_rating_distribution()
        df_wide = self.metrics.get_rating_distribution_wide()
        
        # Create figures
        dist_fig = self.visualizer.plot_rating_distribution(df_dist)
        stats_fig = self.visualizer.plot_rating_statistics(df_stats)
        heatmap_fig = self.visualizer.plot_rating_heatmap(df_wide) if not df_wide.empty else None
        
        result = {
            'rating_distribution_chart': self._fig_to_html(dist_fig),
            'rating_stats_chart': self._fig_to_html(stats_fig),
            'rating_stats_table': self._df_to_html(df_stats),
        }
        
        if heatmap_fig:
            result['rating_heatmap'] = self._fig_to_html(heatmap_fig)
        else:
            result['rating_heatmap'] = ''
        
        return result
    
    def generate_temporal_section(self) -> Dict[str, str]:
        """Generate temporal analysis section content."""
        logger.info("Generating temporal section...")
        
        # Extract data
        df_overview = self.metrics.get_temporal_overview()
        df_monthly = self.metrics.get_temporal_monthly_all()
        
        # Create figures
        overview_fig = self.visualizer.plot_temporal_overview(df_overview)
        trends_fig = self.visualizer.plot_temporal_trends(df_monthly)
        
        return {
            'temporal_overview_chart': self._fig_to_html(overview_fig),
            'temporal_trends_chart': self._fig_to_html(trends_fig),
            'temporal_table': self._df_to_html(df_overview),
        }
    
    def generate_behavior_section(self) -> Dict[str, str]:
        """Generate user/item behavior section content."""
        logger.info("Generating behavior section...")
        
        # Extract data
        df_five_core = self.metrics.get_five_core_stats()
        tier_data = self.metrics.get_tier_analysis()
        
        # Create figures
        five_core_fig = self.visualizer.plot_five_core_stats(df_five_core)
        tier_fig = self.visualizer.plot_tier_analysis(
            tier_data['users'], 
            tier_data['items']
        )
        
        return {
            'five_core_stats_chart': self._fig_to_html(five_core_fig),
            'tier_analysis_chart': self._fig_to_html(tier_fig),
        }
    
    def generate_multimodal_section(self) -> Dict[str, str]:
        """Generate multimodal coverage section content."""
        logger.info("Generating multimodal section...")
        
        # Extract data
        df_coverage = self.metrics.get_metadata_coverage()
        df_completeness = self.metrics.get_feature_completeness()
        df_lengths = self.metrics.get_text_lengths()
        
        # Create figures
        coverage_fig = self.visualizer.plot_modality_coverage(df_coverage)
        completeness_fig = self.visualizer.plot_feature_completeness_heatmap(df_completeness)
        lengths_fig = self.visualizer.plot_text_lengths(df_lengths)
        
        return {
            'modality_coverage_chart': self._fig_to_html(coverage_fig),
            'feature_completeness_chart': self._fig_to_html(completeness_fig),
            'text_lengths_chart': self._fig_to_html(lengths_fig),
            'coverage_table': self._df_to_html(df_coverage),
        }
    
    def generate_embedding_section(self) -> Dict[str, str]:
        """Generate embedding quality section content."""
        logger.info("Generating embedding section...")
        
        # Extract data
        df_visual = self.metrics.get_visual_embedding_stats()
        df_text = self.metrics.get_text_embedding_stats()
        df_collapse = self.metrics.get_feature_collapse_analysis()
        df_anisotropy = self.metrics.get_anisotropy_analysis()
        
        # Create figures
        stats_fig = self.visualizer.plot_embedding_extraction_stats(df_visual, df_text)
        collapse_fig = self.visualizer.plot_feature_collapse(df_collapse)
        anisotropy_fig = self.visualizer.plot_anisotropy_comparison(df_anisotropy)
        
        # Prepare table with selected columns
        collapse_cols = ['Dataset', 'Mean Similarity', 'Is Pass', 'Interpretation']
        collapse_table = df_collapse[[c for c in collapse_cols if c in df_collapse.columns]]
        
        anisotropy_cols = ['Dataset', 'Before Centering (Avg Cosine)', 
                          'After Centering (Avg Cosine)', 'Is Anisotropic', 'Recommendation']
        anisotropy_table = df_anisotropy[[c for c in anisotropy_cols if c in df_anisotropy.columns]]
        
        return {
            'embedding_stats_chart': self._fig_to_html(stats_fig),
            'feature_collapse_chart': self._fig_to_html(collapse_fig),
            'anisotropy_chart': self._fig_to_html(anisotropy_fig),
            'collapse_table': self._df_to_html(collapse_table),
            'anisotropy_table': self._df_to_html(anisotropy_table),
        }
    
    def generate_alignment_section(self) -> Dict[str, str]:
        """Generate modality alignment section content."""
        logger.info("Generating alignment section...")
        
        # Extract data
        df_visual = self.metrics.get_visual_alignment()
        df_semantic = self.metrics.get_semantic_alignment()
        df_cross = self.metrics.get_cross_modal_consistency()
        df_cca = self.metrics.get_cca_alignment()
        
        # Create figures
        alignment_fig = self.visualizer.plot_alignment_comparison(df_visual, df_semantic)
        cross_fig = self.visualizer.plot_cross_modal_consistency(df_cross)
        cca_fig = self.visualizer.plot_cca_correlations(df_cca)
        
        # Prepare tables
        alignment_cols = ['Dataset', 'Pearson r', 'Interpretation']
        visual_table = df_visual[[c for c in alignment_cols if c in df_visual.columns]]
        
        cross_cols = ['Dataset', 'Mean Agreement', 'Alignment Status', 'Recommendation']
        cross_table = df_cross[[c for c in cross_cols if c in df_cross.columns]]
        
        return {
            'alignment_comparison_chart': self._fig_to_html(alignment_fig),
            'cross_modal_chart': self._fig_to_html(cross_fig),
            'cca_chart': self._fig_to_html(cca_fig),
            'alignment_table': self._df_to_html(visual_table),
            'cross_modal_table': self._df_to_html(cross_table),
        }
    
    def generate_graph_section(self) -> Dict[str, str]:
        """Generate graph analysis section content."""
        logger.info("Generating graph section...")
        
        # Extract data
        df_connectivity = self.metrics.get_graph_connectivity()
        df_kcore = self.metrics.get_kcore_analysis()
        
        # Create figures
        connectivity_fig = self.visualizer.plot_graph_connectivity(df_connectivity)
        kcore_fig = self.visualizer.plot_kcore_retention(df_kcore)
        
        # Prepare table
        graph_cols = ['Dataset', 'N Components', 'Giant Component (%)', 'Is Pass', 'Interpretation']
        graph_table = df_connectivity[[c for c in graph_cols if c in df_connectivity.columns]]
        
        return {
            'graph_connectivity_chart': self._fig_to_html(connectivity_fig),
            'kcore_chart': self._fig_to_html(kcore_fig),
            'graph_table': self._df_to_html(graph_table),
        }
    
    def generate_bpr_section(self) -> Dict[str, str]:
        """Generate BPR hardness section content."""
        logger.info("Generating BPR section...")
        
        # Extract data
        df_bpr = self.metrics.get_bpr_hardness()
        
        # Create figure
        bpr_fig = self.visualizer.plot_bpr_hardness(df_bpr)
        
        # Prepare table
        bpr_cols = ['Dataset', 'Mean Visual Distance', 'Easy (%)', 'Medium (%)', 
                   'Hard (%)', 'Recommendation']
        bpr_table = df_bpr[[c for c in bpr_cols if c in df_bpr.columns]]
        
        return {
            'bpr_hardness_chart': self._fig_to_html(bpr_fig),
            'bpr_table': self._df_to_html(bpr_table),
        }
    
    def generate_consistency_section(self) -> Dict[str, str]:
        """Generate user consistency section content."""
        logger.info("Generating consistency section...")
        
        # Extract data
        df_consistency = self.metrics.get_user_consistency()
        
        # Create figure
        consistency_fig = self.visualizer.plot_user_consistency(df_consistency)
        
        # Prepare table
        cons_cols = ['Dataset', 'Mean Local Distance', 'Mean Global Distance',
                    'Consistency Ratio', 'Is Consistent', 'Recommendation']
        cons_table = df_consistency[[c for c in cons_cols if c in df_consistency.columns]]
        
        return {
            'user_consistency_chart': self._fig_to_html(consistency_fig),
            'consistency_table': self._df_to_html(cons_table),
        }
    
    def generate_feasibility_section(self) -> Dict[str, str]:
        """Generate model feasibility section content."""
        logger.info("Generating feasibility section...")
        
        # Extract data
        df_feasibility = self.metrics.get_lattice_feasibility()
        
        # Create figures
        radar_fig = self.visualizer.plot_lattice_feasibility(df_feasibility)
        summary_fig = self.visualizer.plot_feasibility_summary(df_feasibility)
        
        return {
            'feasibility_radar': self._fig_to_html(radar_fig),
            'feasibility_summary': self._fig_to_html(summary_fig),
            'feasibility_table': self._df_to_html(df_feasibility),
        }
    
    def generate_recommendations_section(self) -> Dict[str, str]:
        """Generate recommendations section content."""
        logger.info("Generating recommendations section...")
        
        # Extract data
        df_recommendations = self.metrics.get_all_recommendations()
        
        # Create figure
        if not df_recommendations.empty:
            rec_fig = self.visualizer.plot_recommendations_summary(df_recommendations)
            return {
                'recommendations_chart': self._fig_to_html(rec_fig),
            }
        
        return {'recommendations_chart': ''}
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate executive summary data."""
        logger.info("Generating summary...")
        
        df_summary = self.metrics.get_summary_statistics()
        
        # Prepare dataset info for template
        datasets = []
        display_names = get_dataset_display_names()
        
        for name, data in self.eda_results.items():
            datasets.append({
                'name': name,
                'display_name': display_names.get(name, name),
                'interactions': data.get('interactions_load_stats', {}).get('total_records', 0),
            })
        
        return {
            'datasets': datasets,
            'summary_table': self._df_to_html(df_summary),
        }
    
    # =========================================================================
    # Main Generation Methods
    # =========================================================================
    
    def generate_all_sections(self) -> Dict[str, str]:
        """Generate all report sections."""
        sections = {}
        
        # Generate each section
        sections.update(self.generate_overview_section())
        sections.update(self.generate_rating_section())
        sections.update(self.generate_temporal_section())
        sections.update(self.generate_behavior_section())
        sections.update(self.generate_multimodal_section())
        sections.update(self.generate_embedding_section())
        sections.update(self.generate_alignment_section())
        sections.update(self.generate_graph_section())
        sections.update(self.generate_bpr_section())
        sections.update(self.generate_consistency_section())
        sections.update(self.generate_feasibility_section())
        sections.update(self.generate_recommendations_section())
        
        return sections
    
    def generate_report(self, output_filename: str = 'eda_report.html') -> Path:
        """
        Generate the complete HTML report.
        
        Args:
            output_filename: Name of the output HTML file
            
        Returns:
            Path to the generated report
        """
        logger.info("Starting report generation...")
        
        # Generate all sections
        sections = self.generate_all_sections()
        
        # Generate summary
        summary = self.generate_summary()
        
        # Prepare template context
        context = {
            'generation_date': timestamp_to_str(),
            'dataset_names': list(self.eda_results.keys()),
            **summary,
            **sections,
        }
        
        # Render template
        template = self.jinja_env.get_template('report_template.html')
        html_content = template.render(**context)
        
        # Write output
        output_path = self.output_dir / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report generated: {output_path}")
        
        return output_path
    
    def export_figures(self, format: str = 'png', dpi: int = 150) -> List[Path]:
        """
        Export all figures as individual files.
        
        Args:
            format: Output format ('png', 'svg', 'pdf')
            dpi: Resolution for raster formats
            
        Returns:
            List of paths to exported figures
        """
        logger.info(f"Exporting figures as {format}...")
        
        exported = []
        figures_dir = self.output_dirs['figures']
        
        # Generate figures for each section
        figure_generators = [
            ('01_scale_comparison', lambda: self.visualizer.plot_dataset_scale_comparison(
                self.metrics.get_dataset_overview())),
            ('02_rating_distribution', lambda: self.visualizer.plot_rating_distribution(
                self.metrics.get_rating_distribution())),
            ('03_temporal_trends', lambda: self.visualizer.plot_temporal_trends(
                self.metrics.get_temporal_monthly_all())),
            ('04_modality_coverage', lambda: self.visualizer.plot_modality_coverage(
                self.metrics.get_metadata_coverage())),
            ('05_feature_collapse', lambda: self.visualizer.plot_feature_collapse(
                self.metrics.get_feature_collapse_analysis())),
            ('06_anisotropy', lambda: self.visualizer.plot_anisotropy_comparison(
                self.metrics.get_anisotropy_analysis())),
            ('07_alignment', lambda: self.visualizer.plot_alignment_comparison(
                self.metrics.get_visual_alignment(),
                self.metrics.get_semantic_alignment())),
            ('08_graph_connectivity', lambda: self.visualizer.plot_graph_connectivity(
                self.metrics.get_graph_connectivity())),
            ('09_bpr_hardness', lambda: self.visualizer.plot_bpr_hardness(
                self.metrics.get_bpr_hardness())),
            ('10_user_consistency', lambda: self.visualizer.plot_user_consistency(
                self.metrics.get_user_consistency())),
            ('11_feasibility', lambda: self.visualizer.plot_lattice_feasibility(
                self.metrics.get_lattice_feasibility())),
        ]
        
        for name, generator in figure_generators:
            try:
                fig = generator()
                filepath = figures_dir / f"{name}.{format}"
                
                if format == 'html':
                    fig.write_html(str(filepath))
                else:
                    fig.write_image(str(filepath), format=format, scale=2)
                
                exported.append(filepath)
                logger.info(f"  Exported: {filepath}")
            except Exception as e:
                logger.error(f"  Failed to export {name}: {e}")
        
        return exported
    
    def export_tables(self, format: str = 'csv') -> List[Path]:
        """
        Export all data tables.
        
        Args:
            format: Output format ('csv', 'excel')
            
        Returns:
            List of paths to exported tables
        """
        logger.info(f"Exporting tables as {format}...")
        
        exported = []
        tables_dir = self.output_dirs['tables']
        
        # Define tables to export
        tables = {
            'dataset_overview': self.metrics.get_dataset_overview(),
            'scale_comparison': self.metrics.get_scale_comparison(),
            'five_core_stats': self.metrics.get_five_core_stats(),
            'rating_statistics': self.metrics.get_rating_statistics(),
            'rating_distribution': self.metrics.get_rating_distribution(),
            'temporal_overview': self.metrics.get_temporal_overview(),
            'metadata_coverage': self.metrics.get_metadata_coverage(),
            'feature_completeness': self.metrics.get_feature_completeness(),
            'visual_embedding_stats': self.metrics.get_visual_embedding_stats(),
            'text_embedding_stats': self.metrics.get_text_embedding_stats(),
            'feature_collapse': self.metrics.get_feature_collapse_analysis(),
            'anisotropy': self.metrics.get_anisotropy_analysis(),
            'visual_alignment': self.metrics.get_visual_alignment(),
            'semantic_alignment': self.metrics.get_semantic_alignment(),
            'cross_modal_consistency': self.metrics.get_cross_modal_consistency(),
            'cca_alignment': self.metrics.get_cca_alignment(),
            'graph_connectivity': self.metrics.get_graph_connectivity(),
            'kcore_analysis': self.metrics.get_kcore_analysis(),
            'bpr_hardness': self.metrics.get_bpr_hardness(),
            'user_consistency': self.metrics.get_user_consistency(),
            'lattice_feasibility': self.metrics.get_lattice_feasibility(),
            'sparsity_analysis': self.metrics.get_sparsity_analysis(),
            'summary_statistics': self.metrics.get_summary_statistics(),
            'recommendations': self.metrics.get_all_recommendations(),
        }
        
        for name, df in tables.items():
            if df.empty:
                continue
            
            try:
                if format == 'csv':
                    filepath = tables_dir / f"{name}.csv"
                    df.to_csv(filepath, index=False)
                elif format == 'excel':
                    filepath = tables_dir / f"{name}.xlsx"
                    df.to_excel(filepath, index=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                exported.append(filepath)
                logger.info(f"  Exported: {filepath}")
            except Exception as e:
                logger.error(f"  Failed to export {name}: {e}")
        
        return exported
    
    def export_json_summary(self) -> Path:
        """Export a JSON summary of all metrics."""
        logger.info("Exporting JSON summary...")
        
        summary = {
            'generation_date': timestamp_to_str(),
            'datasets': list(self.eda_results.keys()),
            'summary_statistics': self.metrics.get_summary_statistics().to_dict('records'),
            'recommendations': self.metrics.get_all_recommendations().to_dict('records'),
            'feasibility': self.metrics.get_lattice_feasibility().to_dict('records'),
        }
        
        filepath = self.output_dir / 'eda_summary.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"JSON summary exported: {filepath}")
        return filepath


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description='Generate EDA analysis reports from JSON results'
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        default='docs',
        help='Directory containing EDA JSON files (default: docs)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='outputs/eda_reports',
        help='Output directory for reports (default: outputs/eda_reports)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['html', 'all'],
        default='html',
        help='Output format (default: html)'
    )
    
    parser.add_argument(
        '--export-figures',
        action='store_true',
        help='Export individual figures as PNG'
    )
    
    parser.add_argument(
        '--export-tables',
        action='store_true',
        help='Export data tables as CSV'
    )
    
    parser.add_argument(
        '--export-json',
        action='store_true',
        help='Export JSON summary'
    )
    
    args = parser.parse_args()
    
    # Generate report
    generator = EDAReportGenerator(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Generate HTML report
    report_path = generator.generate_report()
    print(f"\n✅ HTML Report generated: {report_path}")
    
    # Optional exports
    if args.export_figures or args.format == 'all':
        figures = generator.export_figures(format='png')
        print(f"✅ Exported {len(figures)} figures")
    
    if args.export_tables or args.format == 'all':
        tables = generator.export_tables(format='csv')
        print(f"✅ Exported {len(tables)} tables")
    
    if args.export_json or args.format == 'all':
        json_path = generator.export_json_summary()
        print(f"✅ JSON summary exported: {json_path}")
    
    print(f"\n📁 All outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
