"""
Visualization module for EDA results analysis.

This module provides the EDAVisualizer class with comprehensive plotting
functions using Plotly for interactive visualizations.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .utils import get_color_palette, get_dataset_display_names


class EDAVisualizer:
    """
    Create interactive visualizations for EDA results analysis.
    """
    
    def __init__(self, color_palette: Dict[str, str] = None):
        """
        Initialize the visualizer with a color palette.
        
        Args:
            color_palette: Dictionary mapping dataset names to colors
        """
        self.colors = color_palette or get_color_palette()
        self.display_names = get_dataset_display_names()
        
        # Default plot settings
        self.default_height = 500
        self.default_width = 900
        self.template = 'plotly_white'
    
    def _get_dataset_color(self, dataset: str) -> str:
        """Get the color for a dataset."""
        return self.colors.get(dataset, self.colors.get('primary', '#2C3E50'))
    
    def _get_dataset_colors(self, datasets: List[str]) -> List[str]:
        """Get colors for a list of datasets."""
        return [self._get_dataset_color(d) for d in datasets]
    
    # =========================================================================
    # Section 1: Dataset Overview & Scale
    # =========================================================================
    
    def plot_dataset_scale_comparison(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a grouped bar chart comparing dataset scales.
        
        Args:
            df: DataFrame with Dataset, Total Users, Total Items, Total Interactions columns
        """
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Users', 'Items', 'Interactions'),
            shared_yaxes=False
        )
        
        colors = self._get_dataset_colors(df['Dataset'].tolist())
        
        # Map display names to actual column names
        col_mapping = {
            'Users': 'Total Users',
            'Items': 'Total Items', 
            'Interactions': 'Total Interactions'
        }
        
        for i, (display_name, col) in enumerate(col_mapping.items(), 1):
            if col in df.columns:
                fig.add_trace(
                    go.Bar(
                        x=df['Dataset'],
                        y=df[col],
                        marker_color=colors,
                        text=df[col].apply(lambda x: f'{x:,.0f}'),
                        textposition='outside',
                        name=display_name,
                        showlegend=False
                    ),
                    row=1, col=i
                )
        
        fig.update_layout(
            title='Dataset Scale Comparison',
            height=450,
            template=self.template,
            bargap=0.3,
        )
        
        return fig
    
    def plot_sparsity_comparison(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a bar chart comparing sparsity across datasets.
        
        Args:
            df: DataFrame with Dataset and Sparsity columns
        """
        # Parse sparsity percentage from string
        df = df.copy()
        df['Sparsity_Pct'] = df['Sparsity'].str.replace('%', '').astype(float)
        
        fig = go.Figure()
        
        # Show sparsity as bars
        fig.add_trace(go.Bar(
            x=df['Dataset'],
            y=df['Sparsity_Pct'],
            marker_color=self._get_dataset_colors(df['Dataset'].tolist()),
            text=df['Sparsity'],
            textposition='outside',
        ))
        
        fig.update_layout(
            title='Dataset Sparsity Comparison',
            yaxis_title='Sparsity (%)',
            yaxis_range=[99.99, 100],
            height=400,
            template=self.template,
        )
        
        return fig
    
    def plot_memory_usage(self, df: pd.DataFrame) -> go.Figure:
        """Create a bar chart showing memory usage by dataset."""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['Dataset'],
            y=df['Memory (MB)'],
            marker_color=self._get_dataset_colors(df['Dataset'].tolist()),
            text=df['Memory (MB)'].apply(lambda x: f'{x:,.0f} MB'),
            textposition='outside',
        ))
        
        fig.update_layout(
            title='Memory Usage by Dataset',
            yaxis_title='Memory (MB)',
            height=400,
            template=self.template,
        )
        
        return fig
    
    # =========================================================================
    # Section 2: Rating Distribution
    # =========================================================================
    
    def plot_rating_distribution(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a grouped bar chart of rating distributions.
        
        Args:
            df: DataFrame with Dataset, Rating, Count, Percentage columns
        """
        fig = px.bar(
            df,
            x='Rating',
            y='Percentage',
            color='Dataset',
            barmode='group',
            color_discrete_map={d: self._get_dataset_color(d) for d in df['Dataset'].unique()},
            title='Rating Distribution by Dataset',
            labels={'Percentage': 'Percentage (%)', 'Rating': 'Rating (Stars)'},
        )
        
        fig.update_layout(
            height=450,
            template=self.template,
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        )
        
        return fig
    
    def plot_rating_statistics(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a box plot showing rating statistics comparison.
        
        Args:
            df: DataFrame with Dataset, Mean Rating, Std Rating columns
        """
        fig = go.Figure()
        
        for _, row in df.iterrows():
            dataset = row['Dataset']
            mean = row['Mean Rating']
            std = row['Std Rating']
            
            fig.add_trace(go.Box(
                y=[mean - std, mean - std/2, mean, mean + std/2, mean + std],
                name=dataset,
                marker_color=self._get_dataset_color(dataset),
                boxpoints=False,
            ))
        
        fig.update_layout(
            title='Rating Statistics by Dataset',
            yaxis_title='Rating',
            height=400,
            template=self.template,
        )
        
        return fig
    
    def plot_rating_heatmap(self, df_wide: pd.DataFrame) -> go.Figure:
        """
        Create a heatmap of rating distributions.
        
        Args:
            df_wide: Wide-format DataFrame with datasets as rows and ratings as columns
        """
        datasets = df_wide['Dataset'].tolist()
        ratings = [col for col in df_wide.columns if col != 'Dataset']
        values = df_wide[ratings].values
        
        fig = go.Figure(data=go.Heatmap(
            z=values,
            x=[f'{r} ★' for r in ratings],
            y=datasets,
            colorscale='RdYlGn',
            text=np.round(values, 1),
            texttemplate='%{text}%',
            textfont={'size': 12},
            hoverongaps=False,
        ))
        
        fig.update_layout(
            title='Rating Distribution Heatmap',
            height=350,
            template=self.template,
        )
        
        return fig
    
    # =========================================================================
    # Section 3: Temporal Analysis
    # =========================================================================
    
    def plot_temporal_trends(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a line plot of interaction trends over time.
        
        Args:
            df: DataFrame with year_month, n_interactions, Dataset columns
        """
        if df.empty or 'year_month' not in df.columns:
            return self._empty_figure("No temporal data available")
        
        # Determine the y-column (handle different column names)
        y_col = 'n_interactions' if 'n_interactions' in df.columns else 'count'
        if y_col not in df.columns:
            return self._empty_figure("No interaction count column found")
        
        fig = px.line(
            df,
            x='year_month',
            y=y_col,
            color='Dataset',
            color_discrete_map={d: self._get_dataset_color(d) for d in df['Dataset'].unique()},
            title='Monthly Interaction Trends',
            labels={'year_month': 'Date', y_col: 'Interactions'},
        )
        
        fig.update_layout(
            height=450,
            template=self.template,
            xaxis_title='Month',
            yaxis_title='Number of Interactions',
        )
        
        return fig
    
    def plot_rating_trends(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a line plot of average rating trends over time.
        """
        if df.empty or 'year_month' not in df.columns:
            return self._empty_figure("No temporal data available")
        
        fig = px.line(
            df,
            x='year_month',
            y='avg_rating',
            color='Dataset',
            color_discrete_map={d: self._get_dataset_color(d) for d in df['Dataset'].unique()},
            title='Average Rating Trends Over Time',
            labels={'year_month': 'Date', 'avg_rating': 'Average Rating'},
        )
        
        fig.update_layout(
            height=450,
            template=self.template,
            yaxis_range=[3.5, 5],
        )
        
        return fig
    
    def plot_temporal_overview(self, df: pd.DataFrame) -> go.Figure:
        """Create a timeline visualization of dataset date ranges."""
        fig = go.Figure()
        
        for i, row in df.iterrows():
            dataset = row['Dataset']
            date_min = row['Date Min']
            date_max = row['Date Max']
            days = row['Date Range (Days)']
            
            fig.add_trace(go.Bar(
                y=[dataset],
                x=[days],
                orientation='h',
                marker_color=self._get_dataset_color(dataset),
                text=f"{date_min} → {date_max}",
                textposition='inside',
                name=dataset,
                showlegend=False,
            ))
        
        fig.update_layout(
            title='Dataset Temporal Coverage',
            xaxis_title='Days of Data',
            height=300,
            template=self.template,
            bargap=0.4,
        )
        
        return fig
    
    # =========================================================================
    # Section 4: User & Item Behavior
    # =========================================================================
    
    def plot_tier_analysis(self, df_users: pd.DataFrame, df_items: pd.DataFrame) -> go.Figure:
        """
        Create tier analysis visualization showing concentration of interactions.
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('User Tier Analysis', 'Item Tier Analysis'),
        )
        
        datasets = df_users['Dataset'].unique() if not df_users.empty else []
        
        for dataset in datasets:
            color = self._get_dataset_color(dataset)
            
            # User tiers
            user_data = df_users[df_users['Dataset'] == dataset]
            if not user_data.empty and 'tier_pct' in user_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=user_data['tier_pct'],
                        y=user_data['pct_of_total_interactions'],
                        mode='lines+markers',
                        name=f'{dataset} (Users)',
                        line=dict(color=color),
                        legendgroup=dataset,
                    ),
                    row=1, col=1
                )
            
            # Item tiers
            item_data = df_items[df_items['Dataset'] == dataset]
            if not item_data.empty and 'tier_pct' in item_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=item_data['tier_pct'],
                        y=item_data['pct_of_total_interactions'],
                        mode='lines+markers',
                        name=f'{dataset} (Items)',
                        line=dict(color=color, dash='dash'),
                        legendgroup=dataset,
                        showlegend=False,
                    ),
                    row=1, col=2
                )
        
        fig.update_layout(
            title='Interaction Concentration (Power Law Analysis)',
            height=450,
            template=self.template,
        )
        
        fig.update_xaxes(title_text='Tier', row=1, col=1)
        fig.update_xaxes(title_text='Tier', row=1, col=2)
        fig.update_yaxes(title_text='% of Total Interactions', row=1, col=1)
        fig.update_yaxes(title_text='% of Total Interactions', row=1, col=2)
        
        return fig
    
    def plot_five_core_stats(self, df: pd.DataFrame) -> go.Figure:
        """Create a comparison of 5-core filtering statistics."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('User Interaction Stats', 'Item Interaction Stats'),
        )
        
        datasets = df['Dataset'].tolist()
        colors = self._get_dataset_colors(datasets)
        
        # User stats
        for i, (_, row) in enumerate(df.iterrows()):
            fig.add_trace(
                go.Bar(
                    x=['Avg', 'Median', 'Max'],
                    y=[row['Avg User Interactions'], 
                       row['Median User Interactions'], 
                       row['Max User Interactions']],
                    name=row['Dataset'],
                    marker_color=colors[i],
                    legendgroup=row['Dataset'],
                ),
                row=1, col=1
            )
        
        # Item stats
        for i, (_, row) in enumerate(df.iterrows()):
            fig.add_trace(
                go.Bar(
                    x=['Avg', 'Median', 'Max'],
                    y=[row['Avg Item Interactions'], 
                       row['Median Item Interactions'], 
                       row['Max Item Interactions']],
                    name=row['Dataset'],
                    marker_color=colors[i],
                    legendgroup=row['Dataset'],
                    showlegend=False,
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title='User and Item Interaction Statistics',
            height=450,
            template=self.template,
            barmode='group',
        )
        
        return fig
    
    # =========================================================================
    # Section 5: Multimodal Coverage
    # =========================================================================
    
    def plot_modality_coverage(self, df: pd.DataFrame) -> go.Figure:
        """Create a grouped bar chart of modality coverage."""
        coverage_cols = [col for col in df.columns if 'Coverage' in col or col == 'Avg Images per Item']
        
        fig = go.Figure()
        
        for dataset in df['Dataset']:
            row = df[df['Dataset'] == dataset].iloc[0]
            values = [row[col] for col in coverage_cols if col in df.columns]
            labels = [col.replace(' (%)', '').replace('Coverage', '').strip() 
                      for col in coverage_cols if col in df.columns]
            
            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                name=dataset,
                marker_color=self._get_dataset_color(dataset),
            ))
        
        fig.update_layout(
            title='Multimodal Feature Coverage',
            yaxis_title='Coverage (%)',
            height=450,
            template=self.template,
            barmode='group',
        )
        
        return fig
    
    def plot_feature_completeness_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create a heatmap of feature completeness across datasets."""
        if df.empty:
            return self._empty_figure("No feature completeness data")
        
        pivot = df.pivot(index='Dataset', columns='Feature', values='Coverage (%)')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale='RdYlGn',
            text=np.round(pivot.values, 1),
            texttemplate='%{text}%',
            textfont={'size': 11},
        ))
        
        fig.update_layout(
            title='Feature Completeness Matrix',
            height=350,
            template=self.template,
        )
        
        return fig
    
    def plot_text_lengths(self, df: pd.DataFrame) -> go.Figure:
        """Create a bar chart comparing average text lengths."""
        fig = go.Figure()
        
        for dataset in df['Dataset']:
            row = df[df['Dataset'] == dataset].iloc[0]
            fig.add_trace(go.Bar(
                x=['Title', 'Description'],
                y=[row['Avg Title Length'], row['Avg Description Length']],
                name=dataset,
                marker_color=self._get_dataset_color(dataset),
            ))
        
        fig.update_layout(
            title='Average Text Lengths by Dataset',
            yaxis_title='Characters',
            height=400,
            template=self.template,
            barmode='group',
        )
        
        return fig
    
    # =========================================================================
    # Section 6: Embedding Quality
    # =========================================================================
    
    def plot_embedding_extraction_stats(self, df_visual: pd.DataFrame, 
                                        df_text: pd.DataFrame) -> go.Figure:
        """Create comparison of embedding extraction statistics."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Visual Embeddings', 'Text Embeddings'),
        )
        
        # Visual embeddings - processing speed
        if not df_visual.empty:
            fig.add_trace(
                go.Bar(
                    x=df_visual['Dataset'],
                    y=df_visual['Items/Second'],
                    marker_color=self._get_dataset_colors(df_visual['Dataset'].tolist()),
                    text=df_visual['Items/Second'].apply(lambda x: f'{x:.1f}'),
                    textposition='outside',
                    name='Visual',
                    showlegend=False,
                ),
                row=1, col=1
            )
        
        # Text embeddings - processing speed
        if not df_text.empty:
            fig.add_trace(
                go.Bar(
                    x=df_text['Dataset'],
                    y=df_text['Items/Second'],
                    marker_color=self._get_dataset_colors(df_text['Dataset'].tolist()),
                    text=df_text['Items/Second'].apply(lambda x: f'{x:.1f}'),
                    textposition='outside',
                    name='Text',
                    showlegend=False,
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Embedding Extraction Speed',
            height=400,
            template=self.template,
        )
        
        fig.update_yaxes(title_text='Items/Second', row=1, col=1)
        fig.update_yaxes(title_text='Items/Second', row=1, col=2)
        
        return fig
    
    def plot_feature_collapse(self, df: pd.DataFrame) -> go.Figure:
        """Create visualization of feature collapse analysis."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Cosine Similarity Distribution', 'Status'),
            column_widths=[0.7, 0.3]
        )
        
        # Similarity statistics as box plot proxy
        for i, (_, row) in enumerate(df.iterrows()):
            dataset = row['Dataset']
            mean = row['Mean Similarity']
            std = row['Std Similarity']
            min_val = row['Min Similarity']
            max_val = row['Max Similarity']
            
            fig.add_trace(
                go.Box(
                    y=[min_val, mean-std, mean, mean+std, max_val],
                    name=dataset,
                    marker_color=self._get_dataset_color(dataset),
                    boxpoints=False,
                ),
                row=1, col=1
            )
        
        # Pass/Fail status
        status_colors = [self.colors['success'] if row['Is Pass'] else self.colors['danger'] 
                        for _, row in df.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=df['Dataset'],
                y=[1] * len(df),
                marker_color=status_colors,
                text=['PASS' if row['Is Pass'] else 'FAIL' for _, row in df.iterrows()],
                textposition='inside',
                showlegend=False,
            ),
            row=1, col=2
        )
        
        # Add threshold line
        fig.add_hline(y=0.5, line_dash='dash', line_color='red', 
                     annotation_text='Threshold', row=1, col=1)
        
        fig.update_layout(
            title='Feature Collapse Analysis',
            height=450,
            template=self.template,
        )
        
        fig.update_yaxes(title_text='Cosine Similarity', row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=1, col=2)
        
        return fig
    
    def plot_anisotropy_comparison(self, df: pd.DataFrame) -> go.Figure:
        """Create before/after comparison of anisotropy correction."""
        fig = go.Figure()
        
        x = np.arange(len(df))
        width = 0.35
        
        fig.add_trace(go.Bar(
            x=df['Dataset'],
            y=df['Before Centering (Avg Cosine)'],
            name='Before Centering',
            marker_color=self.colors['warning'],
            text=df['Before Centering (Avg Cosine)'].apply(lambda x: f'{x:.3f}'),
            textposition='outside',
        ))
        
        fig.add_trace(go.Bar(
            x=df['Dataset'],
            y=df['After Centering (Avg Cosine)'],
            name='After Centering',
            marker_color=self.colors['success'],
            text=df['After Centering (Avg Cosine)'].apply(lambda x: f'{x:.3f}'),
            textposition='outside',
        ))
        
        fig.update_layout(
            title='Anisotropy Correction Effect',
            yaxis_title='Average Cosine Similarity',
            height=450,
            template=self.template,
            barmode='group',
        )
        
        return fig
    
    # =========================================================================
    # Section 7: Modality Alignment
    # =========================================================================
    
    def plot_alignment_comparison(self, df_visual: pd.DataFrame, 
                                   df_text: pd.DataFrame) -> go.Figure:
        """Create comparison of visual and text alignment correlations."""
        fig = go.Figure()
        
        # Visual alignment
        fig.add_trace(go.Bar(
            x=df_visual['Dataset'],
            y=df_visual['Pearson r'],
            name='Visual-Interaction',
            marker_color=self.colors['info'],
            text=df_visual['Pearson r'].apply(lambda x: f'{x:.4f}'),
            textposition='outside',
        ))
        
        # Text alignment  
        if not df_text.empty:
            fig.add_trace(go.Bar(
                x=df_text['Dataset'],
                y=df_text['Pearson r'],
                name='Text-Interaction',
                marker_color=self.colors['warning'],
                text=df_text['Pearson r'].apply(lambda x: f'{x:.4f}'),
                textposition='outside',
            ))
        
        fig.add_hline(y=0, line_dash='solid', line_color='gray')
        
        fig.update_layout(
            title='Modality-Interaction Alignment (Pearson Correlation)',
            yaxis_title='Correlation (r)',
            height=450,
            template=self.template,
            barmode='group',
        )
        
        return fig
    
    def plot_cross_modal_consistency(self, df: pd.DataFrame) -> go.Figure:
        """Visualize cross-modal (text-image) consistency."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Agreement Score', 'Agreement Distribution'),
            column_widths=[0.5, 0.5]
        )
        
        # Agreement scores
        fig.add_trace(
            go.Bar(
                x=df['Dataset'],
                y=df['Mean Agreement'],
                marker_color=self._get_dataset_colors(df['Dataset'].tolist()),
                text=df['Mean Agreement'].apply(lambda x: f'{x:.3f}'),
                textposition='outside',
                showlegend=False,
            ),
            row=1, col=1
        )
        
        # Agreement distribution
        for i, (_, row) in enumerate(df.iterrows()):
            dataset = row['Dataset']
            fig.add_trace(
                go.Bar(
                    x=['Low', 'Moderate', 'High'],
                    y=[row['Low Agreement (%)'], 
                       row['Moderate Agreement (%)'], 
                       row['High Agreement (%)']],
                    name=dataset,
                    marker_color=self._get_dataset_color(dataset),
                    legendgroup=dataset,
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Cross-Modal Consistency Analysis',
            height=450,
            template=self.template,
        )
        
        fig.update_yaxes(title_text='Agreement Score', row=1, col=1)
        fig.update_yaxes(title_text='Percentage (%)', row=1, col=2)
        
        return fig
    
    def plot_cca_correlations(self, df: pd.DataFrame) -> go.Figure:
        """Visualize CCA canonical correlations."""
        fig = go.Figure()
        
        for _, row in df.iterrows():
            dataset = row['Dataset']
            correlations = row['Top 3 Correlations']
            
            if correlations:
                fig.add_trace(go.Bar(
                    x=[f'CC{i+1}' for i in range(len(correlations))],
                    y=correlations,
                    name=dataset,
                    marker_color=self._get_dataset_color(dataset),
                ))
        
        fig.update_layout(
            title='CCA Canonical Correlations',
            xaxis_title='Canonical Component',
            yaxis_title='Correlation',
            height=400,
            template=self.template,
            barmode='group',
        )
        
        return fig
    
    # =========================================================================
    # Section 8: Graph Analysis
    # =========================================================================
    
    def plot_graph_connectivity(self, df: pd.DataFrame) -> go.Figure:
        """Visualize graph connectivity analysis."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Giant Component Coverage', 'Pass/Fail Status'),
            column_widths=[0.7, 0.3]
        )
        
        colors = self._get_dataset_colors(df['Dataset'].tolist())
        
        # Coverage
        fig.add_trace(
            go.Bar(
                x=df['Dataset'],
                y=df['Giant Component (%)'],
                marker_color=colors,
                text=df['Giant Component (%)'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside',
                showlegend=False,
            ),
            row=1, col=1
        )
        
        # Add threshold line
        fig.add_hline(y=50, line_dash='dash', line_color='red',
                     annotation_text='Threshold (50%)', row=1, col=1)
        
        # Status
        status_colors = [self.colors['success'] if row['Is Pass'] else self.colors['danger']
                        for _, row in df.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=df['Dataset'],
                y=[1] * len(df),
                marker_color=status_colors,
                text=['PASS' if row['Is Pass'] else 'FAIL' for _, row in df.iterrows()],
                textposition='inside',
                showlegend=False,
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Graph Connectivity Analysis (LATTICE Feasibility)',
            height=400,
            template=self.template,
        )
        
        fig.update_yaxes(title_text='Coverage (%)', row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=1, col=2)
        
        return fig
    
    def plot_kcore_retention(self, df: pd.DataFrame) -> go.Figure:
        """Plot k-core retention curves."""
        if df.empty:
            return self._empty_figure("No k-core data available")
        
        fig = go.Figure()
        
        for dataset in df['Dataset'].unique():
            data = df[df['Dataset'] == dataset].sort_values('K')
            
            fig.add_trace(go.Scatter(
                x=data['K'],
                y=data['Interaction Retention (%)'],
                mode='lines+markers',
                name=dataset,
                line=dict(color=self._get_dataset_color(dataset)),
            ))
        
        fig.update_layout(
            title='K-Core Filtering: Interaction Retention',
            xaxis_title='K Value',
            yaxis_title='Retention (%)',
            height=400,
            template=self.template,
        )
        
        return fig
    
    # =========================================================================
    # Section 9: BPR Hardness
    # =========================================================================
    
    def plot_bpr_hardness(self, df: pd.DataFrame) -> go.Figure:
        """Visualize BPR negative sampling hardness."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Visual Distance Statistics', 'Hardness Distribution'),
            column_widths=[0.5, 0.5]
        )
        
        colors = self._get_dataset_colors(df['Dataset'].tolist())
        
        # Distance statistics as error bars
        for i, (_, row) in enumerate(df.iterrows()):
            dataset = row['Dataset']
            fig.add_trace(
                go.Scatter(
                    x=[dataset],
                    y=[row['Mean Visual Distance']],
                    error_y=dict(
                        type='data',
                        array=[row['Std Visual Distance']],
                        visible=True
                    ),
                    mode='markers',
                    marker=dict(size=15, color=colors[i]),
                    name=dataset,
                    showlegend=False,
                ),
                row=1, col=1
            )
        
        # Hardness distribution stacked bar
        for _, row in df.iterrows():
            dataset = row['Dataset']
            fig.add_trace(
                go.Bar(
                    x=['Easy', 'Medium', 'Hard'],
                    y=[row['Easy (%)'], row['Medium (%)'], row['Hard (%)']],
                    name=dataset,
                    marker_color=self._get_dataset_color(dataset),
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title='BPR Negative Sampling Hardness Analysis',
            height=450,
            template=self.template,
        )
        
        fig.update_yaxes(title_text='Visual Distance', row=1, col=1)
        fig.update_yaxes(title_text='Percentage (%)', row=1, col=2)
        
        return fig
    
    # =========================================================================
    # Section 10: User Consistency
    # =========================================================================
    
    def plot_user_consistency(self, df: pd.DataFrame) -> go.Figure:
        """Visualize user visual consistency analysis."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Local vs Global Distance', 'Consistency Ratio'),
            column_widths=[0.6, 0.4]
        )
        
        # Local vs Global comparison
        for i, (_, row) in enumerate(df.iterrows()):
            dataset = row['Dataset']
            color = self._get_dataset_color(dataset)
            
            fig.add_trace(
                go.Bar(
                    x=['Local', 'Global'],
                    y=[row['Mean Local Distance'], row['Mean Global Distance']],
                    name=dataset,
                    marker_color=color,
                    legendgroup=dataset,
                ),
                row=1, col=1
            )
        
        # Consistency ratio
        fig.add_trace(
            go.Bar(
                x=df['Dataset'],
                y=df['Consistency Ratio'],
                marker_color=self._get_dataset_colors(df['Dataset'].tolist()),
                text=df['Consistency Ratio'].apply(lambda x: f'{x:.3f}'),
                textposition='outside',
                showlegend=False,
            ),
            row=1, col=2
        )
        
        # Add threshold
        fig.add_hline(y=1.0, line_dash='dash', line_color='gray',
                     annotation_text='Ratio = 1.0', row=1, col=2)
        
        fig.update_layout(
            title='User Visual Preference Consistency',
            height=450,
            template=self.template,
            barmode='group',
        )
        
        fig.update_yaxes(title_text='Distance', row=1, col=1)
        fig.update_yaxes(title_text='Ratio', row=1, col=2)
        
        return fig
    
    # =========================================================================
    # Section 11: Model Feasibility
    # =========================================================================
    
    def plot_lattice_feasibility(self, df: pd.DataFrame) -> go.Figure:
        """Create LATTICE feasibility assessment visualization."""
        fig = go.Figure()
        
        checks = ['Alignment', 'Connectivity', 'Collapse']
        
        for _, row in df.iterrows():
            dataset = row['Dataset']
            
            # Create binary pass/fail for radar
            values = [
                1 if row['Alignment Pass'] else 0,
                1 if row['Connectivity Pass'] else 0,
                1 if row['Collapse Pass'] else 0,
            ]
            values.append(values[0])  # Close the radar
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=checks + [checks[0]],
                name=dataset,
                line=dict(color=self._get_dataset_color(dataset)),
                fill='toself',
                opacity=0.6,
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0, 1],
                    ticktext=['FAIL', 'PASS'],
                )
            ),
            title='LATTICE Model Feasibility Assessment',
            height=500,
            template=self.template,
        )
        
        return fig
    
    def plot_feasibility_summary(self, df: pd.DataFrame) -> go.Figure:
        """Create a summary table visualization of feasibility."""
        fig = go.Figure()
        
        # Prepare data for table
        header_values = ['Dataset', 'Alignment', 'Connectivity', 'Collapse', 'Decision']
        
        cell_values = [
            df['Dataset'].tolist(),
            ['✓ PASS' if v else '✗ FAIL' for v in df['Alignment Pass']],
            ['✓ PASS' if v else '✗ FAIL' for v in df['Connectivity Pass']],
            ['✓ PASS' if v else '✗ FAIL' for v in df['Collapse Pass']],
            df['Decision'].tolist(),
        ]
        
        # Color cells based on pass/fail
        cell_colors = [
            [self._get_dataset_color(d) for d in df['Dataset']],
            [self.colors['success'] if v else self.colors['danger'] for v in df['Alignment Pass']],
            [self.colors['success'] if v else self.colors['danger'] for v in df['Connectivity Pass']],
            [self.colors['success'] if v else self.colors['danger'] for v in df['Collapse Pass']],
            [self.colors['success'] if d == 'PROCEED' else self.colors['danger'] for d in df['Decision']],
        ]
        
        fig.add_trace(go.Table(
            header=dict(
                values=header_values,
                fill_color=self.colors['primary'],
                font=dict(color='white', size=14),
                align='center',
            ),
            cells=dict(
                values=cell_values,
                fill_color=cell_colors,
                font=dict(color='white', size=12),
                align='center',
                height=40,
            )
        ))
        
        fig.update_layout(
            title='Model Feasibility Summary',
            height=250,
        )
        
        return fig
    
    # =========================================================================
    # Summary & Comparison
    # =========================================================================
    
    def plot_summary_radar(self, df: pd.DataFrame) -> go.Figure:
        """Create a radar chart comparing key metrics across datasets."""
        # Normalize metrics to 0-1 scale
        metrics = [
            'Mean Rating',
            'CCA Correlation', 
            'Graph Connected (%)',
        ]
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in df.columns]
        
        if not available_metrics:
            return self._empty_figure("No metrics available for radar chart")
        
        fig = go.Figure()
        
        for _, row in df.iterrows():
            dataset = row['Dataset']
            values = []
            
            for metric in available_metrics:
                val = row.get(metric, 0)
                if isinstance(val, (int, float)):
                    # Normalize to 0-1
                    if 'Rating' in metric:
                        values.append(val / 5.0)
                    elif '%' in metric:
                        values.append(val / 100.0)
                    else:
                        values.append(min(val, 1.0))
                else:
                    values.append(0)
            
            values.append(values[0])  # Close radar
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=available_metrics + [available_metrics[0]],
                name=dataset,
                line=dict(color=self._get_dataset_color(dataset)),
                fill='toself',
                opacity=0.6,
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            title='Dataset Quality Comparison',
            height=500,
            template=self.template,
        )
        
        return fig
    
    def plot_recommendations_summary(self, df: pd.DataFrame) -> go.Figure:
        """Create a summary visualization of recommendations."""
        if df.empty:
            return self._empty_figure("No recommendations available")
        
        fig = go.Figure()
        
        fig.add_trace(go.Table(
            header=dict(
                values=['Dataset', 'Section', 'Recommendation'],
                fill_color=self.colors['primary'],
                font=dict(color='white', size=13),
                align='left',
            ),
            cells=dict(
                values=[
                    df['Dataset'].tolist(),
                    df['Section'].tolist(),
                    df['Recommendation'].tolist(),
                ],
                fill_color=[
                    [self._get_dataset_color(d) for d in df['Dataset']],
                    'white',
                    'white',
                ],
                font=dict(size=11),
                align='left',
                height=35,
            )
        ))
        
        fig.update_layout(
            title='Analysis Recommendations',
            height=max(300, len(df) * 40 + 100),
        )
        
        return fig
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref='paper',
            yref='paper',
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color='gray'),
        )
        fig.update_layout(
            height=300,
            template=self.template,
        )
        return fig
    
    def save_figure(self, fig: go.Figure, filepath: str, 
                   format: str = 'html', **kwargs) -> None:
        """
        Save a figure to file.
        
        Args:
            fig: Plotly figure
            filepath: Output file path
            format: 'html', 'png', 'svg', 'pdf'
            **kwargs: Additional arguments for write functions
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'html':
            fig.write_html(str(filepath), **kwargs)
        else:
            fig.write_image(str(filepath), format=format, **kwargs)
