"""
EDA Results Analysis and Visualization Module
==============================================

This module provides comprehensive tools for extracting, analyzing, 
and visualizing multimodal recommendation system EDA results.

Main Components:
- eda_report_generator: Main script for generating HTML reports
- visualizations: Plotly-based interactive visualization functions
- metrics: Metric extraction and computation utilities
- utils: Helper functions and data loading utilities
"""

from .metrics import MetricsExtractor
from .visualizations import EDAVisualizer
from .eda_report_generator import EDAReportGenerator

__all__ = ['MetricsExtractor', 'EDAVisualizer', 'EDAReportGenerator']
__version__ = '1.0.0'
