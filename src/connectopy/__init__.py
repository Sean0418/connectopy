"""Brain Connectome Analysis Package.

A Python package for analyzing brain structural and functional connectomes
from the Human Connectome Project (HCP).
"""

__version__ = "0.1.0"
__author__ = "Riley Harper, Sean Shen, Yinyu Yao"

from connectopy.analysis.dimorphism import DimorphismAnalysis
from connectopy.analysis.pca import ConnectomePCA
from connectopy.data.loader import ConnectomeDataLoader
from connectopy.models.classifiers import (
    ConnectomeEBM,
    ConnectomeRandomForest,
    ConnectomeXGBoost,
)

__all__ = [
    # Data
    "ConnectomeDataLoader",
    # Analysis
    "ConnectomePCA",
    "DimorphismAnalysis",
    # Models
    "ConnectomeRandomForest",
    "ConnectomeXGBoost",
    "ConnectomeEBM",
]
