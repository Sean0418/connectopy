"""Brain Connectome Analysis Package.

A Python package for analyzing brain structural and functional connectomes
from the Human Connectome Project (HCP).
"""

__version__ = "0.1.0"
__author__ = "Riley Harper, Sean Shen, Yinyu Yao"

from brain_connectome.data.loader import ConnectomeDataLoader
from brain_connectome.analysis.pca import ConnectomePCA
from brain_connectome.analysis.dimorphism import DimorphismAnalysis

__all__ = [
    "ConnectomeDataLoader",
    "ConnectomePCA",
    "DimorphismAnalysis",
]

