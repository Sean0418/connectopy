"""Analysis modules for brain connectome data."""

from brain_connectome.analysis.pca import ConnectomePCA
from brain_connectome.analysis.vae import ConnectomeVAE
from brain_connectome.analysis.dimorphism import DimorphismAnalysis

__all__ = ["ConnectomePCA", "ConnectomeVAE", "DimorphismAnalysis"]

