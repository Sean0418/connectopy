"""Analysis modules for brain connectome data."""

from brain_connectome.analysis.dimorphism import DimorphismAnalysis
from brain_connectome.analysis.pca import ConnectomePCA
from brain_connectome.analysis.vae import ConnectomeVAE

__all__ = ["ConnectomePCA", "ConnectomeVAE", "DimorphismAnalysis"]
