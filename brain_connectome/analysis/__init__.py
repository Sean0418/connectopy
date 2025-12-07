"""Analysis modules for brain connectome data."""

from brain_connectome.analysis.dimorphism import DimorphismAnalysis
from brain_connectome.analysis.mediation import (
    MediationAnalysis,
    MediationResult,
    SexStratifiedMediation,
    SexStratifiedResult,
    run_multiple_mediations,
)
from brain_connectome.analysis.pca import ConnectomePCA
from brain_connectome.analysis.vae import ConnectomeVAE

__all__ = [
    "ConnectomePCA",
    "ConnectomeVAE",
    "DimorphismAnalysis",
    "MediationAnalysis",
    "MediationResult",
    "SexStratifiedMediation",
    "SexStratifiedResult",
    "run_multiple_mediations",
]
