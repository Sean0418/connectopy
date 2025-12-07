"""Variational Autoencoder for connectome dimensionality reduction.

This module implements a VAE for learning nonlinear latent representations
of brain connectome data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VAEEncoder(nn.Module):
    """Encoder network for the VAE.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dim : int
        Number of hidden units.
    latent_dim : int
        Dimension of latent space.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder.

        Parameters
        ----------
        x : Tensor of shape (batch_size, input_dim)
            Input data.

        Returns
        -------
        mu : Tensor of shape (batch_size, latent_dim)
            Mean of latent distribution.
        logvar : Tensor of shape (batch_size, latent_dim)
            Log variance of latent distribution.
        """
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """Decoder network for the VAE.

    Parameters
    ----------
    latent_dim : int
        Dimension of latent space.
    hidden_dim : int
        Number of hidden units.
    output_dim : int
        Number of output features.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.

        Parameters
        ----------
        z : Tensor of shape (batch_size, latent_dim)
            Latent representation.

        Returns
        -------
        x_recon : Tensor of shape (batch_size, output_dim)
            Reconstructed data.
        """
        h = torch.relu(self.fc1(z))
        h = self.dropout(h)
        return self.fc2(h)  # type: ignore[no-any-return]


class ConnectomeVAE:
    """Variational Autoencoder for brain connectome analysis.

    This class implements a VAE for learning nonlinear latent representations
    of brain connectivity data, providing an alternative to PCA for
    dimensionality reduction.

    Parameters
    ----------
    latent_dim : int, default=60
        Dimension of latent space.
    hidden_dim : int, default=256
        Number of hidden units in encoder/decoder.
    dropout : float, default=0.2
        Dropout probability for regularization.
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer.
    weight_decay : float, default=0.0005
        Weight decay (L2 regularization) for optimizer.
    device : str, default="auto"
        Device to use ("cpu", "cuda", or "auto" for automatic detection).

    Attributes
    ----------
    encoder : VAEEncoder
        Encoder network.
    decoder : VAEDecoder
        Decoder network.
    train_losses : list
        Training loss history.
    val_losses : list
        Validation loss history.

    Examples
    --------
    >>> vae = ConnectomeVAE(latent_dim=60)
    >>> vae.fit(X_train, X_val, epochs=200)
    >>> latent = vae.encode(X)
    """

    def __init__(
        self,
        latent_dim: int = 60,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
        device: str = "auto",
    ) -> None:
        """Initialize the VAE model."""
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model components (initialized in fit)
        self.encoder: VAEEncoder | None = None
        self.decoder: VAEDecoder | None = None
        self.input_dim: int | None = None

        # Training history
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def _build_model(self, input_dim: int) -> None:
        """Build encoder and decoder networks.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        """
        self.input_dim = input_dim
        self.encoder = VAEEncoder(input_dim, self.hidden_dim, self.latent_dim, self.dropout).to(
            self.device
        )
        self.decoder = VAEDecoder(self.latent_dim, self.hidden_dim, input_dim, self.dropout).to(
            self.device
        )

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE.

        Parameters
        ----------
        mu : Tensor
            Mean of latent distribution.
        logvar : Tensor
            Log variance of latent distribution.

        Returns
        -------
        z : Tensor
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _vae_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Compute VAE loss (reconstruction + KL divergence).

        Parameters
        ----------
        x : Tensor
            Original input.
        x_recon : Tensor
            Reconstructed input.
        mu : Tensor
            Latent mean.
        logvar : Tensor
            Latent log variance.

        Returns
        -------
        loss : Tensor
            Combined MSE + KLD loss.
        """
        mse = nn.functional.mse_loss(x_recon, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kld

    def fit(
        self,
        X_train: NDArray[np.float64],
        X_val: NDArray[np.float64] | None = None,
        epochs: int = 200,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> ConnectomeVAE:
        """Fit the VAE model.

        Parameters
        ----------
        X_train : ndarray of shape (n_samples, n_features)
            Training data.
        X_val : ndarray, optional
            Validation data.
        epochs : int, default=200
            Number of training epochs.
        batch_size : int, default=64
            Batch size for training.
        verbose : bool, default=True
            Whether to show progress bar.

        Returns
        -------
        self : ConnectomeVAE
            Fitted model.
        """
        # Build model
        self._build_model(X_train.shape[1])

        # Assert models are built for type checker
        assert self.encoder is not None
        assert self.decoder is not None

        # Create data loaders
        train_tensor = torch.tensor(X_train, dtype=torch.float32)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_tensor: torch.Tensor | None = None
        if X_val is not None:
            val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        # Optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        # Training loop
        iterator = tqdm(range(epochs), desc="Training VAE") if verbose else range(epochs)

        for _ in iterator:
            # Training
            self.encoder.train()
            self.decoder.train()
            train_loss = 0.0

            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()

                # Forward pass
                mu, logvar = self.encoder(batch_x)
                z = self._reparameterize(mu, logvar)
                x_recon = self.decoder(z)

                # Loss
                loss = self._vae_loss(batch_x, x_recon, mu, logvar)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            self.train_losses.append(train_loss / len(X_train))

            # Validation
            if X_val is not None and val_tensor is not None:
                self.encoder.eval()
                self.decoder.eval()
                with torch.no_grad():
                    mu, logvar = self.encoder(val_tensor)
                    z = self._reparameterize(mu, logvar)
                    x_recon = self.decoder(z)
                    val_loss = self._vae_loss(val_tensor, x_recon, mu, logvar)
                    self.val_losses.append(val_loss.item() / len(X_val))

        return self

    def encode(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Encode data to latent space.

        Uses the mean (mu) of the latent distribution for deterministic encoding.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        latent : ndarray of shape (n_samples, latent_dim)
            Latent representations.
        """
        if self.encoder is None:
            raise ValueError("Model must be fit before encoding")

        self.encoder.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            mu, _ = self.encoder(x_tensor)
            return mu.cpu().numpy()

    def decode(self, latent: NDArray[np.float64]) -> NDArray[np.float64]:
        """Decode latent representations to original space.

        Parameters
        ----------
        latent : ndarray of shape (n_samples, latent_dim)
            Latent representations.

        Returns
        -------
        X_recon : ndarray of shape (n_samples, n_features)
            Reconstructed data.
        """
        if self.decoder is None:
            raise ValueError("Model must be fit before decoding")

        self.decoder.eval()
        with torch.no_grad():
            z_tensor = torch.tensor(latent, dtype=torch.float32).to(self.device)
            x_recon = self.decoder(z_tensor)
            return x_recon.cpu().numpy()

    def to_dataframe(
        self,
        X: NDArray[np.float64],
        subject_ids: NDArray | None = None,
        prefix: str = "VAE_LD",
    ) -> pd.DataFrame:
        """Encode data and return as DataFrame.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to encode.
        subject_ids : ndarray, optional
            Subject identifiers.
        prefix : str, default="VAE_LD"
            Prefix for column names.

        Returns
        -------
        df : DataFrame
            DataFrame with latent dimensions.
        """
        latent = self.encode(X)

        columns = [f"{prefix}{i + 1}" for i in range(self.latent_dim)]
        df = pd.DataFrame(latent, columns=columns)

        if subject_ids is not None:
            df.insert(0, "Subject", subject_ids)
        else:
            df.insert(0, "Subject", range(len(df)))

        return df
