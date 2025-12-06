"""Plotting functions for brain connectome visualization.

This module provides visualization functions for connectome analysis results,
including connectivity matrices, PCA scatter plots, and feature importance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def plot_connectome_matrix(
    matrix: NDArray[np.float64],
    title: str = "Connectome Matrix",
    cmap: str = "Reds",
    log_scale: bool = True,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (8, 8),
) -> tuple[Figure, Axes]:
    """Plot a brain connectivity matrix as a heatmap.

    Parameters
    ----------
    matrix : ndarray of shape (n_regions, n_regions)
        Connectivity matrix.
    title : str, default="Connectome Matrix"
        Plot title.
    cmap : str, default="Reds"
        Colormap name.
    log_scale : bool, default=True
        Whether to apply log scaling (log(x+1)).
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.
    figsize : tuple of (width, height), default=(8, 8)
        Figure size if creating new figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]  # type: ignore[assignment]

    if log_scale:
        matrix = np.log1p(matrix)

    im = ax.imshow(matrix, cmap=cmap, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("Region")
    ax.set_ylabel("Region")
    label = "Log(Connection Strength + 1)" if log_scale else "Connection Strength"
    plt.colorbar(im, ax=ax, label=label)

    return fig, ax


def plot_pca_scatter(
    data: pd.DataFrame,
    pc_x: str = "PC1",
    pc_y: str = "PC2",
    hue: str = "Gender",
    palette: dict | None = None,
    add_ellipse: bool = True,
    title: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> tuple[Figure, Axes]:
    """Create a scatter plot of PCA components colored by group.

    Parameters
    ----------
    data : DataFrame
        Data containing PC columns and hue variable.
    pc_x : str, default="PC1"
        Column name for x-axis.
    pc_y : str, default="PC2"
        Column name for y-axis.
    hue : str, default="Gender"
        Column to color by.
    palette : dict, optional
        Color mapping. Default is {"M": "#1f77b4", "F": "#d62728"}.
    add_ellipse : bool, default=True
        Whether to add 95% confidence ellipses.
    title : str, optional
        Plot title.
    ax : Axes, optional
        Matplotlib axes to plot on.
    figsize : tuple of (width, height), default=(10, 8)
        Figure size if creating new figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    if palette is None:
        palette = {"M": "#1f77b4", "F": "#d62728"}

    # Scatter plot
    for group in data[hue].unique():
        group_data = data[data[hue] == group]
        color = palette.get(group)
        ax.scatter(
            group_data[pc_x],
            group_data[pc_y],
            label=group,
            c=color,
            alpha=0.5,
            s=30,
        )

        if add_ellipse:
            _add_confidence_ellipse(
                group_data[pc_x].values,
                group_data[pc_y].values,
                ax,
                color=color,
                alpha=0.2,
            )

    ax.set_xlabel(pc_x)
    ax.set_ylabel(pc_y)
    ax.legend()

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{pc_x} vs {pc_y} by {hue}")

    return fig, ax


def _add_confidence_ellipse(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    ax: Axes,
    n_std: float = 2.0,
    color: str | None = None,
    alpha: float = 0.2,
) -> None:
    """Add a confidence ellipse to an axes.

    Parameters
    ----------
    x : ndarray
        X coordinates.
    y : ndarray
        Y coordinates.
    ax : Axes
        Axes to add ellipse to.
    n_std : float, default=2.0
        Number of standard deviations for ellipse.
    color : str, optional
        Ellipse color.
    alpha : float, default=0.2
        Ellipse transparency.
    """
    import matplotlib.transforms as transforms
    from matplotlib.patches import Ellipse

    if len(x) < 2:
        return

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=color,
        alpha=alpha,
        edgecolor=color,
    )

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(float(np.mean(x)), float(np.mean(y)))
    )

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def plot_dimorphism_comparison(
    data: pd.DataFrame,
    feature: str,
    gender_col: str = "Gender",
    title: str | None = None,
    palette: dict | None = None,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> tuple[Figure, Axes]:
    """Create a boxplot comparing a feature between genders.

    Parameters
    ----------
    data : DataFrame
        Data containing feature and gender columns.
    feature : str
        Column name of feature to compare.
    gender_col : str, default="Gender"
        Column name for gender.
    title : str, optional
        Plot title.
    palette : dict, optional
        Color mapping for genders.
    ax : Axes, optional
        Matplotlib axes to plot on.
    figsize : tuple of (width, height), default=(8, 6)
        Figure size if creating new figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    if palette is None:
        palette = {"M": "#1f77b4", "F": "#d62728"}

    sns.boxplot(
        data=data,
        x=gender_col,
        y=feature,
        palette=palette,
        ax=ax,
    )

    sns.stripplot(
        data=data,
        x=gender_col,
        y=feature,
        color="black",
        alpha=0.1,
        ax=ax,
    )

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Distribution of {feature} by {gender_col}")

    return fig, ax


def plot_feature_importance(
    importances: pd.DataFrame,
    n_features: int = 10,
    title: str = "Top Feature Importances",
    ax: Axes | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> tuple[Figure, Axes]:
    """Create a horizontal bar plot of feature importances.

    Parameters
    ----------
    importances : DataFrame
        DataFrame with 'Feature' and 'Importance' columns.
    n_features : int, default=10
        Number of top features to show.
    title : str, default="Top Feature Importances"
        Plot title.
    ax : Axes, optional
        Matplotlib axes to plot on.
    figsize : tuple of (width, height), default=(10, 6)
        Figure size if creating new figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    top_features = importances.head(n_features).iloc[::-1]

    colors = plt.colormaps["viridis"](np.linspace(0.3, 0.9, n_features))

    ax.barh(
        top_features["Feature"],
        top_features["Importance"],
        color=colors,
    )

    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(title)

    return fig, ax


def plot_scree(
    variance_explained: NDArray[np.float64],
    n_components: int | None = None,
    title: str = "Scree Plot",
    ax: Axes | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> tuple[Figure, Axes]:
    """Create a scree plot showing variance explained by PCA components.

    Parameters
    ----------
    variance_explained : ndarray
        Array of variance explained ratios.
    n_components : int, optional
        Number of components to show. If None, shows all.
    title : str, default="Scree Plot"
        Plot title.
    ax : Axes, optional
        Matplotlib axes to plot on.
    figsize : tuple of (width, height), default=(10, 6)
        Figure size if creating new figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    if n_components is not None:
        variance_explained = variance_explained[:n_components]

    n = len(variance_explained)
    cumulative = np.cumsum(variance_explained)

    ax.plot(range(1, n + 1), variance_explained, "bo-", label="Individual")
    ax.plot(range(1, n + 1), cumulative, "ro-", label="Cumulative")

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax
