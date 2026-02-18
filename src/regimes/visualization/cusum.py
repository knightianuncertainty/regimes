"""Visualization functions for CUSUM and CUSUM-SQ tests.

This module provides plotting functions for CUSUM and CUSUM-SQ test
results, following the regimes PLOTTING_STYLE.md specification (Section 4.4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from regimes.visualization.style import REGIMES_COLORS, use_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from regimes.tests.cusum import CUSUMResults, CUSUMSQResults


def plot_cusum(
    results: CUSUMResults,
    ax: Axes | None = None,
    title: str | None = None,
    xlabel: str = "Observation",
    ylabel: str = "CUSUM statistic",
    statistic_color: str | None = None,
    bound_color: str | None = None,
    shade_rejection: bool = True,
    rejection_alpha: float = 0.10,
    show_crossings: bool = True,
    crossing_color: str | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot the CUSUM test statistic with critical boundaries.

    Parameters
    ----------
    results : CUSUMResults
        Results from a CUSUM test.
    ax : Axes | None
        Axes to plot on. If None, creates a new figure.
    title : str | None
        Plot title. Defaults to "CUSUM test statistic".
    xlabel : str
        X-axis label. Default is "Observation".
    ylabel : str
        Y-axis label. Default is "CUSUM statistic".
    statistic_color : str | None
        Color for the statistic path. Defaults to REGIMES_COLORS["blue"].
    bound_color : str | None
        Color for critical boundaries. Defaults to REGIMES_COLORS["red"].
    shade_rejection : bool
        Whether to shade the rejection region. Default is True.
    rejection_alpha : float
        Alpha for rejection region shading. Default is 0.10.
    show_crossings : bool
        Whether to mark boundary crossings. Default is True.
    crossing_color : str | None
        Color for crossing markers. Defaults to REGIMES_COLORS["grey"].
    figsize : tuple[float, float]
        Figure size. Default is (10, 5).

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes.
    """
    import matplotlib.pyplot as plt

    if statistic_color is None:
        statistic_color = REGIMES_COLORS["blue"]
    if bound_color is None:
        bound_color = REGIMES_COLORS["red"]
    if crossing_color is None:
        crossing_color = REGIMES_COLORS["grey"]

    with use_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()  # type: ignore[union-attr]

        n = len(results.statistic_path)
        x = np.arange(n)

        # Plot statistic path
        ax.plot(
            x,
            results.statistic_path,
            color=statistic_color,
            linewidth=2.0,
            label="CUSUM",
        )

        # Plot critical boundaries
        ax.plot(
            x,
            results.upper_bound,
            color=bound_color,
            linewidth=1.0,
            linestyle="--",
            label=f"{int((1 - results.significance) * 100)}% bounds",
        )
        ax.plot(
            x,
            results.lower_bound,
            color=bound_color,
            linewidth=1.0,
            linestyle="--",
        )

        # Shade rejection region
        if shade_rejection:
            ax.fill_between(
                x,
                results.upper_bound,
                np.full(n, ax.get_ylim()[1] if n > 0 else 10),
                alpha=rejection_alpha,
                color=bound_color,
            )
            ax.fill_between(
                x,
                np.full(n, ax.get_ylim()[0] if n > 0 else -10),
                results.lower_bound,
                alpha=rejection_alpha,
                color=bound_color,
            )

        # Mark boundary crossings
        if show_crossings and results.crossing_indices:
            for idx in results.crossing_indices:
                ax.axvline(
                    x=idx,
                    color=crossing_color,
                    linewidth=0.8,
                    linestyle=":",
                    alpha=0.7,
                )

        # Zero line
        ax.axhline(y=0, color=REGIMES_COLORS["near_black"], linewidth=0.5, alpha=0.3)

        ax.set_title(title or "CUSUM test statistic")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", frameon=False)

    return fig, ax  # type: ignore[return-value]


def plot_cusum_sq(
    results: CUSUMSQResults,
    ax: Axes | None = None,
    title: str | None = None,
    xlabel: str = "Observation",
    ylabel: str = "CUSUM-SQ statistic",
    statistic_color: str | None = None,
    bound_color: str | None = None,
    expected_color: str | None = None,
    shade_rejection: bool = True,
    rejection_alpha: float = 0.10,
    show_crossings: bool = True,
    show_expected: bool = True,
    crossing_color: str | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot the CUSUM-SQ test statistic with critical boundaries.

    Parameters
    ----------
    results : CUSUMSQResults
        Results from a CUSUM-SQ test.
    ax : Axes | None
        Axes to plot on. If None, creates a new figure.
    title : str | None
        Plot title. Defaults to "CUSUM-SQ test statistic".
    xlabel : str
        X-axis label. Default is "Observation".
    ylabel : str
        Y-axis label. Default is "CUSUM-SQ statistic".
    statistic_color : str | None
        Color for the statistic path. Defaults to REGIMES_COLORS["blue"].
    bound_color : str | None
        Color for critical boundaries. Defaults to REGIMES_COLORS["red"].
    expected_color : str | None
        Color for expected diagonal. Defaults to REGIMES_COLORS["grey"].
    shade_rejection : bool
        Whether to shade the rejection region. Default is True.
    rejection_alpha : float
        Alpha for rejection region shading. Default is 0.10.
    show_crossings : bool
        Whether to mark boundary crossings. Default is True.
    show_expected : bool
        Whether to show the expected diagonal. Default is True.
    crossing_color : str | None
        Color for crossing markers. Defaults to REGIMES_COLORS["grey"].
    figsize : tuple[float, float]
        Figure size. Default is (10, 5).

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes.
    """
    import matplotlib.pyplot as plt

    if statistic_color is None:
        statistic_color = REGIMES_COLORS["blue"]
    if bound_color is None:
        bound_color = REGIMES_COLORS["red"]
    if expected_color is None:
        expected_color = REGIMES_COLORS["grey"]
    if crossing_color is None:
        crossing_color = REGIMES_COLORS["grey"]

    with use_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()  # type: ignore[union-attr]

        n = len(results.statistic_path)
        x = np.arange(n)

        # Plot expected diagonal
        if show_expected:
            ax.plot(
                x,
                results.expected_path,
                color=expected_color,
                linewidth=1.0,
                linestyle="--",
                alpha=0.6,
                label="Expected",
            )

        # Plot statistic path
        ax.plot(
            x,
            results.statistic_path,
            color=statistic_color,
            linewidth=2.0,
            label="CUSUM-SQ",
        )

        # Plot critical boundaries
        ax.plot(
            x,
            results.upper_bound,
            color=bound_color,
            linewidth=1.0,
            linestyle="--",
            label=f"{int((1 - results.significance) * 100)}% bounds",
        )
        ax.plot(
            x,
            results.lower_bound,
            color=bound_color,
            linewidth=1.0,
            linestyle="--",
        )

        # Shade rejection region
        if shade_rejection:
            ax.fill_between(
                x,
                results.upper_bound,
                np.ones(n),
                alpha=rejection_alpha,
                color=bound_color,
            )
            ax.fill_between(
                x,
                np.zeros(n),
                results.lower_bound,
                alpha=rejection_alpha,
                color=bound_color,
                where=results.lower_bound > 0,
            )

        # Mark boundary crossings
        if show_crossings and results.crossing_indices:
            for idx in results.crossing_indices:
                ax.axvline(
                    x=idx,
                    color=crossing_color,
                    linewidth=0.8,
                    linestyle=":",
                    alpha=0.7,
                )

        ax.set_title(title or "CUSUM-SQ test statistic")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", frameon=False)

    return fig, ax  # type: ignore[return-value]
