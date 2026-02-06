"""Visualization utilities for structural breaks.

This module provides plotting functions for visualizing time series
with structural breaks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from regimes.visualization.style import (
    REGIMES_COLOR_CYCLE,
    REGIMES_COLORS,
    use_style,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

    from regimes.results.base import BreakResultsBase
    from regimes.tests.base import BreakTestResultsBase


def plot_breaks(
    y: ArrayLike,
    breaks: Sequence[int] | None = None,
    results: BreakResultsBase | BreakTestResultsBase | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    xlabel: str = "Observation",
    ylabel: str = "Value",
    break_color: str | None = None,
    break_linestyle: str = "--",
    break_linewidth: float = 0.8,
    break_alpha: float = 0.7,
    series_color: str | None = None,
    series_linewidth: float = 2.0,
    shade_regimes: bool = False,
    regime_colors: Sequence[str] | None = None,
    regime_alpha: float = 0.15,
    show_legend: bool = True,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot time series with structural break lines.

    Parameters
    ----------
    y : ArrayLike
        Time series data to plot.
    breaks : Sequence[int] | None
        Break point indices. If None and results is provided, uses
        breaks from results.
    results : BreakResultsBase | BreakTestResultsBase | None
        Results object containing break information. Alternative to
        specifying breaks directly.
    ax : Axes | None
        Matplotlib axes to plot on. If None, creates new figure and axes.
    title : str | None
        Plot title. If None, uses default.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    break_color : str
        Color for break lines.
    break_linestyle : str
        Line style for break lines.
    break_linewidth : float
        Line width for break lines.
    break_alpha : float
        Alpha (transparency) for break lines.
    series_color : str
        Color for the time series line.
    series_linewidth : float
        Line width for the time series.
    shade_regimes : bool
        Whether to shade different regimes with colors.
    regime_colors : Sequence[str] | None
        Colors for regime shading. If None, uses default color cycle.
    regime_alpha : float
        Alpha for regime shading.
    show_legend : bool
        Whether to show legend.
    figsize : tuple[float, float]
        Figure size (width, height) in inches.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.

    Examples
    --------
    >>> import numpy as np
    >>> from regimes.visualization import plot_breaks
    >>> y = np.concatenate([np.random.randn(50), np.random.randn(50) + 2])
    >>> fig, ax = plot_breaks(y, breaks=[50])
    >>> fig.savefig("breaks_plot.png")
    """
    import matplotlib.pyplot as plt

    # Apply palette defaults
    if break_color is None:
        break_color = REGIMES_COLORS["grey"]
    if series_color is None:
        series_color = REGIMES_COLORS["blue"]
    if regime_colors is None:
        regime_colors = REGIMES_COLOR_CYCLE

    y_arr = np.asarray(y)
    n = len(y_arr)

    # Get breaks from results if not provided directly
    if breaks is None and results is not None:
        breaks = list(results.break_indices)
    elif breaks is None:
        breaks = []
    else:
        breaks = list(breaks)

    with use_style():
        # Create figure and axes if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()  # type: ignore[union-attr]

        # Plot time series
        x = np.arange(n)
        ax.plot(x, y_arr, color=series_color, linewidth=series_linewidth, label="Data")

        # Shade regimes if requested
        if shade_regimes and breaks:
            all_breaks = [0] + sorted(breaks) + [n]
            for i in range(len(all_breaks) - 1):
                start, end = all_breaks[i], all_breaks[i + 1]
                color = regime_colors[i % len(regime_colors)]
                ax.axvspan(
                    start,
                    end,
                    alpha=regime_alpha,
                    color=color,
                    label=f"Regime {i + 1}" if i < 3 else None,
                )

        # Plot break lines
        for i, b in enumerate(sorted(breaks)):
            label = "Breaks" if i == 0 else None
            ax.axvline(
                x=b,
                color=break_color,
                linestyle=break_linestyle,
                linewidth=break_linewidth,
                alpha=break_alpha,
                label=label,
            )

        # Labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title is None:
            n_breaks = len(breaks)
            title = f"Time Series with {n_breaks} Structural Break{'s' if n_breaks != 1 else ''}"
        ax.set_title(title)

        # Legend
        if show_legend and breaks:
            ax.legend(loc="best")

        ax.set_xlim(0, n - 1)

    return fig, ax


def plot_regime_means(
    y: ArrayLike,
    breaks: Sequence[int],
    ax: Axes | None = None,
    title: str | None = None,
    xlabel: str = "Observation",
    ylabel: str = "Value",
    series_color: str | None = None,
    series_alpha: float = 0.6,
    mean_color: str | None = None,
    mean_linewidth: float = 2.0,
    break_color: str | None = None,
    show_legend: bool = True,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot time series with regime-specific means.

    Overlays horizontal lines showing the mean within each regime,
    useful for visualizing mean shift models.

    Parameters
    ----------
    y : ArrayLike
        Time series data.
    breaks : Sequence[int]
        Break point indices.
    ax : Axes | None
        Matplotlib axes. If None, creates new figure.
    title : str | None
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    series_color : str
        Color for data points.
    series_alpha : float
        Alpha for data points.
    mean_color : str
        Color for regime mean lines.
    mean_linewidth : float
        Line width for regime means.
    break_color : str
        Color for break lines.
    show_legend : bool
        Whether to show legend.
    figsize : tuple[float, float]
        Figure size.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes.
    """
    import matplotlib.pyplot as plt

    # Apply palette defaults
    if series_color is None:
        series_color = REGIMES_COLORS["blue"]
    if mean_color is None:
        mean_color = REGIMES_COLORS["red"]
    if break_color is None:
        break_color = REGIMES_COLORS["grey"]

    y_arr = np.asarray(y)
    n = len(y_arr)
    breaks_list = sorted(list(breaks))

    with use_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()  # type: ignore[union-attr]

        # Plot data
        x = np.arange(n)
        ax.plot(
            x,
            y_arr,
            "o",
            color=series_color,
            alpha=series_alpha,
            markersize=3,
            label="Data",
        )

        # Compute and plot regime means
        all_breaks = [0] + breaks_list + [n]
        for i in range(len(all_breaks) - 1):
            start, end = all_breaks[i], all_breaks[i + 1]
            regime_mean = np.mean(y_arr[start:end])
            label = "Regime means" if i == 0 else None
            ax.hlines(
                regime_mean,
                start,
                end - 1,
                colors=mean_color,
                linewidth=mean_linewidth,
                label=label,
            )

        # Plot break lines
        for i, b in enumerate(breaks_list):
            label = "Breaks" if i == 0 else None
            ax.axvline(
                x=b,
                color=break_color,
                linestyle="--",
                alpha=0.7,
                linewidth=0.8,
                label=label,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is None:
            title = "Time Series with Regime Means"
        ax.set_title(title)

        if show_legend:
            ax.legend(loc="best")

        ax.set_xlim(0, n - 1)

    return fig, ax


def plot_break_confidence(
    y: ArrayLike,
    breaks: Sequence[int],
    confidence_intervals: Sequence[tuple[int, int]] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    ci_alpha: float = 0.15,
    ci_color: str | None = None,
    figsize: tuple[float, float] = (10, 5),
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot time series with break confidence intervals.

    Parameters
    ----------
    y : ArrayLike
        Time series data.
    breaks : Sequence[int]
        Break point indices.
    confidence_intervals : Sequence[tuple[int, int]] | None
        Confidence intervals for each break as (lower, upper) tuples.
    ax : Axes | None
        Matplotlib axes.
    title : str | None
        Plot title.
    ci_alpha : float
        Alpha for confidence interval shading.
    ci_color : str
        Color for confidence interval shading.
    figsize : tuple[float, float]
        Figure size.
    **kwargs
        Additional arguments passed to plot_breaks.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes.
    """
    # Apply palette defaults
    if ci_color is None:
        ci_color = REGIMES_COLORS["red"]

    # First create base plot (use_style is already applied inside plot_breaks)
    fig, ax = plot_breaks(y, breaks=breaks, ax=ax, figsize=figsize, **kwargs)

    # Add confidence intervals
    if confidence_intervals is not None:
        for lower, upper in confidence_intervals:
            ax.axvspan(lower, upper, alpha=ci_alpha, color=ci_color)

    if title is not None:
        ax.set_title(title)

    return fig, ax
