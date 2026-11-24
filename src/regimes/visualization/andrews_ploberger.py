"""Visualization functions for Andrews-Ploberger test results.

This module provides plotting functions for the F-statistic sequence
from the Andrews-Ploberger structural break test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from regimes.visualization.style import REGIMES_COLORS, use_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from regimes.tests.andrews_ploberger import AndrewsPlobergerResults


def plot_f_sequence(
    results: AndrewsPlobergerResults,
    ax: Axes | None = None,
    title: str | None = None,
    xlabel: str = "Observation",
    ylabel: str = "F-statistic",
    statistic_color: str | None = None,
    critical_color: str | None = None,
    break_color: str | None = None,
    show_critical: bool = True,
    show_break: bool = True,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot the F-statistic sequence from an Andrews-Ploberger test.

    Displays the F(τ) path across candidate break dates, with a
    horizontal line at the SupF critical value and a vertical line
    at the estimated break date.

    Parameters
    ----------
    results : AndrewsPlobergerResults
        Results from an Andrews-Ploberger test.
    ax : Axes | None
        Axes to plot on. If None, creates a new figure.
    title : str | None
        Plot title. Defaults to "Andrews-Ploberger F-statistic sequence".
    xlabel : str
        X-axis label. Default is "Observation".
    ylabel : str
        Y-axis label. Default is "F-statistic".
    statistic_color : str | None
        Color for the F-statistic path. Defaults to REGIMES_COLORS["blue"].
    critical_color : str | None
        Color for the critical value line. Defaults to REGIMES_COLORS["red"].
    break_color : str | None
        Color for the break date line. Defaults to REGIMES_COLORS["grey"].
    show_critical : bool
        Whether to show the SupF critical value line. Default is True.
    show_break : bool
        Whether to show the estimated break date. Default is True.
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
    if critical_color is None:
        critical_color = REGIMES_COLORS["red"]
    if break_color is None:
        break_color = REGIMES_COLORS["grey"]

    with use_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = cast("Figure", ax.get_figure())

        # Plot F-statistic path
        ax.plot(
            results.candidate_indices,
            results.f_sequence,
            color=statistic_color,
            linewidth=2.0,
            label="F(τ)",
        )

        # Plot SupF critical value line
        if show_critical:
            cv = results.sup_f_critical.get(results.significance_level, np.nan)
            if not np.isnan(cv):
                ax.axhline(
                    y=cv,
                    color=critical_color,
                    linewidth=1.0,
                    linestyle="--",
                    label=f"{int((1 - results.significance_level) * 100)}% CV ({cv:.2f})",
                )

        # Plot estimated break date
        if show_break and results.n_breaks > 0:
            ax.axvline(
                x=results.sup_f_break_index,
                color=break_color,
                linewidth=0.8,
                linestyle=":",
                alpha=0.7,
                label=f"Break at {results.sup_f_break_index}",
            )

        # Zero line
        ax.axhline(y=0, color=REGIMES_COLORS["near_black"], linewidth=0.5, alpha=0.3)

        ax.set_title(title or "Andrews-Ploberger F-statistic sequence")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", frameon=False)

    return fig, ax
