"""Visualization for GETS indicator saturation results.

Provides plotting functions for SIS and MIS results, showing estimated
regime levels as step functions over time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from regimes.visualization.style import (
    REGIMES_COLOR_CYCLE,
    REGIMES_COLORS,
    add_break_dates,
    add_confidence_band,
    use_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from regimes.gets.saturation import SaturationResults


def plot_sis_coefficients(
    results: SaturationResults,
    ax: Axes | None = None,
    title: str | None = None,
    xlabel: str = "Observation",
    ylabel: str = "Intercept level",
    show_ci: bool = True,
    ci_alpha: float = 0.15,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot SIS results: intercept regime levels as a step function.

    Parameters
    ----------
    results : SaturationResults
        Results from ``isat()`` with SIS.
    ax : Axes | None
        Existing axes to plot on. If None, creates a new figure.
    title : str | None
        Plot title. If None, uses a default.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    show_ci : bool
        If True, show confidence interval bands around regime levels.
    ci_alpha : float
        Transparency for confidence bands.
    figsize : tuple[float, float]
        Figure size if creating a new figure.

    Returns
    -------
    tuple[Figure, Axes]
    """
    import matplotlib.pyplot as plt

    with use_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        regimes_dict = results.regime_levels.param_regimes
        if "const" not in regimes_dict:
            ax.text(
                0.5,
                0.5,
                "No intercept regimes found",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            if title:
                ax.set_title(title)
            return fig, ax

        regimes = regimes_dict["const"]
        color = REGIMES_COLORS["blue"]

        # Plot step function
        for regime in regimes:
            x = np.array([regime.start, regime.end])
            y_level = np.array([regime.level, regime.level])
            ax.plot(x, y_level, color=color, linewidth=2.0, solid_capstyle="butt")

            if show_ci:
                upper = regime.level + 1.96 * regime.level_se
                lower = regime.level - 1.96 * regime.level_se
                add_confidence_band(
                    ax,
                    x,
                    np.array([lower, lower]),
                    np.array([upper, upper]),
                    color=color,
                    alpha=ci_alpha,
                )

        # Add break date lines
        if results.break_dates:
            add_break_dates(ax, results.break_dates)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title or "SIS: Intercept Regime Levels")

    return fig, ax


def plot_mis_coefficients(
    results: SaturationResults,
    params: list[str] | None = None,
    axes: NDArray[Any] | None = None,
    title: str | None = None,
    xlabel: str = "Observation",
    show_ci: bool = True,
    ci_alpha: float = 0.15,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, NDArray[Any]]:
    """Plot MIS results: coefficient regime levels per parameter.

    Parameters
    ----------
    results : SaturationResults
        Results from ``isat()`` with MIS.
    params : list[str] | None
        Which parameters to plot. If None, plots all non-constant
        parameters that have regimes.
    axes : NDArray | None
        Existing axes array. If None, creates a new figure.
    title : str | None
        Suptitle for the figure.
    xlabel : str
        X-axis label.
    show_ci : bool
        If True, show confidence interval bands.
    ci_alpha : float
        Transparency for confidence bands.
    figsize : tuple[float, float] | None
        Figure size. If None, auto-computed from number of panels.

    Returns
    -------
    tuple[Figure, NDArray]
        Figure and axes array.
    """
    import matplotlib.pyplot as plt

    with use_style():
        regimes_dict = results.regime_levels.param_regimes

        # Select parameters to plot
        if params is None:
            params = [p for p in regimes_dict if p != "const"]
        if not params:
            params = list(regimes_dict.keys())

        n_params = len(params)
        if n_params == 0:
            fig, ax = plt.subplots(figsize=figsize or (10, 5))
            ax.text(
                0.5,
                0.5,
                "No parameter regimes found",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return fig, np.array([[ax]])

        if figsize is None:
            figsize = (10, 3.5 * n_params)

        if axes is None:
            fig, axes_arr = plt.subplots(n_params, 1, figsize=figsize, squeeze=False)
        else:
            axes_arr = np.atleast_2d(axes)
            fig = axes_arr.flat[0].get_figure()

        colors = REGIMES_COLOR_CYCLE

        for i, param_name in enumerate(params):
            ax = axes_arr.flat[i]
            color = colors[i % len(colors)]

            if param_name not in regimes_dict:
                ax.set_title(f"{param_name}: no regimes")
                continue

            regimes = regimes_dict[param_name]

            for regime in regimes:
                x = np.array([regime.start, regime.end])
                y_level = np.array([regime.level, regime.level])
                ax.plot(x, y_level, color=color, linewidth=2.0, solid_capstyle="butt")

                if show_ci:
                    upper = regime.level + 1.96 * regime.level_se
                    lower = regime.level - 1.96 * regime.level_se
                    add_confidence_band(
                        ax,
                        x,
                        np.array([lower, lower]),
                        np.array([upper, upper]),
                        color=color,
                        alpha=ci_alpha,
                    )

            if results.break_dates:
                add_break_dates(ax, results.break_dates)

            ax.set_ylabel(param_name)
            if i == n_params - 1:
                ax.set_xlabel(xlabel)

        if title:
            fig.suptitle(title, fontweight="bold", fontsize=13)
        fig.tight_layout()

    return fig, axes_arr


def plot_regime_levels(
    results: SaturationResults,
    params: list[str] | None = None,
    axes: NDArray[Any] | None = None,
    title: str | None = None,
    xlabel: str = "Observation",
    show_ci: bool = True,
    ci_alpha: float = 0.15,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, NDArray[Any]]:
    """Plot all regime levels (combined SIS + MIS) in a panel.

    Parameters
    ----------
    results : SaturationResults
        Results from ``isat()``.
    params : list[str] | None
        Which parameters to plot. If None, plots all parameters
        with regime changes.
    axes : NDArray | None
        Existing axes array. If None, creates a new figure.
    title : str | None
        Suptitle for the figure.
    xlabel : str
        X-axis label.
    show_ci : bool
        If True, show confidence interval bands.
    ci_alpha : float
        Transparency for confidence bands.
    figsize : tuple[float, float] | None
        Figure size. If None, auto-computed.

    Returns
    -------
    tuple[Figure, NDArray]
        Figure and axes array.
    """
    import matplotlib.pyplot as plt

    with use_style():
        regimes_dict = results.regime_levels.param_regimes

        # Select parameters: all that have more than 1 regime
        if params is None:
            params = [p for p, regs in regimes_dict.items() if len(regs) > 1]
            # Fall back to all if none have breaks
            if not params:
                params = list(regimes_dict.keys())

        n_params = len(params)
        if n_params == 0:
            fig, ax = plt.subplots(figsize=figsize or (10, 5))
            ax.text(
                0.5,
                0.5,
                "No regime changes detected",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return fig, np.array([[ax]])

        if figsize is None:
            figsize = (10, 3.5 * n_params)

        if axes is None:
            fig, axes_arr = plt.subplots(n_params, 1, figsize=figsize, squeeze=False)
        else:
            axes_arr = np.atleast_2d(axes)
            fig = axes_arr.flat[0].get_figure()

        colors = REGIMES_COLOR_CYCLE

        for i, param_name in enumerate(params):
            ax = axes_arr.flat[i]
            color = colors[i % len(colors)]

            if param_name not in regimes_dict:
                ax.set_title(f"{param_name}: no regimes")
                continue

            regimes = regimes_dict[param_name]

            for regime in regimes:
                x = np.array([regime.start, regime.end])
                y_level = np.array([regime.level, regime.level])
                ax.plot(x, y_level, color=color, linewidth=2.0, solid_capstyle="butt")

                if show_ci:
                    upper = regime.level + 1.96 * regime.level_se
                    lower = regime.level - 1.96 * regime.level_se
                    add_confidence_band(
                        ax,
                        x,
                        np.array([lower, lower]),
                        np.array([upper, upper]),
                        color=color,
                        alpha=ci_alpha,
                    )

            if results.break_dates:
                add_break_dates(ax, results.break_dates)

            ax.set_ylabel(param_name)
            if i == n_params - 1:
                ax.set_xlabel(xlabel)

        suptitle = title or f"Regime Levels ({results.saturation_type})"
        fig.suptitle(suptitle, fontweight="bold", fontsize=13)
        fig.tight_layout()

    return fig, axes_arr
