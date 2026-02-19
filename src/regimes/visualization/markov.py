"""Visualization utilities for Markov regime-switching models.

This module provides four core plot functions for MS model results,
plus an IC plot for regime number selection.
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
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray

    from regimes.markov.results import MarkovSwitchingResultsBase


def plot_smoothed_probabilities(
    results: MarkovSwitchingResultsBase,
    ax: NDArray[Any] | None = None,
    title: str | None = None,
    xlabel: str = "Observation",
    figsize: tuple[float, float] = (10, 3),
    alpha: float = 0.4,
) -> tuple[Figure, NDArray[Any]]:
    """Plot smoothed regime probabilities P(S_t = j | Y_1,...,Y_T).

    Creates one subplot per regime, stacked vertically.

    Parameters
    ----------
    results : MarkovSwitchingResultsBase
        Fitted Markov switching model results.
    ax : NDArray[Axes] | None
        Array of axes to plot on. If None, creates new figure.
    title : str | None
        Overall figure title.
    xlabel : str
        X-axis label.
    figsize : tuple[float, float]
        Per-panel figure size (width, height_per_panel).
    alpha : float
        Fill alpha for the probability area.

    Returns
    -------
    tuple[Figure, NDArray[Axes]]
        Figure and array of axes (one per regime).
    """
    import matplotlib.pyplot as plt

    k = results.k_regimes
    probs = results.smoothed_marginal_probabilities

    with use_style():
        if ax is None:
            fig, axes = plt.subplots(
                k,
                1,
                figsize=(figsize[0], figsize[1] * k),
                sharex=True,
                squeeze=False,
            )
            axes = axes.flatten()
        else:
            axes = np.asarray(ax).flatten()
            fig = axes[0].get_figure()

        x = np.arange(len(probs))

        for j in range(k):
            color = REGIMES_COLOR_CYCLE[j % len(REGIMES_COLOR_CYCLE)]
            axes[j].fill_between(
                x,
                0,
                probs[:, j],
                color=color,
                alpha=alpha,
                linewidth=0,
            )
            axes[j].plot(x, probs[:, j], color=color, linewidth=1.5)
            axes[j].set_ylim(0, 1)
            axes[j].set_ylabel(f"P(Regime {j})")

        axes[-1].set_xlabel(xlabel)

        if title is None:
            title = "Smoothed Regime Probabilities"
        fig.suptitle(title, fontweight="bold", fontsize=13)
        fig.tight_layout()

    return fig, axes


def plot_regime_shading(
    y: ArrayLike,
    results: MarkovSwitchingResultsBase,
    ax: Axes | None = None,
    title: str | None = None,
    xlabel: str = "Observation",
    ylabel: str = "Value",
    series_color: str | None = None,
    series_linewidth: float = 2.0,
    alpha_min: float = 0.05,
    alpha_max: float = 0.3,
    show_legend: bool = True,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot time series with regime-colored background shading.

    Background color at each time t is the most likely regime's color,
    with alpha proportional to the smoothed probability for visual
    representation of uncertainty.

    Parameters
    ----------
    y : ArrayLike
        Time series data to plot.
    results : MarkovSwitchingResultsBase
        Fitted Markov switching model results.
    ax : Axes | None
        Matplotlib axes to plot on. If None, creates new figure.
    title : str | None
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    series_color : str | None
        Color for the time series line.
    series_linewidth : float
        Line width for the series.
    alpha_min : float
        Minimum shading alpha.
    alpha_max : float
        Maximum shading alpha.
    show_legend : bool
        Whether to show the regime legend.
    figsize : tuple[float, float]
        Figure size.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    y_arr = np.asarray(y)
    k = results.k_regimes
    probs = results.smoothed_marginal_probabilities

    with use_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Plot the time series
        if series_color is None:
            series_color = REGIMES_COLORS["near_black"]
        ax.plot(y_arr, color=series_color, linewidth=series_linewidth, zorder=2)

        # Shade background by regime periods
        periods = results.regime_periods()
        for regime, start, end in periods:
            color = REGIMES_COLOR_CYCLE[regime % len(REGIMES_COLOR_CYCLE)]
            # Compute average probability in this period for alpha
            avg_prob = float(np.mean(probs[start:end, regime]))
            alpha = alpha_min + (alpha_max - alpha_min) * avg_prob
            ax.axvspan(start, end, facecolor=color, alpha=alpha, zorder=0)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is None:
            title = "Time Series with Regime Shading"
        ax.set_title(title)

        if show_legend:
            patches = []
            for j in range(k):
                color = REGIMES_COLOR_CYCLE[j % len(REGIMES_COLOR_CYCLE)]
                patches.append(
                    mpatches.Patch(color=color, alpha=0.3, label=f"Regime {j}")
                )
            ax.legend(handles=patches, loc="best")

        fig.tight_layout()

    return fig, ax


def plot_transition_matrix(
    results: MarkovSwitchingResultsBase,
    ax: Axes | None = None,
    title: str | None = None,
    cmap: str = "Blues",
    annotate: bool = True,
    fmt: str = ".3f",
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Plot heatmap of the transition probability matrix.

    Parameters
    ----------
    results : MarkovSwitchingResultsBase
        Fitted Markov switching model results.
    ax : Axes | None
        Matplotlib axes. If None, creates new figure.
    title : str | None
        Plot title.
    cmap : str
        Colormap for the heatmap.
    annotate : bool
        Whether to annotate cells with probability values.
    fmt : str
        Format string for annotations.
    figsize : tuple[float, float] | None
        Figure size. If None, auto-sized based on k_regimes.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes.
    """
    import matplotlib.pyplot as plt

    k = results.k_regimes
    P = results.regime_transition

    with use_style():
        if figsize is None:
            size = max(3, k * 1.5)
            figsize = (size, size)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        im = ax.imshow(P, cmap=cmap, vmin=0, vmax=1, aspect="equal")

        if annotate:
            for i in range(k):
                for j in range(k):
                    val = P[i, j]
                    # Use dark text on light cells, light on dark
                    text_color = "white" if val > 0.5 else REGIMES_COLORS["near_black"]
                    if (
                        results.restricted_transitions
                        and (i, j) in results.restricted_transitions
                    ):
                        text_color = REGIMES_COLORS["grey"]
                    ax.text(
                        j,
                        i,
                        f"{val:{fmt}}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=11,
                        fontweight="bold",
                    )

        # Highlight restricted (zero) cells
        if results.restricted_transitions:
            for (i, j), val in results.restricted_transitions.items():
                if val == 0.0:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=True,
                            facecolor=REGIMES_COLORS["light_grey"],
                            edgecolor=REGIMES_COLORS["grey"],
                            linewidth=1,
                            zorder=1,
                        )
                    )
                    ax.text(
                        j,
                        i,
                        "0",
                        ha="center",
                        va="center",
                        color=REGIMES_COLORS["grey"],
                        fontsize=11,
                    )

        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels([f"Regime {j}" for j in range(k)])
        ax.set_yticklabels([f"Regime {i}" for i in range(k)])
        ax.set_xlabel("To Regime (S$_{t-1}$)")
        ax.set_ylabel("From Regime (S$_t$)")

        if title is None:
            title = "Transition Probability Matrix"
        ax.set_title(title)

        fig.colorbar(im, ax=ax, label="Probability", shrink=0.8)
        fig.tight_layout()

    return fig, ax


def plot_parameter_time_series(
    results: MarkovSwitchingResultsBase,
    param_name: str | None = None,
    ax: Axes | NDArray[Any] | None = None,
    title: str | None = None,
    xlabel: str = "Observation",
    weighted: bool = False,
    show_regime_shading: bool = True,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Any]:
    """Plot regime-dependent parameter values over time.

    Shows theta(s_t) for each time point based on the most likely regime
    (step function), or the probability-weighted parameter if weighted=True.

    Parameters
    ----------
    results : MarkovSwitchingResultsBase
        Fitted Markov switching model results.
    param_name : str | None
        Name of the parameter to plot (must be a key in regime_params).
        If None, creates a panel with one subplot per switching parameter.
    ax : Axes | NDArray[Axes] | None
        Axes to plot on. If None, creates new figure.
    title : str | None
        Plot title.
    xlabel : str
        X-axis label.
    weighted : bool
        If True, plot probability-weighted parameter instead of step function.
    show_regime_shading : bool
        Whether to shade background by regime.
    figsize : tuple[float, float]
        Figure size.

    Returns
    -------
    tuple[Figure, Axes | NDArray[Axes]]
        Figure and axes.
    """
    import matplotlib.pyplot as plt

    k = results.k_regimes
    probs = results.smoothed_marginal_probabilities

    # Identify switching parameters (parameters that differ across regimes)
    if param_name is not None:
        param_names = [param_name]
    else:
        # Find all parameters that vary across regimes
        all_params = set()
        for params in results.regime_params.values():
            all_params.update(params.keys())

        switching_params = []
        for p in sorted(all_params):
            values = [
                results.regime_params[j].get(p)
                for j in range(k)
                if p in results.regime_params.get(j, {})
            ]
            if (
                len(values) >= 2
                and len({f"{v:.6f}" for v in values if v is not None}) > 1
            ):
                switching_params.append(p)

        param_names = switching_params if switching_params else list(all_params)[:1]

    n_params = len(param_names)

    with use_style():
        if ax is None:
            fig, axes = plt.subplots(
                n_params,
                1,
                figsize=(figsize[0], figsize[1] * n_params / 2),
                sharex=True,
                squeeze=False,
            )
            axes = axes.flatten()
        else:
            axes = np.asarray(ax).flatten()
            fig = axes[0].get_figure()

        x = np.arange(results.nobs)

        for idx, pname in enumerate(param_names):
            # Get per-regime values
            regime_values = {}
            for j in range(k):
                if pname in results.regime_params.get(j, {}):
                    regime_values[j] = results.regime_params[j][pname]

            if not regime_values:
                continue

            if weighted:
                # Probability-weighted parameter
                param_ts = np.zeros(results.nobs)
                for j, val in regime_values.items():
                    param_ts += probs[:, j] * val
                axes[idx].plot(
                    x,
                    param_ts,
                    color=REGIMES_COLORS["blue"],
                    linewidth=2.0,
                )
            else:
                # Step function based on most likely regime
                assignments = results.most_likely_regime
                param_ts = np.array(
                    [
                        regime_values.get(int(assignments[t]), np.nan)
                        for t in range(results.nobs)
                    ]
                )
                axes[idx].step(
                    x,
                    param_ts,
                    color=REGIMES_COLORS["blue"],
                    linewidth=2.0,
                    where="post",
                )

            # Horizontal reference lines for each regime value
            for j, val in regime_values.items():
                color = REGIMES_COLOR_CYCLE[j % len(REGIMES_COLOR_CYCLE)]
                axes[idx].axhline(
                    y=val,
                    color=color,
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.7,
                )
                # Label at right edge
                axes[idx].annotate(
                    f"Regime {j}: {val:.3f}",
                    xy=(1.02, val),
                    xycoords=("axes fraction", "data"),
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                    va="center",
                    clip_on=False,
                )

            # Regime shading
            if show_regime_shading:
                periods = results.regime_periods()
                for regime, start, end in periods:
                    color = REGIMES_COLOR_CYCLE[regime % len(REGIMES_COLOR_CYCLE)]
                    axes[idx].axvspan(
                        start,
                        end,
                        facecolor=color,
                        alpha=0.08,
                        zorder=0,
                    )

            axes[idx].set_ylabel(pname)

        axes[-1].set_xlabel(xlabel)

        if title is None:
            title = "Regime-Dependent Parameters Over Time"
        fig.suptitle(title, fontweight="bold", fontsize=13)
        fig.tight_layout()

    return fig, axes


def plot_ic(
    ic_table: Any,
    selected_k: int | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    criteria: list[str] | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> tuple[Figure, Axes]:
    """Plot information criteria vs number of regimes.

    Parameters
    ----------
    ic_table : pd.DataFrame
        DataFrame with columns K, AIC, BIC, HQIC.
    selected_k : int | None
        Selected number of regimes to highlight.
    ax : Axes | None
        Axes to plot on.
    title : str | None
        Plot title.
    criteria : list[str] | None
        Which criteria to plot. Default ["AIC", "BIC", "HQIC"].
    figsize : tuple[float, float]
        Figure size.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes.
    """
    import matplotlib.pyplot as plt

    if criteria is None:
        criteria = ["AIC", "BIC", "HQIC"]

    with use_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        colors = [REGIMES_COLORS["blue"], REGIMES_COLORS["red"], REGIMES_COLORS["teal"]]

        for i, crit in enumerate(criteria):
            if crit in ic_table.columns:
                color = colors[i % len(colors)]
                ax.plot(
                    ic_table["K"],
                    ic_table[crit],
                    marker="o",
                    color=color,
                    linewidth=2,
                    markersize=6,
                    label=crit,
                )

                # Mark minimum
                min_idx = ic_table[crit].idxmin()
                min_k = ic_table.loc[min_idx, "K"]
                min_val = ic_table.loc[min_idx, crit]
                ax.plot(
                    min_k,
                    min_val,
                    marker="*",
                    color=color,
                    markersize=15,
                    zorder=5,
                )

        if selected_k is not None:
            ax.axvline(
                x=selected_k,
                color=REGIMES_COLORS["grey"],
                linestyle="--",
                linewidth=0.8,
                alpha=0.7,
                label=f"Selected: K={selected_k}",
            )

        ax.set_xlabel("Number of Regimes (K)")
        ax.set_ylabel("Information Criterion")
        ax.set_xticks(ic_table["K"].values)

        if title is None:
            title = "Information Criteria vs Number of Regimes"
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()

    return fig, ax
