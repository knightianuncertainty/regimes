"""Visualization utilities for rolling and recursive estimation results.

This module provides plotting functions for visualizing rolling and
recursive coefficient estimates over time with confidence bands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from regimes.visualization.style import REGIMES_COLORS, use_style

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from regimes.rolling.base import RollingResultsBase


def plot_rolling_coefficients(
    results: RollingResultsBase,
    variables: Sequence[str] | None = None,
    alpha: float = 0.05,
    ax: Axes | Sequence[Axes] | None = None,
    figsize: tuple[float, float] = (10, 5),
    ncols: int = 1,
    show_zero_line: bool = True,
    ci_alpha: float = 0.15,
    color: str | None = None,
    title: str | None = None,
) -> tuple[Figure, NDArray[Any] | Axes]:
    """Plot rolling or recursive coefficient estimates with confidence bands.

    Displays time-varying parameter estimates from rolling or recursive
    estimation with confidence intervals.

    Parameters
    ----------
    results : RollingResultsBase
        Results from RollingOLS, RecursiveOLS, RollingAR, or RecursiveAR.
    variables : Sequence[str] | None
        Which variables to plot. If None, plots all parameters.
    alpha : float
        Significance level for confidence intervals (default 0.05 = 95% CI).
    ax : Axes | Sequence[Axes] | None
        Matplotlib axes to plot on. If None, creates new figure.
    figsize : tuple[float, float]
        Figure size (width, height) in inches.
    ncols : int
        Number of columns in subplot grid.
    show_zero_line : bool
        Whether to show a horizontal line at y=0.
    ci_alpha : float
        Alpha (transparency) for confidence interval shading.
    color : str | None
        Color for the plot. If None, uses matplotlib default.
    title : str | None
        Overall figure title. If None, generates a default title.

    Returns
    -------
    tuple[Figure, NDArray | Axes]
        Matplotlib figure and axes objects.

    Examples
    --------
    >>> import numpy as np
    >>> from regimes.rolling import RollingOLS
    >>> from regimes.visualization.rolling import plot_rolling_coefficients
    >>> np.random.seed(42)
    >>> n = 200
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = X @ [1, 2] + np.random.randn(n)
    >>> rolling = RollingOLS(y, X, window=60)
    >>> results = rolling.fit()
    >>> fig, axes = plot_rolling_coefficients(results)

    Notes
    -----
    - Coefficient estimates are plotted as lines over time
    - Confidence intervals use shaded regions
    - NaN values (before first valid estimate) are not plotted
    """
    import matplotlib.pyplot as plt

    # Apply palette defaults
    if color is None:
        color = REGIMES_COLORS["blue"]

    # Determine which variables to plot
    param_names = results.param_names
    if variables is None:
        vars_to_plot = param_names
    else:
        # Validate variable names
        vars_to_plot = []
        for v in variables:
            if v in param_names:
                vars_to_plot.append(v)
            else:
                raise ValueError(f"Variable '{v}' not found. Available: {param_names}")

    if not vars_to_plot:
        raise ValueError("No variables to plot")

    # Create subplots
    n_vars = len(vars_to_plot)
    nrows = int(np.ceil(n_vars / ncols))

    with use_style():
        if ax is None:
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        else:
            if isinstance(ax, np.ndarray):
                axes = ax.reshape(-1, ncols) if ax.ndim == 1 else ax
                fig = axes.flat[0].get_figure()
            else:
                # Single axes provided
                axes = np.array([[ax]])
                fig = ax.get_figure()

        # Flatten axes for easy iteration
        axes_flat = axes.flatten()

        # Get confidence intervals
        ci = results.conf_int(alpha=alpha)

        # Time index
        t = np.arange(results.nobs)

        # Plot each variable
        for idx, var_name in enumerate(vars_to_plot):
            if idx >= len(axes_flat):
                break

            ax_curr = axes_flat[idx]
            var_idx = param_names.index(var_name)

            # Get data for this variable
            params = results.params[:, var_idx]
            ci_lower = ci[:, var_idx, 0]
            ci_upper = ci[:, var_idx, 1]

            # Find valid (non-NaN) values
            valid_mask = ~np.isnan(params)
            t_valid = t[valid_mask]
            params_valid = params[valid_mask]
            ci_lower_valid = ci_lower[valid_mask]
            ci_upper_valid = ci_upper[valid_mask]

            if len(t_valid) == 0:
                ax_curr.set_title(f"{var_name} (no valid estimates)")
                continue

            # Plot coefficient estimates
            ax_curr.plot(t_valid, params_valid, color=color, label=var_name)

            # Plot confidence interval
            ax_curr.fill_between(
                t_valid,
                ci_lower_valid,
                ci_upper_valid,
                color=color,
                alpha=ci_alpha,
            )

            # Add zero line
            if show_zero_line:
                ax_curr.axhline(
                    y=0,
                    color=REGIMES_COLORS["grey"],
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.7,
                )

            # Labels
            ax_curr.set_title(var_name)
            ax_curr.set_xlabel("Observation")
            ax_curr.set_ylabel("Estimate")

        # Hide unused subplots
        for idx in range(n_vars, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # Set overall title
        if title is None:
            est_type = "Rolling" if results.is_rolling else "Recursive"
            if results.is_rolling:
                title = f"{est_type} Estimates (window={results.window})"
            else:
                title = f"{est_type} Estimates (min_nobs={results.min_nobs})"

        fig.suptitle(title)

        # Adjust layout
        fig.tight_layout()

    # Return appropriate type
    if n_vars == 1 and ax is not None and not isinstance(ax, np.ndarray):
        return fig, axes_flat[0]

    return fig, axes
