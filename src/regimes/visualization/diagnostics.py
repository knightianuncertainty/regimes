"""PcGive-style misspecification diagnostic plots.

This module provides visualization tools for regression diagnostics,
mimicking the style of OxMetrics/PcGive. The four main diagnostic plots are:
1. Actual vs Fitted - time series of y and fitted values
2. Scaled Residuals - vertical index plot of residuals/sigma
3. Residual Distribution - histogram with standard normal overlay
4. Residual ACF/PACF - autocorrelation and partial autocorrelation functions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from regimes.visualization.style import REGIMES_COLORS, use_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray

    from regimes.results.base import RegressionResultsBase


def plot_actual_fitted(
    results: RegressionResultsBase,
    endog: ArrayLike | None = None,
    time_index: ArrayLike | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    xlabel: str = "Observation",
    ylabel: str = "Value",
    actual_color: str | None = None,
    fitted_color: str | None = None,
    actual_linewidth: float = 1.5,
    fitted_linewidth: float = 2.0,
    actual_alpha: float = 0.7,
    fitted_alpha: float = 0.9,
    show_legend: bool = True,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot actual vs fitted values over time.

    Parameters
    ----------
    results : RegressionResultsBase
        Results object containing fittedvalues and resid.
    endog : ArrayLike | None
        Dependent variable (actual values). If None, reconstructed as
        fittedvalues + resid.
    time_index : ArrayLike | None
        Time index for x-axis. If None, uses observation numbers.
    ax : Axes | None
        Matplotlib axes to plot on. If None, creates new figure and axes.
    title : str | None
        Plot title. If None, uses "Actual vs Fitted".
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    actual_color : str
        Color for actual values line.
    fitted_color : str
        Color for fitted values line.
    actual_linewidth : float
        Line width for actual values.
    fitted_linewidth : float
        Line width for fitted values.
    actual_alpha : float
        Alpha (transparency) for actual values line.
    fitted_alpha : float
        Alpha (transparency) for fitted values line.
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
    >>> import regimes as rg
    >>> np.random.seed(42)
    >>> y = np.random.randn(100)
    >>> X = np.column_stack([np.ones(100), np.random.randn(100)])
    >>> results = rg.OLS(y, X, has_constant=False).fit()
    >>> fig, ax = rg.plot_actual_fitted(results)
    """
    import matplotlib.pyplot as plt

    # Apply palette defaults
    if actual_color is None:
        actual_color = REGIMES_COLORS["grey"]
    if fitted_color is None:
        fitted_color = REGIMES_COLORS["blue"]

    # Get fitted values and residuals
    fitted = np.asarray(results.fittedvalues)
    resid = np.asarray(results.resid)

    # Reconstruct actual values if not provided
    if endog is None:
        actual = fitted + resid
    else:
        actual = np.asarray(endog)

    n = len(fitted)

    # Create time index if not provided
    if time_index is None:
        # Check for _nobs_original attribute (AR/ADL models lose observations)
        nobs_original = getattr(results, "_nobs_original", None)
        if nobs_original is not None and nobs_original > n:
            # Offset to align with original time index
            offset = nobs_original - n
            time_index = np.arange(offset, nobs_original)
        else:
            time_index = np.arange(n)
    else:
        time_index = np.asarray(time_index)

    with use_style():
        # Create figure and axes if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()  # type: ignore[union-attr]

        # Plot actual and fitted
        ax.plot(
            time_index,
            actual,
            color=actual_color,
            linewidth=actual_linewidth,
            alpha=actual_alpha,
            label="Actual",
        )
        ax.plot(
            time_index,
            fitted,
            color=fitted_color,
            linewidth=fitted_linewidth,
            alpha=fitted_alpha,
            label="Fitted",
        )

        # Labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is None:
            title = "Actual vs Fitted"
        ax.set_title(title)

        if show_legend:
            ax.legend(loc="best")

    return fig, ax


def plot_scaled_residuals(
    results: RegressionResultsBase,
    time_index: ArrayLike | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    xlabel: str = "Observation",
    ylabel: str = "Scaled Residual",
    color: str | None = None,
    linewidth: float = 0.8,
    alpha: float = 0.8,
    show_bands: bool = True,
    band_color: str | None = None,
    band_alpha: float = 0.3,
    band_linestyle: str = "--",
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot scaled residuals as vertical index lines.

    Scaled residuals are computed as resid / sigma, which should be
    approximately N(0,1) under correct specification. The vertical line
    format makes it easier to spot autocorrelation patterns.

    Parameters
    ----------
    results : RegressionResultsBase
        Results object containing resid and sigma.
    time_index : ArrayLike | None
        Time index for x-axis. If None, uses observation numbers.
    ax : Axes | None
        Matplotlib axes to plot on. If None, creates new figure and axes.
    title : str | None
        Plot title. If None, uses "Scaled Residuals".
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    color : str
        Color for residual lines.
    linewidth : float
        Line width for residual lines.
    alpha : float
        Alpha (transparency) for residual lines.
    show_bands : bool
        Whether to show +/- 2 reference bands.
    band_color : str
        Color for reference bands.
    band_alpha : float
        Alpha for reference bands.
    band_linestyle : str
        Line style for reference bands.
    figsize : tuple[float, float]
        Figure size (width, height) in inches.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.

    Examples
    --------
    >>> import numpy as np
    >>> import regimes as rg
    >>> np.random.seed(42)
    >>> y = np.random.randn(100)
    >>> X = np.column_stack([np.ones(100), np.random.randn(100)])
    >>> results = rg.OLS(y, X, has_constant=False).fit()
    >>> fig, ax = rg.plot_scaled_residuals(results)
    """
    import matplotlib.pyplot as plt

    # Apply palette defaults
    if color is None:
        color = REGIMES_COLORS["blue"]
    if band_color is None:
        band_color = REGIMES_COLORS["red"]

    # Get residuals and sigma
    resid = np.asarray(results.resid)
    sigma = results.sigma

    # Compute scaled residuals
    scaled_resid = resid / sigma

    n = len(resid)

    # Create time index if not provided
    if time_index is None:
        nobs_original = getattr(results, "_nobs_original", None)
        if nobs_original is not None and nobs_original > n:
            offset = nobs_original - n
            time_index = np.arange(offset, nobs_original)
        else:
            time_index = np.arange(n)
    else:
        time_index = np.asarray(time_index)

    with use_style():
        # Create figure and axes if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()  # type: ignore[union-attr]

        # Plot scaled residuals as vertical lines
        ax.vlines(
            time_index,
            ymin=0,
            ymax=scaled_resid,
            colors=color,
            linewidth=linewidth,
            alpha=alpha,
        )

        # Plot zero line
        ax.axhline(y=0, color=REGIMES_COLORS["near_black"], linewidth=0.5)

        # Plot +/- 2 reference bands
        if show_bands:
            ax.axhline(
                y=2,
                color=band_color,
                linestyle=band_linestyle,
                alpha=band_alpha,
                linewidth=1.0,
            )
            ax.axhline(
                y=-2,
                color=band_color,
                linestyle=band_linestyle,
                alpha=band_alpha,
                linewidth=1.0,
            )

        # Labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is None:
            title = "Scaled Residuals"
        ax.set_title(title)

    return fig, ax


def plot_residual_distribution(
    results: RegressionResultsBase,
    ax: Axes | None = None,
    title: str | None = None,
    xlabel: str = "Scaled Residual",
    ylabel: str = "Density",
    bins: int | str = "auto",
    hist_color: str | None = None,
    hist_alpha: float = 0.7,
    normal_color: str | None = None,
    normal_linewidth: float = 2.0,
    show_normal: bool = True,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot histogram of scaled residuals with standard normal overlay.

    Scaled residuals (resid/sigma) should follow N(0,1) under correct
    model specification. This plot helps assess normality.

    Parameters
    ----------
    results : RegressionResultsBase
        Results object containing resid and sigma.
    ax : Axes | None
        Matplotlib axes to plot on. If None, creates new figure and axes.
    title : str | None
        Plot title. If None, uses "Residual Distribution".
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    bins : int | str
        Number of histogram bins or binning strategy.
    hist_color : str
        Color for histogram bars.
    hist_alpha : float
        Alpha (transparency) for histogram.
    normal_color : str
        Color for normal distribution overlay.
    normal_linewidth : float
        Line width for normal distribution.
    show_normal : bool
        Whether to show the N(0,1) overlay.
    figsize : tuple[float, float]
        Figure size (width, height) in inches.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.

    Examples
    --------
    >>> import numpy as np
    >>> import regimes as rg
    >>> np.random.seed(42)
    >>> y = np.random.randn(100)
    >>> X = np.column_stack([np.ones(100), np.random.randn(100)])
    >>> results = rg.OLS(y, X, has_constant=False).fit()
    >>> fig, ax = rg.plot_residual_distribution(results)
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    # Apply palette defaults
    if hist_color is None:
        hist_color = REGIMES_COLORS["blue"]
    if normal_color is None:
        normal_color = REGIMES_COLORS["red"]

    # Get residuals and sigma
    resid = np.asarray(results.resid)
    sigma = results.sigma

    # Compute scaled residuals
    scaled_resid = resid / sigma

    with use_style():
        # Create figure and axes if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()  # type: ignore[union-attr]

        # Plot histogram
        ax.hist(
            scaled_resid,
            bins=bins,
            density=True,
            color=hist_color,
            alpha=hist_alpha,
            edgecolor="white",
        )

        # Overlay standard normal
        if show_normal:
            x = np.linspace(scaled_resid.min() - 0.5, scaled_resid.max() + 0.5, 200)
            ax.plot(
                x,
                stats.norm.pdf(x),
                color=normal_color,
                linewidth=normal_linewidth,
                label="N(0,1)",
            )
            ax.legend(loc="best")

        # Labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is None:
            title = "Residual Distribution"
        ax.set_title(title)

    return fig, ax


def plot_residual_acf(
    results: RegressionResultsBase,
    nlags: int = 20,
    alpha: float = 0.05,
    ax: NDArray[Any] | None = None,
    title: str | None = None,
    bar_color: str | None = None,
    ci_color: str | None = None,
    ci_alpha: float = 0.15,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, NDArray[Any]]:
    """Plot ACF and PACF of residuals.

    Parameters
    ----------
    results : RegressionResultsBase
        Results object containing resid.
    nlags : int
        Number of lags to compute.
    alpha : float
        Significance level for confidence bands.
    ax : NDArray[Axes] | None
        Array of two axes for ACF and PACF. If None, creates new figure.
    title : str | None
        Overall figure title.
    bar_color : str
        Color for ACF/PACF bars.
    ci_color : str
        Color for confidence bands.
    ci_alpha : float
        Alpha for confidence band fill.
    figsize : tuple[float, float]
        Figure size (width, height) in inches.

    Returns
    -------
    tuple[Figure, NDArray[Axes]]
        Matplotlib figure and array of two axes (ACF, PACF).

    Examples
    --------
    >>> import numpy as np
    >>> import regimes as rg
    >>> np.random.seed(42)
    >>> y = np.random.randn(100)
    >>> X = np.column_stack([np.ones(100), np.random.randn(100)])
    >>> results = rg.OLS(y, X, has_constant=False).fit()
    >>> fig, axes = rg.plot_residual_acf(results, nlags=15)
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    from statsmodels.tsa.stattools import acf, pacf

    # Apply palette defaults
    if bar_color is None:
        bar_color = REGIMES_COLORS["blue"]
    if ci_color is None:
        ci_color = REGIMES_COLORS["teal"]

    # Get residuals
    resid = np.asarray(results.resid)
    n = len(resid)

    # Ensure nlags doesn't exceed sample size limits
    nlags = min(nlags, n // 2 - 1)

    # Compute ACF and PACF
    acf_values = acf(resid, nlags=nlags, fft=False)
    pacf_values = pacf(resid, nlags=nlags)

    # Confidence band (approximate for white noise)
    # Under null of no autocorrelation, acf ~ N(0, 1/n) for large n
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_bound = z_crit / np.sqrt(n)

    with use_style():
        # Create figure and axes if needed
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
        else:
            axes = np.asarray(ax)
            fig = axes[0].get_figure()  # type: ignore[union-attr]

        lags = np.arange(nlags + 1)

        # Plot ACF
        ax_acf = axes[0]
        ax_acf.bar(lags, acf_values, color=bar_color, width=0.6)
        ax_acf.axhline(y=0, color=REGIMES_COLORS["near_black"], linewidth=0.5)
        ax_acf.axhline(y=ci_bound, color=ci_color, linestyle="--", alpha=0.8)
        ax_acf.axhline(y=-ci_bound, color=ci_color, linestyle="--", alpha=0.8)
        ax_acf.fill_between(lags, -ci_bound, ci_bound, color=ci_color, alpha=ci_alpha)
        ax_acf.set_xlabel("Lag")
        ax_acf.set_ylabel("ACF")
        ax_acf.set_title("Autocorrelation Function")
        ax_acf.set_xlim(-0.5, nlags + 0.5)

        # Plot PACF
        ax_pacf = axes[1]
        ax_pacf.bar(lags, pacf_values, color=bar_color, width=0.6)
        ax_pacf.axhline(y=0, color=REGIMES_COLORS["near_black"], linewidth=0.5)
        ax_pacf.axhline(y=ci_bound, color=ci_color, linestyle="--", alpha=0.8)
        ax_pacf.axhline(y=-ci_bound, color=ci_color, linestyle="--", alpha=0.8)
        ax_pacf.fill_between(lags, -ci_bound, ci_bound, color=ci_color, alpha=ci_alpha)
        ax_pacf.set_xlabel("Lag")
        ax_pacf.set_ylabel("PACF")
        ax_pacf.set_title("Partial Autocorrelation Function")
        ax_pacf.set_xlim(-0.5, nlags + 0.5)

        # Set overall title if provided
        if title is not None:
            fig.suptitle(title)

        fig.tight_layout()

    return fig, axes


def plot_diagnostics(
    results: RegressionResultsBase,
    endog: ArrayLike | None = None,
    nlags: int = 20,
    alpha: float = 0.05,
    figsize: tuple[float, float] = (12, 8),
) -> tuple[Figure, NDArray[Any]]:
    """Plot PcGive-style misspecification diagnostic panel.

    Creates a 2x2 panel with:
    - Top left: Actual vs Fitted values
    - Top right: Residual Distribution histogram
    - Bottom left: Scaled Residuals (vertical index plot)
    - Bottom right: ACF and PACF (stacked)

    Parameters
    ----------
    results : RegressionResultsBase
        Results object from OLS, AR, or ADL model.
    endog : ArrayLike | None
        Dependent variable (actual values). If None, reconstructed as
        fittedvalues + resid.
    nlags : int
        Number of lags for ACF/PACF plots.
    alpha : float
        Significance level for ACF/PACF confidence bands.
    figsize : tuple[float, float]
        Figure size (width, height) in inches.

    Returns
    -------
    tuple[Figure, NDArray[Axes]]
        Matplotlib figure and 2x2 array of axes.

    Examples
    --------
    >>> import numpy as np
    >>> import regimes as rg
    >>> np.random.seed(42)
    >>> y = np.random.randn(100)
    >>> X = np.column_stack([np.ones(100), np.random.randn(100)])
    >>> results = rg.OLS(y, X, has_constant=False).fit()
    >>> fig, axes = rg.plot_diagnostics(results)

    >>> # Also available as method on results
    >>> fig, axes = results.plot_diagnostics()

    Notes
    -----
    This diagnostic panel mimics the standard output from OxMetrics/PcGive,
    providing a quick visual assessment of model fit and residual properties.
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    from statsmodels.tsa.stattools import acf, pacf

    # Define colors from palette
    primary_color = REGIMES_COLORS["blue"]
    secondary_color = REGIMES_COLORS["red"]
    ci_color = REGIMES_COLORS["teal"]
    actual_color = REGIMES_COLORS["grey"]
    near_black = REGIMES_COLORS["near_black"]

    with use_style():
        # Create 2x2 figure with custom layout
        # The ACF/PACF panel needs special handling (two subplots stacked)
        fig = plt.figure(figsize=figsize)

        # Create grid: 2 rows, 2 columns
        # But the bottom-right cell will have ACF and PACF stacked
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        ax_actual_fitted = fig.add_subplot(gs[0, 0])
        ax_distribution = fig.add_subplot(gs[0, 1])
        ax_scaled_resid = fig.add_subplot(gs[1, 0])

        # For ACF/PACF, create a nested gridspec
        gs_acf = gs[1, 1].subgridspec(2, 1, hspace=0.4)
        ax_acf = fig.add_subplot(gs_acf[0])
        ax_pacf = fig.add_subplot(gs_acf[1])

        # Get data
        fitted = np.asarray(results.fittedvalues)
        resid = np.asarray(results.resid)
        sigma = results.sigma
        n = len(resid)

        if endog is None:
            actual = fitted + resid
        else:
            actual = np.asarray(endog)

        # Create time index
        nobs_original = getattr(results, "_nobs_original", None)
        if nobs_original is not None and nobs_original > n:
            offset = nobs_original - n
            time_index = np.arange(offset, nobs_original)
        else:
            time_index = np.arange(n)

        # Scaled residuals
        scaled_resid = resid / sigma

        # Panel 1: Actual vs Fitted
        ax_actual_fitted.plot(
            time_index,
            actual,
            color=actual_color,
            linewidth=1.5,
            alpha=0.7,
            label="Actual",
        )
        ax_actual_fitted.plot(
            time_index,
            fitted,
            color=primary_color,
            linewidth=2.0,
            alpha=0.9,
            label="Fitted",
        )
        ax_actual_fitted.set_xlabel("Observation")
        ax_actual_fitted.set_ylabel("Value")
        ax_actual_fitted.set_title("Actual vs Fitted")
        ax_actual_fitted.legend(loc="best")

        # Panel 2: Residual Distribution
        ax_distribution.hist(
            scaled_resid,
            bins="auto",
            density=True,
            color=primary_color,
            alpha=0.7,
            edgecolor="white",
        )
        x_norm = np.linspace(scaled_resid.min() - 0.5, scaled_resid.max() + 0.5, 200)
        ax_distribution.plot(
            x_norm,
            stats.norm.pdf(x_norm),
            color=secondary_color,
            linewidth=2.0,
            label="N(0,1)",
        )
        ax_distribution.set_xlabel("Scaled Residual")
        ax_distribution.set_ylabel("Density")
        ax_distribution.set_title("Residual Distribution")
        ax_distribution.legend(loc="best")

        # Panel 3: Scaled Residuals
        ax_scaled_resid.vlines(
            time_index,
            ymin=0,
            ymax=scaled_resid,
            colors=primary_color,
            linewidth=0.8,
            alpha=0.8,
        )
        ax_scaled_resid.axhline(y=0, color=near_black, linewidth=0.5)
        ax_scaled_resid.axhline(
            y=2, color=secondary_color, linestyle="--", alpha=0.3, linewidth=1.0
        )
        ax_scaled_resid.axhline(
            y=-2, color=secondary_color, linestyle="--", alpha=0.3, linewidth=1.0
        )
        ax_scaled_resid.set_xlabel("Observation")
        ax_scaled_resid.set_ylabel("Scaled Residual")
        ax_scaled_resid.set_title("Scaled Residuals")

        # Panel 4: ACF and PACF
        nlags_actual = min(nlags, n // 2 - 1)
        acf_values = acf(resid, nlags=nlags_actual, fft=False)
        pacf_values = pacf(resid, nlags=nlags_actual)
        lags = np.arange(nlags_actual + 1)

        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_bound = z_crit / np.sqrt(n)

        # ACF
        ax_acf.bar(lags, acf_values, color=primary_color, width=0.6)
        ax_acf.axhline(y=0, color=near_black, linewidth=0.5)
        ax_acf.axhline(y=ci_bound, color=ci_color, linestyle="--", alpha=0.8)
        ax_acf.axhline(y=-ci_bound, color=ci_color, linestyle="--", alpha=0.8)
        ax_acf.fill_between(lags, -ci_bound, ci_bound, color=ci_color, alpha=0.15)
        ax_acf.set_ylabel("ACF")
        ax_acf.set_title("ACF")
        ax_acf.set_xlim(-0.5, nlags_actual + 0.5)

        # PACF
        ax_pacf.bar(lags, pacf_values, color=primary_color, width=0.6)
        ax_pacf.axhline(y=0, color=near_black, linewidth=0.5)
        ax_pacf.axhline(y=ci_bound, color=ci_color, linestyle="--", alpha=0.8)
        ax_pacf.axhline(y=-ci_bound, color=ci_color, linestyle="--", alpha=0.8)
        ax_pacf.fill_between(lags, -ci_bound, ci_bound, color=ci_color, alpha=0.15)
        ax_pacf.set_xlabel("Lag")
        ax_pacf.set_ylabel("PACF")
        ax_pacf.set_title("PACF")
        ax_pacf.set_xlim(-0.5, nlags_actual + 0.5)

        # Overall title
        model_name = getattr(results, "model_name", "Model")
        fig.suptitle(f"{model_name} Diagnostic Plots", fontsize=12, fontweight="bold")

    # Collect axes into array for return
    axes = np.array(
        [[ax_actual_fitted, ax_distribution], [ax_scaled_resid, None]], dtype=object
    )
    # Store ACF/PACF axes in a nested way
    axes[1, 1] = np.array([ax_acf, ax_pacf])

    return fig, axes
