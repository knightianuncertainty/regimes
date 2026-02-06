"""Visualization utilities for parameter estimates over time.

This module provides plotting functions for visualizing regime-specific
parameter estimates as step functions over the sample period.
"""

from __future__ import annotations

import re
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
    from numpy.typing import NDArray

    from regimes.models.ar import ARResults
    from regimes.models.ols import OLSResults
    from regimes.tests.bai_perron import BaiPerronResults, BaiPerronTest

    # Type alias for result objects this function can accept
    ResultType = OLSResults | ARResults | BaiPerronTest | BaiPerronResults


def _normalize_results_input(
    results: ResultType | list[ResultType] | dict[str, ResultType],
) -> dict[str, Any]:
    """Convert single/list/dict input to {label: result} dict.

    Parameters
    ----------
    results : ResultType | list[ResultType] | dict[str, ResultType]
        Input results in various formats.

    Returns
    -------
    dict[str, Any]
        Dictionary mapping labels to result objects.
    """
    if isinstance(results, dict):
        return results
    elif isinstance(results, list):
        return {f"Model {i + 1}": r for i, r in enumerate(results)}
    else:
        return {"Model": results}


def _parse_param_name(name: str) -> tuple[str, int | None]:
    """Parse parameter name to extract base name and regime number.

    Parameters
    ----------
    name : str
        Parameter name, possibly with regime suffix.

    Returns
    -------
    tuple[str, int | None]
        Base parameter name and regime number (1-indexed), or None if no regime.

    Examples
    --------
    >>> _parse_param_name("const_regime2")
    ('const', 2)
    >>> _parse_param_name("x1")
    ('x1', None)
    >>> _parse_param_name("y.L1_regime1")
    ('y.L1', 1)
    """
    match = re.match(r"^(.+)_regime(\d+)$", name)
    if match:
        return match.group(1), int(match.group(2))
    return name, None


def _get_base_param_names(result: Any) -> list[str]:
    """Extract unique base parameter names from a result object.

    Parameters
    ----------
    result : Any
        A results object (OLSResults, ARResults, BaiPerronTest, or BaiPerronResults).

    Returns
    -------
    list[str]
        List of unique base parameter names (without regime suffix).
    """
    # Handle BaiPerronTest - it has exog_break and q
    if hasattr(result, "exog_break") and hasattr(result, "q"):
        q = result.q
        if q == 1:
            return ["const"]
        return [f"beta{i}" for i in range(q)]

    # Handle BaiPerronResults - it has q but not exog_break
    if hasattr(result, "breaks_by_m") and hasattr(result, "q"):
        q = result.q
        if q == 1:
            return ["const"]
        return [f"beta{i}" for i in range(q)]

    param_names = getattr(result, "param_names", None)
    if param_names is None:
        nparams = len(getattr(result, "params", []))
        param_names = [f"x{i}" for i in range(nparams)]

    # Extract unique base names while preserving order
    seen = set()
    base_names = []
    for name in param_names:
        base_name, _ = _parse_param_name(name)
        if base_name not in seen:
            seen.add(base_name)
            base_names.append(base_name)

    return base_names


def _get_nobs(result: Any) -> int:
    """Get number of observations from a result object.

    Parameters
    ----------
    result : Any
        A results object.

    Returns
    -------
    int
        Number of observations.
    """
    # For regression results with breaks, use original nobs if available
    if hasattr(result, "_nobs_original") and result._nobs_original is not None:
        return result._nobs_original
    return getattr(result, "nobs", 0)


def _extract_regime_boundaries(result: Any, nobs: int) -> list[tuple[int, int]]:
    """Get (start, end) indices for each regime from result.

    Parameters
    ----------
    result : Any
        A results object.
    nobs : int
        Number of observations.

    Returns
    -------
    list[tuple[int, int]]
        List of (start, end) tuples for each regime.
    """
    breaks: list[int] | None = None

    # Check for _breaks attribute (OLSResults, ARResults)
    if hasattr(result, "_breaks") and result._breaks:
        breaks = list(result._breaks)
    # Check for break_indices (BaiPerronResults)
    elif hasattr(result, "break_indices") and result.break_indices:
        breaks = list(result.break_indices)

    if not breaks:
        return [(0, nobs)]

    boundaries = []
    sorted_breaks = sorted(breaks)
    prev = 0
    for bp in sorted_breaks:
        boundaries.append((prev, bp))
        prev = bp
    boundaries.append((prev, nobs))

    return boundaries


def _extract_param_data(
    result: Any, alpha: float = 0.05
) -> dict[str, list[dict[str, Any]]]:
    """Extract parameter data organized by base parameter name.

    Parameters
    ----------
    result : Any
        A results object.
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Dictionary mapping base parameter names to lists of segment data.
        Each segment has: regime, start, end, value, ci_lower, ci_upper.
    """
    from scipy import stats

    nobs = _get_nobs(result)

    # Handle BaiPerronTest and BaiPerronResults
    if hasattr(result, "exog_break") and hasattr(result, "get_regime_estimates"):
        return _extract_baiperron_param_data(result, alpha)
    elif hasattr(result, "breaks_by_m") and hasattr(result, "break_indices"):
        # BaiPerronResults - need to get the test object or extract from stored data
        return _extract_baiperron_results_param_data(result, alpha)

    # Handle regression results (OLSResults, ARResults)
    param_names = getattr(result, "param_names", None)
    params = getattr(result, "params", None)
    bse = getattr(result, "bse", None)
    df_resid = getattr(
        result, "df_resid", nobs - len(params) if params is not None else nobs
    )

    if params is None or param_names is None:
        return {}

    # Get confidence interval multiplier
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)

    # Get regime boundaries
    regime_boundaries = _extract_regime_boundaries(result, nobs)
    n_regimes = len(regime_boundaries)

    # Organize parameters by base name
    param_data: dict[str, list[dict[str, Any]]] = {}

    for i, name in enumerate(param_names):
        base_name, regime_num = _parse_param_name(name)

        if base_name not in param_data:
            param_data[base_name] = []

        value = float(params[i])
        se = float(bse[i]) if bse is not None else 0.0
        ci_lower = value - t_crit * se
        ci_upper = value + t_crit * se

        # Determine regime boundaries for this parameter
        if regime_num is not None:
            # Parameter has explicit regime suffix
            regime_idx = regime_num - 1  # Convert to 0-indexed
            if regime_idx < len(regime_boundaries):
                start, end = regime_boundaries[regime_idx]
            else:
                start, end = 0, nobs
        elif n_regimes == 1:
            # No breaks, single regime
            start, end = 0, nobs
        else:
            # Parameter without regime suffix in a model with breaks
            # This means it's constant across all regimes
            start, end = 0, nobs

        param_data[base_name].append(
            {
                "regime": regime_num or 1,
                "start": start,
                "end": end,
                "value": value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )

    return param_data


def _extract_baiperron_param_data(
    test: Any, alpha: float = 0.05
) -> dict[str, list[dict[str, Any]]]:
    """Extract parameter data from a BaiPerronTest object.

    Parameters
    ----------
    test : BaiPerronTest
        The Bai-Perron test object.
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Dictionary mapping parameter names to segment data.
    """
    from scipy import stats

    nobs = test.nobs
    q = test.q

    # Get break indices - prefer those from a results attribute if available
    break_indices: list[int] = []
    if hasattr(test, "_selected_breaks"):
        break_indices = list(test._selected_breaks)
    elif hasattr(test, "break_indices"):
        break_indices = list(test.break_indices)

    # Get regime estimates
    regime_estimates = test.get_regime_estimates(break_indices)

    # Compute boundaries
    boundaries = [0] + list(break_indices) + [nobs]

    # Parameter names
    if q == 1:
        param_names = ["const"]
    else:
        param_names = [f"beta{i}" for i in range(q)]

    param_data: dict[str, list[dict[str, Any]]] = {name: [] for name in param_names}

    for regime_idx, (beta, ssr) in enumerate(regime_estimates):
        start = boundaries[regime_idx]
        end = boundaries[regime_idx + 1]
        n_seg = end - start

        # Compute standard errors for this regime
        # sigma^2 = SSR / (n - q)
        df = n_seg - q
        if df > 0 and ssr > 0:
            sigma2 = ssr / df

            # Get the design matrix for this segment
            if test.exog_break is not None:
                X_seg = test.exog_break[start:end]
                try:
                    XtX_inv = np.linalg.inv(X_seg.T @ X_seg)
                    se = np.sqrt(np.diag(sigma2 * XtX_inv))
                except np.linalg.LinAlgError:
                    se = np.full(q, np.nan)
            else:
                se = np.full(q, np.nan)

            # t critical value
            t_crit = stats.t.ppf(1 - alpha / 2, df)
        else:
            se = np.full(q, np.nan)
            t_crit = 0.0

        # Store parameter data
        for j, name in enumerate(param_names):
            value = float(beta[j]) if j < len(beta) else np.nan
            se_j = float(se[j]) if j < len(se) else np.nan
            ci_lower = value - t_crit * se_j if not np.isnan(se_j) else np.nan
            ci_upper = value + t_crit * se_j if not np.isnan(se_j) else np.nan

            param_data[name].append(
                {
                    "regime": regime_idx + 1,
                    "start": start,
                    "end": end,
                    "value": value,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            )

    return param_data


def _extract_baiperron_results_param_data(
    results: Any, alpha: float = 0.05
) -> dict[str, list[dict[str, Any]]]:
    """Extract parameter data from BaiPerronResults.

    This is a simplified version that uses stored SSR values.

    Parameters
    ----------
    results : BaiPerronResults
        The Bai-Perron results object.
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Dictionary mapping parameter names to segment data.
    """
    # BaiPerronResults stores break_indices and n_breaks
    # but doesn't store the actual parameter estimates
    # We need to return empty data or placeholder values

    nobs = results.nobs
    break_indices = list(results.break_indices)
    q = getattr(results, "q", 1)

    if q == 1:
        param_names = ["const"]
    else:
        param_names = [f"beta{i}" for i in range(q)]

    # Compute boundaries
    boundaries = [0] + list(break_indices) + [nobs]

    param_data: dict[str, list[dict[str, Any]]] = {name: [] for name in param_names}

    # Without the original test object, we can't compute estimates
    # Return placeholder data with NaN values
    for regime_idx in range(len(boundaries) - 1):
        start = boundaries[regime_idx]
        end = boundaries[regime_idx + 1]

        for name in param_names:
            param_data[name].append(
                {
                    "regime": regime_idx + 1,
                    "start": start,
                    "end": end,
                    "value": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                }
            )

    return param_data


def _get_all_base_params(models: dict[str, Any]) -> list[str]:
    """Get all unique base parameter names across models.

    Parameters
    ----------
    models : dict[str, Any]
        Dictionary of model results.

    Returns
    -------
    list[str]
        List of unique base parameter names preserving order.
    """
    seen = set()
    all_params = []

    for result in models.values():
        for name in _get_base_param_names(result):
            if name not in seen:
                seen.add(name)
                all_params.append(name)

    return all_params


def plot_params_over_time(
    results: ResultType | list[ResultType] | dict[str, ResultType],
    params: Sequence[str] | None = None,
    alpha: float = 0.05,
    ax: Axes | Sequence[Axes] | None = None,
    figsize: tuple[float, float] = (10, 5),
    ncols: int = 1,
    show_breaks: bool = True,
    break_color: str | None = None,
    break_linestyle: str = "--",
    ci_alpha: float = 0.15,
    colors: Sequence[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
) -> tuple[Figure, NDArray[Any] | Axes]:
    """Plot parameter estimates as step functions over time.

    Displays regime-specific parameter estimates with confidence intervals,
    showing how coefficients change at structural break points. Supports
    overlaying multiple models for comparison.

    Parameters
    ----------
    results : ResultType | list[ResultType] | dict[str, ResultType]
        Results to plot. Can be:
        - Single result object (OLSResults, ARResults, or BaiPerronTest)
        - List of results (auto-labeled as "Model 1", "Model 2", etc.)
        - Dict mapping labels to results for custom labels
    params : Sequence[str] | None
        Parameter names to plot. If None, plots all parameters.
    alpha : float
        Significance level for confidence intervals (default 0.05 = 95% CI).
    ax : Axes | Sequence[Axes] | None
        Matplotlib axes to plot on. If None, creates new figure.
        Can be a single Axes (for single parameter) or array of Axes.
    figsize : tuple[float, float]
        Figure size (width, height) in inches.
    ncols : int
        Number of columns in subplot grid.
    show_breaks : bool
        Whether to show vertical lines at break points.
    break_color : str
        Color for break lines.
    break_linestyle : str
        Line style for break lines.
    ci_alpha : float
        Alpha (transparency) for confidence interval shading.
    colors : Sequence[str] | None
        Colors for each model. If None, uses matplotlib default cycle.
    show_legend : bool
        Whether to show legend (only shown when multiple models).
    title : str | None
        Overall figure title. If None, no title is added.

    Returns
    -------
    tuple[Figure, NDArray | Axes]
        Matplotlib figure and axes objects.

    Examples
    --------
    >>> import numpy as np
    >>> from regimes import OLS, plot_params_over_time
    >>> np.random.seed(42)
    >>> y = np.concatenate([np.random.randn(100), np.random.randn(100) + 2])
    >>> X = np.column_stack([np.ones(200), np.random.randn(200)])
    >>>
    >>> # Compare OLS with different break assumptions
    >>> ols_no_break = OLS(y, X).fit()
    >>> ols_known_break = OLS(y, X, breaks=[100]).fit()
    >>>
    >>> fig, axes = plot_params_over_time({
    ...     "No breaks": ols_no_break,
    ...     "Break at 100": ols_known_break,
    ... })

    Notes
    -----
    - Parameters are plotted as step functions using `drawstyle='steps-post'`
    - Confidence intervals use shaded regions matching the step function
    - Break lines are shown for each model that has breaks
    - When plotting multiple models, different colors distinguish them
    """
    import matplotlib.pyplot as plt

    # Apply palette defaults
    if break_color is None:
        break_color = REGIMES_COLORS["grey"]
    if colors is None:
        colors = REGIMES_COLOR_CYCLE

    # Normalize input to dict
    models = _normalize_results_input(results)

    # Get all base parameter names across models
    all_params = _get_all_base_params(models)
    params_to_plot = list(params) if params is not None else all_params

    if not params_to_plot:
        raise ValueError("No parameters to plot")

    # Create subplots
    n_params = len(params_to_plot)
    nrows = int(np.ceil(n_params / ncols))

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

        # Track which breaks have been plotted (to avoid duplicate lines)
        plotted_breaks: set[tuple[int, int]] = set()

        # Plot each parameter
        for idx, param_name in enumerate(params_to_plot):
            if idx >= len(axes_flat):
                break

            ax_curr = axes_flat[idx]

            for model_idx, (label, result) in enumerate(models.items()):
                param_data = _extract_param_data(result, alpha)

                if param_name not in param_data:
                    continue

                segments = param_data[param_name]
                if not segments:
                    continue

                # Get nobs for this model
                nobs = _get_nobs(result)
                if nobs == 0:
                    continue

                # Build step-function arrays
                # We need to handle potentially overlapping segments
                # For models with breaks, segments are non-overlapping
                # For models without breaks, there's one segment covering all data

                x_vals = []
                y_vals = []
                ci_low_vals = []
                ci_high_vals = []

                for seg in sorted(segments, key=lambda s: s["start"]):
                    start = seg["start"]
                    end = seg["end"]
                    value = seg["value"]
                    ci_lower = seg["ci_lower"]
                    ci_upper = seg["ci_upper"]

                    # Add points for step function
                    x_vals.extend([start, end])
                    y_vals.extend([value, value])
                    ci_low_vals.extend([ci_lower, ci_lower])
                    ci_high_vals.extend([ci_upper, ci_upper])

                if not x_vals:
                    continue

                # Convert to arrays
                x_arr = np.array(x_vals)
                y_arr = np.array(y_vals)
                ci_low_arr = np.array(ci_low_vals)
                ci_high_arr = np.array(ci_high_vals)

                # Get color from color cycle
                color = colors[model_idx % len(colors)]

                # Plot the step function
                ax_curr.plot(
                    x_arr, y_arr, color=color, label=label, drawstyle="steps-post"
                )

                # Plot confidence interval
                if not np.all(np.isnan(ci_low_arr)):
                    ax_curr.fill_between(
                        x_arr,
                        ci_low_arr,
                        ci_high_arr,
                        color=color,
                        alpha=ci_alpha,
                        step="post",
                    )

                # Plot break lines for this model
                if show_breaks:
                    breaks = None
                    if hasattr(result, "_breaks") and result._breaks:
                        breaks = result._breaks
                    elif hasattr(result, "break_indices") and result.break_indices:
                        breaks = result.break_indices

                    if breaks:
                        for bp in breaks:
                            # Avoid duplicate break lines
                            key = (idx, bp)
                            if key not in plotted_breaks:
                                ax_curr.axvline(
                                    bp,
                                    color=break_color,
                                    linestyle=break_linestyle,
                                    linewidth=0.8,
                                    alpha=0.7,
                                )
                                plotted_breaks.add(key)

            # Set subplot title
            ax_curr.set_title(param_name)
            ax_curr.set_xlabel("Observation")
            ax_curr.set_ylabel("Estimate")

            # Add legend only for multiple models
            if show_legend and len(models) > 1:
                ax_curr.legend(loc="best")

        # Hide unused subplots
        for idx in range(n_params, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # Set overall title
        if title is not None:
            fig.suptitle(title)

        # Adjust layout
        fig.tight_layout()

    # Return appropriate type
    if n_params == 1 and ax is not None and not isinstance(ax, np.ndarray):
        return fig, axes_flat[0]

    return fig, axes
