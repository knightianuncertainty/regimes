"""Tests for rolling/recursive coefficient visualization."""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

import regimes as rg
from regimes.visualization.rolling import plot_rolling_coefficients

# Use non-interactive backend for testing
matplotlib.use("Agg")


# =============================================================================
# Basic Tests
# =============================================================================


class TestPlotRollingCoefficientsBasic:
    """Basic tests for plot_rolling_coefficients."""

    def test_returns_figure_and_axes(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test that function returns Figure and Axes."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()
        fig, axes = plot_rolling_coefficients(results)

        assert fig is not None
        assert axes is not None
        plt.close(fig)

    def test_rolling_ols_results(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test with RollingOLS results."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()
        fig, axes = plot_rolling_coefficients(results)

        assert fig is not None
        plt.close(fig)

    def test_recursive_ols_results(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test with RecursiveOLS results."""
        y, X = regression_data
        results = rg.RecursiveOLS(y, X, min_nobs=20).fit()
        fig, axes = plot_rolling_coefficients(results)

        assert fig is not None
        plt.close(fig)

    def test_rolling_ar_results(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test with RollingAR results."""
        y = ar1_data
        results = rg.RollingAR(y, lags=1, window=40).fit()
        fig, axes = plot_rolling_coefficients(results)

        assert fig is not None
        plt.close(fig)

    def test_recursive_ar_results(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test with RecursiveAR results."""
        y = ar1_data
        results = rg.RecursiveAR(y, lags=1, min_nobs=30).fit()
        fig, axes = plot_rolling_coefficients(results)

        assert fig is not None
        plt.close(fig)

    def test_rolling_adl_results(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test with RollingADL results."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40).fit()
        fig, axes = plot_rolling_coefficients(results)

        assert fig is not None
        plt.close(fig)

    def test_recursive_adl_results(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test with RecursiveADL results."""
        y, x = adl_data
        results = rg.RecursiveADL(y, x, lags=1, exog_lags=1, min_nobs=30).fit()
        fig, axes = plot_rolling_coefficients(results)

        assert fig is not None
        plt.close(fig)


# =============================================================================
# Variable Selection Tests
# =============================================================================


class TestVariableSelection:
    """Test variable selection for plotting."""

    def test_select_all_variables(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test plotting all variables (default)."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()
        fig, axes = plot_rolling_coefficients(results)

        # Should have subplots for all parameters
        assert fig is not None
        plt.close(fig)

    def test_select_subset_variables(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test plotting subset of variables."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()
        fig, axes = plot_rolling_coefficients(results, variables=["x0"])

        assert fig is not None
        plt.close(fig)

    def test_select_multiple_variables(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test selecting multiple specific variables."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40).fit()
        # ADL has: const, y.L1, x0, x0.L1
        fig, axes = plot_rolling_coefficients(results, variables=["y.L1", "x0"])

        assert fig is not None
        plt.close(fig)

    def test_invalid_variable_raises(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test error for invalid variable name."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()

        with pytest.raises(ValueError, match="not found"):
            plot_rolling_coefficients(results, variables=["nonexistent"])


# =============================================================================
# Alpha/Confidence Interval Tests
# =============================================================================


class TestConfidenceIntervals:
    """Test confidence interval settings."""

    def test_default_alpha(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test default alpha (0.05)."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()
        fig, axes = plot_rolling_coefficients(results)

        assert fig is not None
        plt.close(fig)

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10, 0.20])
    def test_various_alpha_values(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
        alpha: float,
    ) -> None:
        """Test various alpha values."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()
        fig, axes = plot_rolling_coefficients(results, alpha=alpha)

        assert fig is not None
        plt.close(fig)

    def test_ci_alpha_transparency(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test CI shading transparency."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()
        fig, axes = plot_rolling_coefficients(results, ci_alpha=0.3)

        assert fig is not None
        plt.close(fig)


# =============================================================================
# Subplot Layout Tests
# =============================================================================


class TestSubplotLayout:
    """Test subplot layout options."""

    def test_single_column(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test single column layout (default)."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40).fit()
        fig, axes = plot_rolling_coefficients(results, ncols=1)

        assert fig is not None
        plt.close(fig)

    def test_two_columns(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test two column layout."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40).fit()
        fig, axes = plot_rolling_coefficients(results, ncols=2)

        assert fig is not None
        plt.close(fig)

    def test_custom_figsize(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test custom figure size."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()
        fig, axes = plot_rolling_coefficients(results, figsize=(14, 10))

        assert fig is not None
        plt.close(fig)


# =============================================================================
# Custom Axes Tests
# =============================================================================


class TestCustomAxes:
    """Test plotting on provided axes."""

    def test_single_ax(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test providing single axes for single variable."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()

        custom_fig, custom_ax = plt.subplots()
        fig, ax = plot_rolling_coefficients(results, variables=["x0"], ax=custom_ax)

        assert fig is custom_fig
        plt.close(fig)

    def test_array_of_axes(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test providing array of axes."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40).fit()

        custom_fig, custom_axes = plt.subplots(2, 2)
        fig, axes = plot_rolling_coefficients(results, ax=custom_axes)

        assert fig is custom_fig
        plt.close(fig)


# =============================================================================
# Appearance Options
# =============================================================================


class TestAppearanceOptions:
    """Test appearance customization."""

    def test_show_zero_line(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test zero line option."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()

        # With zero line (default)
        fig1, _ = plot_rolling_coefficients(results, show_zero_line=True)
        plt.close(fig1)

        # Without zero line
        fig2, _ = plot_rolling_coefficients(results, show_zero_line=False)
        plt.close(fig2)

    def test_custom_color(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test custom plot color."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()
        fig, axes = plot_rolling_coefficients(results, color="red")

        assert fig is not None
        plt.close(fig)

    def test_custom_title(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test custom title."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()
        fig, axes = plot_rolling_coefficients(results, title="My Custom Title")

        # Check suptitle was set
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "My Custom Title"
        plt.close(fig)

    def test_default_title_rolling(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test default title for rolling estimation."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()
        fig, axes = plot_rolling_coefficients(results)

        assert fig._suptitle is not None
        assert "Rolling" in fig._suptitle.get_text()
        assert "window=30" in fig._suptitle.get_text()
        plt.close(fig)

    def test_default_title_recursive(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test default title for recursive estimation."""
        y, X = regression_data
        results = rg.RecursiveOLS(y, X, min_nobs=20).fit()
        fig, axes = plot_rolling_coefficients(results)

        assert fig._suptitle is not None
        assert "Recursive" in fig._suptitle.get_text()
        assert "min_nobs=20" in fig._suptitle.get_text()
        plt.close(fig)


# =============================================================================
# Results Method Tests
# =============================================================================


class TestResultsMethod:
    """Test plot_coefficients method on results objects."""

    def test_rolling_ols_method(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test RollingOLS results method."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()
        fig, axes = results.plot_coefficients()

        assert fig is not None
        plt.close(fig)

    def test_recursive_ols_method(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test RecursiveOLS results method."""
        y, X = regression_data
        results = rg.RecursiveOLS(y, X, min_nobs=20).fit()
        fig, axes = results.plot_coefficients()

        assert fig is not None
        plt.close(fig)

    def test_rolling_ar_method(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test RollingAR results method."""
        y = ar1_data
        results = rg.RollingAR(y, lags=1, window=40).fit()
        fig, axes = results.plot_coefficients()

        assert fig is not None
        plt.close(fig)

    def test_rolling_adl_method(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test RollingADL results method."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40).fit()
        fig, axes = results.plot_coefficients()

        assert fig is not None
        plt.close(fig)

    def test_method_with_parameters(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test method accepts parameters."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()
        fig, axes = results.plot_coefficients(
            variables=["x0"], alpha=0.10, ncols=1, figsize=(8, 4)
        )

        assert fig is not None
        plt.close(fig)


# =============================================================================
# API Export Tests
# =============================================================================


class TestAPIExports:
    """Tests for API-level exports."""

    def test_plot_rolling_coefficients_in_api(self) -> None:
        """Test that plot_rolling_coefficients is exported from regimes."""
        assert hasattr(rg, "plot_rolling_coefficients")

    def test_function_callable_from_api(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test function is callable via API."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()
        fig, axes = rg.plot_rolling_coefficients(results)

        assert fig is not None
        plt.close(fig)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases for rolling visualization."""

    def test_small_window(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test with small window size."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=10).fit()
        fig, axes = plot_rolling_coefficients(results)

        assert fig is not None
        plt.close(fig)

    def test_large_window(self, rng: np.random.Generator) -> None:
        """Test with large window size."""
        n = 500
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [1, 2] + rng.standard_normal(n)
        results = rg.RollingOLS(y, X, window=200).fit()
        fig, axes = plot_rolling_coefficients(results)

        assert fig is not None
        plt.close(fig)

    def test_single_parameter(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test with single parameter model."""
        y = ar1_data
        results = rg.RollingAR(y, lags=1, window=40, trend="n").fit()
        fig, axes = plot_rolling_coefficients(results)

        assert fig is not None
        plt.close(fig)

    def test_many_parameters(self, rng: np.random.Generator) -> None:
        """Test with many parameters."""
        n = 200
        X = np.column_stack([np.ones(n)] + [rng.standard_normal(n) for _ in range(5)])
        y = rng.standard_normal(n)
        results = rg.RollingOLS(y, X, window=60).fit()
        fig, axes = plot_rolling_coefficients(results, ncols=2)

        assert fig is not None
        plt.close(fig)
