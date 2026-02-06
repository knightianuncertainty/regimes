"""Tests for PcGive-style diagnostic plots."""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import regimes as rg
from regimes.visualization.diagnostics import (
    plot_actual_fitted,
    plot_diagnostics,
    plot_residual_acf,
    plot_residual_distribution,
    plot_scaled_residuals,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


class TestPlotActualFitted:
    """Tests for plot_actual_fitted function."""

    def test_returns_figure_and_axes(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test that function returns Figure and Axes."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, ax = plot_actual_fitted(results)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_custom_axes(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test plotting on provided axes."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        custom_fig, custom_ax = plt.subplots()
        fig, ax = plot_actual_fitted(results, ax=custom_ax)

        assert fig is custom_fig
        assert ax is custom_ax
        plt.close(fig)

    def test_with_endog_provided(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test with explicit endog array."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, ax = plot_actual_fitted(results, endog=y)

        assert fig is not None
        plt.close(fig)

    def test_with_ar_model(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test with AR model results."""
        y, break_point = ar1_data_with_break
        results = rg.AR(y, lags=1).fit()

        fig, ax = plot_actual_fitted(results)

        assert fig is not None
        plt.close(fig)

    def test_custom_title(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test custom title parameter."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, ax = plot_actual_fitted(results, title="My Custom Title")

        assert ax.get_title() == "My Custom Title"
        plt.close(fig)


class TestPlotScaledResiduals:
    """Tests for plot_scaled_residuals function."""

    def test_returns_figure_and_axes(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test that function returns Figure and Axes."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, ax = plot_scaled_residuals(results)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_custom_axes(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test plotting on provided axes."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        custom_fig, custom_ax = plt.subplots()
        fig, ax = plot_scaled_residuals(results, ax=custom_ax)

        assert fig is custom_fig
        assert ax is custom_ax
        plt.close(fig)

    def test_no_bands(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test with reference bands disabled."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, ax = plot_scaled_residuals(results, show_bands=False)

        assert fig is not None
        plt.close(fig)

    def test_with_ar_model(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test with AR model results."""
        y, _ = ar1_data_with_break
        results = rg.AR(y, lags=1).fit()

        fig, ax = plot_scaled_residuals(results)

        assert fig is not None
        plt.close(fig)


class TestPlotResidualDistribution:
    """Tests for plot_residual_distribution function."""

    def test_returns_figure_and_axes(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test that function returns Figure and Axes."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, ax = plot_residual_distribution(results)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_custom_axes(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test plotting on provided axes."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        custom_fig, custom_ax = plt.subplots()
        fig, ax = plot_residual_distribution(results, ax=custom_ax)

        assert fig is custom_fig
        assert ax is custom_ax
        plt.close(fig)

    def test_no_normal_overlay(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test with normal overlay disabled."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, ax = plot_residual_distribution(results, show_normal=False)

        assert fig is not None
        plt.close(fig)

    def test_custom_bins(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test with custom number of bins."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, ax = plot_residual_distribution(results, bins=20)

        assert fig is not None
        plt.close(fig)


class TestPlotResidualAcf:
    """Tests for plot_residual_acf function."""

    def test_returns_figure_and_axes(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test that function returns Figure and array of Axes."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, axes = plot_residual_acf(results)

        assert fig is not None
        assert axes is not None
        assert len(axes) == 2  # ACF and PACF
        plt.close(fig)

    def test_custom_axes(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test plotting on provided axes."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        custom_fig, custom_axes = plt.subplots(1, 2)
        fig, axes = plot_residual_acf(results, ax=custom_axes)

        assert fig is custom_fig
        plt.close(fig)

    def test_custom_nlags(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test with custom number of lags."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, axes = plot_residual_acf(results, nlags=10)

        assert fig is not None
        plt.close(fig)

    def test_custom_alpha(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test with custom significance level."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, axes = plot_residual_acf(results, alpha=0.01)

        assert fig is not None
        plt.close(fig)


class TestPlotDiagnostics:
    """Tests for plot_diagnostics combined function."""

    def test_returns_figure_and_axes(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test that function returns Figure and array of Axes."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, axes = plot_diagnostics(results)

        assert fig is not None
        assert axes is not None
        assert axes.shape == (2, 2)
        plt.close(fig)

    def test_with_ols_results(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test with OLSResults."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, axes = plot_diagnostics(results)

        assert fig is not None
        plt.close(fig)

    def test_with_ar_results(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test with ARResults."""
        y, _ = ar1_data_with_break
        results = rg.AR(y, lags=1).fit()

        fig, axes = plot_diagnostics(results)

        assert fig is not None
        plt.close(fig)

    def test_with_adl_results(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test with ADLResults."""
        y, x = adl_data
        results = rg.ADL(y, x, lags=1, exog_lags=1).fit()

        fig, axes = plot_diagnostics(results)

        assert fig is not None
        plt.close(fig)

    def test_custom_nlags(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test with custom number of lags for ACF/PACF."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, axes = plot_diagnostics(results, nlags=10)

        assert fig is not None
        plt.close(fig)

    def test_custom_figsize(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test with custom figure size."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, axes = plot_diagnostics(results, figsize=(14, 12))

        assert fig is not None
        plt.close(fig)

    def test_with_breaks(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test with model that has structural breaks."""
        y, X, break_point = regression_data_with_break
        results = rg.OLS(y, X, breaks=[break_point], has_constant=False).fit()

        fig, axes = plot_diagnostics(results)

        assert fig is not None
        plt.close(fig)


class TestMethodAccess:
    """Tests for plot_diagnostics method on results classes."""

    def test_ols_results_method(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test that OLSResults has plot_diagnostics method."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, axes = results.plot_diagnostics()

        assert fig is not None
        plt.close(fig)

    def test_ar_results_method(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test that ARResults has plot_diagnostics method."""
        y, _ = ar1_data_with_break
        results = rg.AR(y, lags=1).fit()

        fig, axes = results.plot_diagnostics()

        assert fig is not None
        plt.close(fig)

    def test_adl_results_method(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test that ADLResults has plot_diagnostics method."""
        y, x = adl_data
        results = rg.ADL(y, x, lags=1, exog_lags=1).fit()

        fig, axes = results.plot_diagnostics()

        assert fig is not None
        plt.close(fig)

    def test_method_with_params(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test method accepts parameters."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, axes = results.plot_diagnostics(nlags=15, alpha=0.01, figsize=(10, 8))

        assert fig is not None
        plt.close(fig)


class TestAPIExports:
    """Tests for API-level exports."""

    def test_plot_diagnostics_in_api(self) -> None:
        """Test that plot_diagnostics is exported from regimes."""
        assert hasattr(rg, "plot_diagnostics")

    def test_plot_actual_fitted_in_api(self) -> None:
        """Test that plot_actual_fitted is exported from regimes."""
        assert hasattr(rg, "plot_actual_fitted")

    def test_plot_scaled_residuals_in_api(self) -> None:
        """Test that plot_scaled_residuals is exported from regimes."""
        assert hasattr(rg, "plot_scaled_residuals")

    def test_plot_residual_distribution_in_api(self) -> None:
        """Test that plot_residual_distribution is exported from regimes."""
        assert hasattr(rg, "plot_residual_distribution")

    def test_plot_residual_acf_in_api(self) -> None:
        """Test that plot_residual_acf is exported from regimes."""
        assert hasattr(rg, "plot_residual_acf")

    def test_functions_callable_from_api(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test that all functions are callable via API."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        # Test each function
        fig1, _ = rg.plot_diagnostics(results)
        plt.close(fig1)

        fig2, _ = rg.plot_actual_fitted(results)
        plt.close(fig2)

        fig3, _ = rg.plot_scaled_residuals(results)
        plt.close(fig3)

        fig4, _ = rg.plot_residual_distribution(results)
        plt.close(fig4)

        fig5, _ = rg.plot_residual_acf(results)
        plt.close(fig5)
