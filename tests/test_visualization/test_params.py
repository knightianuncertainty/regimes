"""Tests for plot_params_over_time visualization."""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

import regimes as rg
from regimes.visualization.params import (
    _extract_param_data,
    _get_base_param_names,
    _normalize_results_input,
    _parse_param_name,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


class TestParseParamName:
    """Tests for _parse_param_name helper."""

    def test_simple_name(self) -> None:
        """Test parsing a simple parameter name without regime."""
        base, regime = _parse_param_name("const")
        assert base == "const"
        assert regime is None

    def test_name_with_regime(self) -> None:
        """Test parsing a parameter name with regime suffix."""
        base, regime = _parse_param_name("const_regime2")
        assert base == "const"
        assert regime == 2

    def test_lag_name_without_regime(self) -> None:
        """Test parsing AR lag parameter name."""
        base, regime = _parse_param_name("y.L1")
        assert base == "y.L1"
        assert regime is None

    def test_lag_name_with_regime(self) -> None:
        """Test parsing AR lag parameter with regime suffix."""
        base, regime = _parse_param_name("y.L1_regime1")
        assert base == "y.L1"
        assert regime == 1

    def test_exog_name_with_regime(self) -> None:
        """Test parsing exogenous variable name with regime."""
        base, regime = _parse_param_name("x1_regime3")
        assert base == "x1"
        assert regime == 3


class TestNormalizeResultsInput:
    """Tests for _normalize_results_input helper."""

    def test_single_result(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test normalizing a single result object."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        normalized = _normalize_results_input(results)
        assert isinstance(normalized, dict)
        assert len(normalized) == 1
        assert "Model" in normalized

    def test_list_of_results(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test normalizing a list of results."""
        y, X = regression_data
        results1 = rg.OLS(y, X, has_constant=False).fit()
        results2 = rg.OLS(y, X, has_constant=False).fit(cov_type="HC0")

        normalized = _normalize_results_input([results1, results2])
        assert isinstance(normalized, dict)
        assert len(normalized) == 2
        assert "Model 1" in normalized
        assert "Model 2" in normalized

    def test_dict_of_results(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test normalizing a dict of results (passthrough)."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        input_dict = {"My Model": results}
        normalized = _normalize_results_input(input_dict)

        assert normalized is input_dict


class TestGetBaseParamNames:
    """Tests for _get_base_param_names helper."""

    def test_ols_no_breaks(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test extracting base names from OLS without breaks."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        base_names = _get_base_param_names(results)
        # X has columns named x0, x1 (auto-generated names)
        assert len(base_names) == 2
        assert "x0" in base_names
        assert "x1" in base_names

    def test_ols_with_breaks(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test extracting base names from OLS with breaks."""
        y, X, break_point = regression_data_with_break
        results = rg.OLS(y, X, breaks=[break_point], has_constant=False).fit()

        base_names = _get_base_param_names(results)
        # Should get unique base names, not regime-specific
        assert len(base_names) == 2
        assert "x0" in base_names
        assert "x1" in base_names


class TestExtractParamData:
    """Tests for _extract_param_data helper."""

    def test_ols_no_breaks(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test extracting param data from OLS without breaks."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        param_data = _extract_param_data(results)
        # X has columns named x0, x1 (auto-generated names)
        assert "x0" in param_data
        assert "x1" in param_data

        # Single segment for each parameter
        assert len(param_data["x0"]) == 1
        assert len(param_data["x1"]) == 1

        # Check segment structure
        seg = param_data["x0"][0]
        assert "start" in seg
        assert "end" in seg
        assert "value" in seg
        assert "ci_lower" in seg
        assert "ci_upper" in seg

    def test_ols_with_breaks(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test extracting param data from OLS with breaks."""
        y, X, break_point = regression_data_with_break
        results = rg.OLS(y, X, breaks=[break_point], has_constant=False).fit()

        param_data = _extract_param_data(results)
        # X has columns named x0, x1 (auto-generated names)
        assert "x0" in param_data
        assert "x1" in param_data

        # Two segments for each parameter (one per regime)
        assert len(param_data["x0"]) == 2
        assert len(param_data["x1"]) == 2

        # Check that segments cover the full sample
        x0_segs = param_data["x0"]
        assert x0_segs[0]["start"] == 0
        assert x0_segs[0]["end"] == break_point
        assert x0_segs[1]["start"] == break_point
        assert x0_segs[1]["end"] == len(y)


class TestPlotParamsOverTime:
    """Tests for plot_params_over_time function."""

    def test_single_ols_no_breaks(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test plotting OLS results without breaks."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, axes = rg.plot_params_over_time(results)

        assert fig is not None
        assert axes is not None
        plt.close(fig)

    def test_single_ols_with_breaks(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test plotting OLS results with breaks."""
        y, X, break_point = regression_data_with_break
        results = rg.OLS(y, X, breaks=[break_point], has_constant=False).fit()

        fig, axes = rg.plot_params_over_time(results)

        assert fig is not None
        # Should have subplots for each parameter
        plt.close(fig)

    def test_multiple_models_overlay(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test plotting multiple models for comparison."""
        y, X, break_point = regression_data_with_break

        # Model without breaks
        results_no_break = rg.OLS(y, X, has_constant=False).fit()
        # Model with breaks
        results_with_break = rg.OLS(
            y, X, breaks=[break_point], has_constant=False
        ).fit()

        fig, axes = rg.plot_params_over_time(
            {
                "No breaks": results_no_break,
                "With break": results_with_break,
            }
        )

        assert fig is not None
        plt.close(fig)

    def test_multiple_models_as_list(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test plotting multiple models passed as list."""
        y, X, break_point = regression_data_with_break

        results1 = rg.OLS(y, X, has_constant=False).fit()
        results2 = rg.OLS(y, X, breaks=[break_point], has_constant=False).fit()

        fig, axes = rg.plot_params_over_time([results1, results2])

        assert fig is not None
        plt.close(fig)

    def test_param_selection(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test plotting only selected parameters."""
        y, X, break_point = regression_data_with_break
        results = rg.OLS(y, X, breaks=[break_point], has_constant=False).fit()

        # Plot only the first parameter (x0)
        fig, axes = rg.plot_params_over_time(results, params=["x0"])

        assert fig is not None
        plt.close(fig)

    def test_custom_axes(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test plotting on provided axes."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        # Create custom figure and axes
        custom_fig, custom_axes = plt.subplots(1, 2, figsize=(12, 4))

        fig, axes = rg.plot_params_over_time(results, ax=custom_axes)

        # Should use the provided figure
        assert fig is custom_fig
        plt.close(fig)

    def test_ci_in_plot(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test that confidence intervals are plotted."""
        y, X, break_point = regression_data_with_break
        results = rg.OLS(y, X, breaks=[break_point], has_constant=False).fit()

        fig, axes = rg.plot_params_over_time(results, ci_alpha=0.3)

        # The plot should contain filled regions (PolyCollection) for CIs
        # This is a basic smoke test
        assert fig is not None
        plt.close(fig)

    def test_ar_results(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test plotting AR model results."""
        y, break_point = ar1_data_with_break
        results = rg.AR(y, lags=1, breaks=[break_point]).fit()

        fig, axes = rg.plot_params_over_time(results)

        assert fig is not None
        plt.close(fig)

    def test_show_breaks_option(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test show_breaks parameter."""
        y, X, break_point = regression_data_with_break
        results = rg.OLS(y, X, breaks=[break_point], has_constant=False).fit()

        # With breaks shown
        fig1, _ = rg.plot_params_over_time(results, show_breaks=True)
        plt.close(fig1)

        # Without breaks shown
        fig2, _ = rg.plot_params_over_time(results, show_breaks=False)
        plt.close(fig2)

    def test_custom_colors(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test custom colors parameter."""
        y, X = regression_data
        results1 = rg.OLS(y, X, has_constant=False).fit()
        results2 = rg.OLS(y, X, has_constant=False).fit(cov_type="HC0")

        fig, axes = rg.plot_params_over_time(
            [results1, results2],
            colors=["blue", "green"],
        )

        assert fig is not None
        plt.close(fig)

    def test_ncols_parameter(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test ncols parameter for subplot layout."""
        y, X, break_point = regression_data_with_break
        results = rg.OLS(y, X, breaks=[break_point], has_constant=False).fit()

        # Plot with 2 columns
        fig, axes = rg.plot_params_over_time(results, ncols=2)

        assert fig is not None
        plt.close(fig)

    def test_title_parameter(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test title parameter."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        fig, axes = rg.plot_params_over_time(results, title="My Custom Title")

        assert fig is not None
        # Check that the suptitle was set
        assert fig._suptitle is not None
        plt.close(fig)

    def test_empty_params_raises(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test that empty params list raises error."""
        y, X = regression_data
        results = rg.OLS(y, X, has_constant=False).fit()

        with pytest.raises(ValueError, match="No parameters to plot"):
            rg.plot_params_over_time(results, params=[])


class TestPlotParamsOverTimeBaiPerron:
    """Tests for plot_params_over_time with BaiPerronTest."""

    def test_baiperron_test(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test plotting BaiPerronTest results."""
        y, _ = data_with_mean_shift

        test = rg.BaiPerronTest(y)
        # We need to manually set breaks for the test to extract data
        test._selected_breaks = [100]  # type: ignore[attr-defined]

        fig, axes = rg.plot_params_over_time(test)

        assert fig is not None
        plt.close(fig)

    def test_baiperron_results(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test plotting BaiPerronResults."""
        y, _ = data_with_mean_shift

        test = rg.BaiPerronTest(y)
        results = test.fit(max_breaks=2)

        # BaiPerronResults doesn't store parameter estimates directly
        # so we expect NaN values but the plot should still work
        fig, axes = rg.plot_params_over_time(results)

        assert fig is not None
        plt.close(fig)
