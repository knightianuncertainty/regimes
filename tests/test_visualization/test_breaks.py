"""Tests for break visualization functions."""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import regimes as rg
from regimes.visualization.breaks import (
    plot_break_confidence,
    plot_breaks,
    plot_regime_means,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


# =============================================================================
# plot_breaks Tests
# =============================================================================


class TestPlotBreaks:
    """Tests for plot_breaks function."""

    def test_returns_figure_and_axes(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test that function returns Figure and Axes."""
        y, break_point = data_with_mean_shift
        fig, ax = plot_breaks(y, breaks=[break_point])

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_no_breaks(self, simple_data: NDArray[np.floating[Any]]) -> None:
        """Test plotting with no breaks."""
        fig, ax = plot_breaks(simple_data, breaks=[])

        assert fig is not None
        plt.close(fig)

    def test_single_break(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test plotting with single break."""
        y, break_point = data_with_mean_shift
        fig, ax = plot_breaks(y, breaks=[break_point])

        assert fig is not None
        plt.close(fig)

    def test_multiple_breaks(
        self, data_with_two_breaks: tuple[NDArray[np.floating[Any]], list[int]]
    ) -> None:
        """Test plotting with multiple breaks."""
        y, breaks = data_with_two_breaks
        fig, ax = plot_breaks(y, breaks=breaks)

        assert fig is not None
        plt.close(fig)

    def test_with_results_object(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test using breaks from results object."""
        y, _ = data_with_mean_shift
        test = rg.BaiPerronTest(y)
        results = test.fit(max_breaks=2)

        fig, ax = plot_breaks(y, results=results)

        assert fig is not None
        plt.close(fig)

    def test_custom_axes(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test plotting on provided axes."""
        y, break_point = data_with_mean_shift
        custom_fig, custom_ax = plt.subplots()

        fig, ax = plot_breaks(y, breaks=[break_point], ax=custom_ax)

        assert fig is custom_fig
        assert ax is custom_ax
        plt.close(fig)

    def test_custom_title(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test custom title."""
        y, break_point = data_with_mean_shift
        fig, ax = plot_breaks(y, breaks=[break_point], title="My Custom Title")

        assert ax.get_title() == "My Custom Title"
        plt.close(fig)

    def test_default_title(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test default title generation."""
        y, break_point = data_with_mean_shift
        fig, ax = plot_breaks(y, breaks=[break_point])

        assert "1 Structural Break" in ax.get_title()
        plt.close(fig)

    def test_custom_colors(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test custom colors."""
        y, break_point = data_with_mean_shift
        fig, ax = plot_breaks(
            y, breaks=[break_point], break_color="red", series_color="green"
        )

        assert fig is not None
        plt.close(fig)

    def test_shade_regimes(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test regime shading option."""
        y, break_point = data_with_mean_shift
        fig, ax = plot_breaks(y, breaks=[break_point], shade_regimes=True)

        assert fig is not None
        plt.close(fig)

    def test_custom_regime_colors(
        self, data_with_two_breaks: tuple[NDArray[np.floating[Any]], list[int]]
    ) -> None:
        """Test custom regime shading colors."""
        y, breaks = data_with_two_breaks
        fig, ax = plot_breaks(
            y, breaks=breaks, shade_regimes=True, regime_colors=["blue", "green", "red"]
        )

        assert fig is not None
        plt.close(fig)

    def test_hide_legend(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test hiding legend."""
        y, break_point = data_with_mean_shift
        fig, ax = plot_breaks(y, breaks=[break_point], show_legend=False)

        assert fig is not None
        plt.close(fig)

    def test_custom_figsize(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test custom figure size."""
        y, break_point = data_with_mean_shift
        fig, ax = plot_breaks(y, breaks=[break_point], figsize=(12, 6))

        assert fig is not None
        # Check approximate figure size (may not be exact due to margins)
        plt.close(fig)

    def test_custom_line_properties(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test custom line properties."""
        y, break_point = data_with_mean_shift
        fig, ax = plot_breaks(
            y,
            breaks=[break_point],
            break_linestyle="-",
            break_linewidth=2.0,
            break_alpha=0.5,
            series_linewidth=1.0,
        )

        assert fig is not None
        plt.close(fig)


# =============================================================================
# plot_regime_means Tests
# =============================================================================


class TestPlotRegimeMeans:
    """Tests for plot_regime_means function."""

    def test_returns_figure_and_axes(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test that function returns Figure and Axes."""
        y, break_point = data_with_mean_shift
        fig, ax = plot_regime_means(y, breaks=[break_point])

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_single_break(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test with single break."""
        y, break_point = data_with_mean_shift
        fig, ax = plot_regime_means(y, breaks=[break_point])

        assert fig is not None
        plt.close(fig)

    def test_multiple_breaks(
        self, data_with_two_breaks: tuple[NDArray[np.floating[Any]], list[int]]
    ) -> None:
        """Test with multiple breaks."""
        y, breaks = data_with_two_breaks
        fig, ax = plot_regime_means(y, breaks=breaks)

        assert fig is not None
        plt.close(fig)

    def test_custom_axes(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test plotting on provided axes."""
        y, break_point = data_with_mean_shift
        custom_fig, custom_ax = plt.subplots()

        fig, ax = plot_regime_means(y, breaks=[break_point], ax=custom_ax)

        assert fig is custom_fig
        assert ax is custom_ax
        plt.close(fig)

    def test_custom_title(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test custom title."""
        y, break_point = data_with_mean_shift
        fig, ax = plot_regime_means(y, breaks=[break_point], title="My Custom Title")

        assert ax.get_title() == "My Custom Title"
        plt.close(fig)

    def test_custom_colors(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test custom colors."""
        y, break_point = data_with_mean_shift
        fig, ax = plot_regime_means(
            y,
            breaks=[break_point],
            series_color="blue",
            mean_color="green",
            break_color="red",
        )

        assert fig is not None
        plt.close(fig)

    def test_hide_legend(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test hiding legend."""
        y, break_point = data_with_mean_shift
        fig, ax = plot_regime_means(y, breaks=[break_point], show_legend=False)

        assert fig is not None
        plt.close(fig)

    def test_custom_line_properties(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test custom line properties."""
        y, break_point = data_with_mean_shift
        fig, ax = plot_regime_means(
            y, breaks=[break_point], series_alpha=0.3, mean_linewidth=3.0
        )

        assert fig is not None
        plt.close(fig)


# =============================================================================
# plot_break_confidence Tests
# =============================================================================


class TestPlotBreakConfidence:
    """Tests for plot_break_confidence function."""

    def test_returns_figure_and_axes(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test that function returns Figure and Axes."""
        y, break_point = data_with_mean_shift
        ci = [(break_point - 10, break_point + 10)]
        fig, ax = plot_break_confidence(
            y, breaks=[break_point], confidence_intervals=ci
        )

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_without_confidence_intervals(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test without confidence intervals."""
        y, break_point = data_with_mean_shift
        fig, ax = plot_break_confidence(y, breaks=[break_point])

        assert fig is not None
        plt.close(fig)

    def test_multiple_confidence_intervals(
        self, data_with_two_breaks: tuple[NDArray[np.floating[Any]], list[int]]
    ) -> None:
        """Test with multiple confidence intervals."""
        y, breaks = data_with_two_breaks
        ci = [(b - 5, b + 5) for b in breaks]
        fig, ax = plot_break_confidence(y, breaks=breaks, confidence_intervals=ci)

        assert fig is not None
        plt.close(fig)

    def test_custom_ci_appearance(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test custom CI appearance."""
        y, break_point = data_with_mean_shift
        ci = [(break_point - 10, break_point + 10)]
        fig, ax = plot_break_confidence(
            y,
            breaks=[break_point],
            confidence_intervals=ci,
            ci_alpha=0.3,
            ci_color="green",
        )

        assert fig is not None
        plt.close(fig)

    def test_custom_title(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test custom title."""
        y, break_point = data_with_mean_shift
        ci = [(break_point - 10, break_point + 10)]
        fig, ax = plot_break_confidence(
            y, breaks=[break_point], confidence_intervals=ci, title="Custom Title"
        )

        assert ax.get_title() == "Custom Title"
        plt.close(fig)

    def test_inherits_plot_breaks_kwargs(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test that kwargs pass through to plot_breaks."""
        y, break_point = data_with_mean_shift
        ci = [(break_point - 10, break_point + 10)]
        fig, ax = plot_break_confidence(
            y,
            breaks=[break_point],
            confidence_intervals=ci,
            shade_regimes=True,
            series_color="green",
        )

        assert fig is not None
        plt.close(fig)


# =============================================================================
# API Export Tests
# =============================================================================


class TestAPIExports:
    """Tests for API-level exports."""

    def test_plot_breaks_in_api(self) -> None:
        """Test that plot_breaks is exported from regimes."""
        assert hasattr(rg, "plot_breaks")

    def test_plot_regime_means_in_api(self) -> None:
        """Test that plot_regime_means is exported from regimes."""
        assert hasattr(rg, "plot_regime_means")

    def test_plot_break_confidence_in_api(self) -> None:
        """Test that plot_break_confidence is exported from regimes."""
        assert hasattr(rg, "plot_break_confidence")

    def test_functions_callable_from_api(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test that all functions are callable via API."""
        y, break_point = data_with_mean_shift

        fig1, _ = rg.plot_breaks(y, breaks=[break_point])
        plt.close(fig1)

        fig2, _ = rg.plot_regime_means(y, breaks=[break_point])
        plt.close(fig2)

        fig3, _ = rg.plot_break_confidence(y, breaks=[break_point])
        plt.close(fig3)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases for break visualization."""

    def test_short_series(self, rng: np.random.Generator) -> None:
        """Test with short time series."""
        y = rng.standard_normal(20)
        fig, ax = plot_breaks(y, breaks=[10])

        assert fig is not None
        plt.close(fig)

    def test_long_series(self, rng: np.random.Generator) -> None:
        """Test with long time series."""
        y = rng.standard_normal(1000)
        fig, ax = plot_breaks(y, breaks=[300, 600])

        assert fig is not None
        plt.close(fig)

    def test_break_at_boundary(self, rng: np.random.Generator) -> None:
        """Test with break near boundary."""
        y = rng.standard_normal(100)
        # Break near start
        fig1, _ = plot_breaks(y, breaks=[5])
        plt.close(fig1)

        # Break near end
        fig2, _ = plot_breaks(y, breaks=[95])
        plt.close(fig2)

    def test_many_breaks(self, rng: np.random.Generator) -> None:
        """Test with many breaks."""
        y = rng.standard_normal(200)
        breaks = [30, 60, 90, 120, 150, 180]
        fig, ax = plot_breaks(y, breaks=breaks)

        assert fig is not None
        plt.close(fig)
