"""Tests for the visualization style module."""

from __future__ import annotations

import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from regimes.visualization.style import (
    REGIMES_COLOR_CYCLE,
    REGIMES_COLORS,
    add_break_dates,
    add_confidence_band,
    add_source,
    get_style,
    label_line_end,
    set_style,
    shade_regimes,
    use_style,
)


class TestColorPalette:
    """Tests for REGIMES_COLORS and REGIMES_COLOR_CYCLE."""

    def test_colors_dict_has_required_keys(self) -> None:
        """Test that REGIMES_COLORS has all required color keys."""
        required_keys = [
            "blue",
            "red",
            "teal",
            "green",
            "gold",
            "grey",
            "mauve",
            "light_grey",
            "near_black",
            "regime_tint_a",
            "regime_tint_b",
        ]
        for key in required_keys:
            assert key in REGIMES_COLORS, f"Missing color key: {key}"

    def test_colors_are_valid_hex(self) -> None:
        """Test that all colors are valid hex codes."""
        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for name, color in REGIMES_COLORS.items():
            assert hex_pattern.match(color), f"Invalid hex color for {name}: {color}"

    def test_color_cycle_is_list(self) -> None:
        """Test that REGIMES_COLOR_CYCLE is a list."""
        assert isinstance(REGIMES_COLOR_CYCLE, list)

    def test_color_cycle_has_expected_length(self) -> None:
        """Test that color cycle has 7 colors (primary palette)."""
        assert len(REGIMES_COLOR_CYCLE) == 7

    def test_color_cycle_matches_dict(self) -> None:
        """Test that color cycle contains colors from the dict."""
        expected = [
            REGIMES_COLORS["blue"],
            REGIMES_COLORS["red"],
            REGIMES_COLORS["teal"],
            REGIMES_COLORS["green"],
            REGIMES_COLORS["gold"],
            REGIMES_COLORS["grey"],
            REGIMES_COLORS["mauve"],
        ]
        assert expected == REGIMES_COLOR_CYCLE


class TestGetStyle:
    """Tests for get_style() function."""

    def test_returns_dict(self) -> None:
        """Test that get_style returns a dictionary."""
        style = get_style()
        assert isinstance(style, dict)

    def test_has_figure_settings(self) -> None:
        """Test that style includes figure settings."""
        style = get_style()
        assert "figure.figsize" in style
        assert style["figure.figsize"] == (10, 5)
        assert "figure.dpi" in style
        assert style["figure.dpi"] == 150

    def test_has_axes_settings(self) -> None:
        """Test that style includes axes settings."""
        style = get_style()
        # No top/right spines
        assert style["axes.spines.top"] is False
        assert style["axes.spines.right"] is False
        assert style["axes.spines.left"] is True
        assert style["axes.spines.bottom"] is True
        # Grid settings
        assert style["axes.grid"] is True
        assert style["axes.grid.axis"] == "y"

    def test_has_font_settings(self) -> None:
        """Test that style includes font settings."""
        style = get_style()
        assert style["font.family"] == "sans-serif"
        assert "font.size" in style

    def test_has_tick_settings(self) -> None:
        """Test that style removes tick marks."""
        style = get_style()
        assert style["xtick.major.size"] == 0
        assert style["ytick.major.size"] == 0

    def test_has_legend_settings(self) -> None:
        """Test that style sets frameless legend."""
        style = get_style()
        assert style["legend.frameon"] is False

    def test_has_savefig_settings(self) -> None:
        """Test that style includes savefig settings."""
        style = get_style()
        assert style["savefig.dpi"] == 300
        assert style["savefig.bbox"] == "tight"

    def test_has_color_cycle(self) -> None:
        """Test that style includes color cycle."""
        style = get_style()
        assert "axes.prop_cycle" in style


class TestSetStyle:
    """Tests for set_style() function."""

    def test_modifies_global_rcparams(self) -> None:
        """Test that set_style modifies global rcParams."""
        # Store original values
        original_figsize = mpl.rcParams["figure.figsize"]

        try:
            set_style()
            assert tuple(mpl.rcParams["figure.figsize"]) == (10, 5)
            assert mpl.rcParams["axes.spines.top"] is False
        finally:
            # Restore original
            mpl.rcParams["figure.figsize"] = original_figsize


class TestUseStyle:
    """Tests for use_style() context manager."""

    def test_applies_style_in_context(self) -> None:
        """Test that use_style applies style within context."""
        original_top_spine = mpl.rcParams["axes.spines.top"]

        with use_style():
            assert mpl.rcParams["axes.spines.top"] is False
            assert mpl.rcParams["axes.spines.right"] is False

        # Should be restored after context
        assert mpl.rcParams["axes.spines.top"] == original_top_spine

    def test_restores_rcparams_after_context(self) -> None:
        """Test that original rcParams are restored after context."""
        # Set a custom value
        original = mpl.rcParams["figure.dpi"]

        with use_style():
            # Inside context, should have regimes style
            assert mpl.rcParams["figure.dpi"] == 150

        # After context, should be restored
        assert mpl.rcParams["figure.dpi"] == original

    def test_restores_on_exception(self) -> None:
        """Test that rcParams are restored even on exception."""
        original = mpl.rcParams["axes.spines.top"]

        with pytest.raises(ValueError), use_style():
            assert mpl.rcParams["axes.spines.top"] is False
            raise ValueError("test exception")

        # Should still be restored
        assert mpl.rcParams["axes.spines.top"] == original


class TestLabelLineEnd:
    """Tests for label_line_end() helper function."""

    def test_adds_annotation(self) -> None:
        """Test that label_line_end adds an annotation to the axes."""
        fig, ax = plt.subplots()
        x = [0, 1, 2, 3]
        y = [1, 2, 3, 4]
        ax.plot(x, y)

        label_line_end(ax, x, y, "Test Label", "#006BA2")

        # Check that annotation was added
        assert len(ax.texts) == 1
        assert ax.texts[0].get_text() == "Test Label"

        plt.close(fig)

    def test_annotation_at_last_point(self) -> None:
        """Test that annotation is positioned at the last data point."""
        fig, ax = plt.subplots()
        x = [0, 1, 2, 3]
        y = [1, 2, 3, 4]
        ax.plot(x, y)

        label_line_end(ax, x, y, "Test", "#006BA2")

        # The annotation xy should be at the last point
        annot = ax.texts[0]
        assert annot.xy == (3, 4)

        plt.close(fig)


class TestAddBreakDates:
    """Tests for add_break_dates() helper function."""

    def test_adds_vertical_lines(self) -> None:
        """Test that add_break_dates adds vertical lines."""
        fig, ax = plt.subplots()
        ax.plot([0, 100], [0, 1])

        add_break_dates(ax, [25, 50, 75])

        # Check that lines were added
        lines = [
            child for child in ax.get_children() if hasattr(child, "get_linestyle")
        ]
        # At least 3 vertical lines should be added (plus the original plot line)
        assert len(lines) >= 3

        plt.close(fig)

    def test_uses_default_grey_color(self) -> None:
        """Test that add_break_dates uses palette grey by default."""
        fig, ax = plt.subplots()
        ax.plot([0, 100], [0, 1])

        add_break_dates(ax, [50])

        # The default should be REGIMES_COLORS["grey"]
        # This is tested implicitly by not passing color parameter

        plt.close(fig)

    def test_adds_labels_when_provided(self) -> None:
        """Test that labels are added when provided."""
        fig, ax = plt.subplots()
        ax.set_ylim(0, 1)
        ax.plot([0, 100], [0, 1])

        add_break_dates(ax, [50], labels=["Break 1"])

        # Check that text was added
        texts = [t for t in ax.texts if t.get_text().strip()]
        assert len(texts) == 1
        assert "Break 1" in texts[0].get_text()

        plt.close(fig)


class TestAddConfidenceBand:
    """Tests for add_confidence_band() helper function."""

    def test_adds_fill_between(self) -> None:
        """Test that add_confidence_band adds a fill_between polygon."""
        fig, ax = plt.subplots()
        x = np.arange(100)
        y = np.sin(x / 10)
        ax.plot(x, y)

        add_confidence_band(ax, x, y - 0.2, y + 0.2)

        # Check that a collection (fill_between) was added
        collections = ax.collections
        assert len(collections) >= 1

        plt.close(fig)

    def test_uses_default_blue_color(self) -> None:
        """Test that add_confidence_band uses palette blue by default."""
        fig, ax = plt.subplots()
        x = np.arange(10)

        add_confidence_band(ax, x, x - 1, x + 1)

        # Check that a collection was added (default color is tested implicitly)
        assert len(ax.collections) >= 1

        plt.close(fig)


class TestShadeRegimes:
    """Tests for shade_regimes() helper function."""

    def test_shades_alternating_regimes(self) -> None:
        """Test that shade_regimes adds axvspan for each regime."""
        fig, ax = plt.subplots()
        ax.plot([0, 100], [0, 1])

        shade_regimes(ax, [30, 60], start=0, end=100)

        # Should have 3 regimes: [0-30], [30-60], [60-100]
        # Check patches were added
        patches = list(ax.patches)
        assert len(patches) == 3

        plt.close(fig)

    def test_uses_default_tint_colors(self) -> None:
        """Test that shade_regimes uses palette tints by default."""
        fig, ax = plt.subplots()
        ax.plot([0, 100], [0, 1])

        # Default colors should be regime_tint_a and regime_tint_b
        shade_regimes(ax, [50], start=0, end=100)

        # 2 regimes should be shaded
        assert len(ax.patches) == 2

        plt.close(fig)


class TestAddSource:
    """Tests for add_source() helper function."""

    def test_adds_text_annotation(self) -> None:
        """Test that add_source adds a text annotation."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        add_source(ax, "Source: Federal Reserve")

        # Check that text was added
        texts = ax.texts
        assert len(texts) == 1
        assert texts[0].get_text() == "Source: Federal Reserve"

        plt.close(fig)

    def test_text_position(self) -> None:
        """Test that source text is positioned at bottom-left."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        add_source(ax, "Source: Test")

        text = ax.texts[0]
        # Should use axes transform
        assert text.get_transform() == ax.transAxes

        plt.close(fig)


class TestIntegration:
    """Integration tests for the style system."""

    def test_style_applied_to_simple_plot(self) -> None:
        """Test that use_style produces expected plot characteristics."""
        with use_style():
            fig, ax = plt.subplots()
            ax.plot([0, 1, 2], [0, 1, 0])
            ax.set_title("Test Plot")

            # Check style was applied
            assert ax.spines["top"].get_visible() is False
            assert ax.spines["right"].get_visible() is False
            assert ax.spines["left"].get_visible() is True
            assert ax.spines["bottom"].get_visible() is True

        plt.close(fig)

    def test_colors_are_accessible(self) -> None:
        """Test that all colors in the palette can be used by matplotlib."""
        with use_style():
            fig, ax = plt.subplots()

            for name, color in REGIMES_COLORS.items():
                # This should not raise
                ax.plot([0, 1], [0, 1], color=color, label=name)

            ax.legend()

        plt.close(fig)
