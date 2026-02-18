"""Tests for CUSUM and CUSUM-SQ visualization functions."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from regimes import CUSUMSQTest, CUSUMTest, plot_cusum, plot_cusum_sq


@pytest.fixture
def cusum_results(rng: np.random.Generator):
    """CUSUM results for stable data."""
    y = rng.standard_normal(200)
    test = CUSUMTest(y)
    return test.fit(significance=0.05)


@pytest.fixture
def cusum_results_rejected(rng: np.random.Generator):
    """CUSUM results with detected instability."""
    y = np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 100)])
    test = CUSUMTest(y)
    return test.fit(significance=0.05)


@pytest.fixture
def cusumsq_results(rng: np.random.Generator):
    """CUSUM-SQ results for stable data."""
    y = rng.standard_normal(200)
    test = CUSUMSQTest(y)
    return test.fit(significance=0.05)


@pytest.fixture
def cusumsq_results_rejected(rng: np.random.Generator):
    """CUSUM-SQ results with detected variance instability."""
    y = np.concatenate([rng.normal(0, 0.5, 150), rng.normal(0, 5.0, 150)])
    test = CUSUMSQTest(y)
    return test.fit(significance=0.05)


class TestPlotCUSUM:
    """Tests for plot_cusum function."""

    def test_returns_fig_and_ax(self, cusum_results) -> None:
        """Should return a figure and axes."""
        fig, ax = plot_cusum(cusum_results)
        assert fig is not None
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_custom_axes(self, cusum_results) -> None:
        """Should work with a provided axes object."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot_cusum(cusum_results, ax=ax)
        assert returned_ax is ax
        plt.close(fig)

    def test_custom_title(self, cusum_results) -> None:
        """Should use custom title."""
        fig, ax = plot_cusum(cusum_results, title="Custom Title")
        assert ax.get_title() == "Custom Title"
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_default_title(self, cusum_results) -> None:
        """Should use default title."""
        fig, ax = plot_cusum(cusum_results)
        assert ax.get_title() == "CUSUM test statistic"
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_custom_colors(self, cusum_results) -> None:
        """Should accept custom colors."""
        fig, ax = plot_cusum(
            cusum_results,
            statistic_color="#FF0000",
            bound_color="#00FF00",
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_hide_crossings(self, cusum_results_rejected) -> None:
        """Should work with crossings hidden."""
        fig, ax = plot_cusum(cusum_results_rejected, show_crossings=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_no_rejection_shading(self, cusum_results) -> None:
        """Should work without rejection region shading."""
        fig, ax = plot_cusum(cusum_results, shade_rejection=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_with_crossings(self, cusum_results_rejected) -> None:
        """Should plot crossings when present."""
        fig, ax = plot_cusum(cusum_results_rejected, show_crossings=True)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_results_plot_method(self, cusum_results) -> None:
        """Results .plot() convenience method should work."""
        fig, ax = cusum_results.plot()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotCUSUMSQ:
    """Tests for plot_cusum_sq function."""

    def test_returns_fig_and_ax(self, cusumsq_results) -> None:
        """Should return a figure and axes."""
        fig, ax = plot_cusum_sq(cusumsq_results)
        assert fig is not None
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_custom_axes(self, cusumsq_results) -> None:
        """Should work with a provided axes object."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot_cusum_sq(cusumsq_results, ax=ax)
        assert returned_ax is ax
        plt.close(fig)

    def test_default_title(self, cusumsq_results) -> None:
        """Should use default title."""
        fig, ax = plot_cusum_sq(cusumsq_results)
        assert ax.get_title() == "CUSUM-SQ test statistic"
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_show_expected_path(self, cusumsq_results) -> None:
        """Should show expected diagonal by default."""
        fig, ax = plot_cusum_sq(cusumsq_results, show_expected=True)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_hide_expected_path(self, cusumsq_results) -> None:
        """Should allow hiding expected path."""
        fig, ax = plot_cusum_sq(cusumsq_results, show_expected=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_custom_colors(self, cusumsq_results) -> None:
        """Should accept custom colors."""
        fig, ax = plot_cusum_sq(
            cusumsq_results,
            statistic_color="#FF0000",
            bound_color="#00FF00",
            expected_color="#0000FF",
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_with_rejection(self, cusumsq_results_rejected) -> None:
        """Should plot with rejection region."""
        fig, ax = plot_cusum_sq(cusumsq_results_rejected)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_results_plot_method(self, cusumsq_results) -> None:
        """Results .plot() convenience method should work."""
        fig, ax = cusumsq_results.plot()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)
