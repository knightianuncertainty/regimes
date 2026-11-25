"""Tests for GETS indicator saturation visualization."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from regimes.gets.saturation import isat
from regimes.visualization.gets import (
    plot_mis_coefficients,
    plot_regime_levels,
    plot_sis_coefficients,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def sis_results(rng):
    """SIS results with a known mean shift."""
    y = np.concatenate(
        [
            rng.normal(0, 0.5, 100),
            rng.normal(3, 0.5, 100),
        ]
    )
    return isat(y, sis=True, diagnostics=False)


@pytest.fixture
def mis_results(rng):
    """MIS results with AR coefficient change."""
    n = 200
    y = np.zeros(n)
    for t in range(1, 100):
        y[t] = 0.3 * y[t - 1] + rng.standard_normal()
    for t in range(100, n):
        y[t] = 0.8 * y[t - 1] + rng.standard_normal()
    return isat(y, ar_lags=1, sis=True, mis=True, diagnostics=False)


@pytest.fixture
def null_results(rng):
    """Results under null DGP (no breaks)."""
    y = rng.standard_normal(200)
    return isat(y, sis=True, diagnostics=False)


# ---------------------------------------------------------------------------
# plot_sis_coefficients
# ---------------------------------------------------------------------------


class TestPlotSISCoefficients:
    def test_returns_figure_and_axes(self, sis_results):
        fig, ax = plot_sis_coefficients(sis_results)
        assert fig is not None
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_custom_title(self, sis_results):
        fig, ax = plot_sis_coefficients(sis_results, title="Custom Title")
        assert ax.get_title() == "Custom Title"
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_without_ci(self, sis_results):
        fig, ax = plot_sis_coefficients(sis_results, show_ci=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_on_existing_axes(self, sis_results):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        fig2, ax2 = plot_sis_coefficients(sis_results, ax=ax)
        assert ax2 is ax
        plt.close(fig)

    def test_null_dgp(self, null_results):
        fig, ax = plot_sis_coefficients(null_results)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_convenience_method(self, sis_results):
        fig, ax = sis_results.plot_sis()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_mis_coefficients
# ---------------------------------------------------------------------------


class TestPlotMISCoefficients:
    def test_returns_figure_and_axes(self, mis_results):
        fig, axes = plot_mis_coefficients(mis_results)
        assert fig is not None
        assert axes is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_specific_params(self, mis_results):
        # Plot only y.L1 if present
        regimes_dict = mis_results.regime_levels.param_regimes
        params = [p for p in regimes_dict if p != "const"]
        if params:
            fig, axes = plot_mis_coefficients(mis_results, params=params[:1])
            assert fig is not None
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_with_title(self, mis_results):
        fig, axes = plot_mis_coefficients(mis_results, title="MIS Test")
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_without_ci(self, mis_results):
        fig, axes = plot_mis_coefficients(mis_results, show_ci=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_convenience_method(self, mis_results):
        fig, axes = mis_results.plot_mis()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_regime_levels
# ---------------------------------------------------------------------------


class TestPlotRegimeLevels:
    def test_returns_figure_and_axes(self, sis_results):
        fig, axes = plot_regime_levels(sis_results)
        assert fig is not None
        assert axes is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_mis_results(self, mis_results):
        fig, axes = plot_regime_levels(mis_results)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_null_dgp(self, null_results):
        fig, axes = plot_regime_levels(null_results)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_title(self, sis_results):
        fig, axes = plot_regime_levels(sis_results, title="All Regimes")
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_specific_params(self, sis_results):
        fig, axes = plot_regime_levels(sis_results, params=["const"])
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_without_ci(self, sis_results):
        fig, axes = plot_regime_levels(sis_results, show_ci=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_convenience_method(self, sis_results):
        fig, axes = sis_results.plot_regime_levels()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)
