"""Integration tests for GETS indicator saturation.

Tests end-to-end workflows: known DGP detection, null retention,
dual representation round-trips, and convenience methods.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimes.gets.representation import levels_to_shifts
from regimes.gets.saturation import SaturationResults, isat


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Known DGP detection
# ---------------------------------------------------------------------------


class TestKnownDGP:
    def test_sis_detects_known_mean_shift(self, rng):
        """SIS detects a single large mean shift near the true break."""
        y = np.concatenate(
            [
                rng.normal(0, 0.5, 100),
                rng.normal(3, 0.5, 100),
            ]
        )
        result = isat(y, sis=True, alpha=0.05, diagnostics=False)

        assert result.n_indicators_retained >= 1
        assert len(result.break_dates) >= 1
        # The detected break should be near t=100
        assert any(90 <= bd <= 110 for bd in result.break_dates), (
            f"Expected break near 100, got {result.break_dates}"
        )

        # Regime levels should approximate the true means
        regimes = result.regime_levels.param_regimes.get("const", [])
        if len(regimes) >= 2:
            # First regime level near 0
            assert abs(regimes[0].level) < 1.0
            # Second regime level near 3
            assert abs(regimes[-1].level - 3.0) < 1.0

    def test_mis_detects_ar_coefficient_change(self, rng):
        """MIS detects a change in AR coefficient."""
        n = 200
        y = np.zeros(n)
        for t in range(1, 100):
            y[t] = 0.2 * y[t - 1] + rng.standard_normal()
        for t in range(100, n):
            y[t] = 0.8 * y[t - 1] + rng.standard_normal()

        result = isat(
            y,
            ar_lags=1,
            mis=True,
            sis=True,
            alpha=0.05,
            diagnostics=False,
        )
        assert result.n_indicators_retained >= 1

    def test_combined_sis_mis_two_breaks(self, rng):
        """Combined SIS+MIS: level shift + AR break at different dates."""
        n = 200
        y = np.zeros(n)
        for t in range(1, 80):
            y[t] = 0 + 0.3 * y[t - 1] + rng.standard_normal()
        for t in range(80, 150):
            y[t] = 2 + 0.3 * y[t - 1] + rng.standard_normal()
        for t in range(150, n):
            y[t] = 2 + 0.7 * y[t - 1] + rng.standard_normal()

        result = isat(
            y,
            ar_lags=1,
            sis=True,
            mis=True,
            alpha=0.05,
            diagnostics=False,
        )
        assert result.n_indicators_retained >= 1
        assert isinstance(result, SaturationResults)


# ---------------------------------------------------------------------------
# Null DGP retention rate
# ---------------------------------------------------------------------------


class TestNullRetention:
    def test_sis_null_retention_rate(self, rng):
        """Under null DGP, SIS false detection rate should be controlled."""
        n = 200
        y = rng.standard_normal(n)
        result = isat(y, sis=True, alpha=0.01, diagnostics=False)

        retention_rate = result.n_indicators_retained / max(
            result.n_indicators_initial, 1
        )
        assert retention_rate < 0.15, (
            f"Null retention rate {retention_rate:.2%} exceeds 15% threshold"
        )

    def test_iis_null_retention_rate(self, rng):
        """IIS null retention also controlled."""
        n = 200
        y = rng.standard_normal(n)
        result = isat(y, iis=True, alpha=0.01, diagnostics=False)

        retention_rate = result.n_indicators_retained / max(
            result.n_indicators_initial, 1
        )
        assert retention_rate < 0.15


# ---------------------------------------------------------------------------
# Dual representation round-trip
# ---------------------------------------------------------------------------


class TestDualRepresentation:
    def test_shifts_to_levels_round_trip(self, rng):
        """shifts -> levels -> shifts preserves break dates and magnitudes."""
        y = np.concatenate(
            [
                rng.normal(0, 0.5, 100),
                rng.normal(3, 0.5, 100),
            ]
        )
        result = isat(y, sis=True, diagnostics=False)

        # Round-trip
        recovered_shifts = levels_to_shifts(result.regime_levels)

        # Break dates should be the same
        assert recovered_shifts.break_dates == result.shifts.break_dates

        # Initial levels should be close
        for param in result.shifts.initial_levels:
            assert (
                abs(
                    recovered_shifts.initial_levels[param]
                    - result.shifts.initial_levels[param]
                )
                < 1e-10
            )

    def test_levels_have_correct_count(self, rng):
        """Number of regimes = number of breaks + 1 for each parameter."""
        y = np.concatenate(
            [
                rng.normal(0, 0.5, 100),
                rng.normal(3, 0.5, 100),
            ]
        )
        result = isat(y, sis=True, diagnostics=False)

        for param, regimes in result.regime_levels.param_regimes.items():
            n_breaks_for_param = len(result.shifts.shifts.get(param, {}))
            assert len(regimes) == n_breaks_for_param + 1


# ---------------------------------------------------------------------------
# Targeted MIS
# ---------------------------------------------------------------------------


class TestTargetedMIS:
    def test_mis_specific_variable(self, rng):
        """MIS with specific variable selection."""
        n = 200
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + rng.standard_normal()

        result = isat(
            y,
            ar_lags=1,
            mis=["y.L1"],
            diagnostics=False,
        )
        assert "MIS" in result.saturation_type

    def test_mis_true_interacts_all(self, rng):
        """MIS=True interacts with all non-constant core regressors."""
        n = 200
        x = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + 0.3 * x[t] + rng.standard_normal()

        result = isat(
            y,
            exog=x,
            ar_lags=1,
            mis=True,
            diagnostics=False,
        )
        assert "MIS" in result.saturation_type


# ---------------------------------------------------------------------------
# Custom user indicators
# ---------------------------------------------------------------------------


class TestCustomIndicators:
    def test_user_indicator_single(self, rng):
        """A single custom indicator at the true break point."""
        n = 200
        y = np.concatenate(
            [
                rng.normal(0, 0.5, 100),
                rng.normal(3, 0.5, 100),
            ]
        )
        user = np.zeros((n, 1))
        user[100:, 0] = 1.0

        result = isat(
            y,
            user_indicators=user,
            user_indicator_names=["custom_step_100"],
            diagnostics=False,
        )
        assert "USER" in result.saturation_type
        assert result.n_indicators_retained >= 1


# ---------------------------------------------------------------------------
# Convenience methods
# ---------------------------------------------------------------------------


class TestConvenienceMethods:
    def test_ols_isat(self, rng):
        """OLS.isat() works end-to-end."""
        from regimes.models.ols import OLS

        y = np.concatenate(
            [
                rng.normal(0, 0.5, 100),
                rng.normal(3, 0.5, 100),
            ]
        )
        model = OLS(y, has_constant=True)
        result = model.isat(sis=True, diagnostics=False)

        assert isinstance(result, SaturationResults)
        assert result.n_indicators_retained >= 1

    def test_ar_isat(self, rng):
        """AR.isat() works end-to-end."""
        from regimes.models.ar import AR

        n = 200
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + rng.standard_normal()

        model = AR(y, lags=1)
        result = model.isat(sis=True, diagnostics=False)
        assert isinstance(result, SaturationResults)

    def test_adl_isat(self, rng):
        """ADL.isat() works end-to-end."""
        from regimes.models.adl import ADL

        n = 200
        x = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + 0.3 * x[t] + rng.standard_normal()

        model = ADL(y, x, lags=1, exog_lags=1)
        result = model.isat(sis=True, diagnostics=False)
        assert isinstance(result, SaturationResults)


# ---------------------------------------------------------------------------
# Top-level API access
# ---------------------------------------------------------------------------


class TestTopLevelAPI:
    def test_isat_from_top_level(self, rng):
        """isat is accessible from regimes top-level."""
        import regimes as rg

        y = rng.standard_normal(200)
        result = rg.isat(y, sis=True, diagnostics=False)
        assert isinstance(result, SaturationResults)

    def test_gets_search_from_top_level(self, rng):
        """gets_search is accessible from regimes top-level."""
        import regimes as rg

        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [1, 2] + rng.standard_normal(n)
        result = rg.gets_search(y, X, diagnostics=False)
        assert hasattr(result, "selected_model")

    def test_results_classes_from_top_level(self):
        """Result classes are accessible from top-level."""
        import regimes as rg

        assert hasattr(rg, "GETSResults")
        assert hasattr(rg, "SaturationResults")
        assert hasattr(rg, "TerminalModel")


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_sis_completes_quickly(self, rng):
        """SIS on n=200 should complete in reasonable time."""
        import time

        y = rng.standard_normal(200)
        start = time.time()
        isat(y, sis=True, diagnostics=False)
        elapsed = time.time() - start
        assert elapsed < 30, f"SIS took {elapsed:.1f}s, expected <30s"

    def test_summary_output(self, rng):
        """Summary produces non-empty string."""
        y = np.concatenate(
            [
                rng.normal(0, 0.5, 100),
                rng.normal(3, 0.5, 100),
            ]
        )
        result = isat(y, sis=True, diagnostics=False)
        s = result.summary()
        assert len(s) > 100
        assert "Indicator Saturation" in s
        assert "Regime Levels" in s
