"""Tests for indicator saturation orchestration (isat)."""

from __future__ import annotations

import numpy as np
import pytest

from regimes.gets.saturation import (
    SaturationResults,
    _build_core_regressors,
    _extract_tau,
    isat,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestExtractTau:
    def test_step(self):
        assert _extract_tau("step_50") == 50

    def test_impulse(self):
        assert _extract_tau("impulse_120") == 120

    def test_trend(self):
        assert _extract_tau("trend_30") == 30

    def test_mis(self):
        assert _extract_tau("x0*step_75") == 75

    def test_non_indicator(self):
        assert _extract_tau("const") is None
        assert _extract_tau("y.L1") is None

    def test_user(self):
        assert _extract_tau("user_0") is None


class TestBuildCoreRegressors:
    def test_constant_only(self, rng):
        y_in = rng.standard_normal(100)
        y, X, names, maxlag = _build_core_regressors(
            y_in, None, None, None, has_constant=True
        )
        assert len(y) == 100
        assert X.shape == (100, 1)
        assert names == ["const"]
        assert maxlag == 0

    def test_ar_lags(self, rng):
        y_in = rng.standard_normal(200)
        y, X, names, maxlag = _build_core_regressors(
            y_in, None, ar_lags=2, exog_lags=None, has_constant=True
        )
        assert maxlag == 2
        assert len(y) == 198
        assert X.shape == (198, 3)  # const + y.L1 + y.L2
        assert names == ["const", "y.L1", "y.L2"]

    def test_exog(self, rng):
        y_in = rng.standard_normal(100)
        x = rng.standard_normal(100)
        y, X, names, maxlag = _build_core_regressors(
            y_in, x, None, None, has_constant=True
        )
        assert X.shape == (100, 2)  # const + x0
        assert names == ["const", "x0"]

    def test_ar_and_exog(self, rng):
        y_in = rng.standard_normal(200)
        x = rng.standard_normal(200)
        y, X, names, maxlag = _build_core_regressors(
            y_in, x, ar_lags=1, exog_lags=None, has_constant=True
        )
        assert maxlag == 1
        assert len(y) == 199
        assert X.shape == (199, 3)  # const + y.L1 + x0
        assert names == ["const", "y.L1", "x0"]

    def test_exog_lags(self, rng):
        y_in = rng.standard_normal(200)
        x = rng.standard_normal(200)
        y, X, names, maxlag = _build_core_regressors(
            y_in, x, ar_lags=None, exog_lags=1, has_constant=True
        )
        assert maxlag == 1
        assert len(y) == 199
        assert "x0.L1" in names

    def test_no_constant(self, rng):
        y_in = rng.standard_normal(100)
        y, X, names, maxlag = _build_core_regressors(
            y_in, None, None, None, has_constant=False
        )
        assert X.shape[1] == 0
        assert names == []


# ---------------------------------------------------------------------------
# isat — basic
# ---------------------------------------------------------------------------


class TestIsatBasic:
    def test_sis_returns_saturation_results(self, rng):
        y = rng.standard_normal(200)
        result = isat(y, sis=True, diagnostics=False)
        assert isinstance(result, SaturationResults)
        assert result.saturation_type == "SIS"
        assert result.selected_results is not None

    def test_iis(self, rng):
        y = rng.standard_normal(200)
        result = isat(y, iis=True, diagnostics=False)
        assert result.saturation_type == "IIS"

    def test_sis_plus_mis(self, rng):
        n = 200
        y = np.zeros(n)
        y[1:] = 0.5 * y[:-1].copy()
        y += rng.standard_normal(n)
        result = isat(y, ar_lags=1, sis=True, mis=True, diagnostics=False)
        assert "SIS" in result.saturation_type
        assert "MIS" in result.saturation_type

    def test_no_indicators_raises(self, rng):
        y = rng.standard_normal(100)
        with pytest.raises(ValueError, match="No indicators"):
            isat(y, diagnostics=False)

    def test_summary(self, rng):
        y = rng.standard_normal(200)
        result = isat(y, sis=True, diagnostics=False)
        s = result.summary()
        assert "Indicator Saturation" in s
        assert "SIS" in s

    def test_break_dates_list(self, rng):
        y = rng.standard_normal(200)
        result = isat(y, sis=True, diagnostics=False)
        assert isinstance(result.break_dates, list)

    def test_retained_mask_shape(self, rng):
        y = rng.standard_normal(200)
        result = isat(y, sis=True, diagnostics=False)
        assert result.retained_indicator_mask.dtype == bool
        assert len(result.retained_indicator_mask) == result.n_indicators_initial

    def test_custom_user_indicators(self, rng):
        n = 200
        y = rng.standard_normal(n)
        # One custom indicator at t=100
        user = np.zeros((n, 1))
        user[100:, 0] = 1.0
        result = isat(
            y,
            user_indicators=user,
            user_indicator_names=["custom_step"],
            diagnostics=False,
        )
        assert "USER" in result.saturation_type

    def test_user_indicators_wrong_size_raises(self, rng):
        y = rng.standard_normal(200)
        user = np.zeros((100, 1))  # Wrong size
        with pytest.raises(ValueError, match="rows"):
            isat(y, user_indicators=user, diagnostics=False)


# ---------------------------------------------------------------------------
# isat — known DGP
# ---------------------------------------------------------------------------


class TestIsatKnownDGP:
    def test_sis_detects_mean_shift(self, rng):
        """SIS should detect a clear level shift."""
        y = np.concatenate(
            [
                rng.normal(0, 0.5, 100),
                rng.normal(3, 0.5, 100),
            ]
        )
        result = isat(y, sis=True, alpha=0.05, diagnostics=False)
        assert result.n_indicators_retained >= 1
        # Break should be near t=100
        if result.break_dates:
            assert any(90 <= bd <= 110 for bd in result.break_dates), (
                f"Expected break near 100, got {result.break_dates}"
            )

    def test_mis_detects_ar_break(self, rng):
        """MIS should detect a change in AR coefficient."""
        n = 300
        y = np.zeros(n)
        # AR(1) with phi=0.3 for t<150, phi=0.8 for t>=150
        for t in range(1, 150):
            y[t] = 0.3 * y[t - 1] + rng.standard_normal()
        for t in range(150, n):
            y[t] = 0.8 * y[t - 1] + rng.standard_normal()

        result = isat(
            y,
            ar_lags=1,
            mis=True,
            sis=True,
            alpha=0.05,
            diagnostics=False,
        )
        # Should detect at least one break
        assert result.n_indicators_retained >= 1

    def test_combined_sis_mis(self, rng):
        """Level shift + AR break at different dates."""
        n = 300
        y = np.zeros(n)
        # Regime 1: const=0, phi=0.3
        for t in range(1, 100):
            y[t] = 0 + 0.3 * y[t - 1] + rng.standard_normal()
        # Regime 2: const=2, phi=0.3
        for t in range(100, 200):
            y[t] = 2 + 0.3 * y[t - 1] + rng.standard_normal()
        # Regime 3: const=2, phi=0.8
        for t in range(200, n):
            y[t] = 2 + 0.8 * y[t - 1] + rng.standard_normal()

        result = isat(
            y,
            ar_lags=1,
            sis=True,
            mis=True,
            alpha=0.05,
            diagnostics=False,
        )
        assert result.n_indicators_retained >= 1

    def test_null_dgp_low_retention(self, rng):
        """Under null (pure noise), retention should be low."""
        n = 200
        y = rng.standard_normal(n)
        result = isat(y, sis=True, alpha=0.01, diagnostics=False)
        # With alpha=0.01 on noise, expect ~2% false retention
        retention_rate = result.n_indicators_retained / result.n_indicators_initial
        assert retention_rate < 0.15, (
            f"Retention rate {retention_rate:.2%} too high for null DGP"
        )


# ---------------------------------------------------------------------------
# isat — block splitting
# ---------------------------------------------------------------------------


class TestIsatBlocks:
    def test_explicit_blocks(self, rng):
        n = 200
        y = rng.standard_normal(n)
        result = isat(y, sis=True, n_blocks=4, diagnostics=False)
        assert isinstance(result, SaturationResults)

    def test_max_block_size(self, rng):
        n = 200
        y = rng.standard_normal(n)
        result = isat(y, sis=True, max_block_size=50, diagnostics=False)
        assert isinstance(result, SaturationResults)

    def test_single_block_equivalent(self, rng):
        """With max_block_size >= n_indicators, behaves as single block."""
        n = 100
        y = rng.standard_normal(n)
        result = isat(y, sis=True, max_block_size=1000, diagnostics=False)
        assert isinstance(result, SaturationResults)


# ---------------------------------------------------------------------------
# isat — regime levels
# ---------------------------------------------------------------------------


class TestIsatRegimeLevels:
    def test_regime_levels_populated(self, rng):
        y = np.concatenate(
            [
                rng.normal(0, 0.5, 100),
                rng.normal(3, 0.5, 100),
            ]
        )
        result = isat(y, sis=True, diagnostics=False)
        assert result.regime_levels is not None
        assert "const" in result.regime_levels.param_regimes

    def test_shifts_populated(self, rng):
        y = np.concatenate(
            [
                rng.normal(0, 0.5, 100),
                rng.normal(3, 0.5, 100),
            ]
        )
        result = isat(y, sis=True, diagnostics=False)
        assert result.shifts is not None
        assert "const" in result.shifts.initial_levels

    def test_n_regimes(self, rng):
        y = rng.standard_normal(200)
        result = isat(y, sis=True, diagnostics=False)
        assert result.n_regimes >= 1


# ---------------------------------------------------------------------------
# isat — MIS targeted
# ---------------------------------------------------------------------------


class TestIsatMISTargeted:
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

    def test_mis_true_all_variables(self, rng):
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
# isat — covariance types
# ---------------------------------------------------------------------------


class TestIsatCovTypes:
    @pytest.mark.parametrize("cov_type", ["nonrobust", "HC1", "HAC"])
    def test_cov_type(self, rng, cov_type):
        y = rng.standard_normal(200)
        result = isat(y, sis=True, cov_type=cov_type, diagnostics=False)
        assert isinstance(result, SaturationResults)


# ---------------------------------------------------------------------------
# isat — selection criteria
# ---------------------------------------------------------------------------


class TestIsatSelection:
    @pytest.mark.parametrize("criterion", ["bic", "aic", "hq"])
    def test_selection_criterion(self, rng, criterion):
        y = rng.standard_normal(200)
        result = isat(y, sis=True, selection=criterion, diagnostics=False)
        assert result.gets_results.selection_criterion == criterion
