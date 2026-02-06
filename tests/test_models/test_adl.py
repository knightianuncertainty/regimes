"""Tests for ADL model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import regimes as rg

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def adl_data(
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Simulated ADL(1,1) process.

    y_t = 0.5 + 0.6*y_{t-1} + 0.3*x_t + 0.15*x_{t-1} + e_t

    Returns
    -------
    tuple[NDArray[np.floating], NDArray[np.floating]]
        y and x arrays.
    """
    n = 200
    x = rng.standard_normal(n)
    y = np.zeros(n)

    for t in range(1, n):
        y[t] = (
            0.5 + 0.6 * y[t - 1] + 0.3 * x[t] + 0.15 * x[t - 1] + rng.standard_normal()
        )

    return y, x


@pytest.fixture
def adl_data_multiple_exog(
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Simulated ADL with multiple exogenous variables.

    y_t = 0.5 + 0.5*y_{t-1} + 0.3*x1_t + 0.1*x1_{t-1} + 0.2*x2_t + e_t

    Returns
    -------
    tuple[NDArray[np.floating], NDArray[np.floating]]
        y and X (n, 2) arrays.
    """
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = np.zeros(n)

    for t in range(1, n):
        y[t] = (
            0.5
            + 0.5 * y[t - 1]
            + 0.3 * x1[t]
            + 0.1 * x1[t - 1]
            + 0.2 * x2[t]
            + rng.standard_normal()
        )

    X = np.column_stack([x1, x2])
    return y, X


@pytest.fixture
def adl_data_with_break(
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], int]:
    """ADL process with structural break in AR coefficient.

    First half: y_t = 0.5 + 0.3*y_{t-1} + 0.4*x_t + e_t
    Second half: y_t = 0.5 + 0.8*y_{t-1} + 0.4*x_t + e_t

    Returns
    -------
    tuple[NDArray[np.floating], NDArray[np.floating], int]
        y, x arrays, and break location.
    """
    n = 200
    break_point = 100
    x = rng.standard_normal(n)
    y = np.zeros(n)

    # First regime: phi = 0.3
    for t in range(1, break_point):
        y[t] = 0.5 + 0.3 * y[t - 1] + 0.4 * x[t] + rng.standard_normal()

    # Second regime: phi = 0.8
    for t in range(break_point, n):
        y[t] = 0.5 + 0.8 * y[t - 1] + 0.4 * x[t] + rng.standard_normal()

    return y, x, break_point


# ============================================================================
# Basic ADL Tests
# ============================================================================


class TestADLBasic:
    """Basic ADL model tests."""

    def test_adl11_estimation(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test ADL(1,1) coefficient estimation."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)
        results = model.fit()

        # Check we have the right number of parameters
        # const + y.L1 + x0 + x0.L1 = 4
        assert len(results.params) == 4

        # True AR coef = 0.6, should estimate close to that
        ar_coef = results.ar_params[0] if results.ar_params is not None else 0
        assert np.isclose(ar_coef, 0.6, atol=0.15)

    def test_adl_parameter_names(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test ADL parameter naming convention."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)
        results = model.fit()

        param_names = results.param_names or []
        assert "const" in param_names
        assert "y.L1" in param_names
        assert "x0" in param_names
        assert "x0.L1" in param_names

    def test_adl_stationarity(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test stationarity check for ADL."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)
        results = model.fit()

        # AR(1) with phi=0.6 is stationary
        assert results.is_stationary

    def test_adl_nobs(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test effective number of observations."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=2, exog_lags=3)
        results = model.fit()

        # Should lose max(2, 3) = 3 observations to lags
        assert results.nobs == len(y) - 3

    def test_adl_no_ar_lags(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test ADL with no AR lags (pure distributed lag model)."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=0, exog_lags=2)
        results = model.fit()

        # Should have: const + x0 + x0.L1 + x0.L2 = 4
        assert len(results.params) == 4
        assert results.ar_params is not None
        assert len(results.ar_params) == 0


class TestADLMultipleExog:
    """Test ADL with multiple exogenous variables."""

    def test_adl_multiple_exog_same_lags(
        self,
        adl_data_multiple_exog: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
    ) -> None:
        """Test ADL with multiple exog variables, same lag structure."""
        y, X = adl_data_multiple_exog
        model = rg.ADL(y, X, lags=1, exog_lags=1)
        results = model.fit()

        # const + y.L1 + x0 + x0.L1 + x1 + x1.L1 = 6
        assert len(results.params) == 6

        param_names = results.param_names or []
        assert "x0" in param_names
        assert "x0.L1" in param_names
        assert "x1" in param_names
        assert "x1.L1" in param_names

    def test_adl_multiple_exog_different_lags(
        self,
        adl_data_multiple_exog: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
    ) -> None:
        """Test ADL with multiple exog variables, different lag structures."""
        y, X = adl_data_multiple_exog
        # x0: lags 0, 1, 2; x1: contemporaneous only
        model = rg.ADL(y, X, lags=1, exog_lags={0: 2, 1: 0})
        results = model.fit()

        # const + y.L1 + x0 + x0.L1 + x0.L2 + x1 = 6
        assert len(results.params) == 6

        param_names = results.param_names or []
        assert "x0" in param_names
        assert "x0.L1" in param_names
        assert "x0.L2" in param_names
        assert "x1" in param_names
        assert "x1.L1" not in param_names


# ============================================================================
# Distributed Lag Properties Tests
# ============================================================================


class TestADLDistributedLagProperties:
    """Test distributed lag effect calculations."""

    def test_cumulative_effect(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test cumulative effect calculation."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)
        results = model.fit()

        cumulative = results.cumulative_effect

        # Should have entry for x0
        assert "x0" in cumulative

        # Cumulative effect = sum of all distributed lag coefficients
        # True: 0.3 + 0.15 = 0.45
        assert np.isclose(cumulative["x0"], 0.45, atol=0.15)

    def test_long_run_multiplier(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test long-run multiplier calculation."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)
        results = model.fit()

        lr_mult = results.long_run_multiplier

        # Should have entry for x0
        assert "x0" in lr_mult

        # Long-run = cumulative / (1 - AR sum)
        # True: (0.3 + 0.15) / (1 - 0.6) = 0.45 / 0.4 = 1.125
        assert np.isclose(lr_mult["x0"], 1.125, atol=0.4)

    def test_long_run_multiplier_nonstationary(self, rng: np.random.Generator) -> None:
        """Test long-run multiplier returns NaN for non-stationary model."""
        # Create data where estimated AR coefficient will be close to 1
        n = 200
        x = rng.standard_normal(n)
        y = np.cumsum(rng.standard_normal(n))  # Random walk

        model = rg.ADL(y, x, lags=1, exog_lags=0)
        results = model.fit()

        # For non-stationary case, AR sum >= 1, LR multiplier undefined
        if not results.is_stationary:
            lr_mult = results.long_run_multiplier
            assert np.isnan(lr_mult.get("x0", np.nan))


# ============================================================================
# Covariance Types Tests
# ============================================================================


class TestADLCovarianceTypes:
    """Test ADL with different covariance estimators."""

    def test_adl_nonrobust(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test nonrobust standard errors."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)
        results = model.fit(cov_type="nonrobust")

        assert results.cov_type == "nonrobust"
        assert all(results.bse > 0)

    def test_adl_hac(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test HAC standard errors."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)
        results = model.fit(cov_type="HAC")

        assert results.cov_type == "HAC"
        assert all(results.bse > 0)

    def test_adl_hc0(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test HC0 standard errors."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)
        results = model.fit(cov_type="HC0")

        assert results.cov_type == "HC0"
        assert all(results.bse > 0)


# ============================================================================
# Structural Breaks Tests
# ============================================================================


class TestADLWithBreaks:
    """Test ADL model with structural breaks."""

    def test_adl_with_break(
        self,
        adl_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test ADL with known structural break."""
        y, x, break_point = adl_data_with_break

        model = rg.ADL(y, x, lags=1, exog_lags=0, breaks=[break_point])
        results = model.fit()

        # Should have parameters for both regimes
        # 2 regimes * (const + y.L1 + x0) = 6 params
        assert len(results.params) == 6

    def test_adl_break_estimates(
        self,
        adl_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test that AR coefficients differ across regimes."""
        y, x, break_point = adl_data_with_break

        model = rg.ADL(y, x, lags=1, exog_lags=0, breaks=[break_point])
        results = model.fit()

        param_names = results.param_names or []
        ar_regime1_idx = [i for i, n in enumerate(param_names) if "y.L1_regime1" in n]
        ar_regime2_idx = [i for i, n in enumerate(param_names) if "y.L1_regime2" in n]

        if ar_regime1_idx and ar_regime2_idx:
            ar_coef_0 = results.params[ar_regime1_idx[0]]
            ar_coef_1 = results.params[ar_regime2_idx[0]]

            # Regime 0: phi ≈ 0.3, Regime 1: phi ≈ 0.8
            assert np.isclose(ar_coef_0, 0.3, atol=0.2)
            assert np.isclose(ar_coef_1, 0.8, atol=0.2)


class TestADLVariableBreaks:
    """Test ADL model with variable-specific breaks."""

    def test_variable_breaks_validation_error(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test that breaks and variable_breaks cannot both be specified."""
        y, x = adl_data
        with pytest.raises(ValueError, match="Cannot specify both"):
            rg.ADL(
                y, x, lags=1, exog_lags=0, breaks=[100], variable_breaks={"const": [50]}
            )

    def test_variable_breaks_const_only(
        self,
        adl_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test variable breaks on constant only (intercept shift)."""
        y, x, break_point = adl_data_with_break

        model = rg.ADL(
            y, x, lags=1, exog_lags=0, variable_breaks={"const": [break_point]}
        )
        results = model.fit()

        param_names = results.param_names or []
        assert "const_regime1" in param_names
        assert "const_regime2" in param_names
        assert "y.L1" in param_names
        assert "y.L1_regime1" not in param_names

    def test_variable_breaks_ar_only(
        self,
        adl_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test variable breaks on AR coefficient only."""
        y, x, break_point = adl_data_with_break

        model = rg.ADL(
            y, x, lags=1, exog_lags=0, variable_breaks={"y.L1": [break_point]}
        )
        results = model.fit()

        param_names = results.param_names or []
        assert "const" in param_names
        assert "y.L1_regime1" in param_names
        assert "y.L1_regime2" in param_names
        assert "const_regime1" not in param_names


# ============================================================================
# Lag Selection Tests
# ============================================================================


class TestADLLagSelection:
    """Test lag selection functionality."""

    def test_select_lags_aic(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test lag selection with AIC criterion."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=0)

        optimal_p, optimal_q = model.select_lags(
            max_ar_lags=3, max_exog_lags=3, criterion="aic"
        )

        # Should select some reasonable lag structure
        assert 0 <= optimal_p <= 3
        assert 0 <= optimal_q <= 3

    def test_select_lags_bic(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test lag selection with BIC criterion."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=0)

        optimal_p, optimal_q = model.select_lags(
            max_ar_lags=3, max_exog_lags=3, criterion="bic"
        )

        # BIC typically selects more parsimonious models
        assert 0 <= optimal_p <= 3
        assert 0 <= optimal_q <= 3


# ============================================================================
# Results Tests
# ============================================================================


class TestADLResults:
    """Test ADL results object."""

    def test_summary(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test summary generation."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)
        results = model.fit()

        summary = results.summary()
        assert "ADL Model Results" in summary
        assert "ADL(1,1)" in summary
        assert "y.L1" in summary
        assert "x0" in summary
        assert "Distributed Lag Effects" in summary

    def test_information_criteria(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test AIC and BIC calculation."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)
        results = model.fit()

        assert not np.isnan(results.aic)
        assert not np.isnan(results.bic)
        # BIC penalizes more than AIC for large samples
        assert results.bic > results.aic

    def test_roots(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test AR polynomial roots."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)
        results = model.fit()

        roots = results.roots
        assert len(roots) == 1

        # For stationary AR(1), root should be outside unit circle
        assert np.abs(roots[0]) > 1

    def test_exog_lags_in_results(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test that exog_lags is correctly stored in results."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=2)
        results = model.fit()

        assert "x0" in results.exog_lags
        assert results.exog_lags["x0"] == [0, 1, 2]


# ============================================================================
# Trend Tests
# ============================================================================


class TestADLTrend:
    """Test ADL with different trend specifications."""

    def test_adl_no_constant(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test ADL without constant."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=0, trend="n")
        results = model.fit()

        # y.L1 + x0 only
        assert len(results.params) == 2

    def test_adl_with_trend(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test ADL with constant and trend."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=0, trend="ct")
        results = model.fit()

        # const + trend + y.L1 + x0
        assert len(results.params) == 4
        param_names = results.param_names or []
        assert any("trend" in n for n in param_names)


# ============================================================================
# Rolling/Recursive Tests
# ============================================================================


class TestADLRolling:
    """Test rolling ADL estimation."""

    def test_rolling_adl(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test rolling ADL estimation."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)
        rolling_results = model.rolling(window=60).fit()

        assert rolling_results.nobs == len(y) - 1  # maxlag = 1
        assert rolling_results.window == 60
        assert rolling_results.n_valid > 0
        assert rolling_results.lags == [1]

    def test_recursive_adl(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test recursive ADL estimation."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)
        recursive_results = model.recursive(min_nobs=30).fit()

        assert recursive_results.nobs == len(y) - 1
        assert recursive_results.window is None
        assert recursive_results.n_valid > 0

    def test_rolling_adl_summary(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test rolling ADL summary."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)
        rolling_results = model.rolling(window=60).fit()

        summary = rolling_results.summary()
        assert "Rolling ADL(1,1) Results" in summary
        assert "Window Size" in summary


# ============================================================================
# Bai-Perron Integration Tests
# ============================================================================


class TestADLBaiPerron:
    """Test Bai-Perron integration with ADL."""

    def test_bai_perron_from_adl(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test Bai-Perron test from ADL model."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=0)
        bp_results = model.bai_perron(max_breaks=2)

        # Should run without error
        assert bp_results.nobs > 0
        assert bp_results.n_breaks >= 0

    def test_bai_perron_from_model_adl(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test BaiPerronTest.from_model with ADL."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=0)
        test = rg.BaiPerronTest.from_model(model)
        bp_results = test.fit(max_breaks=2)

        assert bp_results.nobs > 0


# ============================================================================
# Edge Cases
# ============================================================================


class TestADLEdgeCases:
    """Test edge cases and error handling."""

    def test_adl_missing_exog_raises(self, rng: np.random.Generator) -> None:
        """Test that ADL without exog raises error."""
        y = rng.standard_normal(100)
        with pytest.raises(ValueError, match="exog is required"):
            rg.ADL(y, None, lags=1)  # type: ignore[arg-type]

    def test_adl_invalid_trend(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test error on invalid trend."""
        y, x = adl_data
        with pytest.raises(ValueError, match="trend must be"):
            rg.ADL(y, x, lags=1, exog_lags=0, trend="invalid")

    def test_adl_invalid_lags(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test error on invalid AR lags."""
        y, x = adl_data
        with pytest.raises(ValueError, match="non-negative"):
            rg.ADL(y, x, lags=-1, exog_lags=0)

    def test_adl_invalid_exog_lag_name(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test error on invalid exog variable name."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags={"invalid_var": 1})
        with pytest.raises(ValueError, match="not found"):
            model.fit()

    def test_adl_1d_exog(self, rng: np.random.Generator) -> None:
        """Test ADL handles 1D exog correctly."""
        n = 100
        y = rng.standard_normal(n)
        x = rng.standard_normal(n)  # 1D array

        model = rg.ADL(y, x, lags=1, exog_lags=0)
        results = model.fit()

        assert len(results.params) == 3  # const + y.L1 + x0
