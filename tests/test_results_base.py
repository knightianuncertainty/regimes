"""Tests for results base classes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import regimes as rg


class TestRegressionResultsBase:
    """Tests for RegressionResultsBase properties and methods."""

    @pytest.fixture
    def ols_results(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> rg.OLSResults:
        """Create OLS results for testing."""
        y, X = regression_data
        return rg.OLS(y, X, has_constant=False).fit()

    def test_df_model(self, ols_results: rg.OLSResults) -> None:
        """Test df_model property."""
        assert ols_results.df_model == len(ols_results.params)
        assert ols_results.df_model == 2  # const + slope

    def test_df_resid(self, ols_results: rg.OLSResults) -> None:
        """Test df_resid property."""
        assert ols_results.df_resid == ols_results.nobs - ols_results.df_model

    def test_tvalues(self, ols_results: rg.OLSResults) -> None:
        """Test tvalues property."""
        expected = ols_results.params / ols_results.bse
        np.testing.assert_array_almost_equal(ols_results.tvalues, expected)

    def test_pvalues_range(self, ols_results: rg.OLSResults) -> None:
        """Test that pvalues are between 0 and 1."""
        assert all(0 <= p <= 1 for p in ols_results.pvalues)

    def test_ssr(self, ols_results: rg.OLSResults) -> None:
        """Test ssr property."""
        expected = np.sum(ols_results.resid**2)
        assert np.isclose(ols_results.ssr, expected)

    def test_mse_resid(self, ols_results: rg.OLSResults) -> None:
        """Test mse_resid property."""
        expected = ols_results.ssr / ols_results.df_resid
        assert np.isclose(ols_results.mse_resid, expected)

    def test_rsquared_bounds(self, ols_results: rg.OLSResults) -> None:
        """Test rsquared is between 0 and 1."""
        assert 0 <= ols_results.rsquared <= 1

    def test_rsquared_adj_bounds(self, ols_results: rg.OLSResults) -> None:
        """Test rsquared_adj is reasonable."""
        # Adjusted R-squared can be negative but should be close to R-squared
        assert ols_results.rsquared_adj <= ols_results.rsquared

    def test_conf_int_shape(self, ols_results: rg.OLSResults) -> None:
        """Test conf_int shape."""
        ci = ols_results.conf_int()
        assert ci.shape == (ols_results.df_model, 2)

    def test_conf_int_lower_less_than_upper(self, ols_results: rg.OLSResults) -> None:
        """Test CI lower bounds are less than upper bounds."""
        ci = ols_results.conf_int()
        assert all(ci[:, 0] < ci[:, 1])

    def test_conf_int_contains_params(self, ols_results: rg.OLSResults) -> None:
        """Test CI contains the point estimates."""
        ci = ols_results.conf_int()
        assert all(ci[:, 0] < ols_results.params)
        assert all(ci[:, 1] > ols_results.params)

    def test_conf_int_different_alpha(self, ols_results: rg.OLSResults) -> None:
        """Test CI width changes with alpha."""
        ci_95 = ols_results.conf_int(alpha=0.05)
        ci_99 = ols_results.conf_int(alpha=0.01)

        # 99% CI should be wider than 95% CI
        width_95 = ci_95[:, 1] - ci_95[:, 0]
        width_99 = ci_99[:, 1] - ci_99[:, 0]
        assert all(width_99 > width_95)

    def test_cov_params(self, ols_results: rg.OLSResults) -> None:
        """Test cov_params method."""
        cov = ols_results.cov_params()
        assert cov.shape == (ols_results.df_model, ols_results.df_model)
        # Should be symmetric
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_summary_contains_key_info(self, ols_results: rg.OLSResults) -> None:
        """Test summary contains key information."""
        summary = ols_results.summary()

        assert "R-squared" in summary
        assert "coef" in summary
        assert "std err" in summary
        assert "P>|t|" in summary

    def test_to_dataframe(self, ols_results: rg.OLSResults) -> None:
        """Test to_dataframe conversion."""
        df = ols_results.to_dataframe()

        assert "coef" in df.columns
        assert "std_err" in df.columns
        assert "t" in df.columns
        assert "P>|t|" in df.columns
        assert "ci_lower" in df.columns
        assert "ci_upper" in df.columns

    def test_to_dataframe_index(self, ols_results: rg.OLSResults) -> None:
        """Test to_dataframe uses param_names as index."""
        df = ols_results.to_dataframe()

        expected_names = ols_results.param_names or [
            f"x{i}" for i in range(len(ols_results.params))
        ]
        assert list(df.index) == list(expected_names)


class TestARResults:
    """Tests for AR-specific results properties."""

    @pytest.fixture
    def ar_results(self, ar1_data: NDArray[np.floating[Any]]) -> rg.ARResults:
        """Create AR results for testing."""
        return rg.AR(ar1_data, lags=1).fit()

    def test_ar_params(self, ar_results: rg.ARResults) -> None:
        """Test ar_params extraction."""
        ar_params = ar_results.ar_params
        assert ar_params is not None
        assert len(ar_params) == 1  # AR(1)

    def test_roots(self, ar_results: rg.ARResults) -> None:
        """Test AR polynomial roots."""
        roots = ar_results.roots
        assert len(roots) == 1
        # For stationary AR(1), root should be outside unit circle
        assert abs(roots[0]) > 1

    def test_is_stationary(self, ar_results: rg.ARResults) -> None:
        """Test stationarity check."""
        # AR(1) with phi=0.7 should be stationary
        assert ar_results.is_stationary

    def test_aic_bic(self, ar_results: rg.ARResults) -> None:
        """Test AIC and BIC calculation."""
        assert not np.isnan(ar_results.aic)
        assert not np.isnan(ar_results.bic)

    def test_sigma_properties(self, ar_results: rg.ARResults) -> None:
        """Test sigma properties."""
        assert ar_results.sigma_squared > 0
        assert np.isclose(ar_results.sigma, np.sqrt(ar_results.sigma_squared))


class TestADLResults:
    """Tests for ADL-specific results properties."""

    @pytest.fixture
    def adl_results(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> rg.ADLResults:
        """Create ADL results for testing."""
        y, x = adl_data
        return rg.ADL(y, x, lags=1, exog_lags=1).fit()

    def test_ar_params(self, adl_results: rg.ADLResults) -> None:
        """Test AR params extraction."""
        ar_params = adl_results.ar_params
        assert ar_params is not None
        assert len(ar_params) == 1

    def test_dl_params(self, adl_results: rg.ADLResults) -> None:
        """Test distributed lag params."""
        dl_params = adl_results.dl_params
        assert dl_params is not None
        assert "x0" in dl_params
        # ADL(1,1): x0 has lags 0 and 1 -> 2 coefficients
        assert len(dl_params["x0"]) == 2

    def test_cumulative_effect(self, adl_results: rg.ADLResults) -> None:
        """Test cumulative effect calculation."""
        cum_eff = adl_results.cumulative_effect
        assert "x0" in cum_eff
        # Should be sum of DL coefficients
        expected = np.sum(adl_results.dl_params["x0"])
        assert np.isclose(cum_eff["x0"], expected)

    def test_long_run_multiplier(self, adl_results: rg.ADLResults) -> None:
        """Test long-run multiplier calculation."""
        lr_mult = adl_results.long_run_multiplier
        assert "x0" in lr_mult
        # For stationary model, should be finite
        if adl_results.is_stationary:
            assert not np.isnan(lr_mult["x0"])

    def test_exog_lags_stored(self, adl_results: rg.ADLResults) -> None:
        """Test exog_lags stored correctly."""
        assert "x0" in adl_results.exog_lags
        assert adl_results.exog_lags["x0"] == [0, 1]


class TestBreakResultsBase:
    """Tests for BreakResultsBase properties."""

    @pytest.fixture
    def bp_results(
        self, data_with_mean_shift: tuple[NDArray[np.floating[Any]], int]
    ) -> rg.BaiPerronResults:
        """Create BaiPerron results for testing."""
        y, _ = data_with_mean_shift
        test = rg.BaiPerronTest(y)
        return test.fit(max_breaks=2)

    def test_n_breaks(self, bp_results: rg.BaiPerronResults) -> None:
        """Test n_breaks property."""
        assert bp_results.n_breaks >= 0

    def test_n_regimes(self, bp_results: rg.BaiPerronResults) -> None:
        """Test n_regimes property."""
        assert bp_results.n_regimes == bp_results.n_breaks + 1

    def test_break_indices(self, bp_results: rg.BaiPerronResults) -> None:
        """Test break_indices property."""
        if bp_results.n_breaks > 0:
            assert len(bp_results.break_indices) == bp_results.n_breaks
            # Should be sorted
            assert list(bp_results.break_indices) == sorted(bp_results.break_indices)

    def test_break_dates_alias(self, bp_results: rg.BaiPerronResults) -> None:
        """Test break_dates is alias for break_indices."""
        assert bp_results.break_dates == bp_results.break_indices

    def test_summary_no_breaks(self, simple_data: NDArray[np.floating[Any]]) -> None:
        """Test summary when no breaks detected."""
        test = rg.BaiPerronTest(simple_data)
        results = test.fit(max_breaks=1)

        summary = results.summary()
        if results.n_breaks == 0:
            assert "No structural breaks" in summary or "0" in summary


class TestRollingResultsBase:
    """Tests for RollingResultsBase properties."""

    @pytest.fixture
    def rolling_results(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> rg.RollingOLSResults:
        """Create rolling results for testing."""
        y, X = regression_data
        return rg.RollingOLS(y, X, window=30).fit()

    def test_is_rolling(self, rolling_results: rg.RollingOLSResults) -> None:
        """Test is_rolling property."""
        assert rolling_results.is_rolling

    def test_window(self, rolling_results: rg.RollingOLSResults) -> None:
        """Test window property."""
        assert rolling_results.window == 30

    def test_n_valid(self, rolling_results: rg.RollingOLSResults) -> None:
        """Test n_valid property."""
        assert rolling_results.n_valid > 0

    def test_params_shape(self, rolling_results: rg.RollingOLSResults) -> None:
        """Test params array shape."""
        assert rolling_results.params.shape[0] == rolling_results.nobs
        assert rolling_results.params.shape[1] == rolling_results.n_params

    def test_tvalues(self, rolling_results: rg.RollingOLSResults) -> None:
        """Test tvalues computation."""
        expected = rolling_results.params / rolling_results.bse
        np.testing.assert_array_almost_equal(rolling_results.tvalues, expected)

    def test_conf_int_shape(self, rolling_results: rg.RollingOLSResults) -> None:
        """Test conf_int shape."""
        ci = rolling_results.conf_int()
        assert ci.shape == (rolling_results.nobs, rolling_results.n_params, 2)

    def test_to_dataframe(self, rolling_results: rg.RollingOLSResults) -> None:
        """Test DataFrame conversion."""
        df = rolling_results.to_dataframe()
        assert len(df) == rolling_results.nobs
        assert len(df.columns) == rolling_results.n_params


class TestRecursiveResults:
    """Tests for recursive estimation results."""

    @pytest.fixture
    def recursive_results(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> rg.RecursiveOLSResults:
        """Create recursive results for testing."""
        y, X = regression_data
        return rg.RecursiveOLS(y, X, min_nobs=20).fit()

    def test_is_not_rolling(self, recursive_results: rg.RecursiveOLSResults) -> None:
        """Test is_rolling is False for recursive."""
        assert not recursive_results.is_rolling

    def test_window_is_none(self, recursive_results: rg.RecursiveOLSResults) -> None:
        """Test window is None for recursive."""
        assert recursive_results.window is None

    def test_min_nobs(self, recursive_results: rg.RecursiveOLSResults) -> None:
        """Test min_nobs property."""
        assert recursive_results.min_nobs == 20
