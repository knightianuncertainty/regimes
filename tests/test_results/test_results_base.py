"""Tests for results base classes."""

from __future__ import annotations

import numpy as np
import pytest

from regimes.results.base import BreakResultsBase, RegressionResultsBase


class TestRegressionResultsBase:
    """Test RegressionResultsBase properties and methods."""

    @pytest.fixture
    def basic_results(self) -> RegressionResultsBase:
        """Create a minimal RegressionResultsBase for testing."""
        n = 50
        params = np.array([1.0, 2.0])
        bse = np.array([0.1, 0.2])
        resid = np.random.default_rng(42).standard_normal(n)
        fittedvalues = np.random.default_rng(42).standard_normal(n)
        cov_matrix = np.diag(bse**2)
        return RegressionResultsBase(
            params=params,
            bse=bse,
            resid=resid,
            fittedvalues=fittedvalues,
            nobs=n,
            cov_params_matrix=cov_matrix,
            cov_type="nonrobust",
            model_name="TestModel",
            param_names=["const", "x1"],
        )

    def test_ssr_computed(self, basic_results: RegressionResultsBase) -> None:
        """SSR should be computed from residuals when _ssr is None."""
        expected = float(np.sum(basic_results.resid**2))
        assert abs(basic_results.ssr - expected) < 1e-10

    def test_ssr_cached(self) -> None:
        """SSR should return cached value when _ssr is set."""
        n = 50
        results = RegressionResultsBase(
            params=np.array([1.0]),
            bse=np.array([0.1]),
            resid=np.zeros(n),
            fittedvalues=np.zeros(n),
            nobs=n,
            cov_params_matrix=np.array([[0.01]]),
            model_name="TestModel",
            _ssr=42.0,
        )
        assert results.ssr == 42.0

    def test_rsquared_no_tss(self, basic_results: RegressionResultsBase) -> None:
        """R-squared should be NaN when _tss is None."""
        assert np.isnan(basic_results.rsquared)

    def test_rsquared_adj_no_tss(self, basic_results: RegressionResultsBase) -> None:
        """Adjusted R-squared should be NaN when _tss is None."""
        assert np.isnan(basic_results.rsquared_adj)

    def test_rsquared_zero_tss(self) -> None:
        """R-squared should be NaN when _tss is zero."""
        n = 50
        results = RegressionResultsBase(
            params=np.array([1.0]),
            bse=np.array([0.1]),
            resid=np.zeros(n),
            fittedvalues=np.zeros(n),
            nobs=n,
            cov_params_matrix=np.array([[0.01]]),
            model_name="TestModel",
            _tss=0.0,
        )
        assert np.isnan(results.rsquared)
        assert np.isnan(results.rsquared_adj)

    def test_rsquared_with_tss(self) -> None:
        """R-squared should be computed when _tss is set."""
        n = 50
        resid = np.ones(n) * 0.1
        results = RegressionResultsBase(
            params=np.array([1.0]),
            bse=np.array([0.1]),
            resid=resid,
            fittedvalues=np.zeros(n),
            nobs=n,
            cov_params_matrix=np.array([[0.01]]),
            model_name="TestModel",
            _ssr=float(np.sum(resid**2)),
            _tss=100.0,
        )
        expected = 1 - results.ssr / 100.0
        assert abs(results.rsquared - expected) < 1e-10
        assert np.isfinite(results.rsquared_adj)

    def test_summary(self, basic_results: RegressionResultsBase) -> None:
        """Summary should produce formatted text with all sections."""
        s = basic_results.summary()
        assert "=" * 78 in s
        assert "TestModel" in s
        assert "coef" in s
        assert "std err" in s
        assert "const" in s
        assert "x1" in s

    def test_summary_auto_names(self) -> None:
        """Summary should auto-generate names when param_names is None."""
        n = 50
        results = RegressionResultsBase(
            params=np.array([1.0, 2.0]),
            bse=np.array([0.1, 0.2]),
            resid=np.zeros(n),
            fittedvalues=np.zeros(n),
            nobs=n,
            cov_params_matrix=np.diag([0.01, 0.04]),
            model_name="TestModel",
        )
        s = results.summary()
        assert "x0" in s
        assert "x1" in s


class TestBreakResultsBase:
    """Test BreakResultsBase properties and methods."""

    def test_n_regimes(self) -> None:
        results = BreakResultsBase(
            params=np.array([1.0, 2.0, 3.0]),
            nobs=200,
            model_name="BreakModel",
            break_indices=[100],
            n_breaks=1,
        )
        assert results.n_regimes == 2

    def test_break_dates(self) -> None:
        results = BreakResultsBase(
            params=np.array([1.0, 2.0]),
            nobs=200,
            model_name="BreakModel",
            break_indices=[80, 160],
            n_breaks=2,
        )
        assert results.break_dates == [80, 160]

    def test_df_model(self) -> None:
        results = BreakResultsBase(
            params=np.array([1.0, 2.0, 3.0]),
            nobs=200,
            model_name="BreakModel",
            break_indices=[100],
            n_breaks=1,
        )
        assert results.df_model == 3

    def test_summary_with_breaks(self) -> None:
        results = BreakResultsBase(
            params=np.array([1.0, 2.0]),
            nobs=200,
            model_name="BreakModel",
            break_indices=[80, 160],
            n_breaks=2,
        )
        s = results.summary()
        assert "BreakModel" in s
        assert "Number of breaks:       2" in s
        assert "Number of regimes:      3" in s
        assert "Break 1: observation 80" in s
        assert "Break 2: observation 160" in s

    def test_summary_no_breaks(self) -> None:
        results = BreakResultsBase(
            params=np.array([1.0]),
            nobs=200,
            model_name="BreakModel",
            break_indices=[],
            n_breaks=0,
        )
        s = results.summary()
        assert "No structural breaks detected." in s
        assert "Number of breaks:       0" in s

    def test_df_resid(self) -> None:
        results = BreakResultsBase(
            params=np.array([1.0, 2.0]),
            nobs=200,
            model_name="BreakModel",
            break_indices=[100],
            n_breaks=1,
        )
        assert results.df_resid == 198  # 200 - 2
