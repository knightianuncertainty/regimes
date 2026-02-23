"""Tests for AR model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import regimes as rg


class TestARBasic:
    """Basic AR model tests."""

    def test_ar1_estimation(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test AR(1) coefficient estimation."""
        model = rg.AR(ar1_data, lags=1)
        results = model.fit()

        # True phi = 0.7, should estimate close to that
        ar_coef = results.ar_params[0] if results.ar_params is not None else 0
        assert np.isclose(ar_coef, 0.7, atol=0.15)

    def test_ar1_stationarity(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test stationarity check for AR(1)."""
        model = rg.AR(ar1_data, lags=1)
        results = model.fit()

        # AR(1) with phi=0.7 is stationary
        assert results.is_stationary

    def test_ar_multiple_lags(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test AR with multiple lags."""
        model = rg.AR(ar1_data, lags=3)
        results = model.fit()

        assert len(results.params) == 4  # constant + 3 AR coefficients
        assert len(results.lags) == 3
        assert results.lags == [1, 2, 3]

    def test_ar_specific_lags(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test AR with specific lag indices."""
        model = rg.AR(ar1_data, lags=[1, 4])
        results = model.fit()

        assert len(results.lags) == 2
        assert results.lags == [1, 4]

    def test_ar_nobs(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test effective number of observations."""
        model = rg.AR(ar1_data, lags=5)
        results = model.fit()

        # Should lose 5 observations to lags
        assert results.nobs == len(ar1_data) - 5


class TestARCovarianceTypes:
    """Test AR with different covariance estimators."""

    def test_ar_nonrobust(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test nonrobust standard errors."""
        model = rg.AR(ar1_data, lags=1)
        results = model.fit(cov_type="nonrobust")

        assert results.cov_type == "nonrobust"
        assert all(results.bse > 0)

    def test_ar_hac(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test HAC standard errors."""
        model = rg.AR(ar1_data, lags=1)
        results = model.fit(cov_type="HAC")

        assert results.cov_type == "HAC"
        assert all(results.bse > 0)


class TestARWithBreaks:
    """Test AR model with structural breaks."""

    def test_ar_with_break(
        self,
        ar1_data_with_break: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test AR(1) with known structural break."""
        y, break_point = ar1_data_with_break

        model = rg.AR(y, lags=1, breaks=[break_point])
        results = model.fit()

        # Should have parameters for both regimes
        # 2 regimes * (1 constant + 1 AR coef) = 4 params
        assert len(results.params) == 4

    def test_ar_break_estimates(
        self,
        ar1_data_with_break: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test that AR coefficients differ across regimes."""
        y, break_point = ar1_data_with_break

        model = rg.AR(y, lags=1, breaks=[break_point])
        results = model.fit()

        # Extract AR coefficients for each regime from param names
        param_names = results.param_names or []
        ar_regime1_idx = [i for i, n in enumerate(param_names) if "y.L1_regime1" in n]
        ar_regime2_idx = [i for i, n in enumerate(param_names) if "y.L1_regime2" in n]

        if ar_regime1_idx and ar_regime2_idx:
            ar_coef_0 = results.params[ar_regime1_idx[0]]
            ar_coef_1 = results.params[ar_regime2_idx[0]]

            # Regime 0: phi ≈ 0.3, Regime 1: phi ≈ 0.8
            assert np.isclose(ar_coef_0, 0.3, atol=0.2)
            assert np.isclose(ar_coef_1, 0.8, atol=0.2)


class TestARResults:
    """Test AR results object."""

    def test_summary(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test summary generation."""
        model = rg.AR(ar1_data, lags=1)
        results = model.fit()

        summary = results.summary()
        assert "AR Model Results" in summary
        assert "AR(1)" in summary
        assert "y.L1" in summary

    def test_roots(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test AR polynomial roots."""
        model = rg.AR(ar1_data, lags=1)
        results = model.fit()

        roots = results.roots
        assert len(roots) == 1

        # For stationary AR(1), root should be outside unit circle
        assert np.abs(roots[0]) > 1

    def test_information_criteria(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test AIC and BIC calculation."""
        model = rg.AR(ar1_data, lags=1)
        results = model.fit()

        assert not np.isnan(results.aic)
        assert not np.isnan(results.bic)
        # BIC penalizes more than AIC for large samples
        assert results.bic > results.aic

    def test_sigma_properties(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test residual standard error and variance."""
        model = rg.AR(ar1_data, lags=1)
        results = model.fit()

        # sigma_squared should be positive
        assert results.sigma_squared > 0
        # sigma should be sqrt of sigma_squared
        assert np.isclose(results.sigma, np.sqrt(results.sigma_squared))
        # Summary should include these values
        summary = results.summary(diagnostics=False)
        assert "Residual Std Err" in summary
        assert "Residual Variance" in summary


class TestARTrend:
    """Test AR with different trend specifications."""

    def test_ar_no_constant(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test AR without constant."""
        model = rg.AR(ar1_data, lags=1, trend="n")
        results = model.fit()

        # Only AR coefficient, no constant
        assert len(results.params) == 1

    def test_ar_with_trend(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test AR with constant and trend."""
        model = rg.AR(ar1_data, lags=1, trend="ct")
        results = model.fit()

        # constant + trend + AR coefficient
        assert len(results.params) == 3
        param_names = results.param_names or []
        assert any("trend" in n for n in param_names)


class TestARVariableBreaks:
    """Test AR model with variable-specific breaks."""

    def test_variable_breaks_validation_error(
        self, ar1_data: NDArray[np.floating[Any]]
    ) -> None:
        """Test that breaks and variable_breaks cannot both be specified."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            rg.AR(ar1_data, lags=1, breaks=[100], variable_breaks={"const": [50]})

    def test_variable_breaks_invalid_name(
        self, ar1_data: NDArray[np.floating[Any]]
    ) -> None:
        """Test error on invalid variable name."""
        model = rg.AR(ar1_data, lags=1, variable_breaks={"invalid_var": [100]})
        with pytest.raises(ValueError, match="not found"):
            model.fit()

    def test_variable_breaks_invalid_index(
        self, ar1_data: NDArray[np.floating[Any]]
    ) -> None:
        """Test error on out-of-bounds variable index."""
        model = rg.AR(ar1_data, lags=1, variable_breaks={99: [100]})
        with pytest.raises(ValueError, match="out of bounds"):
            model.fit()

    def test_variable_breaks_invalid_break_point(
        self, ar1_data: NDArray[np.floating[Any]]
    ) -> None:
        """Test error on invalid break point."""
        model = rg.AR(ar1_data, lags=1, variable_breaks={"const": [0]})
        with pytest.raises(ValueError, match="out of bounds"):
            model.fit()

    def test_variable_breaks_const_only(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test variable breaks on constant only (intercept shift)."""
        y, break_point = ar1_data_with_break

        model = rg.AR(y, lags=1, variable_breaks={"const": [break_point]})
        results = model.fit()

        # Should have: const_regime1, const_regime2, y.L1 (no regime suffix)
        param_names = results.param_names or []
        assert "const_regime1" in param_names
        assert "const_regime2" in param_names
        assert "y.L1" in param_names
        # No regime suffix on y.L1
        assert "y.L1_regime1" not in param_names
        assert "y.L1_regime2" not in param_names

    def test_variable_breaks_ar_only(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test variable breaks on AR coefficient only."""
        y, break_point = ar1_data_with_break

        model = rg.AR(y, lags=1, variable_breaks={"y.L1": [break_point]})
        results = model.fit()

        # Should have: const (no suffix), y.L1_regime1, y.L1_regime2
        param_names = results.param_names or []
        assert "const" in param_names
        assert "y.L1_regime1" in param_names
        assert "y.L1_regime2" in param_names
        # No regime suffix on const
        assert "const_regime1" not in param_names

    def test_variable_breaks_estimation(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test that variable breaks produce reasonable estimates."""
        y, break_point = ar1_data_with_break

        # True model: phi=0.3 for t < 100, phi=0.8 for t >= 100
        model = rg.AR(y, lags=1, variable_breaks={"y.L1": [break_point]})
        results = model.fit()

        param_names = results.param_names or []
        ar_regime1_idx = param_names.index("y.L1_regime1")
        ar_regime2_idx = param_names.index("y.L1_regime2")

        ar_coef_0 = results.params[ar_regime1_idx]
        ar_coef_1 = results.params[ar_regime2_idx]

        # Should be close to true values
        assert np.isclose(ar_coef_0, 0.3, atol=0.2)
        assert np.isclose(ar_coef_1, 0.8, atol=0.2)

    def test_variable_breaks_index_key(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test variable breaks using integer index as key."""
        y, break_point = ar1_data_with_break

        # Index 0 is const, index 1 is y.L1 in AR(1) with constant
        model = rg.AR(y, lags=1, variable_breaks={1: [break_point]})
        results = model.fit()

        param_names = results.param_names or []
        assert "const" in param_names
        assert "y.L1_regime1" in param_names
        assert "y.L1_regime2" in param_names


class TestARFitByRegime:
    """Test AR.fit_by_regime() method."""

    def test_fit_by_regime_no_breaks(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test fit_by_regime with no breaks returns single result."""
        model = rg.AR(ar1_data, lags=1)
        results = model.fit_by_regime()

        assert len(results) == 1
        assert results[0].nobs == len(ar1_data) - 1  # Lose 1 obs to lag

    def test_fit_by_regime_with_breaks(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test fit_by_regime with break returns two results."""
        y, break_point = ar1_data_with_break

        model = rg.AR(y, lags=1, breaks=[break_point])
        results = model.fit_by_regime()

        assert len(results) == 2

        # Each regime loses 1 observation to lag initialization
        # Regime 0: obs 0-99 (100 obs) -> 99 effective
        # Regime 1: obs 100-199 (100 obs) -> 99 effective
        assert results[0].nobs == break_point - 1
        assert results[1].nobs == len(y) - break_point - 1

    def test_fit_by_regime_estimates(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test that fit_by_regime produces correct estimates per regime."""
        y, break_point = ar1_data_with_break

        model = rg.AR(y, lags=1, breaks=[break_point])
        results = model.fit_by_regime()

        # True values: phi=0.3 for regime 0, phi=0.8 for regime 1
        ar_coef_0 = results[0].ar_params[0] if results[0].ar_params is not None else 0
        ar_coef_1 = results[1].ar_params[0] if results[1].ar_params is not None else 0

        assert np.isclose(ar_coef_0, 0.3, atol=0.2)
        assert np.isclose(ar_coef_1, 0.8, atol=0.2)

    def test_fit_by_regime_cov_type(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test fit_by_regime with HAC covariance."""
        y, break_point = ar1_data_with_break

        model = rg.AR(y, lags=1, breaks=[break_point])
        results = model.fit_by_regime(cov_type="HAC")

        assert results[0].cov_type == "HAC"
        assert results[1].cov_type == "HAC"


class TestARSummaryBreaks:
    """Test AR summary output with breaks."""

    def test_summary_common_breaks(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test that summary includes break information for common breaks."""
        y, break_point = ar1_data_with_break

        model = rg.AR(y, lags=1, breaks=[break_point])
        results = model.fit()

        summary = results.summary(diagnostics=False)

        assert "Structural Breaks" in summary
        assert f"Break at {break_point}" in summary
        assert "Regime 1" in summary
        assert "Regime 2" in summary

    def test_summary_variable_breaks(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test that summary includes break information for variable breaks."""
        y, break_point = ar1_data_with_break

        model = rg.AR(y, lags=1, variable_breaks={"const": [break_point]})
        results = model.fit()

        summary = results.summary(diagnostics=False)

        assert "Variable-Specific Structural Breaks" in summary
        assert "const:" in summary
        assert f"break at {break_point}" in summary

    def test_ar_summary_by_regime(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test ar_summary_by_regime function."""
        y, break_point = ar1_data_with_break

        model = rg.AR(y, lags=1, breaks=[break_point])
        regime_results = model.fit_by_regime()

        summary = rg.ar_summary_by_regime(
            regime_results, breaks=[break_point], nobs_total=len(y)
        )

        assert "AR Model Results by Regime" in summary
        assert f"Breaks at: {break_point}" in summary
        assert "Regime 1" in summary
        assert "Regime 2" in summary
        assert "y.L1" in summary
        # Both regimes should have R-squared shown
        assert summary.count("R-squared:") >= 2

    def test_ar_summary_by_regime_no_breaks_provided(
        self, ar1_data: NDArray[np.floating[Any]]
    ) -> None:
        """Test ar_summary_by_regime without breaks parameter."""
        model = rg.AR(ar1_data, lags=1)
        results = model.fit_by_regime()

        summary = rg.ar_summary_by_regime(results)

        assert "AR Model Results by Regime" in summary
        assert "Regime 1" in summary
