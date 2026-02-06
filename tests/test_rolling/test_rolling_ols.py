"""Tests for rolling OLS estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import regimes as rg


class TestRollingOLSBasic:
    """Basic RollingOLS tests."""

    def test_rolling_ols_creation(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test RollingOLS can be created."""
        y, X = regression_data
        rolling = rg.RollingOLS(y, X, window=30)

        assert rolling.nobs == len(y)
        assert rolling.n_params == X.shape[1]
        assert rolling.window == 30
        assert rolling.is_rolling

    def test_rolling_ols_from_model(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test RollingOLS.from_model() factory method."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        rolling = rg.RollingOLS.from_model(model, window=30)

        assert rolling.nobs == model.nobs
        assert rolling.n_params == model.k_exog

    def test_rolling_ols_model_integration(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test model.rolling() integration method."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        rolling = model.rolling(window=30)

        assert isinstance(rolling, rg.RollingOLS)
        assert rolling.window == 30

    def test_rolling_ols_fit(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test RollingOLS.fit() produces results."""
        y, X = regression_data
        rolling = rg.RollingOLS(y, X, window=30)
        results = rolling.fit()

        assert isinstance(results, rg.RollingOLSResults)
        assert results.nobs == len(y)
        assert results.params.shape == (len(y), X.shape[1])
        assert results.bse.shape == (len(y), X.shape[1])
        assert results.window == 30
        assert results.is_rolling

    def test_rolling_ols_nan_padding(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test that early observations are NaN-padded."""
        y, X = regression_data
        window = 30
        rolling = rg.RollingOLS(y, X, window=window)
        results = rolling.fit()

        # First window-1 observations should be NaN
        assert np.all(np.isnan(results.params[: window - 1]))
        assert np.all(np.isnan(results.bse[: window - 1]))

        # From window-1 onwards should have values
        assert not np.any(np.isnan(results.params[window - 1 :]))
        assert not np.any(np.isnan(results.bse[window - 1 :]))

    def test_rolling_ols_n_valid(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test n_valid count is correct."""
        y, X = regression_data
        window = 30
        rolling = rg.RollingOLS(y, X, window=window)
        results = rolling.fit()

        expected_valid = len(y) - window + 1
        assert results.n_valid == expected_valid


class TestRecursiveOLSBasic:
    """Basic RecursiveOLS tests."""

    def test_recursive_ols_creation(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test RecursiveOLS can be created."""
        y, X = regression_data
        recursive = rg.RecursiveOLS(y, X, min_nobs=20)

        assert recursive.nobs == len(y)
        assert recursive.n_params == X.shape[1]
        assert recursive.min_nobs == 20
        assert not recursive.is_rolling

    def test_recursive_ols_from_model(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test RecursiveOLS.from_model() factory method."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        recursive = rg.RecursiveOLS.from_model(model, min_nobs=20)

        assert recursive.nobs == model.nobs
        assert recursive.min_nobs == 20

    def test_recursive_ols_model_integration(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test model.recursive() integration method."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        recursive = model.recursive(min_nobs=20)

        assert isinstance(recursive, rg.RecursiveOLS)
        assert recursive.min_nobs == 20

    def test_recursive_ols_fit(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test RecursiveOLS.fit() produces results."""
        y, X = regression_data
        recursive = rg.RecursiveOLS(y, X, min_nobs=20)
        results = recursive.fit()

        assert isinstance(results, rg.RecursiveOLSResults)
        assert results.nobs == len(y)
        assert results.window is None
        assert not results.is_rolling

    def test_recursive_ols_nan_padding(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test that early observations are NaN-padded."""
        y, X = regression_data
        min_nobs = 20
        recursive = rg.RecursiveOLS(y, X, min_nobs=min_nobs)
        results = recursive.fit()

        # First min_nobs-1 observations should be NaN
        assert np.all(np.isnan(results.params[: min_nobs - 1]))

        # From min_nobs-1 onwards should have values
        assert not np.any(np.isnan(results.params[min_nobs - 1 :]))


class TestRollingOLSComparison:
    """Test RollingOLS against statsmodels RollingOLS."""

    def test_rolling_ols_matches_statsmodels(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Compare RollingOLS estimates to statsmodels."""
        from statsmodels.regression.rolling import RollingOLS as SM_RollingOLS

        y, X = regression_data
        window = 30

        # Our implementation
        our_results = rg.RollingOLS(y, X, window=window).fit()

        # statsmodels implementation
        sm_rolling = SM_RollingOLS(y, X, window=window)
        sm_results = sm_rolling.fit()

        # Compare parameters (both should have NaN at start)
        np.testing.assert_array_almost_equal(
            our_results.params, sm_results.params, decimal=10
        )

        # Compare standard errors
        np.testing.assert_array_almost_equal(
            our_results.bse, sm_results.bse, decimal=10
        )


class TestRollingOLSCovTypes:
    """Test different covariance types."""

    def test_rolling_ols_nonrobust(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test nonrobust covariance."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit(cov_type="nonrobust")
        assert results.cov_type == "nonrobust"

    def test_rolling_ols_hc0(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test HC0 covariance."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit(cov_type="HC0")
        assert results.cov_type == "HC0"


class TestRollingOLSResults:
    """Test RollingOLS results properties."""

    def test_tvalues(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test t-values computation."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()

        # Manual computation
        expected_t = results.params / results.bse
        np.testing.assert_array_almost_equal(results.tvalues, expected_t)

    def test_conf_int(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test confidence interval computation."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()
        ci = results.conf_int(alpha=0.05)

        # Check shape
        assert ci.shape == (len(y), X.shape[1], 2)

        # Lower should be less than upper where valid
        valid_idx = ~np.isnan(results.params[:, 0])
        assert np.all(ci[valid_idx, :, 0] < ci[valid_idx, :, 1])

    def test_to_dataframe(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test DataFrame conversion."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()
        df = results.to_dataframe()

        assert len(df) == len(y)
        assert len(df.columns) == X.shape[1]

    def test_summary(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test summary generation."""
        y, X = regression_data
        results = rg.RollingOLS(y, X, window=30).fit()
        summary = results.summary()

        assert "Rolling OLS" in summary
        assert "Window Size" in summary
        assert "Valid Estimates" in summary


class TestRollingOLSValidation:
    """Test input validation."""

    def test_window_too_small(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test error when window is smaller than n_params + 1."""
        y, X = regression_data
        with pytest.raises(ValueError, match="window.*must be at least"):
            rg.RollingOLS(y, X, window=2)  # X has 2 cols, need at least 3

    def test_mismatched_lengths(self) -> None:
        """Test error when endog and exog have different lengths."""
        y = np.random.randn(100)
        X = np.random.randn(50, 2)
        with pytest.raises(ValueError, match="same length"):
            rg.RollingOLS(y, X, window=30)


class TestRollingOLSWorkflow:
    """Test complete workflow integration."""

    def test_full_workflow(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test model -> rolling -> fit -> plot workflow."""
        y, X = regression_data

        # Define model
        model = rg.OLS(y, X, has_constant=False)

        # Rolling estimation via model method
        rolling_results = model.rolling(window=30).fit()

        # Check results
        assert rolling_results.n_valid > 0
        assert isinstance(rolling_results.params, np.ndarray)

        # Test plot method doesn't error (don't actually display)
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        fig, axes = rolling_results.plot_coefficients()
        assert fig is not None

    def test_recursive_workflow(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test model -> recursive -> fit workflow."""
        y, X = regression_data

        # Define model
        model = rg.OLS(y, X, has_constant=False)

        # Recursive estimation
        recursive_results = model.recursive(min_nobs=20).fit()

        # Check expanding estimates
        assert recursive_results.n_valid > 0
        assert not recursive_results.is_rolling
