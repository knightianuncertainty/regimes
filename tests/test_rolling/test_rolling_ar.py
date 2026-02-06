"""Tests for rolling AR estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import regimes as rg


class TestRollingARBasic:
    """Basic RollingAR tests."""

    def test_rolling_ar_creation(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test RollingAR can be created."""
        y = ar1_data
        rolling = rg.RollingAR(y, lags=1, window=30)

        # After lag adjustment, nobs is effective sample size
        assert rolling.nobs == len(y) - 1  # One lag
        assert rolling.window == 30
        assert rolling.is_rolling

    def test_rolling_ar_from_model(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test RollingAR.from_model() factory method."""
        y = ar1_data
        model = rg.AR(y, lags=1)
        rolling = rg.RollingAR.from_model(model, window=30)

        assert rolling.window == 30
        assert rolling.lags == [1]

    def test_rolling_ar_model_integration(
        self, ar1_data: NDArray[np.floating[Any]]
    ) -> None:
        """Test model.rolling() integration method."""
        y = ar1_data
        model = rg.AR(y, lags=1)
        rolling = model.rolling(window=30)

        assert isinstance(rolling, rg.RollingAR)
        assert rolling.window == 30

    def test_rolling_ar_fit(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test RollingAR.fit() produces results."""
        y = ar1_data
        rolling = rg.RollingAR(y, lags=1, window=30)
        results = rolling.fit()

        assert isinstance(results, rg.RollingARResults)
        assert results.window == 30
        assert results.is_rolling
        assert results.lags == [1]

    def test_rolling_ar_nan_padding(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test that early observations are NaN-padded."""
        y = ar1_data
        window = 30
        rolling = rg.RollingAR(y, lags=1, window=window)
        results = rolling.fit()

        # First window-1 observations should be NaN
        assert np.all(np.isnan(results.params[: window - 1]))

        # From window-1 onwards should have values
        assert not np.any(np.isnan(results.params[window - 1 :]))

    def test_rolling_ar_multiple_lags(
        self, ar1_data: NDArray[np.floating[Any]]
    ) -> None:
        """Test RollingAR with multiple lags."""
        y = ar1_data
        rolling = rg.RollingAR(y, lags=3, window=40)
        results = rolling.fit()

        assert results.lags == [1, 2, 3]
        # Should have: const + 3 AR params = 4 parameters
        assert results.n_params == 4

    def test_rolling_ar_specific_lags(
        self, ar1_data: NDArray[np.floating[Any]]
    ) -> None:
        """Test RollingAR with specific lag indices."""
        y = ar1_data
        rolling = rg.RollingAR(y, lags=[1, 4], window=40)
        results = rolling.fit()

        assert results.lags == [1, 4]
        # Should have: const + 2 AR params = 3 parameters
        assert results.n_params == 3


class TestRecursiveARBasic:
    """Basic RecursiveAR tests."""

    def test_recursive_ar_creation(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test RecursiveAR can be created."""
        y = ar1_data
        recursive = rg.RecursiveAR(y, lags=1, min_nobs=20)

        assert recursive.min_nobs == 20
        assert not recursive.is_rolling

    def test_recursive_ar_from_model(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test RecursiveAR.from_model() factory method."""
        y = ar1_data
        model = rg.AR(y, lags=1)
        recursive = rg.RecursiveAR.from_model(model, min_nobs=20)

        assert recursive.min_nobs == 20

    def test_recursive_ar_model_integration(
        self, ar1_data: NDArray[np.floating[Any]]
    ) -> None:
        """Test model.recursive() integration method."""
        y = ar1_data
        model = rg.AR(y, lags=1)
        recursive = model.recursive(min_nobs=20)

        assert isinstance(recursive, rg.RecursiveAR)
        assert recursive.min_nobs == 20

    def test_recursive_ar_fit(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test RecursiveAR.fit() produces results."""
        y = ar1_data
        recursive = rg.RecursiveAR(y, lags=1, min_nobs=20)
        results = recursive.fit()

        assert isinstance(results, rg.RecursiveARResults)
        assert results.window is None
        assert not results.is_rolling

    def test_recursive_ar_nan_padding(
        self, ar1_data: NDArray[np.floating[Any]]
    ) -> None:
        """Test that early observations are NaN-padded."""
        y = ar1_data
        min_nobs = 20
        recursive = rg.RecursiveAR(y, lags=1, min_nobs=min_nobs)
        results = recursive.fit()

        # First min_nobs-1 observations should be NaN
        assert np.all(np.isnan(results.params[: min_nobs - 1]))


class TestRollingARResults:
    """Test RollingAR results properties."""

    def test_ar_params_extraction(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test ar_params property extracts AR coefficients."""
        y = ar1_data
        results = rg.RollingAR(y, lags=2, window=40).fit()

        ar_params = results.ar_params
        # Should have shape (nobs, 2) for AR(2)
        assert ar_params.shape[1] == 2

    def test_summary(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test summary generation."""
        y = ar1_data
        results = rg.RollingAR(y, lags=1, window=30).fit()
        summary = results.summary()

        assert "Rolling AR(1)" in summary
        assert "Window Size" in summary
        assert "y.L1" in summary

    def test_to_dataframe(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test DataFrame conversion."""
        y = ar1_data
        results = rg.RollingAR(y, lags=1, window=30).fit()
        df = results.to_dataframe()

        # Should have columns for const and y.L1
        assert "const" in df.columns
        assert "y.L1" in df.columns


class TestRollingARCovTypes:
    """Test different covariance types."""

    def test_rolling_ar_nonrobust(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test nonrobust covariance."""
        y = ar1_data
        results = rg.RollingAR(y, lags=1, window=30).fit(cov_type="nonrobust")
        assert results.cov_type == "nonrobust"

    def test_rolling_ar_hc0(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test HC0 covariance."""
        y = ar1_data
        results = rg.RollingAR(y, lags=1, window=30).fit(cov_type="HC0")
        assert results.cov_type == "HC0"


class TestRollingARTrend:
    """Test different trend options."""

    def test_rolling_ar_constant(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test AR with constant."""
        y = ar1_data
        rolling = rg.RollingAR(y, lags=1, window=30, trend="c")
        results = rolling.fit()

        assert "const" in results.param_names

    def test_rolling_ar_no_trend(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test AR without deterministic terms."""
        y = ar1_data
        rolling = rg.RollingAR(y, lags=1, window=30, trend="n")
        results = rolling.fit()

        assert "const" not in results.param_names
        # Should only have AR param
        assert results.n_params == 1


class TestRollingARValidation:
    """Test input validation."""

    def test_invalid_lags(self) -> None:
        """Test error for invalid lags."""
        y = np.random.randn(100)
        with pytest.raises(ValueError, match="positive integers"):
            rg.RollingAR(y, lags=[0, 1], window=30)

    def test_empty_lags(self) -> None:
        """Test error for empty lags."""
        y = np.random.randn(100)
        with pytest.raises(ValueError, match="At least one lag"):
            rg.RollingAR(y, lags=[], window=30)

    def test_invalid_trend(self) -> None:
        """Test error for invalid trend."""
        y = np.random.randn(100)
        with pytest.raises(ValueError, match="trend must be"):
            rg.RollingAR(y, lags=1, window=30, trend="invalid")


class TestRollingARWorkflow:
    """Test complete workflow integration."""

    def test_full_workflow(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test model -> rolling -> fit -> plot workflow."""
        y = ar1_data

        # Define model
        model = rg.AR(y, lags=1)

        # Rolling estimation via model method
        rolling_results = model.rolling(window=30).fit()

        # Check results
        assert rolling_results.n_valid > 0

        # Test plot method doesn't error
        import matplotlib

        matplotlib.use("Agg")
        fig, axes = rolling_results.plot_coefficients()
        assert fig is not None

    def test_recursive_workflow(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Test model -> recursive -> fit workflow."""
        y = ar1_data

        # Define model
        model = rg.AR(y, lags=1)

        # Recursive estimation
        recursive_results = model.recursive(min_nobs=20).fit()

        # Check expanding estimates
        assert recursive_results.n_valid > 0
        assert not recursive_results.is_rolling

    def test_ar_with_break_detection_workflow(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test AR with both rolling estimation and break detection."""
        y, break_point = ar1_data_with_break

        # Rolling estimation to visualize parameter evolution
        model = rg.AR(y, lags=1)
        rolling_results = model.rolling(window=40).fit()

        # Check that AR coefficient shows some variation
        valid_ar = rolling_results.ar_params[
            ~np.isnan(rolling_results.ar_params[:, 0]), 0
        ]
        assert len(valid_ar) > 0
        # There should be some variation if there's a break
        assert np.std(valid_ar) > 0
