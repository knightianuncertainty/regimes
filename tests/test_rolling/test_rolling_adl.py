"""Tests for rolling and recursive ADL estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import regimes as rg

# =============================================================================
# Basic RollingADL Tests
# =============================================================================


class TestRollingADLBasic:
    """Basic RollingADL tests."""

    def test_rolling_adl_creation(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test RollingADL can be created."""
        y, x = adl_data
        rolling = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40)

        # After lag adjustment, nobs is effective sample size
        assert rolling.nobs == len(y) - 1  # maxlag = max(1, 1) = 1
        assert rolling.window == 40
        assert rolling.is_rolling

    def test_rolling_adl_from_model(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test RollingADL.from_model() factory method."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)
        rolling = rg.RollingADL.from_model(model, window=40)

        assert rolling.window == 40
        assert rolling.lags == [1]

    def test_rolling_adl_model_integration(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test model.rolling() integration method."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)
        rolling = model.rolling(window=40)

        assert isinstance(rolling, rg.RollingADL)
        assert rolling.window == 40

    def test_rolling_adl_fit(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test RollingADL.fit() produces results."""
        y, x = adl_data
        rolling = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40)
        results = rolling.fit()

        assert isinstance(results, rg.RollingADLResults)
        assert results.window == 40
        assert results.is_rolling
        assert results.lags == [1]

    def test_rolling_adl_nan_padding(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test that early observations are NaN-padded."""
        y, x = adl_data
        window = 40
        rolling = rg.RollingADL(y, x, lags=1, exog_lags=1, window=window)
        results = rolling.fit()

        # First window-1 observations should be NaN
        assert np.all(np.isnan(results.params[: window - 1]))

        # From window-1 onwards should have values
        assert not np.any(np.isnan(results.params[window - 1 :]))

    def test_rolling_adl_n_valid(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test n_valid count is correct."""
        y, x = adl_data
        window = 40
        rolling = rg.RollingADL(y, x, lags=1, exog_lags=1, window=window)
        results = rolling.fit()

        # Effective sample = n - maxlag
        n_eff = len(y) - 1
        expected_valid = n_eff - window + 1
        assert results.n_valid == expected_valid


# =============================================================================
# Basic RecursiveADL Tests
# =============================================================================


class TestRecursiveADLBasic:
    """Basic RecursiveADL tests."""

    def test_recursive_adl_creation(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test RecursiveADL can be created."""
        y, x = adl_data
        recursive = rg.RecursiveADL(y, x, lags=1, exog_lags=1, min_nobs=30)

        assert recursive.min_nobs == 30
        assert not recursive.is_rolling

    def test_recursive_adl_from_model(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test RecursiveADL.from_model() factory method."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)
        recursive = rg.RecursiveADL.from_model(model, min_nobs=30)

        assert recursive.min_nobs == 30

    def test_recursive_adl_model_integration(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test model.recursive() integration method."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)
        recursive = model.recursive(min_nobs=30)

        assert isinstance(recursive, rg.RecursiveADL)
        assert recursive.min_nobs == 30

    def test_recursive_adl_fit(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test RecursiveADL.fit() produces results."""
        y, x = adl_data
        recursive = rg.RecursiveADL(y, x, lags=1, exog_lags=1, min_nobs=30)
        results = recursive.fit()

        assert isinstance(results, rg.RecursiveADLResults)
        assert results.window is None
        assert not results.is_rolling

    def test_recursive_adl_nan_padding(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test that early observations are NaN-padded."""
        y, x = adl_data
        min_nobs = 30
        recursive = rg.RecursiveADL(y, x, lags=1, exog_lags=1, min_nobs=min_nobs)
        results = recursive.fit()

        # First min_nobs-1 observations should be NaN
        assert np.all(np.isnan(results.params[: min_nobs - 1]))


# =============================================================================
# Window Size Variations
# =============================================================================


class TestRollingADLWindowSizes:
    """Test various window sizes."""

    @pytest.mark.parametrize("window", [30, 40, 50, 60])
    def test_different_window_sizes(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
        window: int,
    ) -> None:
        """Test RollingADL with various window sizes."""
        y, x = adl_data
        rolling = rg.RollingADL(y, x, lags=1, exog_lags=1, window=window)
        results = rolling.fit()

        assert results.window == window
        n_eff = len(y) - 1
        expected_valid = n_eff - window + 1
        assert results.n_valid == expected_valid

    def test_minimum_window_size(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test minimum valid window size."""
        y, x = adl_data
        # ADL(1,1) with constant: 4 parameters -> window must be >= 5
        rolling = rg.RollingADL(y, x, lags=1, exog_lags=1, window=10)
        results = rolling.fit()

        assert results.n_valid > 0

    def test_window_too_small_raises(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test error when window is too small."""
        y, x = adl_data
        # ADL(1,1) with constant has 4 params (const, y.L1, x0, x0.L1)
        # Window must be >= n_params + 1
        with pytest.raises(ValueError, match="window.*must be at least"):
            rg.RollingADL(y, x, lags=1, exog_lags=1, window=3)


# =============================================================================
# Recursive Min_nobs Variations
# =============================================================================


class TestRecursiveADLMinNobs:
    """Test various min_nobs settings."""

    @pytest.mark.parametrize("min_nobs", [20, 30, 40, 50])
    def test_different_min_nobs(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
        min_nobs: int,
    ) -> None:
        """Test RecursiveADL with various min_nobs values."""
        y, x = adl_data
        recursive = rg.RecursiveADL(y, x, lags=1, exog_lags=1, min_nobs=min_nobs)
        results = recursive.fit()

        assert results.min_nobs == min_nobs
        n_eff = len(y) - 1
        expected_valid = n_eff - min_nobs + 1
        assert results.n_valid == expected_valid

    def test_default_min_nobs(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test RecursiveADL with default min_nobs."""
        y, x = adl_data
        recursive = rg.RecursiveADL(y, x, lags=1, exog_lags=1)
        results = recursive.fit()

        # Default min_nobs should be n_params + 1
        assert results.min_nobs == recursive.n_params + 1


# =============================================================================
# Results Shape Tests
# =============================================================================


class TestRollingADLResultsShape:
    """Test result array shapes."""

    def test_params_shape(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test params array shape."""
        y, x = adl_data
        rolling = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40)
        results = rolling.fit()

        n_eff = len(y) - 1
        # ADL(1,1) with constant: const, y.L1, x0, x0.L1 = 4 params
        assert results.params.shape == (n_eff, 4)

    def test_bse_shape(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test bse array shape."""
        y, x = adl_data
        rolling = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40)
        results = rolling.fit()

        n_eff = len(y) - 1
        assert results.bse.shape == (n_eff, 4)

    def test_ssr_shape(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test ssr array shape."""
        y, x = adl_data
        rolling = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40)
        results = rolling.fit()

        n_eff = len(y) - 1
        assert results.ssr is not None
        assert results.ssr.shape == (n_eff,)

    def test_rsquared_shape(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test rsquared array shape."""
        y, x = adl_data
        rolling = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40)
        results = rolling.fit()

        n_eff = len(y) - 1
        assert results.rsquared is not None
        assert results.rsquared.shape == (n_eff,)


# =============================================================================
# Edge Cases
# =============================================================================


class TestRollingADLEdgeCases:
    """Test edge cases for RollingADL."""

    def test_short_data(self, rng: np.random.Generator) -> None:
        """Test with minimal data length."""
        n = 60
        x = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + 0.3 * x[t] + rng.standard_normal()

        rolling = rg.RollingADL(y, x, lags=1, exog_lags=0, window=20)
        results = rolling.fit()

        assert results.n_valid > 0

    def test_single_exog_var(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test with single exogenous variable."""
        y, x = adl_data
        # x is already 1D
        rolling = rg.RollingADL(y, x, lags=1, exog_lags=0, window=40)
        results = rolling.fit()

        # const, y.L1, x0 = 3 params
        assert results.n_params == 3
        assert results.n_valid > 0

    def test_multiple_exog_vars(self, rng: np.random.Generator) -> None:
        """Test with multiple exogenous variables."""
        n = 200
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        X = np.column_stack([x1, x2])
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + 0.3 * x1[t] + 0.2 * x2[t] + rng.standard_normal()

        rolling = rg.RollingADL(y, X, lags=1, exog_lags=0, window=40)
        results = rolling.fit()

        # const, y.L1, x0, x1 = 4 params
        assert results.n_params == 4
        assert results.n_valid > 0

    def test_dict_style_exog_lags(self, rng: np.random.Generator) -> None:
        """Test with dict-style exog_lags specification."""
        n = 200
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        X = np.column_stack([x1, x2])
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = (
                0.5 * y[t - 1]
                + 0.3 * x1[t]
                + 0.15 * x1[t - 1]
                + 0.2 * x2[t]
                + rng.standard_normal()
            )

        # Dict-style: x0 has lags 0,1; x1 has only lag 0
        rolling = rg.RollingADL(y, X, lags=1, exog_lags={0: 1, 1: 0}, window=40)
        results = rolling.fit()

        # const, y.L1, x0, x0.L1, x1 = 5 params
        assert results.n_params == 5
        assert results.n_valid > 0

    def test_no_ar_lags(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test ADL(0,q) - distributed lag model with no AR terms."""
        y, x = adl_data
        rolling = rg.RollingADL(y, x, lags=0, exog_lags=2, window=40)
        results = rolling.fit()

        # const, x0, x0.L1, x0.L2 = 4 params
        assert results.lags == []
        assert results.n_valid > 0

    def test_high_ar_order(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test with higher AR order."""
        y, x = adl_data
        rolling = rg.RollingADL(y, x, lags=3, exog_lags=0, window=50)
        results = rolling.fit()

        assert results.lags == [1, 2, 3]
        assert results.n_valid > 0


# =============================================================================
# Results Properties
# =============================================================================


class TestRollingADLResultsProperties:
    """Test RollingADL results properties."""

    def test_ar_params_extraction(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test ar_params property extracts AR coefficients."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=2, exog_lags=1, window=50).fit()

        ar_params = results.ar_params
        # Should have shape (nobs, 2) for AR(2)
        assert ar_params.shape[1] == 2

    def test_tvalues(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test t-values computation."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40).fit()

        # Manual computation
        expected_t = results.params / results.bse
        np.testing.assert_array_almost_equal(results.tvalues, expected_t)

    def test_conf_int(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test confidence interval computation."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40).fit()
        ci = results.conf_int(alpha=0.05)

        n_eff = len(y) - 1
        # Check shape (nobs, n_params, 2)
        assert ci.shape == (n_eff, 4, 2)

        # Lower should be less than upper where valid
        valid_idx = ~np.isnan(results.params[:, 0])
        assert np.all(ci[valid_idx, :, 0] < ci[valid_idx, :, 1])

    def test_to_dataframe(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test DataFrame conversion."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40).fit()
        df = results.to_dataframe()

        n_eff = len(y) - 1
        assert len(df) == n_eff
        # Should have columns for all params
        assert "const" in df.columns
        assert "y.L1" in df.columns

    def test_param_names(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test param_names property."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40).fit()

        expected_names = ["const", "y.L1", "x0", "x0.L1"]
        assert results.param_names == expected_names

    def test_exog_lags_in_results(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test exog_lags stored in results."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=2, window=40).fit()

        assert "x0" in results.exog_lags
        assert results.exog_lags["x0"] == [0, 1, 2]


# =============================================================================
# Summary Tests
# =============================================================================


class TestRollingADLSummary:
    """Test summary generation."""

    def test_summary_rolling(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test rolling summary generation."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40).fit()
        summary = results.summary()

        assert "Rolling ADL(1,1)" in summary
        assert "Window Size" in summary
        assert "Valid Estimates" in summary
        assert "y.L1" in summary
        assert "x0" in summary

    def test_summary_recursive(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test recursive summary generation."""
        y, x = adl_data
        results = rg.RecursiveADL(y, x, lags=1, exog_lags=1, min_nobs=30).fit()
        summary = results.summary()

        assert "Recursive ADL(1,1)" in summary
        assert "Min Observations" in summary
        assert "Valid Estimates" in summary


# =============================================================================
# Covariance Type Tests
# =============================================================================


class TestRollingADLCovTypes:
    """Test different covariance types."""

    def test_rolling_adl_nonrobust(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test nonrobust covariance."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40).fit(
            cov_type="nonrobust"
        )
        assert results.cov_type == "nonrobust"

    def test_rolling_adl_hc0(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test HC0 covariance."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40).fit(
            cov_type="HC0"
        )
        assert results.cov_type == "HC0"

    def test_recursive_adl_nonrobust(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test nonrobust covariance for recursive."""
        y, x = adl_data
        results = rg.RecursiveADL(y, x, lags=1, exog_lags=1, min_nobs=30).fit(
            cov_type="nonrobust"
        )
        assert results.cov_type == "nonrobust"

    def test_recursive_adl_hc0(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test HC0 covariance for recursive."""
        y, x = adl_data
        results = rg.RecursiveADL(y, x, lags=1, exog_lags=1, min_nobs=30).fit(
            cov_type="HC0"
        )
        assert results.cov_type == "HC0"


# =============================================================================
# Trend Tests
# =============================================================================


class TestRollingADLTrend:
    """Test different trend options."""

    def test_rolling_adl_constant(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test ADL with constant (default)."""
        y, x = adl_data
        rolling = rg.RollingADL(y, x, lags=1, exog_lags=0, window=40, trend="c")
        results = rolling.fit()

        assert "const" in results.param_names

    def test_rolling_adl_no_trend(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test ADL without deterministic terms."""
        y, x = adl_data
        rolling = rg.RollingADL(y, x, lags=1, exog_lags=0, window=40, trend="n")
        results = rolling.fit()

        assert "const" not in results.param_names
        # y.L1, x0 = 2 params
        assert results.n_params == 2

    def test_rolling_adl_constant_and_trend(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test ADL with constant and trend."""
        y, x = adl_data
        rolling = rg.RollingADL(y, x, lags=1, exog_lags=0, window=40, trend="ct")
        results = rolling.fit()

        assert "const" in results.param_names
        assert "trend" in results.param_names


# =============================================================================
# Validation Tests
# =============================================================================


class TestRollingADLValidation:
    """Test input validation."""

    def test_invalid_ar_lags(self) -> None:
        """Test error for invalid AR lags."""
        y = np.random.randn(100)
        x = np.random.randn(100)
        with pytest.raises(ValueError, match="positive integers"):
            rg.RollingADL(y, x, lags=[0, 1], exog_lags=0, window=30)

    def test_invalid_trend(self) -> None:
        """Test error for invalid trend."""
        y = np.random.randn(100)
        x = np.random.randn(100)
        with pytest.raises(ValueError, match="trend must be"):
            rg.RollingADL(y, x, lags=1, exog_lags=0, window=30, trend="invalid")


# =============================================================================
# Plotting Tests
# =============================================================================


class TestRollingADLPlotting:
    """Test plotting methods."""

    def test_plot_coefficients(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test plot_coefficients method."""
        import matplotlib

        matplotlib.use("Agg")

        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40).fit()
        fig, axes = results.plot_coefficients()

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_coefficients_subset(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test plot_coefficients with variable subset."""
        import matplotlib

        matplotlib.use("Agg")

        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40).fit()
        fig, axes = results.plot_coefficients(variables=["y.L1", "x0"])

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_coefficients_custom_alpha(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test plot_coefficients with custom alpha."""
        import matplotlib

        matplotlib.use("Agg")

        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=1, window=40).fit()
        fig, axes = results.plot_coefficients(alpha=0.10)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_recursive_plot_coefficients(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test plot_coefficients for recursive estimation."""
        import matplotlib

        matplotlib.use("Agg")

        y, x = adl_data
        results = rg.RecursiveADL(y, x, lags=1, exog_lags=1, min_nobs=30).fit()
        fig, axes = results.plot_coefficients()

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


# =============================================================================
# Workflow Tests
# =============================================================================


class TestRollingADLWorkflow:
    """Test complete workflow integration."""

    def test_full_workflow(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test model -> rolling -> fit -> plot workflow."""
        import matplotlib

        matplotlib.use("Agg")

        y, x = adl_data

        # Define model
        model = rg.ADL(y, x, lags=1, exog_lags=1)

        # Rolling estimation via model method
        rolling_results = model.rolling(window=40).fit()

        # Check results
        assert rolling_results.n_valid > 0
        assert isinstance(rolling_results.params, np.ndarray)

        # Test plot method doesn't error
        fig, axes = rolling_results.plot_coefficients()
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_recursive_workflow(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test model -> recursive -> fit workflow."""
        y, x = adl_data

        # Define model
        model = rg.ADL(y, x, lags=1, exog_lags=1)

        # Recursive estimation
        recursive_results = model.recursive(min_nobs=30).fit()

        # Check expanding estimates
        assert recursive_results.n_valid > 0
        assert not recursive_results.is_rolling

    def test_adl_with_break_detection_workflow(
        self,
        adl_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test ADL with both rolling estimation and break detection."""
        y, x, break_point = adl_data_with_break

        # Rolling estimation to visualize parameter evolution
        model = rg.ADL(y, x, lags=1, exog_lags=0)
        rolling_results = model.rolling(window=50).fit()

        # Check that AR coefficient shows some variation
        ar_params = rolling_results.ar_params
        valid_ar = ar_params[~np.isnan(ar_params[:, 0]), 0]
        assert len(valid_ar) > 0
        # There should be some variation if there's a break
        assert np.std(valid_ar) > 0

    def test_compare_rolling_recursive(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test comparison of rolling and recursive estimates."""
        y, x = adl_data
        model = rg.ADL(y, x, lags=1, exog_lags=1)

        rolling_results = model.rolling(window=40).fit()
        recursive_results = model.recursive(min_nobs=40).fit()

        # Both should produce results
        assert rolling_results.n_valid > 0
        assert recursive_results.n_valid > 0

        # Final rolling and recursive estimates differ because:
        # - Rolling uses only last 40 observations
        # - Recursive uses all observations from start
        # But both should produce valid parameter estimates (not NaN)
        last_rolling = rolling_results.params[-1]
        last_recursive = recursive_results.params[-1]

        assert not np.any(np.isnan(last_rolling))
        assert not np.any(np.isnan(last_recursive))


# =============================================================================
# Specific Lag Configuration Tests
# =============================================================================


class TestRollingADLLagConfigurations:
    """Test various lag configurations."""

    def test_adl_10_single_exog(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test ADL(1,0) - AR(1) with contemporaneous exog."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=0, window=40).fit()

        assert results.lags == [1]
        assert "x0" in results.exog_lags
        assert results.exog_lags["x0"] == [0]

    def test_adl_12(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test ADL(1,2)."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=1, exog_lags=2, window=40).fit()

        assert results.lags == [1]
        assert results.exog_lags["x0"] == [0, 1, 2]
        # const, y.L1, x0, x0.L1, x0.L2 = 5 params
        assert results.n_params == 5

    def test_adl_21(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test ADL(2,1)."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=2, exog_lags=1, window=40).fit()

        assert results.lags == [1, 2]
        assert results.exog_lags["x0"] == [0, 1]
        # const, y.L1, y.L2, x0, x0.L1 = 5 params
        assert results.n_params == 5

    def test_adl_specific_ar_lags(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test ADL with specific AR lag indices."""
        y, x = adl_data
        results = rg.RollingADL(y, x, lags=[1, 4], exog_lags=0, window=50).fit()

        assert results.lags == [1, 4]
        # const, y.L1, y.L4, x0 = 4 params
        assert results.n_params == 4
