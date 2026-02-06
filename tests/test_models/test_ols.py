"""Tests for OLS model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import regimes as rg


class TestOLSBasic:
    """Basic OLS tests without breaks."""

    def test_ols_constant_only(self, simple_data: NDArray[np.floating[Any]]) -> None:
        """Test OLS with constant only (mean estimation)."""
        model = rg.OLS(simple_data)
        results = model.fit()

        # Should estimate the mean
        assert len(results.params) == 1
        assert np.isclose(results.params[0], np.mean(simple_data), atol=0.01)

    def test_ols_simple_regression(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test OLS on simple regression."""
        y, X = regression_data

        model = rg.OLS(y, X, has_constant=False)  # X already has constant
        results = model.fit()

        # Check coefficient estimates (true: [1, 2])
        assert len(results.params) == 2
        assert np.isclose(results.params[0], 1.0, atol=0.3)
        assert np.isclose(results.params[1], 2.0, atol=0.3)

    def test_ols_rsquared(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test R-squared calculation."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        # R-squared should be between 0 and 1
        assert 0 <= results.rsquared <= 1
        assert 0 <= results.rsquared_adj <= 1

    def test_ols_residuals(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test residuals sum to approximately zero."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        # Residuals should sum to approximately zero
        assert np.isclose(np.sum(results.resid), 0, atol=1e-10)

        # Fitted values + residuals = y
        np.testing.assert_array_almost_equal(results.fittedvalues + results.resid, y)


class TestOLSCovarianceTypes:
    """Test different covariance estimators."""

    def test_ols_nonrobust(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test nonrobust standard errors."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        results = model.fit(cov_type="nonrobust")

        assert results.cov_type == "nonrobust"
        assert len(results.bse) == 2
        assert all(results.bse > 0)

    def test_ols_hc0(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test HC0 heteroskedasticity-robust standard errors."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        results = model.fit(cov_type="HC0")

        assert results.cov_type == "HC0"
        assert len(results.bse) == 2
        assert all(results.bse > 0)

    def test_ols_hac(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test HAC (Newey-West) standard errors."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        results = model.fit(cov_type="HAC")

        assert results.cov_type == "HAC"
        assert len(results.bse) == 2
        assert all(results.bse > 0)


class TestOLSWithBreaks:
    """Test OLS with structural breaks."""

    def test_ols_with_break(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test OLS with known structural break."""
        y, X, break_point = regression_data_with_break

        model = rg.OLS(y, X, breaks=[break_point], has_constant=False)
        results = model.fit()

        # Should have 4 parameters (2 per regime)
        assert len(results.params) == 4

    def test_ols_fit_by_regime(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test fitting separate models by regime."""
        y, X, break_point = regression_data_with_break

        model = rg.OLS(y, X, breaks=[break_point], has_constant=False)
        regime_results = model.fit_by_regime()

        assert len(regime_results) == 2

        # First regime should have slope near 0.5
        assert np.isclose(regime_results[0].params[1], 0.5, atol=0.3)

        # Second regime should have slope near 2.0
        assert np.isclose(regime_results[1].params[1], 2.0, atol=0.3)


class TestOLSResults:
    """Test OLS results object."""

    def test_summary(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test summary generation."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        summary = results.summary()
        assert "OLS Regression Results" in summary
        assert "R-squared" in summary
        assert "coef" in summary

    def test_conf_int(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test confidence interval calculation."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        ci = results.conf_int(alpha=0.05)
        assert ci.shape == (2, 2)

        # Lower bound should be less than point estimate
        assert all(ci[:, 0] < results.params)
        # Upper bound should be greater than point estimate
        assert all(ci[:, 1] > results.params)

    def test_to_dataframe(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test conversion to DataFrame."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        df = results.to_dataframe()
        assert "coef" in df.columns
        assert "std_err" in df.columns
        assert "t" in df.columns
        assert len(df) == 2


# =============================================================================
# Additional Covariance Type Tests
# =============================================================================


class TestOLSAllCovTypes:
    """Test all covariance estimators."""

    @pytest.mark.parametrize(
        "cov_type", ["nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC"]
    )
    def test_all_cov_types(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
        cov_type: str,
    ) -> None:
        """Test all covariance types work."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        results = model.fit(cov_type=cov_type)  # type: ignore[arg-type]

        assert results.cov_type == cov_type
        assert len(results.bse) == 2
        assert all(results.bse > 0)

    def test_hac_custom_maxlags(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test HAC with custom maxlags."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": 5})

        assert results.cov_type == "HAC"
        assert all(results.bse > 0)


# =============================================================================
# Multiple Breaks Tests
# =============================================================================


class TestOLSMultipleBreaks:
    """Test OLS with multiple breaks."""

    def test_two_breaks(
        self,
        data_with_two_breaks: tuple[NDArray[np.floating[Any]], list[int]],
    ) -> None:
        """Test OLS with two breaks."""
        y, breaks = data_with_two_breaks

        model = rg.OLS(y, breaks=breaks)
        results = model.fit()

        # 3 regimes * 1 parameter (constant) = 3 params
        assert len(results.params) == 3

    def test_three_breaks(self, rng: np.random.Generator) -> None:
        """Test OLS with three breaks."""
        # Generate data with 3 breaks -> 4 regimes
        y1 = rng.standard_normal(50)
        y2 = rng.standard_normal(50) + 1
        y3 = rng.standard_normal(50) + 2
        y4 = rng.standard_normal(50) + 3
        y = np.concatenate([y1, y2, y3, y4])

        breaks = [50, 100, 150]
        model = rg.OLS(y, breaks=breaks)
        results = model.fit()

        # 4 regimes * 1 parameter = 4 params
        assert len(results.params) == 4

    def test_breaks_summary_multiple(
        self, data_with_two_breaks: tuple[NDArray[np.floating[Any]], list[int]]
    ) -> None:
        """Test summary with multiple breaks."""
        y, breaks = data_with_two_breaks

        model = rg.OLS(y, breaks=breaks)
        results = model.fit()
        summary = results.summary()

        assert "Structural Breaks" in summary
        assert "Regime 1" in summary
        assert "Regime 2" in summary
        assert "Regime 3" in summary


# =============================================================================
# Edge Cases and Validation
# =============================================================================


class TestOLSEdgeCases:
    """Test edge cases and validation."""

    def test_minimal_sample(self, rng: np.random.Generator) -> None:
        """Test with minimal sample size."""
        n = 10
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [1, 2] + rng.standard_normal(n) * 0.1

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        assert len(results.params) == 2
        assert results.nobs == n

    def test_many_regressors(self, rng: np.random.Generator) -> None:
        """Test with many regressors."""
        n = 200
        k = 10
        X = np.column_stack(
            [np.ones(n)] + [rng.standard_normal(n) for _ in range(k - 1)]
        )
        y = X @ np.ones(k) + rng.standard_normal(n)

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        assert len(results.params) == k
        assert results.df_model == k

    def test_auto_add_constant(self, rng: np.random.Generator) -> None:
        """Test automatic constant addition."""
        n = 100
        x = rng.standard_normal(n)
        y = 1 + 2 * x + rng.standard_normal(n) * 0.5

        # Single column without constant - should add one automatically
        model = rg.OLS(y, x.reshape(-1, 1), has_constant=True)
        results = model.fit()

        assert len(results.params) == 2  # constant + x

    def test_perfect_fit(self, rng: np.random.Generator) -> None:
        """Test with near-perfect fit."""
        n = 50
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [1, 2]  # No noise

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        # R-squared should be ~1
        assert results.rsquared > 0.999

    def test_pvalues_and_tvalues(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test p-values and t-values calculation."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        # t-values should be params / bse
        expected_t = results.params / results.bse
        np.testing.assert_array_almost_equal(results.tvalues, expected_t)

        # p-values should be between 0 and 1
        assert all(0 <= p <= 1 for p in results.pvalues)

    def test_fittedvalues_resid_relationship(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test that fittedvalues + resid = y."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        np.testing.assert_array_almost_equal(results.fittedvalues + results.resid, y)


# =============================================================================
# Model Integration Tests
# =============================================================================


class TestOLSModelIntegration:
    """Test OLS model integration methods."""

    def test_rolling_method(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test model.rolling() method."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        rolling = model.rolling(window=30)

        assert rolling.window == 30
        results = rolling.fit()
        assert results.n_valid > 0

    def test_recursive_method(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test model.recursive() method."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        recursive = model.recursive(min_nobs=20)

        assert recursive.min_nobs == 20
        results = recursive.fit()
        assert results.n_valid > 0

    def test_bai_perron_method(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test model.bai_perron() method."""
        y, _ = data_with_mean_shift
        model = rg.OLS(y)
        bp_results = model.bai_perron(max_breaks=2)

        assert bp_results.nobs > 0
        # Should detect the break
        assert bp_results.n_breaks >= 1


# =============================================================================
# Diagnostics Tests
# =============================================================================


class TestOLSDiagnostics:
    """Test diagnostic methods."""

    def test_diagnostics_method(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test diagnostics() method on results."""
        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        diag = results.diagnostics()

        assert diag.normality is not None
        assert diag.heteroskedasticity is not None
        assert diag.autocorrelation is not None

    def test_plot_diagnostics_method(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test plot_diagnostics() method on results."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        y, X = regression_data
        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        fig, axes = results.plot_diagnostics()

        assert fig is not None
        plt.close(fig)


# =============================================================================
# Variable-Specific Breaks Tests
# =============================================================================


class TestOLSVariableBreaks:
    """Test OLS with variable-specific breaks."""

    def test_variable_breaks_by_index(self, rng: np.random.Generator) -> None:
        """Test variable_breaks with integer indices."""
        n = 200
        X = np.column_stack(
            [np.ones(n), rng.standard_normal(n), rng.standard_normal(n)]
        )
        y = rng.standard_normal(n)

        # Variable 1 breaks at obs 100, variable 2 has no breaks
        model = rg.OLS(y, X, has_constant=False, variable_breaks={1: [100]})
        results = model.fit()

        # Should have more parameters due to regime-specific coefficients
        assert results is not None
        assert results.nobs == n

    def test_variable_breaks_by_name(self, rng: np.random.Generator) -> None:
        """Test variable_breaks with variable names."""
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)

        # Use auto-generated variable name (x1 is second column)
        model = rg.OLS(y, X, has_constant=False, variable_breaks={"x1": [100]})
        results = model.fit()

        assert results is not None

    def test_variable_breaks_multiple_vars(self, rng: np.random.Generator) -> None:
        """Test variable_breaks with multiple variables having different breaks."""
        n = 200
        X = np.column_stack(
            [
                np.ones(n),
                rng.standard_normal(n),
                rng.standard_normal(n),
            ]
        )
        y = rng.standard_normal(n)

        # Different breaks for different variables using auto-generated names
        model = rg.OLS(
            y,
            X,
            has_constant=False,
            variable_breaks={"x1": [100], "x2": [50, 150]},
        )
        results = model.fit()

        assert results is not None

    def test_variable_breaks_summary(self, rng: np.random.Generator) -> None:
        """Test summary includes variable-specific break info."""
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)

        # Use auto-generated name x1 for second column
        model = rg.OLS(y, X, has_constant=False, variable_breaks={"x1": [100]})
        results = model.fit()

        summary = results.summary()
        assert "Variable-Specific Structural Breaks" in summary
        assert "x1" in summary
        assert "Regime 1" in summary
        assert "Regime 2" in summary

    def test_variable_breaks_invalid_index_raises(
        self, rng: np.random.Generator
    ) -> None:
        """Test that invalid variable index raises ValueError."""
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)

        model = rg.OLS(y, X, has_constant=False, variable_breaks={5: [50]})

        with pytest.raises(ValueError, match="out of bounds"):
            model.fit()

    def test_variable_breaks_invalid_name_raises(
        self, rng: np.random.Generator
    ) -> None:
        """Test that invalid variable name raises ValueError."""
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)

        # x99 doesn't exist - columns are x0 and x1
        model = rg.OLS(y, X, has_constant=False, variable_breaks={"x99": [50]})

        with pytest.raises(ValueError, match="not found"):
            model.fit()

    def test_variable_breaks_invalid_break_point_raises(
        self, rng: np.random.Generator
    ) -> None:
        """Test that break point at boundary raises ValueError."""
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)

        # Break at 0 is invalid
        model = rg.OLS(y, X, has_constant=False, variable_breaks={1: [0]})

        with pytest.raises(ValueError, match="out of bounds"):
            model.fit()


# =============================================================================
# Summary By Regime Tests
# =============================================================================


class TestSummaryByRegime:
    """Test summary_by_regime function."""

    def test_summary_by_regime_single_regime(self, rng: np.random.Generator) -> None:
        """Test summary_by_regime with single result."""
        from regimes.models.ols import summary_by_regime

        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [1, 2] + rng.standard_normal(n) * 0.5

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        summary = summary_by_regime([results])
        assert "OLS Regression Results by Regime" in summary
        assert "Regime 1" in summary

    def test_summary_by_regime_multiple_regimes(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test summary_by_regime with multiple regimes."""
        from regimes.models.ols import summary_by_regime

        y, X, break_point = regression_data_with_break

        model = rg.OLS(y, X, breaks=[break_point], has_constant=False)
        regime_results = model.fit_by_regime()

        summary = summary_by_regime(
            regime_results, breaks=[break_point], nobs_total=len(y)
        )
        assert "OLS Regression Results by Regime" in summary
        assert "Regime 1" in summary
        assert "Regime 2" in summary
        assert f"Breaks at observations: {break_point}" in summary

    def test_summary_by_regime_empty_list(self) -> None:
        """Test summary_by_regime with empty list."""
        from regimes.models.ols import summary_by_regime

        summary = summary_by_regime([])
        assert "No results to summarize" in summary

    def test_summary_by_regime_hac_note(self, rng: np.random.Generator) -> None:
        """Test summary_by_regime includes HAC note."""
        from regimes.models.ols import summary_by_regime

        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit(cov_type="HAC")

        summary = summary_by_regime([results])
        assert "HAC" in summary

    def test_summary_by_regime_hc_note(self, rng: np.random.Generator) -> None:
        """Test summary_by_regime includes HC note."""
        from regimes.models.ols import summary_by_regime

        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit(cov_type="HC1")

        summary = summary_by_regime([results])
        assert "HC1" in summary

    def test_summary_by_regime_infer_boundaries(self, rng: np.random.Generator) -> None:
        """Test summary_by_regime infers boundaries from results."""
        from regimes.models.ols import summary_by_regime

        # Create two results from different data
        n1, n2 = 50, 75
        X1 = np.column_stack([np.ones(n1), rng.standard_normal(n1)])
        X2 = np.column_stack([np.ones(n2), rng.standard_normal(n2)])
        y1 = rng.standard_normal(n1)
        y2 = rng.standard_normal(n2)

        results1 = rg.OLS(y1, X1, has_constant=False).fit()
        results2 = rg.OLS(y2, X2, has_constant=False).fit()

        # No breaks or nobs_total provided - should infer
        summary = summary_by_regime([results1, results2])
        assert "Regime 1" in summary
        assert "Regime 2" in summary
