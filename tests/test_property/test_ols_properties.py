"""Property-based tests for OLS model invariants.

These tests verify that mathematical properties of OLS hold for ANY valid input,
using the Hypothesis library for property-based testing.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from scipy import stats

import regimes as rg

from .conftest import regression_data


class TestOLSResidualProperties:
    """Test residual-related OLS properties."""

    @given(data=regression_data(with_constant=True))
    @settings(max_examples=50, deadline=None)
    def test_residuals_sum_to_zero_with_constant(
        self,
        data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Residuals should sum to approximately zero when model has constant.

        This is a fundamental property of OLS: with an intercept term, the
        residuals are orthogonal to the constant, meaning they sum to zero.
        """
        y, X = data

        model = rg.OLS(y, X, has_constant=False)  # X already has constant
        results = model.fit()

        # Residuals should sum to approximately zero (relative to scale)
        # Use a tolerance scaled by the magnitude of residuals
        resid_scale = (
            np.std(results.resid) * len(results.resid)
            if np.std(results.resid) > 0
            else 1.0
        )
        relative_sum = abs(np.sum(results.resid)) / resid_scale

        assert relative_sum < 1e-6, (
            f"Residuals sum to {np.sum(results.resid)} (relative: {relative_sum}), expected ~0"
        )

    @given(data=regression_data())
    @settings(max_examples=50, deadline=None)
    def test_fitted_plus_residual_equals_y(
        self,
        data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Fitted values plus residuals should exactly equal y.

        This is the definitional identity: y = y_hat + e.
        """
        y, X = data

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        # y = fittedvalues + resid (exact identity)
        reconstructed = results.fittedvalues + results.resid
        np.testing.assert_allclose(
            reconstructed,
            y,
            rtol=1e-10,
            atol=1e-10,
            err_msg="y != fittedvalues + resid",
        )


class TestOLSRSquaredProperties:
    """Test R-squared related properties."""

    @given(data=regression_data())
    @settings(max_examples=50, deadline=None)
    def test_rsquared_bounds(
        self,
        data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """R-squared should be between 0 and 1.

        For OLS with an intercept, 0 <= R^2 <= 1. Without an intercept,
        R^2 can technically be negative, but with our test data generation
        it should typically be non-negative.
        """
        y, X = data

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        # Allow small numerical tolerance
        assert -1e-10 <= results.rsquared <= 1 + 1e-10, (
            f"R-squared {results.rsquared} not in [0, 1]"
        )
        # Adjusted R-squared can be negative for poor fits
        assert results.rsquared_adj <= 1 + 1e-10, (
            f"Adjusted R-squared {results.rsquared_adj} exceeds 1"
        )

    @given(data=regression_data(with_constant=True))
    @settings(max_examples=50, deadline=None)
    def test_tss_decomposition(
        self,
        data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Total Sum of Squares = Explained SS + Residual SS.

        When model includes an intercept:
        TSS = sum((y - y_mean)^2)
        ESS = sum((y_hat - y_mean)^2)
        RSS = sum(e^2)
        TSS = ESS + RSS
        """
        y, X = data

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        y_mean = np.mean(y)
        tss = np.sum((y - y_mean) ** 2)
        ess = np.sum((results.fittedvalues - y_mean) ** 2)
        rss = np.sum(results.resid**2)

        # Skip test if TSS is near zero (degenerate case)
        if tss < 1e-10:
            return

        # TSS = ESS + RSS (with relative tolerance)
        np.testing.assert_allclose(
            tss,
            ess + rss,
            rtol=1e-6,
            err_msg=f"TSS decomposition failed: {tss} != {ess} + {rss}",
        )

        # Also verify R^2 = ESS/TSS = 1 - RSS/TSS
        rsq_from_ess = ess / tss
        rsq_from_rss = 1 - rss / tss
        np.testing.assert_allclose(
            results.rsquared,
            rsq_from_ess,
            rtol=1e-5,
            err_msg="R^2 != ESS/TSS",
        )
        np.testing.assert_allclose(
            results.rsquared,
            rsq_from_rss,
            rtol=1e-5,
            err_msg="R^2 != 1 - RSS/TSS",
        )


class TestOLSStandardErrorProperties:
    """Test standard error related properties."""

    @given(data=regression_data())
    @settings(max_examples=50, deadline=None)
    def test_standard_errors_positive(
        self,
        data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Standard errors must be non-negative.

        SE is the square root of diagonal of covariance matrix,
        which must be positive semi-definite. In degenerate cases
        (perfect fit), SE could be zero.
        """
        y, X = data

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        # SE should be non-negative (can be zero in degenerate cases)
        assert np.all(results.bse >= 0), (
            f"Found negative standard errors: {results.bse}"
        )
        # In typical cases with noise, SE should be positive
        if results.ssr > 1e-10:  # Non-degenerate case
            assert np.all(results.bse > 0), (
                f"Found zero standard errors with non-zero RSS: {results.bse}"
            )

    @given(data=regression_data())
    @settings(max_examples=50, deadline=None)
    def test_confidence_interval_contains_estimate(
        self,
        data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Confidence interval must contain the point estimate.

        For any valid CI construction:
        lower_bound < point_estimate < upper_bound
        """
        y, X = data

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        # Skip if SE is zero (degenerate case where CI is undefined)
        if np.any(results.bse == 0):
            return

        ci = results.conf_int(alpha=0.05)

        # Lower bound < estimate
        assert np.all(ci[:, 0] < results.params), (
            "Lower CI bound not less than point estimate"
        )

        # Upper bound > estimate
        assert np.all(ci[:, 1] > results.params), (
            "Upper CI bound not greater than point estimate"
        )

        # CI width should be positive
        ci_width = ci[:, 1] - ci[:, 0]
        assert np.all(ci_width > 0), "CI width not positive"


class TestOLSInformationCriteriaProperties:
    """Test information criteria properties."""

    @given(data=regression_data())
    @settings(max_examples=50, deadline=None)
    def test_aic_formula(
        self,
        data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """AIC = -2 * log-likelihood + 2 * df_model.

        This is the standard AIC definition.
        """
        y, X = data

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        # AIC formula
        expected_aic = -2 * results.llf + 2 * results.df_model
        np.testing.assert_allclose(
            results.aic,
            expected_aic,
            rtol=1e-10,
            err_msg="AIC formula mismatch",
        )

    @given(data=regression_data())
    @settings(max_examples=50, deadline=None)
    def test_bic_formula(
        self,
        data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """BIC = -2 * log-likelihood + log(n) * df_model.

        This is the standard BIC (Schwarz criterion) definition.
        """
        y, X = data

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        # BIC formula
        expected_bic = -2 * results.llf + np.log(results.nobs) * results.df_model
        np.testing.assert_allclose(
            results.bic,
            expected_bic,
            rtol=1e-10,
            err_msg="BIC formula mismatch",
        )

    @given(data=regression_data())
    @settings(max_examples=50, deadline=None)
    def test_bic_greater_than_aic_for_sufficient_n(
        self,
        data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """BIC >= AIC when n >= e^2 (approx 7.4).

        BIC penalizes complexity more than AIC for samples larger than e^2.
        Since log(n) > 2 when n > e^2 ≈ 7.4, and we generate n >= 30,
        BIC should always have a larger penalty and thus be larger.
        """
        y, X = data

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        # For n >= 8, log(n) > 2, so BIC penalty > AIC penalty
        # This means BIC >= AIC
        if results.nobs >= 8 and results.df_model > 0:
            assert results.bic >= results.aic - 1e-10, (
                f"BIC {results.bic} < AIC {results.aic} for n={results.nobs}"
            )


class TestOLSCovarianceProperties:
    """Test covariance matrix properties."""

    @given(data=regression_data())
    @settings(max_examples=50, deadline=None)
    def test_cov_matrix_symmetric(
        self,
        data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Covariance matrix must be symmetric."""
        y, X = data

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        cov = results.cov_params()
        np.testing.assert_allclose(
            cov,
            cov.T,
            rtol=1e-10,
            err_msg="Covariance matrix not symmetric",
        )

    @given(data=regression_data())
    @settings(max_examples=50, deadline=None)
    def test_cov_matrix_positive_semidefinite(
        self,
        data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Covariance matrix must be positive semi-definite.

        All eigenvalues should be non-negative.
        """
        y, X = data

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        cov = results.cov_params()
        eigenvalues = np.linalg.eigvalsh(cov)

        # Allow small numerical errors
        assert np.all(eigenvalues >= -1e-10), (
            f"Covariance matrix has negative eigenvalues: {eigenvalues}"
        )


class TestOLSInferenceProperties:
    """Test inference-related properties."""

    @given(data=regression_data())
    @settings(max_examples=50, deadline=None)
    def test_tvalues_formula(
        self,
        data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """t-values should equal params / bse."""
        y, X = data

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        expected_tvalues = results.params / results.bse
        np.testing.assert_allclose(
            results.tvalues,
            expected_tvalues,
            rtol=1e-10,
            err_msg="t-values formula mismatch",
        )

    @given(data=regression_data())
    @settings(max_examples=50, deadline=None)
    def test_pvalues_bounds(
        self,
        data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """p-values must be between 0 and 1 (when defined)."""
        y, X = data

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        # Filter out NaN p-values (which occur when SE is zero)
        valid_pvalues = results.pvalues[~np.isnan(results.pvalues)]

        if len(valid_pvalues) > 0:
            assert np.all(valid_pvalues >= 0), "p-values must be >= 0"
            assert np.all(valid_pvalues <= 1), "p-values must be <= 1"

    @given(data=regression_data())
    @settings(max_examples=50, deadline=None)
    def test_pvalues_symmetric_in_tvalues(
        self,
        data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """p-values should be same for +t and -t (two-sided test)."""
        y, X = data

        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()

        # Compute p-values manually using absolute t-values
        df = results.df_resid
        expected_pvalues = 2 * (1 - stats.t.cdf(np.abs(results.tvalues), df))

        np.testing.assert_allclose(
            results.pvalues,
            expected_pvalues,
            rtol=1e-6,
            err_msg="p-values not symmetric in t-values",
        )
