"""Property-based tests for AR model invariants.

These tests verify that mathematical properties of autoregressive models
hold for ANY valid input, using the Hypothesis library.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings

import regimes as rg

from .conftest import stationary_ar_data


class TestARRootProperties:
    """Test AR characteristic polynomial root properties."""

    @given(data=stationary_ar_data(max_p=3))
    @settings(max_examples=50, deadline=None)
    def test_root_count_equals_ar_order(
        self,
        data: tuple[np.ndarray, list[int], np.ndarray],
    ) -> None:
        """Number of roots should equal AR order (number of lags).

        The characteristic polynomial of an AR(p) model has degree p,
        so it has exactly p roots (counting multiplicities).
        """
        y, lags, _ = data

        model = rg.AR(y, lags=lags)
        results = model.fit()

        n_roots = len(results.roots)
        n_lags = len(lags)

        assert n_roots == n_lags, f"Root count {n_roots} != AR order {n_lags}"

    @given(data=stationary_ar_data(max_p=3))
    @settings(max_examples=50, deadline=None)
    def test_stationarity_check_matches_roots(
        self,
        data: tuple[np.ndarray, list[int], np.ndarray],
    ) -> None:
        """is_stationary should be True iff all roots outside unit circle.

        For stationarity, all roots of the characteristic polynomial
        must have modulus > 1 (outside unit circle).
        """
        y, lags, _ = data

        model = rg.AR(y, lags=lags)
        results = model.fit()

        if len(results.roots) == 0:
            # No AR terms, trivially stationary
            assert results.is_stationary is True
        else:
            roots_outside = np.all(np.abs(results.roots) > 1)
            assert results.is_stationary == roots_outside, (
                f"is_stationary={results.is_stationary} but "
                f"roots outside unit circle={roots_outside}, "
                f"root moduli={np.abs(results.roots)}"
            )


class TestARSampleSizeProperties:
    """Test AR effective sample size properties."""

    @given(data=stationary_ar_data(max_p=3))
    @settings(max_examples=50, deadline=None)
    def test_effective_sample_size(
        self,
        data: tuple[np.ndarray, list[int], np.ndarray],
    ) -> None:
        """Effective sample should equal original minus maxlag.

        AR models lose observations at the beginning to construct lags.
        nobs_effective = nobs_original - maxlag
        """
        y, lags, _ = data

        model = rg.AR(y, lags=lags)
        results = model.fit()

        n_original = len(y)
        maxlag = max(lags) if lags else 0
        expected_nobs = n_original - maxlag

        assert results.nobs == expected_nobs, (
            f"nobs={results.nobs}, expected {n_original} - {maxlag} = {expected_nobs}"
        )


class TestARParameterProperties:
    """Test AR parameter-related properties."""

    @given(data=stationary_ar_data(max_p=3))
    @settings(max_examples=50, deadline=None)
    def test_ar_params_count(
        self,
        data: tuple[np.ndarray, list[int], np.ndarray],
    ) -> None:
        """Number of ar_params should equal number of lags."""
        y, lags, _ = data

        model = rg.AR(y, lags=lags)
        results = model.fit()

        assert len(results.ar_params) == len(lags), (
            f"ar_params length {len(results.ar_params)} != lags length {len(lags)}"
        )

    @given(data=stationary_ar_data(max_p=3))
    @settings(max_examples=50, deadline=None)
    def test_ar_params_extracted_correctly(
        self,
        data: tuple[np.ndarray, list[int], np.ndarray],
    ) -> None:
        """AR params should be subset of full params.

        The ar_params attribute should contain only the AR coefficients,
        not the constant or other deterministic terms.
        """
        y, lags, _ = data

        model = rg.AR(y, lags=lags)
        results = model.fit()

        # AR params should appear in full params
        # Find indices of y.L* parameters
        ar_indices = [
            i for i, name in enumerate(results.param_names) if name.startswith("y.L")
        ]

        expected_ar_params = results.params[ar_indices]
        np.testing.assert_array_equal(
            results.ar_params,
            expected_ar_params,
            err_msg="ar_params not correctly extracted from params",
        )


class TestARInformationCriteriaProperties:
    """Test AR information criteria properties."""

    @given(data=stationary_ar_data(max_p=3))
    @settings(max_examples=50, deadline=None)
    def test_aic_formula(
        self,
        data: tuple[np.ndarray, list[int], np.ndarray],
    ) -> None:
        """AIC = -2 * log-likelihood + 2 * df_model."""
        y, lags, _ = data

        model = rg.AR(y, lags=lags)
        results = model.fit()

        expected_aic = -2 * results.llf + 2 * results.df_model
        np.testing.assert_allclose(
            results.aic,
            expected_aic,
            rtol=1e-10,
            err_msg="AIC formula mismatch",
        )

    @given(data=stationary_ar_data(max_p=3))
    @settings(max_examples=50, deadline=None)
    def test_bic_formula(
        self,
        data: tuple[np.ndarray, list[int], np.ndarray],
    ) -> None:
        """BIC = -2 * log-likelihood + log(n) * df_model."""
        y, lags, _ = data

        model = rg.AR(y, lags=lags)
        results = model.fit()

        expected_bic = -2 * results.llf + np.log(results.nobs) * results.df_model
        np.testing.assert_allclose(
            results.bic,
            expected_bic,
            rtol=1e-10,
            err_msg="BIC formula mismatch",
        )


class TestARInheritedOLSProperties:
    """Test that AR inherits OLS properties (since it's estimated via OLS)."""

    @given(data=stationary_ar_data(max_p=3))
    @settings(max_examples=50, deadline=None)
    def test_residuals_sum_to_zero(
        self,
        data: tuple[np.ndarray, list[int], np.ndarray],
    ) -> None:
        """AR residuals should sum to zero (has constant by default)."""
        y, lags, _ = data

        model = rg.AR(y, lags=lags)
        results = model.fit()

        assert abs(np.sum(results.resid)) < 1e-8, (
            f"AR residuals sum to {np.sum(results.resid)}, expected ~0"
        )

    @given(data=stationary_ar_data(max_p=3))
    @settings(max_examples=50, deadline=None)
    def test_fitted_plus_residual_equals_y_effective(
        self,
        data: tuple[np.ndarray, list[int], np.ndarray],
    ) -> None:
        """Fitted values + residuals should equal y (effective sample)."""
        y, lags, _ = data

        model = rg.AR(y, lags=lags)
        results = model.fit()

        maxlag = max(lags) if lags else 0
        y_effective = y[maxlag:]

        reconstructed = results.fittedvalues + results.resid
        np.testing.assert_allclose(
            reconstructed,
            y_effective,
            rtol=1e-10,
            atol=1e-10,
            err_msg="y != fittedvalues + resid",
        )

    @given(data=stationary_ar_data(max_p=3))
    @settings(max_examples=50, deadline=None)
    def test_standard_errors_positive(
        self,
        data: tuple[np.ndarray, list[int], np.ndarray],
    ) -> None:
        """Standard errors must be positive."""
        y, lags, _ = data

        model = rg.AR(y, lags=lags)
        results = model.fit()

        assert np.all(results.bse > 0), (
            f"Found non-positive standard errors: {results.bse}"
        )

    @given(data=stationary_ar_data(max_p=3))
    @settings(max_examples=50, deadline=None)
    def test_rsquared_bounds(
        self,
        data: tuple[np.ndarray, list[int], np.ndarray],
    ) -> None:
        """R-squared should be between 0 and 1."""
        y, lags, _ = data

        model = rg.AR(y, lags=lags)
        results = model.fit()

        assert 0 <= results.rsquared <= 1, f"R-squared {results.rsquared} not in [0, 1]"
