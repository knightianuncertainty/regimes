"""Property-based tests for ADL model invariants.

These tests verify that mathematical properties of autoregressive distributed
lag models hold for ANY valid input, using the Hypothesis library.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings

import regimes as rg

from .conftest import adl_data


class TestADLCumulativeEffectProperties:
    """Test cumulative effect properties."""

    @given(data=adl_data())
    @settings(max_examples=50, deadline=None)
    def test_cumulative_effect_equals_sum_of_dl_params(
        self,
        data: tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray],
    ) -> None:
        """Cumulative effect should equal sum of distributed lag coefficients.

        For each exogenous variable x:
        cumulative_effect[x] = sum(dl_params[x])
        """
        y, x, p, q, _, _ = data

        model = rg.ADL(y, x, lags=p if p > 0 else [], exog_lags=q)
        results = model.fit()

        for var_name, dl_coefs in (results.dl_params or {}).items():
            expected_cum = np.sum(dl_coefs)
            actual_cum = results.cumulative_effect.get(var_name, np.nan)

            np.testing.assert_allclose(
                actual_cum,
                expected_cum,
                rtol=1e-10,
                err_msg=f"Cumulative effect for {var_name} mismatch: "
                f"{actual_cum} != sum({dl_coefs}) = {expected_cum}",
            )


class TestADLLongRunMultiplierProperties:
    """Test long-run multiplier properties."""

    @given(data=adl_data())
    @settings(max_examples=50, deadline=None)
    def test_long_run_multiplier_formula(
        self,
        data: tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray],
    ) -> None:
        """Long-run multiplier = cumulative_effect / (1 - sum(ar_params)).

        The long-run multiplier measures the equilibrium effect of a permanent
        unit change in x, accounting for autoregressive dynamics:
        LRM = sum(beta_j) / (1 - sum(phi_i))
        """
        y, x, p, q, _, _ = data

        model = rg.ADL(y, x, lags=p if p > 0 else [], exog_lags=q)
        results = model.fit()

        ar_sum = (
            np.sum(results.ar_params)
            if results.ar_params is not None and len(results.ar_params) > 0
            else 0.0
        )

        # Check if stationary
        if abs(ar_sum) >= 1:
            # Non-stationary: LRM should be NaN
            for var_name in results.cumulative_effect:
                lrm = results.long_run_multiplier.get(var_name, 0)
                assert np.isnan(lrm), (
                    f"LRM for {var_name} should be NaN when non-stationary "
                    f"(ar_sum={ar_sum}), but got {lrm}"
                )
        else:
            # Stationary: LRM = cum_effect / (1 - ar_sum)
            multiplier = 1.0 / (1.0 - ar_sum)

            for var_name, cum_eff in results.cumulative_effect.items():
                expected_lrm = cum_eff * multiplier
                actual_lrm = results.long_run_multiplier.get(var_name, np.nan)

                np.testing.assert_allclose(
                    actual_lrm,
                    expected_lrm,
                    rtol=1e-10,
                    err_msg=f"Long-run multiplier for {var_name} mismatch: "
                    f"{actual_lrm} != {cum_eff} / (1 - {ar_sum}) = {expected_lrm}",
                )

    @given(data=adl_data())
    @settings(max_examples=50, deadline=None)
    def test_lrm_relationship_to_cumulative_effect(
        self,
        data: tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray],
    ) -> None:
        """LRM >= cumulative_effect when AR coefficients are positive.

        If sum(ar_params) > 0, then 1/(1-sum(ar_params)) > 1,
        so LRM > cumulative_effect (in absolute terms for same sign).
        """
        y, x, p, q, _, _ = data

        model = rg.ADL(y, x, lags=p if p > 0 else [], exog_lags=q)
        results = model.fit()

        ar_sum = (
            np.sum(results.ar_params)
            if results.ar_params is not None and len(results.ar_params) > 0
            else 0.0
        )

        # Only check for stationary case with positive AR sum
        if 0 < ar_sum < 1:
            for var_name, cum_eff in results.cumulative_effect.items():
                lrm = results.long_run_multiplier.get(var_name, np.nan)

                if not np.isnan(lrm):
                    # |LRM| > |cum_eff| because multiplier > 1
                    assert abs(lrm) >= abs(cum_eff) - 1e-10, (
                        f"|LRM| = {abs(lrm)} should be >= |cum_eff| = {abs(cum_eff)} "
                        f"when ar_sum = {ar_sum} > 0"
                    )


class TestADLRootProperties:
    """Test ADL characteristic polynomial root properties."""

    @given(data=adl_data())
    @settings(max_examples=50, deadline=None)
    def test_root_count_equals_ar_order(
        self,
        data: tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray],
    ) -> None:
        """Number of roots should equal AR order."""
        y, x, p, q, _, _ = data

        model = rg.ADL(y, x, lags=p if p > 0 else [], exog_lags=q)
        results = model.fit()

        n_roots = len(results.roots)
        n_ar_params = len(results.ar_params) if results.ar_params is not None else 0

        assert n_roots == n_ar_params, (
            f"Root count {n_roots} != AR params count {n_ar_params}"
        )

    @given(data=adl_data())
    @settings(max_examples=50, deadline=None)
    def test_stationarity_check_matches_roots(
        self,
        data: tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray],
    ) -> None:
        """is_stationary should be True iff all roots outside unit circle."""
        y, x, p, q, _, _ = data

        model = rg.ADL(y, x, lags=p if p > 0 else [], exog_lags=q)
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


class TestADLSampleSizeProperties:
    """Test ADL effective sample size properties."""

    @given(data=adl_data())
    @settings(max_examples=50, deadline=None)
    def test_effective_sample_size(
        self,
        data: tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray],
    ) -> None:
        """Effective sample should equal original minus maxlag.

        ADL models lose observations to construct lags.
        maxlag = max(AR_maxlag, DL_maxlag)
        nobs_effective = nobs_original - maxlag
        """
        y, x, p, q, _, _ = data

        model = rg.ADL(y, x, lags=p if p > 0 else [], exog_lags=q)
        results = model.fit()

        n_original = len(y)
        maxlag = max(p, q)
        expected_nobs = n_original - maxlag

        assert results.nobs == expected_nobs, (
            f"nobs={results.nobs}, expected {n_original} - {maxlag} = {expected_nobs}"
        )


class TestADLInheritedOLSProperties:
    """Test that ADL inherits OLS properties."""

    @given(data=adl_data())
    @settings(max_examples=50, deadline=None)
    def test_residuals_sum_to_zero(
        self,
        data: tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray],
    ) -> None:
        """ADL residuals should sum to zero (has constant by default)."""
        y, x, p, q, _, _ = data

        model = rg.ADL(y, x, lags=p if p > 0 else [], exog_lags=q)
        results = model.fit()

        assert abs(np.sum(results.resid)) < 1e-8, (
            f"ADL residuals sum to {np.sum(results.resid)}, expected ~0"
        )

    @given(data=adl_data())
    @settings(max_examples=50, deadline=None)
    def test_fitted_plus_residual_equals_y_effective(
        self,
        data: tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray],
    ) -> None:
        """Fitted values + residuals should equal y (effective sample)."""
        y, x, p, q, _, _ = data

        model = rg.ADL(y, x, lags=p if p > 0 else [], exog_lags=q)
        results = model.fit()

        maxlag = max(p, q)
        y_effective = y[maxlag:]

        reconstructed = results.fittedvalues + results.resid
        np.testing.assert_allclose(
            reconstructed,
            y_effective,
            rtol=1e-10,
            atol=1e-10,
            err_msg="y != fittedvalues + resid",
        )

    @given(data=adl_data())
    @settings(max_examples=50, deadline=None)
    def test_standard_errors_positive(
        self,
        data: tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray],
    ) -> None:
        """Standard errors must be positive."""
        y, x, p, q, _, _ = data

        model = rg.ADL(y, x, lags=p if p > 0 else [], exog_lags=q)
        results = model.fit()

        assert np.all(results.bse > 0), (
            f"Found non-positive standard errors: {results.bse}"
        )

    @given(data=adl_data())
    @settings(max_examples=50, deadline=None)
    def test_rsquared_bounds(
        self,
        data: tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray],
    ) -> None:
        """R-squared should be between 0 and 1."""
        y, x, p, q, _, _ = data

        model = rg.ADL(y, x, lags=p if p > 0 else [], exog_lags=q)
        results = model.fit()

        assert 0 <= results.rsquared <= 1, f"R-squared {results.rsquared} not in [0, 1]"


class TestADLInformationCriteriaProperties:
    """Test ADL information criteria properties."""

    @given(data=adl_data())
    @settings(max_examples=50, deadline=None)
    def test_aic_formula(
        self,
        data: tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray],
    ) -> None:
        """AIC = -2 * log-likelihood + 2 * df_model."""
        y, x, p, q, _, _ = data

        model = rg.ADL(y, x, lags=p if p > 0 else [], exog_lags=q)
        results = model.fit()

        expected_aic = -2 * results.llf + 2 * results.df_model
        np.testing.assert_allclose(
            results.aic,
            expected_aic,
            rtol=1e-10,
            err_msg="AIC formula mismatch",
        )

    @given(data=adl_data())
    @settings(max_examples=50, deadline=None)
    def test_bic_formula(
        self,
        data: tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray],
    ) -> None:
        """BIC = -2 * log-likelihood + log(n) * df_model."""
        y, x, p, q, _, _ = data

        model = rg.ADL(y, x, lags=p if p > 0 else [], exog_lags=q)
        results = model.fit()

        expected_bic = -2 * results.llf + np.log(results.nobs) * results.df_model
        np.testing.assert_allclose(
            results.bic,
            expected_bic,
            rtol=1e-10,
            err_msg="BIC formula mismatch",
        )
