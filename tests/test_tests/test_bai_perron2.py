"""
Tests for Bai-Perron (1998, 2003) implementation correctness.

Verifies:
1. Partial structural change models (mixed breaking/non-breaking regressors).
2. Multicollinearity handling in model initialization.
3. Parameter mapping when converting to OLS representation.
4. Edge cases: exog=None, convergence, multiple constant columns.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimes import OLS, BaiPerronTest


class TestBaiPerronCorrectness:
    """Verification of Bai-Perron estimation procedures."""

    def test_partial_structural_change_consistency(self) -> None:
        """
        Verify consistency of partial structural change estimation.

        Ensures that non-breaking regressors are constrained to have global
        coefficients, while breaking regressors vary across regimes.
        Compares the estimated SSR against a restricted OLS benchmark.
        """
        np.random.seed(42)
        n = 100

        # DGP: y_t = 2*x_t + z_t * (1 if t < 50 else 3) + e_t
        # x is non-breaking, z is breaking (intercept)
        x = np.random.randn(n, 1)
        z = np.ones((n, 1))

        y = 2 * x.flatten()
        y[:50] += 1
        y[50:] += 3
        y += np.random.randn(n) * 0.1

        bp = BaiPerronTest(y, exog=x, exog_break=z)
        results = bp.fit(max_breaks=1)

        # Benchmark: Restricted OLS
        # y = beta*x + delta1*D1 + delta2*D2
        X_restricted = np.zeros((n, 3))
        X_restricted[:, 0] = x.flatten()
        X_restricted[:50, 1] = 1
        X_restricted[50:, 2] = 1

        beta_r = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
        resid_r = y - X_restricted @ beta_r
        ssr_restricted = np.sum(resid_r**2)

        # The iterative procedure should converge to the restricted OLS solution
        assert np.isclose(results.ssr[1], ssr_restricted, rtol=1e-3), (
            f"SSR divergence: BP={results.ssr[1]:.4f}, Restricted={ssr_restricted:.4f}"
        )

    def test_initialization_collinearity(self) -> None:
        """
        Verify handling of constant terms in model initialization.

        When break_vars="const", the constant should be moved from the
        non-breaking set (exog) to the breaking set (exog_break) to
        prevent perfect multicollinearity.  After partitioning, fitting
        must succeed without errors.
        """
        np.random.seed(123)
        n = 50
        x1 = np.random.randn(n)
        X = np.column_stack([np.ones(n), x1])
        y = 1 + x1 + np.random.randn(n)

        model = OLS(y, X, has_constant=False)
        bp = BaiPerronTest.from_model(model, break_vars="const")

        # Check partition of regressors
        has_const_in_exog = False
        if bp.exog is not None:
            has_const_in_exog = np.any(np.all(np.isclose(bp.exog, 1.0), axis=0))

        has_const_in_break = False
        if bp.exog_break is not None:
            has_const_in_break = np.any(np.all(np.isclose(bp.exog_break, 1.0), axis=0))

        assert has_const_in_break, "Constant missing from breaking regressors"
        assert not has_const_in_exog, (
            "Constant retained in non-breaking regressors (collinearity risk)"
        )

        # Verify that fitting actually succeeds without errors
        results = bp.fit(max_breaks=1)
        assert results is not None, "fit() returned None after collinearity fix"
        assert 0 in results.ssr, "SSR dict missing m=0 entry"
        assert np.isfinite(results.ssr[0]), "SSR for m=0 is non-finite"

    def test_ols_parameter_mapping_via_detection(self) -> None:
        """
        Verify parameter mapping in OLS conversion using the actual
        detection pipeline (no manual override of break_indices).

        Non-breaking regressors should map to a single global parameter.
        Breaking regressors should map to (m+1) regime-specific parameters.
        """
        np.random.seed(42)
        n = 200

        x_global = np.random.randn(n, 1)
        x_break = np.ones((n, 1))

        # Large mean shift to ensure detection succeeds
        y = 0.5 * x_global.flatten()
        y[:100] += 1.0
        y[100:] += 5.0
        y += np.random.randn(n) * 0.1

        bp = BaiPerronTest(y, exog=x_global, exog_break=x_break)
        results = bp.fit(max_breaks=2)

        assert results.n_breaks >= 1, (
            f"Detection should find at least 1 break, got {results.n_breaks}"
        )

        ols = results.to_ols()

        # Expected: 1 global param + (n_breaks+1) regime intercepts
        expected_params = 1 + (results.n_breaks + 1)
        assert len(ols.params) == expected_params, (
            f"Incorrect parameter count: {len(ols.params)} (expected {expected_params})"
        )

    def test_pure_structural_change_exog_none(self) -> None:
        """
        Verify that the model runs correctly when exog=None (pure
        structural change: all regressors are breaking).
        """
        np.random.seed(99)
        n = 100

        y = np.zeros(n)
        y[:50] = 1.0 + np.random.randn(50) * 0.1
        y[50:] = 3.0 + np.random.randn(50) * 0.1

        # Pure structural change: only exog_break, no exog
        bp = BaiPerronTest(y, exog=None, exog_break=np.ones((n, 1)))
        results = bp.fit(max_breaks=2)

        assert results is not None
        assert results.n_breaks >= 1, (
            "Should detect at least one break in a clear mean-shift DGP"
        )
        assert np.isfinite(results.ssr[0])
        assert np.isfinite(results.ssr[1])

    def test_default_mean_shift_exog_none(self) -> None:
        """
        Verify that the default constructor (no exog, no exog_break)
        creates a valid mean-shift model and fit() does not crash.
        """
        np.random.seed(77)
        y = np.concatenate([np.random.randn(50), np.random.randn(50) + 4.0])

        bp = BaiPerronTest(y)
        results = bp.fit(max_breaks=2)

        assert results is not None
        assert results.n_breaks >= 1

    def test_convergence_partial_structural_change(self) -> None:
        """
        Verify the iterative procedure converges: the SSR with 1 break
        should be strictly less than SSR with 0 breaks when the DGP
        has a genuine break.
        """
        np.random.seed(10)
        n = 120

        x = np.random.randn(n, 1)
        z = np.ones((n, 1))

        y = 1.5 * x.flatten()
        y[:60] += 2.0
        y[60:] += 6.0
        y += np.random.randn(n) * 0.2

        bp = BaiPerronTest(y, exog=x, exog_break=z)
        results = bp.fit(max_breaks=1)

        assert results.ssr[1] < results.ssr[0], (
            f"SSR with 1 break ({results.ssr[1]:.4f}) should be less than "
            f"SSR with 0 breaks ({results.ssr[0]:.4f})"
        )

    def test_multiple_constant_columns(self) -> None:
        """
        Verify that from_model correctly handles exog matrices that
        contain multiple constant-like columns (e.g., duplicated intercept).
        Only one should be moved to exog_break.
        """
        np.random.seed(55)
        n = 60
        x1 = np.random.randn(n)
        # Two constant columns (duplicated intercept)
        X = np.column_stack([np.ones(n), np.ones(n), x1])
        y = 1.0 + x1 + np.random.randn(n) * 0.5

        model = OLS(y, X, has_constant=False)
        bp = BaiPerronTest.from_model(model, break_vars="const")

        # At least one constant should be in exog_break
        assert bp.exog_break is not None
        has_const_in_break = np.any(np.all(np.isclose(bp.exog_break, 1.0), axis=0))
        assert has_const_in_break, "Constant missing from breaking regressors"

        # The remaining exog (if any) should not have all-ones columns
        if bp.exog is not None:
            for col in range(bp.exog.shape[1]):
                assert not np.allclose(bp.exog[:, col], 1.0), (
                    f"Column {col} in exog is still a constant after from_model()"
                )

    def test_to_ols_without_test_reference_raises(self) -> None:
        """
        Verify that to_ols() raises ValueError when _test is None.
        """
        from regimes.tests.bai_perron import BaiPerronResults

        results = BaiPerronResults(
            test_name="Bai-Perron",
            nobs=100,
            n_breaks=1,
            break_indices=[50],
            _test=None,
        )
        with pytest.raises(ValueError, match="Cannot convert to OLS"):
            results.to_ols()

    def test_selection_sequential(self) -> None:
        """
        Verify that selection='sequential' works for partial structural
        change and returns a valid result.
        """
        np.random.seed(42)
        n = 200

        x = np.random.randn(n, 1)
        z = np.ones((n, 1))

        y = 1.0 * x.flatten()
        y[:100] += 1.0
        y[100:] += 5.0
        y += np.random.randn(n) * 0.1

        bp = BaiPerronTest(y, exog=x, exog_break=z)
        results = bp.fit(max_breaks=3, selection="sequential")

        assert results.selection_method == "sequential"
        assert results.n_breaks >= 1, (
            "Sequential selection should detect at least 1 break"
        )
        # Sequential F-stats should be populated
        assert len(results.seqf_stats) > 0

    def test_selection_lwz(self) -> None:
        """
        Verify that selection='lwz' works and selects a valid number
        of breaks.
        """
        np.random.seed(42)
        n = 200

        y = np.zeros(n)
        y[:100] = 1.0 + np.random.randn(100) * 0.1
        y[100:] = 5.0 + np.random.randn(100) * 0.1

        bp = BaiPerronTest(y)
        results = bp.fit(max_breaks=3, selection="lwz")

        assert results.selection_method == "lwz"
        assert results.n_breaks >= 0
        assert len(results.lwz) > 0
        # LWZ values should be finite
        for m, val in results.lwz.items():
            assert np.isfinite(val), f"LWZ for m={m} is not finite"

    def test_from_model_ar(self) -> None:
        """
        Verify that from_model works correctly with an AR model.
        """
        from regimes import AR

        np.random.seed(42)
        n = 200
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.7 * y[t - 1] + np.random.randn()

        model = AR(y, lags=1)
        bp = BaiPerronTest.from_model(model, break_vars="all")
        results = bp.fit(max_breaks=2)

        assert results is not None
        assert results.nobs > 0
        assert 0 in results.ssr

    def test_from_model_adl(self) -> None:
        """
        Verify that from_model works correctly with an ADL model.
        """
        from regimes import ADL

        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + 0.3 * x[t] + np.random.randn()

        model = ADL(y, x, lags=1, exog_lags=1)
        bp = BaiPerronTest.from_model(model, break_vars="all")
        results = bp.fit(max_breaks=2)

        assert results is not None
        assert results.nobs > 0
        assert 0 in results.ssr

    def test_convergence_warning_not_raised_on_easy_problem(self) -> None:
        """
        Verify that no convergence warning is emitted for a simple
        well-separated DGP.
        """
        import warnings

        np.random.seed(42)
        n = 100
        x = np.random.randn(n, 1)
        z = np.ones((n, 1))

        y = 2.0 * x.flatten()
        y[:50] += 1.0
        y[50:] += 10.0
        y += np.random.randn(n) * 0.05

        bp = BaiPerronTest(y, exog=x, exog_break=z)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bp.fit(max_breaks=1)
            convergence_warnings = [
                x for x in w if "did not converge" in str(x.message)
            ]
            assert len(convergence_warnings) == 0, (
                "Convergence warning should not fire on well-separated data"
            )
