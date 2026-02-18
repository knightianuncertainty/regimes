"""Tests for CUSUM and CUSUM-SQ structural break tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import regimes as rg
from regimes import ADL, AR, OLS, CUSUMResults, CUSUMSQResults, CUSUMSQTest, CUSUMTest
from regimes.tests.cusum import _compute_recursive_residuals


class TestRecursiveResiduals:
    """Tests for the recursive residuals computation."""

    def test_length(self, rng: np.random.Generator) -> None:
        """Recursive residuals should have length T - k."""
        T, k = 100, 3
        X = np.column_stack([np.ones(T), rng.standard_normal((T, k - 1))])
        y = X @ rng.standard_normal(k) + rng.standard_normal(T)

        w, f = _compute_recursive_residuals(y, X)

        assert len(w) == T - k
        assert len(f) == T - k

    def test_zero_mean_under_null(self, rng: np.random.Generator) -> None:
        """Under parameter stability, recursive residuals should have zero mean."""
        T = 500
        X = np.column_stack([np.ones(T), rng.standard_normal(T)])
        beta = np.array([1.0, 2.0])
        y = X @ beta + rng.standard_normal(T) * 0.5

        w, f = _compute_recursive_residuals(y, X)

        # Mean should be close to zero (not exact due to finite sample)
        assert abs(np.mean(w)) < 0.3

    def test_positive_scaling_factors(self, rng: np.random.Generator) -> None:
        """Scaling factors f_t should all be positive (> 1)."""
        T = 100
        X = np.column_stack([np.ones(T), rng.standard_normal(T)])
        y = rng.standard_normal(T)

        w, f = _compute_recursive_residuals(y, X)

        assert np.all(f > 1.0)

    def test_constant_only_model(self, rng: np.random.Generator) -> None:
        """Should work with a constant-only model (k=1)."""
        T = 50
        X = np.ones((T, 1))
        y = 3.0 + rng.standard_normal(T)

        w, f = _compute_recursive_residuals(y, X)

        assert len(w) == T - 1
        assert len(f) == T - 1

    def test_insufficient_observations(self) -> None:
        """Should raise error when T <= k."""
        X = np.ones((3, 3))
        y = np.ones(3)

        with pytest.raises(ValueError, match="Need more observations"):
            _compute_recursive_residuals(y, X)


class TestCUSUMBasic:
    """Basic CUSUM test functionality."""

    def test_stable_data_not_rejected(
        self, simple_data: NDArray[np.floating[Any]]
    ) -> None:
        """Test on stable data — should not reject."""
        test = CUSUMTest(simple_data)
        results = test.fit(significance=0.05)

        assert not results.reject
        assert results.is_stable
        assert results.n_breaks == 0
        assert len(results.break_indices) == 0

    def test_mean_shift_detected(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test detection of mean shift."""
        y, true_break = data_with_mean_shift

        test = CUSUMTest(y)
        results = test.fit(significance=0.05)

        assert results.reject
        assert not results.is_stable
        assert results.n_breaks >= 1

    def test_statistic_path_shape(self, rng: np.random.Generator) -> None:
        """Statistic path should have length T - k."""
        T = 100
        y = rng.standard_normal(T)
        test = CUSUMTest(y)  # constant only, k=1
        results = test.fit(significance=0.05)

        assert len(results.statistic_path) == T - 1
        assert len(results.upper_bound) == T - 1
        assert len(results.lower_bound) == T - 1

    def test_symmetric_bounds(self, rng: np.random.Generator) -> None:
        """Upper and lower bounds should be symmetric around zero."""
        y = rng.standard_normal(100)
        test = CUSUMTest(y)
        results = test.fit(significance=0.05)

        np.testing.assert_allclose(
            results.upper_bound, -results.lower_bound, rtol=1e-10
        )

    def test_diverging_bounds(self, rng: np.random.Generator) -> None:
        """Bounds should be diverging (increasing over time)."""
        y = rng.standard_normal(200)
        test = CUSUMTest(y)
        results = test.fit(significance=0.05)

        # Upper bound should be strictly increasing
        diffs = np.diff(results.upper_bound)
        assert np.all(diffs > 0)

    def test_with_regressors(self, rng: np.random.Generator) -> None:
        """Test with explicit regressors."""
        T = 200
        X = np.column_stack([np.ones(T), rng.standard_normal(T)])
        y = X @ [1.0, 2.0] + rng.standard_normal(T)

        test = CUSUMTest(y, exog=X)
        results = test.fit(significance=0.05)

        assert len(results.statistic_path) == T - 2  # T - k, k=2
        assert results.n_params == 2


class TestCUSUMStatistics:
    """Tests for CUSUM statistic properties."""

    def test_sigma_hat_positive(self, rng: np.random.Generator) -> None:
        """Sigma hat should be positive."""
        y = rng.standard_normal(100)
        test = CUSUMTest(y)
        results = test.fit(significance=0.05)

        assert results.sigma_hat > 0

    def test_max_statistic_property(self, rng: np.random.Generator) -> None:
        """max_statistic should equal max |statistic_path|."""
        y = rng.standard_normal(100)
        test = CUSUMTest(y)
        results = test.fit(significance=0.05)

        assert results.max_statistic == float(np.max(np.abs(results.statistic_path)))

    @pytest.mark.parametrize("significance", [0.01, 0.05, 0.10])
    def test_significance_levels(
        self, significance: float, rng: np.random.Generator
    ) -> None:
        """Test all supported significance levels."""
        y = rng.standard_normal(100)
        test = CUSUMTest(y)
        results = test.fit(significance=significance)

        assert results.significance == significance

    def test_narrower_bounds_at_higher_significance(
        self, rng: np.random.Generator
    ) -> None:
        """Bounds at 10% should be narrower than at 1%."""
        y = rng.standard_normal(100)
        test = CUSUMTest(y)

        results_01 = test.fit(significance=0.01)
        results_10 = test.fit(significance=0.10)

        # 10% bounds should be narrower (less conservative)
        assert results_10.upper_bound[-1] < results_01.upper_bound[-1]

    def test_recursive_residuals_stored(self, rng: np.random.Generator) -> None:
        """Recursive residuals should be stored in results."""
        y = rng.standard_normal(100)
        test = CUSUMTest(y)
        results = test.fit(significance=0.05)

        assert len(results.recursive_residuals) == 99  # T - k = 100 - 1


class TestCUSUMSQBasic:
    """Basic CUSUM-SQ test functionality."""

    def test_stable_data_not_rejected(
        self, simple_data: NDArray[np.floating[Any]]
    ) -> None:
        """Test on stable data — should not reject."""
        test = CUSUMSQTest(simple_data)
        results = test.fit(significance=0.05)

        assert not results.reject
        assert results.is_stable

    def test_path_in_unit_interval(self, rng: np.random.Generator) -> None:
        """CUSUM-SQ path should be in [0, 1]."""
        y = rng.standard_normal(100)
        test = CUSUMSQTest(y)
        results = test.fit(significance=0.05)

        assert np.all(results.statistic_path >= 0)
        assert np.all(results.statistic_path <= 1.0 + 1e-10)

    def test_path_monotonic(self, rng: np.random.Generator) -> None:
        """CUSUM-SQ path should be monotonically increasing."""
        y = rng.standard_normal(100)
        test = CUSUMSQTest(y)
        results = test.fit(significance=0.05)

        diffs = np.diff(results.statistic_path)
        assert np.all(diffs >= -1e-10)

    def test_path_starts_near_zero(self, rng: np.random.Generator) -> None:
        """First value of CUSUM-SQ path should be near 0."""
        y = rng.standard_normal(200)
        test = CUSUMSQTest(y)
        results = test.fit(significance=0.05)

        # First observation contributes w_1^2 / sum(w^2), should be small
        assert results.statistic_path[0] < 0.1

    def test_path_ends_at_one(self, rng: np.random.Generator) -> None:
        """Last value of CUSUM-SQ path should be exactly 1."""
        y = rng.standard_normal(100)
        test = CUSUMSQTest(y)
        results = test.fit(significance=0.05)

        np.testing.assert_allclose(results.statistic_path[-1], 1.0, atol=1e-10)

    def test_expected_path_diagonal(self, rng: np.random.Generator) -> None:
        """Expected path should be the diagonal from 0 to 1."""
        T = 100
        y = rng.standard_normal(T)
        test = CUSUMSQTest(y)
        results = test.fit(significance=0.05)

        n = T - 1  # k=1 for constant only
        expected = np.arange(1, n + 1) / n
        np.testing.assert_allclose(results.expected_path, expected)

    def test_variance_change_detected(self, rng: np.random.Generator) -> None:
        """CUSUM-SQ should detect a variance change."""
        # Strong variance shift: sd = 0.5 then sd = 5.0
        y1 = rng.normal(0, 0.5, 150)
        y2 = rng.normal(0, 5.0, 150)
        y = np.concatenate([y1, y2])

        test = CUSUMSQTest(y)
        results = test.fit(significance=0.05)

        assert results.reject

    def test_max_deviation_property(self, rng: np.random.Generator) -> None:
        """max_deviation should equal max |S_r - E[S_r]|."""
        y = rng.standard_normal(100)
        test = CUSUMSQTest(y)
        results = test.fit(significance=0.05)

        expected = float(np.max(np.abs(results.statistic_path - results.expected_path)))
        assert results.max_deviation == pytest.approx(expected)


class TestCUSUMFromModel:
    """Tests for creating CUSUM tests from model objects."""

    def test_from_ols_model(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Create CUSUMTest from an OLS model."""
        y, X = regression_data
        model = OLS(y, X, has_constant=False)

        test = CUSUMTest.from_model(model)
        results = test.fit(significance=0.05)

        assert isinstance(results, CUSUMResults)
        assert results.n_params == X.shape[1]

    def test_from_ar_model(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """Create CUSUMTest from an AR model."""
        model = AR(ar1_data, lags=1)

        test = CUSUMTest.from_model(model)
        results = test.fit(significance=0.05)

        assert isinstance(results, CUSUMResults)

    def test_from_adl_model(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Create CUSUMTest from an ADL model."""
        y, x = adl_data
        model = ADL(y, x, lags=1, exog_lags=1)

        test = CUSUMTest.from_model(model)
        results = test.fit(significance=0.05)

        assert isinstance(results, CUSUMResults)

    def test_cusumsq_from_ols_model(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Create CUSUMSQTest from an OLS model."""
        y, X = regression_data
        model = OLS(y, X, has_constant=False)

        test = CUSUMSQTest.from_model(model)
        results = test.fit(significance=0.05)

        assert isinstance(results, CUSUMSQResults)

    def test_invalid_model_type(self) -> None:
        """Should raise TypeError for unsupported model types."""
        with pytest.raises(TypeError, match="model must be OLS, AR, or ADL"):
            CUSUMTest.from_model("not a model")  # type: ignore[arg-type]

    def test_cusumsq_invalid_model_type(self) -> None:
        """Should raise TypeError for unsupported model types in CUSUM-SQ."""
        with pytest.raises(TypeError, match="model must be OLS, AR, or ADL"):
            CUSUMSQTest.from_model(42)  # type: ignore[arg-type]


class TestOLSCUSUMMethod:
    """Tests for OLS convenience methods."""

    def test_cusum_test_method(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """OLS.cusum_test() should return CUSUMResults."""
        y, X = regression_data
        model = OLS(y, X, has_constant=False)

        results = model.cusum_test(significance=0.05)

        assert isinstance(results, CUSUMResults)
        assert results.test_name == "CUSUM"

    def test_cusum_sq_test_method(
        self,
        regression_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """OLS.cusum_sq_test() should return CUSUMSQResults."""
        y, X = regression_data
        model = OLS(y, X, has_constant=False)

        results = model.cusum_sq_test(significance=0.05)

        assert isinstance(results, CUSUMSQResults)
        assert results.test_name == "CUSUM-SQ"


class TestARCUSUMMethod:
    """Tests for AR convenience methods."""

    def test_cusum_test_method(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """AR.cusum_test() should return CUSUMResults."""
        model = AR(ar1_data, lags=1)
        results = model.cusum_test(significance=0.05)

        assert isinstance(results, CUSUMResults)

    def test_cusum_sq_test_method(self, ar1_data: NDArray[np.floating[Any]]) -> None:
        """AR.cusum_sq_test() should return CUSUMSQResults."""
        model = AR(ar1_data, lags=1)
        results = model.cusum_sq_test(significance=0.05)

        assert isinstance(results, CUSUMSQResults)


class TestADLCUSUMMethod:
    """Tests for ADL convenience methods."""

    def test_cusum_test_method(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """ADL.cusum_test() should return CUSUMResults."""
        y, x = adl_data
        model = ADL(y, x, lags=1, exog_lags=1)
        results = model.cusum_test(significance=0.05)

        assert isinstance(results, CUSUMResults)

    def test_cusum_sq_test_method(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """ADL.cusum_sq_test() should return CUSUMSQResults."""
        y, x = adl_data
        model = ADL(y, x, lags=1, exog_lags=1)
        results = model.cusum_sq_test(significance=0.05)

        assert isinstance(results, CUSUMSQResults)


class TestCUSUMResults:
    """Tests for CUSUM results objects."""

    def test_summary_format(self, rng: np.random.Generator) -> None:
        """Summary should be a formatted string."""
        y = rng.standard_normal(100)
        test = CUSUMTest(y)
        results = test.fit(significance=0.05)

        summary = results.summary()
        assert isinstance(summary, str)
        assert "CUSUM Test for Parameter Instability" in summary
        assert "Number of observations" in summary
        assert "Sigma hat" in summary

    def test_cusumsq_summary_format(self, rng: np.random.Generator) -> None:
        """CUSUM-SQ summary should be a formatted string."""
        y = rng.standard_normal(100)
        test = CUSUMSQTest(y)
        results = test.fit(significance=0.05)

        summary = results.summary()
        assert isinstance(summary, str)
        assert "CUSUM-SQ Test for Variance Instability" in summary

    def test_n_regimes(self, rng: np.random.Generator) -> None:
        """n_regimes should be n_breaks + 1."""
        y = rng.standard_normal(100)
        test = CUSUMTest(y)
        results = test.fit(significance=0.05)

        assert results.n_regimes == results.n_breaks + 1

    def test_break_dates_alias(self, rng: np.random.Generator) -> None:
        """break_dates should be an alias for break_indices."""
        y = rng.standard_normal(100)
        test = CUSUMTest(y)
        results = test.fit(significance=0.05)

        assert results.break_dates == results.break_indices

    def test_rejected_summary_includes_crossing(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """When rejected, summary should mention crossing index."""
        y, _ = data_with_mean_shift
        test = CUSUMTest(y)
        results = test.fit(significance=0.05)

        if results.reject:
            summary = results.summary()
            assert "REJECT" in summary
            assert "crossing" in summary.lower()


class TestCUSUMEdgeCases:
    """Tests for edge cases."""

    def test_minimum_sample_size(self, rng: np.random.Generator) -> None:
        """Test with minimum feasible sample size."""
        # k=1 (constant), need T > k, so T=3 is minimum
        y = rng.standard_normal(5)
        test = CUSUMTest(y)
        results = test.fit(significance=0.05)

        assert len(results.statistic_path) == 4  # T - k = 5 - 1

    def test_constant_only_model(self, rng: np.random.Generator) -> None:
        """Test with constant-only model (no explicit regressors)."""
        y = rng.standard_normal(100)
        test = CUSUMTest(y)
        results = test.fit(significance=0.05)

        assert results.n_params == 1  # just the constant

    def test_large_sample(self, rng: np.random.Generator) -> None:
        """Test with a larger sample to check performance."""
        y = rng.standard_normal(1000)
        test = CUSUMTest(y)
        results = test.fit(significance=0.05)

        assert len(results.statistic_path) == 999

    def test_invalid_significance(self, rng: np.random.Generator) -> None:
        """CUSUM should reject invalid significance levels."""
        y = rng.standard_normal(100)
        test = CUSUMTest(y)

        with pytest.raises(ValueError, match="significance must be one of"):
            test.fit(significance=0.03)

    def test_cusumsq_invalid_significance(self, rng: np.random.Generator) -> None:
        """CUSUM-SQ should reject out-of-range significance levels."""
        y = rng.standard_normal(100)
        test = CUSUMSQTest(y)

        with pytest.raises(ValueError, match="significance must be in"):
            test.fit(significance=0.0)

        with pytest.raises(ValueError, match="significance must be in"):
            test.fit(significance=1.0)

    def test_cusumsq_any_significance(self, rng: np.random.Generator) -> None:
        """CUSUM-SQ should accept any significance in (0, 1)."""
        y = rng.standard_normal(100)
        test = CUSUMSQTest(y)

        # These should all work
        for sig in [0.001, 0.025, 0.05, 0.10, 0.20, 0.50]:
            results = test.fit(significance=sig)
            assert results.significance == sig

    def test_crossing_indices_empty_when_stable(self, rng: np.random.Generator) -> None:
        """crossing_indices should be empty when stable."""
        y = rng.standard_normal(100)
        test = CUSUMTest(y)
        results = test.fit(significance=0.05)

        if results.is_stable:
            assert len(results.crossing_indices) == 0

    def test_cusumsq_crossing_indices(self, rng: np.random.Generator) -> None:
        """CUSUM-SQ crossing_indices should be consistent with reject."""
        y1 = rng.normal(0, 0.5, 150)
        y2 = rng.normal(0, 5.0, 150)
        y = np.concatenate([y1, y2])

        test = CUSUMSQTest(y)
        results = test.fit(significance=0.05)

        if results.reject:
            assert len(results.crossing_indices) > 0


class TestCUSUMImports:
    """Tests for import accessibility."""

    def test_cusum_test_from_regimes(self) -> None:
        """CUSUMTest should be importable from regimes."""
        assert hasattr(rg, "CUSUMTest")
        assert rg.CUSUMTest is CUSUMTest

    def test_cusum_results_from_regimes(self) -> None:
        """CUSUMResults should be importable from regimes."""
        assert hasattr(rg, "CUSUMResults")
        assert rg.CUSUMResults is CUSUMResults

    def test_cusumsq_test_from_regimes(self) -> None:
        """CUSUMSQTest should be importable from regimes."""
        assert hasattr(rg, "CUSUMSQTest")
        assert rg.CUSUMSQTest is CUSUMSQTest

    def test_cusumsq_results_from_regimes(self) -> None:
        """CUSUMSQResults should be importable from regimes."""
        assert hasattr(rg, "CUSUMSQResults")
        assert rg.CUSUMSQResults is CUSUMSQResults

    def test_plot_cusum_from_regimes(self) -> None:
        """plot_cusum should be importable from regimes."""
        assert hasattr(rg, "plot_cusum")

    def test_plot_cusum_sq_from_regimes(self) -> None:
        """plot_cusum_sq should be importable from regimes."""
        assert hasattr(rg, "plot_cusum_sq")
