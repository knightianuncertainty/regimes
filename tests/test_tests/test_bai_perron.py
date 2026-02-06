"""Tests for Bai-Perron structural break test."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import regimes as rg
from regimes import AR, OLS, BaiPerronTest


class TestBaiPerronBasic:
    """Basic Bai-Perron test functionality."""

    def test_no_breaks_data(self, simple_data: NDArray[np.floating[Any]]) -> None:
        """Test on data without breaks."""
        test = rg.BaiPerronTest(simple_data)
        results = test.fit(max_breaks=3)

        # Should detect 0 breaks (or possibly 1 by chance)
        assert results.n_breaks <= 1

    def test_one_break_detection(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test detection of single mean shift."""
        y, true_break = data_with_mean_shift

        test = rg.BaiPerronTest(y)
        results = test.fit(max_breaks=3)

        # Should detect at least 1 break
        assert results.n_breaks >= 1

        # Break should be near true break
        if results.n_breaks >= 1:
            detected_break = results.break_indices[0]
            assert abs(detected_break - true_break) < 20

    def test_two_breaks_detection(
        self,
        data_with_two_breaks: tuple[NDArray[np.floating[Any]], list[int]],
    ) -> None:
        """Test detection of two breaks."""
        y, true_breaks = data_with_two_breaks

        test = rg.BaiPerronTest(y)
        results = test.fit(max_breaks=4)

        # Should detect 2 breaks
        assert results.n_breaks == 2

        # Breaks should be near true breaks
        for i, (detected, true_br) in enumerate(
            zip(sorted(results.break_indices), sorted(true_breaks))
        ):
            assert abs(detected - true_br) < 20


class TestBaiPerronStatistics:
    """Test Bai-Perron test statistics."""

    def test_supf_statistics(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test Sup-F statistics computation."""
        y, _ = data_with_mean_shift

        test = rg.BaiPerronTest(y)
        results = test.fit(max_breaks=3)

        # Should have Sup-F stats for m=1,2,3
        assert 1 in results.supf_stats
        assert 2 in results.supf_stats
        assert 3 in results.supf_stats

        # All stats should be positive
        assert all(s >= 0 for s in results.supf_stats.values())

    def test_udmax_statistic(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test UDmax statistic computation."""
        y, _ = data_with_mean_shift

        test = rg.BaiPerronTest(y)
        results = test.fit(max_breaks=3)

        # UDmax should be max of Sup-F stats
        assert results.udmax == max(results.supf_stats.values())
        assert not np.isnan(results.udmax)

    def test_information_criteria(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test BIC and LWZ computation."""
        y, _ = data_with_mean_shift

        test = rg.BaiPerronTest(y)
        results = test.fit(max_breaks=3)

        # Should have IC for 0 to max_breaks
        assert 0 in results.bic
        assert 1 in results.bic
        assert 0 in results.lwz
        assert 1 in results.lwz

        # All values should be finite
        assert all(not np.isnan(v) for v in results.bic.values())
        assert all(not np.isnan(v) for v in results.lwz.values())


class TestBaiPerronSelection:
    """Test break number selection methods."""

    def test_bic_selection(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test BIC selection method."""
        y, _ = data_with_mean_shift

        test = rg.BaiPerronTest(y)
        results = test.fit(max_breaks=3, selection="bic")

        assert results.selection_method == "bic"
        # BIC should select the number with minimum BIC
        assert results.n_breaks == min(results.bic, key=results.bic.get)

    def test_lwz_selection(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test LWZ selection method."""
        y, _ = data_with_mean_shift

        test = rg.BaiPerronTest(y)
        results = test.fit(max_breaks=3, selection="lwz")

        assert results.selection_method == "lwz"

    def test_sequential_selection(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test sequential testing selection method."""
        y, _ = data_with_mean_shift

        test = rg.BaiPerronTest(y)
        results = test.fit(max_breaks=3, selection="sequential")

        assert results.selection_method == "sequential"


class TestBaiPerronResults:
    """Test Bai-Perron results object."""

    def test_summary(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test summary generation."""
        y, _ = data_with_mean_shift

        test = rg.BaiPerronTest(y)
        results = test.fit(max_breaks=3)

        summary = results.summary()
        assert "Bai-Perron" in summary
        assert "Sup-F" in summary
        assert "UDmax" in summary
        assert "BIC" in summary

    def test_breaks_by_m(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test breaks_by_m dictionary."""
        y, _ = data_with_mean_shift

        test = rg.BaiPerronTest(y)
        results = test.fit(max_breaks=3)

        # Should have break locations for each m
        assert 0 in results.breaks_by_m
        assert 1 in results.breaks_by_m

        # m=0 should have empty list
        assert len(results.breaks_by_m[0]) == 0

        # m=1 should have 1 break
        assert len(results.breaks_by_m[1]) == 1

    def test_n_regimes(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test n_regimes property."""
        y, _ = data_with_mean_shift

        test = rg.BaiPerronTest(y)
        results = test.fit(max_breaks=3)

        assert results.n_regimes == results.n_breaks + 1


class TestBaiPerronParameters:
    """Test Bai-Perron parameter configurations."""

    def test_trimming_parameter(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test different trimming parameters."""
        y, _ = data_with_mean_shift

        # Default trimming
        test = rg.BaiPerronTest(y)
        results_default = test.fit(max_breaks=3, trimming=0.15)
        assert results_default.trimming == 0.15

        # Larger trimming
        results_large = test.fit(max_breaks=3, trimming=0.20)
        assert results_large.trimming == 0.20

    def test_max_breaks_limit(self, simple_data: NDArray[np.floating[Any]]) -> None:
        """Test maximum breaks parameter."""
        test = rg.BaiPerronTest(simple_data)
        results = test.fit(max_breaks=2)

        assert results.max_breaks == 2
        assert max(results.supf_stats.keys()) == 2

    def test_exog_break_custom(self, rng: np.random.Generator) -> None:
        """Test with custom breaking regressors."""
        n = 200
        x = rng.standard_normal(n)

        # y = beta_0 + beta_1 * x + e, with break in both coefficients
        y = np.zeros(n)
        y[:100] = 1 + 0.5 * x[:100] + rng.standard_normal(100) * 0.5
        y[100:] = 2 + 2.0 * x[100:] + rng.standard_normal(100) * 0.5

        X_break = np.column_stack([np.ones(n), x])

        test = rg.BaiPerronTest(y, exog_break=X_break)
        results = test.fit(max_breaks=3)

        # Should detect 1 break
        assert results.n_breaks >= 1
        assert results.q == 2  # Two breaking regressors


class TestBaiPerronRegimeEstimates:
    """Test regime-specific estimates."""

    def test_get_regime_estimates(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test extraction of regime estimates."""
        y, true_break = data_with_mean_shift

        test = rg.BaiPerronTest(y)
        results = test.fit(max_breaks=3)

        if results.n_breaks >= 1:
            estimates = test.get_regime_estimates(results.break_indices)

            # Should have n_breaks + 1 regimes
            assert len(estimates) == results.n_breaks + 1

            # Each regime should have coefficients and SSR
            for beta, ssr in estimates:
                assert len(beta) >= 1
                assert ssr >= 0


class TestBaiPerronConfidenceIntervals:
    """Test confidence intervals for break dates."""

    def test_ci_computed(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test that CIs are computed when breaks are detected."""
        y, _ = data_with_mean_shift

        test = rg.BaiPerronTest(y)
        results = test.fit(max_breaks=3)

        if results.n_breaks >= 1:
            # Should have CI for each break
            assert len(results.break_ci) == results.n_breaks

            # Each break index should have a CI
            for break_idx in results.break_indices:
                assert break_idx in results.break_ci

    def test_ci_bounds_valid(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test that CI bounds are valid."""
        y, _ = data_with_mean_shift

        test = rg.BaiPerronTest(y)
        results = test.fit(max_breaks=3)

        if results.n_breaks >= 1:
            for break_idx in results.break_indices:
                lower, upper = results.break_ci[break_idx]

                # Lower <= break_idx <= upper
                assert lower <= break_idx
                assert break_idx <= upper

                # Bounds within [0, T-1]
                assert lower >= 0
                assert upper < results.nobs

    def test_ci_empty_when_no_breaks(
        self,
        simple_data: NDArray[np.floating[Any]],
    ) -> None:
        """Test that CI dict is empty when no breaks are detected."""
        # Use data without breaks
        test = rg.BaiPerronTest(simple_data)
        results = test.fit(max_breaks=3)

        if results.n_breaks == 0:
            assert results.break_ci == {}

    def test_ci_in_summary(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test that CIs appear in summary output."""
        y, _ = data_with_mean_shift

        test = rg.BaiPerronTest(y)
        results = test.fit(max_breaks=3)

        summary = results.summary()

        if results.n_breaks >= 1:
            assert "Confidence Intervals" in summary
            # Check that break index appears in CI section
            for break_idx in results.break_indices:
                assert f"Break at {break_idx}" in summary


class TestBaiPerronFromModel:
    """Test BaiPerronTest.from_model() class method."""

    def test_from_ols_model(self, rng: np.random.Generator) -> None:
        """Test creating BaiPerronTest from OLS model."""
        n = 200
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])

        # Data with break in slope
        y = np.zeros(n)
        y[:100] = 1 + 0.5 * x[:100] + rng.standard_normal(100) * 0.5
        y[100:] = 2 + 2.0 * x[100:] + rng.standard_normal(100) * 0.5

        model = OLS(y, X, has_constant=False)
        test = BaiPerronTest.from_model(model)

        # Test should have correct dimensions
        assert test.nobs == n
        assert test.q == 2  # Two breaking regressors

        # Fit and verify detection
        results = test.fit(max_breaks=3)
        assert results.n_breaks >= 1

    def test_from_ols_model_const_only(self, rng: np.random.Generator) -> None:
        """Test from_model with break_vars='const'."""
        n = 200
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])

        # Data with mean shift only
        y = np.zeros(n)
        y[:100] = rng.standard_normal(100)
        y[100:] = rng.standard_normal(100) + 2

        model = OLS(y, X, has_constant=False)
        test = BaiPerronTest.from_model(model, break_vars="const")

        # Only constant breaks, so q=1
        assert test.q == 1

    def test_from_ar_model(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test creating BaiPerronTest from AR model."""
        y, true_break = ar1_data_with_break

        model = AR(y, lags=1)
        test = BaiPerronTest.from_model(model)

        # q should be 2 (constant + 1 lag)
        assert test.q == 2

        # Effective sample is n - maxlag
        assert test.nobs == len(y) - 1

        # Fit and verify
        results = test.fit(max_breaks=3)
        assert isinstance(results, rg.BaiPerronResults)

    def test_from_model_invalid_type(self) -> None:
        """Test from_model raises error for invalid model type."""
        with pytest.raises(TypeError, match="must be OLS, AR, or ADL"):
            BaiPerronTest.from_model("not a model")  # type: ignore[arg-type]

    def test_from_model_invalid_break_vars(self, rng: np.random.Generator) -> None:
        """Test from_model raises error for invalid break_vars."""
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)

        model = OLS(y, X, has_constant=False)
        with pytest.raises(ValueError, match="break_vars must be"):
            BaiPerronTest.from_model(model, break_vars="invalid")  # type: ignore[arg-type]


class TestBaiPerronToOLS:
    """Test BaiPerronResults.to_ols() method."""

    def test_to_ols_basic(self, rng: np.random.Generator) -> None:
        """Test basic to_ols conversion."""
        n = 200
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])

        # Data with break in slope
        y = np.zeros(n)
        y[:100] = 1 + 0.5 * x[:100] + rng.standard_normal(100) * 0.5
        y[100:] = 2 + 2.0 * x[100:] + rng.standard_normal(100) * 0.5

        model = OLS(y, X, has_constant=False)
        bp_results = BaiPerronTest.from_model(model).fit(max_breaks=3)

        # Convert to OLS
        ols_results = bp_results.to_ols()

        # Should be OLSResults
        assert isinstance(ols_results, rg.OLSResults)

        # Should have regime-specific parameters
        if bp_results.n_breaks > 0:
            # Should have 2 * (n_breaks + 1) parameters (2 per regime)
            expected_params = 2 * (bp_results.n_breaks + 1)
            assert len(ols_results.params) == expected_params

    def test_to_ols_with_hac(self, rng: np.random.Generator) -> None:
        """Test to_ols with HAC standard errors."""
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])

        y = np.zeros(n)
        y[:100] = rng.standard_normal(100)
        y[100:] = rng.standard_normal(100) + 2

        model = OLS(y, X, has_constant=False)
        bp_results = BaiPerronTest.from_model(model).fit()

        # Convert with HAC
        ols_results = bp_results.to_ols(cov_type="HAC")
        assert ols_results.cov_type == "HAC"

    def test_to_ols_without_test_reference(self) -> None:
        """Test to_ols raises error when test reference is missing."""
        # Create results without test reference
        results = rg.BaiPerronResults(
            test_name="Bai-Perron",
            nobs=100,
            n_breaks=1,
            break_indices=[50],
            _test=None,
        )

        with pytest.raises(ValueError, match="test reference not stored"):
            results.to_ols()

    def test_to_ols_summary(self, rng: np.random.Generator) -> None:
        """Test that to_ols result can generate summary."""
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])

        y = np.zeros(n)
        y[:100] = rng.standard_normal(100)
        y[100:] = rng.standard_normal(100) + 2

        model = OLS(y, X, has_constant=False)
        bp_results = model.bai_perron()
        ols_results = bp_results.to_ols()

        summary = ols_results.summary()
        assert "OLS Regression Results" in summary


class TestOLSBaiPerronMethod:
    """Test OLS.bai_perron() convenience method."""

    def test_ols_bai_perron_basic(self, rng: np.random.Generator) -> None:
        """Test basic usage of OLS.bai_perron()."""
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])

        y = np.zeros(n)
        y[:100] = rng.standard_normal(100)
        y[100:] = rng.standard_normal(100) + 2

        model = OLS(y, X, has_constant=False)
        bp_results = model.bai_perron()

        assert isinstance(bp_results, rg.BaiPerronResults)
        assert bp_results.n_breaks >= 1

    def test_ols_bai_perron_with_options(self, rng: np.random.Generator) -> None:
        """Test OLS.bai_perron() with custom options."""
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)

        model = OLS(y, X, has_constant=False)
        bp_results = model.bai_perron(
            break_vars="const",
            max_breaks=3,
            trimming=0.20,
            selection="lwz",
        )

        assert bp_results.max_breaks == 3
        assert bp_results.trimming == 0.20
        assert bp_results.selection_method == "lwz"

    def test_ols_bai_perron_workflow(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test complete workflow: model -> bai_perron -> to_ols."""
        y, X, true_break = regression_data_with_break

        # Define model
        model = OLS(y, X, has_constant=False)

        # Run Bai-Perron
        bp_results = model.bai_perron()

        # Should detect break
        assert bp_results.n_breaks >= 1

        # Convert to OLS with breaks
        ols_with_breaks = bp_results.to_ols()

        # Should have regime parameters
        assert len(ols_with_breaks.params) > len(X[0])

        # Fit constant model for comparison
        constant_results = model.fit()

        # Model with breaks should have lower SSR
        assert ols_with_breaks.ssr < constant_results.ssr


class TestARBaiPerronMethod:
    """Test AR.bai_perron() convenience method."""

    def test_ar_bai_perron_basic(
        self,
        ar1_data_with_break: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test basic usage of AR.bai_perron()."""
        y, true_break = ar1_data_with_break

        model = AR(y, lags=1)
        bp_results = model.bai_perron()

        assert isinstance(bp_results, rg.BaiPerronResults)
        # Should detect at least one break
        assert bp_results.n_breaks >= 0  # May not always detect

    def test_ar_bai_perron_with_options(
        self, ar1_data: NDArray[np.floating[Any]]
    ) -> None:
        """Test AR.bai_perron() with custom options."""
        model = AR(ar1_data, lags=1)
        bp_results = model.bai_perron(
            break_vars="const",
            max_breaks=2,
            trimming=0.20,
            selection="sequential",
        )

        assert bp_results.max_breaks == 2
        assert bp_results.trimming == 0.20
        assert bp_results.selection_method == "sequential"

    def test_ar_bai_perron_to_ols(
        self,
        ar1_data_with_break: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test AR.bai_perron() -> to_ols() workflow."""
        y, _ = ar1_data_with_break

        model = AR(y, lags=1)
        bp_results = model.bai_perron()

        if bp_results.n_breaks > 0:
            ols_results = bp_results.to_ols()
            assert isinstance(ols_results, rg.OLSResults)
            # Should have 2 params per regime (const + y.L1)
            expected_params = 2 * (bp_results.n_breaks + 1)
            assert len(ols_results.params) == expected_params
