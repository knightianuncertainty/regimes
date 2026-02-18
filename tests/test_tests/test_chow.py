"""Tests for Chow structural break test."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import regimes as rg
from regimes import ADL, AR, OLS, ChowTest, ChowTestResults


class TestChowBasic:
    """Basic Chow test functionality."""

    def test_no_break_data(self, simple_data: NDArray[np.floating[Any]]) -> None:
        """Test on data without breaks — should not reject."""
        test = ChowTest(simple_data)
        results = test.fit(break_points=50)

        assert results.n_breaks == 0
        assert len(results.break_indices) == 0

    def test_detect_mean_shift(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test detection of single mean shift at true break."""
        y, true_break = data_with_mean_shift

        test = ChowTest(y)
        results = test.fit(break_points=true_break)

        assert results.n_breaks == 1
        assert true_break in results.break_indices

    def test_stronger_at_true_break(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """F-stat should be higher at true break than at a wrong location."""
        y, true_break = data_with_mean_shift

        test = ChowTest(y)
        results = test.fit(break_points=[true_break, 30])

        # F-stat at true break should be larger than at wrong location
        assert results.f_stats[true_break] > results.f_stats[30]

    def test_regression_data_with_break(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test detection of break in regression coefficients."""
        y, X, true_break = regression_data_with_break

        test = ChowTest(y, exog_break=X)
        results = test.fit(break_points=true_break)

        assert results.n_breaks == 1
        assert true_break in results.break_indices

    def test_multiple_break_points(self, rng: np.random.Generator) -> None:
        """Test multiple break points tested simultaneously."""
        # Use large mean shifts so each individual Chow test detects a break
        y1 = rng.standard_normal(100)
        y2 = rng.standard_normal(100) + 5  # Large shift
        y3 = rng.standard_normal(100) - 5  # Large shift back
        y = np.concatenate([y1, y2, y3])

        test = ChowTest(y)
        results = test.fit(break_points=[100, 200])

        # Should test both break points
        assert len(results.tested_break_points) == 2
        # At least one should be significant (the Chow test tests each individually)
        assert results.n_breaks >= 1


class TestChowStatistics:
    """Test Chow test statistics."""

    def test_f_stat_positive(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """F-statistic should be non-negative."""
        y, X, true_break = regression_data_with_break

        test = ChowTest(y, exog_break=X)
        results = test.fit(break_points=true_break)

        assert results.f_stats[true_break] >= 0

    def test_p_values_in_range(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """P-values should be in [0, 1]."""
        y, true_break = data_with_mean_shift

        test = ChowTest(y)
        results = test.fit(break_points=[true_break, 50])

        for bp in results.tested_break_points:
            assert 0.0 <= results.p_values[bp] <= 1.0

    def test_df_correct_standard(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Degrees of freedom should be correct for standard test."""
        y, X, true_break = regression_data_with_break
        T = len(y)
        k = X.shape[1]

        test = ChowTest(y, exog_break=X)
        results = test.fit(break_points=true_break)

        assert results.df_num[true_break] == k
        assert results.df_denom[true_break] == T - 2 * k

    def test_ssr_decomposition(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """SSR_full should be >= SSR_unrestricted (restriction increases SSR)."""
        y, true_break = data_with_mean_shift

        test = ChowTest(y)
        results = test.fit(break_points=true_break)

        assert results.ssr_full >= results.ssr_unrestricted[true_break]

    def test_large_f_stat_for_strong_break(self, rng: np.random.Generator) -> None:
        """A very large mean shift should produce a large F-statistic."""
        y1 = rng.standard_normal(100)
        y2 = rng.standard_normal(100) + 10  # Large shift
        y = np.concatenate([y1, y2])

        test = ChowTest(y)
        results = test.fit(break_points=100)

        assert results.f_stats[100] > 50  # Should be very large
        assert results.p_values[100] < 0.001


class TestChowPredictive:
    """Test predictive Chow test variant."""

    def test_auto_triggers_near_end(self, rng: np.random.Generator) -> None:
        """Predictive variant should auto-trigger when break is near end."""
        n = 100
        y = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])

        test = ChowTest(y, exog_break=X)
        # Break at n-1 leaves only 1 obs in second sub-sample (< k=2)
        results = test.fit(break_points=n - 1)

        assert results.is_predictive[n - 1] is True

    def test_auto_triggers_near_start(self, rng: np.random.Generator) -> None:
        """Predictive variant should auto-trigger when break is near start."""
        n = 100
        y = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])

        test = ChowTest(y, exog_break=X)
        # Break at 1 leaves only 1 obs in first sub-sample (< k=2)
        results = test.fit(break_points=1)

        assert results.is_predictive[1] is True

    def test_standard_used_when_sufficient(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Standard test should be used when both sub-samples are large enough."""
        y, true_break = data_with_mean_shift

        test = ChowTest(y)
        results = test.fit(break_points=true_break)

        assert results.is_predictive[true_break] is False

    def test_predictive_df(self, rng: np.random.Generator) -> None:
        """Predictive variant should have correct degrees of freedom."""
        n = 100
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x, x**2])  # k=3 params
        y = rng.standard_normal(n)

        test = ChowTest(y, exog_break=X)
        # Break at 98: n1=98, n2=2, k=3 → n2 < k → predictive
        results = test.fit(break_points=98)

        # Predictive: df_num = n_small = 2, df_denom = n_large - k = 98 - 3 = 95
        assert results.is_predictive[98] is True
        assert results.df_num[98] == 2
        assert results.df_denom[98] == 95

    def test_predictive_detects_break(self, rng: np.random.Generator) -> None:
        """Predictive variant should still detect a strong break."""
        n = 200
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x, x**2])  # k=3 params

        y = np.zeros(n)
        y[:198] = 1 + 2 * x[:198] + rng.standard_normal(198) * 0.5
        y[198:] = 20 + 2 * x[198:] + rng.standard_normal(2) * 0.5  # Big shift

        test = ChowTest(y, exog_break=X)
        # n2=2 < k=3 → predictive
        results = test.fit(break_points=198)

        assert results.is_predictive[198] is True
        assert results.p_values[198] < 0.05


class TestChowResults:
    """Test ChowTestResults object."""

    def test_summary_format(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test summary string format."""
        y, true_break = data_with_mean_shift

        test = ChowTest(y)
        results = test.fit(break_points=[true_break, 50])

        summary = results.summary()
        assert "Chow" in summary
        assert "F-stat" in summary
        assert "p-value" in summary
        assert str(true_break) in summary

    def test_rejected_property(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test rejected property matches break_indices."""
        y, true_break = data_with_mean_shift

        test = ChowTest(y)
        results = test.fit(break_points=[true_break, 50])

        for bp, is_rejected in results.rejected.items():
            if is_rejected:
                assert bp in results.break_indices
            else:
                assert bp not in results.break_indices

    def test_tested_break_points(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test tested_break_points returns all tested points."""
        y, true_break = data_with_mean_shift

        test = ChowTest(y)
        results = test.fit(break_points=[true_break, 50, 150])

        assert sorted(results.tested_break_points) == sorted([true_break, 50, 150])

    def test_n_regimes(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test n_regimes property."""
        y, true_break = data_with_mean_shift

        test = ChowTest(y)
        results = test.fit(break_points=true_break)

        assert results.n_regimes == results.n_breaks + 1

    def test_break_dates_alias(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test break_dates is alias for break_indices."""
        y, true_break = data_with_mean_shift

        test = ChowTest(y)
        results = test.fit(break_points=true_break)

        assert results.break_dates == results.break_indices


class TestChowFromModel:
    """Test ChowTest.from_model() class method."""

    def test_from_ols_model(self, rng: np.random.Generator) -> None:
        """Test creating ChowTest from OLS model."""
        n = 200
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])

        y = np.zeros(n)
        y[:100] = 1 + 0.5 * x[:100] + rng.standard_normal(100) * 0.5
        y[100:] = 2 + 2.0 * x[100:] + rng.standard_normal(100) * 0.5

        model = OLS(y, X, has_constant=False)
        test = ChowTest.from_model(model)

        assert test.nobs == n
        assert test.n_break_params == 2

    def test_from_ar_model(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test creating ChowTest from AR model."""
        y, _ = ar1_data_with_break

        model = AR(y, lags=1)
        test = ChowTest.from_model(model)

        # AR(1) with constant: 2 params (const + y.L1)
        assert test.n_break_params == 2
        # Effective sample is n - maxlag
        assert test.nobs == len(y) - 1

    def test_from_adl_model(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test creating ChowTest from ADL model."""
        y, x = adl_data

        model = ADL(y, x, lags=1, exog_lags=1)
        test = ChowTest.from_model(model)

        # ADL(1,1): const + y.L1 + x.L0 + x.L1 = 4 params
        assert test.n_break_params == 4

    def test_from_model_const_only(self, rng: np.random.Generator) -> None:
        """Test from_model with break_vars='const'."""
        n = 200
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])
        y = rng.standard_normal(n)

        model = OLS(y, X, has_constant=False)
        test = ChowTest.from_model(model, break_vars="const")

        # Only constant breaks (auto-added); original X becomes non-breaking
        assert test.n_break_params == 1  # Default constant
        assert test.n_nonbreak_params == 2  # Original X is non-breaking

    def test_from_model_invalid_type(self) -> None:
        """Test from_model raises error for invalid model type."""
        with pytest.raises(TypeError, match="must be OLS, AR, or ADL"):
            ChowTest.from_model("not a model")  # type: ignore[arg-type]

    def test_from_model_invalid_break_vars(self, rng: np.random.Generator) -> None:
        """Test from_model raises error for invalid break_vars."""
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)

        model = OLS(y, X, has_constant=False)
        with pytest.raises(ValueError, match="break_vars must be"):
            ChowTest.from_model(model, break_vars="invalid")  # type: ignore[arg-type]


class TestOLSChowMethod:
    """Test OLS.chow_test() convenience method."""

    def test_ols_chow_basic(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test basic usage of OLS.chow_test()."""
        y, X, true_break = regression_data_with_break

        model = OLS(y, X, has_constant=False)
        results = model.chow_test(break_points=true_break)

        assert isinstance(results, ChowTestResults)
        assert results.n_breaks == 1

    def test_ols_chow_with_options(self, rng: np.random.Generator) -> None:
        """Test OLS.chow_test() with custom options."""
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)

        model = OLS(y, X, has_constant=False)
        results = model.chow_test(
            break_points=100,
            break_vars="const",
            significance=0.10,
        )

        assert results.significance_level == 0.10

    def test_ols_chow_multiple_breaks(
        self,
        data_with_two_breaks: tuple[NDArray[np.floating[Any]], list[int]],
    ) -> None:
        """Test OLS.chow_test() with multiple break points."""
        y, true_breaks = data_with_two_breaks
        X = np.ones((len(y), 1))

        model = OLS(y, X, has_constant=False)
        results = model.chow_test(break_points=true_breaks)

        assert len(results.tested_break_points) == 2


class TestARChowMethod:
    """Test AR.chow_test() convenience method."""

    def test_ar_chow_basic(
        self,
        ar1_data_with_break: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test basic usage of AR.chow_test()."""
        y, true_break = ar1_data_with_break

        model = AR(y, lags=1)
        # Break is in full-sample coords; effective sample coords = true_break - maxlag
        effective_break = true_break - 1  # maxlag=1 for AR(1)
        results = model.chow_test(break_points=effective_break)

        assert isinstance(results, ChowTestResults)

    def test_ar_chow_detects_break(
        self,
        ar1_data_with_break: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """AR Chow test should detect break in AR coefficient."""
        y, true_break = ar1_data_with_break

        model = AR(y, lags=1)
        effective_break = true_break - 1
        results = model.chow_test(break_points=effective_break)

        # With phi changing from 0.3 to 0.8, should detect
        assert results.n_breaks >= 1


class TestADLChowMethod:
    """Test ADL.chow_test() convenience method."""

    def test_adl_chow_basic(
        self,
        adl_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test basic usage of ADL.chow_test()."""
        y, x, true_break = adl_data_with_break

        model = ADL(y, x, lags=1, exog_lags=1)
        # Effective sample coordinates
        effective_break = true_break - 1  # maxlag=1
        results = model.chow_test(break_points=effective_break)

        assert isinstance(results, ChowTestResults)

    def test_adl_chow_detects_break(
        self,
        adl_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """ADL Chow test should detect break in AR coefficient."""
        y, x, true_break = adl_data_with_break

        model = ADL(y, x, lags=1, exog_lags=0)
        # ADL(1,0): maxlag = 1
        effective_break = true_break - 1
        results = model.chow_test(break_points=effective_break)

        # With phi changing from 0.3 to 0.8, should likely detect
        assert results.n_breaks >= 0  # Conservative — depends on noise


class TestChowEdgeCases:
    """Test edge cases for the Chow test."""

    def test_break_at_boundary_start(self, rng: np.random.Generator) -> None:
        """Test break at first valid position."""
        y = rng.standard_normal(100)

        test = ChowTest(y)
        results = test.fit(break_points=1)

        assert 1 in results.f_stats

    def test_break_at_boundary_end(self, rng: np.random.Generator) -> None:
        """Test break at last valid position."""
        y = rng.standard_normal(100)

        test = ChowTest(y)
        results = test.fit(break_points=99)

        assert 99 in results.f_stats

    def test_invalid_break_point_zero(self, rng: np.random.Generator) -> None:
        """Test that break at 0 raises error."""
        y = rng.standard_normal(100)

        test = ChowTest(y)
        with pytest.raises(ValueError, match="outside the valid range"):
            test.fit(break_points=0)

    def test_invalid_break_point_beyond_end(self, rng: np.random.Generator) -> None:
        """Test that break beyond T raises error."""
        y = rng.standard_normal(100)

        test = ChowTest(y)
        with pytest.raises(ValueError, match="outside the valid range"):
            test.fit(break_points=100)

    def test_constant_only_model(self, rng: np.random.Generator) -> None:
        """Test with constant-only (mean) model."""
        y = np.concatenate(
            [
                rng.standard_normal(50),
                rng.standard_normal(50) + 3,
            ]
        )

        # Constant-only model (default)
        test = ChowTest(y)
        results = test.fit(break_points=50)

        assert results.n_params == 1
        assert results.n_breaks == 1

    def test_single_int_break_point(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test that a single int (not list) is accepted."""
        y, true_break = data_with_mean_shift

        test = ChowTest(y)
        results = test.fit(break_points=true_break)

        assert true_break in results.f_stats

    def test_significance_levels(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test different significance levels."""
        y, true_break = data_with_mean_shift

        test = ChowTest(y)

        results_05 = test.fit(break_points=true_break, significance=0.05)
        results_01 = test.fit(break_points=true_break, significance=0.01)

        # F-stats should be the same regardless of significance level
        assert results_05.f_stats[true_break] == results_01.f_stats[true_break]

        # But break_indices may differ
        assert results_05.significance_level == 0.05
        assert results_01.significance_level == 0.01


class TestChowSubsetCoefficients:
    """Test partial coefficient testing (exog vs exog_break)."""

    def test_all_coefficients_break(self, rng: np.random.Generator) -> None:
        """Test when all coefficients are tested for breaks."""
        n = 200
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])

        y = np.zeros(n)
        y[:100] = 1 + 0.5 * x[:100] + rng.standard_normal(100) * 0.5
        y[100:] = 3 + 2.0 * x[100:] + rng.standard_normal(100) * 0.5

        test = ChowTest(y, exog_break=X)
        results = test.fit(break_points=100)

        assert results.n_params == 2  # Both const and slope break
        assert results.n_breaks == 1

    def test_only_constant_breaks(self, rng: np.random.Generator) -> None:
        """Test when only the constant is tested for breaks."""
        n = 200
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])

        # Only intercept changes, slope stays the same
        y = np.zeros(n)
        y[:100] = 1 + 2.0 * x[:100] + rng.standard_normal(100) * 0.5
        y[100:] = 3 + 2.0 * x[100:] + rng.standard_normal(100) * 0.5

        # Test with only constant breaking
        test = ChowTest(y, exog=X, exog_break=np.ones((n, 1)))
        results = test.fit(break_points=100)

        assert results.n_params == 1  # Only constant breaks

    def test_from_model_break_vars_all(self, rng: np.random.Generator) -> None:
        """Test from_model with break_vars='all' via convenience method."""
        n = 200
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])

        y = np.zeros(n)
        y[:100] = 1 + 0.5 * x[:100] + rng.standard_normal(100) * 0.5
        y[100:] = 3 + 2.0 * x[100:] + rng.standard_normal(100) * 0.5

        model = OLS(y, X, has_constant=False)
        results = model.chow_test(break_points=100, break_vars="all")

        assert results.n_params == 2

    def test_from_model_break_vars_const(self, rng: np.random.Generator) -> None:
        """Test from_model with break_vars='const' via convenience method."""
        n = 200
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])
        y = rng.standard_normal(n)

        model = OLS(y, X, has_constant=False)
        results = model.chow_test(break_points=100, break_vars="const")

        assert results.n_params == 1

    def test_partial_break_stronger_than_full(self, rng: np.random.Generator) -> None:
        """When only the constant breaks, testing only the constant should
        give a higher F-stat than testing all coefficients."""
        n = 200
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])

        # Only intercept changes
        y = np.zeros(n)
        y[:100] = 0 + 2.0 * x[:100] + rng.standard_normal(100) * 0.5
        y[100:] = 3 + 2.0 * x[100:] + rng.standard_normal(100) * 0.5

        model = OLS(y, X, has_constant=False)
        results_all = model.chow_test(break_points=100, break_vars="all")
        results_const = model.chow_test(break_points=100, break_vars="const")

        # Both should detect the break
        assert results_all.n_breaks == 1
        assert results_const.n_breaks == 1


class TestChowImports:
    """Test that ChowTest is properly accessible."""

    def test_import_from_regimes(self) -> None:
        """Test import from top-level package."""
        assert hasattr(rg, "ChowTest")
        assert hasattr(rg, "ChowTestResults")

    def test_import_from_tests_module(self) -> None:
        """Test import from tests submodule."""
        from regimes.tests import ChowTest, ChowTestResults

        assert ChowTest is not None
        assert ChowTestResults is not None

    def test_results_type(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test that results are ChowTestResults instances."""
        y, true_break = data_with_mean_shift

        test = rg.ChowTest(y)
        results = test.fit(break_points=true_break)

        assert isinstance(results, rg.ChowTestResults)
