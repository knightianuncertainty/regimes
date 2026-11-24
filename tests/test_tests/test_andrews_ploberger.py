"""Tests for Andrews-Ploberger structural break test."""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
import pytest
from numpy.typing import NDArray

matplotlib.use("Agg")

import regimes as rg
from regimes import ADL, AR, OLS, AndrewsPlobergerResults, AndrewsPlobergerTest


class TestBasic:
    """Basic Andrews-Ploberger test functionality."""

    def test_no_break_data(self, simple_data: NDArray[np.floating[Any]]) -> None:
        """Test on data without breaks — should not reject."""
        test = AndrewsPlobergerTest(simple_data)
        results = test.fit()

        assert results.n_breaks == 0
        assert len(results.break_indices) == 0

    def test_detect_mean_shift(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test detection of single mean shift."""
        y, true_break = data_with_mean_shift

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        assert results.n_breaks == 1
        assert len(results.break_indices) == 1

    def test_break_location_near_truth(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Estimated break should be near the true break."""
        y, true_break = data_with_mean_shift

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        # Break should be within 20 observations of true location
        assert abs(results.sup_f_break_index - true_break) < 20

    def test_strong_break_rejects(self, rng: np.random.Generator) -> None:
        """A very large mean shift should be detected by all three statistics."""
        y = np.concatenate(
            [
                rng.standard_normal(100),
                rng.standard_normal(100) + 10,
            ]
        )

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        assert results.rejected["SupF"] is True
        assert results.rejected["ExpF"] is True
        assert results.rejected["AveF"] is True

    def test_regression_data_with_break(
        self,
        regression_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]], int
        ],
    ) -> None:
        """Test detection of break in regression coefficients."""
        y, X, true_break = regression_data_with_break

        test = AndrewsPlobergerTest(y, exog_break=X)
        results = test.fit()

        assert results.n_breaks == 1


class TestStatistics:
    """Test Andrews-Ploberger test statistics."""

    def test_sup_f_non_negative(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """SupF should be non-negative."""
        y, _ = data_with_mean_shift

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        assert results.sup_f >= 0

    def test_ave_f_non_negative(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """AveF should be non-negative."""
        y, _ = data_with_mean_shift

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        assert results.ave_f >= 0

    def test_sup_f_geq_ave_f(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """SupF (max) should be >= AveF (mean)."""
        y, _ = data_with_mean_shift

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        assert results.sup_f >= results.ave_f

    def test_large_for_strong_break(self, rng: np.random.Generator) -> None:
        """Statistics should be large for a strong break."""
        y = np.concatenate(
            [
                rng.standard_normal(100),
                rng.standard_normal(100) + 10,
            ]
        )

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        assert results.sup_f > 50

    def test_small_for_stable_data(self, rng: np.random.Generator) -> None:
        """Statistics should be small for stable data."""
        y = rng.standard_normal(200)

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        # SupF should be moderate — not enormous
        assert results.sup_f < 30


class TestFSequence:
    """Test F-statistic sequence properties."""

    def test_length_matches_trimmed_range(self, rng: np.random.Generator) -> None:
        """F-sequence length should match the number of candidate dates."""
        y = rng.standard_normal(200)

        test = AndrewsPlobergerTest(y)
        results = test.fit(trimming=0.15)

        assert len(results.f_sequence) == len(results.candidate_indices)
        assert len(results.f_sequence) > 0

    def test_max_equals_sup_f(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Maximum of F-sequence should equal SupF."""
        y, _ = data_with_mean_shift

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        assert np.isclose(np.max(results.f_sequence), results.sup_f)

    def test_mean_equals_ave_f(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Mean of F-sequence should equal AveF."""
        y, _ = data_with_mean_shift

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        assert np.isclose(np.mean(results.f_sequence), results.ave_f)

    def test_all_non_negative(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """All F-statistics should be non-negative."""
        y, _ = data_with_mean_shift

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        assert np.all(results.f_sequence >= 0)

    def test_candidate_indices_in_range(self, rng: np.random.Generator) -> None:
        """Candidate indices should be within the trimmed range."""
        n = 200
        y = rng.standard_normal(n)
        trimming = 0.15

        test = AndrewsPlobergerTest(y)
        results = test.fit(trimming=trimming)

        min_idx = int(np.floor(n * trimming))
        max_idx = n - min_idx

        assert results.candidate_indices[0] >= min_idx
        assert results.candidate_indices[-1] <= max_idx


class TestCriticalValues:
    """Test critical value lookup."""

    def test_critical_values_populated(self, rng: np.random.Generator) -> None:
        """Critical value dicts should be populated."""
        y = rng.standard_normal(200)

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        for cv_dict in [
            results.sup_f_critical,
            results.exp_f_critical,
            results.ave_f_critical,
        ]:
            assert 0.10 in cv_dict
            assert 0.05 in cv_dict
            assert 0.01 in cv_dict

    def test_critical_values_increasing_with_alpha(
        self, rng: np.random.Generator
    ) -> None:
        """Critical values should increase for stricter significance levels."""
        y = rng.standard_normal(200)

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        for cv_dict in [
            results.sup_f_critical,
            results.exp_f_critical,
            results.ave_f_critical,
        ]:
            assert cv_dict[0.10] < cv_dict[0.05] < cv_dict[0.01]

    def test_critical_values_vary_with_trimming(self, rng: np.random.Generator) -> None:
        """Critical values should differ for different trimming fractions."""
        y = rng.standard_normal(200)

        test = AndrewsPlobergerTest(y)
        r1 = test.fit(trimming=0.10)
        r2 = test.fit(trimming=0.20)

        # CVs should differ
        assert r1.sup_f_critical[0.05] != r2.sup_f_critical[0.05]


class TestPValues:
    """Test p-value computation."""

    def test_pvalues_in_valid_range(self, rng: np.random.Generator) -> None:
        """P-values should be in a valid range."""
        y = rng.standard_normal(200)

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        for pv in [results.sup_f_pvalue, results.exp_f_pvalue, results.ave_f_pvalue]:
            assert 0.0 < pv <= 1.0

    def test_small_pvalues_for_break(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """P-values should be small when there is a break."""
        y, _ = data_with_mean_shift

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        # At least one p-value should be small
        assert (
            min(results.sup_f_pvalue, results.exp_f_pvalue, results.ave_f_pvalue) < 0.10
        )

    def test_large_pvalues_for_no_break(self, rng: np.random.Generator) -> None:
        """P-values should be large when there is no break."""
        y = rng.standard_normal(200)

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        # All p-values should be > 0.05 (most of the time)
        assert results.sup_f_pvalue > 0.01

    def test_pvalues_consistent_with_cvs(self, rng: np.random.Generator) -> None:
        """P-values should be consistent with critical value comparison."""
        y = np.concatenate(
            [
                rng.standard_normal(100),
                rng.standard_normal(100) + 5,
            ]
        )

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        # If SupF exceeds 5% CV, p-value should be < 0.05
        cv05 = results.sup_f_critical[0.05]
        if results.sup_f > cv05:
            assert results.sup_f_pvalue < 0.05


class TestTrimming:
    """Test trimming parameter."""

    def test_default_trimming(self, rng: np.random.Generator) -> None:
        """Default trimming should be 0.15."""
        y = rng.standard_normal(200)

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        assert results.trimming == 0.15

    @pytest.mark.parametrize("trimming", [0.05, 0.10, 0.15, 0.20])
    def test_various_trimming_values(
        self, rng: np.random.Generator, trimming: float
    ) -> None:
        """Test various trimming fractions."""
        y = rng.standard_normal(200)

        test = AndrewsPlobergerTest(y)
        results = test.fit(trimming=trimming)

        assert results.trimming == trimming
        assert len(results.f_sequence) > 0

    def test_narrower_trimming_more_candidates(self, rng: np.random.Generator) -> None:
        """Narrower trimming should produce more candidate dates."""
        y = rng.standard_normal(200)

        test = AndrewsPlobergerTest(y)
        r1 = test.fit(trimming=0.10)
        r2 = test.fit(trimming=0.20)

        assert len(r1.f_sequence) > len(r2.f_sequence)

    def test_invalid_trimming(self, rng: np.random.Generator) -> None:
        """Trimming outside (0, 0.5) should raise."""
        y = rng.standard_normal(200)

        test = AndrewsPlobergerTest(y)
        with pytest.raises(ValueError, match="trimming must be in"):
            test.fit(trimming=0.0)
        with pytest.raises(ValueError, match="trimming must be in"):
            test.fit(trimming=0.5)


class TestResults:
    """Test AndrewsPlobergerResults object."""

    def test_summary_format(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test summary string format."""
        y, _ = data_with_mean_shift

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        summary = results.summary()
        assert "Andrews-Ploberger" in summary
        assert "SupF" in summary
        assert "ExpF" in summary
        assert "AveF" in summary
        assert "CV(5%)" in summary

    def test_rejected_property(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test rejected property returns dict with three keys."""
        y, _ = data_with_mean_shift

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        assert "SupF" in results.rejected
        assert "ExpF" in results.rejected
        assert "AveF" in results.rejected
        assert all(isinstance(v, bool) for v in results.rejected.values())

    def test_n_regimes(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test n_regimes property."""
        y, _ = data_with_mean_shift

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        assert results.n_regimes == results.n_breaks + 1

    def test_break_dates_alias(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test break_dates is alias for break_indices."""
        y, _ = data_with_mean_shift

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        assert results.break_dates == results.break_indices

    def test_test_name(self, rng: np.random.Generator) -> None:
        """Test name should be 'Andrews-Ploberger'."""
        y = rng.standard_normal(200)

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        assert results.test_name == "Andrews-Ploberger"

    def test_q_and_p_stored(self, rng: np.random.Generator) -> None:
        """Test q and p are stored correctly."""
        n = 200
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])
        y = rng.standard_normal(n)

        # All break: q=2, p=0
        test = AndrewsPlobergerTest(y, exog_break=X)
        results = test.fit()
        assert results.q == 2
        assert results.p == 0

        # Partial break: q=1 (constant), p=1 (the slope; the constant in X
        # is the breaking regressor and is not counted twice)
        test2 = AndrewsPlobergerTest(y, exog=X)
        results2 = test2.fit()
        assert results2.q == 1
        assert results2.p == 1


class TestFromModel:
    """Test AndrewsPlobergerTest.from_model() class method."""

    def test_from_ols_model(self, rng: np.random.Generator) -> None:
        """Test creating from OLS model."""
        n = 200
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])
        y = rng.standard_normal(n)

        model = OLS(y, X, has_constant=False)
        test = AndrewsPlobergerTest.from_model(model)

        assert test.nobs == n
        assert test.n_break_params == 2

    def test_from_ar_model(
        self, ar1_data_with_break: tuple[NDArray[np.floating[Any]], int]
    ) -> None:
        """Test creating from AR model."""
        y, _ = ar1_data_with_break

        model = AR(y, lags=1)
        test = AndrewsPlobergerTest.from_model(model)

        assert test.n_break_params == 2  # const + y.L1
        assert test.nobs == len(y) - 1  # effective sample

    def test_from_adl_model(
        self, adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    ) -> None:
        """Test creating from ADL model."""
        y, x = adl_data

        model = ADL(y, x, lags=1, exog_lags=1)
        test = AndrewsPlobergerTest.from_model(model)

        assert test.n_break_params == 4  # const + y.L1 + x.L0 + x.L1

    def test_from_model_const_only(self, rng: np.random.Generator) -> None:
        """Test from_model with break_vars='const'."""
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)

        model = OLS(y, X, has_constant=False)
        test = AndrewsPlobergerTest.from_model(model, break_vars="const")

        assert test.n_break_params == 1
        assert test.n_nonbreak_params == 1

    def test_from_model_invalid_type(self) -> None:
        """Test from_model raises error for invalid model type."""
        with pytest.raises(TypeError, match="must be OLS, AR, or ADL"):
            AndrewsPlobergerTest.from_model("not a model")  # type: ignore[arg-type]

    def test_from_model_invalid_break_vars(self, rng: np.random.Generator) -> None:
        """Test from_model raises error for invalid break_vars."""
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)

        model = OLS(y, X, has_constant=False)
        with pytest.raises(ValueError, match="break_vars must be"):
            AndrewsPlobergerTest.from_model(model, break_vars="invalid")  # type: ignore[arg-type]


class TestConvenienceMethods:
    """Test model.andrews_ploberger() convenience methods."""

    def test_ols_convenience(self, rng: np.random.Generator) -> None:
        """Test OLS.andrews_ploberger() convenience method."""
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = np.zeros(n)
        y[:100] = 1 + 0.5 * X[:100, 1] + rng.standard_normal(100) * 0.5
        y[100:] = 3 + 2.0 * X[100:, 1] + rng.standard_normal(100) * 0.5

        model = OLS(y, X, has_constant=False)
        results = model.andrews_ploberger()

        assert isinstance(results, AndrewsPlobergerResults)
        assert results.n_breaks == 1

    def test_ar_convenience(
        self,
        ar1_data_with_break: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test AR.andrews_ploberger() convenience method."""
        y, _ = ar1_data_with_break

        model = AR(y, lags=1)
        results = model.andrews_ploberger()

        assert isinstance(results, AndrewsPlobergerResults)

    def test_adl_convenience(
        self,
        adl_data: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ) -> None:
        """Test ADL.andrews_ploberger() convenience method."""
        y, x = adl_data

        model = ADL(y, x, lags=1, exog_lags=1)
        results = model.andrews_ploberger()

        assert isinstance(results, AndrewsPlobergerResults)

    def test_convenience_with_options(self, rng: np.random.Generator) -> None:
        """Test convenience method with custom options."""
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)

        model = OLS(y, X, has_constant=False)
        results = model.andrews_ploberger(
            break_vars="const",
            trimming=0.10,
            significance=0.10,
        )

        assert results.trimming == 0.10
        assert results.significance_level == 0.10
        assert results.q == 1  # const only


class TestEdgeCases:
    """Test edge cases."""

    def test_small_sample(self, rng: np.random.Generator) -> None:
        """Test with a small sample."""
        y = np.concatenate(
            [
                rng.standard_normal(30),
                rng.standard_normal(30) + 5,
            ]
        )

        test = AndrewsPlobergerTest(y)
        results = test.fit(trimming=0.15)

        # Should still compute without error
        assert len(results.f_sequence) > 0

    def test_partial_break(self, rng: np.random.Generator) -> None:
        """Test with non-breaking regressors."""
        n = 200
        x = rng.standard_normal(n)
        X_nonbreak = np.column_stack([np.ones(n), x])
        X_break = np.column_stack([np.ones(n)])

        y = np.zeros(n)
        y[:100] = 1 + 2 * x[:100] + rng.standard_normal(100) * 0.5
        y[100:] = 3 + 2 * x[100:] + rng.standard_normal(100) * 0.5

        test = AndrewsPlobergerTest(y, exog=X_nonbreak, exog_break=X_break)
        results = test.fit()

        assert results.q == 1
        assert results.p == 1

    def test_strong_break_all_reject(self, rng: np.random.Generator) -> None:
        """Very strong break should cause all three statistics to reject."""
        y = np.concatenate(
            [
                rng.standard_normal(100),
                rng.standard_normal(100) + 10,
            ]
        )

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        assert all(results.rejected.values())

    def test_constant_only_model(self, rng: np.random.Generator) -> None:
        """Test with default constant-only (mean-shift) model."""
        y = np.concatenate(
            [
                rng.standard_normal(100),
                rng.standard_normal(100) + 3,
            ]
        )

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        assert results.q == 1
        assert results.p == 0


class TestPlot:
    """Test plot functionality."""

    def test_plot_returns_figure_axes(self, rng: np.random.Generator) -> None:
        """Test that plot() returns (Figure, Axes)."""
        import matplotlib.pyplot as plt

        y = rng.standard_normal(200)

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        fig, ax = results.plot()

        assert isinstance(fig, plt.Figure)
        assert ax is not None
        plt.close(fig)

    def test_plot_with_existing_axes(self, rng: np.random.Generator) -> None:
        """Test that plot works with pre-existing axes."""
        import matplotlib.pyplot as plt

        y = rng.standard_normal(200)

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        fig, ax = plt.subplots()
        fig2, ax2 = results.plot(ax=ax)

        assert fig2 is fig
        plt.close(fig)

    def test_plot_f_sequence_function(self, rng: np.random.Generator) -> None:
        """Test the standalone plot_f_sequence function."""
        import matplotlib.pyplot as plt

        y = np.concatenate(
            [
                rng.standard_normal(100),
                rng.standard_normal(100) + 3,
            ]
        )

        test = AndrewsPlobergerTest(y)
        results = test.fit()

        fig, ax = rg.plot_f_sequence(results)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestImports:
    """Test that Andrews-Ploberger is properly accessible."""

    def test_import_from_regimes(self) -> None:
        """Test import from top-level package."""
        assert hasattr(rg, "AndrewsPlobergerTest")
        assert hasattr(rg, "AndrewsPlobergerResults")
        assert hasattr(rg, "plot_f_sequence")

    def test_import_from_tests_module(self) -> None:
        """Test import from tests submodule."""
        from regimes.tests import AndrewsPlobergerResults, AndrewsPlobergerTest

        assert AndrewsPlobergerTest is not None
        assert AndrewsPlobergerResults is not None

    def test_import_from_visualization_module(self) -> None:
        """Test import from visualization submodule."""
        from regimes.visualization import plot_f_sequence

        assert plot_f_sequence is not None

    def test_results_type(
        self,
        data_with_mean_shift: tuple[NDArray[np.floating[Any]], int],
    ) -> None:
        """Test that results are AndrewsPlobergerResults instances."""
        y, _ = data_with_mean_shift

        test = rg.AndrewsPlobergerTest(y)
        results = test.fit()

        assert isinstance(results, rg.AndrewsPlobergerResults)
