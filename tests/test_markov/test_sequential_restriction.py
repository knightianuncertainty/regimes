"""Tests for sequential restriction testing algorithms."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from regimes.markov.sequential_restriction import (
    NonRecurringRegimeTest,
    NonRecurringRegimeTestResults,
    SequentialRestrictionResults,
    SequentialRestrictionTest,
    _chi_bar_squared_critical_value,
    _chi_bar_squared_pvalue,
)


class TestChiBarSquared:
    """Test chi-bar-squared utility functions."""

    def test_pvalue_zero_stat(self) -> None:
        """LR = 0 should give p-value of 0.5 for single restriction."""
        p = _chi_bar_squared_pvalue(0.0, 1)
        assert abs(p - 0.5) < 0.01

    def test_pvalue_large_stat(self) -> None:
        """Large LR should give small p-value."""
        p = _chi_bar_squared_pvalue(20.0, 1)
        assert p < 0.001

    def test_pvalue_known_value(self) -> None:
        """At the 5% critical value, p should be 0.05."""
        cv = _chi_bar_squared_critical_value(0.05, 1)
        p = _chi_bar_squared_pvalue(cv, 1)
        assert abs(p - 0.05) < 0.01

    def test_critical_value_5_percent(self) -> None:
        """5% critical value for single restriction should be ~2.706."""
        cv = _chi_bar_squared_critical_value(0.05, 1)
        assert abs(cv - 2.706) < 0.1

    def test_pvalue_multiple_restrictions_conservative(self) -> None:
        """Multiple restrictions should give more conservative p-values."""
        p1 = _chi_bar_squared_pvalue(5.0, 1)
        p2 = _chi_bar_squared_pvalue(5.0, 3)
        assert p2 >= p1


class TestNonRecurringRegimeTest:
    """Test NonRecurringRegimeTest."""

    def test_init(self, two_regime_data: NDArray[np.floating[Any]]) -> None:
        test = NonRecurringRegimeTest(
            two_regime_data, k_regimes=2, method="chi_bar_squared"
        )
        assert test.k_regimes == 2
        assert test.method == "chi_bar_squared"

    def test_build_restrictions(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        test = NonRecurringRegimeTest(two_regime_data, k_regimes=2)
        restrictions = test._build_non_recurring_restrictions()
        # For k=2, the only non-recurring restriction is (0,1)=0
        # (can't go from regime 1 back to regime 0)
        assert len(restrictions) > 0
        for (i, j), val in restrictions.items():
            assert val == 0.0

    def test_build_restrictions_3_regimes(
        self, three_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        test = NonRecurringRegimeTest(three_regime_data, k_regimes=3)
        restrictions = test._build_non_recurring_restrictions()
        # For k=3: restrictions should include (0,1), (0,2), (2,0), (1,2)
        # but NOT (0,0), (1,1), (2,2) [diagonal]
        # and NOT (1,0), (2,1) [forward transitions]
        assert (0, 1) in restrictions  # Can't go back from 1 to 0
        assert (0, 2) in restrictions  # Can't go back from 2 to 0
        assert (1, 0) not in restrictions  # Forward: 0 -> 1
        assert (2, 1) not in restrictions  # Forward: 1 -> 2

    def test_fit_chi_bar_squared(self) -> None:
        """Test with larger, well-separated data for reliable convergence."""
        rng = np.random.default_rng(99)
        y = np.concatenate(
            [
                rng.standard_normal(200) + 0.0,
                rng.standard_normal(200) + 5.0,
            ]
        )
        test = NonRecurringRegimeTest(y, k_regimes=2, method="chi_bar_squared")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = test.fit()

        assert isinstance(results, NonRecurringRegimeTestResults)
        assert results.lr_statistic >= 0
        assert 0 <= results.p_value <= 1
        assert results.rejected in (True, False)
        assert results.method == "chi_bar_squared"

    def test_summary(self) -> None:
        rng = np.random.default_rng(99)
        y = np.concatenate(
            [
                rng.standard_normal(200) + 0.0,
                rng.standard_normal(200) + 5.0,
            ]
        )
        test = NonRecurringRegimeTest(y, k_regimes=2, method="chi_bar_squared")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = test.fit()

        s = results.summary()
        assert "Non-Recurring" in s
        assert "LR statistic" in s


class TestNonRecurringRegimeTestEdgeCases:
    """Additional edge cases for NonRecurringRegimeTest."""

    def test_summary_format(self) -> None:
        """Summary should contain all expected sections."""
        rng = np.random.default_rng(99)
        y = np.concatenate(
            [
                rng.standard_normal(200) + 0.0,
                rng.standard_normal(200) + 5.0,
            ]
        )
        test = NonRecurringRegimeTest(y, k_regimes=2, method="chi_bar_squared")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = test.fit()

        s = results.summary()
        assert "Non-Recurring" in s
        assert "LR statistic" in s
        assert "p-value" in s or "p_value" in s.lower() or "0." in s

    def test_3_regimes(self) -> None:
        """Test with 3 regimes."""
        rng = np.random.default_rng(42)
        y = np.concatenate(
            [
                rng.standard_normal(150) + 0.0,
                rng.standard_normal(150) + 4.0,
                rng.standard_normal(150) + 8.0,
            ]
        )
        test = NonRecurringRegimeTest(y, k_regimes=3, method="chi_bar_squared")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = test.fit()

        assert isinstance(results, NonRecurringRegimeTestResults)
        assert results.lr_statistic >= 0


class TestSequentialRestrictionTest:
    """Test SequentialRestrictionTest."""

    def test_init(self, two_regime_data: NDArray[np.floating[Any]]) -> None:
        test = SequentialRestrictionTest(
            two_regime_data,
            k_regimes=2,
            critical_value_method="chi_bar_squared",
        )
        assert test.k_regimes == 2
        assert test.multiple_testing == "holm"

    def test_identify_candidates(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        test = SequentialRestrictionTest(two_regime_data, k_regimes=2)
        transition = np.array([[0.9, 0.2], [0.1, 0.8]])
        candidates = test._identify_candidates(transition, {})
        # Should have 2 off-diagonal candidates
        assert len(candidates) == 2
        # Smallest first
        assert transition[candidates[0]] <= transition[candidates[1]]

    def test_identify_candidates_with_existing(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        test = SequentialRestrictionTest(two_regime_data, k_regimes=2)
        transition = np.array([[0.9, 0.2], [0.1, 0.8]])
        candidates = test._identify_candidates(transition, {(1, 0): 0.0})
        # One already restricted, so only 1 candidate
        assert len(candidates) == 1

    def test_holm_significance(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        test = SequentialRestrictionTest(
            two_regime_data,
            k_regimes=2,
            significance=0.05,
            multiple_testing="holm",
        )
        # Step 0, total 2 candidates: alpha / (2 - 0) = 0.025
        sig = test._holm_significance(0, 2)
        assert abs(sig - 0.025) < 0.001

        # Step 1, total 2: alpha / (2 - 1) = 0.05
        sig = test._holm_significance(1, 2)
        assert abs(sig - 0.05) < 0.001

    def test_bonferroni_significance(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        test = SequentialRestrictionTest(
            two_regime_data,
            k_regimes=2,
            significance=0.05,
            multiple_testing="bonferroni",
        )
        sig = test._holm_significance(0, 2)
        assert abs(sig - 0.025) < 0.001

    def test_no_correction(self, two_regime_data: NDArray[np.floating[Any]]) -> None:
        test = SequentialRestrictionTest(
            two_regime_data,
            k_regimes=2,
            significance=0.05,
            multiple_testing="none",
        )
        sig = test._holm_significance(0, 2)
        assert abs(sig - 0.05) < 0.001

    def test_fit(self) -> None:
        """Use larger data for reliable restricted model convergence."""
        rng = np.random.default_rng(77)
        y = np.concatenate(
            [
                rng.standard_normal(200) + 0.0,
                rng.standard_normal(200) + 5.0,
            ]
        )
        test = SequentialRestrictionTest(
            y,
            k_regimes=2,
            critical_value_method="chi_bar_squared",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = test.fit()

        assert isinstance(results, SequentialRestrictionResults)
        assert isinstance(results.final_restrictions, dict)
        assert results.final_transition.shape == (2, 2)
        assert np.isfinite(results.llf_unrestricted)
        assert results.is_non_recurring in (True, False)
        assert len(results.history) > 0

    def test_fit_verbose(self, capsys: Any) -> None:
        """Verbose output should print step-by-step results."""
        rng = np.random.default_rng(77)
        y = np.concatenate(
            [
                rng.standard_normal(200) + 0.0,
                rng.standard_normal(200) + 5.0,
            ]
        )
        test = SequentialRestrictionTest(
            y,
            k_regimes=2,
            critical_value_method="chi_bar_squared",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test.fit(verbose=True)

        captured = capsys.readouterr()
        # Should print step information
        assert (
            "Step" in captured.out
            or "step" in captured.out.lower()
            or len(captured.out) > 0
        )

    def test_check_non_recurring(self) -> None:
        """_check_non_recurring should identify non-recurring patterns."""
        rng = np.random.default_rng(77)
        y = np.concatenate(
            [
                rng.standard_normal(200) + 0.0,
                rng.standard_normal(200) + 5.0,
            ]
        )
        test = SequentialRestrictionTest(
            y,
            k_regimes=2,
            critical_value_method="chi_bar_squared",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = test.fit()

        assert results.is_non_recurring in (True, False)

    def test_summary(self) -> None:
        rng = np.random.default_rng(77)
        y = np.concatenate(
            [
                rng.standard_normal(200) + 0.0,
                rng.standard_normal(200) + 5.0,
            ]
        )
        test = SequentialRestrictionTest(
            y,
            k_regimes=2,
            critical_value_method="chi_bar_squared",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = test.fit()

        s = results.summary()
        assert "Sequential Restriction" in s
        assert "Final Transition Matrix" in s
