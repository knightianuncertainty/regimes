"""Tests for restricted Markov switching models."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from regimes.markov.restricted import RestrictedMarkovAR, RestrictedMarkovRegression


class TestRestrictedMarkovRegression:
    """Test RestrictedMarkovRegression."""

    def test_basic_restriction(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        """Test that a simple restriction is enforced."""
        restrictions = {(0, 1): 0.0}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = RestrictedMarkovRegression(
                two_regime_data, k_regimes=2, restrictions=restrictions
            )
            results = model.fit(search_reps=5)

        # The restricted entry should be exactly 0
        assert results.regime_transition[0, 1] == 0.0
        assert results.restricted_transitions == restrictions
        assert np.isfinite(results.llf), "Restricted model should produce finite llf"

    def test_non_recurring_factory(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        """Test the non_recurring() factory method."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = RestrictedMarkovRegression.non_recurring(
                two_regime_data, k_regimes=2
            )
            results = model.fit(search_reps=5)

        assert isinstance(results.restricted_transitions, dict)
        assert results.k_regimes == 2
        assert np.isfinite(results.llf), "Non-recurring model should produce finite llf"

    def test_non_recurring_3_regimes(
        self, three_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        """Test non-recurring structure with 3 regimes."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = RestrictedMarkovRegression.non_recurring(
                three_regime_data, k_regimes=3
            )

        # Check that restrictions are correctly set
        expected_zeros = {(2, 0), (0, 2)}  # Can't skip or go back
        for pos in expected_zeros:
            assert pos in model.restrictions
            assert model.restrictions[pos] == 0.0

    def test_restricted_llf_leq_unrestricted(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        """Restricted model should have lower (or equal) log-likelihood."""
        from regimes.markov import MarkovRegression

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Unrestricted
            u_model = MarkovRegression(two_regime_data, k_regimes=2, ordering=None)
            u_results = u_model.fit(search_reps=5)

            # Restricted
            r_model = RestrictedMarkovRegression(
                two_regime_data,
                k_regimes=2,
                restrictions={(0, 1): 0.0},
                ordering=None,
            )
            r_results = r_model.fit(search_reps=5)

        assert np.isfinite(u_results.llf), (
            "Unrestricted model should produce finite llf"
        )
        assert np.isfinite(r_results.llf), "Restricted model should produce finite llf"
        # Restricted should be <= unrestricted (with small tolerance)
        assert r_results.llf <= u_results.llf + 1.0

    def test_result_type(self, two_regime_data: NDArray[np.floating[Any]]) -> None:
        from regimes.markov.results import MarkovRegressionResults

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = RestrictedMarkovRegression(
                two_regime_data,
                k_regimes=2,
                restrictions={(0, 1): 0.0},
            )
            results = model.fit(search_reps=5)

        assert isinstance(results, MarkovRegressionResults)


class TestNonRecurringSmoothedProbabilities:
    """Tests verifying that smoothed probabilities respect transition restrictions.

    These tests target the probability leakage bug where statsmodels'
    log(max(0, 1e-20)) ≈ -46 allows forbidden transitions to accumulate
    over many time steps.
    """

    @staticmethod
    def _recurring_data(rng: np.random.Generator) -> NDArray[np.floating[Any]]:
        """Data that goes 0→1→0 (recurring pattern)."""
        return np.concatenate(
            [
                rng.standard_normal(100) + 0.0,
                rng.standard_normal(100) + 5.0,
                rng.standard_normal(100) + 0.0,
            ]
        )

    def test_non_recurring_smoothed_probs_monotonic(
        self, rng: np.random.Generator
    ) -> None:
        """Once smoothed P(regime 1) > 0.99, it stays > 0.99 for all t."""
        y = self._recurring_data(rng)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = RestrictedMarkovRegression.non_recurring(y, k_regimes=2)
            results = model.fit(search_reps=5)

        probs = results.smoothed_marginal_probabilities
        assert np.isfinite(results.llf), "Model should produce finite llf"

        # Find regime 1 (the high-mean regime)
        # Determine which column corresponds to the higher mean
        # Identify the absorbing (high-mean) regime by its constant
        consts = {r: results.regime_params[r]["const"] for r in range(2)}
        high_regime = max(consts, key=consts.get)

        p_high = probs[:, high_regime]

        # Find the first time P(high regime) > 0.99
        above = np.where(p_high > 0.99)[0]
        if len(above) > 0:
            first_above = above[0]
            # All subsequent probabilities should stay > 0.99
            assert np.all(p_high[first_above:] > 0.99), (
                f"Smoothed P(regime {high_regime}) dropped below 0.99 after "
                f"first exceeding it at t={first_above}. "
                f"Min after: {p_high[first_above:].min():.6f}"
            )

    def test_non_recurring_filtered_probs_absorbing(
        self, rng: np.random.Generator
    ) -> None:
        """After transition, filtered P(absorbing regime) stays near 1."""
        y = self._recurring_data(rng)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = RestrictedMarkovRegression.non_recurring(y, k_regimes=2)
            results = model.fit(search_reps=5)

        probs = results.filtered_marginal_probabilities
        assert np.isfinite(results.llf)

        consts = {r: results.regime_params[r]["const"] for r in range(2)}
        high_regime = max(consts, key=consts.get)
        p_high = probs[:, high_regime]

        # At t=200 (well into the high-mean regime), filtered prob should be
        # near 1 and should not drop back
        if p_high[150] > 0.99:
            assert p_high[200] > 0.99, (
                f"Filtered P(regime {high_regime}) at t=200 is "
                f"{p_high[200]:.6f}, expected > 0.99"
            )

    def test_restricted_ar_smoothed_probs(self, rng: np.random.Generator) -> None:
        """Same monotonicity test for RestrictedMarkovAR.non_recurring()."""
        y = self._recurring_data(rng)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = RestrictedMarkovAR.non_recurring(y, k_regimes=2, order=1)
            results = model.fit(search_reps=5)

        probs = results.smoothed_marginal_probabilities
        assert np.isfinite(results.llf), "AR model should produce finite llf"

        consts = {r: results.regime_params[r]["const"] for r in range(2)}
        high_regime = max(consts, key=consts.get)
        p_high = probs[:, high_regime]

        above = np.where(p_high > 0.99)[0]
        if len(above) > 0:
            first_above = above[0]
            assert np.all(p_high[first_above:] > 0.99), (
                f"AR smoothed P(regime {high_regime}) dropped below 0.99 "
                f"after first exceeding it at t={first_above}. "
                f"Min after: {p_high[first_above:].min():.6f}"
            )


class TestRestrictedMarkovRegressionEdgeCases:
    """Additional edge case tests for RestrictedMarkovRegression."""

    def test_switching_variance(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        """Test restricted model with switching variance."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = RestrictedMarkovRegression(
                two_regime_data,
                k_regimes=2,
                restrictions={(0, 1): 0.0},
                switching_variance=True,
            )
            results = model.fit(search_reps=5)

        assert results.regime_transition[0, 1] == 0.0

    def test_3_regime_non_recurring_fit(
        self, three_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        """Test non-recurring 3-regime model fitting."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = RestrictedMarkovRegression.non_recurring(
                three_regime_data, k_regimes=3
            )
            results = model.fit(search_reps=5)

        assert results.k_regimes == 3
        assert isinstance(results.restricted_transitions, dict)


class TestRestrictedMarkovAR:
    """Test RestrictedMarkovAR."""

    def test_basic_restriction(
        self, two_regime_ar_data: NDArray[np.floating[Any]]
    ) -> None:
        restrictions = {(0, 1): 0.0}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = RestrictedMarkovAR(
                two_regime_ar_data,
                k_regimes=2,
                order=1,
                restrictions=restrictions,
            )
            results = model.fit(search_reps=5)

        assert results.regime_transition[0, 1] == 0.0
        assert results.restricted_transitions == restrictions
        assert np.isfinite(results.llf), "Restricted AR model should produce finite llf"

    def test_non_recurring_factory(
        self, two_regime_ar_data: NDArray[np.floating[Any]]
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = RestrictedMarkovAR.non_recurring(
                two_regime_ar_data, k_regimes=2, order=1
            )
            results = model.fit(search_reps=5)

        assert isinstance(results.restricted_transitions, dict)
        assert np.isfinite(results.llf), (
            "Non-recurring AR model should produce finite llf"
        )

    def test_result_type(self, two_regime_ar_data: NDArray[np.floating[Any]]) -> None:
        from regimes.markov.results import MarkovARResults

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = RestrictedMarkovAR(
                two_regime_ar_data,
                k_regimes=2,
                order=1,
                restrictions={(0, 1): 0.0},
            )
            results = model.fit(search_reps=5)

        assert isinstance(results, MarkovARResults)
