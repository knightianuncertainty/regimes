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

        if np.isfinite(r_results.llf) and np.isfinite(u_results.llf):
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
