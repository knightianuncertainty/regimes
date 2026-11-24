"""Tests for MarkovAR model."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from regimes.markov.models import MarkovAR
from regimes.markov.results import MarkovARResults


class TestMarkovARInit:
    """Test MarkovAR initialization."""

    def test_basic_init(self, two_regime_ar_data: NDArray[np.floating[Any]]) -> None:
        model = MarkovAR(two_regime_ar_data, k_regimes=2, order=1)
        assert model.k_regimes == 2
        assert model.order == 1
        assert model.switching_ar is True

    def test_non_switching_ar(
        self, two_regime_ar_data: NDArray[np.floating[Any]]
    ) -> None:
        model = MarkovAR(two_regime_ar_data, k_regimes=2, order=1, switching_ar=False)
        assert model.switching_ar is False

    def test_order_2(self, two_regime_ar_data: NDArray[np.floating[Any]]) -> None:
        model = MarkovAR(two_regime_ar_data, k_regimes=2, order=2)
        assert model.order == 2


class TestMarkovARFit:
    """Test MarkovAR.fit()."""

    def test_basic_fit(self, ms_ar_results: MarkovARResults) -> None:
        assert isinstance(ms_ar_results, MarkovARResults)

    def test_result_fields(self, ms_ar_results: MarkovARResults) -> None:
        r = ms_ar_results
        assert r.k_regimes == 2
        assert r.order == 1
        assert r.regime_transition.shape == (2, 2)
        assert r.smoothed_marginal_probabilities.shape[1] == 2
        assert np.isfinite(r.llf)

    def test_ar_params(self, ms_ar_results: MarkovARResults) -> None:
        r = ms_ar_results
        # ar_params should have entries for each regime
        assert 0 in r.ar_params
        assert 1 in r.ar_params
        # Each should have "ar.L1" or similar
        for j in range(2):
            assert len(r.ar_params[j]) >= 1

    def test_transition_matrix_valid(self, ms_ar_results: MarkovARResults) -> None:
        P = ms_ar_results.regime_transition
        col_sums = P.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=0.01)
        assert np.all(P >= -0.01)

    def test_summary(self, ms_ar_results: MarkovARResults) -> None:
        s = ms_ar_results.summary()
        assert "Markov" in s
        assert "AR" in s or "ar" in s.lower()

    def test_regime_params(self, ms_ar_results: MarkovARResults) -> None:
        r = ms_ar_results
        assert 0 in r.regime_params
        assert 1 in r.regime_params


class TestMarkovARFromModel:
    """Test MarkovAR.from_model()."""

    def test_from_ar(self, two_regime_ar_data: NDArray[np.floating[Any]]) -> None:
        from regimes.models import AR

        ar = AR(two_regime_ar_data, lags=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ms = MarkovAR.from_model(ar, k_regimes=2)
            results = ms.fit()

        assert isinstance(results, MarkovARResults)
        assert results.k_regimes == 2
        assert results.order == 1


class TestMarkovARConvenience:
    """Test AR.markov_switching() convenience method."""

    def test_ar_markov_switching(
        self, two_regime_ar_data: NDArray[np.floating[Any]]
    ) -> None:
        from regimes.models import AR

        model = AR(two_regime_ar_data, lags=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = model.markov_switching(k_regimes=2)

        assert isinstance(results, MarkovARResults)
        assert results.k_regimes == 2


class TestSwitchingVarianceHigherOrder:
    """``order >= 2`` with ``switching_variance=True`` (a statsmodels defect)."""

    def test_variance_follows_the_current_regime(self) -> None:
        """``MarkovAR`` estimates with the corrected likelihood.

        statsmodels 0.14 and 0.15 give the variance of period ``t`` to the
        regime of period ``t - order + 1``. That the corrected class has the
        right likelihood is checked against a direct sum over regime paths
        in ``test_simulate.py``; an estimate on a long series with
        persistent regimes is close under either likelihood, so the
        assertion that matters here is that ``MarkovAR`` builds the
        corrected class. The other assertions check that estimation works.
        """
        from regimes.markov.models import _SMMarkovAutoregression

        rng = np.random.default_rng(5)
        n = 400
        sigma = np.where(np.arange(n) < n // 2, 0.3, 2.0)
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = 0.4 * y[t - 1] - 0.2 * y[t - 2] + sigma[t] * rng.standard_normal()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = MarkovAR(
                y,
                k_regimes=2,
                order=2,
                switching_ar=False,
                switching_trend=False,
                switching_variance=True,
            ).fit(search_reps=3)
        assert isinstance(results._sm_results.model, _SMMarkovAutoregression)
        variances = sorted(results.regime_params[j]["sigma2"] for j in (0, 1))
        assert variances[0] == pytest.approx(0.09, rel=0.35)
        assert variances[1] == pytest.approx(4.0, rel=0.35)
        # the regime path follows the variance, period by period
        noisy = int(np.argmax([results.regime_params[j]["sigma2"] for j in (0, 1)]))
        prob_noisy = np.asarray(results.smoothed_marginal_probabilities)[:, noisy]
        assert prob_noisy[: n // 2 - 30].mean() < 0.1
        assert prob_noisy[n // 2 + 10 :].mean() > 0.9
