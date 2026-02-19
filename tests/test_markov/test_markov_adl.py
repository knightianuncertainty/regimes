"""Tests for MarkovADL model."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from regimes.markov.models import MarkovADL
from regimes.markov.results import MarkovADLResults


@pytest.fixture
def two_regime_adl_data(
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """ADL data with two regime structure."""
    n = 200
    x = rng.standard_normal(n)
    y = np.zeros(n)
    # Regime 1
    for t in range(1, 100):
        y[t] = 1.0 + 0.3 * y[t - 1] + 0.5 * x[t] + rng.standard_normal()
    # Regime 2
    for t in range(100, n):
        y[t] = 3.0 + 0.7 * y[t - 1] + 1.0 * x[t] + rng.standard_normal()
    return y, x


class TestMarkovADLInit:
    """Test MarkovADL initialization."""

    def test_basic_init(
        self,
        two_regime_adl_data: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
    ) -> None:
        y, x = two_regime_adl_data
        model = MarkovADL(y, x, k_regimes=2, ar_order=1, exog_lags=1)
        assert model.k_regimes == 2
        assert model.ar_order == 1
        assert model.exog_lags_raw == 1

    def test_no_exog_lags(
        self,
        two_regime_adl_data: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
    ) -> None:
        y, x = two_regime_adl_data
        model = MarkovADL(y, x, k_regimes=2, ar_order=1, exog_lags=0)
        assert model.exog_lags_raw == 0


class TestMarkovADLFit:
    """Test MarkovADL.fit()."""

    def test_basic_fit(
        self,
        two_regime_adl_data: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
    ) -> None:
        y, x = two_regime_adl_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MarkovADL(y, x, k_regimes=2, ar_order=1, exog_lags=0)
            results = model.fit(search_reps=3)

        assert isinstance(results, MarkovADLResults)
        assert results.k_regimes == 2

    def test_result_fields(
        self,
        two_regime_adl_data: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
    ) -> None:
        y, x = two_regime_adl_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MarkovADL(y, x, k_regimes=2, ar_order=1, exog_lags=0)
            r = model.fit(search_reps=3)

        assert r.regime_transition.shape == (2, 2)
        assert r.smoothed_marginal_probabilities.shape[1] == 2
        assert np.isfinite(r.llf)
        assert len(r.params) > 0

    def test_summary(
        self,
        two_regime_adl_data: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
    ) -> None:
        y, x = two_regime_adl_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MarkovADL(y, x, k_regimes=2, ar_order=1, exog_lags=0)
            r = model.fit(search_reps=3)

        s = r.summary()
        assert "Markov" in s


class TestMarkovADLExogLags:
    """Test MarkovADL with various exog_lags configurations."""

    def test_exog_lags_dict_int_keys(
        self,
        two_regime_adl_data: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
    ) -> None:
        """Dict-form exog_lags with integer keys."""
        y, x = two_regime_adl_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MarkovADL(y, x, k_regimes=2, ar_order=1, exog_lags={0: 1})
            results = model.fit(search_reps=3)

        assert isinstance(results, MarkovADLResults)

    def test_exog_lags_with_positive_lags(
        self,
        two_regime_adl_data: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
    ) -> None:
        """exog_lags > 0 should include lagged exog columns."""
        y, x = two_regime_adl_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MarkovADL(y, x, k_regimes=2, ar_order=1, exog_lags=2)
            results = model.fit(search_reps=3)

        assert isinstance(results, MarkovADLResults)
        # Should have more parameters due to lagged exog
        assert len(results.params) > 4  # const, ar.L1, x0, x0.L1, x0.L2 per regime

    def test_maxlag_property(
        self,
        two_regime_adl_data: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
    ) -> None:
        """maxlag should be max of ar_order and exog_lags."""
        y, x = two_regime_adl_data
        model = MarkovADL(y, x, k_regimes=2, ar_order=1, exog_lags=3)
        assert model.maxlag == 3

    def test_multiple_exog(self, rng: np.random.Generator) -> None:
        """Test with multiple exogenous variables."""
        n = 200
        x = rng.standard_normal((n, 2))
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = (
                1.0
                + 0.3 * y[t - 1]
                + 0.5 * x[t, 0]
                + 0.2 * x[t, 1]
                + rng.standard_normal()
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MarkovADL(y, x, k_regimes=2, ar_order=1, exog_lags=0)
            results = model.fit(search_reps=3)

        assert isinstance(results, MarkovADLResults)


class TestMarkovADLFromModel:
    """Test MarkovADL.from_model()."""

    def test_from_adl(
        self,
        two_regime_adl_data: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
    ) -> None:
        from regimes.models import ADL

        y, x = two_regime_adl_data
        adl = ADL(y, x, lags=1, exog_lags=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ms = MarkovADL.from_model(adl, k_regimes=2)
            results = ms.fit()

        assert isinstance(results, MarkovADLResults)
        assert results.k_regimes == 2


class TestMarkovADLConvenience:
    """Test ADL.markov_switching() convenience method."""

    def test_adl_markov_switching(
        self,
        two_regime_adl_data: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
    ) -> None:
        from regimes.models import ADL

        y, x = two_regime_adl_data
        model = ADL(y, x, lags=1, exog_lags=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = model.markov_switching(k_regimes=2)

        assert isinstance(results, MarkovADLResults)
