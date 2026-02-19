"""Tests for MarkovRegression model."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from regimes.markov.models import MarkovRegression
from regimes.markov.results import MarkovRegressionResults, MarkovSwitchingResultsBase


class TestMarkovRegressionInit:
    """Test MarkovRegression initialization."""

    def test_basic_init(self, two_regime_data: NDArray[np.floating[Any]]) -> None:
        model = MarkovRegression(two_regime_data, k_regimes=2)
        assert model.k_regimes == 2
        assert model.trend == "c"
        assert model.ordering == "first_appearance"

    def test_custom_ordering(self, two_regime_data: NDArray[np.floating[Any]]) -> None:
        model = MarkovRegression(two_regime_data, k_regimes=2, ordering="intercept")
        assert model.ordering == "intercept"

    def test_no_ordering(self, two_regime_data: NDArray[np.floating[Any]]) -> None:
        model = MarkovRegression(two_regime_data, k_regimes=2, ordering=None)
        assert model.ordering is None

    def test_with_exog(
        self,
        two_regime_regression_data: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
    ) -> None:
        y, X = two_regime_regression_data
        model = MarkovRegression(y, k_regimes=2, exog=X, trend="n")
        assert model.exog is not None

    def test_switching_variance(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        model = MarkovRegression(two_regime_data, k_regimes=2, switching_variance=True)
        assert model.switching_variance is True


class TestMarkovRegressionFit:
    """Test MarkovRegression.fit()."""

    def test_basic_fit(self, ms_regression_results: MarkovRegressionResults) -> None:
        results = ms_regression_results
        assert isinstance(results, MarkovRegressionResults)
        assert isinstance(results, MarkovSwitchingResultsBase)

    def test_result_fields(
        self, ms_regression_results: MarkovRegressionResults
    ) -> None:
        r = ms_regression_results
        assert r.k_regimes == 2
        assert r.regime_transition.shape == (2, 2)
        assert r.smoothed_marginal_probabilities.shape[1] == 2
        assert r.filtered_marginal_probabilities.shape[1] == 2
        assert len(r.params) > 0
        assert len(r.bse) == len(r.params)
        assert np.isfinite(r.llf)
        assert len(r.resid) == r.nobs
        assert len(r.fittedvalues) == r.nobs

    def test_transition_matrix_valid(
        self, ms_regression_results: MarkovRegressionResults
    ) -> None:
        P = ms_regression_results.regime_transition
        # Columns should sum to ~1
        col_sums = P.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=0.01)
        # All entries should be non-negative
        assert np.all(P >= -0.01)

    def test_information_criteria(
        self, ms_regression_results: MarkovRegressionResults
    ) -> None:
        r = ms_regression_results
        assert np.isfinite(r.aic)
        assert np.isfinite(r.bic)
        assert np.isfinite(r.hqic)
        # BIC should penalize more than AIC for moderate sample sizes
        # (not always, depends on n and k, but generally true)

    def test_regime_params(
        self, ms_regression_results: MarkovRegressionResults
    ) -> None:
        r = ms_regression_results
        assert 0 in r.regime_params
        assert 1 in r.regime_params
        # Each regime should have a constant
        assert "const" in r.regime_params[0] or any(
            "const" in k for k in r.regime_params[0]
        )

    def test_expected_durations(
        self, ms_regression_results: MarkovRegressionResults
    ) -> None:
        d = ms_regression_results.expected_durations
        assert len(d) == 2
        assert all(d_i >= 1.0 for d_i in d)

    def test_most_likely_regime(
        self, ms_regression_results: MarkovRegressionResults
    ) -> None:
        mlr = ms_regression_results.most_likely_regime
        assert len(mlr) == ms_regression_results.nobs
        assert set(mlr).issubset({0, 1})

    def test_regime_assignments(
        self, ms_regression_results: MarkovRegressionResults
    ) -> None:
        ra = ms_regression_results.regime_assignments
        assert len(ra) == ms_regression_results.nobs
        assert all(0 <= r <= 1 for r in ra)

    def test_regime_periods(
        self, ms_regression_results: MarkovRegressionResults
    ) -> None:
        periods = ms_regression_results.regime_periods()
        assert len(periods) > 0
        for regime, start, end in periods:
            assert regime in (0, 1)
            assert start < end

    def test_tvalues_and_pvalues(
        self, ms_regression_results: MarkovRegressionResults
    ) -> None:
        r = ms_regression_results
        assert len(r.tvalues) == len(r.params)
        assert len(r.pvalues) == len(r.params)
        assert all(0 <= p <= 1 for p in r.pvalues)

    def test_conf_int(self, ms_regression_results: MarkovRegressionResults) -> None:
        ci = ms_regression_results.conf_int()
        assert ci.shape == (len(ms_regression_results.params), 2)
        # Lower bound should be less than upper bound
        assert all(ci[i, 0] < ci[i, 1] for i in range(ci.shape[0]))

    def test_summary(self, ms_regression_results: MarkovRegressionResults) -> None:
        s = ms_regression_results.summary()
        assert "Markov" in s
        assert "Regime" in s

    def test_to_dataframe(self, ms_regression_results: MarkovRegressionResults) -> None:
        import pandas as pd

        df = ms_regression_results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(ms_regression_results.params)


class TestMarkovRegressionFromModel:
    """Test MarkovRegression.from_model()."""

    def test_from_ols(
        self,
        two_regime_regression_data: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
    ) -> None:
        from regimes.models import OLS

        y, X = two_regime_regression_data
        ols = OLS(y, X, has_constant=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ms = MarkovRegression.from_model(ols, k_regimes=2)
            results = ms.fit()

        assert isinstance(results, MarkovRegressionResults)
        assert results.k_regimes == 2


class TestMarkovRegressionConvenience:
    """Test OLS.markov_switching() convenience method."""

    def test_ols_markov_switching(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        from regimes.models import OLS

        model = OLS(two_regime_data, has_constant=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = model.markov_switching(k_regimes=2)

        assert isinstance(results, MarkovRegressionResults)
        assert results.k_regimes == 2


class TestMarkovRegressionInterceptOrdering:
    """Test MarkovRegression with ordering='intercept'."""

    def test_ordering_intercept_fit(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        """Intercept ordering should produce valid results with two distinct means."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MarkovRegression(two_regime_data, k_regimes=2, ordering="intercept")
            results = model.fit(search_reps=5)

        assert isinstance(results, MarkovRegressionResults)
        # Both regimes should have regime params
        means = [results.regime_params[j].get("const", 0) for j in range(2)]
        assert abs(means[0] - means[1]) > 1.0  # regimes should be distinct

    def test_from_model_no_exog(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        """from_model() with exog=None should work (mean-only model)."""
        from regimes.models import OLS

        ols = OLS(two_regime_data, has_constant=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ms = MarkovRegression.from_model(ols, k_regimes=2)
            results = ms.fit()

        assert isinstance(results, MarkovRegressionResults)


class TestMarkovRegressionSummaryEdgeCases:
    """Test summary edge cases for coverage of results.py lines 304-308, 335."""

    def test_summary_with_restricted_transitions(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        """Summary should include restricted transitions section."""
        from regimes.markov.restricted import RestrictedMarkovRegression

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = RestrictedMarkovRegression(
                two_regime_data, k_regimes=2, restrictions={(0, 1): 0.0}
            )
            results = model.fit(search_reps=5)

        s = results.summary()
        assert "Restricted transitions" in s
        assert "P(0,1)" in s

    def test_plot_regime_shading_no_y(
        self, ms_regression_results: MarkovRegressionResults
    ) -> None:
        """plot_regime_shading() without y should reconstruct from fitted+resid."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = ms_regression_results.plot_regime_shading()
        assert fig is not None
        plt.close(fig)


class TestMarkovRegressionParameterRecovery:
    """Test that the model recovers true parameters on well-separated data."""

    def test_recovers_mean_shift(self) -> None:
        rng = np.random.default_rng(123)
        y = np.concatenate(
            [
                rng.standard_normal(150) + 0.0,
                rng.standard_normal(150) + 5.0,
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MarkovRegression(y, k_regimes=2)
            results = model.fit(search_reps=5)

        # Two regimes should have different means
        means = [results.regime_params[j].get("const", 0) for j in range(2)]
        low_mean = min(means)
        high_mean = max(means)
        # Check the two means are well separated and roughly correct
        assert high_mean - low_mean > 3.0
        assert abs(low_mean - 0.0) < 1.5
        assert abs(high_mean - 5.0) < 1.5
