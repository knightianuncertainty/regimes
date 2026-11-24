"""The model without switching, nested estimation and the bootstrap LR test
of ``RegimeNumberSelection``, with reference values from the R package
MSTest (``tests/reference_data/ms_mstest_reference.json``)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.regime_switching.markov_regression import (
    MarkovRegression as SMMarkovRegression,
)

import regimes.markov.selection as module
from regimes.markov._simulate import MonteCarloPValue, simulate_linear
from regimes.markov.models import _objective_search_fit, _SMMarkovAutoregression
from regimes.markov.selection import (
    OneRegimeResults,
    RegimeNumberSelection,
    _fit_one_regime,
    _split_starts,
    _variance_penalty,
)

REF = Path(__file__).resolve().parents[1] / "reference_data"


@pytest.fixture(autouse=True)
def _fewer_restarts(monkeypatch: Any) -> None:
    """One random restart per estimation instead of five (see the module)."""
    monkeypatch.setattr(RegimeNumberSelection, "_search_reps", 1)


@pytest.fixture(scope="module")
def reference() -> tuple[np.ndarray, dict[str, Any]]:
    y = pd.read_csv(REF / "ms_two_regime_series.csv")["y"].to_numpy()
    with (REF / "ms_mstest_reference.json").open(encoding="utf-8") as fh:
        return y, json.load(fh)


def _regression_data(seed: int = 0, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    return 1.0 + 2.0 * x + rng.standard_normal(n), x


class TestOneRegimeModel:
    def test_regression_is_ols(self) -> None:
        y, x = _regression_data()
        design = np.column_stack([np.ones(len(y)), x])
        ours = _fit_one_regime(y, design, 0, ["const", "x1"])
        theirs = sm.OLS(y, design).fit()
        np.testing.assert_allclose(ours.params, theirs.params, rtol=1e-10)
        assert ours.llf == pytest.approx(theirs.llf, rel=1e-12)
        assert ours.sigma2 == pytest.approx(theirs.ssr / len(y), rel=1e-12)
        # two coefficients and the variance
        assert ours.df_model == 3
        assert ours.bic == pytest.approx(-2 * ours.llf + np.log(len(y)) * 3)

    def test_autoregression_without_regressors_is_autoreg(self) -> None:
        rng = np.random.default_rng(1)
        y = np.zeros(300)
        for t in range(2, 300):
            y[t] = 1.0 + 0.5 * y[t - 1] - 0.2 * y[t - 2] + rng.standard_normal()
        ours = _fit_one_regime(y, np.ones((300, 1)), 2, ["const"])
        theirs = AutoReg(y, lags=2, trend="c").fit()
        np.testing.assert_allclose(ours.ar_params, theirs.params[1:], rtol=1e-6)
        # the constant is the mean: intercept / (1 - sum of the coefficients)
        assert ours.params[0] == pytest.approx(
            theirs.params[0] / (1 - theirs.params[1:].sum()), rel=1e-6
        )
        assert ours.nobs == 298
        assert ours.llf == pytest.approx(theirs.llf, rel=1e-8)

    def test_autoregression_with_regressors_is_the_mean_adjusted_form(self) -> None:
        """The same likelihood as ``MarkovAR`` with identical regimes, maximized."""
        rng = np.random.default_rng(2)
        n = 250
        x = rng.standard_normal(n)
        u = np.zeros(n)
        for t in range(1, n):
            u[t] = 0.6 * u[t - 1] + rng.standard_normal()
        y = 2.0 + 1.5 * x + u
        design = np.column_stack([np.ones(n), x])
        ours = _fit_one_regime(y, design, 1, ["const", "x1"])
        assert ours.params == pytest.approx([2.0, 1.5], abs=0.4)
        assert ours.ar_params[0] == pytest.approx(0.6, abs=0.1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _SMMarkovAutoregression(
                y, k_regimes=2, order=1, exog=x.reshape(-1, 1), switching_ar=False
            )

            def loglike(beta: Any, phi: float, sigma2: float) -> float:
                params = np.zeros(model.k_params)
                params[model.parameters["regime_transition"]] = [0.7, 0.4]
                for regime in (0, 1):
                    params[model.parameters[regime, "exog"]] = beta
                params[model.parameters["autoregressive"]] = phi
                params[model.parameters["variance"]] = sigma2
                return float(model.loglike(params))

            at_estimate = loglike(ours.params, ours.ar_params[0], ours.sigma2)
            assert ours.llf == pytest.approx(at_estimate, abs=1e-8)
            # a maximum: moving any parameter lowers the likelihood
            for d_beta, d_phi in (([0.01, 0], 0), ([0, 0.01], 0), ([0, 0], 0.01)):
                for sign in (-1, 1):
                    moved = loglike(
                        ours.params + sign * np.array(d_beta),
                        ours.ar_params[0] + sign * d_phi,
                        ours.sigma2,
                    )
                    assert moved < at_estimate

    def test_too_few_observations(self) -> None:
        with pytest.raises(ValueError, match="too few observations"):
            _fit_one_regime(np.arange(4.0), np.ones((4, 1)), 3, ["const"])


class TestSplitStarts:
    """A model with one regime more contains the smaller model."""

    def test_two_regimes_from_one(self) -> None:
        y, x = _regression_data(seed=3, n=120)
        design = np.column_stack([np.ones(len(y)), x])
        lower = _fit_one_regime(y, design, 0, ["const", "x1"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SMMarkovRegression(y, k_regimes=2, exog=x)
            starts = _split_starts(model, lower)
            # the exact split has the likelihood of the smaller model
            assert model.loglike(starts[0]) == pytest.approx(lower.llf, abs=1e-8)
            # the other start moves the constants apart, not the slope
            moved = starts[1] - starts[0]
            constant = [
                np.atleast_1d(np.arange(model.k_params)[model.parameters[j, "exog"]])[0]
                for j in (0, 1)
            ]
            assert moved[constant[0]] == pytest.approx(-0.5 * np.sqrt(lower.sigma2))
            assert moved[constant[1]] == pytest.approx(0.5 * np.sqrt(lower.sigma2))
            assert np.count_nonzero(moved) == 2

    @pytest.mark.parametrize("switching_variance", [False, True])
    def test_three_regimes_from_two(self, switching_variance: bool) -> None:
        from regimes.markov import MarkovRegression

        rng = np.random.default_rng(4)
        y = np.concatenate([rng.normal(m, 1, 50) for m in (0, 3, 0, 3)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lower = MarkovRegression(
                y, k_regimes=2, switching_variance=switching_variance
            ).fit()
            model = SMMarkovRegression(
                y, k_regimes=3, switching_variance=switching_variance
            )
            starts = _split_starts(model, lower)
            # two regimes to split, an exact and a moved start for each
            assert len(starts) == 4
            for exact in (starts[0], starts[2]):
                matrix = model.regime_transition_matrix(exact)[..., 0]
                np.testing.assert_allclose(matrix.sum(axis=0), 1.0)
                assert model.loglike(exact) == pytest.approx(lower.llf, abs=1e-6)

    def test_autoregression(self) -> None:
        rng = np.random.default_rng(5)
        y = np.zeros(160)
        for t in range(1, 160):
            y[t] = 0.5 + 0.4 * y[t - 1] + rng.standard_normal()
        lower = _fit_one_regime(y, np.ones((160, 1)), 1, ["const"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _SMMarkovAutoregression(y, k_regimes=2, order=1)
            starts = _split_starts(model, lower)
            assert model.loglike(starts[0]) == pytest.approx(lower.llf, abs=1e-8)

    def test_only_the_variance_switches(self) -> None:
        y, x = _regression_data(seed=6, n=100)
        lower = _fit_one_regime(y, np.ones((100, 1)), 0, ["const"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SMMarkovRegression(
                y, k_regimes=2, switching_trend=False, switching_variance=True
            )
            starts = _split_starts(model, lower)
        variances = starts[1][model.parameters["variance"]]
        np.testing.assert_allclose(sorted(variances / lower.sigma2), [0.5, 2.0])


class TestBaselineAndCounts:
    def test_baseline_has_the_regressors_of_the_model(self) -> None:
        """``y = 1 + 2x + e`` has one regime (the baseline used to ignore x)."""
        y, x = _regression_data(seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = RegimeNumberSelection(
                y, k_max=2, method="bic", exog=x, random_state=0
            ).fit()
        assert results.selected_k == 1
        table = results.ic_table.set_index("K")
        one = results.results_by_k[1]
        assert isinstance(one, OneRegimeResults)
        assert one.param_names == ["const", "x1"]
        assert one.params == pytest.approx([1.0, 2.0], abs=0.25)
        # constant, slope and variance; two of each plus two probabilities
        assert table.loc[1, "n_params"] == 3
        assert table.loc[2, "n_params"] == 7
        lr = 2 * (table.loc[2, "llf"] - table.loc[1, "llf"])
        assert 0 <= lr < 25  # it was 345
        assert table.loc[1, "BIC"] == pytest.approx(
            -2 * table.loc[1, "llf"] + 3 * np.log(len(y))
        )

    @pytest.mark.parametrize("trend", ["c", "ct", "n"])
    def test_baseline_is_nested_for_every_trend(self, trend: str) -> None:
        rng = np.random.default_rng(7)
        y = 0.02 * np.arange(150) + rng.standard_normal(150)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = RegimeNumberSelection(
                y, k_max=2, trend=trend, random_state=0
            ).fit()
        table = results.ic_table.set_index("K")
        assert table.loc[2, "llf"] >= table.loc[1, "llf"] - 1e-6

    def test_autoregressive_type(self) -> None:
        rng = np.random.default_rng(8)
        y = np.zeros(200)
        for t in range(1, 200):
            y[t] = 0.5 * y[t - 1] + rng.standard_normal()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = RegimeNumberSelection(
                y, k_max=2, model_type="ar", order=1, random_state=0
            ).fit()
        one = results.results_by_k[1]
        assert one.nobs == 199
        assert one.ar_params[0] == pytest.approx(0.5, abs=0.15)
        table = results.ic_table.set_index("K")
        assert table.loc[2, "llf"] >= table.loc[1, "llf"] - 1e-6
        assert results.selected_k == 1

    def test_argument_checks(self) -> None:
        y = np.random.default_rng(0).standard_normal(30)
        with pytest.raises(ValueError, match="method must be one of"):
            RegimeNumberSelection(y, method="lrt")
        with pytest.raises(ValueError, match="model_type"):
            RegimeNumberSelection(y, model_type="adl")
        # 19 replications give an exact test at 5 percent; 18 cannot reject
        with pytest.raises(ValueError, match="use at least 19 replications"):
            RegimeNumberSelection(y, method="sequential", n_bootstrap=18)
        RegimeNumberSelection(y, method="sequential", n_bootstrap=19)
        RegimeNumberSelection(y, method="bic", n_bootstrap=5)
        # a misspelled model argument is an error, not "one regime"
        with pytest.raises(TypeError, match="switching_varaince"):
            RegimeNumberSelection(y, switching_varaince=True)
        with pytest.raises(ValueError, match="order must be a positive integer"):
            RegimeNumberSelection(y, model_type="ar", order=0)
        with pytest.raises(ValueError, match="search_reps"):
            RegimeNumberSelection(y, search_reps=-1)
        assert RegimeNumberSelection(y, search_reps=7)._search_reps == 7

    @pytest.mark.parametrize(
        ("trend", "names"), [("c", ["const"]), ("ct", ["const", "trend"]), ("n", [])]
    )
    def test_one_regime_model_has_the_trend_of_the_model(
        self, trend: str, names: list[str]
    ) -> None:
        rng = np.random.default_rng(12)
        n = 90
        y = 1.0 + 0.03 * np.arange(n) + rng.standard_normal(n)
        test = RegimeNumberSelection(y, trend=trend)
        one = test._fit_k1()["results"]
        assert one.param_names == names
        design = {
            "c": np.ones((n, 1)),
            "ct": np.column_stack([np.ones(n), np.arange(1, n + 1)]),
            "n": np.zeros((n, 0)),
        }[trend]
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0] if names else 0.0
        ssr = float(np.sum((y - fitted) ** 2))
        assert one.llf == pytest.approx(
            -0.5 * n * (np.log(2 * np.pi * ssr / n) + 1), abs=1e-8
        )
        assert one.df_model == len(names) + 1

    def test_criteria_and_counts_of_the_one_regime_autoregression(self) -> None:
        rng = np.random.default_rng(13)
        n = 160
        x = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = 0.4 * y[t - 1] + 0.2 * y[t - 2] + 0.5 * x[t] + rng.standard_normal()
        test = RegimeNumberSelection(y, model_type="ar", order=2, exog=x)
        info = test._fit_k1()
        one = info["results"]
        # constant, slope, two autoregressive coefficients and the variance
        assert one.df_model == 5
        assert one.nobs == n - 2
        assert info["n_params"] == 5
        assert info["aic"] == pytest.approx(-2 * one.llf + 2 * 5)
        assert info["bic"] == pytest.approx(-2 * one.llf + 5 * np.log(n - 2))
        assert info["hqic"] == pytest.approx(
            -2 * one.llf + 2 * 5 * np.log(np.log(n - 2))
        )
        table = test._build_ic_table({1: info}).set_index("K")
        assert table.loc[1, "AIC"] == info["aic"]
        assert table.loc[1, "BIC"] == info["bic"]
        assert table.loc[1, "HQIC"] == info["hqic"]


class TestVariancePenalty:
    """Regime-specific variances: the penalty of Kasahara and Shimotsu."""

    def test_formula(self) -> None:
        y = np.random.default_rng(14).standard_normal(100)
        common = SMMarkovRegression(y, k_regimes=2)
        assert _variance_penalty(common, 1.3) is None
        model = SMMarkovRegression(y, k_regimes=3, switching_variance=True)
        penalty = _variance_penalty(model, 1.3)
        params = np.asarray(model.start_params, dtype=float)
        positions = np.arange(model.k_params)[model.parameters["variance"]]
        params[positions] = 1.3
        assert penalty(params) == pytest.approx(0.0, abs=1e-12)
        params[positions] = [1.3, 0.13, 2.6]
        a_n = 20 / np.sqrt(100)
        expected = -a_n * sum(1.3 / v + np.log(v / 1.3) - 1 for v in (1.3, 0.13, 2.6))
        assert penalty(params) == pytest.approx(expected)
        # a variance at the lower bound of the paper costs about 10^4 a_n
        params[positions] = [1.3, 1.3 * 0.01**2, 1.3]
        assert penalty(params) < -9000 * a_n

    def test_no_variance_spike_and_the_same_result_for_every_seed(
        self, reference: tuple[np.ndarray, dict[str, Any]]
    ) -> None:
        """Without the penalty the three-regime fit put a variance of 0.002
        on a few observations, a different spike for every seed (likelihood
        ratio statistics from 3.9 to 9.7)."""
        y, _ref = reference
        statistics = []
        for seed in (1, 2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = RegimeNumberSelection(
                    y, k_max=3, random_state=seed, switching_variance=True
                ).fit()
            table = results.ic_table.set_index("K")
            sm_results = results.results_by_k[3]._sm_results
            variances = sm_results.params[sm_results.model.parameters["variance"]]
            assert variances.min() > 0.2 * results.results_by_k[1].sigma2
            # the plain log-likelihood is reported, not the penalized one
            assert table.loc[3, "llf"] == pytest.approx(
                sm_results.model.loglike(sm_results.params), abs=1e-8
            )
            statistics.append(2 * (table.loc[3, "llf"] - table.loc[2, "llf"]))
        assert statistics[0] == pytest.approx(statistics[1], abs=0.05)
        assert 0 <= statistics[0] < 6


class TestAgainstMSTest:
    """Log-likelihoods of the R package MSTest on a stored series."""

    @staticmethod
    def _params(model: Any, estimates: dict[str, Any]) -> np.ndarray:
        k = model.k_regimes
        params = np.zeros(model.k_params)
        transition = np.asarray(estimates["transition"], dtype=float)
        index = np.arange(model.k_params)[
            model.parameters["regime_transition"]
        ].reshape(k - 1, k)
        params[index] = transition[: k - 1, :]
        for j in range(k):
            params[model.parameters[j, "exog"]] = estimates["mu"][j]
            params[model.parameters[j, "variance"]] = estimates["sigma2"][j]
        return params

    @pytest.mark.parametrize("which", [0, 1])
    def test_same_likelihood_function(
        self, reference: tuple[np.ndarray, dict[str, Any]], which: int
    ) -> None:
        """At MSTest's estimates the likelihood is the value MSTest reports."""
        y, ref = reference
        case = ref["models"][which]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k in (2, 3):
                model = SMMarkovRegression(
                    y, k_regimes=k, switching_variance=case["switching_variance"]
                )
                params = self._params(model, case[f"k{k}"])
                assert model.loglike(params) == pytest.approx(
                    case["llf"][f"k{k}"], abs=1e-6
                )

    @pytest.mark.parametrize("which", [0, 1])
    def test_estimates(
        self, reference: tuple[np.ndarray, dict[str, Any]], which: int
    ) -> None:
        y, ref = reference
        case = ref["models"][which]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = RegimeNumberSelection(
                y,
                k_max=2,
                method="bic",
                random_state=0,
                switching_variance=case["switching_variance"],
            ).fit()
        table = results.ic_table.set_index("K")
        assert table.loc[1, "llf"] == pytest.approx(case["llf"]["k1"], abs=1e-8)
        if case["switching_variance"]:
            # estimated with the variance penalty, which MSTest does not have
            assert table.loc[2, "llf"] == pytest.approx(case["llf"]["k2"], abs=0.1)
        else:
            # the maximum is at least the value MSTest's EM algorithm stops at
            assert table.loc[2, "llf"] >= case["llf"]["k2"] - 1e-6
            assert table.loc[2, "llf"] == pytest.approx(case["llf"]["k2"], abs=0.05)
        lr = 2 * (table.loc[2, "llf"] - table.loc[1, "llf"])
        assert lr == pytest.approx(case["lr"]["k1_vs_k2"], abs=0.1)
        assert results.selected_k == 2

    def test_three_regimes_against_the_reference(
        self, reference: tuple[np.ndarray, dict[str, Any]]
    ) -> None:
        """Common variance: the search ends at one of two local maxima, the
        better of which is the value MSTest reports (-458.13 and -458.33)."""
        y, ref = reference
        case = next(c for c in ref["models"] if not c["switching_variance"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = RegimeNumberSelection(y, k_max=3, random_state=0).fit()
        table = results.ic_table.set_index("K")
        assert table.loc[3, "llf"] >= table.loc[2, "llf"] - 1e-6
        assert table.loc[3, "llf"] == pytest.approx(case["llf"]["k3"], abs=0.25)
        assert table.loc[3, "n_params"] == 10


class _StubResults:
    def __init__(self, params: Any, converged: bool = True) -> None:
        self.params = np.asarray(params, dtype=float)
        self.mle_retvals = {"converged": converged}


class _StubModel:
    """A model with two candidate optima, for the search alone."""

    k_params = 1
    start_params = np.array([0.0])

    def __init__(self, converged_at: dict[float, bool] | None = None) -> None:
        self.calls: list[tuple[float, int, str]] = []
        self.converged_at = converged_at or {}

    def loglike(self, params: Any, transformed: bool = True) -> float:
        # the plain likelihood prefers the point 0, by one unit
        return -10.0 if float(np.real(params[0])) == 0.0 else -11.0

    def transform_params(self, params: Any) -> Any:
        return params

    def fit(self, start_params: Any = None, **kwargs: Any) -> _StubResults:
        point = 0.0 if start_params is None else float(start_params[0])
        self.calls.append((point, kwargs["maxiter"], kwargs["cov_type"]))
        return _StubResults([point], self.converged_at.get(point, True))

    def smooth(self, params: Any, transformed: bool, cov_type: str) -> _StubResults:
        self.calls.append((float(params[0]), -1, cov_type))
        return _StubResults(params)


class TestObjectiveSearch:
    def test_starting_values_are_compared_on_the_penalized_objective(self) -> None:
        model = _StubModel()
        # the penalty reverses the order: 5 units against the point 0
        results = _objective_search_fit(
            model,
            "bfgs",
            100,
            5,
            0,
            None,
            [np.array([1.0])],
            lambda params: -5.0 if float(np.real(params[0])) == 0.0 else 0.0,
            True,
        )
        assert results.params[0] == 1.0
        assert "loglike" not in vars(model)  # the override is removed
        without = _objective_search_fit(
            _StubModel(), "bfgs", 100, 5, 0, None, [np.array([1.0])], None, True
        )
        assert without.params[0] == 0.0

    def test_covariance_once_and_only_when_asked(self) -> None:
        for inference, expected in ((True, "approx"), (False, "none")):
            model = _StubModel()
            _objective_search_fit(
                model, "bfgs", 100, 5, 0, None, [np.array([1.0])], None, inference
            )
            assert [c[2] for c in model.calls] == ["none", "none", expected]

    def test_a_fit_that_stops_at_the_iteration_limit_is_continued(self) -> None:
        model = _StubModel(converged_at={0.0: False})
        results = _objective_search_fit(
            model, "bfgs", 100, 5, 0, None, [np.array([1.0])], None, True
        )
        # continued once from the best point, with five times the iterations
        assert (0.0, 500, "none") in model.calls
        assert results.mle_retvals == {"converged": False}

    def test_convergence_flag_reaches_the_results(self) -> None:
        """It was lost: every model of a selection reported "converged"."""
        from regimes.markov import MarkovRegression

        rng = np.random.default_rng(17)
        y = np.concatenate([rng.normal(0, 1, 60), rng.normal(3, 1, 60)])
        for maxiter, expected in ((1, False), (200, True)):
            model = MarkovRegression(y, k_regimes=2, switching_variance=True)
            model._search_objective = lambda sm_model: None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = model.fit(maxiter=maxiter, em_iter=0)
            assert results.converged is expected


class TestPenaltyInTheFits:
    @pytest.mark.parametrize("model_type", ["regression", "ar"])
    def test_every_switching_variance_fit_uses_the_variance_of_its_sample(
        self, monkeypatch: Any, model_type: str
    ) -> None:
        """Also for autoregressions, and with the variance (not the standard
        deviation) of the one-regime model of the same sample."""
        rng = np.random.default_rng(18)
        y = np.concatenate([rng.normal(0, 1, 50), rng.normal(3, 2, 50)])
        seen: list[tuple[str, float, int]] = []
        original = module._variance_penalty

        def recording(sm_model: Any, sigma2: float) -> Any:
            seen.append((type(sm_model).__name__, sigma2, int(sm_model.nobs)))
            return original(sm_model, sigma2)

        monkeypatch.setattr(module, "_variance_penalty", recording)
        test = RegimeNumberSelection(
            y, k_max=2, model_type=model_type, random_state=0, switching_variance=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = test._fit_up_to(2, y, inference=True)
            sample = test._simulate(fitted[1]["results"], np.random.default_rng(3))
            on_sample = test._fit_up_to(2, sample, inference=False)
        assert fitted[2]["penalized"] and on_sample[2]["penalized"]
        assert len(seen) == 2
        nobs = len(y) - (1 if model_type == "ar" else 0)
        assert seen[0][1:] == (fitted[1]["results"].sigma2, nobs)
        assert seen[1][1:] == (on_sample[1]["results"].sigma2, nobs)
        assert seen[0][1] != seen[1][1]
        # standard errors for the data, none for a bootstrap sample
        assert np.isfinite(fitted[2]["results"].bse).sum() >= 4
        assert not np.isfinite(on_sample[2]["results"].bse).any()

    def test_estimates_maximize_the_penalized_likelihood(self) -> None:
        rng = np.random.default_rng(19)
        y = np.concatenate([rng.normal(0, 1, 60), rng.normal(3, 2, 60)])
        test = RegimeNumberSelection(
            y, k_max=2, random_state=0, switching_variance=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = test._fit_up_to(2, y, inference=False)
        sm_results = fitted[2]["results"]._sm_results
        sm_model = sm_results.model
        penalty = _variance_penalty(sm_model, fitted[1]["results"].sigma2)
        free = np.asarray(sm_model.untransform_params(sm_results.params), dtype=float)

        def objective(point: np.ndarray, with_penalty: bool) -> float:
            constrained = sm_model.transform_params(point)
            value = sm_model.loglike(constrained)
            return float(value + penalty(constrained)) if with_penalty else float(value)

        def gradient(with_penalty: bool) -> np.ndarray:
            step = 1e-5
            out = np.empty(len(free))
            for i in range(len(free)):
                up, down = free.copy(), free.copy()
                up[i] += step
                down[i] -= step
                out[i] = (
                    objective(up, with_penalty) - objective(down, with_penalty)
                ) / (2 * step)
            return out

        assert np.max(np.abs(gradient(True))) < 1e-2
        assert np.max(np.abs(gradient(False))) > 0.1
        # the log-likelihood reported is the plain one
        assert fitted[2]["llf"] == pytest.approx(objective(free, False), abs=1e-8)

    def test_k_max_is_flagged_for_a_criterion_too(self, monkeypatch: Any) -> None:
        test = RegimeNumberSelection(np.zeros(50), k_max=2, method="aic")
        monkeypatch.setattr(test, "_fit_k1", lambda endog=None: _fake_fit(-100.0, 1))
        monkeypatch.setattr(
            test, "_fit_k_regimes", lambda k, lower, one, *a, **kw: _fake_fit(-50.0, k)
        )
        monkeypatch.setattr(test, "_select_by_ic", lambda table, criterion: 2)
        results = test.fit()
        assert results.notes == [
            "AIC is smallest at k_max = 2: more regimes were not considered"
        ]


def _fake_fit(llf: float, k: int) -> dict[str, Any]:
    return {
        "results": OneRegimeResults(
            params=np.zeros(1),
            param_names=["const"],
            ar_params=np.zeros(0),
            sigma2=1.0,
            llf=llf,
            nobs=50,
            df_model=1 + 3 * (k - 1),
        ),
        "llf": llf,
        "aic": 0.0,
        "bic": 0.0,
        "hqic": 0.0,
        "n_params": 1 + 3 * (k - 1),
        "penalized": False,
    }


class TestSequentialBootstrap:
    def _table(
        self, test: RegimeNumberSelection, monkeypatch: Any, llf: dict[int, float]
    ) -> list[int]:
        """Replace the estimations by a table of log-likelihoods."""
        fitted: list[int] = []

        def fit_k1(endog: Any = None) -> Any:
            fitted.append(1)
            return _fake_fit(llf[1], 1)

        def fit_k(
            k: int, lower: Any, one: Any, endog: Any = None, inference: bool = True
        ) -> Any:
            fitted.append(k)
            if isinstance(llf[k], Exception):
                raise llf[k]
            return _fake_fit(llf[k], k)

        monkeypatch.setattr(test, "_fit_k1", fit_k1)
        monkeypatch.setattr(test, "_fit_k_regimes", fit_k)
        return fitted

    @staticmethod
    def _p_values(monkeypatch: Any, values: list[float]) -> list[float]:
        observed: list[float] = []
        iterator = iter(values)

        def fake_monte_carlo(
            lr: float, simulate: Any, statistic: Any, n: int, rng: Any
        ) -> Any:
            observed.append(lr)
            return MonteCarloPValue(
                p_value=next(iterator),
                statistics=np.array([1.0, 2.0]),
                n_valid=n,
                n_failed=0,
            )

        monkeypatch.setattr(module, "monte_carlo_pvalue", fake_monte_carlo)
        return observed

    def test_stops_at_the_first_test_that_does_not_reject(
        self, monkeypatch: Any
    ) -> None:
        test = RegimeNumberSelection(np.zeros(50), k_max=4, method="sequential")
        fitted = self._table(
            test, monkeypatch, {1: -100.0, 2: -80.0, 3: -79.0, 4: -60.0}
        )
        observed = self._p_values(monkeypatch, [0.01, 0.40])
        results = test.fit()
        assert observed == [40.0, 2.0]
        assert results.selected_k == 2
        # a model is estimated only when a test needs it
        assert fitted == [1, 2, 3]
        assert sorted(results.results_by_k) == [1, 2, 3]
        tests = results.sequential_tests
        assert tests is not None and len(tests) == 2
        assert [t["rejected"] for t in tests] == [True, False]
        assert tests[0]["n_extra_params"] == 3
        assert tests[0]["n_bootstrap"] == 199
        assert results.notes == []
        assert "K=2 vs K=3" in results.summary()
        assert "parametric bootstrap" in results.summary()

    def test_rejects_at_the_boundary(self, monkeypatch: Any) -> None:
        """p = 0.05 rejects at 5 percent (10 of 200 with the default 199)."""
        test = RegimeNumberSelection(np.zeros(50), k_max=3, method="sequential")
        self._table(test, monkeypatch, {1: -100.0, 2: -80.0, 3: -79.0})
        self._p_values(monkeypatch, [10 / 200, 11 / 200])
        results = test.fit()
        assert [t["rejected"] for t in results.sequential_tests] == [True, False]
        assert results.selected_k == 2

    def test_selected_number_equal_to_k_max_is_flagged(self, monkeypatch: Any) -> None:
        test = RegimeNumberSelection(np.zeros(50), k_max=2, method="sequential")
        self._table(test, monkeypatch, {1: -100.0, 2: -80.0})
        self._p_values(monkeypatch, [0.01])
        results = test.fit()
        assert results.selected_k == 2
        assert results.notes == ["2 regimes is k_max: it was not tested against 3"]
        assert "Note: 2 regimes is k_max" in results.summary()

    @pytest.mark.parametrize("method", ["bic", "sequential"])
    def test_model_that_cannot_be_estimated_is_reported(
        self, monkeypatch: Any, method: str
    ) -> None:
        """It used to be dropped silently, and the answer was a smaller K."""
        test = RegimeNumberSelection(np.zeros(50), k_max=4, method=method)
        fitted = self._table(
            test,
            monkeypatch,
            {1: -100.0, 2: -80.0, 3: np.linalg.LinAlgError("singular"), 4: -60.0},
        )
        self._p_values(monkeypatch, [0.01])
        with pytest.warns(UserWarning, match="3 regimes could not be estimated"):
            results = test.fit()
        assert fitted == [1, 2, 3]
        assert sorted(results.results_by_k) == [1, 2]
        assert len(results.notes) == 1
        assert "LinAlgError: singular" in results.notes[0]
        assert "could not be estimated" in results.summary()
        if method == "sequential":
            assert results.selected_k == 2
            assert len(results.sequential_tests) == 1

    def test_zero_statistic_needs_no_bootstrap(self, monkeypatch: Any) -> None:
        test = RegimeNumberSelection(np.zeros(50), k_max=2, method="sequential")
        self._table(test, monkeypatch, {1: -100.0, 2: -100.0})

        def no_bootstrap(*args: Any) -> Any:
            raise AssertionError("no bootstrap is needed for a statistic of zero")

        monkeypatch.setattr(module, "monte_carlo_pvalue", no_bootstrap)
        results = test.fit()
        assert results.selected_k == 1
        assert results.sequential_tests[0]["p_value"] == 1.0

    def test_no_valid_replication_raises(self, monkeypatch: Any) -> None:
        test = RegimeNumberSelection(np.zeros(50), k_max=2, method="sequential")
        self._table(test, monkeypatch, {1: -100.0, 2: -90.0})
        monkeypatch.setattr(
            module,
            "monte_carlo_pvalue",
            lambda *args: MonteCarloPValue(
                p_value=float("nan"),
                statistics=np.array([]),
                n_valid=0,
                n_failed=199,
                last_error=KeyError("exog"),
            ),
        )
        with pytest.raises(RuntimeError, match="any bootstrap replication") as info:
            test.fit()
        assert isinstance(info.value.__cause__, KeyError)

    def test_too_few_valid_replications_warn(self, monkeypatch: Any) -> None:
        test = RegimeNumberSelection(np.zeros(50), k_max=2, method="sequential")
        self._table(test, monkeypatch, {1: -100.0, 2: -90.0})
        monkeypatch.setattr(
            module,
            "monte_carlo_pvalue",
            lambda *args: MonteCarloPValue(
                p_value=1 / 11,
                statistics=np.arange(10.0),
                n_valid=10,
                n_failed=189,
            ),
        )
        with pytest.warns(UserWarning, match="too few for a test at level 0.05"):
            results = test.fit()
        assert not results.sequential_tests[0]["rejected"]

    def test_samples_and_statistic(self, monkeypatch: Any) -> None:
        """Samples come from the fitted null; the statistic refits both models."""
        y, x = _regression_data(seed=9, n=120)
        test = RegimeNumberSelection(
            y, k_max=2, method="sequential", exog=x, random_state=3
        )
        seen: dict[str, Any] = {}

        def fake_monte_carlo(
            lr: float, simulate: Any, statistic: Any, n: int, rng: Any
        ) -> Any:
            seen["observed"] = lr
            seen["samples"] = [simulate(rng) for _ in range(200)]
            seen["on_data"] = statistic(y)
            seen["on_sample"] = statistic(seen["samples"][0])
            return MonteCarloPValue(
                p_value=0.5, statistics=np.array([1.0]), n_valid=n, n_failed=0
            )

        monkeypatch.setattr(module, "monte_carlo_pvalue", fake_monte_carlo)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = test.fit()
        one = results.results_by_k[1]
        samples = np.array(seen["samples"])
        assert samples.shape == (200, 120)
        # drawn from y = b0 + b1 x + e with the estimated coefficients
        design = np.column_stack([np.ones(120), x])
        coefs = np.linalg.lstsq(design, samples.mean(axis=0), rcond=None)[0]
        np.testing.assert_allclose(coefs, one.params, atol=0.05)
        residual_variance = np.var(samples - design @ one.params)
        assert residual_variance == pytest.approx(one.sigma2, rel=0.05)
        # the statistic of the data is the observed one
        assert seen["on_data"] == pytest.approx(seen["observed"], abs=1e-3)
        assert seen["on_sample"] >= 0.0
        assert results.selected_k == 1

    def test_samples_of_a_one_regime_autoregression(self) -> None:
        """The null of ``model_type="ar"``: an autoregression around the mean,
        started from the first observations of the data."""
        rng = np.random.default_rng(15)
        n = 400
        y = np.full(n, 5.0)
        for t in range(2, n):
            y[t] = (
                5.0
                + 0.5 * (y[t - 1] - 5.0)
                + 0.3 * (y[t - 2] - 5.0)
                + rng.standard_normal()
            )
        test = RegimeNumberSelection(y, model_type="ar", order=2)
        null = test._fit_k1()["results"]
        assert null.ar_params[0] > 0.3
        draws = np.random.default_rng(16)
        estimates = []
        for _ in range(40):
            sample = test._simulate(null, draws)
            assert sample.shape == (n,)
            np.testing.assert_array_equal(sample[:2], y[:2])
            estimates.append(test._fit_k1(sample)["results"].ar_params)
        # white noise would give coefficients near zero
        np.testing.assert_allclose(
            np.mean(estimates, axis=0), null.ar_params, atol=0.04
        )
        assert np.mean(sample) == pytest.approx(5.0, abs=1.0)

    def test_every_sample_is_treated_like_the_data(self, monkeypatch: Any) -> None:
        """Two regimes against three: on a sample, as on the data, the model
        with two regimes is built up from one regime and the model with
        three from the two-regime fit of that sample."""
        rng = np.random.default_rng(10)
        y = np.concatenate([rng.normal(m, 1, 40) for m in (0, 3, 0, 3)])
        test = RegimeNumberSelection(y, k_max=3, method="sequential", random_state=1)
        test._rng = np.random.default_rng(1)
        calls: list[dict[str, Any]] = []
        original = test._fit_k_regimes

        def recording(
            k: int, lower: Any, one: Any, endog: Any = None, inference: bool = True
        ) -> Any:
            out = original(k, lower, one, endog, inference)
            calls.append(
                {
                    "k": k,
                    "lower": lower,
                    "one": one,
                    "endog": endog,
                    "inference": inference,
                    "results": out["results"],
                }
            )
            return out

        monkeypatch.setattr(test, "_fit_k_regimes", recording)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            on_data = test._fit_up_to(3, y, inference=True)
            sample = test._simulate(on_data[2]["results"], np.random.default_rng(2))
            del calls[:]
            statistic = test._lr_statistic(2, sample)
        assert statistic >= 0.0
        assert [c["k"] for c in calls] == [2, 3]
        for call in calls:
            assert call["endog"] is sample
            assert call["inference"] is False
            # the one-regime model of the sample, not of the data
            assert isinstance(call["one"], OneRegimeResults)
            assert call["one"] is calls[0]["lower"]
            assert call["one"].llf != on_data[1]["results"].llf
        # three regimes are started from the two-regime fit of the sample
        assert calls[1]["lower"] is calls[0]["results"]
        assert calls[1]["lower"] is not on_data[2]["results"]

    def test_end_to_end_and_reproducible(self) -> None:
        """A real bootstrap: two recurring regimes are found, twice the same."""
        rng = np.random.default_rng(11)
        y = np.concatenate([rng.normal(m, 1, 30) for m in (0, 3, 0, 3)])
        runs = []
        for _ in range(2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                runs.append(
                    RegimeNumberSelection(
                        y,
                        k_max=2,
                        method="sequential",
                        significance=0.10,
                        n_bootstrap=12,
                        random_state=5,
                    ).fit()
                )
        first, second = (run.sequential_tests[0] for run in runs)
        assert runs[0].selected_k == 2
        assert first["rejected"]
        assert first["p_value"] == pytest.approx(1 / 13)
        assert first["n_bootstrap"] == 12
        assert first["lr_statistic"] > first["critical_value"]
        assert first == second


def test_simulate_linear() -> None:
    x = np.column_stack([np.ones(4000), np.linspace(0, 1, 4000)])
    y = simulate_linear(
        x, np.array([1.0, 2.0]), 0.25, None, None, 4000, np.random.default_rng(10)
    )
    coefs, *_ = np.linalg.lstsq(x, y, rcond=None)
    np.testing.assert_allclose(coefs, [1.0, 2.0], atol=0.06)
    assert np.var(y - x @ coefs) == pytest.approx(0.25, abs=0.02)

    y = simulate_linear(
        np.ones((6000, 1)),
        np.array([5.0]),
        1.0,
        np.array([0.6]),
        np.array([5.0]),
        6000,
        np.random.default_rng(11),
    )
    assert np.mean(y) == pytest.approx(5.0, abs=0.15)
    dev = y - np.mean(y)
    assert (dev[1:] @ dev[:-1]) / (dev[:-1] @ dev[:-1]) == pytest.approx(0.6, abs=0.04)
