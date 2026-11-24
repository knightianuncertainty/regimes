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
from regimes.markov.models import _SMMarkovAutoregression
from regimes.markov.selection import (
    OneRegimeResults,
    RegimeNumberSelection,
    _fit_one_regime,
    _split_starts,
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
        y = np.zeros(30)
        with pytest.raises(ValueError, match="method must be one of"):
            RegimeNumberSelection(y, method="lrt")
        with pytest.raises(ValueError, match="model_type"):
            RegimeNumberSelection(y, model_type="adl")
        with pytest.raises(ValueError, match="could never reject"):
            RegimeNumberSelection(y, method="sequential", n_bootstrap=19)
        RegimeNumberSelection(y, method="bic", n_bootstrap=5)


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
        # the maximum is at least the value MSTest's EM algorithm stops at
        assert table.loc[2, "llf"] >= case["llf"]["k2"] - 1e-6
        assert table.loc[2, "llf"] == pytest.approx(case["llf"]["k2"], abs=0.05)
        lr = 2 * (table.loc[2, "llf"] - table.loc[1, "llf"])
        assert lr == pytest.approx(case["lr"]["k1_vs_k2"], abs=0.1)
        assert results.selected_k == 2


class _Fit:
    def __init__(self, llf: float) -> None:
        self.llf = llf


class TestSequentialBootstrap:
    def _table(
        self, test: RegimeNumberSelection, monkeypatch: Any, llf: dict[int, float]
    ) -> list[Any]:
        calls: list[Any] = []

        def fake_fit(
            k: int, endog: Any = None, lower: Any = None, starts: Any = ()
        ) -> Any:
            calls.append((k, endog is not None, lower is not None, len(starts)))
            return {
                "results": OneRegimeResults(
                    params=np.zeros(1),
                    param_names=["const"],
                    ar_params=np.zeros(0),
                    sigma2=1.0,
                    llf=llf[k],
                    nobs=50,
                    df_model=1 + 3 * (k - 1),
                ),
                "llf": llf[k],
                "aic": 0.0,
                "bic": 0.0,
                "hqic": 0.0,
                "n_params": 1 + 3 * (k - 1),
            }

        monkeypatch.setattr(test, "_fit", fake_fit)
        monkeypatch.setattr(test, "_fit_k1", lambda endog=None: fake_fit(1, endog))
        monkeypatch.setattr(
            test,
            "_fit_k_regimes",
            lambda k, endog=None, lower=None, starts=(): fake_fit(
                k, endog, lower, starts
            ),
        )
        return calls

    def test_stops_at_the_first_test_that_does_not_reject(
        self, monkeypatch: Any
    ) -> None:
        test = RegimeNumberSelection(np.zeros(50), k_max=4, method="sequential")
        self._table(test, monkeypatch, {1: -100.0, 2: -80.0, 3: -79.0, 4: -60.0})
        p_values = iter([0.01, 0.40])
        observed: list[float] = []

        def fake_monte_carlo(
            lr: float, simulate: Any, statistic: Any, n: int, rng: Any
        ) -> Any:
            observed.append(lr)
            return MonteCarloPValue(
                p_value=next(p_values),
                statistics=np.array([1.0, 2.0]),
                n_valid=n,
                n_failed=0,
            )

        monkeypatch.setattr(module, "monte_carlo_pvalue", fake_monte_carlo)
        results = test.fit()
        assert observed == [40.0, 2.0]
        assert results.selected_k == 2
        tests = results.sequential_tests
        assert tests is not None and len(tests) == 2
        assert [t["rejected"] for t in tests] == [True, False]
        assert tests[0]["n_extra_params"] == 3
        assert tests[0]["n_bootstrap"] == 199
        assert "K=2 vs K=3" in results.summary()
        assert "parametric bootstrap" in results.summary()

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

    def test_sample_from_a_markov_null_starts_from_its_parameters(
        self, monkeypatch: Any
    ) -> None:
        rng = np.random.default_rng(10)
        y = np.concatenate([rng.normal(m, 1, 40) for m in (0, 3, 0, 3)])
        test = RegimeNumberSelection(y, k_max=3, method="sequential", random_state=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            null = test._fit_k_regimes(2, lower=test._fit_k1()["results"])["results"]
        calls: list[Any] = []
        original = test._fit

        def recording(
            k: int, endog: Any = None, lower: Any = None, starts: Any = ()
        ) -> Any:
            calls.append((k, lower is not None, [np.asarray(s) for s in starts]))
            return original(k, endog, lower, starts)

        monkeypatch.setattr(test, "_fit", recording)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sample = test._simulate(null, np.random.default_rng(2))
            statistic = test._lr_statistic(2, sample, null)
        assert statistic >= 0.0
        assert [c[0] for c in calls] == [2, 3]
        # the two-regime model is also started from the parameters of the null
        np.testing.assert_array_equal(
            calls[0][2][0], np.asarray(null._sm_results.params)
        )
        # the three-regime model from the two-regime fit of the sample
        assert calls[1][1] is True

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
