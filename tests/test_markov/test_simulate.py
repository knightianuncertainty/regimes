"""Tests for simulation from Markov switching models.

The central check is that the process the simulator draws from is the one
whose likelihood the models evaluate. For small samples the likelihood can
be computed directly, as a sum over every regime path of the probability of
the path times the density of the data given the path, with the timing
conventions of ``regimes.markov._simulate``. It must equal the likelihood
the model reports.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any

import numpy as np
import pytest
from scipy import stats
from statsmodels.tsa.regime_switching.markov_regression import (
    MarkovRegression as SMMarkovRegression,
)

from regimes.markov._simulate import (
    as_generator,
    can_reject,
    draw_regime_path,
    generate_series,
    monte_carlo_pvalue,
    presample_periods,
    rejects,
    replications_needed,
    simulate_markov_switching,
)
from regimes.markov.models import _SMMarkovAutoregression
from regimes.markov.restricted import (
    _RestrictedSMMarkovAutoregression,
    _RestrictedSMMarkovRegression,
)


def _direct_loglike(model: Any, params: Any) -> float:
    """Log-likelihood as a sum over all regime paths (simulator's timing)."""
    order = int(getattr(model, "order", 0))
    k = model.k_regimes
    transition = model.regime_transition_matrix(params)[..., 0]
    initial = model.initial_probabilities(params, transition)
    y = np.asarray(model.orig_endog if order else model.endog).ravel()
    exog = model.orig_exog if order else model.exog
    nobs = len(y)
    means = np.column_stack(
        [exog @ params[model.parameters[j, "exog"]] for j in range(k)]
    )
    sigma = np.sqrt(
        [float(np.squeeze(params[model.parameters[j, "variance"]])) for j in range(k)]
    )
    ar = (
        [params[model.parameters[j, "autoregressive"]] for j in range(k)]
        if order
        else None
    )
    # distribution of the regime of the first period of the sample
    start = np.linalg.matrix_power(transition, presample_periods(order)) @ initial
    total = 0.0
    for path in itertools.product(range(k), repeat=nobs):
        prob = start[path[0]]
        for t in range(1, nobs):
            prob *= transition[path[t], path[t - 1]]
        density = 1.0
        for t in range(order, nobs):
            mean = means[t, path[t]]
            for lag in range(1, order + 1):
                mean += ar[path[t]][lag - 1] * (  # type: ignore[index]
                    y[t - lag] - means[t - lag, path[t - lag]]
                )
            density *= stats.norm.pdf(y[t], mean, sigma[path[t]])
        total += prob * density
    return float(np.log(total))


def _regression_model(cls: Any, switching_variance: bool, seed: int) -> tuple[Any, Any]:
    rng = np.random.default_rng(seed)
    y = rng.normal(size=6)
    x = rng.normal(size=6)
    model = cls(y, k_regimes=2, exog=x, switching_variance=switching_variance)
    model.initialize_known([0.3, 0.7])
    params = model.start_params.copy()
    params[model.parameters["regime_transition"]] = [0.8, 0.35]
    if switching_variance:
        params[model.parameters["variance"]] = [0.5, 2.0]
    return model, params


def _ar_model(
    cls: Any, order: int, switching_variance: bool, seed: int
) -> tuple[Any, Any]:
    rng = np.random.default_rng(seed)
    y = rng.normal(size=6 + order)
    model = cls(
        y,
        k_regimes=2,
        order=order,
        switching_ar=True,
        switching_variance=switching_variance,
    )
    model.initialize_known([0.3, 0.7])
    params = model.start_params.copy()
    params[model.parameters["regime_transition"]] = [0.8, 0.35]
    params[model.parameters["exog"]] = [-0.5, 1.0]
    params[model.parameters["autoregressive"]] = rng.uniform(
        -0.5, 0.5, size=len(params[model.parameters["autoregressive"]])
    )
    if switching_variance:
        params[model.parameters["variance"]] = [0.5, 2.0]
    return model, params


class TestLikelihoodMatchesSimulatedProcess:
    """The model's likelihood is that of the process the simulator draws."""

    @pytest.mark.parametrize("switching_variance", [False, True])
    @pytest.mark.parametrize("cls", [SMMarkovRegression, _RestrictedSMMarkovRegression])
    def test_regression(self, cls: Any, switching_variance: bool) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, params = _regression_model(cls, switching_variance, seed=1)
            assert model.loglike(params) == pytest.approx(
                _direct_loglike(model, params), abs=1e-8
            )

    @pytest.mark.parametrize("switching_variance", [False, True])
    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize(
        "cls", [_SMMarkovAutoregression, _RestrictedSMMarkovAutoregression]
    )
    def test_autoregression(
        self, cls: Any, order: int, switching_variance: bool
    ) -> None:
        """Includes the variance of the current regime for order >= 2.

        statsmodels 0.14 and 0.15 use the variance of an earlier regime
        there; ``_SMMarkovAutoregression`` corrects it.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, params = _ar_model(cls, order, switching_variance, seed=2)
            assert model.loglike(params) == pytest.approx(
                _direct_loglike(model, params), abs=1e-8
            )

    def test_stationary_start(self) -> None:
        """With the stationary distribution the timing does not matter."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rng = np.random.default_rng(3)
            model = SMMarkovRegression(rng.normal(size=6), k_regimes=3)
            params = model.start_params.copy()
            params[model.parameters["regime_transition"]] = [
                0.7,
                0.2,
                0.1,
                0.1,
                0.6,
                0.3,
            ]
            assert model.loglike(params) == pytest.approx(
                _direct_loglike(model, params), abs=1e-8
            )


class TestMoreConfigurations:
    """Trend, regressors that do not switch, three regimes, real restrictions."""

    def _check(self, model: Any, params: Any) -> None:
        assert model.loglike(params) == pytest.approx(
            _direct_loglike(model, params), abs=1e-8
        )

    def _fill(self, model: Any, rng: np.random.Generator) -> Any:
        params = model.start_params.copy()
        k = model.k_regimes
        columns = rng.dirichlet(np.ones(k), size=k).T  # columns sum to one
        params[model.parameters["regime_transition"]] = columns[: k - 1, :].ravel()
        params[model.parameters["exog"]] = rng.normal(
            size=len(params[model.parameters["exog"]])
        )
        params[model.parameters["variance"]] = rng.uniform(
            0.5, 2.0, size=len(params[model.parameters["variance"]])
        )
        if getattr(model, "order", 0) > 0:
            params[model.parameters["autoregressive"]] = rng.uniform(
                -0.4, 0.4, size=len(params[model.parameters["autoregressive"]])
            )
        return params

    @pytest.mark.parametrize("trend", ["n", "c", "t", "ct"])
    @pytest.mark.parametrize("switching_exog", [True, False])
    def test_regression_trend_and_regressors(
        self, trend: str, switching_exog: bool
    ) -> None:
        rng = np.random.default_rng(40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SMMarkovRegression(
                rng.normal(size=6),
                k_regimes=2,
                trend=trend,
                exog=rng.normal(size=(6, 1)),
                switching_exog=switching_exog,
                switching_variance=True,
            )
            model.initialize_known([0.2, 0.8])
            self._check(model, self._fill(model, rng))

    def test_three_regimes_known_start(self) -> None:
        rng = np.random.default_rng(41)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SMMarkovRegression(rng.normal(size=5), k_regimes=3)
            model.initialize_known([0.5, 0.3, 0.2])
            self._check(model, self._fill(model, rng))

    @pytest.mark.parametrize("switching_ar", [True, False])
    def test_autoregression_with_trend_and_regressors(self, switching_ar: bool) -> None:
        rng = np.random.default_rng(42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _SMMarkovAutoregression(
                rng.normal(size=8),
                k_regimes=2,
                order=2,
                trend="ct",
                exog=rng.normal(size=(8, 1)),
                switching_ar=switching_ar,
                switching_variance=True,
            )
            model.initialize_known([0.6, 0.4])
            self._check(model, self._fill(model, rng))

    def test_three_regime_autoregression(self) -> None:
        rng = np.random.default_rng(43)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _SMMarkovAutoregression(
                rng.normal(size=6), k_regimes=3, order=1, switching_variance=True
            )
            model.initialize_known([0.5, 0.3, 0.2])
            self._check(model, self._fill(model, rng))

    def test_restricted_model_with_zeros(self) -> None:
        """The restricted filter (exact zeros) against the direct sum."""
        rng = np.random.default_rng(44)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _RestrictedSMMarkovRegression(
                rng.normal(size=6),
                3,
                restrictions={(2, 0): 0.0, (0, 1): 0.0, (0, 2): 0.0, (1, 2): 0.0},
                switching_variance=True,
            )
            model.initialize_known([1.0, 0.0, 0.0])
            params = self._fill(model, rng)
            # impose the restrictions on the parameter vector
            params = model.transform_params(model.untransform_params(params))
            matrix = model.regime_transition_matrix(params)[..., 0]
            assert matrix[2, 0] == 0.0 and matrix[0, 1] == 0.0 and matrix[2, 2] == 1.0
            self._check(model, params)

    def test_restricted_autoregression_with_zeros(self) -> None:
        rng = np.random.default_rng(45)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _RestrictedSMMarkovAutoregression(
                rng.normal(size=7), 2, 2, restrictions={(0, 1): 0.0}
            )
            model.initialize_known([1.0, 0.0])
            params = self._fill(model, rng)
            params = model.transform_params(model.untransform_params(params))
            self._check(model, params)

    def test_lags_are_applied_in_order(self) -> None:
        """Order 2: the first coefficient multiplies the first lag."""
        path = np.zeros(5, dtype=int)
        y = generate_series(
            path,
            np.zeros((5, 1)),
            np.array([0.0]),
            np.array([[0.5, 0.25]]),
            np.array([1.0, 2.0]),
            np.random.default_rng(0),
        )
        # y2 = 0.5 * 2 + 0.25 * 1; y3 = 0.5 * y2 + 0.25 * 2; ...
        expected = [1.0, 2.0, 1.25, 1.125, 0.875]
        np.testing.assert_allclose(y, expected)

    def test_autoregression_parameters_are_recovered(self) -> None:
        from regimes.markov import MarkovAR

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nobs = 3000
            model = _SMMarkovAutoregression(
                np.zeros(nobs), k_regimes=2, order=1, switching_ar=False
            )
            params = np.zeros(model.k_params)
            params[model.parameters["regime_transition"]] = [0.95, 0.08]
            params[model.parameters[0, "exog"]] = -1.0
            params[model.parameters[1, "exog"]] = 3.0
            params[model.parameters["autoregressive"]] = 0.5
            params[model.parameters["variance"]] = 0.5
            y = simulate_markov_switching(model, params, np.random.default_rng(46))
            fit = MarkovAR(
                y, k_regimes=2, order=1, switching_ar=False, ordering="intercept"
            ).fit()
        assert fit.regime_params[0]["const"] == pytest.approx(-1.0, abs=0.15)
        assert fit.regime_params[1]["const"] == pytest.approx(3.0, abs=0.2)
        assert fit.regime_params[0]["ar.L1"] == pytest.approx(0.5, abs=0.05)
        assert fit.regime_params[0]["sigma2"] == pytest.approx(0.5, abs=0.06)


class TestDrawRegimePath:
    def test_alternating_chain_shows_the_timing(self) -> None:
        transition = np.array([[0.0, 1.0], [1.0, 0.0]])
        initial = np.array([1.0, 0.0])
        rng = np.random.default_rng(0)
        one = draw_regime_path(transition, initial, 6, rng, n_presample=1)
        two = draw_regime_path(transition, initial, 6, rng, n_presample=2)
        np.testing.assert_array_equal(one, [1, 0, 1, 0, 1, 0])
        np.testing.assert_array_equal(two, [0, 1, 0, 1, 0, 1])

    def test_first_regime_frequency(self) -> None:
        transition = np.array([[0.6, 0.0], [0.4, 1.0]])
        initial = np.array([1.0, 0.0])
        rng = np.random.default_rng(1)
        n = 4000
        first = np.array(
            [
                draw_regime_path(transition, initial, 1, rng, n_presample=2)[0]
                for _ in range(n)
            ]
        )
        expected = 0.6**2
        se = np.sqrt(expected * (1 - expected) / n)
        assert abs(np.mean(first == 0) - expected) < 4 * se

    def test_absorbing_regime_is_never_left(self) -> None:
        transition = np.array([[0.9, 0.0], [0.1, 1.0]])
        path = draw_regime_path(
            transition, np.array([1.0, 0.0]), 500, np.random.default_rng(2)
        )
        assert np.all(np.diff(path) >= 0)

    def test_validation(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="probabilities"):
            draw_regime_path(np.array([[0.5, 0.5], [0.4, 0.5]]), [0.5, 0.5], 5, rng)
        with pytest.raises(ValueError, match="n_presample"):
            draw_regime_path(np.eye(2), [0.5, 0.5], 5, rng, n_presample=0)
        with pytest.raises(ValueError, match="one probability per regime"):
            draw_regime_path(np.eye(2), [1.0], 5, rng)


class TestGenerateSeries:
    def test_no_autoregression(self) -> None:
        path = np.array([0, 0, 1, 1, 1])
        means = np.column_stack([np.zeros(5), np.full(5, 10.0)])
        y = generate_series(
            path, means, np.array([0.0, 0.0]), None, None, np.random.default_rng(0)
        )
        np.testing.assert_allclose(y, [0, 0, 10, 10, 10])

    def test_mean_adjusted_recursion(self) -> None:
        """Without shocks the deviation from the regime mean decays."""
        path = np.array([0, 0, 0, 1, 1])
        means = np.column_stack([np.zeros(5), np.full(5, 10.0)])
        y = generate_series(
            path,
            means,
            np.array([0.0, 0.0]),
            np.array([[0.5], [0.5]]),
            np.array([4.0]),
            np.random.default_rng(0),
        )
        # deviations: 4, 2, 1, 0.5, 0.25 around means 0, 0, 0, 10, 10
        np.testing.assert_allclose(y, [4.0, 2.0, 1.0, 10.5, 10.25])

    def test_needs_initial_values(self) -> None:
        with pytest.raises(ValueError, match="initial_values"):
            generate_series(
                np.zeros(4, dtype=int),
                np.zeros((4, 1)),
                np.array([1.0]),
                np.array([[0.5, 0.2]]),
                np.array([1.0]),
                np.random.default_rng(0),
            )


class TestSimulateMarkovSwitching:
    def test_regression_shape_and_seed(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, params = _regression_model(SMMarkovRegression, True, seed=4)
        a = simulate_markov_switching(model, params, np.random.default_rng(5))
        b = simulate_markov_switching(model, params, np.random.default_rng(5))
        assert a.shape == (6,)
        np.testing.assert_array_equal(a, b)

    def test_autoregression_keeps_initial_observations(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, params = _ar_model(_SMMarkovAutoregression, 2, False, seed=6)
        y = simulate_markov_switching(model, params, np.random.default_rng(7))
        assert y.shape == (8,)
        np.testing.assert_array_equal(y[:2], np.asarray(model.orig_endog)[:2])

    def test_parameters_are_recovered(self) -> None:
        """A long simulated series returns the parameters it was drawn from."""
        from regimes.markov import MarkovRegression

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nobs = 3000
            model = SMMarkovRegression(
                np.zeros(nobs), k_regimes=2, switching_variance=True
            )
            params = np.zeros(model.k_params)
            params[model.parameters["regime_transition"]] = [0.95, 0.10]
            params[model.parameters[0, "exog"]] = -1.0
            params[model.parameters[1, "exog"]] = 2.0
            params[model.parameters["variance"]] = [0.25, 1.0]
            y = simulate_markov_switching(model, params, np.random.default_rng(8))
            fit = MarkovRegression(
                y, k_regimes=2, switching_variance=True, ordering="intercept"
            ).fit()
        assert fit.regime_params[0]["const"] == pytest.approx(-1.0, abs=0.06)
        assert fit.regime_params[1]["const"] == pytest.approx(2.0, abs=0.1)
        assert fit.regime_params[0]["sigma2"] == pytest.approx(0.25, abs=0.04)
        assert fit.regime_params[1]["sigma2"] == pytest.approx(1.0, abs=0.15)
        assert fit.regime_transition[0, 0] == pytest.approx(0.95, abs=0.02)
        assert fit.regime_transition[0, 1] == pytest.approx(0.10, abs=0.03)

    def test_non_recurring_model_does_not_return(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nobs = 300
            model = _RestrictedSMMarkovRegression(
                np.zeros(nobs), 2, restrictions={(0, 1): 0.0}
            )
            model.initialize_known([1.0, 0.0])
            params = np.zeros(model.k_params)
            params[model.parameters["regime_transition"]] = [0.98, 0.0]
            params[model.parameters[0, "exog"]] = 0.0
            params[model.parameters[1, "exog"]] = 50.0
            params[model.parameters["variance"]] = 1.0
            y = simulate_markov_switching(model, params, np.random.default_rng(9))
        high = y > 25
        # once in the second regime the series stays there
        assert np.all(np.diff(high.astype(int)) >= 0)


class TestMonteCarloPValue:
    def test_formula(self) -> None:
        draws = iter([0.5, 1.5, 2.5, 3.5])
        out = monte_carlo_pvalue(
            2.0,
            lambda rng: np.zeros(1),
            lambda sample: next(draws),
            4,
            np.random.default_rng(0),
        )
        # two of four simulated statistics are at least 2: (1 + 2) / (4 + 1)
        assert out.p_value == pytest.approx(0.6)
        assert out.n_valid == 4
        assert out.n_failed == 0
        # reject at 0.45 iff (1 + c) / 5 <= 0.45 iff c <= 1 iff the statistic
        # exceeds the second largest simulated value
        assert out.critical_value(0.45) == 2.5
        # at 0.4 the same: (1 + 1) / 5 = 0.4 rejects
        assert out.critical_value(0.4) == 2.5
        # at 0.25 and at 0.2 only c = 0 rejects: above the largest
        assert out.critical_value(0.25) == 3.5
        assert out.critical_value(0.2) == 3.5
        # below 1 / 5 nothing can reject
        assert out.critical_value(0.19) == float("inf")
        for level in (0.45, 0.4, 0.25, 0.2, 0.19):
            for observed in (0.4, 1.0, 2.0, 3.0, 4.0):
                p_value = (1 + sum(v >= observed for v in (0.5, 1.5, 2.5, 3.5))) / 5
                assert rejects(p_value, level) == (observed > out.critical_value(level))

    def test_decision_rule(self) -> None:
        """Reject when p <= level: exact with 19 replications at 5 percent."""
        assert rejects(0.05, 0.05)
        assert rejects(10 / 200, 0.05)
        assert rejects(3 / 30, 0.1)
        assert not rejects(0.0501, 0.05)
        assert not rejects(float("nan"), 0.05)
        assert can_reject(19, 0.05) and not can_reject(18, 0.05)
        assert can_reject(99, 0.01) and not can_reject(98, 0.01)
        assert [replications_needed(a) for a in (0.1, 0.05, 0.025, 0.01)] == [
            9,
            19,
            39,
            99,
        ]
        # the level of the test under the null: floor(alpha (N + 1)) / (N + 1)
        rng = np.random.default_rng(0)
        draws = rng.standard_normal((20000, 20))
        p_values = (1 + (draws[:, 1:] >= draws[:, :1]).sum(axis=1)) / 20
        rate = np.mean([rejects(p, 0.05) for p in p_values])
        assert abs(rate - 0.05) < 0.006

    def test_ties_count_against_the_null(self) -> None:
        out = monte_carlo_pvalue(
            0.0,
            lambda rng: np.zeros(1),
            lambda sample: 0.0,
            9,
            np.random.default_rng(0),
        )
        assert out.p_value == 1.0

    def test_failed_replications_are_left_out(self) -> None:
        calls = {"n": 0}

        def statistic(sample: Any) -> float:
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise RuntimeError("no convergence")
            if calls["n"] == 1:
                return float("nan")
            return 5.0

        with pytest.warns(UserWarning, match="could not be computed"):
            out = monte_carlo_pvalue(
                1.0,
                lambda rng: np.zeros(1),
                statistic,
                10,
                np.random.default_rng(0),
            )
        assert out.n_failed == 6
        assert out.n_valid == 4
        assert out.p_value == pytest.approx(1.0)

    def test_all_failed(self) -> None:
        def statistic(sample: Any) -> float:
            raise RuntimeError("no convergence")

        with pytest.warns(UserWarning):
            out = monte_carlo_pvalue(
                1.0, lambda rng: np.zeros(1), statistic, 3, np.random.default_rng(0)
            )
        assert np.isnan(out.p_value)
        assert np.isnan(out.critical_value(0.05))
        assert isinstance(out.last_error, RuntimeError)

    def test_needs_a_replication(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            monte_carlo_pvalue(
                1.0, lambda rng: np.zeros(1), lambda s: 0.0, 0, np.random.default_rng(0)
            )


def test_as_generator() -> None:
    generator = np.random.default_rng(1)
    assert as_generator(generator) is generator
    a = as_generator(12).random()
    b = as_generator(12).random()
    assert a == b
    assert isinstance(as_generator(None), np.random.Generator)
