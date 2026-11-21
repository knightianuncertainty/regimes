"""Restricted transition matrices: layout, starting regime, inference."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from statsmodels.tools.numdiff import approx_hess

from regimes.markov import RestrictedMarkovAR, RestrictedMarkovRegression
from regimes.markov.restricted import (
    _non_recurring_restrictions,
    _RestrictedSMMarkovRegression,
)
from regimes.markov.sequential_restriction import NonRecurringRegimeTest


def _shifts(means: list[float], n_each: int, seed: int, sd: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(m, sd, n_each) for m in means])


def _fit(model: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(**kwargs)


@pytest.fixture(scope="module")
def three_regime_fit() -> Any:
    """Levels 0 -> 4 -> 8, non-recurring model with three regimes."""
    y = _shifts([0, 4, 8], 100, seed=0)
    return _fit(RestrictedMarkovRegression.non_recurring(y, k_regimes=3))


@pytest.fixture(scope="module")
def two_regime_fit() -> Any:
    y = _shifts([0, 3], 100, seed=10)
    return _fit(RestrictedMarkovRegression.non_recurring(y, k_regimes=2))


class TestLayout:
    """statsmodels stores p[j->i] destination by destination."""

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_transition_index_matches_the_names(self, k: int) -> None:
        model = _RestrictedSMMarkovRegression(
            np.random.default_rng(0).normal(size=80), k, restrictions={}
        )
        index = model._transition_index()
        assert index.shape == (k - 1, k)
        for i in range(k - 1):
            for j in range(k):
                assert model.param_names[index[i, j]] == f"p[{j}->{i}]"

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_transform_imposes_the_restrictions(self, k: int) -> None:
        restrictions = _non_recurring_restrictions(k)
        model = _RestrictedSMMarkovRegression(
            np.random.default_rng(0).normal(size=80), k, restrictions=restrictions
        )
        rng = np.random.default_rng(1)
        for _ in range(20):
            params = model.transform_params(rng.normal(size=model.k_params))
            matrix = model.regime_transition_matrix(params)[..., 0]
            np.testing.assert_allclose(matrix.sum(axis=0), 1.0, atol=1e-12)
            assert np.all(matrix >= 0)
            for (i, j), value in restrictions.items():
                assert matrix[i, j] == value
            # the allowed moves are open
            for j in range(k - 1):
                assert matrix[j, j] > 0 and matrix[j + 1, j] > 0
            assert matrix[k - 1, k - 1] == 1.0

    def test_other_fixed_values(self) -> None:
        model = _RestrictedSMMarkovRegression(
            np.random.default_rng(0).normal(size=80),
            3,
            restrictions={(2, 0): 0.25, (0, 2): 0.0},
        )
        params = model.transform_params(np.random.default_rng(2).normal(size=10))
        matrix = model.regime_transition_matrix(params)[..., 0]
        assert matrix[2, 0] == pytest.approx(0.25)
        assert matrix[0, 2] == 0.0
        np.testing.assert_allclose(matrix.sum(axis=0), 1.0, atol=1e-12)


class TestFreeParameters:
    def test_no_restrictions_all_free(self) -> None:
        model = _RestrictedSMMarkovRegression(
            np.random.default_rng(0).normal(size=80), 3, restrictions={}
        )
        jac = model._free_parameter_jacobian()
        assert jac.shape == (model.k_params, model.k_params)
        np.testing.assert_array_equal(jac @ jac.T, np.eye(model.k_params))

    def test_non_recurring_three_regimes(self) -> None:
        model = _RestrictedSMMarkovRegression(
            np.random.default_rng(0).normal(size=80),
            3,
            restrictions=_non_recurring_restrictions(3),
        )
        jac = model._free_parameter_jacobian()
        # p[0->0], p[1->1], three constants and the variance
        assert jac.shape == (10, 6)
        names = list(model.param_names)
        for fixed in ("p[1->0]", "p[2->0]", "p[2->1]"):
            assert not jac[names.index(fixed)].any()
        # p[0->1] = 1 - p[0->0]
        col = jac[:, np.flatnonzero(jac[names.index("p[0->0]")])[0]]
        assert col[names.index("p[0->0]")] == 1.0
        assert col[names.index("p[0->1]")] == -1.0


class TestThreeRegimes:
    """The case that failed: likelihood -663.9 where -441.0 is attainable."""

    def test_likelihood_and_estimates(self, three_regime_fit: Any) -> None:
        res = three_regime_fit
        assert res.llf > -445.0
        consts = [res.regime_params[j]["const"] for j in range(3)]
        np.testing.assert_allclose(consts, [0.0, 4.0, 8.0], atol=0.3)

    def test_transition_matrix(self, three_regime_fit: Any) -> None:
        matrix = three_regime_fit.regime_transition
        np.testing.assert_allclose(matrix.sum(axis=0), 1.0, atol=1e-10)
        assert np.all(matrix >= 0)
        for (i, j), value in _non_recurring_restrictions(3).items():
            assert matrix[i, j] == value
        # the matrix reported is the one the filter used
        used = np.asarray(three_regime_fit._sm_results.regime_transition).squeeze()
        np.testing.assert_array_equal(matrix, used)

    def test_regimes_come_in_order(self, three_regime_fit: Any) -> None:
        path = three_regime_fit.smoothed_marginal_probabilities.argmax(axis=1)
        assert path[0] == 0 and path[150] == 1 and path[-1] == 2
        assert np.all(np.diff(path) >= 0)

    def test_free_parameters_and_standard_errors(self, three_regime_fit: Any) -> None:
        res = three_regime_fit
        assert len(res.params) == 10
        assert res.df_model == 6
        assert res.aic == pytest.approx(-2 * res.llf + 2 * 6)
        table = dict(zip(res.param_names, res.bse, strict=True))
        for fixed in ("p[1->0]", "p[2->0]", "p[2->1]"):
            assert np.isnan(table[fixed])
        for name in ("p[0->0]", "p[1->1]", "const[0]", "const[1]", "const[2]"):
            assert np.isfinite(table[name]) and table[name] > 0
        # p[0->1] = 1 - p[0->0]: same standard error
        assert table["p[0->1]"] == pytest.approx(table["p[0->0]"])


class TestStartingRegime:
    def test_non_recurring_starts_in_regime_zero(self, two_regime_fit: Any) -> None:
        res = two_regime_fit
        assert res.smoothed_marginal_probabilities[0, 0] == pytest.approx(1.0)
        consts = [res.regime_params[j]["const"] for j in range(2)]
        np.testing.assert_allclose(consts, [0.0, 3.0], atol=0.4)
        # the break is found: regime 0 for the first half, regime 1 after
        path = res.smoothed_marginal_probabilities.argmax(axis=1)
        assert abs(int(np.argmax(path == 1)) - 100) <= 3

    def test_default_start(self) -> None:
        y = _shifts([0, 3], 60, seed=1)
        # every regime reachable from every other: stationary distribution
        open_chain = RestrictedMarkovRegression(
            y, k_regimes=3, restrictions={(2, 0): 0.0}
        )
        assert open_chain.initial_probabilities is None
        # regime 1 absorbing: start in regime 0, the one that cannot be re-entered
        absorbing = RestrictedMarkovRegression(
            y, k_regimes=2, restrictions={(0, 1): 0.0}
        )
        np.testing.assert_array_equal(absorbing.initial_probabilities, [1.0, 0.0])
        # two regimes that feed a third, absorbing one
        fork = RestrictedMarkovRegression(
            y,
            k_regimes=3,
            restrictions={(1, 0): 0.0, (0, 1): 0.0, (0, 2): 0.0, (1, 2): 0.0},
        )
        np.testing.assert_array_equal(fork.initial_probabilities, [0.5, 0.5, 0.0])

    def test_hand_written_restriction_finds_the_break(self) -> None:
        """Same model as non_recurring(), written out; it used to be degenerate."""
        y = _shifts([0, 3], 100, seed=10)
        res = _fit(
            RestrictedMarkovRegression(y, k_regimes=2, restrictions={(0, 1): 0.0})
        )
        consts = [res.regime_params[j]["const"] for j in range(2)]
        np.testing.assert_allclose(consts, [0.0, 3.0], atol=0.4)
        assert 0.9 < res.regime_transition[0, 0] < 1.0

    def test_invalid_initial_probabilities(self) -> None:
        y = _shifts([0, 3], 60, seed=1)
        for bad in ([0.5, 0.6], [1.0], [-0.1, 1.1]):
            with pytest.raises(ValueError, match="initial_probabilities"):
                RestrictedMarkovRegression(
                    y, k_regimes=2, restrictions={}, initial_probabilities=bad
                )

    def test_invalid_restrictions(self) -> None:
        y = _shifts([0, 3], 60, seed=1)
        with pytest.raises(ValueError, match="outside"):
            RestrictedMarkovRegression(y, k_regimes=2, restrictions={(2, 0): 0.0})
        with pytest.raises(ValueError, match="not a probability"):
            RestrictedMarkovRegression(y, k_regimes=2, restrictions={(0, 0): 1.5})
        with pytest.raises(ValueError, match="sum to"):
            RestrictedMarkovRegression(
                y, k_regimes=2, restrictions={(0, 0): 0.3, (1, 0): 0.3}
            )


class TestStandardErrors:
    def test_against_finite_differences_in_the_free_parameters(
        self, two_regime_fit: Any
    ) -> None:
        """Independent check: real finite differences instead of complex steps."""
        res = two_regime_fit
        sm = res._sm_results
        model = sm.model
        params = np.asarray(sm.params, dtype=float)
        jac = model._free_parameter_jacobian()

        def loglike(theta: np.ndarray) -> float:
            return float(model.loglike(params + jac @ theta, transformed=True))

        hessian = approx_hess(np.zeros(jac.shape[1]), loglike)
        cov = jac @ (-np.linalg.inv(hessian)) @ jac.T
        expected = np.sqrt(np.diag(cov))
        estimated = np.isfinite(res.bse)
        assert estimated.sum() == 4  # p[0->0], two constants, the variance
        np.testing.assert_allclose(res.bse[estimated], expected[estimated], rtol=5e-3)

    def test_two_regimes_count(self, two_regime_fit: Any) -> None:
        res = two_regime_fit
        assert len(res.params) == 5
        assert res.df_model == 4
        table = dict(zip(res.param_names, res.bse, strict=True))
        assert np.isnan(table["p[1->0]"])


class TestNonRecurringRegimeTest:
    def test_true_null_is_not_rejected(self) -> None:
        for seed in (10, 11):
            y = _shifts([0, 3], 100, seed=seed)
            result = NonRecurringRegimeTest(y, k_regimes=2).fit()
            assert result.lr_statistic < 1.0
            assert not result.rejected
            # nested models: the unrestricted likelihood is not lower
            assert result.llf_unrestricted >= result.llf_restricted - 1e-4

    def test_recurring_regimes_are_rejected(self) -> None:
        y = _shifts([0, 3, 0, 3], 60, seed=20)
        result = NonRecurringRegimeTest(y, k_regimes=2).fit()
        assert result.lr_statistic > 50
        assert result.rejected


class TestRestrictedAR:
    def test_non_recurring_ar(self) -> None:
        rng = np.random.default_rng(4)
        n = 240
        mu = np.r_[np.zeros(120), np.full(120, 4.0)]
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = mu[t] + 0.4 * (y[t - 1] - mu[t - 1]) + rng.normal(scale=0.6)
        res = _fit(RestrictedMarkovAR.non_recurring(y, k_regimes=2, order=1))
        assert res.regime_transition[0, 1] == 0.0
        assert res.regime_transition[1, 1] == 1.0
        assert res.df_model == len(res.params) - 1
        consts = [res.regime_params[j]["const"] for j in range(2)]
        np.testing.assert_allclose(consts, [0.0, 4.0], atol=0.6)
        assert res.smoothed_marginal_probabilities[0, 0] == pytest.approx(1.0)


class TestStartingValues:
    """The regimes of a non-recurring model follow one another in time."""

    @pytest.mark.parametrize(
        ("means", "k"),
        [([3.0, 0.0], 2), ([8.0, 0.0, 4.0], 3), ([0.0, 4.0, 8.0], 3)],
    )
    def test_first_regime_need_not_be_the_lowest(
        self, means: list[float], k: int
    ) -> None:
        y = _shifts(means, 80, seed=7)
        res = _fit(RestrictedMarkovRegression.non_recurring(y, k_regimes=k))
        consts = [res.regime_params[j]["const"] for j in range(k)]
        np.testing.assert_allclose(consts, means, atol=0.4)
        path = res.smoothed_marginal_probabilities.argmax(axis=1)
        for j in range(1, k):
            assert abs(int(np.argmax(path == j)) - 80 * j) <= 3

    def test_regimes_of_unequal_length(self) -> None:
        y = np.r_[_shifts([5.0], 30, seed=1), _shifts([0.0], 210, seed=2)]
        res = _fit(RestrictedMarkovRegression.non_recurring(y, k_regimes=2))
        path = res.smoothed_marginal_probabilities.argmax(axis=1)
        assert abs(int(np.argmax(path == 1)) - 30) <= 2

    def test_untransform_inverts_transform_on_free_entries(self) -> None:
        model = _RestrictedSMMarkovRegression(
            np.random.default_rng(0).normal(size=80),
            3,
            restrictions=_non_recurring_restrictions(3),
        )
        params = model.transform_params(np.random.default_rng(3).normal(size=10))
        again = model.transform_params(model.untransform_params(params))
        np.testing.assert_allclose(again, params, atol=1e-7)
