"""The parameter table must use the same regime labels as every other result.

After the regimes are relabeled (``ordering="first_appearance"`` or
``"intercept"``), ``params``, ``bse`` and ``param_names`` have to describe the
same regimes as ``regime_params``, the probabilities and the transition matrix.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from statsmodels.tsa.regime_switching.markov_regression import (
    MarkovRegression as SMMarkovRegression,
)

from regimes.markov import (
    MarkovADL,
    MarkovAR,
    MarkovRegression,
    RestrictedMarkovAR,
    RestrictedMarkovRegression,
)
from regimes.markov.models import (
    _extract_regime_params,
    _parameter_layout,
    _relabel_regimes,
    _relabeled_parameters,
)

N = 300


def _two_regime_data(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """High-mean regime first, then low, then high."""
    rng = np.random.default_rng(seed)
    state = np.r_[np.ones(100), np.zeros(100), np.ones(100)].astype(int)
    x = rng.normal(size=N)
    y = np.where(state == 1, 3.0 + 1.5 * x, -1.0 - 0.5 * x)
    return y + rng.normal(scale=0.5, size=N), x


def _three_regime_data(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = np.r_[np.full(100, 2), np.zeros(100), np.ones(100)].astype(int)
    return np.array([0.0, 4.0, 8.0])[state] + rng.normal(scale=0.6, size=N)


def _ar_data(seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 400
    mu = np.where(np.r_[np.ones(200), np.zeros(200)] == 1, 5.0, 0.0)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = mu[t] + 0.5 * (y[t - 1] - mu[t - 1]) + rng.normal(scale=0.5)
    return y, rng.normal(size=n)


def _fit(model: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit()


def _table(results: Any) -> dict[str, tuple[float, float]]:
    return {
        name: (float(results.params[i]), float(results.bse[i]))
        for i, name in enumerate(results.param_names)
    }


def _assert_consistent(
    results: Any, ordering: str | None, exog_names: list[str] | None = None
) -> None:
    """The invariants that tie the parameter table to the other results."""
    k = results.k_regimes
    table = _table(results)
    assert len(table) == len(results.param_names), "duplicate parameter names"

    # Per-regime coefficients equal the table entries with that regime label
    for r in range(k):
        assert results.regime_params[r], f"no parameters for regime {r}"
        for base, value in results.regime_params[r].items():
            name = f"{base}[{r}]" if f"{base}[{r}]" in table else base
            assert name in table, f"{base} of regime {r} is not in the table"
            assert table[name][0] == pytest.approx(value, abs=1e-12)

    # Transition probabilities: p[j->i] is the probability of moving from j to i
    for i in range(k - 1):
        for j in range(k):
            assert table[f"p[{j}->{i}]"][0] == pytest.approx(
                results.regime_transition[i, j], abs=1e-10
            )

    # The table is a relabeling of the statsmodels estimates
    sm = results._sm_results
    perm = _relabel_regimes(sm, ordering, k)
    layout = _parameter_layout(sm.model, exog_names)
    for ix, info in enumerate(layout):
        if info.kind != "coef":
            continue
        base = info.base
        name = base if info.regime is None else f"{base}[{int(perm[info.regime])}]"
        assert table[name][0] == pytest.approx(float(sm.params[ix]), abs=1e-12)
        np.testing.assert_allclose(table[name][1], float(sm.bse[ix]), equal_nan=True)


CASES = {
    "const": lambda o: MarkovRegression(_two_regime_data()[0], k_regimes=2, ordering=o),
    "switching_exog": lambda o: MarkovRegression(
        _two_regime_data()[0],
        k_regimes=2,
        exog=_two_regime_data()[1],
        switching_exog=True,
        ordering=o,
    ),
    "fixed_exog": lambda o: MarkovRegression(
        _two_regime_data()[0],
        k_regimes=2,
        exog=_two_regime_data()[1],
        switching_exog=False,
        ordering=o,
    ),
    "switching_variance": lambda o: MarkovRegression(
        _two_regime_data()[0],
        k_regimes=2,
        exog=_two_regime_data()[1],
        switching_variance=True,
        ordering=o,
    ),
    "three_regimes": lambda o: MarkovRegression(
        _three_regime_data(), k_regimes=3, ordering=o
    ),
    "ar": lambda o: MarkovAR(
        _ar_data()[0], k_regimes=2, order=1, switching_ar=True, ordering=o
    ),
    "adl": lambda o: MarkovADL(
        _ar_data()[0], _ar_data()[1], k_regimes=2, ar_order=1, exog_lags=1, ordering=o
    ),
}


@pytest.fixture(scope="module")
def fitted() -> dict[tuple[str, str | None], Any]:
    out: dict[tuple[str, str | None], Any] = {}
    for case, build in CASES.items():
        for ordering in ("first_appearance", "intercept", None):
            out[(case, ordering)] = _fit(build(ordering))
    return out


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("ordering", ["first_appearance", "intercept", None])
def test_table_matches_other_results(
    fitted: dict[tuple[str, str | None], Any], case: str, ordering: str | None
) -> None:
    exog_names = ["y.L1", "x0", "x0.L1"] if case == "adl" else None
    _assert_consistent(fitted[(case, ordering)], ordering, exog_names)


def test_relabeling_is_not_trivial_in_these_data(
    fitted: dict[tuple[str, str | None], Any],
) -> None:
    """Guard: the cases above must relabel, or they would test nothing."""
    res = fitted[("const", "first_appearance")]
    perm = _relabel_regimes(res._sm_results, "first_appearance", 2)
    assert list(perm) == [1, 0]
    res3 = fitted[("three_regimes", "first_appearance")]
    perm3 = _relabel_regimes(res3._sm_results, "first_appearance", 3)
    assert list(perm3) != [0, 1, 2]


def test_constant_of_regime_zero_is_const_0(
    fitted: dict[tuple[str, str | None], Any],
) -> None:
    """The case in which the table and regime_params disagreed."""
    res = fitted[("const", "first_appearance")]
    table = _table(res)
    assert res.regime_params[0]["const"] == pytest.approx(3.0, abs=0.3)
    assert table["const[0]"][0] == res.regime_params[0]["const"]
    assert table["const[1]"][0] == res.regime_params[1]["const"]
    # standard errors moved with their coefficients
    sm = res._sm_results
    names = list(sm.model.param_names)
    assert table["const[0]"][1] == float(sm.bse[names.index("const[1]")])
    assert table["const[1]"][1] == float(sm.bse[names.index("const[0]")])


def test_first_appearance_starts_in_regime_zero(
    fitted: dict[tuple[str, str | None], Any],
) -> None:
    for case in CASES:
        res = fitted[(case, "first_appearance")]
        assert res.smoothed_marginal_probabilities[0].argmax() == 0


def test_intercept_ordering_sorts_the_constants(
    fitted: dict[tuple[str, str | None], Any],
) -> None:
    for case in CASES:
        res = fitted[(case, "intercept")]
        consts = [res.regime_params[r]["const"] for r in range(res.k_regimes)]
        assert consts == sorted(consts), case
        table = _table(res)
        assert [table[f"const[{r}]"][0] for r in range(res.k_regimes)] == consts


def test_no_ordering_keeps_statsmodels_estimates(
    fitted: dict[tuple[str, str | None], Any],
) -> None:
    for case in CASES:
        res = fitted[(case, None)]
        np.testing.assert_array_equal(res.params, np.asarray(res._sm_results.params))
        np.testing.assert_array_equal(res.bse, np.asarray(res._sm_results.bse))


def test_intercept_ordering_needs_a_switching_constant() -> None:
    y, x = _two_regime_data()
    model = MarkovRegression(
        y,
        k_regimes=2,
        exog=x,
        switching_trend=False,
        switching_exog=True,
        ordering="intercept",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = model.fit()
    assert any("needs a constant" in str(w.message) for w in caught)
    np.testing.assert_array_equal(res.params, np.asarray(res._sm_results.params))


def test_intercept_ordering_uses_a_column_of_ones() -> None:
    """With trend="n" and ones in exog, statsmodels calls that column const."""
    y, x = _two_regime_data()
    X = np.column_stack([np.ones(N), x])
    res = _fit(
        MarkovRegression(
            y, k_regimes=2, exog=X, trend="n", switching_exog=True, ordering="intercept"
        )
    )
    assert res.regime_params[0]["const"] < res.regime_params[1]["const"]
    _assert_consistent(res, "intercept")


def test_coefficient_that_does_not_switch(
    fitted: dict[tuple[str, str | None], Any],
) -> None:
    """statsmodels names it ``x1[1]``; it applies in both regimes."""
    res = fitted[("fixed_exog", "first_appearance")]
    assert "x1[1]" in res._sm_results.model.param_names
    assert "x1" in res.param_names
    assert not any(name.startswith("x1[") for name in res.param_names)
    assert res.regime_params[0]["x1"] == res.regime_params[1]["x1"]
    raw = _extract_regime_params(res._sm_results, 2)
    assert raw[0]["x1"] == raw[1]["x1"]


def test_ar_params_follow_the_labels(
    fitted: dict[tuple[str, str | None], Any],
) -> None:
    for ordering in ("first_appearance", "intercept", None):
        res = fitted[("ar", ordering)]
        table = _table(res)
        for r in range(2):
            assert res.ar_params[r][0] == res.regime_params[r]["ar.L1"]
            assert res.ar_params[r][0] == table[f"ar.L1[{r}]"][0]


def test_adl_uses_its_own_names(fitted: dict[tuple[str, str | None], Any]) -> None:
    res = fitted[("adl", "first_appearance")]
    for name in ("y.L1[0]", "y.L1[1]", "x0[0]", "x0.L1[1]"):
        assert name in res.param_names
    assert set(res.regime_params[0]) == {"const", "y.L1", "x0", "x0.L1", "sigma2"}


def test_summary_and_dataframe_use_the_relabeled_table(
    fitted: dict[tuple[str, str | None], Any],
) -> None:
    res = fitted[("const", "first_appearance")]
    frame = res.to_dataframe()
    assert list(frame.index) == res.param_names
    assert frame.loc["const[0]", "coef"] == res.regime_params[0]["const"]
    text = res.summary()
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("const[0]"))
    assert f"{res.regime_params[0]['const']:.4f}" in line


class TestTransitionProbabilities:
    """Relabeling when statsmodels' last regime does not stay last."""

    @staticmethod
    def _fake(k: int = 3) -> tuple[Any, np.ndarray]:
        rng = np.random.default_rng(0)
        model = SMMarkovRegression(rng.normal(size=120), k_regimes=k)
        # p[j->i], destinations 0..k-2, statsmodels order (by destination, origin)
        p = {
            (0, 0): 0.80,
            (1, 0): 0.05,
            (2, 0): 0.10,
            (0, 1): 0.15,
            (1, 1): 0.90,
            (2, 1): 0.20,
        }
        params = np.array([p[(j, i)] for i in range(k - 1) for j in range(k)])
        params = np.r_[params, [1.0, 2.0, 3.0], 0.5]  # constants, sigma2
        a = rng.normal(size=(len(params), len(params)))
        cov = a @ a.T / 100
        fake = SimpleNamespace(
            model=model,
            params=params,
            bse=np.sqrt(np.diag(cov)),
            cov_params=lambda: cov,
        )
        return fake, cov

    @staticmethod
    def _raw_matrix(params: np.ndarray, k: int) -> np.ndarray:
        """T[i, j] = probability of moving from j to i."""
        t = np.zeros((k, k))
        t[: k - 1, :] = params[: k * (k - 1)].reshape(k - 1, k)
        t[k - 1, :] = 1.0 - t[: k - 1, :].sum(axis=0)
        return t

    @pytest.mark.parametrize(
        "perm", [[0, 1, 2], [1, 0, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1], [2, 1, 0]]
    )
    def test_probabilities_equal_the_permuted_matrix(self, perm: list[int]) -> None:
        fake, _cov = self._fake()
        k = 3
        p = np.array(perm, dtype=np.intp)
        names, params, _bse, regime_params = _relabeled_parameters(fake, p)
        raw = self._raw_matrix(fake.params, k)
        new = np.zeros_like(raw)
        for i in range(k):
            for j in range(k):
                new[p[i], p[j]] = raw[i, j]
        table = dict(zip(names, params, strict=True))
        assert names[: k * (k - 1)] == [
            f"p[{j}->{i}]" for i in range(k - 1) for j in range(k)
        ]
        for i in range(k - 1):
            for j in range(k):
                assert table[f"p[{j}->{i}]"] == pytest.approx(new[i, j], abs=1e-12)
        for old in range(k):
            assert regime_params[int(p[old])]["const"] == float(old + 1)

    def test_standard_error_of_a_derived_probability(self) -> None:
        fake, cov = self._fake()
        # statsmodels' last regime becomes regime 0: p[.->0] are derived
        p = np.array([1, 2, 0], dtype=np.intp)
        names, _params, bse, _ = _relabeled_parameters(fake, p)
        table = dict(zip(names, bse, strict=True))
        raw_names = list(fake.model.param_names)
        for origin_old in range(3):
            own = [raw_names.index(f"p[{origin_old}->{i}]") for i in range(2)]
            expected = np.sqrt(cov[np.ix_(own, own)].sum())
            assert table[f"p[{int(p[origin_old])}->0]"] == pytest.approx(expected)
        # estimated probabilities keep their own standard errors
        assert table["p[1->1]"] == pytest.approx(fake.bse[raw_names.index("p[0->0]")])

    def test_two_regimes_swapped(self) -> None:
        rng = np.random.default_rng(0)
        model = SMMarkovRegression(rng.normal(size=80), k_regimes=2)
        params = np.array([0.9, 0.2, -1.0, 3.0, 0.5])
        bse = np.array([0.01, 0.02, 0.1, 0.2, 0.05])
        fake = SimpleNamespace(
            model=model, params=params, bse=bse, cov_params=lambda: np.diag(bse**2)
        )
        names, new_params, new_bse, _ = _relabeled_parameters(
            fake, np.array([1, 0], dtype=np.intp)
        )
        assert names == ["p[0->0]", "p[1->0]", "const[0]", "const[1]", "sigma2"]
        # old p[1->0] = 0.2 is new p[0->1], so new p[0->0] = 0.8; same for the other
        np.testing.assert_allclose(new_params, [0.8, 0.1, 3.0, -1.0, 0.5])
        np.testing.assert_allclose(new_bse, [0.02, 0.01, 0.2, 0.1, 0.05])


class TestRestrictedModelsKeepTheirLabels:
    """Restrictions refer to the regime labels, so nothing is relabeled."""

    @staticmethod
    def _data() -> np.ndarray:
        rng = np.random.default_rng(5)
        return np.r_[rng.normal(1.0, 0.5, 120), rng.normal(4.0, 0.5, 120)]

    def test_ordering_is_ignored_with_a_warning(self) -> None:
        with pytest.warns(UserWarning, match="ordering is ignored"):
            model = RestrictedMarkovRegression(
                self._data(),
                k_regimes=2,
                restrictions={(0, 1): 0.0},
                ordering="intercept",
            )
        assert model.ordering is None

    def test_default_does_not_relabel(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            model = RestrictedMarkovRegression.non_recurring(self._data(), k_regimes=2)
            ar_model = RestrictedMarkovAR.non_recurring(
                self._data(), k_regimes=2, order=1
            )
        assert model.ordering is None
        assert ar_model.ordering is None

    def test_table_states_the_restriction(self) -> None:
        res = _fit(RestrictedMarkovRegression.non_recurring(self._data(), k_regimes=2))
        table = _table(res)
        # no return from regime 1 to regime 0
        assert table["p[1->0]"][0] == 0.0
        assert res.regime_transition[0, 1] == 0.0
        np.testing.assert_array_equal(res.params, np.asarray(res._sm_results.params))
        _assert_consistent(res, None)
