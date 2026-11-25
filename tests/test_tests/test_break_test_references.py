"""Chow, CUSUM and Andrews-Ploberger tests against other implementations.

References:

* R ``strucchange`` (``Fstats``, ``sctest``) for the sequence of F
  statistics, the sup/ave/exp statistics and their p-values
  (``reference_data/ap_strucchange_reference.json``).
* Hansen's own p-value programs (``reference_data/hansen1997_pvalues.csv``).
* Published critical values.
* statsmodels ``RecursiveLS`` for the recursive residuals, the CUSUM and
  CUSUM-of-squares paths and the CUSUM-of-squares bounds.
* statsmodels ``OLS`` F tests for the Chow test.
* ``BaiPerronTest`` for the SupF statistic with one break.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.recursive_ls import RecursiveLS

import regimes as rg
from regimes.tests._hansen_pvalues import (
    hansen_critical_value,
    hansen_pvalue,
    pi0_from_range,
)
from regimes.tests.cusum import _compute_recursive_residuals, cusum_sq_critical_value

REF = Path(__file__).resolve().parents[1] / "reference_data"


@pytest.fixture(scope="module")
def strucchange() -> dict[str, Any]:
    with (REF / "ap_strucchange_reference.json").open(encoding="utf-8") as fh:
        return {case["name"]: case for case in json.load(fh)["cases"]}


def _case_data(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name.startswith("mean"):
        data = pd.read_csv(REF / "dgp1_mean_shift.csv")
        X = np.ones((len(data), 1))
    else:
        data = pd.read_csv(REF / "dgp2_regression_break.csv")
        X = np.column_stack([np.ones(len(data)), data["x"].to_numpy()])
    y = data["y"].to_numpy()
    if "first_half" in name:
        y, X = y[:100], X[:100]
    return y, X


CASES = [
    "mean_shift_015",
    "mean_shift_010",
    "regression_break_015",
    "mean_first_half_015",
    "mean_first_half_005",
    "regression_first_half_015",
    "regression_first_half_020",
]


class TestAndrewsPlobergerAgainstStrucchange:
    @pytest.mark.parametrize("name", CASES)
    def test_statistics(self, strucchange: dict[str, Any], name: str) -> None:
        case = strucchange[name]
        y, X = _case_data(name)
        results = rg.AndrewsPlobergerTest(y, exog_break=X).fit(trimming=case["from"])

        assert results.q == case["q"]
        # same candidate dates: floor(trimming * T) observations at each end
        assert results.candidate_indices[0] == case["first_break"]
        assert results.candidate_indices[-1] == case["last_break"]
        np.testing.assert_allclose(
            results.f_sequence, case["f_sequence"], rtol=1e-9, atol=1e-9
        )
        assert results.sup_f == pytest.approx(case["supF_statistic"], rel=1e-10)
        assert results.ave_f == pytest.approx(case["aveF_statistic"], rel=1e-10)
        assert results.sup_f_break_index == case["breakpoint"]
        if case["expF_statistic"] is not None:
            # strucchange overflows for a large break; here it is computed
            assert results.exp_f == pytest.approx(case["expF_statistic"], rel=1e-10)

    @pytest.mark.parametrize("name", [c for c in CASES if "first_half" in c])
    def test_pvalues(self, strucchange: dict[str, Any], name: str) -> None:
        case = strucchange[name]
        y, X = _case_data(name)
        results = rg.AndrewsPlobergerTest(y, exog_break=X).fit(trimming=case["from"])
        assert results.sup_f_pvalue == pytest.approx(case["supF_pvalue"], abs=1e-10)
        assert results.ave_f_pvalue == pytest.approx(case["aveF_pvalue"], abs=1e-10)
        assert results.exp_f_pvalue == pytest.approx(case["expF_pvalue"], abs=1e-10)

    def test_large_break_has_a_small_pvalue(self, strucchange: dict[str, Any]) -> None:
        """Where Hansen's formula turns back, the p-value stays small.

        For this case strucchange reports p = 1 for the ave statistic of
        388 with two coefficients: the published polynomial is past its
        turning point.
        """
        case = strucchange["regression_break_015"]
        assert case["aveF_pvalue"] == 1
        y, X = _case_data("regression_break_015")
        results = rg.AndrewsPlobergerTest(y, exog_break=X).fit(trimming=0.15)
        assert results.ave_f_pvalue < 1e-3
        assert results.exp_f_pvalue < 1e-3
        assert results.sup_f_pvalue < 1e-6
        assert all(results.rejected.values())


class TestHansenPValues:
    def test_against_hansens_program(self) -> None:
        ref = pd.read_csv(REF / "hansen1997_pvalues.csv")
        ours = np.array(
            [
                hansen_pvalue(row.statistic, int(row.q), row.pi0, row.kind)
                for row in ref.itertuples()
            ]
        )
        np.testing.assert_allclose(ours, ref["pvalue"].to_numpy(), atol=1e-9)

    def test_published_critical_values(self) -> None:
        """One coefficient, pi_0 = 0.15, 5 percent.

        SupF 8.68 (Andrews, 2003, corrigendum); ExpF 2.06 and AveF 2.88
        (Andrews and Ploberger, 1994). Hansen's approximation is accurate
        to about 0.1.
        """
        assert hansen_critical_value(0.05, 1, 0.15, "sup") == pytest.approx(
            8.68, abs=0.1
        )
        assert hansen_critical_value(0.05, 1, 0.15, "exp") == pytest.approx(
            2.06, abs=0.05
        )
        assert hansen_critical_value(0.05, 1, 0.15, "ave") == pytest.approx(
            2.88, abs=0.05
        )

    @pytest.mark.parametrize("kind", ["sup", "exp", "ave"])
    @pytest.mark.parametrize("q", [1, 2, 3, 4, 10, 40])
    def test_critical_value_inverts_the_pvalue(self, kind: str, q: int) -> None:
        for pi0 in (0.05, 0.15, 0.3):
            for alpha in (0.10, 0.05, 0.01):
                cv = hansen_critical_value(alpha, q, pi0, kind)  # type: ignore[arg-type]
                assert hansen_pvalue(cv, q, pi0, kind) == pytest.approx(  # type: ignore[arg-type]
                    alpha, abs=1e-8
                )

    @pytest.mark.parametrize("kind", ["sup", "exp", "ave"])
    @pytest.mark.parametrize("q", [1, 2, 3, 4, 7])
    def test_pvalue_never_increases_with_the_statistic(self, kind: str, q: int) -> None:
        grid = np.concatenate([np.linspace(0, 60, 601), np.linspace(60, 5000, 200)])
        for pi0 in (0.01, 0.05, 0.15, 0.25, 0.45):
            p = np.array([hansen_pvalue(x, q, pi0, kind) for x in grid])  # type: ignore[arg-type]
            assert np.all(np.diff(p) <= 1e-12), (kind, q, pi0)
            assert p[-1] < 1e-3

    def test_tail_guard_differs_from_the_published_formula(self) -> None:
        """Hansen's own program gives p = 1 here (see its output)."""
        assert hansen_pvalue(31.0, 2, 0.01, "ave") < 1e-3

    def test_single_date_is_the_chi_squared_test(self) -> None:
        for q in (1, 3):
            assert hansen_pvalue(5.0, q, 0.5, "sup") == pytest.approx(
                stats.chi2.sf(5.0, q)
            )

    def test_pi0_from_range(self) -> None:
        assert pi0_from_range(0.15, 0.85) == pytest.approx(0.15)
        assert pi0_from_range(0.5, 0.5) == pytest.approx(0.5)
        lam = 0.9 * 0.8 / (0.2 * 0.1)
        assert pi0_from_range(0.2, 0.9) == pytest.approx(1 / (1 + np.sqrt(lam)))
        with pytest.raises(ValueError):
            pi0_from_range(0.6, 0.4)

    def test_argument_checks(self) -> None:
        with pytest.raises(ValueError, match="1 to 40"):
            hansen_pvalue(3.0, 41, 0.15, "sup")
        with pytest.raises(ValueError, match="pi0"):
            hansen_pvalue(3.0, 1, 0.6, "sup")
        with pytest.raises(ValueError, match="kind"):
            hansen_pvalue(3.0, 1, 0.15, "max")  # type: ignore[arg-type]
        assert np.isnan(hansen_pvalue(float("nan"), 1, 0.15, "sup"))


class TestAndrewsPlobergerDefinitions:
    def test_statistics_are_scaled_by_q(self, rng: np.random.Generator) -> None:
        """F(tau) is q times the Chow F statistic at tau."""
        n = 120
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        y = X @ np.array([1.0, 0.5, -0.5]) + rng.standard_normal(n)
        results = rg.AndrewsPlobergerTest(y, exog_break=X).fit(trimming=0.15)
        for position in (0, 20, len(results.f_sequence) - 1):
            tau = int(results.candidate_indices[position])
            chow = rg.ChowTest(y, exog_break=X).fit(break_points=tau)
            assert results.f_sequence[position] == pytest.approx(
                3 * chow.f_stats[tau], rel=1e-10
            )
        w = results.f_sequence
        assert results.sup_f == pytest.approx(w.max())
        assert results.ave_f == pytest.approx(w.mean())
        assert results.exp_f == pytest.approx(np.log(np.mean(np.exp(w / 2))))

    def test_sup_f_equals_bai_perron_with_one_break(
        self, rng: np.random.Generator
    ) -> None:
        n = 150
        y = np.concatenate([rng.normal(0, 1, 75), rng.normal(0.8, 1, 75)])
        ap = rg.AndrewsPlobergerTest(y).fit(trimming=0.15)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bp = rg.BaiPerronTest(y).fit(max_breaks=1, trimming=0.15, hetvar=False)
        assert ap.sup_f == pytest.approx(bp.supf_stats[1], rel=1e-8)
        assert len(ap.f_sequence) == n - 2 * int(np.floor(0.15 * n)) + 1

    def test_trimming_uses_floor(self, rng: np.random.Generator) -> None:
        y = rng.standard_normal(205)
        results = rg.AndrewsPlobergerTest(y).fit(trimming=0.15)
        # floor(0.15 * 205) = 30, not ceil = 31
        assert results.candidate_indices[0] == 30
        assert results.candidate_indices[-1] == 175
        assert results.pi0 == pytest.approx(30 / 205)

    def test_break_date_is_reported_when_sup_f_rejects(
        self, rng: np.random.Generator
    ) -> None:
        y = rng.standard_normal(200)
        results = rg.AndrewsPlobergerTest(y).fit()
        assert results.n_breaks == int(results.rejected["SupF"])
        assert (len(results.break_indices) == 1) == results.rejected["SupF"]

        # a case in which ExpF or AveF rejects but SupF does not must not
        # report a break: force it through the significance level of SupF
        found = False
        for seed in range(40):
            local = np.random.default_rng(seed)
            shifted = np.concatenate(
                [local.normal(0, 1, 100), local.normal(0.4, 1, 100)]
            )
            r = rg.AndrewsPlobergerTest(shifted).fit()
            if (r.rejected["ExpF"] or r.rejected["AveF"]) and not r.rejected["SupF"]:
                assert r.n_breaks == 0
                assert list(r.break_indices) == []
                found = True
                break
        assert found, "no sample in which only ExpF or AveF rejects"

    def test_continuous_pvalues_and_any_level(self, rng: np.random.Generator) -> None:
        y = rng.standard_normal(200)
        results = rg.AndrewsPlobergerTest(y).fit(significance=0.025)
        for pvalue in (
            results.sup_f_pvalue,
            results.exp_f_pvalue,
            results.ave_f_pvalue,
        ):
            assert pvalue not in (0.005, 0.025, 0.075, 0.20)
        assert 0.025 in results.sup_f_critical
        assert results.sup_f_critical[0.025] == pytest.approx(
            hansen_critical_value(0.025, 1, results.pi0, "sup")
        )

    def test_too_many_coefficients(self, rng: np.random.Generator) -> None:
        n = 400
        X = rng.standard_normal((n, 41))
        y = rng.standard_normal(n)
        with pytest.raises(ValueError, match="up to 40"):
            rg.AndrewsPlobergerTest(y, exog_break=X).fit()


class TestConstantOnlyBreak:
    """``break_vars="const"``: the constant must not be in the design twice."""

    def _model(self, rng: np.random.Generator) -> tuple[Any, np.ndarray, np.ndarray]:
        n = 40
        x = rng.standard_normal(n)
        y = 1.0 + 0.5 * x + rng.standard_normal(n)
        y[20:] += 1.5
        X = np.column_stack([np.ones(n), x])
        return rg.OLS(y, X, has_constant=False), y, x

    def test_chow_degrees_of_freedom(self, rng: np.random.Generator) -> None:
        model, y, x = self._model(rng)
        results = model.chow_test(break_points=20, break_vars="const")
        # T - 2 * 1 - 1 = 37: two intercepts and one slope
        assert results.df_num[20] == 1
        assert results.df_denom[20] == 37

        # the same test from statsmodels: is the step dummy significant?
        step = (np.arange(40) >= 20).astype(float)
        fit = sm.OLS(y, np.column_stack([np.ones(40), x, step])).fit()
        assert results.f_stats[20] == pytest.approx(fit.tvalues[2] ** 2, rel=1e-10)
        assert results.p_values[20] == pytest.approx(fit.pvalues[2], rel=1e-8)

    def test_direct_use_with_a_constant_among_the_fixed_regressors(
        self, rng: np.random.Generator
    ) -> None:
        model, y, x = self._model(rng)
        X = np.column_stack([np.ones(40), x])
        direct = rg.ChowTest(y, exog=X).fit(break_points=20)
        via_model = model.chow_test(break_points=20, break_vars="const")
        assert direct.df_denom[20] == 37
        assert direct.f_stats[20] == pytest.approx(via_model.f_stats[20], rel=1e-12)

    def test_andrews_ploberger_counts(self, rng: np.random.Generator) -> None:
        model, _y, _x = self._model(rng)
        test = rg.AndrewsPlobergerTest.from_model(model, break_vars="const")
        assert test.n_break_params == 1
        assert test.n_nonbreak_params == 1
        results = test.fit()
        chow = model.chow_test(
            break_points=int(results.candidate_indices[5]), break_vars="const"
        )
        assert results.f_sequence[5] == pytest.approx(
            next(iter(chow.f_stats.values())), rel=1e-10
        )

    def test_model_without_a_constant(self, rng: np.random.Generator) -> None:
        n = 60
        x = rng.standard_normal((n, 1))
        y = rng.standard_normal(n)
        model = rg.OLS(y, x, has_constant=False)
        with pytest.raises(ValueError, match="no constant"):
            rg.ChowTest.from_model(model, break_vars="const")
        with pytest.raises(ValueError, match="no constant"):
            rg.AndrewsPlobergerTest.from_model(model, break_vars="const")


class TestChowAgainstStatsmodels:
    def test_full_break(self, rng: np.random.Generator) -> None:
        n = 90
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        y = X @ np.array([1.0, 0.5, -0.5]) + rng.standard_normal(n)
        y[50:] += 0.7
        results = rg.ChowTest(y, exog_break=X).fit(break_points=50)

        step = (np.arange(n) >= 50).astype(float)[:, None]
        unrestricted = sm.OLS(y, np.column_stack([X, X * step])).fit()
        restricted = sm.OLS(y, X).fit()
        f_stat, p_value, df_diff = unrestricted.compare_f_test(restricted)
        assert results.f_stats[50] == pytest.approx(f_stat, rel=1e-10)
        assert results.p_values[50] == pytest.approx(p_value, rel=1e-8)
        assert results.df_num[50] == df_diff == 3
        assert results.df_denom[50] == n - 6

    def test_too_few_observations_raises(self) -> None:
        """T = 5 with three coefficients: no test can be computed."""
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(5), rng.standard_normal((5, 2))])
        y = np.array([0.0, 0.1, 100.0, 100.2, 99.9])
        for break_point in (2, 3):
            with pytest.raises(ValueError, match="cannot be computed"):
                rg.ChowTest(y, exog_break=X).fit(break_points=break_point)

    def test_exact_fit_raises(self) -> None:
        y = np.concatenate([np.zeros(10), np.ones(10)])
        with pytest.raises(ValueError, match="exactly"):
            rg.ChowTest(y).fit(break_points=10)


def _recursive_ls(y: np.ndarray, X: np.ndarray) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return RecursiveLS(y, X).fit()


class TestCusumAgainstStatsmodels:
    def _data(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        n = 120
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        y = X @ np.array([1.0, 0.5, -0.5]) + rng.standard_normal(n)
        return y, X

    def test_recursive_residuals_and_cusum_path(self, rng: np.random.Generator) -> None:
        y, X = self._data(rng)
        results = rg.CUSUMTest(y, exog=X).fit()
        sm_results = _recursive_ls(y, X)
        k = X.shape[1]
        np.testing.assert_allclose(
            results.recursive_residuals, sm_results.resid_recursive[k:], atol=1e-8
        )
        # statsmodels centers the scale estimate; the cumulated sums agree
        np.testing.assert_allclose(
            results.statistic_path * results.sigma_hat,
            np.cumsum(sm_results.resid_recursive[k:]),
            atol=1e-7,
        )
        assert results.sigma_hat == pytest.approx(
            np.sqrt(np.sum(results.recursive_residuals**2) / (len(y) - k))
        )

    def test_cusum_boundary_line(self, rng: np.random.Generator) -> None:
        """Lines through (k, a sqrt(T-k)) and (T, 3 a sqrt(T-k))."""
        y, X = self._data(rng)
        results = rg.CUSUMTest(y, exog=X).fit(significance=0.05)
        n = len(y) - X.shape[1]
        a = 0.948
        assert results.upper_bound[-1] == pytest.approx(3 * a * np.sqrt(n))
        assert results.upper_bound[0] == pytest.approx(
            a * np.sqrt(n) + 2 * a / np.sqrt(n)
        )
        np.testing.assert_allclose(results.lower_bound, -results.upper_bound)

    @pytest.mark.parametrize("alpha", [0.20, 0.10, 0.05, 0.02, 0.01])
    def test_cusum_of_squares(self, rng: np.random.Generator, alpha: float) -> None:
        y, X = self._data(rng)
        results = rg.CUSUMSQTest(y, exog=X).fit(significance=alpha)
        sm_results = _recursive_ls(y, X)
        k = X.shape[1]
        np.testing.assert_allclose(
            results.statistic_path, sm_results.cusum_squares, atol=1e-8
        )
        lower, upper = sm_results._cusum_squares_significance_bounds(
            alpha, points=np.arange(k + 1, len(y) + 1)
        )
        np.testing.assert_allclose(results.upper_bound, upper, atol=1e-10)
        np.testing.assert_allclose(
            results.lower_bound, np.maximum(lower, 0.0), atol=1e-10
        )
        assert results.critical_value == pytest.approx(
            cusum_sq_critical_value(alpha, len(y) - k)
        )

    def test_tabulated_levels_only(self, rng: np.random.Generator) -> None:
        y = rng.standard_normal(100)
        for alpha in (0.03, 0.5, 0.001):
            with pytest.raises(ValueError, match="significance must be one of"):
                rg.CUSUMSQTest(y).fit(significance=alpha)


class TestCusumReporting:
    def test_first_crossing_is_in_sample_observations(self) -> None:
        rng = np.random.default_rng(3)
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
        y = X @ np.array([0.0, 0.5, 0.5, 0.5]) + rng.standard_normal(n)
        y[100:] += 3.0
        results = rg.CUSUMTest(y, exog=X).fit()
        assert results.reject
        k = 4
        outside = np.abs(results.statistic_path) > results.upper_bound
        assert results.first_crossing == int(np.argmax(outside)) + k
        assert results.crossing_indices[0] == results.first_crossing
        np.testing.assert_array_equal(results.observations, np.arange(k, n))
        # no break date is claimed
        assert results.n_breaks == 0
        assert list(results.break_indices) == []
        assert "not an estimate" in results.summary()

    def test_variance_change(self) -> None:
        rng = np.random.default_rng(4)
        y = np.concatenate([rng.normal(0, 1, 120), rng.normal(0, 4, 120)])
        results = rg.CUSUMSQTest(y).fit()
        assert results.reject
        assert results.first_crossing is not None
        assert results.n_breaks == 0
        assert "c_0" in results.summary()

    def test_start_that_does_not_identify_the_coefficients(self) -> None:
        """A dummy that is zero for the first 30 observations."""
        rng = np.random.default_rng(22)
        n = 80
        dummy = (np.arange(n) >= 30).astype(float)
        X = np.column_stack([np.ones(n), dummy])
        y = rng.standard_normal(n)
        # a dummy that never varies is collinear with the constant
        with pytest.raises(ValueError, match="collinear in the sample"):
            rg.CUSUMTest(y, exog=np.column_stack([np.ones(n), np.zeros(n) + 3])).fit()
        # the observation at which the dummy first varies is fitted exactly;
        # every other observation after the first has a residual
        results = rg.CUSUMTest(y, exog=X).fit()
        expected = [t for t in range(n) if t not in (0, 30)]
        np.testing.assert_array_equal(results.observations, expected)


class TestRecursiveResidualsNumerics:
    """The recursion is accurate for badly scaled regressors and starts as
    soon as the coefficients are identified."""

    @staticmethod
    def _direct(y: np.ndarray, X: np.ndarray, start: int) -> np.ndarray:
        out = []
        for t in range(start, len(y)):
            beta, *_ = np.linalg.lstsq(X[:t], y[:t], rcond=None)
            leverage = X[t] @ np.linalg.pinv(X[:t].T @ X[:t]) @ X[t]
            out.append((y[t] - X[t] @ beta) / np.sqrt(1 + leverage))
        return np.array(out)

    def test_trend_in_calendar_years(self) -> None:
        rng = np.random.default_rng(50)
        years = 1960 + np.arange(120) / 4
        X = np.column_stack([np.ones(120), years, years**2])
        y = 3.0 + 0.02 * (years - 1960) + rng.standard_normal(120)
        results = rg.CUSUMTest(y, exog=X).fit()
        # the same residuals from centered and scaled regressors
        z = (years - years.mean()) / years.std()
        scaled = rg.CUSUMTest(y, exog=np.column_stack([np.ones(120), z, z**2])).fit()
        # the raw design has a condition number of 5e14: what double
        # precision leaves is an accuracy of about 1e-5
        np.testing.assert_allclose(
            results.recursive_residuals, scaled.recursive_residuals, atol=1e-4
        )
        np.testing.assert_allclose(
            scaled.recursive_residuals,
            self._direct(y, np.column_stack([np.ones(120), z, z**2]), 3),
            atol=1e-8,
        )

    def test_regressor_in_levels(self) -> None:
        rng = np.random.default_rng(51)
        level = 1e6 + np.cumsum(rng.standard_normal(100))
        X = np.column_stack([np.ones(100), level])
        y = 0.5 * (level - 1e6) + rng.standard_normal(100)
        results = rg.CUSUMTest(y, exog=X).fit()
        centered = np.column_stack([np.ones(100), level - 1e6])
        np.testing.assert_allclose(
            results.recursive_residuals, self._direct(y, centered, 2), atol=1e-6
        )

    def test_nearly_collinear_start(self) -> None:
        """Two first observations that differ by 1e-9 are two design points:
        the second completes the rank, as in exact arithmetic."""
        rng = np.random.default_rng(52)
        x = rng.standard_normal(60)
        x[1] = x[0] + 1e-9
        X = np.column_stack([np.ones(60), x])
        y = 1.0 + 0.5 * x + rng.standard_normal(60)
        results = rg.CUSUMTest(y, exog=X).fit()
        assert len(results.recursive_residuals) == 58
        np.testing.assert_array_equal(results.observations[:3], [2, 3, 4])
        assert np.sum(results.recursive_residuals**2) == pytest.approx(
            sm.OLS(y, X).fit().ssr, rel=1e-8
        )
        # observations that are exactly equal are one design point
        x[1] = x[0]
        X = np.column_stack([np.ones(60), x])
        results = rg.CUSUMTest(y, exog=X).fit()
        np.testing.assert_array_equal(results.observations[:3], [1, 3, 4])
        assert results.recursive_residuals[0] == pytest.approx(
            (y[1] - y[0]) / np.sqrt(2)
        )

    def test_series_that_starts_at_zero(self) -> None:
        """An AR(1) of simulated data: y0 = y1 = 0 are one design point."""
        rng = np.random.default_rng(53)
        y = np.zeros(120)
        for t in range(2, 120):
            y[t] = 0.5 + 0.5 * y[t - 1] + rng.standard_normal()
        for results in (
            rg.AR(y, lags=1).cusum_test(),
            rg.AR(y, lags=1).cusum_sq_test(),
        ):
            # 119 observations in the estimation sample, two coefficients
            assert len(results.recursive_residuals) == 117
            assert np.all(np.isfinite(results.statistic_path))

    def test_dummy_in_the_middle_of_the_sample(self) -> None:
        """The test still covers the whole sample (it used to start after
        the dummy, or to fail)."""
        rng = np.random.default_rng(54)
        n = 200
        impulse = np.zeros(n)
        impulse[150] = 1.0
        X = np.column_stack([np.ones(n), impulse])
        y = rng.standard_normal(n)
        y[60:] *= 3.0  # the variance triples long before the dummy
        cusum_sq = rg.CUSUMSQTest(y, exog=X).fit()
        cusum = rg.CUSUMTest(y, exog=X).fit()
        expected = [t for t in range(n) if t not in (0, 150)]
        for results in (cusum, cusum_sq):
            np.testing.assert_array_equal(results.observations, expected)
            assert len(results.recursive_residuals) == n - 2
            assert np.sum(results.recursive_residuals**2) == pytest.approx(
                sm.OLS(y, X).fit().ssr, rel=1e-10
            )
        assert cusum_sq.reject
        # before the dummy the residuals are those of the constant-only model
        constant_only = rg.CUSUMTest(y[:150]).fit()
        np.testing.assert_allclose(
            cusum.recursive_residuals[:149],
            constant_only.recursive_residuals,
            atol=1e-10,
        )
        # bounds and scale use the number of residuals
        assert cusum.sigma_hat == pytest.approx(
            np.sqrt(np.sum(cusum.recursive_residuals**2) / (n - 2))
        )
        assert cusum.upper_bound[-1] == pytest.approx(3 * 0.948 * np.sqrt(n - 2))
        assert cusum_sq.critical_value == pytest.approx(
            cusum_sq_critical_value(0.05, n - 2)
        )
        # a crossing is reported at its observation, also after the gap
        crossed = np.where(
            (cusum_sq.statistic_path > cusum_sq.upper_bound)
            | (cusum_sq.statistic_path < cusum_sq.lower_bound)
        )[0]
        assert cusum_sq.first_crossing == expected[crossed[0]]
        assert cusum_sq.crossing_indices == [expected[i] for i in crossed]

    def test_dummy_at_the_last_observation(self) -> None:
        rng = np.random.default_rng(67)
        n = 60
        impulse = np.zeros(n)
        impulse[-1] = 1.0
        y = rng.standard_normal(n)
        results = rg.CUSUMTest(y, exog=np.column_stack([np.ones(n), impulse])).fit()
        np.testing.assert_array_equal(results.observations, np.arange(1, n - 1))
        np.testing.assert_allclose(
            results.recursive_residuals,
            rg.CUSUMTest(y[:-1]).fit().recursive_residuals,
            atol=1e-10,
        )

    def test_long_sample_is_fast(self) -> None:
        import time

        rng = np.random.default_rng(68)
        X = np.column_stack([np.ones(6000), rng.standard_normal((6000, 4))])
        y = rng.standard_normal(6000)
        start = time.perf_counter()
        results = rg.CUSUMTest(y, exog=X).fit()
        assert time.perf_counter() - start < 20.0
        assert len(results.recursive_residuals) == 5995

    def test_collinear_regressors_raise(self) -> None:
        rng = np.random.default_rng(55)
        x = rng.standard_normal(40)
        X = np.column_stack([np.ones(40), x, 2 * x + 1])
        with pytest.raises(ValueError, match="collinear in the sample"):
            rg.CUSUMTest(rng.standard_normal(40), exog=X).fit()


class TestCusumSquaresRange:
    def test_small_samples_are_refused(self) -> None:
        """Below 20 recursive residuals the bounds are not accurate."""
        rng = np.random.default_rng(56)
        with pytest.raises(ValueError, match="not accurate"):
            rg.CUSUMSQTest(rng.standard_normal(15)).fit()
        with pytest.raises(ValueError, match="not accurate"):
            cusum_sq_critical_value(0.05, 19)
        assert cusum_sq_critical_value(0.05, 20) > 0
        rg.CUSUMSQTest(rng.standard_normal(21)).fit()

    def test_rejection_rate_under_the_null(self) -> None:
        """A seeded simulation: a wrong coefficient or sample size would show."""
        rng = np.random.default_rng(57)
        rejections = 0
        replications = 400
        for _ in range(replications):
            X = np.column_stack([np.ones(100), rng.standard_normal(100)])
            y = rng.standard_normal(100)
            rejections += rg.CUSUMSQTest(y, exog=X).fit(significance=0.05).reject
        rate = rejections / replications
        # the old bounds gave about 0.24
        assert 0.02 <= rate <= 0.09

    def test_stable_data_have_no_crossing(self) -> None:
        y = np.random.default_rng(58).standard_normal(150)
        for results in (rg.CUSUMTest(y).fit(), rg.CUSUMSQTest(y).fit()):
            assert not results.reject
            assert results.first_crossing is None
            assert results.crossing_indices == []


class TestPlotsUseSampleObservations:
    def test_x_axis(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(59)
        n = 90
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
        y = rng.standard_normal(n)
        y[45:] += 4.0
        for results in (
            rg.CUSUMTest(y, exog=X).fit(),
            rg.CUSUMSQTest(y, exog=X).fit(),
        ):
            fig, ax = results.plot()
            x = ax.lines[0].get_xdata()
            np.testing.assert_array_equal(x, np.arange(4, n))
            if results.first_crossing is not None:
                vertical = [
                    line.get_xdata()[0]
                    for line in ax.lines
                    if len(set(np.atleast_1d(line.get_xdata()))) == 1
                    and line.get_linestyle() == ":"
                ]
                assert vertical == [results.first_crossing]
            plt.close(fig)


class TestHansenEdges:
    def test_exp_statistic_with_a_single_candidate_date(self) -> None:
        """ExpF is half the chi-squared statistic of the one date."""
        for q in (1, 3):
            chi = 7.0
            assert hansen_pvalue(chi, q, 0.5, "sup") == pytest.approx(
                stats.chi2.sf(chi, q)
            )
            assert hansen_pvalue(chi, q, 0.5, "ave") == pytest.approx(
                stats.chi2.sf(chi, q)
            )
            assert hansen_pvalue(chi / 2, q, 0.5, "exp") == pytest.approx(
                stats.chi2.sf(chi, q)
            )
        assert hansen_critical_value(0.01, 1, 0.5, "exp") == pytest.approx(
            stats.chi2.ppf(0.99, 1) / 2
        )

    def test_one_candidate_date_gives_the_same_pvalue_three_times(self) -> None:
        # eight observations and three breaking coefficients: each regime
        # needs four observations, so observation 4 is the only candidate
        rng = np.random.default_rng(60)
        X = np.column_stack([np.ones(8), rng.standard_normal((8, 2))])
        y = rng.standard_normal(8)
        results = rg.AndrewsPlobergerTest(y, exog_break=X).fit(trimming=0.3)
        assert len(results.f_sequence) == 1
        assert results.pi0 == pytest.approx(0.5)
        assert results.exp_f == pytest.approx(results.sup_f / 2)
        assert results.sup_f_pvalue == pytest.approx(results.exp_f_pvalue)
        assert results.sup_f_pvalue == pytest.approx(results.ave_f_pvalue)

    def test_very_small_trimming_warns_once(self) -> None:
        # below the tables: the value for pi_0 = 0.01
        assert hansen_pvalue(12.0, 1, 0.002, "sup") == pytest.approx(
            hansen_pvalue(12.0, 1, 0.01, "sup")
        )
        y = np.random.default_rng(61).standard_normal(2000)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rg.AndrewsPlobergerTest(y).fit(trimming=0.002)
        messages = [str(w.message) for w in caught if "below 0.01" in str(w.message)]
        assert len(messages) == 1
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rg.AndrewsPlobergerTest(y[:300]).fit(trimming=0.05)

    def test_exp_statistic_between_the_last_row_and_a_single_date(self) -> None:
        """For pi_0 between 0.49 and 0.5 the p-value runs linearly from the
        table to the chi-squared tail of twice the statistic."""
        q, statistic = 2, 3.0
        at_049 = hansen_pvalue(statistic, q, 0.49, "exp")
        at_050 = hansen_pvalue(statistic, q, 0.5, "exp")
        assert at_050 == pytest.approx(stats.chi2.sf(2 * statistic, q))
        middle = hansen_pvalue(statistic, q, 0.495, "exp")
        assert middle == pytest.approx(0.5 * (at_049 + at_050), rel=1e-9)
        assert min(at_049, at_050) <= middle <= max(at_049, at_050)

    def test_critical_value_below_the_floor_is_nan(self) -> None:
        assert np.isnan(hansen_critical_value(1e-6, 2, 0.15, "exp"))
        assert np.isfinite(hansen_critical_value(1e-6, 2, 0.15, "sup"))


class TestPartialBreakAgainstStatsmodels:
    def test_two_breaking_and_two_fixed_coefficients(self) -> None:
        rng = np.random.default_rng(62)
        n = 140
        breaking = np.column_stack([np.ones(n), rng.standard_normal(n)])
        fixed = rng.standard_normal((n, 2))
        y = breaking @ [1.0, 0.5] + fixed @ [0.3, -0.2] + rng.standard_normal(n)
        results = rg.AndrewsPlobergerTest(y, exog=fixed, exog_break=breaking).fit()
        assert (results.q, results.p) == (2, 2)
        restricted = sm.OLS(y, np.column_stack([breaking, fixed])).fit()
        for position in (0, 30, len(results.f_sequence) - 1):
            tau = int(results.candidate_indices[position])
            step = (np.arange(n) >= tau).astype(float)[:, None]
            unrestricted = sm.OLS(
                y, np.column_stack([breaking, breaking * step, fixed])
            ).fit()
            f_stat, _p, df_diff = unrestricted.compare_f_test(restricted)
            assert df_diff == 2
            assert unrestricted.df_resid == n - 2 * 2 - 2
            assert results.f_sequence[position] == pytest.approx(2 * f_stat, rel=1e-9)

    def test_ar_model_with_a_break_in_the_constant_only(self) -> None:
        rng = np.random.default_rng(63)
        y = np.zeros(150)
        for t in range(1, 150):
            y[t] = (0.0 if t < 80 else 1.5) + 0.4 * y[t - 1] + rng.standard_normal()
        model = rg.AR(y, lags=1)
        test = rg.AndrewsPlobergerTest.from_model(model, break_vars="const")
        assert (test.n_break_params, test.n_nonbreak_params) == (1, 1)
        chow = rg.ChowTest.from_model(model, break_vars="const")
        assert (chow.n_break_params, chow.n_nonbreak_params) == (1, 1)


class TestLargeMeanAndRedundantBreak:
    def test_series_with_a_large_mean(self) -> None:
        """A small residual variance relative to the mean is not an exact fit."""
        rng = np.random.default_rng(64)
        noise = 0.5 * rng.standard_normal(80)
        noise[40:] += 0.6
        centered = rg.ChowTest(noise).fit(break_points=40)
        shifted = rg.ChowTest(1e6 + noise).fit(break_points=40)
        assert shifted.f_stats[40] == pytest.approx(centered.f_stats[40], rel=1e-5)
        ap_centered = rg.AndrewsPlobergerTest(noise).fit()
        ap_shifted = rg.AndrewsPlobergerTest(1e6 + noise).fit()
        assert ap_shifted.sup_f == pytest.approx(ap_centered.sup_f, rel=1e-5)

    def test_break_that_is_already_in_the_model(self) -> None:
        rng = np.random.default_rng(65)
        n = 80
        step = (np.arange(n) >= 40).astype(float)
        X = np.column_stack([rng.standard_normal(n), step])
        y = rng.standard_normal(n) + step
        with pytest.raises(ValueError, match="already contain this break"):
            rg.ChowTest(y, exog=X).fit(break_points=40)
        # another date is a real test
        results = rg.ChowTest(y, exog=X).fit(break_points=30)
        assert results.df_num[30] == 1

    def test_constant_that_is_not_one(self) -> None:
        rng = np.random.default_rng(66)
        n = 60
        X = np.column_stack([np.full(n, 2.0), rng.standard_normal(n)])
        model = rg.OLS(rng.standard_normal(n), X, has_constant=False)
        test = rg.ChowTest.from_model(model, break_vars="const")
        assert (test.n_break_params, test.n_nonbreak_params) == (1, 1)


class TestChowDegenerateDesigns:
    """Designs in which there is nothing to test raise a ValueError."""

    def test_breaking_regressor_that_is_zero_in_a_sub_sample(self) -> None:
        rng = np.random.default_rng(69)
        n = 100
        x = rng.standard_normal(n)
        step = (np.arange(n) >= 60).astype(float)
        y = rng.standard_normal(n)
        fixed = np.column_stack([np.ones(n), x])
        with pytest.raises(ValueError, match="not identified"):
            rg.ChowTest(y, exog=fixed, exog_break=step).fit(break_points=50)
        with pytest.raises(ValueError, match="not identified"):
            rg.ChowTest(y, exog=x, exog_break=np.column_stack([np.ones(n), step])).fit(
                break_points=60
            )
        # a step among the regressors that do not break, tested elsewhere,
        # is a real test of the two breaking coefficients
        results = rg.ChowTest(
            y, exog=step, exog_break=np.column_stack([np.ones(n), x])
        ).fit(break_points=80)
        assert results.df_num[80] == 2
        assert results.f_stats[80] > 0

    def test_predictive_variant_with_an_impulse_in_the_predicted_part(self) -> None:
        rng = np.random.default_rng(70)
        n = 100
        impulse = np.zeros(n)
        impulse[99] = 1.0
        X = np.column_stack([rng.standard_normal(n), impulse])
        y = rng.standard_normal(n)
        with pytest.raises(ValueError, match="nothing to test"):
            rg.ChowTest(y, exog=X).fit(break_points=99)
        # without the dummy the predictive test of the last observation runs
        results = rg.ChowTest(y, exog=X[:, :1]).fit(break_points=99)
        assert results.is_predictive[99]

    def test_break_points_are_integers(self) -> None:
        y = np.random.default_rng(71).standard_normal(80)
        test = rg.ChowTest(y)
        for bad in (40.7, [40.7], [True], "40"):
            with pytest.raises(TypeError, match="must be integers"):
                test.fit(break_points=bad)  # type: ignore[arg-type]
        assert list(test.fit(break_points=[np.int64(40), 50]).f_stats) == [40, 50]

    def test_andrews_ploberger_exact_fit(self) -> None:
        y = np.concatenate([np.zeros(30), np.ones(30)])
        with pytest.raises(ValueError, match="exactly"):
            rg.AndrewsPlobergerTest(y).fit()


class TestRecursiveResidualsDoNotDependOnScale:
    """The rank decision used a tolerance relative to the largest singular
    value of the whole sample: a regressor that is small early in the sample
    was treated as absent there, and the tests rejected a stable model."""

    @staticmethod
    def _check(y: np.ndarray, X: np.ndarray, no_residual: list[int]) -> None:
        w, _f, positions = _compute_recursive_residuals(y, X)
        expected = [t for t in range(len(y)) if t not in no_residual]
        np.testing.assert_array_equal(positions, expected)
        scaled = X / np.max(np.abs(X), axis=0)
        beta, *_ = np.linalg.lstsq(scaled, y, rcond=None)
        ssr = float(np.sum((y - scaled @ beta) ** 2))
        assert np.sum(w**2) == pytest.approx(ssr, rel=2e-6)

    @pytest.mark.parametrize("rate", [0.05, 0.07])
    def test_regressor_that_grows_by_orders_of_magnitude(self, rate: float) -> None:
        rng = np.random.default_rng(72)
        n = 300
        x = np.exp(rate * np.arange(n))  # up to 3e6 and 1e9
        X = np.column_stack([np.ones(n), x])
        y = 1.0 + x + rng.standard_normal(n)
        self._check(y, X, [0, 1])
        # the coefficients are stable: with rate 0.07 the sum of squared
        # residuals was 3.7 million where the regression has 291, and both
        # tests rejected
        assert not rg.CUSUMTest(y, exog=X).fit().reject
        assert not rg.CUSUMSQTest(y, exog=X).fit().reject

    def test_one_very_large_value(self) -> None:
        rng = np.random.default_rng(73)
        n = 300
        x = rng.standard_normal(n)
        x[200] = 1e8
        X = np.column_stack([np.ones(n), x])
        y = 1.0 + x + rng.standard_normal(n)
        self._check(y, X, [0, 1])
        assert not rg.CUSUMSQTest(y, exog=X).fit().reject

    def test_long_polynomial_trends(self) -> None:
        """The first k observations complete the rank (the old rule dropped
        observation 8 of a quadratic trend with 2000 observations)."""
        rng = np.random.default_rng(74)
        for n, degree in ((2000, 2), (500, 3)):
            t = np.arange(n, dtype=float)
            X = np.column_stack([t**j for j in range(degree + 1)])
            self._check(rng.standard_normal(n), X, list(range(degree + 1)))

    def test_rescaling_a_regressor_changes_nothing(self) -> None:
        rng = np.random.default_rng(75)
        n = 120
        step = (np.arange(n) >= 40).astype(float)
        X = np.column_stack([np.ones(n), rng.standard_normal(n), step])
        y = rng.standard_normal(n)
        w, _f, positions = _compute_recursive_residuals(y, X)
        for factors in ([1.0, 1e8, 1.0], [1e-6, 1.0, 1e9]):
            w2, _f2, positions2 = _compute_recursive_residuals(y, X * factors)
            np.testing.assert_array_equal(positions2, positions)
            np.testing.assert_allclose(w2, w, rtol=1e-8, atol=1e-10)

    def test_polynomials_in_calendar_years(self) -> None:
        """A cubic in quarterly calendar years is computed, to about four
        digits; a quartic is beyond double precision and raises."""
        rng = np.random.default_rng(76)
        years = 1990 + np.arange(60) / 4
        elapsed = years - years[0]
        y = rng.standard_normal(60)
        calendar = rg.CUSUMTest(
            y, exog=np.column_stack([years**j for j in range(4)])
        ).fit()
        from_start = rg.CUSUMTest(
            y, exog=np.column_stack([elapsed**j for j in range(4)])
        ).fit()
        np.testing.assert_array_equal(calendar.observations, np.arange(4, 60))
        np.testing.assert_allclose(
            calendar.recursive_residuals, from_start.recursive_residuals, atol=2e-3
        )
        assert calendar.reject == from_start.reject
        years = 1990 + np.arange(80) / 4
        with pytest.raises(ValueError, match="could not be computed accurately"):
            rg.CUSUMTest(
                rng.standard_normal(80),
                exog=np.column_stack([years**j for j in range(5)]),
            ).fit()


class TestCrossingBeforeALateDummy:
    """Positions when the observation without a residual comes late."""

    def test_cusum_crossing_and_plot(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(77)
        n = 200
        impulse = np.zeros(n)
        impulse[150] = 1.0
        X = np.column_stack([np.ones(n), impulse])
        y = rng.standard_normal(n)
        y[40:] += 3.0  # the mean shifts long before the dummy
        results = rg.CUSUMTest(y, exog=X).fit()
        assert results.reject
        outside = np.where(
            (results.statistic_path > results.upper_bound)
            | (results.statistic_path < results.lower_bound)
        )[0]
        # residual number i belongs to observation i + 1 before the dummy
        assert results.first_crossing == outside[0] + 1
        assert results.first_crossing < 150
        assert results.crossing_indices[0] == results.first_crossing
        fig, ax = results.plot()
        np.testing.assert_array_equal(
            ax.lines[0].get_xdata(), [t for t in range(n) if t not in (0, 150)]
        )
        plt.close(fig)


class TestChowWhenEveryCoefficientBreaks:
    def test_dummy_among_the_regressors(self) -> None:
        """It returned F(3, 94) = 0.71 where the test of the two coefficients
        that can change is F(2, 95) = 1.08."""
        rng = np.random.default_rng(78)
        n = 100
        x = rng.standard_normal(n)
        impulse = np.zeros(n)
        impulse[10] = 1.0
        y = 1.0 + 0.5 * x + rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x, impulse])
        with pytest.raises(ValueError, match="Pass such a regressor as `exog`"):
            rg.ChowTest(y, exog_break=X).fit(break_points=50)
        with pytest.raises(ValueError, match="not identified"):
            rg.ChowTest.from_model(rg.OLS(y, X, has_constant=False)).fit(
                break_points=50
            )
        # the way out that the message names
        results = rg.ChowTest(y, exog=impulse, exog_break=X[:, :2]).fit(break_points=50)
        assert (results.df_num[50], results.df_denom[50]) == (2, 95)
        reference = sm.OLS(
            y,
            np.column_stack(
                [
                    X[:, :2] * (np.arange(n) < 50)[:, None],
                    X[:, :2] * (np.arange(n) >= 50)[:, None],
                    impulse,
                ]
            ),
        ).fit()
        restricted = sm.OLS(y, X).fit()
        f_stat = ((restricted.ssr - reference.ssr) / 2) / (reference.ssr / 95)
        assert results.f_stats[50] == pytest.approx(f_stat, rel=1e-9)

    def test_other_unidentified_designs(self) -> None:
        rng = np.random.default_rng(79)
        n = 100
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        step = (np.arange(n) >= 50).astype(float)
        late = np.where(np.arange(n) >= 60, x, 0.0)
        for X in (
            np.column_stack([np.ones(n), x, step]),  # the break itself
            np.column_stack([np.ones(n), x, x]),  # a duplicated column
            np.column_stack([np.ones(n), late]),  # zero before the break
        ):
            with pytest.raises(ValueError, match="not identified|collinear"):
                rg.ChowTest(y, exog_break=X).fit(break_points=50)
        # predictive variant with collinear regressors
        with pytest.raises(ValueError, match="collinear in the sample"):
            rg.ChowTest(y, exog_break=np.column_stack([np.ones(n), x, x])).fit(
                break_points=98
            )
        # and an ordinary design is not affected
        results = rg.ChowTest(y, exog_break=np.column_stack([np.ones(n), x])).fit(
            break_points=[50, 99]
        )
        assert results.df_num[50] == 2
        assert results.is_predictive[99]

    def test_break_points_given_as_an_array_scalar(self) -> None:
        y = np.random.default_rng(80).standard_normal(60)
        with pytest.raises(TypeError, match="must be integers"):
            rg.ChowTest(y).fit(break_points=np.array(30.5))


def _exact_rank_rises(X: np.ndarray) -> list[int]:
    """Observations that raise the rank of the design, in rational arithmetic."""
    from fractions import Fraction

    k = X.shape[1]
    rows: list[tuple[int, list[Fraction]]] = []
    rises = []
    for t in range(X.shape[0]):
        v = [Fraction(float(a)) for a in X[t]]
        for pivot, row in rows:
            if v[pivot] != 0:
                c = v[pivot] / row[pivot]
                v = [a - c * b for a, b in zip(v, row)]
        nonzero = [j for j in range(k) if v[j] != 0]
        if nonzero:
            rows.append((nonzero[0], v))
            rises.append(t)
    return rises


class TestDependenciesAgainstExactArithmetic:
    """Which observations have no recursive residual, against rational
    arithmetic, for designs with exact linear dependencies among the columns
    in part of the sample."""

    @staticmethod
    def _check(y: np.ndarray, X: np.ndarray) -> None:
        w, _f, positions = _compute_recursive_residuals(y, X)
        missing = sorted(set(range(len(y))) - set(positions.tolist()))
        assert missing == _exact_rank_rises(X)
        scaled = X / np.max(np.abs(X), axis=0)
        beta, *_ = np.linalg.lstsq(scaled, y, rcond=None)
        ssr = float(np.sum((y - scaled @ beta) ** 2))
        assert np.sum(w**2) == pytest.approx(ssr, rel=1e-8)

    def test_a_rate_with_zeros_and_its_first_regime(self) -> None:
        """Elimination left 5e-17 where a coefficient is exactly zero; at an
        observation where the rate is zero that residue looked like a broken
        dependency, and an ordinary design (condition number 3.5) raised."""
        x = np.array(
            [1.0, 3, 0, 2, 0, -1, 2, 0, 3, -2, 7, 0, -5]
            + [2, 0, 2, -7, 0, 2, 5, 0, -2, 2, 0, 0, 6]
        )
        n = len(x)
        X = np.column_stack([np.ones(n), x, np.where(np.arange(n) < 10, x, 0.0)])
        y = np.sin(1.7 * np.arange(n)) + 0.3 * x
        self._check(y, X)
        results = rg.CUSUMTest(y, exog=X).fit()
        expected = [t for t in range(n) if t not in (0, 1, 10)]
        np.testing.assert_array_equal(results.observations, expected)

    def test_random_designs(self) -> None:
        rng = np.random.default_rng(2026)
        steps = [0, 0, 0, 0.25, -0.25, 0.5, -0.5, 0.75, 1.0, -1.25, 1.75]
        for _ in range(60):
            n = int(rng.integers(20, 45))
            x = rng.choice(steps, size=n)
            tau = int(rng.integers(5, n - 5))
            X = np.column_stack([np.ones(n), x, np.where(np.arange(n) < tau, x, 0.0)])
            if len(_exact_rank_rises(X)) == 3:
                self._check(rng.standard_normal(n), X)
        for _ in range(40):
            k = int(rng.integers(4, 9))
            n = int(rng.integers(3 * k, 5 * k))
            X = rng.integers(-3, 4, size=(n, k)).astype(float)
            for j in range(1, k):
                if rng.random() < 0.6:
                    until = int(rng.integers(1, n - k))
                    others = rng.choice(j, size=min(j, 2), replace=False)
                    # coefficients that are exact in binary arithmetic
                    coefs = rng.choice(
                        [1.0, 2.0, -1.0, 0.5, 3.0, -1.5], size=len(others)
                    )
                    X[:until, j] = X[:until, others] @ coefs
            X[:, 0] = 1.0
            if len(_exact_rank_rises(X)) == k:
                self._check(rng.standard_normal(n), X)

    def test_two_dependencies_broken_by_one_observation(self) -> None:
        rng = np.random.default_rng(81)
        n = 40
        a = rng.integers(1, 5, size=n).astype(float)
        b, c = 2 * a, a.copy()
        b[12:] = rng.integers(-3, 4, size=n - 12)
        c[12:] = rng.integers(-3, 4, size=n - 12)
        self._check(rng.standard_normal(n), np.column_stack([a, b, c]))


class TestRecursiveResidualsEdges:
    def test_exact_fit_raises(self) -> None:
        """The residuals would be rounding error (the tests used to run on
        it: CUSUM-of-squares rejected)."""
        rng = np.random.default_rng(82)
        X = np.column_stack([np.ones(60), rng.standard_normal(60)])
        for y in (np.zeros(60), np.full(60, 3.0), X @ [1.0, 2.0]):
            with pytest.raises(ValueError, match="fits the data exactly"):
                rg.CUSUMTest(y, exog=X).fit()
            with pytest.raises(ValueError, match="fits the data exactly"):
                rg.CUSUMSQTest(y, exog=X).fit()

    def test_missing_values_raise(self) -> None:
        y = np.random.default_rng(83).standard_normal(40)
        for bad in (np.nan, np.inf):
            y_bad = y.copy()
            y_bad[7] = bad
            with pytest.raises(ValueError, match="NaN or infinite"):
                rg.CUSUMTest(y_bad).fit()

    def test_large_level_of_the_dependent_variable(self) -> None:
        """A level of 1e8 with residuals of 1e-3: the sum of squares cannot
        be reproduced to 1e-6 in double precision, and need not be."""
        for seed in range(10):
            rng = np.random.default_rng(seed)
            noise = 1e-3 * rng.standard_normal(60)
            w, _f, positions = _compute_recursive_residuals(
                1e8 + noise, np.ones((60, 1))
            )
            reference, _f, _p = _compute_recursive_residuals(noise, np.ones((60, 1)))
            np.testing.assert_array_equal(positions, np.arange(1, 60))
            np.testing.assert_allclose(w, reference, atol=1e-6)


class TestTheCheckOfTheRecursiveResiduals:
    """The sum of squared recursive residuals must equal the residual sum
    of squares; the computations are tried in turn until one does."""

    @staticmethod
    def _data(seed: int = 84, n: int = 80) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(n), np.column_stack(
            [np.ones(n), rng.standard_normal(n)]
        )

    def test_ordinary_design_uses_the_first_computation(self, monkeypatch: Any) -> None:
        import regimes.tests.cusum as module

        def not_needed(*args: Any) -> Any:
            raise AssertionError("second computation used")

        monkeypatch.setattr(module, "_residuals_in_sample_basis", not_needed)
        monkeypatch.setattr(module, "_orthonormal", not_needed)
        y, X = self._data()
        w, _f, positions = _compute_recursive_residuals(y, X)
        np.testing.assert_array_equal(positions, np.arange(2, 80))
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        assert np.sum(w**2) == pytest.approx(np.sum((y - X @ beta) ** 2), rel=1e-12)

    def test_the_computations_agree_on_ordinary_designs(self) -> None:
        import regimes.tests.cusum as module

        for seed in range(20):
            y, X = self._data(seed)
            w, f, positions = _compute_recursive_residuals(y, X)
            w_050, f_050, positions_050 = module._residuals_in_sample_basis(y, X)
            np.testing.assert_array_equal(positions, positions_050)
            np.testing.assert_allclose(w, w_050, atol=1e-11)
            np.testing.assert_allclose(f, f_050, rtol=1e-10)

    def test_falls_back_when_the_first_computation_fails(
        self, monkeypatch: Any
    ) -> None:
        import regimes.tests.cusum as module

        y, X = self._data()
        expected = module._residuals_in_sample_basis(y, X)

        def fails(*args: Any) -> Any:
            raise ValueError("could not be told from rounding error")

        monkeypatch.setattr(module, "_begin_from_dependencies", fails)
        w, _f, positions = _compute_recursive_residuals(y, X)
        np.testing.assert_array_equal(w, expected[0])
        np.testing.assert_array_equal(positions, expected[2])

    def test_falls_back_when_the_first_computation_is_wrong(
        self, monkeypatch: Any
    ) -> None:
        import regimes.tests.cusum as module

        y, X = self._data()
        expected = module._residuals_in_sample_basis(y, X)
        original = module._finish_from_dependencies

        def wrong(*args: Any) -> Any:
            w, f, positions = original(*args)
            return w * (1 + 5e-5), f, positions  # sum of squares off by 1e-4

        monkeypatch.setattr(module, "_finish_from_dependencies", wrong)
        w, _f, _positions = _compute_recursive_residuals(y, X)
        np.testing.assert_array_equal(w, expected[0])

    def test_error_when_no_computation_reproduces_the_sum_of_squares(
        self, monkeypatch: Any
    ) -> None:
        import regimes.tests.cusum as module

        y, X = self._data()
        finish = module._finish_from_dependencies
        released = module._residuals_in_sample_basis

        def off(function: Any) -> Any:
            def wrapped(*args: Any) -> Any:
                w, f, positions = function(*args)
                return w * (1 + 5e-5), f, positions

            return wrapped

        monkeypatch.setattr(module, "_finish_from_dependencies", off(finish))
        monkeypatch.setattr(module, "_residuals_in_sample_basis", off(released))
        with pytest.raises(ValueError, match="could not be computed accurately"):
            _compute_recursive_residuals(y, X)

    def test_the_best_computation_is_used(self, monkeypatch: Any) -> None:
        """Neither within 1e-9, both within 1e-6: the closer one."""
        import regimes.tests.cusum as module

        y, X = self._data()
        finish = module._finish_from_dependencies
        released = module._residuals_in_sample_basis

        def off(function: Any, by: float) -> Any:
            def wrapped(*args: Any) -> Any:
                w, f, positions = function(*args)
                return w * (1 + by), f, positions

            return wrapped

        monkeypatch.setattr(module, "_finish_from_dependencies", off(finish, 2e-7))
        monkeypatch.setattr(module, "_residuals_in_sample_basis", off(released, 2e-8))
        w, _f, _positions = _compute_recursive_residuals(y, X)
        np.testing.assert_array_equal(w, released(y, X)[0] * (1 + 2e-8))

    def test_collinear_in_the_sample(self) -> None:
        y, X = self._data()
        with pytest.raises(ValueError, match="collinear in the sample"):
            _compute_recursive_residuals(y, np.column_stack([X, 2 * X[:, 1]]))


class TestDesignsFromTheSecondReview:
    """Designs on which a version of the computation from dependencies
    raised or left out the wrong observations."""

    @staticmethod
    def _exact(y: np.ndarray, X: np.ndarray) -> None:
        w, _f, positions = _compute_recursive_residuals(y, X)
        missing = sorted(set(range(len(y))) - set(positions.tolist()))
        assert missing == _exact_rank_rises(X)
        scaled = X / np.linalg.norm(X, axis=0)
        beta = np.linalg.lstsq(scaled, y, rcond=None)[0]
        ssr = float(np.sum((y - scaled @ beta) ** 2))
        assert np.sum(w**2) == pytest.approx(ssr, rel=1e-6)

    def test_first_two_observations_almost_equal(self) -> None:
        """The elimination multiplier is a ratio with a small denominator;
        its rounding error looked like a broken dependency."""
        x = np.r_[100.0, 100.001, 101.0 + np.arange(28)]
        t = np.arange(30)
        X = np.column_stack([np.ones(30), x, np.where(t < 20, x, 0.0)])
        y = np.sin(1.7 * t)
        self._exact(y, X)
        results = rg.CUSUMTest(y, exog=X).fit()
        expected = [s for s in range(30) if s not in (0, 1, 20)]
        np.testing.assert_array_equal(results.observations, expected)

    def test_small_first_steps_and_kinks(self) -> None:
        rng = np.random.default_rng(424242)
        for _ in range(60):
            n = int(rng.integers(30, 80))
            step = 10.0 ** -rng.integers(1, 6)
            walk = step + np.cumsum(rng.standard_normal(n - 2))
            x = 100.0 + np.concatenate([[0.0, step], walk])
            tau = int(rng.integers(8, n - 8))
            X = np.column_stack([np.ones(n), x, np.where(np.arange(n) < tau, x, 0.0)])
            self._exact(rng.standard_normal(n), X)
        for _ in range(40):
            n = int(rng.integers(30, 80))
            x = 5.0 + np.cumsum(rng.standard_normal(n))
            cap = float(np.quantile(x, rng.uniform(0.3, 0.8)))
            X = np.column_stack([np.ones(n), x, np.minimum(x, cap)])
            self._exact(rng.standard_normal(n), X)

    def test_one_very_large_value(self) -> None:
        """Also when the dependent variable is unrelated to the regressor:
        the computation of version 0.5.0 reproduces the sum of squares
        there, with the last observation left out instead of the second."""
        for big in (1e8, 1e11):
            rng = np.random.default_rng(86)
            x = rng.standard_normal(100)
            x[-1] = big
            X = np.column_stack([np.ones(100), x])
            self._exact(rng.standard_normal(100), X)
            self._exact(1.0 + 2.0 * x / big + rng.standard_normal(100), X)

    def test_the_released_method_does_not_override_the_observations(self) -> None:
        """Two regressors that differ by 1e-9 in the first observations: the
        method of 0.5.0 leaves out observations 0, 3 and 4 and reproduces
        the sum of squares more closely than the right computation."""
        for seed in range(1, 6):
            rng = np.random.default_rng(seed)
            x, z, y = (rng.standard_normal(40) for _ in range(3))
            x[:3] = 1.0 + 1e-9 * np.arange(3)
            z[:3] = 1.0
            X = np.column_stack([np.ones(40), x, z])
            try:
                _w, _f, positions = _compute_recursive_residuals(y, X)
            except ValueError:
                continue  # at the limit of the arithmetic: an error is right
            missing = sorted(set(range(40)) - set(positions.tolist()))
            assert missing == [0, 1, 3]

    def test_a_dependency_that_is_not_exact_in_binary(self) -> None:
        """c = a/3 + 2b/3 at first: the stored columns are dependent only
        to rounding error, and are treated as dependent."""
        rng = np.random.default_rng(87)
        for _ in range(30):
            n = int(rng.integers(24, 60))
            a = rng.integers(-5, 6, size=n).astype(float)
            b = rng.integers(-5, 6, size=n).astype(float)
            c = a / 3 + 2 * b / 3
            until = int(rng.integers(6, n - 8))
            c[until:] = rng.integers(-5, 6, size=n - until) + 0.5
            X = np.column_stack([np.ones(n), a, b, c])
            y = rng.standard_normal(n)
            w, _f, positions = _compute_recursive_residuals(y, X)
            missing = sorted(set(range(n)) - set(positions.tolist()))
            assert missing[-1] >= until  # not at a rounding residue before
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            assert np.sum(w**2) == pytest.approx(np.sum((y - X @ beta) ** 2), rel=1e-6)


class TestChowDoesNotDependOnUnits:
    def test_rescaled_regressors(self) -> None:
        """Least squares dropped a column fourteen orders of magnitude
        smaller than another, and the statistic changed with the units."""
        rng = np.random.default_rng(1)
        n = 100
        a, b, y = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
        well = rg.ChowTest(y, exog_break=np.column_stack([np.ones(n), a, b])).fit(50)
        badly = rg.ChowTest(
            y, exog_break=np.column_stack([np.ones(n), 0.01 * a, 1e12 * b])
        ).fit(50)
        assert badly.f_stats[50] == pytest.approx(well.f_stats[50], rel=1e-8)
        sup_well = rg.AndrewsPlobergerTest(
            y, exog_break=np.column_stack([np.ones(n), a, b])
        ).fit()
        sup_badly = rg.AndrewsPlobergerTest(
            y, exog_break=np.column_stack([np.ones(n), 0.01 * a, 1e12 * b])
        ).fit()
        assert sup_badly.sup_f == pytest.approx(sup_well.sup_f, rel=1e-8)

    def test_break_points_from_any_iterable(self) -> None:
        y = np.random.default_rng(85).standard_normal(120)
        test = rg.ChowTest(y)
        assert sorted(test.fit({60, 70}).f_stats) == [60, 70]
        assert sorted(test.fit(b for b in (40, 80)).f_stats) == [40, 80]
