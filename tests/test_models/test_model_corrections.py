"""Regression tests for the model defects found in the November 2026 audit,
and permanent comparisons of the estimators with statsmodels."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.regression.recursive_ls import RecursiveLS
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ardl import ARDL, ardl_select_order

import regimes as rg
from regimes.models.base import check_break_design, normalize_breaks


def _adl_data(seed: int = 0, n: int = 220) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = (
            0.5
            + 0.4 * y[t - 1]
            - 0.2 * y[t - 2]
            + 0.8 * x[t]
            + 0.5 * x[t - 1]
            + rng.standard_normal()
        )
    return y, x


class TestSelectLags:
    """All candidates on one sample (audit 4.2: the choice changed with units)."""

    @pytest.mark.parametrize("criterion", ["aic", "bic"])
    def test_matches_statsmodels(self, criterion: str) -> None:
        y, x = _adl_data(seed=1)
        ours = rg.ADL(y, x, lags=1, exog_lags=0).select_lags(
            max_ar_lags=4, max_exog_lags=3, criterion=criterion
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selection = ardl_select_order(
                y, maxlag=4, exog=x.reshape(-1, 1), maxorder=3, trend="c", ic=criterion
            )
        ar_lags = selection.model.ar_lags or []
        dl_lags = next(iter(selection.model.dl_lags.values()))
        assert ours == (max(ar_lags, default=0), max(dl_lags, default=0))

    @pytest.mark.parametrize("criterion", ["aic", "bic"])
    def test_does_not_depend_on_units(self, criterion: str) -> None:
        rng = np.random.default_rng(2)
        n = 136
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        choices = {
            rg.ADL(y * scale, x, lags=1, exog_lags=0).select_lags(
                max_ar_lags=4, max_exog_lags=4, criterion=criterion
            )
            for scale in (1.0, 100.0, 10000.0)
        }
        assert len(choices) == 1

    def test_every_candidate_uses_the_same_sample(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        y, x = _adl_data(seed=3, n=80)
        sizes = set()
        original = rg.ADL.fit

        def recording(self: rg.ADL, *args: object, **kwargs: object) -> object:
            results = original(self, *args, **kwargs)
            sizes.add(results.nobs)
            return results

        monkeypatch.setattr(rg.ADL, "fit", recording)
        rg.ADL(y, x, lags=1, exog_lags=0).select_lags(max_ar_lags=3, max_exog_lags=2)
        assert sizes == {80 - 3}

    def test_argument_checks(self) -> None:
        y, x = _adl_data(seed=4, n=60)
        model = rg.ADL(y, x, lags=1, exog_lags=0)
        with pytest.raises(ValueError, match="criterion"):
            model.select_lags(criterion="hqic")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="non-negative"):
            model.select_lags(max_ar_lags=-1)


class TestBreakValidation:
    def test_normalize_breaks(self) -> None:
        assert normalize_breaks([90, 60], 100) == [60, 90]
        assert normalize_breaks(None, 100) == []
        with pytest.raises(ValueError, match="out of bounds"):
            normalize_breaks([0], 100)
        with pytest.raises(ValueError, match="out of bounds"):
            normalize_breaks([100], 100)
        with pytest.raises(ValueError, match="distinct"):
            normalize_breaks([50, 50], 100)

    def test_break_at_maxlag_is_an_error(self) -> None:
        """It used to return three coefficients equal to 0 without warning."""
        rng = np.random.default_rng(5)
        y = rng.standard_normal(130)
        with pytest.raises(ValueError, match="initial values"):
            rg.AR(y, lags=2, breaks=[2]).fit()
        with pytest.raises(ValueError, match="initial values"):
            rg.AR(y, lags=2, breaks=[1]).fit()
        with pytest.raises(ValueError, match="initial values"):
            rg.AR(y, lags=2, variable_breaks={"const": [2]}).fit()

    def test_regime_with_fewer_observations_than_coefficients(self) -> None:
        """``breaks=[3]`` with two lags: three coefficients on one observation."""
        rng = np.random.default_rng(6)
        y = rng.standard_normal(130)
        with pytest.raises(ValueError, match="not identified"):
            rg.AR(y, lags=2, breaks=[3]).fit()
        X = np.column_stack([np.ones(60), rng.standard_normal((60, 2))])
        with pytest.raises(ValueError, match="not identified"):
            rg.OLS(rng.standard_normal(60), X, has_constant=False, breaks=[2]).fit()
        # the shortest regime that can be estimated
        rg.OLS(rng.standard_normal(60), X, has_constant=False, breaks=[3]).fit()

    def test_adl_breaks(self) -> None:
        y, x = _adl_data(seed=7, n=120)
        with pytest.raises(ValueError, match="initial values"):
            rg.ADL(y, x, lags=1, exog_lags=2, breaks=[2]).fit()
        with pytest.raises(ValueError, match="not identified"):
            rg.ADL(y, x, lags=1, exog_lags=2, breaks=[4]).fit()
        rg.ADL(y, x, lags=1, exog_lags=2, breaks=[60]).fit()

    def test_break_at_zero_and_duplicates(self) -> None:
        y = np.random.default_rng(8).standard_normal(50)
        with pytest.raises(ValueError, match="out of bounds"):
            rg.OLS(y, breaks=[0])
        with pytest.raises(ValueError, match="distinct"):
            rg.OLS(y, breaks=[20, 20])

    def test_unsorted_breaks_are_sorted(self) -> None:
        rng = np.random.default_rng(9)
        y = rng.standard_normal(120)
        unsorted = rg.OLS(y, breaks=[90, 60]).fit()
        ordered = rg.OLS(y, breaks=[60, 90]).fit()
        np.testing.assert_allclose(unsorted.params, ordered.params)
        text = unsorted.summary(diagnostics=False)
        assert "Regime 2: 60-89 (n=30)" in text
        assert "n=-" not in text

    def test_check_break_design(self) -> None:
        X = np.column_stack([np.ones(5), np.zeros(5)])
        with pytest.raises(ValueError, match="no observations.*b"):
            check_break_design(X, ["a", "b"])
        X = np.column_stack([np.ones(5), np.ones(5)])
        with pytest.raises(ValueError, match="rank 1"):
            check_break_design(X, ["a", "b"])


class TestRollingCovType:
    @pytest.mark.parametrize("cov_type", ["HAC", "HC1", "HC3", "hac"])
    def test_unsupported_types_raise(self, cov_type: str) -> None:
        """They used to return HC0 and record the name that was asked for."""
        y, x = _adl_data(seed=10, n=90)
        X = np.column_stack([np.ones(90), x])
        estimators = [
            rg.RollingOLS(y, X, window=40),
            rg.RecursiveOLS(y, X, min_nobs=20),
            rg.AR(y, lags=1).rolling(window=40),
            rg.AR(y, lags=1).recursive(min_nobs=20),
            rg.ADL(y, x, lags=1, exog_lags=1).rolling(window=40),
            rg.ADL(y, x, lags=1, exog_lags=1).recursive(min_nobs=20),
        ]
        for estimator in estimators:
            with pytest.raises(ValueError, match="'nonrobust' and 'HC0'"):
                estimator.fit(cov_type=cov_type)  # type: ignore[arg-type]

    def test_supported_types(self) -> None:
        y, x = _adl_data(seed=11, n=90)
        X = np.column_stack([np.ones(90), x])
        plain = rg.RollingOLS(y, X, window=40).fit(cov_type="nonrobust")
        robust = rg.RollingOLS(y, X, window=40).fit(cov_type="HC0")
        assert plain.cov_type == "nonrobust"
        assert robust.cov_type == "HC0"
        assert not np.allclose(plain.bse[60], robust.bse[60])

    def test_from_model_warns_about_breaks(self) -> None:
        y, x = _adl_data(seed=12, n=120)
        X = np.column_stack([np.ones(120), x])
        with pytest.warns(UserWarning, match="ignores the breaks"):
            rg.OLS(y, X, has_constant=False, breaks=[60]).rolling(window=40)
        with pytest.warns(UserWarning, match="ignores the breaks"):
            rg.AR(y, lags=1, breaks=[60]).recursive(min_nobs=20)
        with pytest.warns(UserWarning, match="ignores the breaks"):
            rg.ADL(y, x, lags=1, exog_lags=1, variable_breaks={"const": [60]}).rolling(
                window=40
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rg.OLS(y, X, has_constant=False).rolling(window=40)


class TestSummaryAndFStatistic:
    def test_constant_only_summary(self) -> None:
        """``rg.OLS(y).fit().summary()`` raised ValueError (White test)."""
        y = np.random.default_rng(13).standard_normal(80)
        results = rg.OLS(y).fit()
        text = results.summary()
        assert "Hetero test (White)" in text
        assert "n/a" in text
        diagnostics = results.diagnostics()
        assert np.isnan(diagnostics.heteroskedasticity.pvalue)
        assert isinstance(diagnostics.all_pass, bool)
        assert np.isnan(results.fvalue)
        assert np.isnan(results.f_pvalue)

    def _compare(self, y: np.ndarray, X: np.ndarray) -> None:
        ours = rg.OLS(y, X, has_constant=False).fit()
        theirs = sm.OLS(y, X).fit()
        assert ours.fvalue == pytest.approx(theirs.fvalue, rel=1e-10)
        assert ours.f_pvalue == pytest.approx(theirs.f_pvalue, rel=1e-8)

    def test_fvalue_with_and_without_a_constant(self) -> None:
        rng = np.random.default_rng(14)
        n = 131
        x = rng.standard_normal((n, 2))
        y = 1.0 + x @ np.array([0.8, -0.4]) + rng.standard_normal(n)
        self._compare(y, np.column_stack([np.ones(n), x]))
        # no constant: all coefficients are tested (it was F with one df less)
        self._compare(y, x)
        # one regressor and no constant: it was NaN
        self._compare(y, x[:, :1])

    def test_fvalue_with_regime_constants(self) -> None:
        """Regime constants span a constant: centered sums of squares."""
        rng = np.random.default_rng(15)
        n = 120
        x = rng.standard_normal(n)
        y = 1.0 + 0.5 * x + rng.standard_normal(n)
        y[60:] += 1.0
        ours = rg.OLS(
            y, np.column_stack([np.ones(n), x]), has_constant=False, breaks=[60]
        ).fit()
        step = (np.arange(n) >= 60).astype(float)
        design = np.column_stack([np.ones(n), x, step, x * step])
        theirs = sm.OLS(y, design).fit()
        assert ours.fvalue == pytest.approx(theirs.fvalue, rel=1e-9)
        assert ours.f_pvalue == pytest.approx(theirs.f_pvalue, rel=1e-6, abs=1e-300)


class TestRoots:
    def test_non_consecutive_lags(self) -> None:
        """Lags 1 and 3: a polynomial of degree three, not two."""
        rng = np.random.default_rng(16)
        n = 600
        y = np.zeros(n)
        for t in range(3, n):
            y[t] = 0.3 * y[t - 1] + 0.4 * y[t - 3] + rng.standard_normal()
        results = rg.AR(y, lags=[1, 3]).fit()
        assert len(results.roots) == 3
        phi1, phi3 = results.ar_params
        expected = np.roots([-phi3, 0.0, -phi1, 1.0])
        np.testing.assert_allclose(
            np.sort_complex(results.roots), np.sort_complex(expected)
        )
        assert results.is_stationary

    def test_every_regime_is_checked(self) -> None:
        """A second regime with a root inside the unit circle."""
        rng = np.random.default_rng(17)
        n = 300
        y = np.zeros(n)
        for t in range(1, n):
            phi = 0.3 if t < 150 else 1.02
            y[t] = phi * y[t - 1] + rng.standard_normal()
        results = rg.AR(y, lags=1, breaks=[150]).fit()
        assert len(results.roots_by_regime) == 2
        first, second = (abs(r[0]) for r in results.roots_by_regime)
        assert first > 1
        assert second < 1
        # `roots` stays the first regime; the check covers both
        np.testing.assert_allclose(results.roots, results.roots_by_regime[0])
        assert not results.is_stationary
        assert "WARNING" in results.summary(diagnostics=False)

    def test_variable_specific_breaks(self) -> None:
        rng = np.random.default_rng(18)
        y = rng.standard_normal(200)
        constant_only = rg.AR(y, lags=2, variable_breaks={"const": [100]}).fit()
        # the autoregressive coefficients do not break: one polynomial
        assert len(constant_only.roots_by_regime) == 1
        one_lag = rg.AR(y, lags=2, variable_breaks={"y.L1": [100]}).fit()
        assert len(one_lag.roots_by_regime) == 2

    def test_adl_roots(self) -> None:
        y, x = _adl_data(seed=19)
        results = rg.ADL(y, x, lags=2, exog_lags=1, breaks=[110]).fit()
        assert len(results.roots_by_regime) == 2
        assert all(len(roots) == 2 for roots in results.roots_by_regime)


class TestLongRunMultiplierByRegime:
    def test_each_regime_is_reported(self) -> None:
        rng = np.random.default_rng(20)
        n = 400
        x = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(1, n):
            beta = 0.5 if t < 200 else 1.5
            y[t] = 0.4 * y[t - 1] + beta * x[t] + 0.5 * rng.standard_normal()
        results = rg.ADL(y, x, lags=1, exog_lags=0, breaks=[200]).fit()
        effects = results.effects_by_regime
        assert [start for start, _ in effects] == [0, 200]
        table = dict(zip(results.param_names, results.params, strict=True))
        for regime, (_start, by_variable) in enumerate(effects, start=1):
            cumulative, long_run = by_variable["x0"]
            beta = table[f"x0_regime{regime}"]
            phi = table[f"y.L1_regime{regime}"]
            assert cumulative == pytest.approx(beta)
            assert long_run == pytest.approx(beta / (1 - phi))
        assert effects[1][1]["x0"][1] > 2 * effects[0][1]["x0"][1]
        # the scalar properties are the first regime, as documented
        assert results.long_run_multiplier["x0"] == pytest.approx(
            effects[0][1]["x0"][1]
        )
        text = results.summary(diagnostics=False)
        assert "Regime 1 (from 0)" in text
        assert "Regime 2 (from 200)" in text

    def test_without_breaks(self) -> None:
        y, x = _adl_data(seed=21)
        results = rg.ADL(y, x, lags=2, exog_lags=1).fit()
        assert len(results.effects_by_regime) == 1
        table = dict(zip(results.param_names, results.params, strict=True))
        cumulative = table["x0"] + table["x0.L1"]
        assert results.cumulative_effect["x0"] == pytest.approx(cumulative)
        assert results.long_run_multiplier["x0"] == pytest.approx(
            cumulative / (1 - table["y.L1"] - table["y.L2"])
        )
        assert "Regime 1" not in results.summary(diagnostics=False)

    def test_unit_root_gives_nan(self) -> None:
        from regimes.models._dynamics import distributed_lag_effects_by_segment

        out = distributed_lag_effects_by_segment(
            ["const", "y.L1", "x0"], np.array([0.0, 1.0, 2.0]), {"x0": [0]}, None, None
        )
        assert np.isnan(out[0][1]["x0"][1])
        assert out[0][1]["x0"][0] == 2.0


class TestSmallItems:
    def test_fit_by_regime_needs_common_breaks(self) -> None:
        rng = np.random.default_rng(22)
        y = rng.standard_normal(100)
        X = np.column_stack([np.ones(100), rng.standard_normal(100)])
        with pytest.raises(ValueError, match="common breaks"):
            rg.OLS(
                y, X, has_constant=False, variable_breaks={"x0": [50]}
            ).fit_by_regime()
        with pytest.raises(ValueError, match="common breaks"):
            rg.AR(y, lags=1, variable_breaks={"const": [50]}).fit_by_regime()

    def test_cov_kwds(self) -> None:
        rng = np.random.default_rng(23)
        y = rng.standard_normal(100)
        X = np.column_stack([np.ones(100), rng.standard_normal(100)])
        model = rg.OLS(y, X, has_constant=False)
        with pytest.warns(UserWarning, match="only used with cov_type='HAC'"):
            model.fit(cov_type="HC1", cov_kwds={"maxlags": 3})
        corrected = model.fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        plain = model.fit(
            cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": False}
        )
        np.testing.assert_allclose(
            corrected.bse / plain.bse, np.sqrt(100 / 98), rtol=1e-10
        )
        with pytest.raises(ValueError, match="Unknown cov_type"):
            model.fit(cov_type="HC4")  # type: ignore[arg-type]

    def test_adl_summary_by_regime_uses_the_longest_lag(self) -> None:
        from regimes.models.adl import _results_maxlag

        y, x = _adl_data(seed=24)
        results = rg.ADL(y, x, lags=1, exog_lags=3).fit()
        assert _results_maxlag(results) == 3


class TestAgainstStatsmodels:
    """The estimators equal statsmodels (audit 4.1), as permanent tests."""

    @pytest.mark.parametrize(
        "cov_type", ["nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC"]
    )
    def test_ols_standard_errors(self, cov_type: str) -> None:
        rng = np.random.default_rng(25)
        n = 150
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        y = X @ np.array([1.0, 0.5, -0.3]) + rng.standard_normal(n) * (1 + X[:, 1] ** 2)
        ours = rg.OLS(y, X, has_constant=False).fit(cov_type=cov_type)  # type: ignore[arg-type]
        model = sm.OLS(y, X)
        if cov_type == "HAC":
            maxlags = int(np.floor(4 * (n / 100) ** (2 / 9)))
            theirs = model.fit(
                cov_type="HAC", cov_kwds={"maxlags": maxlags, "use_correction": True}
            )
        elif cov_type == "nonrobust":
            theirs = model.fit()
        else:
            theirs = model.fit(cov_type=cov_type)
        np.testing.assert_allclose(ours.params, theirs.params, rtol=1e-12)
        np.testing.assert_allclose(ours.bse, theirs.bse, rtol=1e-10)
        assert ours.llf == pytest.approx(theirs.llf, rel=1e-12)
        assert ours.aic == pytest.approx(theirs.aic, rel=1e-12)
        assert ours.bic == pytest.approx(theirs.bic, rel=1e-12)

    def test_regime_coefficients_are_subsample_ols(self) -> None:
        rng = np.random.default_rng(26)
        n = 160
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ np.array([1.0, 0.5]) + rng.standard_normal(n)
        ours = rg.OLS(y, X, has_constant=False, breaks=[70]).fit()
        first = sm.OLS(y[:70], X[:70]).fit()
        second = sm.OLS(y[70:], X[70:]).fit()
        np.testing.assert_allclose(ours.params[:2], first.params, rtol=1e-10)
        np.testing.assert_allclose(ours.params[2:], second.params, rtol=1e-10)

    def test_ar_equals_autoreg(self) -> None:
        y, _x = _adl_data(seed=27)
        ours = rg.AR(y, lags=2).fit()
        theirs = AutoReg(y, lags=2, trend="c").fit()
        np.testing.assert_allclose(ours.params, theirs.params, rtol=1e-9)

    def test_adl_equals_ardl(self) -> None:
        y, x = _adl_data(seed=28)
        ours = rg.ADL(y, x, lags=2, exog_lags=1).fit()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theirs = ARDL(y, lags=2, exog=x.reshape(-1, 1), order=1, trend="c").fit()
        np.testing.assert_allclose(ours.params, theirs.params, rtol=1e-8)

    def test_recursive_ols_equals_recursive_ls(self) -> None:
        rng = np.random.default_rng(29)
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ np.array([1.0, 0.5]) + rng.standard_normal(n)
        ours = rg.RecursiveOLS(y, X, min_nobs=20).fit()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theirs = RecursiveLS(y, X).fit()
        expected = theirs.recursive_coefficients.filtered.T
        np.testing.assert_allclose(ours.params[25:], expected[25:], atol=1e-6)
