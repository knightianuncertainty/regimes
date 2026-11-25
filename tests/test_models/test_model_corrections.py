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


# ---------------------------------------------------------------------------
# Added after the independent review of these corrections
# ---------------------------------------------------------------------------


def _two_regressor_data(seed: int, n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = (
            0.3
            + 0.5 * y[t - 1]
            - 0.25 * y[t - 2]
            + 0.9 * X[t, 0]
            + 0.6 * X[t - 1, 0]
            - 0.7 * X[t, 1]
            + 0.5 * X[t - 1, 1]
            + rng.standard_normal()
        )
    return y, X


class TestSelectLagsGrid:
    """The criteria of every candidate equal statsmodels' ARDL on the common
    sample, also with two regressors and other trends."""

    @pytest.mark.parametrize("trend", ["n", "c", "ct"])
    @pytest.mark.parametrize("criterion", ["aic", "bic"])
    def test_choice_equals_a_grid_of_ardl_models(
        self, trend: str, criterion: str
    ) -> None:
        y, X = _two_regressor_data(seed=30)
        max_p, max_q = 3, 2
        hold_back = max(max_p, max_q)
        best, best_ic = None, np.inf
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                fit = ARDL(y, p, X, q, trend=trend, hold_back=hold_back).fit()
                ic = getattr(fit, criterion)
                if ic < best_ic:
                    best, best_ic = (p, q), ic
        ours = rg.ADL(y, X, lags=1, exog_lags=0, trend=trend).select_lags(
            max_ar_lags=max_p, max_exog_lags=max_q, criterion=criterion
        )
        assert ours == best
        # the data were generated with two lags of y and one of each x
        if criterion == "bic":
            assert ours == (2, 1)

    def test_sample_too_short_raises(self) -> None:
        rng = np.random.default_rng(31)
        y, x = rng.standard_normal(12), rng.standard_normal(12)
        model = rg.ADL(y, x, lags=1, exog_lags=0)
        # 12 - 4 = 8 observations for 1 + 4 + 5 = 10 coefficients
        with pytest.raises(ValueError, match="too short"):
            model.select_lags(max_ar_lags=4, max_exog_lags=4)
        assert model.select_lags(max_ar_lags=1, max_exog_lags=1) in {
            (p, q) for p in range(2) for q in range(2)
        }


class TestBreakDatesAreIntegers:
    def test_non_integers_are_errors(self) -> None:
        for bad in ([100.7], [True], ["100"], [100.0]):
            with pytest.raises(TypeError, match="must be integers"):
                normalize_breaks(bad, 200)  # type: ignore[arg-type]
        assert normalize_breaks([np.int64(120), np.int32(40)], 200) == [40, 120]
        rng = np.random.default_rng(33)
        y = rng.standard_normal(100)
        X = np.column_stack([np.ones(100), rng.standard_normal(100)])
        with pytest.raises(TypeError, match="must be integers"):
            rg.OLS(y, X, has_constant=False, breaks=[50.5])  # type: ignore[list-item]
        with pytest.raises(TypeError, match="must be integers"):
            rg.AR(y, lags=1, variable_breaks={"const": [50.5]}).fit()  # type: ignore[list-item]

    def test_variable_breaks_are_validated_like_common_breaks(self) -> None:
        rng = np.random.default_rng(34)
        y = rng.standard_normal(100)
        X = np.column_stack([np.ones(100), rng.standard_normal(100)])
        for breaks, error in (
            ([50, 50], "distinct"),
            ([0], "out of bounds"),
            ([100], "out of bounds"),
        ):
            with pytest.raises(ValueError, match=error):
                rg.OLS(y, X, has_constant=False, variable_breaks={"x1": breaks}).fit()
            with pytest.raises(ValueError, match=error):
                rg.AR(y, lags=1, variable_breaks={"const": breaks}).fit()
            with pytest.raises(ValueError, match=error):
                rg.ADL(
                    y, X[:, 1], lags=1, exog_lags=0, variable_breaks={"const": breaks}
                ).fit()

    def test_integer_keys(self) -> None:
        rng = np.random.default_rng(35)
        y = rng.standard_normal(120)
        x = rng.standard_normal(120)
        X = np.column_stack([np.ones(120), x])
        by_name = rg.OLS(y, X, has_constant=False, variable_breaks={"x1": [70, 30]})
        for key in (1, np.int64(1)):
            by_number = rg.OLS(
                y, X, has_constant=False, variable_breaks={key: [30, 70]}
            )
            np.testing.assert_allclose(by_number.fit().params, by_name.fit().params)
        ar_name = rg.AR(y, lags=1, variable_breaks={"y.L1": [60]}).fit()
        ar_number = rg.AR(y, lags=1, variable_breaks={1: [60]}).fit()
        np.testing.assert_allclose(ar_number.params, ar_name.params)
        adl_name = rg.ADL(y, x, lags=1, exog_lags=0, variable_breaks={"x0": [60]}).fit()
        adl_number = rg.ADL(y, x, lags=1, exog_lags=0, variable_breaks={2: [60]}).fit()
        np.testing.assert_allclose(adl_number.params, adl_name.params)
        with pytest.raises(TypeError, match="names or column"):
            rg.OLS(y, X, has_constant=False, variable_breaks={True: [60]}).fit()


class TestBreakDesignMessagesAndScale:
    def test_dummy_that_is_zero_in_a_regime(self) -> None:
        """Not an empty regime: the message points to `variable_breaks`."""
        rng = np.random.default_rng(36)
        n = 120
        impulse = np.zeros(n)
        impulse[80] = 1.0
        X = np.column_stack([np.ones(n), rng.standard_normal(n), impulse])
        y = rng.standard_normal(n)
        with pytest.raises(ValueError, match="variable_breaks") as info:
            rg.OLS(y, X, has_constant=False, breaks=[60]).fit()
        assert "zero in every observation" in str(info.value)
        # the way out that the message names
        results = rg.OLS(
            y, X, has_constant=False, variable_breaks={"x0": [60], "x1": [60]}
        ).fit()
        assert len(results.params) == 5

    def test_rank_does_not_depend_on_units(self) -> None:
        rng = np.random.default_rng(37)
        n = 120
        x = rng.standard_normal(n)
        y = 1.0 + 0.5 * x + rng.standard_normal(n)
        small = rg.OLS(
            y, np.column_stack([np.ones(n), x]), has_constant=False, breaks=[60]
        )
        large = rg.OLS(
            y, np.column_stack([np.ones(n), 1e9 * x]), has_constant=False, breaks=[60]
        )
        np.testing.assert_allclose(
            large.fit().params * [1, 1e9, 1, 1e9], small.fit().params, rtol=1e-6
        )
        # collinear within the second regime, whatever the units
        for scale in (1.0, 1e9):
            z = scale * x.copy()
            z[60:] = scale * 2.0
            with pytest.raises(ValueError, match="not identified"):
                rg.OLS(
                    y, np.column_stack([np.ones(n), z]), has_constant=False, breaks=[60]
                ).fit()


class TestFitByRegimeChecks:
    def test_ols(self) -> None:
        rng = np.random.default_rng(38)
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        y = rng.standard_normal(n)
        with pytest.raises(ValueError, match="regime 2 has 3 observations"):
            rg.OLS(y, X, has_constant=False, breaks=[97]).fit_by_regime()
        X[50:, 2] = 4.0
        with pytest.raises(ValueError, match="regime 2: the regressors are collinear"):
            rg.OLS(y, X, has_constant=False, breaks=[50]).fit_by_regime()
        results = rg.OLS(y, X[:, :2], has_constant=False, breaks=[50]).fit_by_regime()
        assert [r.nobs for r in results] == [50, 50]

    def test_ar(self) -> None:
        rng = np.random.default_rng(39)
        y = rng.standard_normal(100)
        # 4 observations in the last regime: 2 initial values, 2 left for 3
        # coefficients
        with pytest.raises(ValueError, match="regime 2 has 2 observations"):
            rg.AR(y, lags=2, breaks=[96]).fit_by_regime()
        with pytest.raises(ValueError, match="initial values"):
            rg.AR(y, lags=4, breaks=[97]).fit_by_regime()
        results = rg.AR(y, lags=2, breaks=[50]).fit_by_regime()
        assert [r.nobs for r in results] == [48, 48]


class TestClassicalFStatistic:
    @pytest.mark.parametrize("cov_type", ["HC1", "HAC"])
    def test_does_not_use_the_covariance_estimator(self, cov_type: str) -> None:
        rng = np.random.default_rng(40)
        n = 150
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        y = X @ [1.0, 0.4, 0.0] + rng.standard_normal(n) * (1 + np.abs(X[:, 1]))
        classical = sm.OLS(y, X).fit()
        robust = rg.OLS(y, X, has_constant=False).fit(cov_type=cov_type)
        assert robust.fvalue == pytest.approx(classical.fvalue, rel=1e-10)
        assert robust.f_pvalue == pytest.approx(classical.f_pvalue, rel=1e-8)
        assert "classical one, not a robust test" in robust.summary(diagnostics=False)
        plain = rg.OLS(y, X, has_constant=False).fit()
        assert "classical one" not in plain.summary(diagnostics=False)

    def test_constant_detection_follows_statsmodels(self) -> None:
        rng = np.random.default_rng(41)
        n = 90
        x = rng.standard_normal(n)
        y = 0.5 * x + rng.standard_normal(n)
        level = 1e6 + rng.standard_normal(n)
        dummies = np.column_stack(
            [(np.arange(n) % 3 == j).astype(float) for j in range(3)]
        )
        designs = {
            "a constant that is not one": np.column_stack([np.full(n, 2.0), x]),
            "a regressor far from zero, no constant": np.column_stack([level, x]),
            "a full set of dummies": np.column_stack([dummies, x]),
            "no constant": np.column_stack([x, x**2]),
        }
        for label, X in designs.items():
            theirs = sm.OLS(y, X).fit()
            ours = rg.OLS(y, X, has_constant=False).fit()
            assert ours.fvalue == pytest.approx(theirs.fvalue, rel=1e-6), label
            assert ours.f_pvalue == pytest.approx(theirs.f_pvalue, rel=1e-6), label


class TestWarningsPointAtTheCaller:
    def test_all_from_model_paths(self) -> None:
        y, x = _adl_data(seed=42, n=120)
        X = np.column_stack([np.ones(120), x])
        ols = rg.OLS(y, X, has_constant=False, breaks=[60])
        ar = rg.AR(y, lags=1, breaks=[60])
        adl = rg.ADL(y, x, lags=1, exog_lags=1, breaks=[60])
        calls = [
            lambda: rg.RollingOLS.from_model(ols, window=40),
            lambda: rg.RecursiveOLS.from_model(ols, min_nobs=20),
            lambda: rg.RollingAR.from_model(ar, window=40),
            lambda: rg.RecursiveAR.from_model(ar, min_nobs=20),
            lambda: rg.RollingADL.from_model(adl, window=40),
            lambda: rg.RecursiveADL.from_model(adl, min_nobs=20),
            lambda: ols.rolling(window=40),
            lambda: ar.recursive(min_nobs=20),
            lambda: adl.rolling(window=40),
        ]
        for call in calls:
            with pytest.warns(UserWarning, match="ignores the breaks") as record:
                call()
            assert len(record) == 1
            assert record[0].filename == __file__

    def test_cov_kwds_warning(self) -> None:
        y, x = _adl_data(seed=43, n=100)
        for model in (
            rg.OLS(y, np.column_stack([np.ones(100), x]), has_constant=False),
            rg.AR(y, lags=1),
            rg.ADL(y, x, lags=1, exog_lags=0),
        ):
            with pytest.warns(UserWarning, match="only used with") as record:
                model.fit(cov_type="HC0", cov_kwds={"maxlags": 2})
            assert record[0].filename == __file__


class TestHacDefaultLag:
    """The default lag is the fixed rule on the estimation sample."""

    def test_ar_and_adl(self) -> None:
        y, x = _adl_data(seed=44, n=260)
        n_eff = 260 - 2
        maxlags = int(np.floor(4 * (n_eff / 100) ** (2 / 9)))
        assert maxlags == 4
        ar = rg.AR(y, lags=2).fit(cov_type="HAC")
        theirs = AutoReg(y, lags=2, trend="c").fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags, "use_correction": True}
        )
        np.testing.assert_allclose(ar.bse, theirs.bse, rtol=1e-8)
        other = AutoReg(y, lags=2, trend="c").fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags + 1, "use_correction": True}
        )
        assert not np.allclose(ar.bse, other.bse, rtol=1e-6)
        adl = rg.ADL(y, x, lags=2, exog_lags=1).fit(cov_type="HAC")
        theirs = ARDL(y, 2, x.reshape(-1, 1), 1, trend="c").fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags, "use_correction": True}
        )
        np.testing.assert_allclose(adl.bse, theirs.bse, rtol=1e-8)


class TestDynamicsValues:
    def test_roots_of_a_known_polynomial(self) -> None:
        """Roots computed from the estimates, regime by regime."""
        rng = np.random.default_rng(45)
        n = 400
        y = np.zeros(n)
        for t in range(2, n):
            if t < 200:
                y[t] = 0.5 * y[t - 1] + 0.2 * y[t - 2] + rng.standard_normal()
            else:
                y[t] = -0.4 * y[t - 1] + 0.3 * y[t - 2] + rng.standard_normal()
        results = rg.AR(y, lags=2, breaks=[200]).fit()
        names = list(results.param_names)
        for regime, roots in enumerate(results.roots_by_regime):
            phi1 = results.params[names.index(f"y.L1_regime{regime + 1}")]
            phi2 = results.params[names.index(f"y.L2_regime{regime + 1}")]
            expected = np.roots([-phi2, -phi1, 1.0])
            np.testing.assert_allclose(
                np.sort_complex(roots), np.sort_complex(expected), rtol=1e-10
            )
        assert np.asarray(results.roots).dtype == np.complex128

    def test_adl_roots_equal_ar_polynomial_roots(self) -> None:
        y, x = _adl_data(seed=46)
        results = rg.ADL(y, x, lags=2, exog_lags=1).fit()
        names = list(results.param_names)
        phi1 = results.params[names.index("y.L1")]
        phi2 = results.params[names.index("y.L2")]
        np.testing.assert_allclose(
            np.sort_complex(results.roots),
            np.sort_complex(np.roots([-phi2, -phi1, 1.0])),
            rtol=1e-10,
        )

    def test_effects_with_variable_specific_breaks(self) -> None:
        """Only the coefficient on x breaks: the stretches are those of x."""
        rng = np.random.default_rng(47)
        n = 400
        x = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(1, n):
            beta = 1.0 if t < 200 else 3.0
            y[t] = 0.5 * y[t - 1] + beta * x[t] + 0.5 * rng.standard_normal()
        results = rg.ADL(y, x, lags=1, exog_lags=0, variable_breaks={"x0": [200]}).fit()
        effects = results.effects_by_regime
        assert [start for start, _ in effects] == [0, 200]
        names = list(results.param_names)
        phi = results.params[names.index("y.L1")]
        for (_start, by_variable), regime in zip(effects, (1, 2)):
            beta = results.params[names.index(f"x0_regime{regime}")]
            cumulative, long_run = by_variable["x0"]
            assert cumulative == pytest.approx(beta)
            assert long_run == pytest.approx(beta / (1 - phi))
        assert effects[0][1]["x0"][1] == pytest.approx(2.0, abs=0.3)
        assert effects[1][1]["x0"][1] == pytest.approx(6.0, abs=0.6)


class TestAdlNamesAndSummaries:
    def test_dataframe_columns_name_the_regressors(self) -> None:
        import pandas as pd

        rng = np.random.default_rng(48)
        frame = pd.DataFrame(
            {"income": rng.standard_normal(120), "rate": rng.standard_normal(120)}
        )
        y = pd.Series(rng.standard_normal(120), name="consumption")
        by_name = rg.ADL(y, frame, lags=1, exog_lags={"income": 2, "rate": 0}).fit()
        assert list(by_name.param_names) == [
            "const",
            "y.L1",
            "income",
            "income.L1",
            "income.L2",
            "rate",
        ]
        by_number = rg.ADL(
            y.to_numpy(), frame.to_numpy(), lags=1, exog_lags={0: 2, 1: 0}
        ).fit()
        np.testing.assert_allclose(by_name.params, by_number.params)
        series = rg.ADL(y, frame["income"], lags=1, exog_lags=1).fit()
        assert "income.L1" in series.param_names

    def test_adl_summary_by_regime(self) -> None:
        y, x = _adl_data(seed=49, n=200)
        pieces = [
            rg.ADL(y[:100], x[:100], lags=1, exog_lags=3).fit(),
            rg.ADL(y[100:], x[100:], lags=1, exog_lags=3).fit(),
        ]
        text = rg.adl_summary_by_regime(pieces, diagnostics=False)
        # each piece lost three initial observations: the regimes are
        # observations 0-99 and 100-199, not 0-97 and 98-195
        assert "Regime 1 (0-99)" in text
        assert "Regime 2 (100-199)" in text
        with_breaks = rg.adl_summary_by_regime(pieces, breaks=[100], diagnostics=False)
        assert "Breaks at: 100" in with_breaks


class TestWhiteTestConstant:
    def test_constant_of_any_value(self) -> None:
        from statsmodels.stats.diagnostic import het_white

        from regimes.diagnostics.misspec import heteroskedasticity_test

        rng = np.random.default_rng(50)
        n = 150
        x = rng.standard_normal(n)
        resid = rng.standard_normal(n) * (1 + 0.5 * np.abs(x))
        only_constant = heteroskedasticity_test(resid, np.full((n, 1), 2.0))
        assert np.isnan(only_constant.statistic)
        with_two = heteroskedasticity_test(resid, np.column_stack([np.full(n, 2.0), x]))
        reference = het_white(resid, np.column_stack([np.ones(n), x]))
        assert with_two.statistic == pytest.approx(reference[0], rel=1e-8)
        assert with_two.pvalue == pytest.approx(reference[1], rel=1e-8)


class TestSecondReview:
    def test_names_reach_the_estimators_built_from_an_adl_model(self) -> None:
        import pandas as pd

        rng = np.random.default_rng(51)
        frame = pd.DataFrame(
            {"income": rng.standard_normal(150), "rate": rng.standard_normal(150)}
        )
        y = rng.standard_normal(150)
        model = rg.ADL(y, frame, lags=1, exog_lags={"income": 1})
        names = ["const", "y.L1", "income", "income.L1", "rate"]
        assert list(model.fit().param_names) == names
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert list(model.rolling(60).fit().param_names) == names
            assert list(model.recursive(min_nobs=30).fit().param_names) == names
            direct = rg.RollingADL(
                y, frame, lags=1, exog_lags={"income": 1}, window=60
            ).fit()
            assert list(direct.param_names) == names
            markov = model.markov_switching(2)
        assert "income.L1[0]" in list(markov.param_names)
        # the estimates are those of the same call with column numbers
        by_number = rg.ADL(y, frame.to_numpy(), lags=1, exog_lags={0: 1})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            np.testing.assert_allclose(
                model.rolling(60).fit().params, by_number.rolling(60).fit().params
            )

    def test_numpy_integers_in_lag_arguments(self) -> None:
        """``exog_lags=np.int64(2)`` silently meant no lags of x."""
        y, x = _adl_data(seed=52, n=150)
        plain = rg.ADL(y, x, lags=2, exog_lags=2).fit()
        numpy_ints = rg.ADL(y, x, lags=np.int64(2), exog_lags=np.int64(2)).fit()
        assert list(numpy_ints.param_names) == list(plain.param_names)
        np.testing.assert_allclose(numpy_ints.params, plain.params)
        keyed = rg.ADL(y, x, lags=2, exog_lags={np.int64(0): np.int64(2)}).fit()
        np.testing.assert_allclose(keyed.params, plain.params)

    def test_a_single_break_is_still_a_list(self) -> None:
        y = np.random.default_rng(53).standard_normal(100)
        with pytest.raises(TypeError, match=r"write \[50\] for a single break"):
            rg.AR(y, lags=1, breaks=50)  # type: ignore[arg-type]
        assert normalize_breaks(range(40, 42), 100) == [40, 41]

    @pytest.mark.parametrize(("trend", "k_trend"), [("n", 0), ("c", 1), ("ct", 2)])
    def test_select_lags_counts_every_coefficient(
        self, trend: str, k_trend: int
    ) -> None:
        """Two regressors: the largest candidate has k_trend + p + 2 (q + 1)
        coefficients, and needs one observation more than that."""
        max_p, max_q = 2, 1
        k_largest = k_trend + max_p + 2 * (max_q + 1)
        n = max(max_p, max_q) + k_largest  # exactly as many as coefficients
        rng = np.random.default_rng(54)
        y, X = rng.standard_normal(n + 1), rng.standard_normal((n + 1, 2))
        short = rg.ADL(y[:n], X[:n], lags=1, exog_lags=0, trend=trend)
        with pytest.raises(ValueError, match=f"{k_largest} coefficients"):
            short.select_lags(max_ar_lags=max_p, max_exog_lags=max_q)
        enough = rg.ADL(y, X, lags=1, exog_lags=0, trend=trend)
        assert len(enough.select_lags(max_ar_lags=max_p, max_exog_lags=max_q)) == 2

    def test_white_test_does_not_take_a_zero_column_for_the_constant(self) -> None:
        from statsmodels.stats.diagnostic import het_white

        from regimes.diagnostics.misspec import heteroskedasticity_test

        rng = np.random.default_rng(55)
        n = 120
        x = rng.standard_normal(n)
        resid = rng.standard_normal(n) * (1 + np.abs(x))
        without_constant = heteroskedasticity_test(resid, x.reshape(-1, 1))
        reference = het_white(resid, np.column_stack([np.ones(n), x]))
        assert without_constant.statistic == pytest.approx(reference[0], rel=1e-8)
        # a column of zeros is not a constant: one is still added
        with_zeros = heteroskedasticity_test(resid, np.column_stack([np.zeros(n), x]))
        assert np.isfinite(with_zeros.statistic)

    def test_warnings_do_not_depend_on_the_spelling_of_the_path(self) -> None:
        import os

        from regimes import _warn

        inside = _warn.__file__
        assert _warn._inside_package(inside)
        assert not _warn._inside_package(__file__)
        if os.name == "nt":
            assert _warn._inside_package(inside.upper())
            assert _warn._inside_package(inside.lower())


class TestAfterRelease:
    """Small corrections made after v0.5.0."""

    def test_regime_counts_in_summaries_add_up(self) -> None:
        """The first regime of an AR or ADL model starts at the first
        observation of the estimation sample (it used to count the initial
        observations for the lags)."""
        rng = np.random.default_rng(56)
        n = 200
        x = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = (0.0 if t < 100 else 2.0) + 0.4 * y[t - 1] + rng.standard_normal()
        ar = rg.AR(y, lags=2, breaks=[100]).fit().summary(diagnostics=False)
        assert "No. Observations:           198" in ar
        assert "Regime 1: 2-99 (n=98)" in ar
        assert "Regime 2: 100-199 (n=100)" in ar
        adl = rg.ADL(y, x, lags=1, exog_lags=3, breaks=[100]).fit()
        text = adl.summary(diagnostics=False)
        assert "Regime 1: 3-99 (n=97)" in text
        assert "Regime 2: 100-199 (n=100)" in text
        partial = rg.ADL(
            y, x, lags=1, exog_lags=3, variable_breaks={"const": [100]}
        ).fit()
        assert "Regime 1: 3-99, Regime 2: 100-199" in partial.summary(diagnostics=False)
        # a regression has no initial observations
        ols = rg.OLS(
            y, np.column_stack([np.ones(n), x]), has_constant=False, breaks=[100]
        ).fit()
        assert "Regime 1: 0-99 (n=100)" in ols.summary(diagnostics=False)

    @pytest.mark.parametrize("name", ["const", "y.L1", "trend"])
    def test_regressor_with_the_name_of_a_model_term(self, name: str) -> None:
        import pandas as pd

        rng = np.random.default_rng(57)
        y = rng.standard_normal(120)
        frame = pd.DataFrame(rng.standard_normal((120, 2)), columns=[name, "rate"])
        trend = "ct" if name == "trend" else "c"
        with pytest.raises(ValueError, match="parameter names are not unique"):
            rg.ADL(y, frame, lags=1, exog_lags=0, trend=trend).fit()
        with pytest.raises(ValueError, match="parameter names are not unique"):
            rg.AR(y, lags=1, exog=frame, trend=trend).fit()
        # other names are fine
        renamed = frame.rename(columns={name: "income"})
        assert (
            "income"
            in rg.ADL(y, renamed, lags=1, exog_lags=0, trend=trend).fit().param_names
        )


class TestIntegerTypesAndNames:
    """Cases the tests of v0.5.0 did not reach."""

    def test_numpy_integer_lags_of_an_autoregression(self) -> None:
        y = np.random.default_rng(58).standard_normal(120)
        plain = rg.AR(y, lags=2).fit()
        numpy_int = rg.AR(y, lags=np.int64(2)).fit()
        np.testing.assert_allclose(numpy_int.params, plain.params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rolling = rg.RollingAR(y, lags=np.int64(2), window=50).fit()
            recursive = rg.RecursiveAR(y, lags=np.int64(2), min_nobs=30).fit()
        assert list(rolling.param_names) == list(plain.param_names)
        assert list(recursive.param_names) == list(plain.param_names)

    def test_numpy_integers_and_names_in_rolling_adl(self) -> None:
        import pandas as pd

        y, x = _adl_data(seed=59, n=150)
        names = ["const", "y.L1", "x0", "x0.L1", "x0.L2"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for estimator in (
                rg.RollingADL(y, x, lags=1, exog_lags=np.int64(2), window=60),
                rg.RecursiveADL(y, x, lags=1, exog_lags=np.int64(2), min_nobs=40),
                rg.RollingADL(
                    y, x, lags=1, exog_lags={np.int64(0): np.int64(2)}, window=60
                ),
            ):
                assert list(estimator.fit().param_names) == names
            series = pd.Series(x, name="income")
            named = rg.RollingADL(y, series, lags=1, exog_lags=1, window=60).fit()
            markov = rg.MarkovADL(y, x, k_regimes=2, ar_order=1, exog_lags=np.int64(1))
        assert list(named.param_names) == ["const", "y.L1", "income", "income.L1"]
        assert markov._process_exog_lags() == {0: [0, 1]}

    def test_messages_for_break_dates_that_are_not_a_list(self) -> None:
        for bad, hint in (
            (50, True),
            (np.int64(50), True),
            (50.0, False),
            ("50", False),
        ):
            with pytest.raises(TypeError, match="list of integers") as info:
                normalize_breaks(bad, 200)  # type: ignore[arg-type]
            assert ("write [50] for a single break" in str(info.value)) is hint
        with pytest.raises(TypeError, match="list of integers"):
            normalize_breaks(np.array(50), 200)  # type: ignore[arg-type]

    def test_break_dates_from_any_iterable(self) -> None:
        """Sets and generators were accepted in 0.5.0 and still are."""
        assert normalize_breaks({120, 40}, 200) == [40, 120]
        assert normalize_breaks((b for b in (120, 40)), 200) == [40, 120]
        assert normalize_breaks(np.array([120, 40]), 200) == [40, 120]

    def test_two_columns_with_the_same_name(self) -> None:
        import pandas as pd

        rng = np.random.default_rng(60)
        frame = pd.DataFrame(rng.standard_normal((80, 2)), columns=["rate", "rate"])
        with pytest.raises(ValueError, match="Two regressors have the same name"):
            rg.ADL(rng.standard_normal(80), frame, lags=1, exog_lags=0).fit()
