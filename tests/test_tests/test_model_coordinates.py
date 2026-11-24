"""Break tests built from AR and ADL models report observation numbers of the
model's data.

A model reads ``breaks=[k]`` as observation ``k`` of its data. The tests are
computed on the sample that remains after the initial observations for the
lags; what they report (break dates, candidate dates, crossings) and what
they accept (``break_points``) is expressed in the same observation numbers,
so a date found by a test can be passed back to the model.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import regimes as rg


def _ar_break_data(seed: int = 0, n: int = 240, break_at: int = 130) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(2, n):
        const = 0.0 if t < break_at else 3.0
        y[t] = const + 0.4 * y[t - 1] - 0.2 * y[t - 2] + rng.standard_normal()
    return y


def _arrays(model: rg.AR | rg.ADL) -> tuple[np.ndarray, np.ndarray, int]:
    y, X, _names = model._build_design_matrix()
    return y, X, int(model.maxlag)


class TestBaiPerron:
    def test_break_dates_are_shifted_by_the_lags(self) -> None:
        model = rg.AR(_ar_break_data(), lags=2)
        y, X, maxlag = _arrays(model)
        assert maxlag == 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from_model = model.bai_perron(max_breaks=2)
            from_arrays = rg.BaiPerronTest(y, exog_break=X).fit(max_breaks=2)

        assert from_model.index_offset == 2
        assert from_arrays.index_offset == 0
        assert from_model.n_breaks == from_arrays.n_breaks >= 1
        assert list(from_model.break_indices) == [
            b + 2 for b in from_arrays.break_indices
        ]
        assert list(from_model.sequential_breaks) == [
            b + 2 for b in from_arrays.sequential_breaks
        ]
        for m, dates in from_arrays.breaks_by_m.items():
            assert list(from_model.breaks_by_m[m]) == [b + 2 for b in dates]
        for ell, date in from_arrays.seqf_break.items():
            assert from_model.seqf_break[ell] == (date + 2 if date >= 0 else date)
        for date, (low, high) in from_arrays.break_ci.items():
            assert from_model.break_ci[date + 2] == (low + 2, high + 2)
        for date, (low, high) in from_arrays.break_ci_90.items():
            assert from_model.break_ci_90[date + 2] == (low + 2, high + 2)
        # statistics do not move
        assert from_model.supf_stats == from_arrays.supf_stats
        assert from_model.ssr == from_arrays.ssr

    def test_break_date_can_be_passed_back_to_the_model(self) -> None:
        """The model with the reported breaks has the test's minimal SSR."""
        y = _ar_break_data(seed=1)
        model = rg.AR(y, lags=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bp = model.bai_perron(max_breaks=2)
        assert bp.n_breaks >= 1
        refit = rg.AR(y, lags=2, breaks=list(bp.break_indices)).fit()
        assert refit.ssr == pytest.approx(bp.ssr[bp.n_breaks], rel=1e-9)
        # one observation earlier or later is a different partition
        shifted = rg.AR(y, lags=2, breaks=[b - 2 for b in bp.break_indices]).fit()
        assert shifted.ssr > refit.ssr + 1e-8

    def test_to_ols_is_unchanged(self) -> None:
        model = rg.AR(_ar_break_data(seed=2), lags=2)
        y, X, _ = _arrays(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from_model = model.bai_perron(max_breaks=2).to_ols()
            from_arrays = rg.BaiPerronTest(y, exog_break=X).fit(max_breaks=2).to_ols()
        np.testing.assert_allclose(from_model.params, from_arrays.params)
        assert from_model.nobs == len(y)

    def test_summary_with_a_full_length_index(self) -> None:
        y = _ar_break_data(seed=3)
        index = pd.period_range("2000-01", periods=len(y), freq="M")
        model = rg.AR(y, lags=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bp = model.bai_perron(max_breaks=1, selection="sequential")
        assert bp.n_breaks == 1
        date = bp.break_indices[0]
        text = bp.summary(index=index)
        assert str(index[date]) in text
        # the sample of the test starts after the initial observations
        assert f"Sample:                   {index[2]} to {index[-1]}" in text
        # the model prints the same date for the same break
        model_text = (
            rg.AR(y, lags=2, breaks=[date])
            .fit()
            .summary(diagnostics=False, index=index)
        )
        assert f"Break at {index[date]}" in model_text

    def test_adl_offset_is_the_longest_lag(self) -> None:
        rng = np.random.default_rng(4)
        n = 200
        x = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(3, n):
            shift = 0.0 if t < 110 else 2.5
            y[t] = (
                shift
                + 0.3 * y[t - 1]
                + 0.5 * x[t]
                + 0.4 * x[t - 3]
                + rng.standard_normal()
            )
        model = rg.ADL(y, x, lags=1, exog_lags=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bp = model.bai_perron(max_breaks=1, selection="sequential")
        assert bp.index_offset == 3
        assert bp.n_breaks == 1
        refit = rg.ADL(y, x, lags=1, exog_lags=3, breaks=list(bp.break_indices)).fit()
        assert refit.ssr == pytest.approx(bp.ssr[1], rel=1e-9)

    def test_ols_models_have_no_offset(self) -> None:
        rng = np.random.default_rng(5)
        y = np.concatenate([rng.normal(0, 1, 80), rng.normal(3, 1, 80)])
        bp = rg.OLS(y).bai_perron(max_breaks=1)
        assert bp.index_offset == 0
        assert rg.BaiPerronTest.from_model(rg.OLS(y)).index_offset == 0


class TestChow:
    def test_break_points_are_model_observations(self) -> None:
        y = _ar_break_data(seed=6)
        model = rg.AR(y, lags=2)
        arrays_y, arrays_X, _ = _arrays(model)
        from_model = model.chow_test(break_points=[130, 100])
        from_arrays = rg.ChowTest(arrays_y, exog_break=arrays_X).fit(
            break_points=[128, 98]
        )
        assert from_model.tested_break_points == [130, 100] or set(
            from_model.tested_break_points
        ) == {130, 100}
        assert from_model.f_stats[130] == pytest.approx(from_arrays.f_stats[128])
        assert from_model.f_stats[100] == pytest.approx(from_arrays.f_stats[98])
        assert from_model.index_offset == 2
        assert all(b in (100, 130) for b in from_model.break_indices)

    def test_equals_the_split_the_model_estimates(self) -> None:
        """Chow at ``k`` compares the regimes of ``AR(..., breaks=[k])``."""
        y = _ar_break_data(seed=7)
        model = rg.AR(y, lags=2)
        chow = model.chow_test(break_points=130)
        with_break = rg.AR(y, lags=2, breaks=[130]).fit()
        without = model.fit()
        f_stat = ((without.ssr - with_break.ssr) / 3) / (
            with_break.ssr / (len(y) - 2 - 6)
        )
        assert chow.f_stats[130] == pytest.approx(f_stat, rel=1e-9)

    def test_valid_range_is_in_model_observations(self) -> None:
        y = _ar_break_data(seed=8, n=100, break_at=50)
        model = rg.AR(y, lags=2)
        with pytest.raises(ValueError, match=r"\[3, 99\]"):
            model.chow_test(break_points=2)
        with pytest.raises(ValueError, match=r"\[3, 99\]"):
            model.chow_test(break_points=100)


class TestAndrewsPloberger:
    def test_candidate_dates_and_break(self) -> None:
        y = _ar_break_data(seed=9)
        model = rg.AR(y, lags=2)
        arrays_y, arrays_X, _ = _arrays(model)
        from_model = model.andrews_ploberger()
        from_arrays = rg.AndrewsPlobergerTest(arrays_y, exog_break=arrays_X).fit()
        np.testing.assert_array_equal(
            from_model.candidate_indices, from_arrays.candidate_indices + 2
        )
        np.testing.assert_allclose(from_model.f_sequence, from_arrays.f_sequence)
        assert from_model.sup_f_break_index == from_arrays.sup_f_break_index + 2
        assert list(from_model.break_indices) == [
            b + 2 for b in from_arrays.break_indices
        ]
        assert from_model.pi0 == pytest.approx(from_arrays.pi0)
        assert from_model.sup_f == pytest.approx(from_arrays.sup_f)
        # the date is where the model with a break fits best
        date = from_model.sup_f_break_index
        best = rg.AR(y, lags=2, breaks=[date]).fit().ssr
        assert best <= rg.AR(y, lags=2, breaks=[date - 2]).fit().ssr


class TestCusum:
    def test_observations_and_first_crossing(self) -> None:
        y = _ar_break_data(seed=10)
        model = rg.AR(y, lags=2)
        arrays_y, arrays_X, _ = _arrays(model)
        for from_model, from_arrays in (
            (model.cusum_test(), rg.CUSUMTest(arrays_y, exog=arrays_X).fit()),
            (model.cusum_sq_test(), rg.CUSUMSQTest(arrays_y, exog=arrays_X).fit()),
        ):
            assert from_model.index_offset == 2
            np.testing.assert_array_equal(
                from_model.observations, from_arrays.observations + 2
            )
            # three regressors and two initial observations
            assert from_model.observations[0] == 5
            assert from_model.observations[-1] == len(y) - 1
            np.testing.assert_allclose(
                from_model.statistic_path, from_arrays.statistic_path
            )
            assert from_model.reject == from_arrays.reject
            if from_arrays.first_crossing is not None:
                assert from_model.first_crossing == from_arrays.first_crossing + 2
                assert from_model.crossing_indices == [
                    i + 2 for i in from_arrays.crossing_indices
                ]

    def test_late_start_of_the_recursion(self) -> None:
        """A series that starts at zero: the first two observations of the
        estimation sample do not identify a constant and a slope."""
        rng = np.random.default_rng(12)
        y = np.zeros(140)
        for t in range(2, 140):
            y[t] = 0.5 + 0.5 * y[t - 1] + rng.standard_normal()
        results = rg.AR(y, lags=1).cusum_test()
        # recursion on observations 1, 2, 3 of the series; the first
        # recursive residual belongs to observation 4
        assert results.observations[0] == 4
        assert results.observations[-1] == 139
        assert len(results.statistic_path) == 136

    def test_mean_shift_is_detected(self) -> None:
        model = rg.AR(_ar_break_data(seed=11), lags=2)
        results = model.cusum_test()
        assert results.reject
        assert results.first_crossing is not None
        assert results.first_crossing >= 130
