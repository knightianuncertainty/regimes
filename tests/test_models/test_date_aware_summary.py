"""Tests for date-aware summary output (index parameter).

Tests that all summary methods correctly show date labels when an index
is provided, and preserve existing integer-based output when not.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

import regimes as rg
from regimes.results.base import _obs_label, _obs_range_label

# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestObsLabel:
    """Tests for _obs_label helper."""

    def test_without_index(self) -> None:
        assert _obs_label(40) == "40"
        assert _obs_label(0) == "0"

    def test_with_period_index(self) -> None:
        idx = pd.period_range("1970Q2", periods=200, freq="Q")
        assert _obs_label(0, idx) == "1970Q2"
        assert _obs_label(40, idx) == "1980Q2"

    def test_with_datetime_index(self) -> None:
        idx = pd.date_range("2000-01-01", periods=50, freq="QE")
        label = _obs_label(0, idx)
        assert "2000" in label

    def test_with_string_sequence(self) -> None:
        idx = [f"T{i}" for i in range(100)]
        assert _obs_label(5, idx) == "T5"


class TestObsRangeLabel:
    """Tests for _obs_range_label helper."""

    def test_without_index(self) -> None:
        assert _obs_range_label(0, 39) == "0-39"

    def test_with_period_index(self) -> None:
        idx = pd.period_range("1970Q2", periods=200, freq="Q")
        result = _obs_range_label(0, 39, idx)
        assert "1970Q2" in result
        assert "1980Q1" in result
        # Uses en-dash separator
        assert "\u2013" in result

    def test_with_string_sequence(self) -> None:
        idx = [f"T{i}" for i in range(100)]
        result = _obs_range_label(0, 9, idx)
        assert "T0" in result
        assert "T9" in result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def period_index() -> pd.PeriodIndex:
    """Quarterly PeriodIndex for 200 observations starting 1970Q2."""
    return pd.period_range("1970Q2", periods=200, freq="Q")


@pytest.fixture()
def ols_data_with_break(
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """OLS data with a single structural break at obs 100."""
    n = 200
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = np.concatenate([rng.standard_normal(100), rng.standard_normal(100) + 2])
    return y, X


# ---------------------------------------------------------------------------
# OLS summary tests
# ---------------------------------------------------------------------------


class TestOLSSummaryIndex:
    """OLS summary with index parameter."""

    def test_summary_without_index_unchanged(
        self,
        ols_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
    ) -> None:
        y, X = ols_data_with_break
        model = rg.OLS(y, X, has_constant=False, breaks=[100])
        results = model.fit()
        text = results.summary(diagnostics=False)
        assert "Break at 100" in text
        assert "0-99" in text

    def test_summary_with_period_index(
        self,
        ols_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
        period_index: pd.PeriodIndex,
    ) -> None:
        y, X = ols_data_with_break
        model = rg.OLS(y, X, has_constant=False, breaks=[100])
        results = model.fit()
        text = results.summary(diagnostics=False, index=period_index)
        assert "1995Q2" in text  # break at obs 100 → 1970Q2+100 = 1995Q2
        assert "1970Q2" in text  # start of regime 1
        assert "Break at 100" not in text

    def test_summary_no_breaks_ignores_index(
        self,
        rng: np.random.Generator,
        period_index: pd.PeriodIndex,
    ) -> None:
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)
        model = rg.OLS(y, X, has_constant=False)
        results = model.fit()
        # No breaks → index should not appear
        text = results.summary(diagnostics=False, index=period_index)
        assert "Structural Breaks" not in text

    def test_summary_variable_breaks_with_index(
        self,
        rng: np.random.Generator,
        period_index: pd.PeriodIndex,
    ) -> None:
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)
        model = rg.OLS(
            y, X, has_constant=False, variable_breaks={"x0": [80], "x1": [120]}
        )
        results = model.fit()
        text = results.summary(diagnostics=False, index=period_index)
        assert "1990Q2" in text  # obs 80 → 1970Q2+80 = 1990Q2
        assert "2000Q2" in text  # obs 120 → 1970Q2+120 = 2000Q2
        # Break labels should use dates, not integer-style "observation N"
        assert "break at observation" not in text.lower()
        assert "break at 80" not in text
        assert "break at 120" not in text


# ---------------------------------------------------------------------------
# AR summary tests
# ---------------------------------------------------------------------------


class TestARSummaryIndex:
    """AR summary with index parameter."""

    def test_summary_with_breaks_and_index(
        self,
        rng: np.random.Generator,
        period_index: pd.PeriodIndex,
    ) -> None:
        n = 200
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + rng.standard_normal()
        model = rg.AR(y, lags=1, breaks=[100])
        results = model.fit()
        text = results.summary(diagnostics=False, index=period_index)
        assert "1995Q2" in text
        assert "Break at 100" not in text

    def test_summary_without_index_unchanged(
        self,
        rng: np.random.Generator,
    ) -> None:
        n = 200
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + rng.standard_normal()
        model = rg.AR(y, lags=1, breaks=[100])
        results = model.fit()
        text = results.summary(diagnostics=False)
        assert "Break at 100" in text


# ---------------------------------------------------------------------------
# ADL summary tests
# ---------------------------------------------------------------------------


class TestADLSummaryIndex:
    """ADL summary with index parameter."""

    def test_summary_with_breaks_and_index(
        self,
        rng: np.random.Generator,
        period_index: pd.PeriodIndex,
    ) -> None:
        n = 200
        x = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + 0.3 * x[t] + rng.standard_normal()
        model = rg.ADL(y, x, lags=1, exog_lags=1, breaks=[100])
        results = model.fit()
        text = results.summary(diagnostics=False, index=period_index)
        assert "1995Q2" in text
        assert "Break at 100" not in text


# ---------------------------------------------------------------------------
# Bai-Perron summary tests
# ---------------------------------------------------------------------------


class TestBaiPerronSummaryIndex:
    """BaiPerronResults.summary() with index parameter."""

    def test_break_dates_with_index(
        self,
        ols_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
        period_index: pd.PeriodIndex,
    ) -> None:
        y, X = ols_data_with_break
        model = rg.OLS(y, X, has_constant=False)
        bp = model.bai_perron(max_breaks=3, trimming=0.15)
        text = bp.summary(index=period_index)
        if bp.n_breaks > 0:
            # Break dates should be period labels, not integers
            for b in bp.break_indices:
                assert str(period_index[b]) in text
            # Confidence intervals should also use labels
            if bp.break_ci:
                assert "95% Confidence Intervals" in text

    def test_summary_without_index_unchanged(
        self,
        ols_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
    ) -> None:
        y, X = ols_data_with_break
        model = rg.OLS(y, X, has_constant=False)
        bp = model.bai_perron(max_breaks=3, trimming=0.15)
        text = bp.summary()
        if bp.n_breaks > 0:
            # Should show integer indices
            for b in bp.break_indices:
                assert str(b) in text

    def test_no_breaks_with_index(
        self,
        rng: np.random.Generator,
        period_index: pd.PeriodIndex,
    ) -> None:
        # Constant data → no breaks
        y = rng.standard_normal(200)
        X = np.column_stack([np.ones(200), rng.standard_normal(200)])
        model = rg.OLS(y, X, has_constant=False)
        bp = model.bai_perron(max_breaks=2, trimming=0.15)
        text = bp.summary(index=period_index)
        assert "Selected number of breaks" in text


# ---------------------------------------------------------------------------
# summary_by_regime tests
# ---------------------------------------------------------------------------


class TestSummaryByRegimeIndex:
    """summary_by_regime() with index parameter."""

    def test_ols_summary_by_regime_with_index(
        self,
        ols_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
        period_index: pd.PeriodIndex,
    ) -> None:
        y, X = ols_data_with_break
        model = rg.OLS(y, X, has_constant=False, breaks=[100])
        regime_results = model.fit_by_regime()
        text = rg.summary_by_regime(
            regime_results, breaks=[100], nobs_total=200, index=period_index
        )
        assert "1995Q2" in text
        assert "1970Q2" in text
        assert "(0-99)" not in text

    def test_ols_summary_by_regime_without_index(
        self,
        ols_data_with_break: tuple[
            NDArray[np.floating[Any]], NDArray[np.floating[Any]]
        ],
    ) -> None:
        y, X = ols_data_with_break
        model = rg.OLS(y, X, has_constant=False, breaks=[100])
        regime_results = model.fit_by_regime()
        text = rg.summary_by_regime(regime_results, breaks=[100], nobs_total=200)
        assert "(0-99)" in text

    def test_ar_summary_by_regime_with_index(
        self,
        rng: np.random.Generator,
        period_index: pd.PeriodIndex,
    ) -> None:
        n = 200
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + rng.standard_normal()
        model = rg.AR(y, lags=1, breaks=[100])
        regime_results = model.fit_by_regime()
        text = rg.ar_summary_by_regime(
            regime_results, breaks=[100], nobs_total=200, index=period_index
        )
        assert "1995Q2" in text
        assert "(0-99)" not in text

    def test_adl_summary_by_regime_with_index(
        self,
        rng: np.random.Generator,
        period_index: pd.PeriodIndex,
    ) -> None:
        n = 200
        x = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + 0.3 * x[t] + rng.standard_normal()
        # ADL does not have fit_by_regime; create two separate ADL fits
        # to simulate per-regime results
        model1 = rg.ADL(y[:100], x[:100], lags=1, exog_lags=1)
        model2 = rg.ADL(y[100:], x[100:], lags=1, exog_lags=1)
        res1 = model1.fit()
        res2 = model2.fit()
        text = rg.adl_summary_by_regime(
            [res1, res2], breaks=[100], nobs_total=200, index=period_index
        )
        assert "1995Q2" in text
        assert "(0-99)" not in text
