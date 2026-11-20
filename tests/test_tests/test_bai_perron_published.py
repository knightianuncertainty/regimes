"""Replication of Bai and Perron (2003), Table I.

US ex-post real interest rate, 1961:1-1986:3 (T = 103), a constant as the
only regressor (q = 1, p = 0), trimming 0.15 (h = 15), up to M = 5 breaks,
with serial correlation and different error variances across regimes allowed
(HAC with quadratic spectral kernel, AR(1) bandwidth and prewhitening).

The series is ``real.dat`` from the paper's entry in the Journal of Applied
Econometrics data archive, stored as
``tests/reference_data/bp2003_real_interest_rate.csv``.

References
----------
Bai, J., & Perron, P. (2003). Computation and analysis of multiple
    structural change models. Journal of Applied Econometrics, 18(1), 1-22.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from regimes import BaiPerronTest
from regimes.tests import _bp_core as bp

_REF_DIR = Path(__file__).resolve().parent.parent / "reference_data"

# quarter index: observation 0 is 1961:1
_Q = {"1966:4": 23, "1972:3": 46, "1980:3": 78}


@pytest.fixture(scope="module")
def real_rate() -> np.ndarray:
    return np.loadtxt(_REF_DIR / "bp2003_real_interest_rate.csv", skiprows=1)


@pytest.fixture(scope="module")
def results(real_rate: np.ndarray):
    return BaiPerronTest(real_rate).fit(
        max_breaks=5,
        trimming=0.15,
        selection="sequential",
        robust=True,
        prewhiten=True,
    )


def test_specification(real_rate: np.ndarray, results) -> None:
    assert len(real_rate) == 103
    assert results.h == 15  # floor(0.15 * 103), as in the table
    assert results.max_breaks == 5
    assert results.q == 1 and results.p == 0


def test_supf_tests(results) -> None:
    """SupF_T(k), k = 1..5: 57.91, 43.01, 33.22, 24.77, 18.33 in the table.

    The program gives 33.32 for k = 3, one digit away from the printed 33.22;
    the other four agree to the two printed decimals.
    """
    stats = [results.supf_stats[k] for k in range(1, 6)]
    np.testing.assert_allclose(stats, [57.91, 43.01, 33.32, 24.77, 18.33], atol=0.006)
    # all significant at the 5% level, as starred in the table
    assert all(results.supf_stats[k] > results.supf_critical[k] for k in range(1, 6))


def test_double_maximum_tests(results) -> None:
    """UDmax = WDmax = 57.91."""
    assert results.udmax == pytest.approx(57.91, abs=0.006)
    assert results.wdmax == pytest.approx(57.91, abs=0.006)
    assert results.udmax > results.udmax_critical
    assert results.wdmax > results.wdmax_critical


def test_sequential_tests(results) -> None:
    """SupF(2|1) = 33.93, SupF(3|2) = 14.72, SupF(4|3) = 0.03."""
    assert results.seqf_stats[1] == pytest.approx(33.93, abs=0.006)
    assert results.seqf_stats[2] == pytest.approx(14.72, abs=0.006)
    assert results.seqf_stats[3] == pytest.approx(0.03, abs=0.006)
    # the first two are starred (5%), the third is not
    assert results.seqf_stats[1] > results.seqf_critical[1]
    assert results.seqf_stats[2] > results.seqf_critical[2]
    assert results.seqf_stats[3] < results.seqf_critical[3]


def test_number_of_breaks_selected(results) -> None:
    """Sequential: 3, LWZ: 2, BIC: 2."""
    assert results.sequential_n_breaks == 3
    assert results.n_breaks == 3
    assert min(results.bic, key=lambda m: results.bic[m]) == 2
    assert min(results.lwz, key=lambda m: results.lwz[m]) == 2


def test_break_dates(results) -> None:
    """T1 = 66:4, T2 = 72:3, T3 = 80:3 (last observation of each regime)."""
    expected = [_Q["1966:4"] + 1, _Q["1972:3"] + 1, _Q["1980:3"] + 1]
    assert list(results.breaks_by_m[3]) == expected
    assert list(results.break_indices) == expected  # sequential dates agree


def test_regime_means(real_rate: np.ndarray, results) -> None:
    """delta = 1.82, 0.87, -1.80, 5.64."""
    vb = np.array(results.breaks_by_m[3]) - 1
    delta = bp.estimate(real_rate, np.ones((103, 1)), None, vb)[0]
    np.testing.assert_allclose(delta, [1.82, 0.87, -1.80, 5.64], atol=0.006)
