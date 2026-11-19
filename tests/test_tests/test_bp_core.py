"""Tests of the Bai-Perron core algorithms (``regimes.tests._bp_core``).

Reference values in ``TestPerronProgram`` were produced by Pierre Perron's
Bai-Perron program on the example data distributed with it (T = 101,
``tests/reference_data/bp_exdata.csv``), with h = 15, trimming 0.15,
hetdat = hetvar = 1, robust = 0. They are the same numbers the OxRegimes
package is validated against (``tests/test_bpcore.ox`` there). Break dates in
this module are in the program's convention: 0-based last index of a regime.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from regimes import BaiPerronTest
from regimes.tests import _bp_core as bp
from regimes.tests import _bp_critical_values as cv

_REF_DIR = Path(__file__).resolve().parent.parent / "reference_data"


def _load(name: str) -> dict[str, np.ndarray]:
    data = np.genfromtxt(_REF_DIR / name, delimiter=",", names=True)
    return {key: np.asarray(data[key], dtype=float) for key in data.dtype.names}


@pytest.fixture(scope="module")
def exdata() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = _load("bp_exdata.csv")
    return d["Y"], d["X1"][:, None], d["Constant"][:, None]


class TestPerronProgram:
    """Numbers from Perron's program on its own example data."""

    H = 15
    OPT = bp.BPOptions()

    def test_pure_change_partitions(self, exdata) -> None:
        y, z, _ = exdata
        breaks, ssr, _ = bp.partition_all(bp.ssr_triangle(y, z, self.H), 5, self.H)
        assert breaks[0].tolist() == [49]
        assert breaks[1].tolist() == [25, 49]
        assert breaks[2].tolist() == [19, 34, 49]
        assert breaks[4].tolist() == [19, 34, 49, 65, 85]
        np.testing.assert_allclose(
            ssr, [220.456, 215.103, 209.067, 206.353, 203.308], rtol=1e-5
        )

    def test_pure_change_supf(self, exdata) -> None:
        y, z, _ = exdata
        breaks, _, _ = bp.partition_all(bp.ssr_triangle(y, z, self.H), 5, self.H)
        stats = [bp.sup_f(y, z, None, b, self.OPT) for b in breaks]
        np.testing.assert_allclose(
            stats, [1238.9, 640.947, 442.258, 338.552, 281.531], rtol=2e-5
        )

    def test_pure_change_seqf(self, exdata) -> None:
        y, z, _ = exdata
        breaks, _, _ = bp.partition_all(bp.ssr_triangle(y, z, self.H), 5, self.H)
        stat, at = bp.seq_f(y, z, None, breaks[0], self.H, self.OPT)
        assert stat == pytest.approx(2.7212, rel=1e-4)
        assert at == 25
        stat, at = bp.seq_f(y, z, None, breaks[1], self.H, self.OPT)
        assert stat == pytest.approx(1.3690, rel=1e-4)
        assert at == 65
        stat, _ = bp.seq_f(y, z, None, breaks[3], self.H, self.OPT)
        assert stat == pytest.approx(1.0406, rel=1e-4)

    def test_pure_change_sequential(self, exdata) -> None:
        y, z, _ = exdata
        n, vb = bp.sequential(y, z, None, 5, self.H, 0.15, 0.05, self.OPT)
        assert n == 1
        assert vb.tolist() == [49]

    def test_partial_change(self, exdata) -> None:
        y, z, x = exdata
        breaks, ssr, _ = bp.partial_partition_all(y, z, x, 5, self.H)
        assert breaks[0].tolist() == [49]
        assert breaks[1].tolist() == [49, 69]
        assert breaks[2].tolist() == [49, 64, 82]
        assert breaks[4].tolist() == [14, 30, 49, 64, 82]
        np.testing.assert_allclose(
            ssr, [111.854, 110.111, 108.617, 106.813, 106.518], rtol=1e-5
        )
        stats = [bp.sup_f(y, z, x, b, self.OPT) for b in breaks]
        np.testing.assert_allclose(
            stats, [2404.2, 1208.75, 822.537, 644.083, 514.767], rtol=2e-5
        )
        stat, at = bp.seq_f(y, z, x, breaks[0], self.H, self.OPT)
        assert stat == pytest.approx(1.6684, rel=1e-4)
        assert at == 30


class TestSsrTriangle:
    def test_matches_direct_ols(self, rng: np.random.Generator) -> None:
        n, h = 60, 8
        z = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = z @ np.array([1.0, 0.5]) + rng.standard_normal(n)
        s = bp.ssr_triangle(y, z, h)
        for i, j in [(0, 7), (0, 59), (10, 30), (25, 59), (52, 59)]:
            beta = np.linalg.lstsq(z[i : j + 1], y[i : j + 1], rcond=None)[0]
            resid = y[i : j + 1] - z[i : j + 1] @ beta
            assert s[i, j] == pytest.approx(float(resid @ resid), rel=1e-9)

    def test_short_segments_are_nan(self, rng: np.random.Generator) -> None:
        y = rng.standard_normal(40)
        s = bp.ssr_triangle(y, np.ones((40, 1)), 6)
        assert np.isnan(s[3, 7])  # length 5 < h
        assert not np.isnan(s[3, 8])  # length 6 = h
        assert np.isnan(s[10, 5])  # below the diagonal

    def test_dynamic_programming_is_global_minimum(
        self, rng: np.random.Generator
    ) -> None:
        """Two-break optimum equals brute-force search over all partitions."""
        n, h = 36, 5
        y = np.concatenate(
            [rng.normal(0, 1, 12), rng.normal(2, 1, 12), rng.normal(-1, 1, 12)]
        )
        z = np.ones((n, 1))
        s = bp.ssr_triangle(y, z, h)
        breaks, ssr, _ = bp.partition_all(s, 2, h)
        best = min(
            (s[0, a] + s[a + 1, b] + s[b + 1, n - 1], a, b)
            for a in range(h - 1, n - 2 * h)
            for b in range(a + h, n - h)
        )
        assert ssr[1] == pytest.approx(best[0], rel=1e-12)
        assert breaks[1].tolist() == [best[1], best[2]]


class TestSegmentLength:
    def test_rounds_down(self) -> None:
        # Bai and Perron (2003, Table I): T = 103, trimming 0.15, h = 15
        assert bp.segment_length(0.15, 103, 1) == 15
        assert bp.segment_length(0.15, 132, 2) == 19
        assert bp.segment_length(0.05, 216, 2) == 10

    def test_exact_products_are_not_rounded_away(self) -> None:
        # 0.1 * 30 = 3.0000000000000004 and 0.29 * 100 = 28.999999999999996
        assert bp.segment_length(0.10, 30, 1) == 3
        assert bp.segment_length(0.29, 100, 1) == 29
        assert bp.segment_length(0.15, 200, 1) == 30

    def test_at_least_q_plus_one(self) -> None:
        assert bp.segment_length(0.05, 40, 4) == 5


class TestInfoCriteria:
    def test_formulas(self) -> None:
        ssr = np.array([100.0, 60.0, 55.0])
        nobs, q, p = 120, 2, 1
        bic, lwz = bp.info_criteria(ssr, nobs, q, p)
        for m in range(3):
            pstar = (m + 1) * q + m + p  # the m break dates are counted
            assert bic[m] == pytest.approx(
                math.log(ssr[m] / nobs) + pstar * math.log(nobs) / nobs
            )
            assert lwz[m] == pytest.approx(
                math.log(ssr[m] / (nobs - pstar))
                + pstar * 0.299 * math.log(nobs) ** 2.1 / nobs
            )

    def test_each_break_costs_q_plus_one_parameters(self) -> None:
        """With equal SSR, one more break raises BIC by (q + 1) ln(T) / T."""
        nobs, q = 150, 3
        bic, _ = bp.info_criteria(np.array([50.0, 50.0]), nobs, q, 0)
        assert bic[1] - bic[0] == pytest.approx((q + 1) * math.log(nobs) / nobs)

    def test_same_ranking_as_perron_program(self) -> None:
        """Perron's ``order``: ln(SSR/T) + ln(T) m (q + 1) / T."""
        ssr = np.array([90.0, 70.0, 66.0, 65.0])
        nobs, q = 100, 2
        bic, _ = bp.info_criteria(ssr, nobs, q, 0)
        perron = np.log(ssr / nobs) + np.log(nobs) * np.arange(4) * (q + 1) / nobs
        np.testing.assert_allclose(bic - bic[0], perron - perron[0])


class TestScalingAndWeights:
    def test_homoskedastic_supf_is_q_times_chow_f(
        self, rng: np.random.Generator
    ) -> None:
        n, q, b = 120, 3, 59
        z = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        y = z @ np.array([1.0, 0.5, -0.3]) + rng.standard_normal(n)
        y[b + 1 :] += 1.0
        stat = bp.sup_f(y, z, None, [b], bp.BPOptions(hetvar=False))
        ssr0 = float(np.sum((y - z @ np.linalg.lstsq(z, y, rcond=None)[0]) ** 2))
        ssr1 = bp.estimate(y, z, None, [b])[3]
        chow = ((ssr0 - ssr1) / q) / (ssr1 / (n - 2 * q))
        assert stat == pytest.approx(q * chow, rel=1e-10)

    def test_wdmax_weights(self, rng: np.random.Generator) -> None:
        y = np.concatenate([rng.normal(0, 1, 80), rng.normal(1.5, 1, 80)])
        res = BaiPerronTest(y).fit(max_breaks=5, trimming=0.15)
        table = cv.supf_table(0.05, 0.15)
        assert table is not None
        expected = max(
            table[0, 0] / table[0, k - 1] * res.supf_stats[k] for k in range(1, 6)
        )
        assert res.wdmax == pytest.approx(expected)
        assert res.wdmax >= res.udmax  # weights c(1)/c(k) >= 1
        assert res.udmax == pytest.approx(max(res.supf_stats.values()))


class TestCriticalValues:
    def test_published_values(self) -> None:
        sup = cv.supf_table(0.05, 0.15)
        seq = cv.seqf_table(0.05, 0.15)
        dmax = cv.dmax_table(0.05, 0.15)
        assert sup is not None and seq is not None and dmax is not None
        np.testing.assert_array_equal(sup[0], [8.58, 7.22, 5.96, 4.99, 3.91])
        np.testing.assert_array_equal(sup[1], [11.47, 9.75, 8.36, 7.19, 5.85])
        np.testing.assert_array_equal(seq[0, :5], [8.58, 10.13, 11.14, 11.83, 12.25])
        np.testing.assert_array_equal(dmax[0], [8.88, 9.91])
        np.testing.assert_array_equal(dmax[1], [11.70, 12.81])
        seq10 = cv.seqf_table(0.10, 0.15)
        assert seq10 is not None
        np.testing.assert_array_equal(seq10[0, :5], [7.04, 8.51, 9.41, 10.04, 10.58])

    def test_shapes_and_structure(self) -> None:
        for eps in cv.TRIMMINGS:
            for lev in cv.LEVELS:
                sup = cv.supf_table(lev, eps)
                seq = cv.seqf_table(lev, eps)
                dmax = cv.dmax_table(lev, eps)
                assert sup is not None and seq is not None and dmax is not None
                assert sup.shape == (10, cv.MAX_BREAKS[eps])
                assert seq.shape == (10, 10)
                assert dmax.shape == (10, 2)
                # sup F(1|0) is the first column of both tables
                np.testing.assert_array_equal(sup[:, 0], seq[:, 0])
                # "scaled by q": critical values rise with q
                assert np.all(np.diff(sup, axis=0) > 0)
                # sup F(k|0) falls in k, sup F(l+1|l) rises in l
                assert np.all(np.diff(sup, axis=1) < 0)
                assert np.all(np.diff(seq, axis=1) >= 0)
                # WDmax >= UDmax >= sup F(1|0)
                assert np.all(dmax[:, 1] >= dmax[:, 0])
                assert np.all(dmax[:, 0] >= sup[:, 0])

    def test_untabulated(self) -> None:
        assert cv.supf_table(0.05, 0.12) is None
        assert cv.seqf_table(0.03, 0.15) is None
        assert cv.max_breaks(0.12) is None
        assert cv.max_breaks(0.15) == 5


class TestSequentialProcedure:
    def test_short_regimes_are_not_split(self, rng: np.random.Generator) -> None:
        y = rng.standard_normal(60)
        z = np.ones((60, 1))
        stat, at = bp.one_break_sup_f(y, z, None, 0, 19, 12, bp.BPOptions())
        assert np.isnan(stat) and at == -1
        # every regime shorter than 2h: the statistic is NaN
        stat, at = bp.seq_f(y, z, None, [19, 39], 12, bp.BPOptions())
        assert np.isnan(stat) and at == -1

    def test_stops_at_first_non_rejection(self, rng: np.random.Generator) -> None:
        y = np.concatenate(
            [rng.normal(0, 1, 100), rng.normal(4, 1, 100), rng.normal(0, 1, 100)]
        )
        z = np.ones((300, 1))
        opt = bp.BPOptions()
        n, vb = bp.sequential(y, z, None, 5, 45, 0.15, 0.05, opt)
        assert n == 2
        assert all(abs(b - t) <= 3 for b, t in zip(vb, [99, 199]))
        # the next test does not reject
        crit = cv.seqf_table(0.05, 0.15)
        assert crit is not None
        assert bp.seq_f(y, z, None, vb, 45, opt)[0] < crit[0, 2]

    def test_no_break_under_the_null(self) -> None:
        rng = np.random.default_rng(12345)
        y = rng.standard_normal(200)
        n, vb = bp.sequential(
            y, np.ones((200, 1)), None, 5, 30, 0.15, 0.01, bp.BPOptions()
        )
        assert n == 0 and vb.size == 0

    def test_needs_tabulated_trimming(self, rng: np.random.Generator) -> None:
        y = rng.standard_normal(100)
        with pytest.raises(ValueError, match="tabulated"):
            bp.sequential(y, np.ones((100, 1)), None, 3, 12, 0.12, 0.05, bp.BPOptions())


class TestLongRun:
    def test_white_noise(self) -> None:
        rng = np.random.default_rng(7)
        v = rng.standard_normal((2000, 2))
        lr = bp.long_run(v, prewhiten=False)
        np.testing.assert_allclose(lr, np.eye(2), atol=0.2)

    def test_ar1(self) -> None:
        """Long-run variance of an AR(1) is sigma^2 / (1 - rho)^2."""
        rng = np.random.default_rng(11)
        n, rho = 3000, 0.5
        e = rng.standard_normal(n)
        v = np.zeros(n)
        for t in range(1, n):
            v[t] = rho * v[t - 1] + e[t]
        for prewhiten in (False, True):
            lr = bp.long_run(v[:, None], prewhiten)[0, 0]
            assert lr == pytest.approx(1 / (1 - rho) ** 2, rel=0.25)

    def test_prewhitening_is_scale_equivariant(self) -> None:
        """Rescaling the columns by D rescales the estimate to about D S D'.

        Only approximately, because the bandwidth weights the columns equally
        whatever their scale. With the VAR(1) coefficient matrix transposed in
        the recoloring step (as in Perron's GAUSS program for more than one
        series) the two differ by orders of magnitude.
        """
        rng = np.random.default_rng(3)
        n = 400
        v = np.zeros((n, 2))
        e = rng.standard_normal((n, 2))
        a = np.array([[0.4, 0.2], [-0.1, 0.3]])
        for t in range(1, n):
            v[t] = a @ v[t - 1] + e[t]
        d = np.diag([1.0, 250.0])
        base = bp.long_run(v, prewhiten=True)
        scaled = bp.long_run(v @ d, prewhiten=True)
        np.testing.assert_allclose(scaled, d @ base @ d, rtol=0.05)

        # the transposed recoloring is not close to equivariant
        def transposed(vv: np.ndarray) -> np.ndarray:
            lag, lead = vv[:-1], vv[1:]
            a_hat = np.linalg.lstsq(lag, lead, rcond=None)[0].T
            inner = bp.long_run(lead - lag @ a_hat.T, prewhiten=False)
            inv = np.linalg.inv(np.eye(2) - a_hat.T)
            return inv @ inner @ inv.T

        ratio = transposed(v @ d) / (d @ transposed(v) @ d)
        assert np.max(np.abs(ratio - 1)) > 0.5


class TestBreakCI:
    def test_symmetric_quantiles(self) -> None:
        """Symmetric case: 7.69 (90%) and 11.03 (95%), cf. Bai (1997)."""
        q = bp.quantiles_argmax(1.0, 1.0, 1.0)
        assert q[0] == pytest.approx(-q[3], abs=1e-3)
        assert q[1] == pytest.approx(-q[2], abs=1e-3)
        assert q[3] == pytest.approx(11.03, abs=0.01)
        assert q[2] == pytest.approx(7.69, abs=0.01)

    def test_cdf_is_a_distribution_function(self) -> None:
        args = (1.5, 1.0, 1.0, 1.5, 1.0)  # symmetric case
        xs = np.linspace(-40, 40, 81)
        vals = [bp.cdf_argmax(x, *args) for x in xs]
        assert np.all(np.diff(vals) >= -1e-10)
        assert vals[0] < 1e-3 and vals[-1] > 1 - 1e-3
        assert bp.cdf_argmax(0.0, *args) == pytest.approx(0.5, abs=1e-6)

    def test_interval_contains_break(self, rng: np.random.Generator) -> None:
        y = np.concatenate([rng.normal(0, 1, 100), rng.normal(1.0, 1, 100)])
        z = np.ones((200, 1))
        ci = bp.break_ci(y, z, None, [99], bp.BPOptions())
        lo95, hi95, lo90, hi90 = ci[0]
        assert lo95 <= 99 <= hi95
        assert lo95 <= lo90 and hi90 <= hi95

    def test_larger_break_gives_shorter_interval(
        self, rng: np.random.Generator
    ) -> None:
        e = rng.standard_normal(200)
        z = np.ones((200, 1))
        widths = []
        for shift in (0.75, 3.0):
            y = e + shift * (np.arange(200) >= 100)
            ci = bp.break_ci(y, z, None, [99], bp.BPOptions())
            widths.append(ci[0, 1] - ci[0, 0])
        assert widths[1] < widths[0]


def _partial_sample(seed: int, nobs: int = 120) -> tuple[np.ndarray, ...]:
    """Two mean shifts next to a persistent fixed regressor."""
    rng = np.random.default_rng(seed)
    t = np.arange(nobs)
    walk = np.cumsum(rng.standard_normal(nobs)) * 0.4
    b1, b2 = rng.integers(25, 50), rng.integers(70, 95)
    y = (
        0.5 * walk
        + rng.uniform(0.5, 1.5) * (t >= b1)
        - rng.uniform(0.5, 1.5) * (t >= b2)
        + rng.standard_normal(nobs)
    )
    return y, np.ones((nobs, 1)), walk[:, None]


def _joint_ssr(y: np.ndarray, z: np.ndarray, x: np.ndarray, vb: list[int]) -> float:
    w = np.column_stack([bp.zbar(z, vb), x])
    resid = y - w @ np.linalg.lstsq(w, y, rcond=None)[0]
    return float(resid @ resid)


class TestPartialModelSearch:
    """Break dates of partial structural change models."""

    H = 18

    @pytest.mark.parametrize("seed", [7, 43, 56, 60])
    def test_one_break_is_the_exact_minimizer(self, seed: int) -> None:
        y, z, x = _partial_sample(seed)
        breaks, ssr, _ = bp.partial_partition_all(y, z, x, 1, self.H)
        every = [_joint_ssr(y, z, x, [b]) for b in range(self.H - 1, len(y) - self.H)]
        assert breaks[0][0] == self.H - 1 + int(np.argmin(every))
        assert ssr[0] == pytest.approx(min(every))

    @pytest.mark.parametrize("seed", [7, 43, 60])
    def test_supf_one_break_has_one_value(self, seed: int) -> None:
        """sup F(1|0) is the same number wherever the results report it."""
        y, z, x = _partial_sample(seed)
        res = BaiPerronTest(y, exog=x, exog_break=z).fit(max_breaks=3)
        assert res.supf_stats[1] == pytest.approx(res.seqf_stats[0])
        assert res.breaks_by_m[1] == [res.seqf_break[0]]

    @pytest.mark.parametrize("seed", range(12))
    def test_two_starts_are_never_worse_than_one(self, seed: int) -> None:
        y, z, x = _partial_sample(seed, nobs=90)
        h = 13
        breaks, ssr, _ = bp.partial_partition_all(y, z, x, 2, h)
        w0 = np.column_stack([z, x])
        beta0 = np.linalg.lstsq(w0, y, rcond=None)[0][1:]
        _, single, _ = bp._partial_iterate(y, z, x, 2, h, beta0, 1e-8, 100)
        assert ssr[1] <= single + 1e-9
        assert ssr[1] == pytest.approx(_joint_ssr(y, z, x, breaks[1].tolist()))

    def test_repartition_uses_the_joint_search(self) -> None:
        y, z, x = _partial_sample(60)
        vb = [22, 80]
        new = bp.repartition(y, z, x, vb, self.H)
        for i, (lo, hi) in enumerate([(0, 80), (23, len(y) - 1)]):
            seg = slice(lo, hi + 1)
            n = hi - lo + 1
            every = [
                _joint_ssr(y[seg], z[seg], x[seg], [b])
                for b in range(self.H - 1, n - self.H)
            ]
            assert new[i] == lo + self.H - 1 + int(np.argmin(every))


class TestRedundantFixedRegressors:
    """Fixed regressors without variation inside a regime."""

    @staticmethod
    def _data(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        nobs = 200
        y = np.concatenate([rng.normal(0, 1, 100), rng.normal(2, 1, 100)])
        impulse = np.zeros((nobs, 1))
        impulse[150] = 1.0
        return y, np.ones((nobs, 1)), impulse

    def test_independent_columns(self, rng: np.random.Generator) -> None:
        nobs = 60
        z = np.column_stack([np.ones(nobs), rng.standard_normal(nobs)])
        x = np.column_stack(
            [
                np.zeros(nobs),  # no variation
                2e9 * z[:, 0],  # multiple of a breaking regressor
                1e-9 * rng.standard_normal(nobs),  # tiny but informative
                z @ np.array([3.0, -1.0]),  # linear combination
                rng.standard_normal(nobs),
            ]
        )
        assert bp.independent_columns(z, x) == [2, 4]
        assert bp.independent_fixed(z, x[:, :2]) is None

    def test_impulse_is_left_out_where_it_is_zero(
        self, rng: np.random.Generator
    ) -> None:
        y, z, x = self._data(rng)
        opt = bp.BPOptions()
        # first regime: the impulse (at 150) is zero, the test is the pure one
        with_x = bp.one_break_sup_f(y, z, x, 0, 99, 30, opt)
        without = bp.one_break_sup_f(y, z, None, 0, 99, 30, opt)
        assert with_x == pytest.approx(without)
        # second regime: the impulse enters
        assert bp.one_break_sup_f(y, z, x, 100, 199, 30, opt)[0] != pytest.approx(
            bp.one_break_sup_f(y, z, None, 100, 199, 30, opt)[0]
        )

    def test_fit_with_impulse_and_step_dummies(self, rng: np.random.Generator) -> None:
        y, _, impulse = self._data(rng)
        res = BaiPerronTest(y, exog=impulse).fit()
        assert res.break_indices == [100]
        assert res.sequential_n_breaks == 1
        assert all(np.isfinite(v) for v in res.seqf_stats.values())
        # a step dummy as a fixed regressor: models with a break at its date
        # are not identified and are not used
        step = (np.arange(200) >= 140).astype(float)
        with pytest.warns(UserWarning):
            res = BaiPerronTest(y, exog=step).fit()
        assert res.break_indices == [100]
        for dates in res.breaks_by_m.values():
            assert 140 not in dates

    def test_search_skips_dates_with_a_singular_design(
        self, rng: np.random.Generator
    ) -> None:
        nobs = 120
        t = np.arange(nobs)
        z = np.column_stack([np.ones(nobs), rng.standard_normal(nobs) * (t >= 30)])
        x = rng.standard_normal((nobs, 1))
        y = 1.0 * (t >= 60) + 0.5 * x[:, 0] + rng.standard_normal(nobs)
        b, ssr = bp.one_break_date(y, z, x, 18)
        assert b >= 30  # the second regressor is zero before observation 30
        assert np.isfinite(ssr)
        stat, _ = bp.one_break_sup_f(y, z, x, 0, nobs - 1, 18, bp.BPOptions())
        assert np.isfinite(stat)


class TestScaling:
    """Results do not depend on the units or the level of the regressors."""

    def test_regressor_with_a_large_mean(self, rng: np.random.Generator) -> None:
        nobs, h = 120, 18
        t = np.arange(nobs)
        v = rng.standard_normal(nobs)
        y = 1.0 + 0.8 * v + 2.0 * (t >= 40) - 1.5 * (t >= 80)
        y = y + rng.standard_normal(nobs)
        ref = bp.partition_all(
            bp.ssr_triangle(y, np.column_stack([np.ones(nobs), v]), h), 3, h
        )
        for level in (1e4, 1e6):
            z = np.column_stack([np.ones(nobs), level + v])
            s = bp.ssr_triangle(y, z, h)
            assert not np.isnan(s[0, h - 1])
            got = bp.partition_all(s, 3, h)
            for m in range(3):
                assert got[0][m].tolist() == ref[0][m].tolist()
            np.testing.assert_allclose(got[1], ref[1], rtol=1e-6)

    def test_units(self, rng: np.random.Generator) -> None:
        nobs = 150
        t = np.arange(nobs)
        gdp = np.exp(0.005 * t) * (1 + 0.01 * rng.standard_normal(nobs))
        rate = 0.03 + 0.01 * rng.standard_normal(nobs)
        y = 1.0 + 2.0 * (t >= 100) + 2.0 * rate + rng.standard_normal(nobs) * 0.5
        base = BaiPerronTest(y, exog=np.column_stack([rate, gdp])).fit(max_breaks=3)
        big = BaiPerronTest(y, exog=np.column_stack([rate, 2e13 * gdp])).fit(
            max_breaks=3
        )
        assert base.p == big.p == 2
        assert base.breaks_by_m == big.breaks_by_m
        np.testing.assert_allclose(
            list(big.ssr.values()), list(base.ssr.values()), rtol=1e-6
        )
        np.testing.assert_allclose(
            list(big.supf_stats.values()), list(base.supf_stats.values()), rtol=1e-5
        )
        # the same with the badly scaled regressor among the breaking ones
        zb = BaiPerronTest(y, exog_break=np.column_stack([np.ones(nobs), gdp])).fit(
            max_breaks=2
        )
        zb_big = BaiPerronTest(
            y, exog_break=np.column_stack([np.ones(nobs), 2e13 * gdp])
        ).fit(max_breaks=2)
        assert zb.breaks_by_m == zb_big.breaks_by_m
        np.testing.assert_allclose(
            list(zb_big.ssr.values()), list(zb.ssr.values()), rtol=1e-6
        )


class TestRobustBreakCI:
    """Robust intervals: phi_2 is scaled by the moment matrix of regime i + 1."""

    def test_agrees_with_the_non_robust_interval_for_iid_errors(self) -> None:
        rng = np.random.default_rng(123)
        nobs = 4000
        t = np.arange(nobs)
        # the regressor has a larger variance in the second regime (eta != 1)
        v = rng.standard_normal(nobs) * np.where(t < 2000, 1.0, 1.7)
        z = np.column_stack([np.ones(nobs), v])
        y = 1.0 + 0.5 * v + 0.1 * v * (t >= 2000) + rng.standard_normal(nobs)
        vb = [1999]
        plain = bp.break_ci(y, z, None, vb, bp.BPOptions())[0]
        robust = bp.break_ci(y, z, None, vb, bp.BPOptions(robust=True))[0]
        width = plain[1] - plain[0]
        assert width > 20
        assert abs(robust[0] - plain[0]) < 0.25 * width
        assert abs(robust[1] - plain[1]) < 0.25 * width

    def test_quantile_outside_the_starting_interval(self) -> None:
        # a wide, very asymmetric distribution: the 97.5% quantile is above 2000
        eta, phi1, phi2 = 0.021, 1.0, 4.8
        q = bp.quantiles_argmax(eta, phi1, phi2)
        assert q[3] > 2000
        a = phi1 / phi2
        b = math.sqrt(phi1 * eta / phi2)
        pars = (
            (1 + 2 * a) / 2,
            a * (1 + a) / 2,
            b,
            math.sqrt(phi2 * eta / phi1) + b / 2,
            ((phi2 / phi1) + 1) * eta / 2,
        )
        assert bp.cdf_argmax(q[3], *pars) == pytest.approx(0.975, abs=1e-5)
        assert bp.cdf_argmax(q[0], *pars) == pytest.approx(0.025, abs=1e-5)
