"""Tests for indicator generation functions."""

from __future__ import annotations

import numpy as np
import pytest

from regimes.gets.indicators import (
    _resolve_taus,
    impulse_indicators,
    multiplicative_indicators,
    step_indicators,
    trend_indicators,
)

# ---------------------------------------------------------------------------
# _resolve_taus
# ---------------------------------------------------------------------------


class TestResolveTaus:
    def test_explicit_taus(self):
        result = _resolve_taus(100, [10, 50, 90], trim=0.0)
        assert result == [10, 50, 90]

    def test_explicit_taus_sorted(self):
        result = _resolve_taus(100, [90, 10, 50], trim=0.0)
        assert result == [10, 50, 90]

    def test_explicit_taus_deduplicated(self):
        result = _resolve_taus(100, [50, 50, 50], trim=0.0)
        assert result == [50]

    def test_full_set_no_trim(self):
        result = _resolve_taus(10, None, trim=0.0)
        assert result == list(range(1, 11))

    def test_full_set_with_trim(self):
        result = _resolve_taus(100, None, trim=0.10)
        assert result[0] == 10
        assert result[-1] == 90

    def test_full_set_trim_05(self):
        result = _resolve_taus(200, None, trim=0.05)
        assert result[0] == 10
        assert result[-1] == 190

    def test_tau_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            _resolve_taus(100, [100], trim=0.0)

    def test_negative_tau_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            _resolve_taus(100, [-1], trim=0.0)

    def test_trim_too_large_raises(self):
        with pytest.raises(ValueError, match="too large"):
            _resolve_taus(3, None, trim=0.5)  # lo=2, hi=1 => lo>hi


# ---------------------------------------------------------------------------
# step_indicators
# ---------------------------------------------------------------------------


class TestStepIndicators:
    def test_shape(self):
        S, names = step_indicators(100, taus=[25, 50, 75])
        assert S.shape == (100, 3)
        assert len(names) == 3

    def test_names(self):
        _, names = step_indicators(100, taus=[25, 50, 75])
        assert names == ["step_25", "step_50", "step_75"]

    def test_forward_convention(self):
        S, _ = step_indicators(100, taus=[50])
        assert S[49, 0] == 0.0
        assert S[50, 0] == 1.0
        assert S[99, 0] == 1.0
        assert S[0, 0] == 0.0

    def test_all_zeros_before_tau(self):
        S, _ = step_indicators(100, taus=[30])
        assert np.all(S[:30, 0] == 0.0)

    def test_all_ones_from_tau(self):
        S, _ = step_indicators(100, taus=[30])
        assert np.all(S[30:, 0] == 1.0)

    def test_full_set_with_trim(self):
        S, names = step_indicators(100, trim=0.1)
        assert len(names) == 81  # 10..90 inclusive
        assert names[0] == "step_10"
        assert names[-1] == "step_90"

    def test_column_sums(self):
        """Step at tau should sum to n - tau."""
        S, _ = step_indicators(100, taus=[20, 60])
        assert S[:, 0].sum() == 80  # 100 - 20
        assert S[:, 1].sum() == 40  # 100 - 60

    def test_dtype_float64(self):
        S, _ = step_indicators(50, taus=[10])
        assert S.dtype == np.float64

    def test_empty_taus(self):
        S, names = step_indicators(100, taus=[])
        assert S.shape == (100, 0)
        assert names == []


# ---------------------------------------------------------------------------
# impulse_indicators
# ---------------------------------------------------------------------------


class TestImpulseIndicators:
    def test_shape(self):
        ind, names = impulse_indicators(100, taus=[10, 90])
        assert ind.shape == (100, 2)
        assert len(names) == 2

    def test_names(self):
        _, names = impulse_indicators(100, taus=[10, 90])
        assert names == ["impulse_10", "impulse_90"]

    def test_single_one(self):
        ind, _ = impulse_indicators(100, taus=[42])
        assert ind[42, 0] == 1.0
        assert ind.sum() == 1.0

    def test_zeros_elsewhere(self):
        ind, _ = impulse_indicators(100, taus=[50])
        assert ind[49, 0] == 0.0
        assert ind[51, 0] == 0.0

    def test_full_set_identity(self):
        """Full set of impulse indicators is essentially an identity."""
        ind, _ = impulse_indicators(5, taus=[0, 1, 2, 3, 4])
        np.testing.assert_array_equal(ind, np.eye(5))

    def test_column_sums_all_one(self):
        ind, _ = impulse_indicators(100, taus=[10, 20, 30])
        for j in range(3):
            assert ind[:, j].sum() == 1.0


# ---------------------------------------------------------------------------
# multiplicative_indicators
# ---------------------------------------------------------------------------


class TestMultiplicativeIndicators:
    @pytest.fixture
    def exog(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal((200, 2))

    def test_shape_all_vars(self, exog):
        M, names = multiplicative_indicators(exog, 200, taus=[50, 100, 150])
        # 2 vars * 3 taus = 6 columns
        assert M.shape == (200, 6)
        assert len(names) == 6

    def test_names_default(self, exog):
        _, names = multiplicative_indicators(exog, 200, taus=[50, 100])
        assert names == [
            "x0*step_50",
            "x0*step_100",
            "x1*step_50",
            "x1*step_100",
        ]

    def test_names_custom(self, exog):
        _, names = multiplicative_indicators(
            exog, 200, taus=[50], exog_names=["y.L1", "x0"]
        )
        assert names == ["y.L1*step_50", "x0*step_50"]

    def test_values_step_times_x(self, exog):
        M, _ = multiplicative_indicators(exog, 200, taus=[100])
        # Before tau: zeros
        np.testing.assert_array_equal(M[:100, 0], 0.0)
        # From tau: exog values
        np.testing.assert_array_almost_equal(M[100:, 0], exog[100:, 0])

    def test_single_variable_by_index(self, exog):
        M, names = multiplicative_indicators(exog, 200, variables=[1], taus=[50, 100])
        assert M.shape == (200, 2)
        assert all("x1" in n for n in names)

    def test_single_variable_by_name(self, exog):
        M, names = multiplicative_indicators(
            exog,
            200,
            variables=["y.L1"],
            taus=[50],
            exog_names=["y.L1", "x0"],
        )
        assert M.shape == (200, 1)
        assert names == ["y.L1*step_50"]

    def test_1d_exog(self):
        x = np.ones(100)
        M, names = multiplicative_indicators(x, 100, taus=[50])
        assert M.shape == (100, 1)
        np.testing.assert_array_equal(M[50:, 0], 1.0)
        np.testing.assert_array_equal(M[:50, 0], 0.0)

    def test_mismatched_n_raises(self, exog):
        with pytest.raises(ValueError, match="rows"):
            multiplicative_indicators(exog, 300, taus=[50])

    def test_bad_variable_index_raises(self, exog):
        with pytest.raises(ValueError, match="out of range"):
            multiplicative_indicators(exog, 200, variables=[5], taus=[50])

    def test_bad_variable_name_raises(self, exog):
        with pytest.raises(ValueError, match="not found"):
            multiplicative_indicators(
                exog,
                200,
                variables=["bad"],
                taus=[50],
                exog_names=["x0", "x1"],
            )

    def test_mismatched_names_raises(self, exog):
        with pytest.raises(ValueError, match="entries"):
            multiplicative_indicators(exog, 200, taus=[50], exog_names=["only_one"])

    def test_full_set_with_trim(self, exog):
        M, names = multiplicative_indicators(exog, 200, trim=0.1)
        # trim=0.1 -> taus from 20..180 = 161 taus, 2 vars => 322 cols
        assert M.shape[1] == 2 * 161


# ---------------------------------------------------------------------------
# trend_indicators
# ---------------------------------------------------------------------------


class TestTrendIndicators:
    def test_shape(self):
        T, names = trend_indicators(100, taus=[30, 60])
        assert T.shape == (100, 2)
        assert len(names) == 2

    def test_names(self):
        _, names = trend_indicators(100, taus=[30, 60])
        assert names == ["trend_30", "trend_60"]

    def test_zero_before_tau(self):
        T, _ = trend_indicators(100, taus=[50])
        assert np.all(T[:50, 0] == 0.0)

    def test_zero_at_tau(self):
        T, _ = trend_indicators(100, taus=[50])
        assert T[50, 0] == 0.0

    def test_linear_after_tau(self):
        T, _ = trend_indicators(100, taus=[50])
        assert T[51, 0] == 1.0
        assert T[52, 0] == 2.0
        assert T[53, 0] == 3.0

    def test_max_value(self):
        T, _ = trend_indicators(100, taus=[50])
        assert T[99, 0] == 49.0

    def test_column_sum(self):
        """Sum should be 0+1+2+...+(n-tau-1) = (n-tau-1)(n-tau)/2."""
        T, _ = trend_indicators(100, taus=[80])
        expected = sum(range(20))  # 0..19
        assert T[:, 0].sum() == expected


# ---------------------------------------------------------------------------
# Composability
# ---------------------------------------------------------------------------


class TestComposability:
    def test_stack_sis_and_iis(self):
        S, s_names = step_indicators(100, taus=[50])
        ind, i_names = impulse_indicators(100, taus=[50])
        combined = np.column_stack([S, ind])
        assert combined.shape == (100, 2)
        assert s_names + i_names == ["step_50", "impulse_50"]

    def test_stack_sis_and_mis(self):
        S, s_names = step_indicators(200, taus=[100])
        X = np.random.default_rng(42).standard_normal((200, 1))
        M, m_names = multiplicative_indicators(X, 200, taus=[100])
        combined = np.column_stack([S, M])
        assert combined.shape == (200, 2)
        all_names = s_names + m_names
        assert all_names == ["step_100", "x0*step_100"]
