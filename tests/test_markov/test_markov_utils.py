"""Tests for Markov switching utility functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from regimes.markov.models import (
    _apply_permutation,
    _ensure_1d,
    _ensure_2d,
    _relabel_regimes,
)
from regimes.markov.restricted import _inverse_softmax, _softmax


class TestSoftmax:
    """Test _softmax utility."""

    def test_output_sums_to_one(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        p = _softmax(x)
        np.testing.assert_allclose(p.sum(), 1.0)

    def test_all_zeros(self) -> None:
        x = np.zeros(3)
        p = _softmax(x)
        np.testing.assert_allclose(p, np.ones(3) / 3)

    def test_numerical_stability_large(self) -> None:
        x = np.array([1000.0, 1001.0, 1002.0])
        p = _softmax(x)
        np.testing.assert_allclose(p.sum(), 1.0)
        assert np.all(np.isfinite(p))
        assert np.all(p > 0)

    def test_ordering_preserved(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        p = _softmax(x)
        assert p[0] < p[1] < p[2]


class TestInverseSoftmax:
    """Test _inverse_softmax utility."""

    def test_basic(self) -> None:
        p = np.array([0.2, 0.3, 0.5])
        log_p = _inverse_softmax(p)
        assert np.all(np.isfinite(log_p))

    def test_clipping_near_zero(self) -> None:
        p = np.array([0.0, 0.5, 0.5])
        log_p = _inverse_softmax(p)
        assert np.all(np.isfinite(log_p))
        # The zero entry should be clipped to 1e-10
        assert log_p[0] == np.log(1e-10)

    def test_roundtrip(self) -> None:
        p = np.array([0.2, 0.3, 0.5])
        log_p = _inverse_softmax(p)
        p_recovered = _softmax(log_p)
        np.testing.assert_allclose(p_recovered, p, atol=1e-6)


class TestEnsure1d:
    """Test _ensure_1d utility."""

    def test_1d_passthrough(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        result = _ensure_1d(arr)
        assert result.ndim == 1
        np.testing.assert_array_equal(result, arr)

    def test_2d_squeeze(self) -> None:
        arr = np.array([[1.0], [2.0], [3.0]])
        result = _ensure_1d(arr)
        assert result.ndim == 1
        assert len(result) == 3

    def test_pandas_series(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0])
        result = _ensure_1d(s)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.dtype == np.float64

    def test_list_input(self) -> None:
        result = _ensure_1d([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64


class TestEnsure2d:
    """Test _ensure_2d utility."""

    def test_none_returns_none(self) -> None:
        assert _ensure_2d(None) is None

    def test_1d_reshapes(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        result = _ensure_2d(arr)
        assert result is not None
        assert result.shape == (3, 1)

    def test_2d_passthrough(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _ensure_2d(arr)
        assert result is not None
        assert result.shape == (2, 2)

    def test_pandas_dataframe(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = _ensure_2d(df)
        assert result is not None
        assert result.shape == (2, 2)
        assert result.dtype == np.float64


class TestRelabelRegimes:
    """Test _relabel_regimes utility."""

    def test_none_ordering_returns_identity(self) -> None:
        mock_results = MagicMock()
        perm = _relabel_regimes(mock_results, ordering=None, k_regimes=3)
        np.testing.assert_array_equal(perm, np.array([0, 1, 2]))

    def test_unknown_ordering_returns_identity(self) -> None:
        mock_results = MagicMock()
        perm = _relabel_regimes(mock_results, ordering="unknown_string", k_regimes=2)
        np.testing.assert_array_equal(perm, np.array([0, 1]))

    def test_first_appearance_ordering(self) -> None:
        mock_results = MagicMock()
        # Regime 1 appears first, then regime 0
        probs = np.zeros((10, 2))
        probs[:5, 1] = 1.0  # regime 1 dominant first
        probs[5:, 0] = 1.0  # regime 0 dominant second
        mock_results.smoothed_marginal_probabilities = probs

        perm = _relabel_regimes(mock_results, ordering="first_appearance", k_regimes=2)
        # Old regime 1 appeared first → should become regime 0
        assert perm[1] == 0
        assert perm[0] == 1

    def test_intercept_ordering(self) -> None:
        mock_results = MagicMock()
        # params[0] = 5.0 (high), params[1] = 1.0 (low)
        mock_results.params = np.array(
            [5.0, 1.0, 0.8, 0.5]
        )  # intercepts then other params

        perm = _relabel_regimes(mock_results, ordering="intercept", k_regimes=2)
        # Regime 0 has param 5.0, regime 1 has param 1.0
        # After sorting: regime 1 (1.0) should become regime 0, regime 0 (5.0) should become regime 1
        assert perm[1] == 0  # old regime 1 (low mean) → new regime 0
        assert perm[0] == 1  # old regime 0 (high mean) → new regime 1

    def test_intercept_ordering_exception_fallback(self) -> None:
        mock_results = MagicMock()
        mock_params = MagicMock()
        mock_params.__getitem__ = MagicMock(side_effect=IndexError("bad index"))
        mock_results.params = mock_params

        perm = _relabel_regimes(mock_results, ordering="intercept", k_regimes=2)
        np.testing.assert_array_equal(perm, np.array([0, 1]))

    def test_first_appearance_unseen_regimes_filled(self) -> None:
        mock_results = MagicMock()
        # Only regime 0 appears (regime 1 never dominant)
        probs = np.zeros((10, 2))
        probs[:, 0] = 1.0  # regime 0 always dominant
        mock_results.smoothed_marginal_probabilities = probs

        perm = _relabel_regimes(mock_results, ordering="first_appearance", k_regimes=2)
        # Regime 0 seen first, regime 1 filled at end
        assert perm[0] == 0
        assert perm[1] == 1


class TestApplyPermutation:
    """Test _apply_permutation utility."""

    def test_identity_returns_same(self) -> None:
        k = 2
        perm = np.array([0, 1], dtype=np.intp)
        transition = np.array([[0.9, 0.3], [0.1, 0.7]])
        smoothed = np.random.rand(10, 2)
        filtered = np.random.rand(10, 2)
        predicted = np.random.rand(10, 2)
        regime_params = {0: {"const": 1.0}, 1: {"const": 3.0}}

        t, s, f, p, rp = _apply_permutation(
            perm, k, transition, smoothed, filtered, predicted, regime_params
        )
        np.testing.assert_array_equal(t, transition)
        np.testing.assert_array_equal(s, smoothed)
        np.testing.assert_array_equal(f, filtered)
        np.testing.assert_array_equal(p, predicted)
        assert rp == regime_params

    def test_swap_permutation(self) -> None:
        k = 2
        perm = np.array([1, 0], dtype=np.intp)  # swap regimes
        transition = np.array([[0.9, 0.3], [0.1, 0.7]])
        smoothed = np.array([[0.8, 0.2], [0.7, 0.3]])
        filtered = np.array([[0.85, 0.15], [0.75, 0.25]])
        predicted = np.array([[0.82, 0.18], [0.72, 0.28]])
        regime_params = {0: {"const": 1.0}, 1: {"const": 3.0}}

        t, s, f, p, rp = _apply_permutation(
            perm, k, transition, smoothed, filtered, predicted, regime_params
        )

        # Check transition matrix is permuted: new[perm[i], perm[j]] = old[i,j]
        assert t[1, 1] == transition[0, 0]  # new[1,1] = old[0,0]
        assert t[0, 0] == transition[1, 1]  # new[0,0] = old[1,1]

        # Check probability columns are swapped
        np.testing.assert_array_equal(s[:, 0], smoothed[:, 1])
        np.testing.assert_array_equal(s[:, 1], smoothed[:, 0])

        # Check regime_params keys are swapped
        assert rp[1]["const"] == 1.0  # old regime 0 → new regime 1
        assert rp[0]["const"] == 3.0  # old regime 1 → new regime 0
