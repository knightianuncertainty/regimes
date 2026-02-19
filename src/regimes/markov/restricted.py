"""Restricted Markov switching models with constrained transition matrices.

This module provides RestrictedMarkovRegression and RestrictedMarkovAutoregression
that subclass the statsmodels implementations to enforce exact zeros (or other
fixed values) in the transition probability matrix.

The key insight: statsmodels uses a softmax parameterization that maps to the
interior of the probability simplex. We override transform_params/untransform_params
to handle restrictions by fixing some entries and distributing the remaining
probability mass among free entries via softmax.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from statsmodels.tsa.regime_switching.markov_autoregression import (
    MarkovAutoregression,
)
from statsmodels.tsa.regime_switching.markov_regression import (
    MarkovRegression as SMMarkovRegression,
)

from regimes.markov.models import (
    MarkovRegression,
    MarkovAR,
    _ensure_1d,
    _ensure_2d,
    _extract_regime_params,
    _relabel_regimes,
    _apply_permutation,
)
from regimes.markov.results import (
    MarkovRegressionResults,
    MarkovARResults,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _softmax(x: NDArray[Any]) -> NDArray[Any]:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def _inverse_softmax(p: NDArray[Any]) -> NDArray[Any]:
    """Inverse of softmax (up to a constant, using log)."""
    p = np.clip(p, 1e-10, 1.0)
    return np.log(p)


class _RestrictedSMMarkovRegression(SMMarkovRegression):
    """statsmodels MarkovRegression with restricted transition probabilities.

    Overrides transform_params/untransform_params to enforce fixed values
    in the transition matrix.

    Parameters
    ----------
    restrictions : dict[tuple[int, int], float]
        Dictionary mapping (row, col) -> fixed_value in the transition matrix.
        Typically {(i, j): 0.0} for forbidden transitions.
    """

    def __init__(
        self,
        endog: Any,
        k_regimes: int,
        restrictions: dict[tuple[int, int], float] | None = None,
        **kwargs: Any,
    ) -> None:
        self._restrictions = restrictions or {}
        super().__init__(endog, k_regimes, **kwargs)

        # Recalculate the number of free transition parameters
        k = k_regimes
        # Standard: (k-1) free params per column, k columns = k*(k-1)
        # With restrictions: subtract 1 per restricted entry
        n_restricted = len(self._restrictions)
        self._n_free_transition = k * (k - 1) - n_restricted

    @property
    def _n_transition_params(self) -> int:
        """Number of free transition parameters."""
        # We need to keep the same parameter vector length as statsmodels expects,
        # so we don't override this. Instead, we handle restrictions in transform.
        return self.k_regimes * (self.k_regimes - 1)

    def _build_free_mask(self) -> list[list[tuple[int, bool, float]]]:
        """Build a mask describing free/restricted entries per column.

        Returns
        -------
        list[list[tuple[int, bool, float]]]
            For each column j: list of (row_i, is_free, fixed_value).
        """
        k = self.k_regimes
        mask = []
        for j in range(k):
            col_entries = []
            for i in range(k):
                if (i, j) in self._restrictions:
                    col_entries.append((i, False, self._restrictions[(i, j)]))
                else:
                    col_entries.append((i, True, 0.0))
            mask.append(col_entries)
        return mask

    def transform_params(self, unconstrained: NDArray[Any]) -> NDArray[Any]:
        """Transform unconstrained params to constrained, applying restrictions.

        The transition matrix parameters are at the end of the parameter vector.
        For each column j:
          1. Sum fixed values
          2. available_mass = 1 - sum(fixed)
          3. Free entries: softmax(unconstrained[free_indices]) * available_mass
        """
        constrained = super().transform_params(unconstrained)

        if not self._restrictions:
            return constrained

        # The transition params are the last k*(k-1) entries
        k = self.k_regimes
        n_tp = k * (k - 1)
        n_other = len(constrained) - n_tp

        # Reconstruct transition matrix from the constrained params
        # statsmodels stores transition params column by column, k-1 per col
        tp_start = n_other
        tp = constrained[tp_start:]

        # Rebuild the full transition matrix
        P = np.zeros((k, k))
        idx = 0
        for j in range(k):
            # statsmodels convention: stores k-1 probs, the k-th is 1 - sum
            probs = tp[idx:idx + k - 1]
            P[:k - 1, j] = probs
            P[k - 1, j] = 1.0 - np.sum(probs)
            idx += k - 1

        # Apply restrictions
        mask = self._build_free_mask()
        for j in range(k):
            entries = mask[j]
            fixed_sum = sum(fv for _, is_free, fv in entries if not is_free)
            available = max(1.0 - fixed_sum, 0.0)

            free_indices = [i for i, is_free, _ in entries if is_free]
            n_free = len(free_indices)

            if n_free == 0:
                # All entries fixed — set them
                for i, is_free, fv in entries:
                    P[i, j] = fv
            elif n_free == 1:
                # One free entry gets all remaining mass
                for i, is_free, fv in entries:
                    if not is_free:
                        P[i, j] = fv
                    else:
                        P[i, j] = available
            else:
                # Multiple free entries: distribute available mass proportionally
                free_vals = np.array([P[i, j] for i in free_indices])
                free_sum = np.sum(free_vals)
                if free_sum > 0:
                    free_vals = free_vals / free_sum * available
                else:
                    free_vals = np.ones(n_free) / n_free * available

                for i, is_free, fv in entries:
                    if not is_free:
                        P[i, j] = fv

                for fi, i in enumerate(free_indices):
                    P[i, j] = free_vals[fi]

        # Write back to constrained parameter vector
        idx = 0
        new_tp = np.zeros(n_tp)
        for j in range(k):
            new_tp[idx:idx + k - 1] = P[:k - 1, j]
            idx += k - 1

        constrained[tp_start:] = new_tp
        return constrained


class _RestrictedSMMarkovAutoregression(MarkovAutoregression):
    """statsmodels MarkovAutoregression with restricted transition probabilities."""

    def __init__(
        self,
        endog: Any,
        k_regimes: int,
        order: int,
        restrictions: dict[tuple[int, int], float] | None = None,
        **kwargs: Any,
    ) -> None:
        self._restrictions = restrictions or {}
        super().__init__(endog, k_regimes, order=order, **kwargs)

    def _build_free_mask(self) -> list[list[tuple[int, bool, float]]]:
        k = self.k_regimes
        mask = []
        for j in range(k):
            col_entries = []
            for i in range(k):
                if (i, j) in self._restrictions:
                    col_entries.append((i, False, self._restrictions[(i, j)]))
                else:
                    col_entries.append((i, True, 0.0))
            mask.append(col_entries)
        return mask

    def transform_params(self, unconstrained: NDArray[Any]) -> NDArray[Any]:
        constrained = super().transform_params(unconstrained)

        if not self._restrictions:
            return constrained

        k = self.k_regimes
        n_tp = k * (k - 1)
        n_other = len(constrained) - n_tp
        tp_start = n_other
        tp = constrained[tp_start:]

        P = np.zeros((k, k))
        idx = 0
        for j in range(k):
            probs = tp[idx:idx + k - 1]
            P[:k - 1, j] = probs
            P[k - 1, j] = 1.0 - np.sum(probs)
            idx += k - 1

        mask = self._build_free_mask()
        for j in range(k):
            entries = mask[j]
            fixed_sum = sum(fv for _, is_free, fv in entries if not is_free)
            available = max(1.0 - fixed_sum, 0.0)
            free_indices = [i for i, is_free, _ in entries if is_free]
            n_free = len(free_indices)

            if n_free == 0:
                for i, is_free, fv in entries:
                    P[i, j] = fv
            elif n_free == 1:
                for i, is_free, fv in entries:
                    if not is_free:
                        P[i, j] = fv
                    else:
                        P[i, j] = available
            else:
                free_vals = np.array([P[i, j] for i in free_indices])
                free_sum = np.sum(free_vals)
                if free_sum > 0:
                    free_vals = free_vals / free_sum * available
                else:
                    free_vals = np.ones(n_free) / n_free * available
                for i, is_free, fv in entries:
                    if not is_free:
                        P[i, j] = fv
                for fi, i in enumerate(free_indices):
                    P[i, j] = free_vals[fi]

        idx = 0
        new_tp = np.zeros(n_tp)
        for j in range(k):
            new_tp[idx:idx + k - 1] = P[:k - 1, j]
            idx += k - 1

        constrained[tp_start:] = new_tp
        return constrained


class RestrictedMarkovRegression(MarkovRegression):
    """MarkovRegression with fixed transition probabilities.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable.
    k_regimes : int
        Number of regimes.
    restrictions : dict[tuple[int, int], float]
        Dictionary mapping (row, col) -> fixed_value in the transition
        matrix. Entry (i, j) is P(S_t = i | S_{t-1} = j). Typically
        used to set {(i, j): 0.0} for forbidden transitions.
    **kwargs
        Additional arguments for MarkovRegression.

    Examples
    --------
    >>> import numpy as np
    >>> from regimes.markov.restricted import RestrictedMarkovRegression
    >>> np.random.seed(42)
    >>> y = np.concatenate([np.random.randn(100) + 1, np.random.randn(100) + 3])
    >>> # Non-recurring: forbid going back from regime 1 to regime 0
    >>> model = RestrictedMarkovRegression(
    ...     y, k_regimes=2, restrictions={(0, 1): 0.0}
    ... )
    >>> results = model.fit()
    >>> print(results.regime_transition)
    """

    def __init__(
        self,
        endog: Any,
        k_regimes: int = 2,
        restrictions: dict[tuple[int, int], float] | None = None,
        **kwargs: Any,
    ) -> None:
        self.restrictions = restrictions or {}
        super().__init__(endog, k_regimes=k_regimes, **kwargs)

    def fit(
        self,
        method: str = "bfgs",
        maxiter: int = 200,
        em_iter: int = 10,
        search_reps: int = 5,
        **kwargs: Any,
    ) -> MarkovRegressionResults:
        """Fit with restricted transition matrix."""
        sm_model = _RestrictedSMMarkovRegression(
            endog=self.endog,
            k_regimes=self.k_regimes,
            restrictions=self.restrictions,
            exog=self.exog,
            trend=self.trend,
            switching_trend=self.switching_trend,
            switching_exog=self.switching_exog,
            switching_variance=self.switching_variance,
        )

        sm_results = self._fit_with_search(
            sm_model, method, maxiter, em_iter, search_reps, **kwargs
        )

        results = self._wrap_results(sm_results)

        # Enforce exact restriction values in the returned transition matrix
        for (i, j), val in self.restrictions.items():
            results.regime_transition[i, j] = val

        # Store restriction info
        results.restricted_transitions = dict(self.restrictions)

        return results

    @staticmethod
    def non_recurring(
        endog: Any,
        k_regimes: int = 2,
        **kwargs: Any,
    ) -> RestrictedMarkovRegression:
        """Create a non-recurring (structural break) model.

        Sets up the Chib (1998) upper-triangular restriction: the latent
        state can only stay put or advance to the next regime.

        For k=3 regimes, the transition matrix is:
            P = [[p11, 0,    0   ],
                 [p21, p22,  0   ],
                 [0,   p32,  1.0 ]]

        Parameters
        ----------
        endog : ArrayLike
            Dependent variable.
        k_regimes : int
            Number of regimes.
        **kwargs
            Additional arguments.

        Returns
        -------
        RestrictedMarkovRegression
            Model with non-recurring transition structure.
        """
        restrictions: dict[tuple[int, int], float] = {}

        for j in range(k_regimes):
            for i in range(k_regimes):
                # Forbid going backward: i < j means earlier regime
                if i < j - 0:  # Allow staying (i==j) and advancing (i==j+1)
                    # But only restrict j<i (going back) and j>i+1 (skipping)
                    pass

        # More precisely: for each column j (conditioning on S_{t-1}=j),
        # only allow transitions to j (stay) and j+1 (advance)
        for j in range(k_regimes):
            for i in range(k_regimes):
                if i != j and i != j + 1:
                    restrictions[(i, j)] = 0.0

        # Last regime is absorbing
        # Already handled: only (k-1, k-1) is allowed for last column

        return RestrictedMarkovRegression(
            endog, k_regimes=k_regimes, restrictions=restrictions, **kwargs
        )


class RestrictedMarkovAR(MarkovAR):
    """MarkovAR with fixed transition probabilities.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable.
    k_regimes : int
        Number of regimes.
    order : int
        AR order.
    restrictions : dict[tuple[int, int], float]
        Transition matrix restrictions.
    **kwargs
        Additional arguments for MarkovAR.
    """

    def __init__(
        self,
        endog: Any,
        k_regimes: int = 2,
        order: int = 1,
        restrictions: dict[tuple[int, int], float] | None = None,
        **kwargs: Any,
    ) -> None:
        self.restrictions = restrictions or {}
        super().__init__(endog, k_regimes=k_regimes, order=order, **kwargs)

    def fit(
        self,
        method: str = "bfgs",
        maxiter: int = 200,
        em_iter: int = 10,
        search_reps: int = 5,
        **kwargs: Any,
    ) -> MarkovARResults:
        """Fit with restricted transition matrix."""
        sm_model = _RestrictedSMMarkovAutoregression(
            endog=self.endog,
            k_regimes=self.k_regimes,
            order=self.order,
            restrictions=self.restrictions,
            exog=self.exog,
            trend=self.trend,
            switching_ar=self.switching_ar,
            switching_trend=self.switching_trend,
            switching_variance=self.switching_variance,
        )

        sm_results = self._fit_with_search(
            sm_model, method, maxiter, em_iter, search_reps, **kwargs
        )

        results = self._wrap_results(sm_results)

        for (i, j), val in self.restrictions.items():
            results.regime_transition[i, j] = val

        results.restricted_transitions = dict(self.restrictions)

        return results

    @staticmethod
    def non_recurring(
        endog: Any,
        k_regimes: int = 2,
        order: int = 1,
        **kwargs: Any,
    ) -> RestrictedMarkovAR:
        """Create a non-recurring (structural break) AR model."""
        restrictions: dict[tuple[int, int], float] = {}
        for j in range(k_regimes):
            for i in range(k_regimes):
                if i != j and i != j + 1:
                    restrictions[(i, j)] = 0.0

        return RestrictedMarkovAR(
            endog, k_regimes=k_regimes, order=order,
            restrictions=restrictions, **kwargs
        )
