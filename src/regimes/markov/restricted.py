"""Restricted Markov switching models with constrained transition matrices.

This module provides RestrictedMarkovRegression and RestrictedMarkovAutoregression
that subclass the statsmodels implementations to enforce exact zeros (or other
fixed values) in the transition probability matrix.

statsmodels maps the unconstrained parameters to the interior of the
probability simplex. ``transform_params`` is overridden to set the restricted
entries of the transition matrix and to rescale the free entries of each
column so that it sums to one.

statsmodels stores the transition probabilities as ``p[j->i]`` for the
destinations ``i = 0, ..., k-2``, destination by destination: reshaped to
``(k-1, k)``, entry ``[i, j]`` is the probability of moving from ``j`` to ``i``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from statsmodels.tools.numdiff import approx_hess_cs
from statsmodels.tsa.regime_switching.markov_regression import (
    MarkovRegression as SMMarkovRegression,
)
from statsmodels.tsa.regime_switching.markov_switching import (
    prefix_hamilton_filter_log_map,
    prefix_kim_smoother_log_map,
)
from statsmodels.tsa.statespace.tools import find_best_blas_type

from regimes.markov.models import (
    MarkovAR,
    MarkovRegression,
    _SMMarkovAutoregression,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from regimes.markov.results import (
        MarkovARResults,
        MarkovRegressionResults,
    )

# Effectively zero in log-space.  statsmodels uses log(max(p, 1e-20)) ≈ -46
# which leaks ~1e-20 probability per step.  Over ~100 steps with strong data
# (likelihood ratio ~10^4/step) the leakage overwhelms the restriction.
# Using -1000 instead eliminates the leakage while avoiding NaN issues that
# -inf would cause in the smoother's backward pass ((-inf) - (-inf) = NaN).
_LOG_ZERO = -1000.0


def _softmax(x: NDArray[Any]) -> NDArray[Any]:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def _inverse_softmax(p: NDArray[Any]) -> NDArray[Any]:
    """Inverse of softmax (up to a constant, using log)."""
    p = np.clip(p, 1e-10, 1.0)
    return np.log(p)


class _RestrictedFilterMixin:
    """Mixin providing filter/smooth overrides that enforce exact zeros in log-space.

    statsmodels' ``cy_hamilton_filter_log`` and ``cy_kim_smoother_log`` convert
    zero transition probabilities to ``log(max(0, 1e-20)) ≈ -46``.  This tiny
    leakage accumulates exponentially when the data strongly favour the
    "forbidden" regime, eventually overwhelming the restriction.  This mixin
    replaces that conversion with ``_LOG_ZERO = -1000`` for any entry that is
    exactly zero in the transition matrix.
    """

    _restrictions: dict[tuple[int, int], float]

    def _transition_index(self) -> NDArray[np.intp]:
        """Positions of the transition probabilities in the parameter vector.

        Returns an array of shape (k-1, k): entry ``[i, j]`` is the position
        of ``p[j->i]``, the probability of moving from regime j to regime i.
        """
        k = self.k_regimes  # type: ignore[attr-defined]
        index = np.arange(self.k_params)[  # type: ignore[attr-defined]
            self.parameters["regime_transition"]  # type: ignore[attr-defined]
        ]
        return np.asarray(index, dtype=np.intp).reshape(k - 1, k)

    def _impose_restrictions(self, matrix: NDArray[Any]) -> NDArray[Any]:
        """Set the restricted entries of a transition matrix.

        ``matrix[i, j]`` is the probability of moving from j to i. In each
        column the restricted entries are set to their values and the free
        entries are rescaled to share the remaining probability.
        """
        k = self.k_regimes  # type: ignore[attr-defined]
        out = np.array(matrix, copy=True)
        for j in range(k):
            fixed = {i: v for (i, jj), v in self._restrictions.items() if jj == j}
            free = [i for i in range(k) if i not in fixed]
            for i, value in fixed.items():
                out[i, j] = value
            if not free:
                continue
            available = max(1.0 - float(sum(fixed.values())), 0.0)
            total = out[free, j].sum()
            if np.real(total) > 0:
                out[free, j] = out[free, j] / total * available
            else:
                out[free, j] = available / len(free)
        return out

    def transform_params(self, unconstrained: NDArray[Any]) -> NDArray[Any]:
        """Transform unconstrained parameters and impose the restrictions."""
        constrained: NDArray[Any] = np.asarray(
            super().transform_params(unconstrained)  # type: ignore[misc]
        )
        if not self._restrictions:
            return constrained

        k = self.k_regimes  # type: ignore[attr-defined]
        index = self._transition_index()
        matrix = np.zeros((k, k), dtype=constrained.dtype)
        matrix[: k - 1, :] = constrained[index]
        matrix[k - 1, :] = 1.0 - matrix[: k - 1, :].sum(axis=0)
        matrix = self._impose_restrictions(matrix)
        constrained[index] = matrix[: k - 1, :]
        return constrained

    def untransform_params(self, constrained: NDArray[Any]) -> NDArray[Any]:
        """Inverse of ``transform_params`` for the free entries.

        statsmodels' version takes logs of the probabilities and cannot
        handle the exact zeros and ones of a restricted matrix. The
        unconstrained value of a restricted entry is irrelevant, because
        ``transform_params`` overwrites it; it is set from a small positive
        probability.
        """
        if not self._restrictions:
            return np.asarray(super().untransform_params(constrained))  # type: ignore[misc]

        k = self.k_regimes  # type: ignore[attr-defined]
        index = self._transition_index()
        values = np.array(constrained, dtype=np.float64, copy=True)
        matrix = np.zeros((k, k))
        matrix[: k - 1, :] = values[index]
        matrix[k - 1, :] = 1.0 - matrix[: k - 1, :].sum(axis=0)
        matrix = np.clip(matrix, 1e-8, None)
        matrix /= matrix.sum(axis=0)

        # The other parameters (variances, AR coefficients) are untransformed
        # by statsmodels; give it interior probabilities to work with
        values[index] = 1.0 / k
        unconstrained: NDArray[Any] = np.asarray(
            super().untransform_params(values)  # type: ignore[misc]
        )
        # statsmodels: p_i = exp(x_i) / (1 + sum_l exp(x_l)), last regime as base
        unconstrained[index] = np.log(matrix[: k - 1, :]) - np.log(matrix[k - 1, :])
        return unconstrained

    def regime_transition_matrix(
        self, params: NDArray[Any], exog_tvtp: Any = None
    ) -> NDArray[Any]:
        """Transition matrix with the restrictions imposed.

        For parameters that satisfy the restrictions this only removes
        rounding errors: the last row is one minus the sum of the others,
        which can leave 1e-16 where a restriction says zero, and the filter
        treats only an exact zero as a forbidden transition. Parameters that
        do not satisfy them (statsmodels' starting values and its EM steps)
        are mapped to a matrix that does.
        """
        matrix: NDArray[Any] = np.asarray(
            super().regime_transition_matrix(params, exog_tvtp)  # type: ignore[misc]
        )
        if self._restrictions:
            matrix[..., 0] = self._impose_restrictions(matrix[..., 0])
        return matrix

    def _em_regime_transition(self, result: Any) -> NDArray[Any]:
        """EM update of the transition probabilities, under the restrictions.

        The probability of moving from j to i is the expected number of such
        moves divided by the expected number of periods the chain moves on
        from j. statsmodels divides by the expected time spent in j over the
        sample instead, which leaves out the starting period and includes
        the last one. The difference is small for a regime that is visited
        often, but when a regime is occupied only briefly at the start, as
        in the first iterations of a model whose chain starts in a regime it
        must leave, statsmodels' ratio reaches one and the model can then
        never leave its first regime.

        Returns an array of shape (k, k-1): row j holds the probabilities of
        moving from j to the regimes 0, ..., k-2.
        """
        k = self.k_regimes  # type: ignore[attr-defined]
        joint = np.asarray(result.smoothed_joint_probabilities)
        for _ in range(joint.ndim - 3):
            joint = joint.sum(axis=-2)
        moves = joint.sum(axis=-1)  # moves[i, j]: expected moves from j to i
        totals = moves.sum(axis=0)
        matrix = np.full((k, k), 1.0 / k)
        left = totals > 0
        matrix[:, left] = moves[:, left] / totals[left]
        matrix = self._impose_restrictions(matrix)
        return np.asarray(matrix[: k - 1, :].T)

    def _free_parameter_jacobian(self) -> NDArray[np.floating[Any]]:
        """Jacobian of the parameter vector in a minimal set of free parameters.

        Parameters other than transition probabilities are free. In column j
        of the transition matrix the free entries sum to a constant, so all
        but the last of them are free and the last is determined.
        """
        k = self.k_regimes  # type: ignore[attr-defined]
        n = int(self.k_params)  # type: ignore[attr-defined]
        index = self._transition_index()
        is_transition = np.zeros(n, dtype=bool)
        is_transition[index.ravel()] = True

        columns = []
        for ix in np.flatnonzero(~is_transition):
            column = np.zeros(n)
            column[ix] = 1.0
            columns.append(column)
        for j in range(k):
            free = [i for i in range(k) if (i, j) not in self._restrictions]
            if not free:
                continue
            determined = free[-1]
            for i in free[:-1]:
                column = np.zeros(n)
                column[index[i, j]] = 1.0
                if determined < k - 1:
                    column[index[determined, j]] = -1.0
                columns.append(column)
        return np.column_stack(columns)

    @staticmethod
    def _restricted_log_transition(
        regime_transition: NDArray[Any],
    ) -> NDArray[Any]:
        """Convert transition matrix to log-space with exact zeros."""
        return np.where(
            regime_transition == 0,
            _LOG_ZERO,
            np.log(np.maximum(regime_transition, 1e-20)),
        )

    def _filter(
        self, params: NDArray[Any], regime_transition: NDArray[Any] | None = None
    ) -> tuple[Any, ...]:
        if not self._restrictions:
            return super()._filter(params, regime_transition)  # type: ignore[misc]

        # Replicate cy_hamilton_filter_log with our strict log-zero
        if regime_transition is None:
            regime_transition = self.regime_transition_matrix(params)
        initial_probabilities = self.initial_probabilities(  # type: ignore[attr-defined]
            params, regime_transition
        )
        conditional_loglikelihoods = self._conditional_loglikelihoods(params)  # type: ignore[attr-defined]

        # --- begin cy_hamilton_filter_log logic with restricted log ---
        k_regimes = len(initial_probabilities)
        nobs = conditional_loglikelihoods.shape[-1]
        order = conditional_loglikelihoods.ndim - 2
        dtype = conditional_loglikelihoods.dtype

        # A known starting distribution can have exact zeros
        log_initial = np.where(
            initial_probabilities == 0,
            _LOG_ZERO,
            np.log(np.maximum(initial_probabilities, 1e-20)),
        )
        log_transition = self._restricted_log_transition(regime_transition)

        filtered_marginal_probabilities = np.zeros((k_regimes, nobs), dtype=dtype)
        predicted_joint_probabilities = np.zeros(
            (k_regimes,) * (order + 1) + (nobs,), dtype=dtype
        )
        joint_loglikelihoods = np.zeros((nobs,), dtype=dtype)
        filtered_joint_probabilities = np.zeros(
            (k_regimes,) * (order + 1) + (nobs + 1,), dtype=dtype
        )

        filtered_marginal_probabilities[:, 0] = log_initial
        tmp = np.copy(log_initial)
        shape = (k_regimes, k_regimes)
        transition_t = 0
        for i in range(order):
            if log_transition.shape[-1] > 1:
                transition_t = i
            tmp = np.reshape(log_transition[..., transition_t], shape + (1,) * i) + tmp
        filtered_joint_probabilities[..., 0] = tmp

        if log_transition.shape[-1] > 1:
            log_transition = log_transition[..., self.order :]  # type: ignore[attr-defined]

        prefix, _dtype_blas, _ = find_best_blas_type(
            (
                log_transition,
                conditional_loglikelihoods,
                joint_loglikelihoods,
                predicted_joint_probabilities,
                filtered_joint_probabilities,
            )
        )
        func = prefix_hamilton_filter_log_map[prefix]
        func(
            nobs,
            k_regimes,
            order,
            log_transition,
            conditional_loglikelihoods.reshape(k_regimes ** (order + 1), nobs),
            joint_loglikelihoods,
            predicted_joint_probabilities.reshape(k_regimes ** (order + 1), nobs),
            filtered_joint_probabilities.reshape(k_regimes ** (order + 1), nobs + 1),
        )

        predicted_joint_probabilities_log = predicted_joint_probabilities
        filtered_joint_probabilities_log = filtered_joint_probabilities

        predicted_joint_probabilities = np.exp(predicted_joint_probabilities)
        filtered_joint_probabilities = np.exp(filtered_joint_probabilities)

        filtered_marginal_probabilities = filtered_joint_probabilities[..., 1:]
        for _i in range(1, filtered_marginal_probabilities.ndim - 1):
            filtered_marginal_probabilities = np.sum(
                filtered_marginal_probabilities, axis=-2
            )
        # --- end cy_hamilton_filter_log logic ---

        return (
            regime_transition,
            initial_probabilities,
            conditional_loglikelihoods,
            filtered_marginal_probabilities,
            predicted_joint_probabilities,
            joint_loglikelihoods,
            filtered_joint_probabilities[..., 1:],
            predicted_joint_probabilities_log,
            filtered_joint_probabilities_log[..., 1:],
        )

    def _smooth(
        self,
        params: NDArray[Any],
        predicted_joint_probabilities_log: NDArray[Any],
        filtered_joint_probabilities_log: NDArray[Any],
        regime_transition: NDArray[Any] | None = None,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        if not self._restrictions:
            return super()._smooth(  # type: ignore[misc]
                params,
                predicted_joint_probabilities_log,
                filtered_joint_probabilities_log,
                regime_transition,
            )

        if regime_transition is None:
            regime_transition = self.regime_transition_matrix(params)

        # --- begin cy_kim_smoother_log logic with restricted log ---
        k_regimes = filtered_joint_probabilities_log.shape[0]
        nobs = filtered_joint_probabilities_log.shape[-1]
        order = filtered_joint_probabilities_log.ndim - 2
        dtype = filtered_joint_probabilities_log.dtype

        smoothed_joint_probabilities = np.zeros(
            (k_regimes,) * (order + 1) + (nobs,), dtype=dtype
        )

        if regime_transition.shape[-1] == nobs + order:
            regime_transition = regime_transition[..., order:]

        log_transition = self._restricted_log_transition(regime_transition)

        prefix, _dtype_blas, _ = find_best_blas_type(
            (
                log_transition,
                predicted_joint_probabilities_log,
                filtered_joint_probabilities_log,
            )
        )
        func = prefix_kim_smoother_log_map[prefix]
        func(
            nobs,
            k_regimes,
            order,
            log_transition,
            predicted_joint_probabilities_log.reshape(k_regimes ** (order + 1), nobs),
            filtered_joint_probabilities_log.reshape(k_regimes ** (order + 1), nobs),
            smoothed_joint_probabilities.reshape(k_regimes ** (order + 1), nobs),
        )

        smoothed_joint_probabilities = np.exp(smoothed_joint_probabilities)

        smoothed_marginal_probabilities = smoothed_joint_probabilities
        for _i in range(1, smoothed_marginal_probabilities.ndim - 1):
            smoothed_marginal_probabilities = np.sum(
                smoothed_marginal_probabilities, axis=-2
            )
        # --- end cy_kim_smoother_log logic ---

        return smoothed_joint_probabilities, smoothed_marginal_probabilities


class _RestrictedSMMarkovRegression(_RestrictedFilterMixin, SMMarkovRegression):
    """statsmodels MarkovRegression with restricted transition probabilities.

    The restrictions are imposed in ``transform_params`` and the filter and
    smoother use a strict log-zero (both in ``_RestrictedFilterMixin``).

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


class _RestrictedSMMarkovAutoregression(
    _RestrictedFilterMixin, _SMMarkovAutoregression
):
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


def _reject_ordering(kwargs: dict[str, Any]) -> None:
    """Drop an ``ordering`` argument, warning if it asks for relabeling.

    The restrictions are stated in terms of the regime labels, so a model
    with restricted transition probabilities keeps the labels of the
    estimation.
    """
    if kwargs.pop("ordering", None) is not None:
        warnings.warn(
            "ordering is ignored in a model with restricted transition "
            "probabilities: the restrictions refer to the regime labels, so "
            "the regimes are not relabeled.",
            UserWarning,
            stacklevel=3,
        )


def _validate_restrictions(
    k_regimes: int, restrictions: dict[tuple[int, int], float]
) -> None:
    """Check that the restrictions describe a possible transition matrix."""
    for (i, j), value in restrictions.items():
        if not (0 <= i < k_regimes and 0 <= j < k_regimes):
            raise ValueError(
                f"restriction ({i}, {j}) is outside a {k_regimes}-regime "
                "transition matrix"
            )
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"restriction ({i}, {j}) = {value} is not a probability")
    for j in range(k_regimes):
        fixed = [v for (_i, jj), v in restrictions.items() if jj == j]
        total = float(sum(fixed))
        if total > 1.0 + 1e-12:
            raise ValueError(
                f"the fixed probabilities of leaving regime {j} sum to {total}"
            )
        if len(fixed) == k_regimes and abs(total - 1.0) > 1e-12:
            raise ValueError(
                f"all probabilities of leaving regime {j} are fixed but sum "
                f"to {total}, not 1"
            )


def _validate_initial_probabilities(
    k_regimes: int, initial_probabilities: Any
) -> NDArray[np.floating[Any]] | None:
    if initial_probabilities is None:
        return None
    probs = np.asarray(initial_probabilities, dtype=np.float64).ravel()
    if probs.shape != (k_regimes,) or np.any(probs < 0) or abs(probs.sum() - 1) > 1e-8:
        raise ValueError(
            f"initial_probabilities must be {k_regimes} non-negative numbers "
            "that sum to 1"
        )
    return probs


def _default_initial_probabilities(
    k_regimes: int, restrictions: dict[tuple[int, int], float]
) -> NDArray[np.floating[Any]] | None:
    """Starting distribution for a chain that the restrictions make reducible.

    If every regime can be reached from every other, the stationary
    distribution is used (None is returned). Otherwise the stationary
    distribution puts no mass on the regimes that the chain eventually leaves
    for good, so the model could never be in them. The chain then starts,
    with equal probabilities, in the regimes that cannot be entered from
    outside their own class: regime 0 of a non-recurring model.
    """
    k = k_regimes
    # allowed[i, j]: the chain can move from j to i
    allowed = np.ones((k, k), dtype=bool)
    for (i, j), value in restrictions.items():
        if value == 0.0:
            allowed[i, j] = False
    reach = allowed.T | np.eye(k, dtype=bool)  # reach[j, i]: i reachable from j
    for m in range(k):
        reach |= reach[:, [m]] & reach[[m], :]
    same_class = reach & reach.T
    if same_class.all():
        return None
    start = np.zeros(k)
    for r in range(k):
        members = same_class[r]
        entered = allowed[np.ix_(members, ~members)].any()
        if not entered:
            start[r] = 1.0
    return start / start.sum()


def _non_recurring_restrictions(k_regimes: int) -> dict[tuple[int, int], float]:
    """Chib (1998): from regime j the chain stays or moves to regime j + 1."""
    return {
        (i, j): 0.0
        for j in range(k_regimes)
        for i in range(k_regimes)
        if i not in (j, j + 1)
    }


def _time_ordered_start_params(sm_model: Any) -> NDArray[np.floating[Any]]:
    """Starting values for a model whose regimes follow one another in time.

    statsmodels' default starting values order the regimes by the size of
    their coefficients, which need not be the order in which they occur. A
    non-recurring model started that way can only leave its first regime at
    once. Here regime j starts from a regression on the j-th of ``k`` equal
    stretches of the sample, and each regime is expected to last ``nobs / k``
    periods.

    Returns
    -------
    NDArray[np.floating]
        Starting values on the scale of the parameters (``transformed=True``).
    """
    k = int(sm_model.k_regimes)
    params = np.array(sm_model.start_params, dtype=np.float64, copy=True)
    parameters = sm_model.parameters
    all_idx = np.arange(sm_model.k_params)
    endog = np.asarray(sm_model.endog, dtype=np.float64).ravel()
    nobs = len(endog)
    bounds = np.linspace(0, nobs, k + 1).astype(int)

    exog = getattr(sm_model, "exog", None)
    if exog is not None and np.size(exog) > 0:
        exog = np.asarray(exog, dtype=np.float64).reshape(nobs, -1)
        full = np.linalg.lstsq(exog, endog, rcond=None)[0]
        switching = np.atleast_1d(parameters.switching["exog"]).astype(bool)
        resid = np.zeros(nobs)
        for j in range(k):
            rows = slice(bounds[j], bounds[j + 1])
            beta = full.copy()
            if bounds[j + 1] - bounds[j] > exog.shape[1]:
                local = np.linalg.lstsq(exog[rows], endog[rows], rcond=None)[0]
                beta[switching] = local[switching]
            params[all_idx[parameters[j, "exog"]]] = beta
            resid[rows] = endog[rows] - exog[rows] @ beta
    else:
        resid = endog - endog.mean()

    for j in range(k):
        rows = slice(bounds[j], bounds[j + 1])
        if np.atleast_1d(parameters.switching["variance"])[0]:
            variance = float(np.mean(resid[rows] ** 2))
        else:
            variance = float(np.mean(resid**2))
        params[all_idx[parameters[j, "variance"]]] = max(variance, 1e-8)

    # Stay with probability 1 - k / nobs, otherwise share the rest among the
    # moves the restrictions allow
    stay = min(max(1.0 - k / nobs, 0.5), 0.99)
    matrix = np.full((k, k), (1.0 - stay) / max(k - 1, 1))
    np.fill_diagonal(matrix, stay)
    matrix = sm_model._impose_restrictions(matrix)
    params[sm_model._transition_index()] = matrix[: k - 1, :]
    return params


def _restricted_inference(
    sm_results: Any,
) -> tuple[NDArray[np.floating[Any]], int]:
    """Standard errors and the number of free parameters of a restricted fit.

    The likelihood is differentiated with respect to a minimal set of free
    parameters, so that every evaluation satisfies the restrictions, and the
    covariance matrix of the full parameter vector follows by the delta
    method (exact: the map is linear). A parameter that the restrictions
    determine gets a NaN standard error.

    statsmodels' own standard errors differentiate with respect to every
    transition probability, restricted or not, which leaves the restricted
    model and counts parameters that are not estimated.
    """
    model = sm_results.model
    params = np.asarray(sm_results.params, dtype=np.float64)
    jacobian = model._free_parameter_jacobian()
    n_free = jacobian.shape[1]

    def loglike(theta: NDArray[Any]) -> Any:
        return model.loglike(params + jacobian @ theta, transformed=True)

    bse = np.full(len(params), np.nan)
    try:
        hessian = np.asarray(approx_hess_cs(np.zeros(n_free), loglike))
        cov_free = -np.linalg.pinv(hessian)
        variances = np.einsum("ij,jk,ik->i", jacobian, cov_free, jacobian)
    except Exception:
        warnings.warn(
            "standard errors of the restricted model could not be computed",
            UserWarning,
            stacklevel=3,
        )
        return bse, n_free
    estimated = np.any(jacobian != 0, axis=1) & (variances > 0)
    bse[estimated] = np.sqrt(variances[estimated])
    return bse, n_free


def _finish_restricted_results(
    results: Any,
    sm_results: Any,
    restrictions: dict[tuple[int, int], float],
    inference: bool = True,
) -> None:
    """Checks and inference common to the two restricted model classes.

    With ``inference=False`` the standard errors are not computed (they need
    a numerical Hessian); the tests that only compare likelihoods use this.
    """
    for (i, j), value in restrictions.items():
        if abs(float(results.regime_transition[i, j]) - value) > 1e-8:
            raise RuntimeError(
                f"the estimated transition matrix violates restriction "
                f"({i}, {j}) = {value}: {results.regime_transition[i, j]}"
            )
    results.restricted_transitions = dict(restrictions)
    if restrictions and inference:
        results.bse, results.n_free_params = _restricted_inference(sm_results)


class RestrictedMarkovRegression(MarkovRegression):
    """MarkovRegression with fixed transition probabilities.

    The regimes are never relabeled (``ordering`` is not available): the
    restrictions refer to the regime labels.

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
    initial_probabilities : ArrayLike | None
        Distribution of the regime before the sample starts: in
        statsmodels' filter, two periods before the first observation (one
        period before it in ``RestrictedMarkovAR``). A chain that starts in
        regime 0 can therefore have left it by the first observation. If
        None (default)
        and every regime can be reached from every other, the stationary
        distribution of the transition matrix is used, as in statsmodels.
        If the restrictions make some regimes impossible to return to, the
        stationary distribution puts no mass on them; the chain then starts,
        with equal probabilities, in the regimes that cannot be entered from
        outside (regime 0 of a non-recurring model).
    **kwargs
        Additional arguments for MarkovRegression.

    Notes
    -----
    Standard errors and ``df_model`` (and so the information criteria) count
    only the parameters that are free under the restrictions; a transition
    probability that the restrictions determine has a NaN standard error.

    Examples
    --------
    >>> import numpy as np
    >>> from regimes.markov.restricted import RestrictedMarkovRegression
    >>> np.random.seed(42)
    >>> y = np.concatenate([np.random.randn(100) + 1, np.random.randn(100) + 3])
    >>> # Non-recurring: forbid going back from regime 1 to regime 0
    >>> model = RestrictedMarkovRegression(
    ...     y, k_regimes=2, restrictions={(0, 1): 0.0},
    ...     initial_probabilities=[1.0, 0.0],
    ... )
    >>> results = model.fit()
    >>> print(results.regime_transition)
    """

    def __init__(
        self,
        endog: Any,
        k_regimes: int = 2,
        restrictions: dict[tuple[int, int], float] | None = None,
        initial_probabilities: Any = None,
        **kwargs: Any,
    ) -> None:
        self.restrictions = restrictions or {}
        _validate_restrictions(k_regimes, self.restrictions)
        self.initial_probabilities = _validate_initial_probabilities(
            k_regimes, initial_probabilities
        )
        if self.initial_probabilities is None:
            self.initial_probabilities = _default_initial_probabilities(
                k_regimes, self.restrictions
            )
        # Set by non_recurring(): the regimes follow one another in time
        self._time_ordered = bool(kwargs.pop("_time_ordered", False))
        _reject_ordering(kwargs)
        super().__init__(endog, k_regimes=k_regimes, ordering=None, **kwargs)

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
        if self.initial_probabilities is not None:
            sm_model.initialize_known(self.initial_probabilities)
        if self._time_ordered and "start_params" not in kwargs:
            kwargs["start_params"] = _time_ordered_start_params(sm_model)

        sm_results = self._fit_with_search(
            sm_model, method, maxiter, em_iter, search_reps, **kwargs
        )

        results = self._wrap_results(sm_results)
        _finish_restricted_results(
            results,
            sm_results,
            self.restrictions,
            inference=not getattr(self, "_skip_inference", False),
        )
        return results

    @staticmethod
    def non_recurring(
        endog: Any,
        k_regimes: int = 2,
        **kwargs: Any,
    ) -> RestrictedMarkovRegression:
        """Create a non-recurring (structural break) model.

        Sets up the Chib (1998) structure: the chain starts in regime 0 and
        can only stay put or advance to the next regime. The last regime is
        absorbing.

        For k=3 regimes, the transition matrix (column = regime of origin) is:
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
            Additional arguments. ``initial_probabilities`` overrides the
            start in regime 0.

        Returns
        -------
        RestrictedMarkovRegression
            Model with non-recurring transition structure.
        """
        kwargs.setdefault("initial_probabilities", np.eye(k_regimes)[0])
        return RestrictedMarkovRegression(
            endog,
            k_regimes=k_regimes,
            restrictions=_non_recurring_restrictions(k_regimes),
            _time_ordered=True,
            **kwargs,
        )


class RestrictedMarkovAR(MarkovAR):
    """MarkovAR with fixed transition probabilities.

    The regimes are never relabeled (``ordering`` is not available): the
    restrictions refer to the regime labels.

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
    initial_probabilities : ArrayLike | None
        Distribution of the regime one period before the first observation;
        see ``RestrictedMarkovRegression``.
    **kwargs
        Additional arguments for MarkovAR.
    """

    def __init__(
        self,
        endog: Any,
        k_regimes: int = 2,
        order: int = 1,
        restrictions: dict[tuple[int, int], float] | None = None,
        initial_probabilities: Any = None,
        **kwargs: Any,
    ) -> None:
        self.restrictions = restrictions or {}
        _validate_restrictions(k_regimes, self.restrictions)
        self.initial_probabilities = _validate_initial_probabilities(
            k_regimes, initial_probabilities
        )
        if self.initial_probabilities is None:
            self.initial_probabilities = _default_initial_probabilities(
                k_regimes, self.restrictions
            )
        # Set by non_recurring(): the regimes follow one another in time
        self._time_ordered = bool(kwargs.pop("_time_ordered", False))
        _reject_ordering(kwargs)
        super().__init__(
            endog, k_regimes=k_regimes, order=order, ordering=None, **kwargs
        )

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
        if self.initial_probabilities is not None:
            sm_model.initialize_known(self.initial_probabilities)
        if self._time_ordered and "start_params" not in kwargs:
            kwargs["start_params"] = _time_ordered_start_params(sm_model)

        sm_results = self._fit_with_search(
            sm_model, method, maxiter, em_iter, search_reps, **kwargs
        )

        results = self._wrap_results(sm_results)
        _finish_restricted_results(
            results,
            sm_results,
            self.restrictions,
            inference=not getattr(self, "_skip_inference", False),
        )
        return results

    @staticmethod
    def non_recurring(
        endog: Any,
        k_regimes: int = 2,
        order: int = 1,
        **kwargs: Any,
    ) -> RestrictedMarkovAR:
        """Create a non-recurring (structural break) AR model.

        The chain starts in regime 0 and can only stay put or advance to the
        next regime; see ``RestrictedMarkovRegression.non_recurring``.
        """
        kwargs.setdefault("initial_probabilities", np.eye(k_regimes)[0])
        return RestrictedMarkovAR(
            endog,
            k_regimes=k_regimes,
            order=order,
            restrictions=_non_recurring_restrictions(k_regimes),
            _time_ordered=True,
            **kwargs,
        )
