"""Simulation from Markov switching models and Monte Carlo p-values.

Internal helpers for the parametric bootstrap tests in
:mod:`regimes.markov.sequential_restriction` and
:mod:`regimes.markov.selection`.

The data-generating process is the one whose likelihood the statsmodels
classes evaluate:

* The regimes follow the transition matrix, whose entry ``[i, j]`` is
  ``P(S_t = i | S_{t-1} = j)``. The chain starts before the sample: the
  model's initial distribution is that of the regime two periods before the
  first observation in a ``MarkovRegression`` and one period before it in a
  ``MarkovAutoregression``. (This is what statsmodels' filter computes; it
  was established by comparing its likelihood with a direct sum over all
  regime paths, see ``tests/test_markov/test_simulate.py``. With the
  stationary distribution the timing does not matter.)
* ``MarkovRegression``: ``y_t = x_t' b(S_t) + s(S_t) e_t``.
* ``MarkovAutoregression`` (Hamilton's mean-adjusted form)::

      y_t = x_t' b(S_t)
            + sum_i phi_i(S_t) (y_{t-i} - x_{t-i}' b(S_{t-i})) + s(S_t) e_t

  The first ``order`` observations are the observed ones: the likelihood is
  conditional on them.

``x_t`` holds the trend terms and the regressors, which are kept at their
observed values. That is appropriate for strictly exogenous regressors only
(Qu and Zhuo, 2021, Section 7): a regressor that depends on past values of
``y`` would have to be simulated jointly with it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

RandomState = int | np.random.Generator | None


def as_generator(random_state: RandomState) -> np.random.Generator:
    """Turn a seed, a generator or None into a ``numpy.random.Generator``."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def presample_periods(order: int) -> int:
    """Periods between the regime drawn from the initial distribution and the
    first period of the sample.

    Two for a ``MarkovRegression`` (``order = 0``) and one for a
    ``MarkovAutoregression``: the timing of statsmodels' filter.
    """
    return 2 if order == 0 else 1


def draw_regime_path(
    transition: NDArray[np.floating[Any]],
    initial: NDArray[np.floating[Any]],
    nobs: int,
    rng: np.random.Generator,
    n_presample: int = 1,
) -> NDArray[np.intp]:
    """Draw ``S_0, ..., S_{nobs-1}`` given the distribution of ``S_{-n_presample}``.

    Parameters
    ----------
    transition : NDArray
        ``(k, k)`` matrix with ``transition[i, j] = P(S_t = i | S_{t-1} = j)``
        (columns sum to one).
    initial : NDArray
        Distribution of the regime ``n_presample`` periods before the first
        one.
    nobs : int
        Length of the path.
    rng : numpy.random.Generator
        Random number generator.
    n_presample : int
        Number of periods between the regime drawn from ``initial`` and the
        first regime of the path (at least 1).
    """
    if n_presample < 1:
        raise ValueError("n_presample must be at least 1")
    transition = np.asarray(transition, dtype=np.float64)
    k = transition.shape[0]
    if transition.shape != (k, k):
        raise ValueError("transition must be a square matrix")
    column_sums = transition.sum(axis=0)
    if np.any(transition < -1e-12) or np.any(np.abs(column_sums - 1.0) > 1e-8):
        raise ValueError("the columns of transition must be probabilities")
    initial = np.asarray(initial, dtype=np.float64).ravel()
    if initial.shape != (k,) or np.any(initial < -1e-12):
        raise ValueError("initial must hold one probability per regime")

    cumulative = np.cumsum(np.clip(transition, 0.0, None), axis=0)
    cumulative /= cumulative[-1]
    start = np.cumsum(np.clip(initial, 0.0, None))
    start /= start[-1]

    draws = rng.random(nobs + n_presample)
    state = min(int(np.searchsorted(start, draws[0], side="right")), k - 1)
    path = np.empty(nobs, dtype=np.intp)
    for step in range(1, nobs + n_presample):
        state = min(
            int(np.searchsorted(cumulative[:, state], draws[step], side="right")),
            k - 1,
        )
        if step >= n_presample:
            path[step - n_presample] = state
    return path


def generate_series(
    path: NDArray[np.intp],
    means: NDArray[np.floating[Any]],
    sigmas: NDArray[np.floating[Any]],
    ar_coefs: NDArray[np.floating[Any]] | None,
    initial_values: NDArray[np.floating[Any]] | None,
    rng: np.random.Generator,
) -> NDArray[np.floating[Any]]:
    """Generate the series given the regime path.

    Parameters
    ----------
    path : NDArray
        Regime of each period, length ``nobs``.
    means : NDArray
        ``(nobs, k)`` array: ``means[t, j] = x_t' b(j)``.
    sigmas : NDArray
        Standard deviation of the error in each regime, length ``k``.
    ar_coefs : NDArray | None
        ``(k, order)`` autoregressive coefficients, or None.
    initial_values : NDArray | None
        The first ``order`` observations (required with ``ar_coefs``).
    rng : numpy.random.Generator
        Random number generator.
    """
    nobs = len(path)
    rows = np.arange(nobs)
    mean_path = np.asarray(means, dtype=np.float64)[rows, path]
    shocks = np.asarray(sigmas, dtype=np.float64)[path] * rng.standard_normal(nobs)
    if ar_coefs is None or np.shape(ar_coefs)[1] == 0:
        return mean_path + shocks

    ar_coefs = np.asarray(ar_coefs, dtype=np.float64)
    order = ar_coefs.shape[1]
    if initial_values is None or len(initial_values) < order:
        raise ValueError("initial_values must hold the first `order` observations")
    y = np.empty(nobs, dtype=np.float64)
    y[:order] = np.asarray(initial_values, dtype=np.float64)[:order]
    deviation = np.zeros(nobs, dtype=np.float64)
    deviation[:order] = y[:order] - mean_path[:order]
    for t in range(order, nobs):
        # deviation[t - 1], ..., deviation[t - order]
        lagged = deviation[t - order : t][::-1]
        deviation[t] = ar_coefs[path[t]] @ lagged + shocks[t]
        y[t] = mean_path[t] + deviation[t]
    return y


def simulate_markov_switching(
    sm_model: Any,
    params: Any,
    rng: np.random.Generator,
) -> NDArray[np.floating[Any]]:
    """Simulate one series from a statsmodels Markov switching model.

    Parameters
    ----------
    sm_model : statsmodels MarkovRegression or MarkovAutoregression
        The model (also the restricted subclasses of this package). Its
        trend, regressors, initial distribution and, for an autoregression,
        its first ``order`` observations are used.
    params : array_like
        Parameter vector in statsmodels' layout (constrained form).
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    NDArray
        A series as long as the sample the model was built on.
    """
    if getattr(sm_model, "tvtp", False):
        raise NotImplementedError(
            "simulation with time-varying transition probabilities is not supported"
        )
    params = np.asarray(params, dtype=np.float64)
    k = int(sm_model.k_regimes)
    order = int(getattr(sm_model, "order", 0))

    transition = np.asarray(sm_model.regime_transition_matrix(params))
    if transition.ndim == 3:
        transition = transition[..., 0]
    transition = np.clip(transition.astype(np.float64), 0.0, None)
    transition /= transition.sum(axis=0, keepdims=True)
    initial = np.asarray(
        sm_model.initial_probabilities(params, transition), dtype=np.float64
    )
    # statsmodels bounds the probabilities at 1e-20 for its log-space filter
    initial = np.where(initial <= 1e-12, 0.0, initial)
    initial /= initial.sum()

    if order > 0:
        endog = np.asarray(sm_model.orig_endog, dtype=np.float64).ravel()
        exog = sm_model.orig_exog if sm_model._k_exog > 0 else None
    else:
        endog = np.asarray(sm_model.endog, dtype=np.float64).ravel()
        exog = sm_model.exog if sm_model._k_exog > 0 else None
    nobs = len(endog)

    means = np.zeros((nobs, k), dtype=np.float64)
    if exog is not None:
        exog = np.asarray(exog, dtype=np.float64).reshape(nobs, -1)
        for j in range(k):
            means[:, j] = exog @ params[sm_model.parameters[j, "exog"]]
    variances = np.array(
        [
            float(np.squeeze(params[sm_model.parameters[j, "variance"]]))
            for j in range(k)
        ]
    )
    ar_coefs = None
    if order > 0:
        ar_coefs = np.vstack(
            [params[sm_model.parameters[j, "autoregressive"]] for j in range(k)]
        )

    path = draw_regime_path(
        transition, initial, nobs, rng, n_presample=presample_periods(order)
    )
    return generate_series(
        path, means, np.sqrt(variances), ar_coefs, endog[:order], rng
    )


def simulate_linear(
    exog: NDArray[np.floating[Any]] | None,
    beta: NDArray[np.floating[Any]] | None,
    sigma2: float,
    ar_coefs: NDArray[np.floating[Any]] | None,
    initial_values: NDArray[np.floating[Any]] | None,
    nobs: int,
    rng: np.random.Generator,
) -> NDArray[np.floating[Any]]:
    """Simulate from the one-regime model, in the same form as above.

    ``y_t = x_t' b + u_t`` with ``u_t = sum_i phi_i u_{t-i} + s e_t``. The
    first ``order`` observations are ``initial_values``.
    """
    means = np.zeros((nobs, 1), dtype=np.float64)
    if exog is not None and beta is not None and np.size(beta) > 0:
        means[:, 0] = np.asarray(exog, dtype=np.float64).reshape(nobs, -1) @ np.asarray(
            beta, dtype=np.float64
        )
    ar = None
    if ar_coefs is not None and np.size(ar_coefs) > 0:
        ar = np.asarray(ar_coefs, dtype=np.float64).reshape(1, -1)
    path = np.zeros(nobs, dtype=np.intp)
    return generate_series(
        path, means, np.array([np.sqrt(sigma2)]), ar, initial_values, rng
    )


@dataclass(kw_only=True)
class MonteCarloPValue:
    """Outcome of a Monte Carlo (parametric bootstrap) test.

    Attributes
    ----------
    p_value : float
        ``(1 + number of simulated statistics >= observed) / (n_valid + 1)``.
    statistics : NDArray
        The simulated statistics that could be computed.
    n_valid : int
        Number of replications in which the statistic could be computed.
    n_failed : int
        Number of replications in which it could not.
    last_error : Exception | None
        The exception of the last replication that failed, if any.
    """

    p_value: float
    statistics: NDArray[np.floating[Any]]
    n_valid: int
    n_failed: int
    last_error: Exception | None = None

    def critical_value(self, significance: float) -> float:
        """The value an observed statistic must exceed to be rejected.

        The p-value ``(1 + #{simulated >= observed}) / (N + 1)`` is below
        ``significance`` exactly when the observed statistic is larger than
        the ``r``-th largest simulated one, ``r = ceil(significance (N + 1))
        - 1``. That value is returned, so that "statistic > critical value"
        and "p-value < significance" are the same decision. It is infinite
        if ``r = 0``: with so few replications the test cannot reject.
        """
        n = len(self.statistics)
        if n == 0:
            return float("nan")
        rank = int(np.ceil(significance * (n + 1) - 1e-12)) - 1
        if rank < 1:
            return float("inf")
        ordered = np.sort(self.statistics)[::-1]
        return float(ordered[min(rank, n) - 1])


def monte_carlo_pvalue(
    observed: float,
    simulate: Callable[[np.random.Generator], NDArray[np.floating[Any]]],
    statistic: Callable[[NDArray[np.floating[Any]]], float],
    n_replications: int,
    rng: np.random.Generator,
) -> MonteCarloPValue:
    """Monte Carlo p-value of a statistic that is large under the alternative.

    Each replication draws a sample under the null hypothesis with
    ``simulate`` and evaluates ``statistic`` on it. The p-value is
    ``(1 + #{simulated >= observed}) / (N + 1)`` (Dufour, 2006), which is
    never zero and counts the observed statistic as one draw.

    A replication in which ``statistic`` raises, or returns a value that is
    not finite, is left out; a warning is issued if more than 10 percent of
    the replications are lost.
    """
    if n_replications < 1:
        raise ValueError("n_replications must be at least 1")
    values: list[float] = []
    n_failed = 0
    last_error: Exception | None = None
    for _ in range(n_replications):
        sample = simulate(rng)
        try:
            value = float(statistic(sample))
        except Exception as error:
            n_failed += 1
            last_error = error
            continue
        if not np.isfinite(value):
            n_failed += 1
            continue
        values.append(value)

    statistics = np.asarray(values, dtype=np.float64)
    n_valid = len(values)
    if n_failed > 0.1 * n_replications:
        warnings.warn(
            f"the statistic could not be computed in {n_failed} of "
            f"{n_replications} bootstrap replications; the p-value rests on "
            f"the remaining {n_valid}",
            UserWarning,
            stacklevel=3,
        )
    if n_valid == 0:
        return MonteCarloPValue(
            p_value=float("nan"),
            statistics=statistics,
            n_valid=0,
            n_failed=n_failed,
            last_error=last_error,
        )
    exceed = int(np.sum(statistics >= observed))
    return MonteCarloPValue(
        p_value=(1 + exceed) / (n_valid + 1),
        statistics=statistics,
        n_valid=n_valid,
        n_failed=n_failed,
        last_error=last_error,
    )


def smallest_pvalue(n_replications: int) -> float:
    """The smallest p-value ``monte_carlo_pvalue`` can return: ``1 / (N + 1)``."""
    return 1.0 / (n_replications + 1)
