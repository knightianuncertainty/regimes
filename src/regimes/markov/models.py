"""Markov regime-switching models wrapping statsmodels.

This module provides MarkovRegression, MarkovAR, and MarkovADL classes
that wrap statsmodels' MarkovRegression and MarkovAutoregression with the
regimes API pattern: Model(...).fit() -> Results.

The one-way mapping from existing OLS/AR/ADL models is provided via
``from_model()`` class methods and ``markov_switching()`` convenience
methods on the original model classes.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_autoregression import (
    MarkovAutoregression,
)
from statsmodels.tsa.regime_switching.markov_regression import (
    MarkovRegression as SMMarkovRegression,
)

from regimes.markov.results import (
    MarkovADLResults,
    MarkovARResults,
    MarkovRegressionResults,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    from regimes.models.adl import ADL
    from regimes.models.ar import AR
    from regimes.models.ols import OLS


def _ensure_1d(y: ArrayLike | pd.Series[Any] | pd.DataFrame) -> NDArray[Any]:
    """Convert endog to 1D numpy array."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.squeeze()
    return arr


def _ensure_2d(
    x: ArrayLike | pd.Series[Any] | pd.DataFrame | None,
) -> NDArray[Any] | None:
    """Convert exog to 2D numpy array, or None."""
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _relabel_regimes(
    sm_results: Any,
    ordering: str | None,
    k_regimes: int,
) -> NDArray[np.intp]:
    """Compute a permutation to relabel regimes.

    Parameters
    ----------
    sm_results : statsmodels results
        Fitted MS model results.
    ordering : str | None
        "first_appearance", "intercept", or None.
    k_regimes : int
        Number of regimes.

    Returns
    -------
    NDArray[np.intp]
        Permutation array: new_label[old_label] = permuted_label.
    """
    identity = np.arange(k_regimes, dtype=np.intp)

    if ordering is None:
        return identity

    if ordering == "first_appearance":
        # Regime 0 is whichever appears first in the most-likely state sequence
        most_likely = sm_results.smoothed_marginal_probabilities.argmax(axis=1)
        seen = []
        for s in most_likely:
            if s not in seen:
                seen.append(int(s))
            if len(seen) == k_regimes:
                break
        # Fill any unseen regimes
        for r in range(k_regimes):
            if r not in seen:
                seen.append(r)
        # seen[i] is the old label that should become regime i
        perm = np.zeros(k_regimes, dtype=np.intp)
        for new_label, old_label in enumerate(seen):
            perm[old_label] = new_label
        return perm

    if ordering == "intercept":
        # Order by the regime-specific constant (regime 0 = lowest). In
        # MarkovAR the constant is the regime mean (Hamilton's form).
        layout = _parameter_layout(sm_results.model)
        params = np.asarray(sm_results.params, dtype=np.float64)
        constants = {
            info.regime: float(params[i])
            for i, info in enumerate(layout)
            if info.kind == "coef" and info.base == "const" and info.regime is not None
        }
        if len(constants) != k_regimes:
            warnings.warn(
                'ordering="intercept" needs a constant that switches across '
                "regimes (trend with a constant and switching_trend=True); "
                "the regimes are left in the order of the estimation.",
                UserWarning,
                stacklevel=3,
            )
            return identity
        sorted_indices = np.argsort(
            [constants[j] for j in range(k_regimes)], kind="stable"
        )
        perm = np.zeros(k_regimes, dtype=np.intp)
        for new_label, old_label in enumerate(sorted_indices):
            perm[old_label] = new_label
        return perm

    return identity


def _apply_permutation(
    perm: NDArray[np.intp],
    k_regimes: int,
    transition: NDArray[Any],
    smoothed: NDArray[Any],
    filtered: NDArray[Any],
    predicted: NDArray[Any],
    regime_params: dict[int, dict[str, float]],
) -> tuple[
    NDArray[Any],
    NDArray[Any],
    NDArray[Any],
    NDArray[Any],
    dict[int, dict[str, float]],
]:
    """Apply a regime permutation to all regime-indexed quantities."""
    if np.array_equal(perm, np.arange(k_regimes)):
        return transition, smoothed, filtered, predicted, regime_params

    # Permute transition matrix: new[perm[i], perm[j]] = old[i, j]
    new_transition = np.zeros_like(transition)
    for i in range(k_regimes):
        for j in range(k_regimes):
            new_transition[perm[i], perm[j]] = transition[i, j]

    # Permute probability columns
    new_smoothed = smoothed[:, np.argsort(perm)]
    new_filtered = filtered[:, np.argsort(perm)]
    new_predicted = predicted[:, np.argsort(perm)]

    # Permute regime_params keys
    new_regime_params = {}
    for old_label, params in regime_params.items():
        new_regime_params[int(perm[old_label])] = params

    return new_transition, new_smoothed, new_filtered, new_predicted, new_regime_params


_REGIME_SUFFIX = re.compile(r"\[\d+\]$")


@dataclass(frozen=True)
class _ParamInfo:
    """What one element of the statsmodels parameter vector is.

    kind : "transition" or "coef"
    base : name without a regime suffix ("p" for transition probabilities)
    regime : for a coefficient, the regime it belongs to (None if it does not
        switch); for a transition probability, the regime of origin
    to : destination regime of a transition probability, else None
    group : position of the parameter's group in the statsmodels vector, used
        to keep the statsmodels order of the groups
    """

    kind: str
    base: str
    regime: int | None
    to: int | None
    group: int


def _parameter_layout(
    sm_model: Any, exog_names: Sequence[str] | None = None
) -> list[_ParamInfo]:
    """Describe each element of the statsmodels parameter vector.

    The layout is read from ``sm_model.parameters``, which records for every
    regime where its parameters are and which of them switch. The names alone
    are not enough: statsmodels gives a non-switching regression coefficient
    the suffix of the last regime (``x1[1]`` with two regimes).

    Parameters
    ----------
    sm_model : statsmodels Markov switching model
        ``MarkovRegression`` or ``MarkovAutoregression``.
    exog_names : Sequence[str] | None
        Names for the columns of ``exog`` (after any trend terms). If None,
        the statsmodels names (``x1``, ``x2``, ...) are kept.

    Returns
    -------
    list[_ParamInfo]
        One entry per parameter, in the statsmodels order.
    """
    k = int(sm_model.k_regimes)
    raw_names = list(sm_model.param_names)
    all_idx = np.arange(len(raw_names))
    parameters = sm_model.parameters
    k_trend = int(getattr(sm_model, "k_trend", 0))

    layout: dict[int, _ParamInfo] = {}
    for ptype in parameters.switching:
        per_regime = [np.atleast_1d(all_idx[parameters[j, ptype]]) for j in range(k)]
        if ptype == "regime_transition":
            # per_regime[j] holds p[j->0], ..., p[j->k-2]
            for j in range(k):
                for i, ix in enumerate(per_regime[j]):
                    layout[int(ix)] = _ParamInfo("transition", "p", j, i, -1)
            continue
        switching = np.atleast_1d(parameters.switching[ptype])
        for pos in range(len(per_regime[0])):
            group = int(min(per_regime[j][pos] for j in range(k)))
            base = _REGIME_SUFFIX.sub("", raw_names[int(per_regime[0][pos])])
            if ptype == "exog" and exog_names is not None and pos >= k_trend:
                base = str(exog_names[pos - k_trend])
            for j in range(k):
                layout[int(per_regime[j][pos])] = _ParamInfo(
                    "coef", base, j if bool(switching[pos]) else None, None, group
                )

    if len(layout) != len(raw_names):
        raise RuntimeError(
            "could not map the statsmodels parameters to regimes: "
            f"{len(layout)} of {len(raw_names)} parameters identified"
        )
    return [layout[i] for i in range(len(raw_names))]


def _relabeled_parameters(
    sm_results: Any,
    perm: NDArray[np.intp],
    exog_names: Sequence[str] | None = None,
) -> tuple[
    list[str],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    dict[int, dict[str, float]],
]:
    """Parameter names, estimates and standard errors under relabeled regimes.

    ``perm[old_label]`` is the new label. A regime-specific coefficient
    ``name[j]`` becomes ``name[perm[j]]`` and the entries are sorted by regime
    within each group, so that ``const[0]`` is the constant of the regime that
    every other result calls regime 0. Coefficients that do not switch carry
    no suffix.

    Transition probabilities are reported as statsmodels reports them, for
    the destinations ``0, ..., k-2``: ``p[j->i]`` is the probability of moving
    from regime ``j`` to regime ``i`` in the new labels. statsmodels estimates
    the probabilities of moving to all regimes but its own last one. Where
    relabeling moves that regime away from the last position, the probability
    of moving to it is one minus the sum of the estimated ones, and its
    standard error follows from their covariance matrix (exact, because the
    function is linear).

    Returns
    -------
    tuple
        ``(param_names, params, bse, regime_params)``; ``regime_params[r]``
        holds every coefficient that applies in regime ``r``, including those
        that do not switch.
    """
    layout = _parameter_layout(sm_results.model, exog_names)
    k = int(sm_results.model.k_regimes)
    raw_params = np.asarray(sm_results.params, dtype=np.float64)
    raw_bse = np.asarray(sm_results.bse, dtype=np.float64)

    # (sort key, name, estimate, standard error)
    entries: list[tuple[tuple[int, int, int], str, float, float]] = []

    # Destination that statsmodels leaves out, in the new labels
    left_out = int(perm[k - 1])
    cov: NDArray[np.floating[Any]] | None = None
    if left_out != k - 1:
        try:
            cov = np.asarray(sm_results.cov_params(), dtype=np.float64)
        except Exception:
            cov = None

    for origin_old in range(k):
        origin = int(perm[origin_old])
        own = [
            ix
            for ix, info in enumerate(layout)
            if info.kind == "transition" and info.regime == origin_old
        ]
        for ix in own:
            to_old = layout[ix].to
            assert to_old is not None
            dest = int(perm[to_old])
            if dest != k - 1:
                value, se = float(raw_params[ix]), float(raw_bse[ix])
            else:
                # Report the probability of moving to the left-out regime
                dest = left_out
                value = 1.0 - float(raw_params[own].sum())
                if cov is None:
                    se = float("nan")
                else:
                    with np.errstate(invalid="ignore"):
                        se = float(np.sqrt(cov[np.ix_(own, own)].sum()))
            entries.append(((0, dest, origin), f"p[{origin}->{dest}]", value, se))

    regime_params: dict[int, dict[str, float]] = {r: {} for r in range(k)}
    for ix, info in enumerate(layout):
        if info.kind == "transition":
            continue
        value, se = float(raw_params[ix]), float(raw_bse[ix])
        if info.regime is None:
            entries.append(((1, info.group, 0), info.base, value, se))
            targets = list(range(k))
        else:
            new_label = int(perm[info.regime])
            entries.append(
                ((1, info.group, new_label), f"{info.base}[{new_label}]", value, se)
            )
            targets = [new_label]
        for r in targets:
            regime_params[r][info.base] = value

    entries.sort(key=lambda entry: entry[0])
    # Keep the statsmodels order of the coefficients within each regime
    for r in regime_params:
        regime_params[r] = {
            base: regime_params[r][base]
            for _key, base in sorted(
                {
                    (info.group, info.base)
                    for info in layout
                    if info.kind == "coef" and info.base in regime_params[r]
                }
            )
        }

    names = [entry[1] for entry in entries]
    params = np.array([entry[2] for entry in entries], dtype=np.float64)
    bse = np.array([entry[3] for entry in entries], dtype=np.float64)
    return names, params, bse, regime_params


def _extract_regime_params(
    sm_results: Any,
    k_regimes: int,
) -> dict[int, dict[str, float]]:
    """Per-regime parameters in the regime order of the statsmodels fit.

    Every coefficient that applies in a regime is included, whether or not it
    switches.
    """
    identity = np.arange(k_regimes, dtype=np.intp)
    return _relabeled_parameters(sm_results, identity)[3]


class MarkovRegression:
    """Markov regime-switching regression.

    Wraps ``statsmodels.tsa.regime_switching.MarkovRegression`` with the
    regimes API: ``MarkovRegression(endog, ...).fit() -> MarkovRegressionResults``.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    k_regimes : int
        Number of regimes. Default is 2.
    exog : ArrayLike | None
        Exogenous regressors (n_obs, k). If None, intercept-only model.
    trend : str
        Trend specification: "c" (constant, default), "ct" (constant + trend),
        "n" (no constant).
    switching_trend : bool
        Whether trend parameters switch across regimes. Default True.
    switching_exog : bool
        Whether exogenous coefficients switch. Default True.
    switching_variance : bool
        Whether the error variance switches. Default False.
    ordering : str | None
        How to order regimes after estimation:
        - "first_appearance" (default): Regime 0 appears first chronologically.
        - "intercept": Regime 0 has the lowest constant (needs a constant
          that switches across regimes).
        - None: Keep statsmodels' arbitrary ordering.

        Every result follows this order, including ``params``, ``bse`` and
        ``param_names``: ``const[0]`` is the constant of regime 0.

    Examples
    --------
    >>> import numpy as np
    >>> from regimes.markov import MarkovRegression
    >>> np.random.seed(42)
    >>> y = np.concatenate([np.random.randn(100), np.random.randn(100) + 3])
    >>> model = MarkovRegression(y, k_regimes=2)
    >>> results = model.fit()
    >>> print(results.summary())
    """

    def __init__(
        self,
        endog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        k_regimes: int = 2,
        exog: ArrayLike | pd.Series[Any] | pd.DataFrame | None = None,
        trend: str = "c",
        switching_trend: bool = True,
        switching_exog: bool = True,
        switching_variance: bool = False,
        ordering: str | None = "first_appearance",
    ) -> None:
        self.endog = _ensure_1d(endog)
        self.exog = _ensure_2d(exog)
        self.k_regimes = k_regimes
        self.trend = trend
        self.switching_trend = switching_trend
        self.switching_exog = switching_exog
        self.switching_variance = switching_variance
        self.ordering = ordering

    def fit(
        self,
        method: str = "bfgs",
        maxiter: int = 100,
        em_iter: int = 5,
        search_reps: int = 0,
        **kwargs: Any,
    ) -> MarkovRegressionResults:
        """Fit the Markov switching regression model.

        Parameters
        ----------
        method : str
            Optimization method for MLE. Default "bfgs".
        maxiter : int
            Maximum iterations for optimizer. Default 100.
        em_iter : int
            Number of EM iterations before switching to MLE. Default 5.
        search_reps : int
            Number of random starting values to try. 0 means use default
            starting values only. Default 0.
        **kwargs
            Additional arguments passed to statsmodels fit().

        Returns
        -------
        MarkovRegressionResults
            Fitted model results.
        """
        sm_model = SMMarkovRegression(
            endog=self.endog,
            k_regimes=self.k_regimes,
            exog=self.exog,
            trend=self.trend,
            switching_trend=self.switching_trend,
            switching_exog=self.switching_exog,
            switching_variance=self.switching_variance,
        )

        sm_results = self._fit_with_search(
            sm_model, method, maxiter, em_iter, search_reps, **kwargs
        )

        return self._wrap_results(sm_results)

    def _fit_with_search(
        self,
        sm_model: Any,
        method: str,
        maxiter: int,
        em_iter: int,
        search_reps: int,
        **kwargs: Any,
    ) -> Any:
        """Fit with optional random starting value search."""
        best_results = None
        best_llf = -np.inf

        # Always try default starting values first
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                results = sm_model.fit(
                    method=method,
                    maxiter=maxiter,
                    em_iter=em_iter,
                    **kwargs,
                )
                if np.isfinite(results.llf) and results.llf > best_llf:
                    best_results = results
                    best_llf = results.llf
            except Exception:
                pass

        # Try random starting values
        for _ in range(search_reps):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    results = sm_model.fit(
                        method=method,
                        maxiter=maxiter,
                        em_iter=0,
                        search_reps=1,
                        **kwargs,
                    )
                    if np.isfinite(results.llf) and results.llf > best_llf:
                        best_results = results
                        best_llf = results.llf
                except Exception:
                    continue

        if best_results is None:
            # Last resort: try without EM
            best_results = sm_model.fit(
                method=method,
                maxiter=maxiter,
                em_iter=0,
                **kwargs,
            )

        return best_results

    def _wrap_results(self, sm_results: Any) -> MarkovRegressionResults:
        """Wrap statsmodels results into MarkovRegressionResults."""
        k = self.k_regimes

        # Extract transition matrix (statsmodels returns (k, k, 1) — squeeze)
        transition = np.array(sm_results.regime_transition).squeeze()
        if transition.ndim == 1:
            transition = transition.reshape(k, k)

        # Extract probabilities
        smoothed = np.array(sm_results.smoothed_marginal_probabilities)
        filtered = np.array(sm_results.filtered_marginal_probabilities)
        predicted = np.array(sm_results.predicted_marginal_probabilities)

        # Relabel the regimes, and with them the parameter table
        perm = _relabel_regimes(sm_results, self.ordering, k)
        param_names, params, bse, regime_params = _relabeled_parameters(
            sm_results, perm
        )
        transition, smoothed, filtered, predicted, _ = _apply_permutation(
            perm, k, transition, smoothed, filtered, predicted, {}
        )

        # Compute fitted values and residuals (probability-weighted)
        fittedvalues = np.zeros(len(self.endog))
        for j in range(k):
            # Get regime-j predictions if available
            try:
                regime_fv = np.array(
                    sm_results.predict(probabilities="regime", which=j)
                )
                fittedvalues += smoothed[:, j] * regime_fv
            except Exception:
                pass

        if np.allclose(fittedvalues, 0):
            # Fallback: use overall predicted values
            try:
                fittedvalues = np.array(sm_results.predict())
            except Exception:
                fittedvalues = np.zeros(len(self.endog))

        resid = self.endog - fittedvalues

        # Get convergence info
        converged = getattr(sm_results, "mle_retvals", {}).get("converged", True)
        n_iterations = getattr(sm_results, "mle_retvals", {}).get("iterations", 0)
        if isinstance(converged, (int, np.integer)):
            converged = bool(converged)

        return MarkovRegressionResults(
            params=params,
            nobs=len(self.endog),
            model_name="Markov Switching Regression",
            k_regimes=k,
            regime_transition=transition,
            smoothed_marginal_probabilities=smoothed,
            filtered_marginal_probabilities=filtered,
            predicted_marginal_probabilities=predicted,
            bse=bse,
            llf=float(sm_results.llf),
            resid=resid,
            fittedvalues=fittedvalues,
            param_names=param_names,
            regime_params=regime_params,
            cov_type="approx",
            converged=converged,
            n_iterations=n_iterations,
            switching_trend=self.switching_trend,
            switching_exog=self.switching_exog,
            switching_variance=self.switching_variance,
            _sm_results=sm_results,
        )

    @classmethod
    def from_model(
        cls,
        model: OLS,
        k_regimes: int = 2,
        **kwargs: Any,
    ) -> MarkovRegression:
        """Create a MarkovRegression from an existing OLS model.

        Parameters
        ----------
        model : OLS
            Fitted or unfitted OLS model to convert.
        k_regimes : int
            Number of regimes.
        **kwargs
            Additional keyword arguments for MarkovRegression.

        Returns
        -------
        MarkovRegression
            New Markov switching model with matching specification.
        """
        # OLS already has exog with constant, so use trend="n"
        # unless there's no exog
        if model.exog is not None:
            return cls(
                endog=model.endog,
                k_regimes=k_regimes,
                exog=model.exog,
                trend="n",
                **kwargs,
            )
        return cls(
            endog=model.endog,
            k_regimes=k_regimes,
            trend="c",
            **kwargs,
        )


class MarkovAR:
    """Markov regime-switching autoregression.

    Wraps ``statsmodels.tsa.regime_switching.MarkovAutoregression`` with the
    regimes API.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    k_regimes : int
        Number of regimes. Default is 2.
    order : int
        AR order. Default is 1.
    exog : ArrayLike | None
        Exogenous regressors.
    trend : str
        Trend specification: "c" (default), "ct", or "n".
    switching_ar : bool
        Whether AR parameters switch. Default True.
    switching_trend : bool
        Whether trend switches. Default True.
    switching_variance : bool
        Whether error variance switches. Default False.
    ordering : str | None
        Regime ordering method. Default "first_appearance".

    Examples
    --------
    >>> import numpy as np
    >>> from regimes.markov import MarkovAR
    >>> np.random.seed(42)
    >>> y = np.zeros(200)
    >>> for t in range(1, 100):
    ...     y[t] = 0.3 * y[t-1] + np.random.randn()
    >>> for t in range(100, 200):
    ...     y[t] = 0.9 * y[t-1] + np.random.randn() * 0.5
    >>> model = MarkovAR(y, k_regimes=2, order=1)
    >>> results = model.fit()
    >>> print(results.summary())
    """

    def __init__(
        self,
        endog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        k_regimes: int = 2,
        order: int = 1,
        exog: ArrayLike | pd.Series[Any] | pd.DataFrame | None = None,
        trend: str = "c",
        switching_ar: bool = True,
        switching_trend: bool = True,
        switching_variance: bool = False,
        ordering: str | None = "first_appearance",
    ) -> None:
        self.endog = _ensure_1d(endog)
        self.exog = _ensure_2d(exog)
        self.k_regimes = k_regimes
        self.order = order
        self.trend = trend
        self.switching_ar = switching_ar
        self.switching_trend = switching_trend
        self.switching_variance = switching_variance
        self.ordering = ordering

    def fit(
        self,
        method: str = "bfgs",
        maxiter: int = 100,
        em_iter: int = 5,
        search_reps: int = 0,
        **kwargs: Any,
    ) -> MarkovARResults:
        """Fit the Markov switching AR model.

        Parameters
        ----------
        method : str
            Optimization method. Default "bfgs".
        maxiter : int
            Maximum iterations. Default 100.
        em_iter : int
            EM iterations before MLE. Default 5.
        search_reps : int
            Random starting value attempts. Default 0.
        **kwargs
            Additional arguments passed to statsmodels fit().

        Returns
        -------
        MarkovARResults
            Fitted model results.
        """
        sm_model = MarkovAutoregression(
            endog=self.endog,
            k_regimes=self.k_regimes,
            order=self.order,
            exog=self.exog,
            trend=self.trend,
            switching_ar=self.switching_ar,
            switching_trend=self.switching_trend,
            switching_variance=self.switching_variance,
        )

        sm_results = self._fit_with_search(
            sm_model, method, maxiter, em_iter, search_reps, **kwargs
        )

        return self._wrap_results(sm_results)

    def _fit_with_search(
        self,
        sm_model: Any,
        method: str,
        maxiter: int,
        em_iter: int,
        search_reps: int,
        **kwargs: Any,
    ) -> Any:
        """Fit with optional random starting value search."""
        best_results = None
        best_llf = -np.inf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                results = sm_model.fit(
                    method=method,
                    maxiter=maxiter,
                    em_iter=em_iter,
                    **kwargs,
                )
                if np.isfinite(results.llf) and results.llf > best_llf:
                    best_results = results
                    best_llf = results.llf
            except Exception:
                pass

        for _ in range(search_reps):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    results = sm_model.fit(
                        method=method,
                        maxiter=maxiter,
                        em_iter=0,
                        search_reps=1,
                        **kwargs,
                    )
                    if np.isfinite(results.llf) and results.llf > best_llf:
                        best_results = results
                        best_llf = results.llf
                except Exception:
                    continue

        if best_results is None:
            best_results = sm_model.fit(
                method=method,
                maxiter=maxiter,
                em_iter=0,
                **kwargs,
            )

        return best_results

    def _wrap_results(self, sm_results: Any) -> MarkovARResults:
        """Wrap statsmodels results into MarkovARResults."""
        k = self.k_regimes

        transition = np.array(sm_results.regime_transition).squeeze()
        if transition.ndim == 1:
            transition = transition.reshape(k, k)
        smoothed = np.array(sm_results.smoothed_marginal_probabilities)
        filtered = np.array(sm_results.filtered_marginal_probabilities)
        predicted = np.array(sm_results.predicted_marginal_probabilities)

        perm = _relabel_regimes(sm_results, self.ordering, k)
        param_names_list, params, bse, regime_params = _relabeled_parameters(
            sm_results, perm
        )
        transition, smoothed, filtered, predicted, _ = _apply_permutation(
            perm, k, transition, smoothed, filtered, predicted, {}
        )

        # Compute fitted values and residuals
        nobs_eff = len(smoothed)
        fittedvalues = np.zeros(nobs_eff)
        for j in range(k):
            try:
                regime_fv = np.array(
                    sm_results.predict(probabilities="regime", which=j)
                )
                fittedvalues += smoothed[:, j] * regime_fv
            except Exception:
                pass

        if np.allclose(fittedvalues, 0):
            try:
                fittedvalues = np.array(sm_results.predict())
            except Exception:
                fittedvalues = np.zeros(nobs_eff)

        # endog may be longer than nobs_eff due to AR lags
        endog_eff = self.endog[-nobs_eff:]
        resid = endog_eff - fittedvalues

        # AR coefficients per regime, under the same labels as everything else
        ar_params_dict: dict[int, NDArray[Any]] = {}
        for j in range(k):
            ar_params_dict[j] = np.array(
                [
                    regime_params[j][f"ar.L{lag}"]
                    for lag in range(1, self.order + 1)
                    if f"ar.L{lag}" in regime_params[j]
                ]
            )

        converged = getattr(sm_results, "mle_retvals", {}).get("converged", True)
        n_iterations = getattr(sm_results, "mle_retvals", {}).get("iterations", 0)
        if isinstance(converged, (int, np.integer)):
            converged = bool(converged)

        return MarkovARResults(
            params=params,
            nobs=nobs_eff,
            model_name="Markov Switching AR",
            k_regimes=k,
            regime_transition=transition,
            smoothed_marginal_probabilities=smoothed,
            filtered_marginal_probabilities=filtered,
            predicted_marginal_probabilities=predicted,
            bse=bse,
            llf=float(sm_results.llf),
            resid=resid,
            fittedvalues=fittedvalues,
            param_names=param_names_list,
            regime_params=regime_params,
            cov_type="approx",
            converged=converged,
            n_iterations=n_iterations,
            order=self.order,
            switching_ar=self.switching_ar,
            switching_trend=self.switching_trend,
            switching_variance=self.switching_variance,
            ar_params=ar_params_dict,
            _sm_results=sm_results,
        )

    @classmethod
    def from_model(
        cls,
        model: AR,
        k_regimes: int = 2,
        **kwargs: Any,
    ) -> MarkovAR:
        """Create a MarkovAR from an existing AR model.

        Parameters
        ----------
        model : AR
            Fitted or unfitted AR model to convert.
        k_regimes : int
            Number of regimes.
        **kwargs
            Additional keyword arguments for MarkovAR.

        Returns
        -------
        MarkovAR
            New Markov switching AR model.
        """
        order = max(model.lags) if model.lags else 1
        return cls(
            endog=model.endog,
            k_regimes=k_regimes,
            order=order,
            exog=model.exog,
            trend=model.trend,
            **kwargs,
        )


class MarkovADL:
    """Markov regime-switching Autoregressive Distributed Lag model.

    Since statsmodels has no MarkovADL, this builds the ADL design matrix
    (lagged y + current/lagged x) and passes it to statsmodels'
    MarkovRegression as exogenous regressors.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    exog : ArrayLike
        Exogenous regressors (n_obs, k). Required.
    k_regimes : int
        Number of regimes. Default is 2.
    ar_order : int
        AR order (p). Default is 1.
    exog_lags : int | dict[str | int, int | Sequence[int]]
        Distributed lags for exogenous variables. Default is 0
        (contemporaneous only).
    trend : str
        Trend specification. Default "c".
    switching_ar : bool
        Whether AR parameters switch. Default True.
    switching_exog : bool
        Whether exogenous coefficients switch. Default True.
    switching_variance : bool
        Whether error variance switches. Default False.
    ordering : str | None
        Regime ordering method. Default "first_appearance".

    Examples
    --------
    >>> import numpy as np
    >>> from regimes.markov import MarkovADL
    >>> np.random.seed(42)
    >>> n = 200
    >>> x = np.random.randn(n)
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = 0.5 * y[t-1] + 0.3 * x[t] + np.random.randn()
    >>> model = MarkovADL(y, x, k_regimes=2, ar_order=1, exog_lags=0)
    >>> results = model.fit()
    >>> print(results.summary())
    """

    def __init__(
        self,
        endog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        exog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        k_regimes: int = 2,
        ar_order: int = 1,
        exog_lags: int | dict[str | int, int | Sequence[int]] = 0,
        trend: str = "c",
        switching_ar: bool = True,
        switching_exog: bool = True,
        switching_variance: bool = False,
        ordering: str | None = "first_appearance",
    ) -> None:
        self.endog = _ensure_1d(endog)
        exog_arr = _ensure_2d(exog)
        if exog_arr is None:
            raise ValueError("exog is required for MarkovADL")
        self.exog = exog_arr
        self.k_regimes = k_regimes
        self.ar_order = ar_order
        self.exog_lags_raw = exog_lags
        self.trend = trend
        self.switching_ar = switching_ar
        self.switching_exog = switching_exog
        self.switching_variance = switching_variance
        self.ordering = ordering

        # Derive exog names
        if isinstance(exog, pd.DataFrame):
            self._exog_names = [str(c) for c in exog.columns]
        elif isinstance(exog, pd.Series):
            self._exog_names = [str(exog.name) if exog.name else "x0"]
        else:
            k_exog = self.exog.shape[1]
            self._exog_names = [f"x{i}" for i in range(k_exog)]

    def _process_exog_lags(self) -> dict[int, list[int]]:
        """Process exog_lags to normalized form."""
        k_exog = self.exog.shape[1]
        result: dict[int, list[int]] = {}

        if isinstance(self.exog_lags_raw, (int, np.integer)):
            for i in range(k_exog):
                result[i] = list(range(0, self.exog_lags_raw + 1))
        elif isinstance(self.exog_lags_raw, dict):
            for key, val in self.exog_lags_raw.items():
                if isinstance(key, (int, np.integer)):
                    idx = key
                else:
                    idx = self._exog_names.index(str(key))
                if isinstance(val, (int, np.integer)):
                    result[idx] = list(range(0, val + 1))
                else:
                    result[idx] = sorted(list(val))
            for i in range(k_exog):
                if i not in result:
                    result[i] = [0]
        else:
            for i in range(k_exog):
                result[i] = [0]

        return result

    @property
    def maxlag(self) -> int:
        """Maximum lag across AR and exog lags."""
        ar_max = self.ar_order
        exog_max = 0
        exog_lags_dict = self._process_exog_lags()
        for lags in exog_lags_dict.values():
            if lags:
                exog_max = max(exog_max, max(lags))
        return max(ar_max, exog_max)

    def _build_design_matrix(
        self,
    ) -> tuple[NDArray[Any], NDArray[Any], list[str]]:
        """Build effective y and X matrices with AR and distributed lags.

        Returns
        -------
        tuple[NDArray, NDArray, list[str]]
            Effective y, design matrix X, and parameter names.
        """
        ml = self.maxlag
        n = len(self.endog)
        n_eff = n - ml

        y_eff = self.endog[ml:]

        components = []
        param_names: list[str] = []

        # AR lags
        for lag in range(1, self.ar_order + 1):
            components.append(self.endog[ml - lag : n - lag])
            param_names.append(f"y.L{lag}")

        # Exogenous lags
        exog_lags_dict = self._process_exog_lags()
        for col_idx in sorted(exog_lags_dict.keys()):
            var_name = (
                self._exog_names[col_idx]
                if col_idx < len(self._exog_names)
                else f"x{col_idx}"
            )
            for lag in exog_lags_dict[col_idx]:
                if lag == 0:
                    components.append(self.exog[ml:, col_idx])
                    param_names.append(var_name)
                else:
                    components.append(self.exog[ml - lag : n - lag, col_idx])
                    param_names.append(f"{var_name}.L{lag}")

        if components:
            X = np.column_stack(components)
        else:
            X = np.empty((n_eff, 0))

        return y_eff, X, param_names

    def fit(
        self,
        method: str = "bfgs",
        maxiter: int = 100,
        em_iter: int = 5,
        search_reps: int = 0,
        **kwargs: Any,
    ) -> MarkovADLResults:
        """Fit the Markov switching ADL model.

        Parameters
        ----------
        method : str
            Optimization method. Default "bfgs".
        maxiter : int
            Maximum iterations. Default 100.
        em_iter : int
            EM iterations before MLE. Default 5.
        search_reps : int
            Random starting value attempts. Default 0.
        **kwargs
            Additional arguments passed to statsmodels fit().

        Returns
        -------
        MarkovADLResults
            Fitted model results.
        """
        y_eff, X, adl_param_names = self._build_design_matrix()

        # Determine switching for each exog column
        # AR params: first ar_order columns
        # Exog params: remaining columns
        n_ar = self.ar_order

        switching_exog_flags = []
        for i in range(X.shape[1]):
            if i < n_ar:
                switching_exog_flags.append(self.switching_ar)
            else:
                switching_exog_flags.append(self.switching_exog)

        sm_model = SMMarkovRegression(
            endog=y_eff,
            k_regimes=self.k_regimes,
            exog=X,
            trend=self.trend,
            switching_trend=True,
            switching_exog=switching_exog_flags,
            switching_variance=self.switching_variance,
        )

        sm_results = self._fit_with_search(
            sm_model, method, maxiter, em_iter, search_reps, **kwargs
        )

        return self._wrap_results(sm_results, adl_param_names)

    def _fit_with_search(
        self,
        sm_model: Any,
        method: str,
        maxiter: int,
        em_iter: int,
        search_reps: int,
        **kwargs: Any,
    ) -> Any:
        """Fit with optional random starting value search."""
        best_results = None
        best_llf = -np.inf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                results = sm_model.fit(
                    method=method,
                    maxiter=maxiter,
                    em_iter=em_iter,
                    **kwargs,
                )
                if np.isfinite(results.llf) and results.llf > best_llf:
                    best_results = results
                    best_llf = results.llf
            except Exception:
                pass

        for _ in range(search_reps):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    results = sm_model.fit(
                        method=method,
                        maxiter=maxiter,
                        em_iter=0,
                        search_reps=1,
                        **kwargs,
                    )
                    if np.isfinite(results.llf) and results.llf > best_llf:
                        best_results = results
                        best_llf = results.llf
                except Exception:
                    continue

        if best_results is None:
            best_results = sm_model.fit(
                method=method,
                maxiter=maxiter,
                em_iter=0,
                **kwargs,
            )

        return best_results

    def _wrap_results(
        self, sm_results: Any, adl_param_names: list[str]
    ) -> MarkovADLResults:
        """Wrap statsmodels results into MarkovADLResults."""
        k = self.k_regimes

        transition = np.array(sm_results.regime_transition).squeeze()
        if transition.ndim == 1:
            transition = transition.reshape(k, k)
        smoothed = np.array(sm_results.smoothed_marginal_probabilities)
        filtered = np.array(sm_results.filtered_marginal_probabilities)
        predicted = np.array(sm_results.predicted_marginal_probabilities)

        perm = _relabel_regimes(sm_results, self.ordering, k)
        param_names, params, bse, regime_params = _relabeled_parameters(
            sm_results, perm, exog_names=adl_param_names
        )
        transition, smoothed, filtered, predicted, _ = _apply_permutation(
            perm, k, transition, smoothed, filtered, predicted, {}
        )

        nobs_eff = len(smoothed)
        fittedvalues = np.zeros(nobs_eff)
        for j in range(k):
            try:
                regime_fv = np.array(
                    sm_results.predict(probabilities="regime", which=j)
                )
                fittedvalues += smoothed[:, j] * regime_fv
            except Exception:
                pass

        if np.allclose(fittedvalues, 0):
            try:
                fittedvalues = np.array(sm_results.predict())
            except Exception:
                fittedvalues = np.zeros(nobs_eff)

        y_eff = self.endog[-nobs_eff:]
        resid = y_eff - fittedvalues

        converged = getattr(sm_results, "mle_retvals", {}).get("converged", True)
        n_iterations = getattr(sm_results, "mle_retvals", {}).get("iterations", 0)
        if isinstance(converged, (int, np.integer)):
            converged = bool(converged)

        # Build exog_lags result dict
        exog_lags_dict = self._process_exog_lags()
        exog_lags_result: dict[str, list[int]] = {}
        for col_idx, lags in exog_lags_dict.items():
            var_name = (
                self._exog_names[col_idx]
                if col_idx < len(self._exog_names)
                else f"x{col_idx}"
            )
            exog_lags_result[var_name] = lags

        return MarkovADLResults(
            params=params,
            nobs=nobs_eff,
            model_name="Markov Switching ADL",
            k_regimes=k,
            regime_transition=transition,
            smoothed_marginal_probabilities=smoothed,
            filtered_marginal_probabilities=filtered,
            predicted_marginal_probabilities=predicted,
            bse=bse,
            llf=float(sm_results.llf),
            resid=resid,
            fittedvalues=fittedvalues,
            param_names=param_names,
            regime_params=regime_params,
            cov_type="approx",
            converged=converged,
            n_iterations=n_iterations,
            ar_order=self.ar_order,
            exog_lags=exog_lags_result,
            switching_ar=self.switching_ar,
            switching_exog=self.switching_exog,
            switching_variance=self.switching_variance,
            _sm_results=sm_results,
        )

    @classmethod
    def from_model(
        cls,
        model: ADL,
        k_regimes: int = 2,
        **kwargs: Any,
    ) -> MarkovADL:
        """Create a MarkovADL from an existing ADL model.

        Parameters
        ----------
        model : ADL
            Fitted or unfitted ADL model to convert.
        k_regimes : int
            Number of regimes.
        **kwargs
            Additional keyword arguments for MarkovADL.

        Returns
        -------
        MarkovADL
            New Markov switching ADL model.
        """
        ar_order = max(model.lags) if model.lags else 0
        instance = cls(
            endog=model.endog,
            exog=model.exog,
            k_regimes=k_regimes,
            ar_order=ar_order,
            exog_lags=model._exog_lags_raw,
            trend=model.trend,
            **kwargs,
        )
        # the model's names for its regressors (column names of a DataFrame),
        # which exog_lags may refer to
        instance._exog_names = list(model._exog_names)
        return instance
