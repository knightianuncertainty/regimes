"""Markov regime-switching models wrapping statsmodels.

This module provides MarkovRegression, MarkovAR, and MarkovADL classes
that wrap statsmodels' MarkovRegression and MarkovAutoregression with the
regimes API pattern: Model(...).fit() -> Results.

The one-way mapping from existing OLS/AR/ADL models is provided via
``from_model()`` class methods and ``markov_switching()`` convenience
methods on the original model classes.
"""

from __future__ import annotations

import warnings
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
        # Order by intercept/mean value (regime 0 = lowest mean)
        # Extract regime means from the smoothed probabilities
        try:
            regime_means = []
            for j in range(k_regimes):
                regime_means.append(sm_results.params[j])  # first param per regime
            sorted_indices = np.argsort(regime_means)
            perm = np.zeros(k_regimes, dtype=np.intp)
            for new_label, old_label in enumerate(sorted_indices):
                perm[old_label] = new_label
            return perm
        except (IndexError, AttributeError):
            return identity

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


def _extract_regime_params(
    sm_results: Any,
    k_regimes: int,
) -> dict[int, dict[str, float]]:
    """Extract per-regime parameters from statsmodels results."""
    regime_params: dict[int, dict[str, float]] = {}
    param_names = sm_results.model.param_names

    for j in range(k_regimes):
        params_j: dict[str, float] = {}
        for i, name in enumerate(param_names):
            # statsmodels uses names like "const[0]", "x1[1]", "sigma2[0]"
            if f"[{j}]" in name:
                clean_name = name.replace(f"[{j}]", "")
                params_j[clean_name] = float(sm_results.params[i])
            elif "[" not in name:
                # Non-switching parameter (same across regimes)
                params_j[name] = float(sm_results.params[i])
        regime_params[j] = params_j

    return regime_params


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
        - "intercept": Regime 0 has the lowest intercept/mean.
        - None: Keep statsmodels' arbitrary ordering.

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

        # Extract regime-specific parameters
        regime_params = _extract_regime_params(sm_results, k)

        # Apply regime relabeling
        perm = _relabel_regimes(sm_results, self.ordering, k)
        transition, smoothed, filtered, predicted, regime_params = _apply_permutation(
            perm, k, transition, smoothed, filtered, predicted, regime_params
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

        # Clean param names
        param_names = list(sm_results.model.param_names)

        # Get convergence info
        converged = getattr(sm_results, "mle_retvals", {}).get("converged", True)
        n_iterations = getattr(sm_results, "mle_retvals", {}).get("iterations", 0)
        if isinstance(converged, (int, np.integer)):
            converged = bool(converged)

        return MarkovRegressionResults(
            params=np.array(sm_results.params),
            nobs=len(self.endog),
            model_name="Markov Switching Regression",
            k_regimes=k,
            regime_transition=transition,
            smoothed_marginal_probabilities=smoothed,
            filtered_marginal_probabilities=filtered,
            predicted_marginal_probabilities=predicted,
            bse=np.array(sm_results.bse),
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

        regime_params = _extract_regime_params(sm_results, k)

        perm = _relabel_regimes(sm_results, self.ordering, k)
        transition, smoothed, filtered, predicted, regime_params = _apply_permutation(
            perm, k, transition, smoothed, filtered, predicted, regime_params
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

        # Extract AR params per regime
        ar_params_dict: dict[int, NDArray[Any]] = {}
        param_names_list = list(sm_results.model.param_names)
        for j in range(k):
            ar_coeffs = []
            for lag in range(1, self.order + 1):
                # Look for switching or non-switching AR params
                if self.switching_ar:
                    target = f"ar.L{lag}[{j}]"
                else:
                    target = f"ar.L{lag}"
                if target in param_names_list:
                    idx = param_names_list.index(target)
                    ar_coeffs.append(float(sm_results.params[idx]))
            if ar_coeffs:
                ar_params_dict[j] = np.array(ar_coeffs)
            else:
                ar_params_dict[j] = np.array([])

        converged = getattr(sm_results, "mle_retvals", {}).get("converged", True)
        n_iterations = getattr(sm_results, "mle_retvals", {}).get("iterations", 0)
        if isinstance(converged, (int, np.integer)):
            converged = bool(converged)

        return MarkovARResults(
            params=np.array(sm_results.params),
            nobs=nobs_eff,
            model_name="Markov Switching AR",
            k_regimes=k,
            regime_transition=transition,
            smoothed_marginal_probabilities=smoothed,
            filtered_marginal_probabilities=filtered,
            predicted_marginal_probabilities=predicted,
            bse=np.array(sm_results.bse),
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

        if isinstance(self.exog_lags_raw, int):
            for i in range(k_exog):
                result[i] = list(range(0, self.exog_lags_raw + 1))
        elif isinstance(self.exog_lags_raw, dict):
            for key, val in self.exog_lags_raw.items():
                if isinstance(key, int):
                    idx = key
                else:
                    idx = self._exog_names.index(str(key))
                if isinstance(val, int):
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

        regime_params = _extract_regime_params(sm_results, k)

        perm = _relabel_regimes(sm_results, self.ordering, k)
        transition, smoothed, filtered, predicted, regime_params = _apply_permutation(
            perm, k, transition, smoothed, filtered, predicted, regime_params
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

        param_names = list(sm_results.model.param_names)

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
            params=np.array(sm_results.params),
            nobs=nobs_eff,
            model_name="Markov Switching ADL",
            k_regimes=k,
            regime_transition=transition,
            smoothed_marginal_probabilities=smoothed,
            filtered_marginal_probabilities=filtered,
            predicted_marginal_probabilities=predicted,
            bse=np.array(sm_results.bse),
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
        return cls(
            endog=model.endog,
            exog=model.exog,
            k_regimes=k_regimes,
            ar_order=ar_order,
            exog_lags=model._exog_lags_raw,
            trend=model.trend,
            **kwargs,
        )
