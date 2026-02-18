"""CUSUM and CUSUM-SQ tests for parameter instability.

This module implements the CUSUM (Brown, Durbin, Evans, 1975) and CUSUM-of-squares
tests for detecting parameter instability in linear regression models. Unlike
Bai-Perron or Chow tests which estimate or test at specific break dates, CUSUM
tests produce a path of test statistics over time — visualization is the primary
output.

CUSUM detects mean shifts in parameters; CUSUM-SQ detects variance changes.

References
----------
Brown, R. L., Durbin, J., & Evans, J. M. (1975). Techniques for testing the
    constancy of regression relationships over time. Journal of the Royal
    Statistical Society: Series B, 37(2), 149-192.

Ploberger, W., & Krämer, W. (1992). The CUSUM test with OLS residuals.
    Econometrica, 60(2), 271-285.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from regimes.tests.base import BreakTestBase, BreakTestResultsBase

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from regimes.models.adl import ADL
    from regimes.models.ar import AR
    from regimes.models.ols import OLS


# CUSUM critical values for significance levels 0.01, 0.05, 0.10
# These come from the Brownian bridge distribution (Table in Brown et al., 1975)
_CUSUM_CRITICAL_VALUES: dict[float, float] = {
    0.01: 1.143,
    0.05: 0.948,
    0.10: 0.850,
}


def _compute_recursive_residuals(
    endog: NDArray[np.floating[Any]],
    exog: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Compute recursive residuals using Sherman-Morrison-Woodbury updating.

    Recursive residuals are one-step-ahead prediction errors from a
    recursively estimated regression. Under parameter stability, they
    are independently and identically distributed as N(0, σ²).

    Parameters
    ----------
    endog : NDArray
        Dependent variable of shape (T,).
    exog : NDArray
        Regressor matrix of shape (T, k), must include constant if desired.

    Returns
    -------
    w : NDArray
        Recursive residuals of shape (T - k,).
    f : NDArray
        Scaling factors of shape (T - k,), where
        f_t = 1 + x_t' (X_{t-1}' X_{t-1})^{-1} x_t.

    Notes
    -----
    Uses the matrix inversion lemma (Sherman-Morrison-Woodbury) for
    efficient O(Tk²) updating rather than O(T²k³) direct inversion.
    The recursive residual at time t is:

        w_t = (y_t - x_t' β_{t-1}) / sqrt(f_t)

    where β_{t-1} is the OLS estimate using observations 1, ..., t-1,
    and f_t = 1 + x_t' (X_{t-1}' X_{t-1})^{-1} x_t.
    """
    T, k = exog.shape

    if k >= T:
        raise ValueError(
            f"Need more observations ({T}) than regressors ({k}) "
            "to compute recursive residuals"
        )

    # Initialize with first k observations
    X_init = exog[:k]
    y_init = endog[:k]

    # (X'X)^{-1} for the first k observations
    XtX_inv = np.linalg.inv(X_init.T @ X_init)
    beta = XtX_inv @ (X_init.T @ y_init)

    n_residuals = T - k
    w = np.empty(n_residuals)
    f = np.empty(n_residuals)

    for t in range(k, T):
        x_t = exog[t]  # (k,)
        y_t = endog[t]

        # One-step-ahead prediction error
        prediction = x_t @ beta
        error = y_t - prediction

        # Scaling factor: f_t = 1 + x_t' (X'X)^{-1} x_t
        f_t = 1.0 + x_t @ XtX_inv @ x_t
        f[t - k] = f_t

        # Recursive residual (standardized)
        w[t - k] = error / np.sqrt(f_t)

        # Sherman-Morrison update of (X'X)^{-1}
        # (X'X + x x')^{-1} = (X'X)^{-1} - (X'X)^{-1} x x' (X'X)^{-1} / f_t
        v = XtX_inv @ x_t  # (k,)
        XtX_inv = XtX_inv - np.outer(v, v) / f_t

        # Update beta: β_t = β_{t-1} + (X'X)^{-1}_t * x_t * w_t * sqrt(f_t)
        # Equivalently: β_t = β_{t-1} + (X'X)_t^{-1} x_t (y_t - x_t' β_{t-1})
        beta = beta + XtX_inv @ x_t * error

    return w, f


@dataclass
class CUSUMResults(BreakTestResultsBase):
    """Results from the CUSUM test for parameter instability.

    The CUSUM test computes the cumulative sum of recursive residuals,
    normalized by the estimated standard deviation. Under the null of
    parameter stability, the statistic stays within expanding boundaries
    derived from the Brownian bridge distribution.

    Attributes
    ----------
    test_name : str
        "CUSUM".
    nobs : int
        Number of observations.
    n_breaks : int
        Number of boundary crossings (0 if stable, 1+ if unstable).
    break_indices : Sequence[int]
        Indices where statistic first crosses the boundary.
    statistic_path : NDArray
        The CUSUM statistic W_r at each time point, shape (T - k,).
    upper_bound : NDArray
        Upper critical boundary at each time point.
    lower_bound : NDArray
        Lower critical boundary at each time point.
    recursive_residuals : NDArray
        The raw recursive residuals w_t.
    sigma_hat : float
        Estimated standard deviation of recursive residuals.
    significance : float
        Significance level used for critical boundaries.
    reject : bool
        Whether the null of parameter stability is rejected.
    n_params : int
        Number of regressors (k).
    _test : CUSUMTest | None
        Back-reference to the test object.
    """

    statistic_path: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.array([])
    )
    upper_bound: NDArray[np.floating[Any]] = field(default_factory=lambda: np.array([]))
    lower_bound: NDArray[np.floating[Any]] = field(default_factory=lambda: np.array([]))
    recursive_residuals: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.array([])
    )
    sigma_hat: float = np.nan
    significance: float = 0.05
    reject: bool = False
    n_params: int = 0
    _test: CUSUMTest | None = field(default=None, repr=False)

    @property
    def is_stable(self) -> bool:
        """Whether the null of parameter stability is not rejected."""
        return not self.reject

    @property
    def max_statistic(self) -> float:
        """Maximum absolute value of the CUSUM statistic."""
        if len(self.statistic_path) == 0:
            return 0.0
        return float(np.max(np.abs(self.statistic_path)))

    @property
    def crossing_indices(self) -> list[int]:
        """Indices where the statistic crosses the critical boundary."""
        if len(self.statistic_path) == 0:
            return []
        above = self.statistic_path > self.upper_bound
        below = self.statistic_path < self.lower_bound
        crossed = np.where(above | below)[0]
        return crossed.tolist()

    def plot(
        self,
        ax: Any = None,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """Plot the CUSUM test statistic with critical boundaries.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None
            Axes to plot on. If None, creates a new figure.
        **kwargs
            Additional arguments passed to ``plot_cusum``.

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib figure and axes.
        """
        from regimes.visualization.cusum import plot_cusum

        return plot_cusum(self, ax=ax, **kwargs)

    def summary(self) -> str:
        """Generate a text summary of CUSUM test results.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = []
        lines.append("=" * 78)
        lines.append(f"{'CUSUM Test for Parameter Instability':^78}")
        lines.append("=" * 78)
        lines.append(f"Number of observations:   {self.nobs:>10}")
        lines.append(f"Number of regressors:     {self.n_params:>10}")
        lines.append(f"Recursive residuals:      {len(self.recursive_residuals):>10}")
        lines.append(f"Significance level:       {self.significance:>10.3f}")
        lines.append(f"Sigma hat:                {self.sigma_hat:>10.4f}")
        lines.append("-" * 78)
        lines.append(f"Max |CUSUM statistic|:    {self.max_statistic:>10.4f}")

        if self.significance in _CUSUM_CRITICAL_VALUES:
            cv = _CUSUM_CRITICAL_VALUES[self.significance]
            lines.append(f"Critical value (a):       {cv:>10.4f}")

        decision = "REJECT" if self.reject else "Do not reject"
        lines.append(f"\nDecision:  {decision} the null of parameter stability")

        if self.reject and self.break_indices:
            lines.append(
                f"First boundary crossing at recursive residual index: "
                f"{self.break_indices[0]}"
            )

        lines.append("=" * 78)
        return "\n".join(lines)


@dataclass
class CUSUMSQResults(BreakTestResultsBase):
    """Results from the CUSUM-of-squares test for variance instability.

    The CUSUM-SQ test computes the cumulative sum of squared recursive
    residuals, normalized to [0, 1]. Under the null, this path tracks
    the diagonal. Critical bounds come from the Kolmogorov-Smirnov
    distribution.

    Attributes
    ----------
    test_name : str
        "CUSUM-SQ".
    nobs : int
        Number of observations.
    n_breaks : int
        Number of boundary crossings.
    break_indices : Sequence[int]
        Indices where statistic first crosses the boundary.
    statistic_path : NDArray
        The CUSUM-SQ statistic S_r at each time point, shape (T - k,).
    expected_path : NDArray
        The expected path under H0 (the diagonal), shape (T - k,).
    upper_bound : NDArray
        Upper critical boundary.
    lower_bound : NDArray
        Lower critical boundary.
    recursive_residuals : NDArray
        The raw recursive residuals w_t.
    significance : float
        Significance level used.
    reject : bool
        Whether the null of stability is rejected.
    n_params : int
        Number of regressors (k).
    _test : CUSUMSQTest | None
        Back-reference to the test object.
    """

    statistic_path: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.array([])
    )
    expected_path: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.array([])
    )
    upper_bound: NDArray[np.floating[Any]] = field(default_factory=lambda: np.array([]))
    lower_bound: NDArray[np.floating[Any]] = field(default_factory=lambda: np.array([]))
    recursive_residuals: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.array([])
    )
    significance: float = 0.05
    reject: bool = False
    n_params: int = 0
    _test: CUSUMSQTest | None = field(default=None, repr=False)

    @property
    def is_stable(self) -> bool:
        """Whether the null of parameter stability is not rejected."""
        return not self.reject

    @property
    def max_deviation(self) -> float:
        """Maximum absolute deviation of the statistic from the expected path."""
        if len(self.statistic_path) == 0:
            return 0.0
        return float(np.max(np.abs(self.statistic_path - self.expected_path)))

    @property
    def crossing_indices(self) -> list[int]:
        """Indices where the statistic crosses the critical boundary."""
        if len(self.statistic_path) == 0:
            return []
        above = self.statistic_path > self.upper_bound
        below = self.statistic_path < self.lower_bound
        crossed = np.where(above | below)[0]
        return crossed.tolist()

    def plot(
        self,
        ax: Any = None,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """Plot the CUSUM-SQ test statistic with critical boundaries.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None
            Axes to plot on. If None, creates a new figure.
        **kwargs
            Additional arguments passed to ``plot_cusum_sq``.

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib figure and axes.
        """
        from regimes.visualization.cusum import plot_cusum_sq

        return plot_cusum_sq(self, ax=ax, **kwargs)

    def summary(self) -> str:
        """Generate a text summary of CUSUM-SQ test results.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = []
        lines.append("=" * 78)
        lines.append(f"{'CUSUM-SQ Test for Variance Instability':^78}")
        lines.append("=" * 78)
        lines.append(f"Number of observations:   {self.nobs:>10}")
        lines.append(f"Number of regressors:     {self.n_params:>10}")
        lines.append(f"Recursive residuals:      {len(self.recursive_residuals):>10}")
        lines.append(f"Significance level:       {self.significance:>10.3f}")
        lines.append("-" * 78)
        lines.append(f"Max |S_r - E[S_r]|:       {self.max_deviation:>10.4f}")

        # KS critical value
        n = len(self.recursive_residuals)
        if n > 0:
            cv = stats.ksone.ppf(1.0 - self.significance / 2, n)
            lines.append(f"KS critical value:        {cv:>10.4f}")

        decision = "REJECT" if self.reject else "Do not reject"
        lines.append(f"\nDecision:  {decision} the null of variance stability")

        if self.reject and self.break_indices:
            lines.append(
                f"First boundary crossing at recursive residual index: "
                f"{self.break_indices[0]}"
            )

        lines.append("=" * 78)
        return "\n".join(lines)


class CUSUMTest(BreakTestBase):
    """CUSUM test for parameter instability (Brown, Durbin, Evans, 1975).

    Tests the null hypothesis of parameter stability by examining the
    cumulative sum of recursive residuals. Under H0, the cumulated sum
    stays within expanding boundaries derived from the Brownian bridge
    distribution. If the path crosses the boundary, we reject stability.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (T,).
    exog : ArrayLike | None
        Regressor matrix. If None, defaults to a constant (intercept-only
        model). Should include a constant column if desired.

    Examples
    --------
    >>> import numpy as np
    >>> from regimes import CUSUMTest
    >>> rng = np.random.default_rng(42)
    >>> y = np.concatenate([rng.normal(0, 1, 100), rng.normal(2, 1, 100)])
    >>> test = CUSUMTest(y)
    >>> results = test.fit(significance=0.05)
    >>> print(results.summary())

    Notes
    -----
    The CUSUM statistic at time r is:

        W_r = (1 / sigma_hat) * sum_{t=k+1}^{r} w_t

    where w_t are the recursive residuals and sigma_hat is their estimated
    standard deviation. The critical boundaries are:

        ±a * sqrt(T - k) ± 2a * (r - k) / sqrt(T - k)

    where a is the critical value from the Brownian bridge distribution
    at the chosen significance level.

    References
    ----------
    Brown, R. L., Durbin, J., & Evans, J. M. (1975). Techniques for testing
        the constancy of regression relationships over time. JRSS-B, 37(2).
    """

    def __init__(
        self,
        endog: ArrayLike,
        exog: ArrayLike | None = None,
        exog_break: ArrayLike | None = None,
    ) -> None:
        """Initialize the CUSUM test."""
        endog_arr = np.asarray(endog, dtype=np.float64)

        # Default to constant if no regressors specified
        if exog is None and exog_break is None:
            exog = np.ones((len(endog_arr), 1))

        # CUSUM doesn't distinguish exog vs exog_break — combine them
        if exog is not None and exog_break is not None:
            exog_arr = np.asarray(exog, dtype=np.float64)
            exog_break_arr = np.asarray(exog_break, dtype=np.float64)
            if exog_arr.ndim == 1:
                exog_arr = exog_arr.reshape(-1, 1)
            if exog_break_arr.ndim == 1:
                exog_break_arr = exog_break_arr.reshape(-1, 1)
            combined = np.column_stack([exog_arr, exog_break_arr])
            super().__init__(endog_arr, exog=combined)
        elif exog_break is not None:
            super().__init__(endog_arr, exog=exog_break)
        else:
            super().__init__(endog_arr, exog=exog)

        # Store the full regressor matrix for computation
        self._exog = (
            self.exog if self.exog is not None else np.ones((len(endog_arr), 1))
        )

    @classmethod
    def from_model(
        cls,
        model: OLS | AR | ADL,
    ) -> CUSUMTest:
        """Create CUSUMTest from an OLS, AR, or ADL model.

        Parameters
        ----------
        model : OLS | AR | ADL
            Model to test for parameter instability.

        Returns
        -------
        CUSUMTest
            Test instance ready for ``.fit()``.

        Notes
        -----
        For AR and ADL models, the test uses the effective sample (after
        dropping initial observations for lags).
        """
        from regimes.models.adl import ADL as ADLModel
        from regimes.models.ar import AR as ARModel
        from regimes.models.ols import OLS as OLSModel

        if isinstance(model, (ARModel, ADLModel)):
            y, X, _ = model._build_design_matrix()
            return cls(y, exog=X)
        elif isinstance(model, OLSModel):
            endog = model.endog
            exog = model.exog
            return cls(endog, exog=exog)
        else:
            raise TypeError(
                f"model must be OLS, AR, or ADL, got {type(model).__name__}"
            )

    @property
    def n_params(self) -> int:
        """Number of regressors (k)."""
        return self._exog.shape[1]

    def fit(
        self,
        significance: float = 0.05,
        **kwargs: Any,
    ) -> CUSUMResults:
        """Perform the CUSUM test.

        Parameters
        ----------
        significance : float
            Significance level. Must be one of 0.01, 0.05, 0.10.

        Returns
        -------
        CUSUMResults
            Results object with test statistic path, boundaries,
            and rejection decision.

        Raises
        ------
        ValueError
            If significance is not one of the supported levels.
        """
        if significance not in _CUSUM_CRITICAL_VALUES:
            raise ValueError(
                f"significance must be one of {sorted(_CUSUM_CRITICAL_VALUES.keys())}, "
                f"got {significance}"
            )

        T = self.nobs
        k = self.n_params

        # Compute recursive residuals
        w, _f = _compute_recursive_residuals(self.endog, self._exog)
        n = len(w)  # T - k

        # Estimate sigma from recursive residuals
        sigma_hat = float(np.sqrt(np.sum(w**2) / n))

        # CUSUM statistic: W_r = cumsum(w) / sigma_hat
        if sigma_hat > 0:
            statistic_path = np.cumsum(w) / sigma_hat
        else:
            statistic_path = np.zeros(n)

        # Critical boundaries (diverging lines from Brownian bridge)
        a = _CUSUM_CRITICAL_VALUES[significance]
        r = np.arange(1, n + 1)  # r = 1, 2, ..., T-k
        sqrt_n = np.sqrt(n)
        upper_bound = a * sqrt_n + 2 * a * r / sqrt_n
        lower_bound = -(a * sqrt_n + 2 * a * r / sqrt_n)

        # Check for boundary crossings
        above = statistic_path > upper_bound
        below = statistic_path < lower_bound
        crossed = np.where(above | below)[0]
        reject = len(crossed) > 0

        # Break indices: first crossing point
        break_indices: list[int] = []
        if reject:
            break_indices = [int(crossed[0])]

        return CUSUMResults(
            test_name="CUSUM",
            nobs=T,
            n_breaks=len(break_indices),
            break_indices=break_indices,
            statistic_path=statistic_path,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            recursive_residuals=w,
            sigma_hat=sigma_hat,
            significance=significance,
            reject=reject,
            n_params=k,
            _test=self,
        )


class CUSUMSQTest(BreakTestBase):
    """CUSUM-of-squares test for variance instability.

    Tests the null hypothesis of constant variance by examining the
    cumulative sum of squared recursive residuals. Under H0, the
    normalized path tracks the diagonal line. Critical bounds come
    from the Kolmogorov-Smirnov distribution.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (T,).
    exog : ArrayLike | None
        Regressor matrix. If None, defaults to a constant.

    Examples
    --------
    >>> import numpy as np
    >>> from regimes import CUSUMSQTest
    >>> rng = np.random.default_rng(42)
    >>> y = np.concatenate([rng.normal(0, 1, 100), rng.normal(0, 3, 100)])
    >>> test = CUSUMSQTest(y)
    >>> results = test.fit(significance=0.05)
    >>> print(results.summary())

    Notes
    -----
    The CUSUM-SQ statistic at time r is:

        S_r = Σ_{t=k+1}^{r} w_t² / Σ_{t=k+1}^{T} w_t²

    Under H0, E[S_r] = (r - k) / (T - k). The critical boundaries are
    parallel lines around this diagonal:

        E[S_r] +/- c_alpha

    where c_alpha is the critical value from the Kolmogorov-Smirnov
    distribution.

    References
    ----------
    Brown, R. L., Durbin, J., & Evans, J. M. (1975). Techniques for testing
        the constancy of regression relationships over time. JRSS-B, 37(2).
    """

    def __init__(
        self,
        endog: ArrayLike,
        exog: ArrayLike | None = None,
        exog_break: ArrayLike | None = None,
    ) -> None:
        """Initialize the CUSUM-SQ test."""
        endog_arr = np.asarray(endog, dtype=np.float64)

        # Default to constant if no regressors specified
        if exog is None and exog_break is None:
            exog = np.ones((len(endog_arr), 1))

        # Combine exog and exog_break like CUSUM
        if exog is not None and exog_break is not None:
            exog_arr = np.asarray(exog, dtype=np.float64)
            exog_break_arr = np.asarray(exog_break, dtype=np.float64)
            if exog_arr.ndim == 1:
                exog_arr = exog_arr.reshape(-1, 1)
            if exog_break_arr.ndim == 1:
                exog_break_arr = exog_break_arr.reshape(-1, 1)
            combined = np.column_stack([exog_arr, exog_break_arr])
            super().__init__(endog_arr, exog=combined)
        elif exog_break is not None:
            super().__init__(endog_arr, exog=exog_break)
        else:
            super().__init__(endog_arr, exog=exog)

        self._exog = (
            self.exog if self.exog is not None else np.ones((len(endog_arr), 1))
        )

    @classmethod
    def from_model(
        cls,
        model: OLS | AR | ADL,
    ) -> CUSUMSQTest:
        """Create CUSUMSQTest from an OLS, AR, or ADL model.

        Parameters
        ----------
        model : OLS | AR | ADL
            Model to test for variance instability.

        Returns
        -------
        CUSUMSQTest
            Test instance ready for ``.fit()``.
        """
        from regimes.models.adl import ADL as ADLModel
        from regimes.models.ar import AR as ARModel
        from regimes.models.ols import OLS as OLSModel

        if isinstance(model, (ARModel, ADLModel)):
            y, X, _ = model._build_design_matrix()
            return cls(y, exog=X)
        elif isinstance(model, OLSModel):
            endog = model.endog
            exog = model.exog
            return cls(endog, exog=exog)
        else:
            raise TypeError(
                f"model must be OLS, AR, or ADL, got {type(model).__name__}"
            )

    @property
    def n_params(self) -> int:
        """Number of regressors (k)."""
        return self._exog.shape[1]

    def fit(
        self,
        significance: float = 0.05,
        **kwargs: Any,
    ) -> CUSUMSQResults:
        """Perform the CUSUM-of-squares test.

        Parameters
        ----------
        significance : float
            Significance level for the Kolmogorov-Smirnov critical value.
            Any value in (0, 1) is supported.

        Returns
        -------
        CUSUMSQResults
            Results object with test statistic path, boundaries,
            and rejection decision.

        Raises
        ------
        ValueError
            If significance is not in (0, 1).
        """
        if not (0 < significance < 1):
            raise ValueError(f"significance must be in (0, 1), got {significance}")

        T = self.nobs
        k = self.n_params

        # Compute recursive residuals
        w, _f = _compute_recursive_residuals(self.endog, self._exog)
        n = len(w)  # T - k

        # CUSUM-SQ statistic: S_r = cumsum(w²) / sum(w²)
        w_sq = w**2
        total_w_sq = np.sum(w_sq)

        if total_w_sq > 0:
            statistic_path = np.cumsum(w_sq) / total_w_sq
        else:
            statistic_path = np.zeros(n)

        # Expected path under H0: (r - k) / (T - k) = r / n for r = 1, ..., n
        # (using local indexing where r goes from 1 to n)
        expected_path = np.arange(1, n + 1) / n

        # Critical value from Kolmogorov-Smirnov distribution
        c_alpha = float(stats.ksone.ppf(1.0 - significance / 2, n))

        # Boundaries: expected ± c_alpha
        upper_bound = expected_path + c_alpha
        lower_bound = expected_path - c_alpha

        # Clamp lower bound at 0 (S_r is always >= 0)
        lower_bound = np.maximum(lower_bound, 0.0)

        # Check for boundary crossings
        above = statistic_path > upper_bound
        below = statistic_path < lower_bound
        crossed = np.where(above | below)[0]
        reject = len(crossed) > 0

        break_indices: list[int] = []
        if reject:
            break_indices = [int(crossed[0])]

        return CUSUMSQResults(
            test_name="CUSUM-SQ",
            nobs=T,
            n_breaks=len(break_indices),
            break_indices=break_indices,
            statistic_path=statistic_path,
            expected_path=expected_path,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            recursive_residuals=w,
            significance=significance,
            reject=reject,
            n_params=k,
            _test=self,
        )
