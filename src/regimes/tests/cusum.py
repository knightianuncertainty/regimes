"""CUSUM and CUSUM-SQ tests for parameter instability.

This module implements the CUSUM and CUSUM-of-squares tests of Brown, Durbin
and Evans (1975) for parameter instability in linear regression models. Both
are built on the recursive residuals. Unlike the Bai-Perron or Chow tests,
which estimate or test break dates, they produce a path of statistics over
time, and the plot of the path against its bounds is the primary output.

CUSUM is sensitive to changes in the regression coefficients; CUSUM-SQ to
changes in the error variance.

What the tests report. A path that leaves its bounds rejects the null of
stability. The observation at which it first does so is reported as
``first_crossing``; it is *not* an estimate of the date of the change, which
can lie well before it (CUSUM) or after it (CUSUM-SQ). The tests therefore
report no break date: ``n_breaks`` is 0 and ``break_indices`` is empty.

Bounds.

* CUSUM: a pair of straight lines through ``(k, +/- a sqrt(T - k))`` and
  ``(T, +/- 3 a sqrt(T - k))``, with ``a`` chosen so that the probability
  that a Brownian motion, the limit of the cumulated recursive residuals,
  crosses either line is the significance level: ``a = 1.143, 0.948, 0.850``
  for 1, 5 and 10 percent (Brown, Durbin and Evans, 1975, Section 2.3). The
  test is somewhat conservative.
* CUSUM-SQ: the lines ``(r - k) / (T - k) +/- c_0``. Under the null and
  normal errors the path behaves like the ordered values of a uniform
  sample of size ``n = (T - k) / 2 - 1``, and ``c_0`` is the critical value
  of the modified Kolmogorov-Smirnov statistic for that ``n`` (Durbin, 1969,
  Table 1). It is computed with the approximation of Edgerton and Wells
  (1994), ``c_0 = a_1 / sqrt(n) + a_2 / n + a_3 / n^1.5``, as statsmodels
  does (``RecursiveLS``), and is available for the two-sided levels 20, 10,
  5, 2 and 1 percent and for at least 20 recursive residuals. The path is
  compared with the bounds at every observation, as in statsmodels, which
  makes the test slightly liberal in small samples (about 6 percent at a
  nominal 5 with 60 observations).

The scale estimate of the CUSUM test is ``sqrt(sum w_t^2 / (T - k))``
(uncentered); statsmodels centers the recursive residuals first, which moves
the size of the test by a few tenths of a percentage point.

References
----------
Brown, R. L., Durbin, J., & Evans, J. M. (1975). Techniques for testing the
    constancy of regression relationships over time. Journal of the Royal
    Statistical Society: Series B, 37(2), 149-192.

Durbin, J. (1969). Tests for serial correlation in regression analysis based
    on the periodogram of least-squares residuals. Biometrika, 56(1), 1-15.

Edgerton, D., & Wells, C. (1994). Critical values for the CUSUMSQ statistic
    in medium and large sized samples. Oxford Bulletin of Economics and
    Statistics, 56(3), 355-365.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from regimes.tests._bp_core import _orthonormal
from regimes.tests.base import BreakTestBase, BreakTestResultsBase

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from regimes.models.adl import ADL
    from regimes.models.ar import AR
    from regimes.models.ols import OLS


# Constant a of the CUSUM boundary lines for significance levels 0.01, 0.05,
# 0.10: the probability that a Brownian motion crosses either line is the
# significance level (Brown, Durbin and Evans, 1975, Section 2.3).
_CUSUM_CRITICAL_VALUES: dict[float, float] = {
    0.01: 1.143,
    0.05: 0.948,
    0.10: 0.850,
}

# Edgerton and Wells (1994): c_0 = a1 / sqrt(n) + a2 / n + a3 / n^1.5 with
# n = (T - k) / 2 - 1 approximates the critical values of Durbin (1969) for
# the CUSUM-of-squares test. Keys are two-sided significance levels (twice the
# one-sided levels 0.10, 0.05, 0.025, 0.01, 0.005 of the table). The
# coefficients are those in statsmodels' ``RecursiveLS``.
_CUSUMSQ_COEFFICIENTS: dict[float, tuple[float, float, float]] = {
    0.20: (1.0729830, -0.6698868, -0.5816458),
    0.10: (1.2238734, -0.6700069, -0.7351697),
    0.05: (1.3581015, -0.6701218, -0.8858694),
    0.02: (1.5174271, -0.6702672, -1.0847745),
    0.01: (1.6276236, -0.6703724, -1.2365861),
}

# Below this number of recursive residuals the approximation of Edgerton and
# Wells is poor: in simulations under the null the test at a nominal 5 percent
# rejects 8 percent of the time with 16 residuals, 12 with 10, and almost
# always with 5.
MIN_CUSUMSQ_RESIDUALS = 20


def cusum_sq_critical_value(significance: float, n_residuals: int) -> float:
    """Half-width ``c_0`` of the CUSUM-of-squares bounds.

    Parameters
    ----------
    significance : float
        Two-sided significance level: 0.20, 0.10, 0.05, 0.02 or 0.01.
    n_residuals : int
        Number of recursive residuals, ``T - k``.

    Returns
    -------
    float
        ``c_0`` from the approximation of Edgerton and Wells (1994) to the
        table of Durbin (1969), with ``n = (T - k) / 2 - 1``.

    Raises
    ------
    ValueError
        For a level that is not tabulated, or fewer than 20 recursive
        residuals.
    """
    key = _match_level(significance, _CUSUMSQ_COEFFICIENTS)
    if key is None:
        raise ValueError(
            "significance must be one of "
            f"{sorted(_CUSUMSQ_COEFFICIENTS)}, got {significance}"
        )
    if n_residuals < MIN_CUSUMSQ_RESIDUALS:
        raise ValueError(
            f"the CUSUM-of-squares bounds are an approximation for medium and "
            f"large samples; with {n_residuals} recursive residuals (fewer "
            f"than {MIN_CUSUMSQ_RESIDUALS}) it is not accurate and the test "
            "is not computed"
        )
    n = 0.5 * n_residuals - 1.0
    a1, a2, a3 = _CUSUMSQ_COEFFICIENTS[key]
    return float(a1 / np.sqrt(n) + a2 / n + a3 / n**1.5)


def _match_level(significance: float, table: dict[float, Any]) -> float | None:
    """The key of ``table`` equal to ``significance``, up to rounding."""
    for key in table:
        if abs(key - significance) < 1e-12:
            return key
    return None


def _compute_recursive_residuals(
    endog: NDArray[np.floating[Any]],
    exog: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Compute recursive residuals.

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
        Recursive residuals of shape (T - m,), where ``m`` is the number of
        observations used to start the recursion (see Notes); the first
        residual belongs to observation ``m``.
    f : NDArray
        Scaling factors of shape (T - m,), where
        f_t = 1 + x_t' (X_{t-1}' X_{t-1})^{-1} x_t.

    Raises
    ------
    ValueError
        If the regressors are collinear in the whole sample, or the
        coefficients are identified too late for a single residual.

    Notes
    -----
    The recursive residual at time t is

        w_t = (y_t - x_t' β_{t-1}) / sqrt(f_t)

    where β_{t-1} is the OLS estimate using observations 1, ..., t-1.

    The recursion is run on an orthonormal basis of the regressors, which
    spans the same space and so gives the same residuals, but keeps the
    computation accurate when the regressors differ in scale or vary little
    around a large mean (a trend in calendar years, a series in levels).

    It starts from the first ``m >= k`` observations that identify the
    ``k`` coefficients. Usually ``m = k``. It is larger when the regressors
    are collinear at the start of the sample: a series whose first values
    are equal (an AR model of simulated data that starts at zero), or a
    dummy variable that is constant at the start.
    """
    T, k = exog.shape

    if k >= T:
        raise ValueError(
            f"Need more observations ({T}) than regressors ({k}) "
            "to compute recursive residuals"
        )

    basis = _orthonormal(np.asarray(exog, dtype=np.float64))
    if basis is exog or np.linalg.matrix_rank(basis) < k:
        raise ValueError(
            "the regressors are collinear in the sample, so the recursive "
            "residuals are not defined"
        )
    start = _recursion_start(basis)

    n_residuals = T - start
    w = np.empty(n_residuals)
    f = np.empty(n_residuals)

    for t in range(start, T):
        # least squares on observations 0, ..., t - 1 through a QR
        # decomposition, which stays accurate when the first observations
        # barely identify the coefficients (updating an inverse would carry
        # the error of that first inverse through the whole sample)
        _q, r = np.linalg.qr(basis[:t])
        beta = np.linalg.solve(r, _q.T @ endog[:t])
        x_t = basis[t]
        v = np.linalg.solve(r.T, x_t)

        # f_t = 1 + x_t' (X'X)^{-1} x_t, with X'X = R'R
        f_t = 1.0 + v @ v
        f[t - start] = f_t
        w[t - start] = (endog[t] - x_t @ beta) / np.sqrt(f_t)

    return w, f


# The first observations identify the coefficients when their regressor
# matrix has a condition number below this (the moment matrix that is
# inverted has the square of it).
_START_CONDITION = 1e6


def _recursion_start(basis: NDArray[np.floating[Any]]) -> int:
    """Number of initial observations that identify the coefficients."""
    T, k = basis.shape
    for m in range(k, T):
        singular = np.linalg.svd(basis[:m], compute_uv=False)
        if singular[-1] > 0 and singular[0] / singular[-1] < _START_CONDITION:
            return m
    raise ValueError(
        f"the {k} coefficients are not identified before the last "
        "observation (the regressors are collinear until then), so no "
        "recursive residual can be computed"
    )


@dataclass
class CUSUMResults(BreakTestResultsBase):
    """Results from the CUSUM test for parameter instability.

    The CUSUM test computes the cumulative sum of recursive residuals,
    normalized by the estimated standard deviation. Under the null of
    parameter stability, the statistic stays within a pair of diverging
    straight lines with the chosen probability (see the module docstring).

    Attributes
    ----------
    test_name : str
        "CUSUM".
    nobs : int
        Number of observations.
    n_breaks : int
        Always 0: the test does not estimate a break date.
    break_indices : Sequence[int]
        Always empty; see ``first_crossing``.
    first_crossing : int | None
        Observation (0-based, in the sample the test was given) at which
        the path first leaves the bounds, or None. This shows when the
        evidence against stability became significant, not when the
        coefficients changed.
    observations : NDArray
        Observation number of each point of the path: ``m, ..., T - 1``,
        where ``m`` is the number of observations that start the recursion
        (the number of regressors ``k``, unless the regressors are collinear
        at the start of the sample).
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
    first_crossing: int | None = None
    _test: CUSUMTest | None = field(default=None, repr=False)

    @property
    def is_stable(self) -> bool:
        """Whether the null of parameter stability is not rejected."""
        return not self.reject

    @property
    def observations(self) -> NDArray[np.intp]:
        """Observation number of each point of the path."""
        start = self.nobs - len(self.statistic_path)
        return np.arange(start, self.nobs)

    @property
    def max_statistic(self) -> float:
        """Maximum absolute value of the CUSUM statistic."""
        if len(self.statistic_path) == 0:
            return 0.0
        return float(np.max(np.abs(self.statistic_path)))

    @property
    def max_ratio(self) -> float:
        """Largest ratio of the path to its bound; above 1 the test rejects."""
        if len(self.statistic_path) == 0:
            return 0.0
        return float(np.max(np.abs(self.statistic_path) / self.upper_bound))

    @property
    def crossing_indices(self) -> list[int]:
        """Observations at which the statistic is outside the bounds."""
        if len(self.statistic_path) == 0:
            return []
        above = self.statistic_path > self.upper_bound
        below = self.statistic_path < self.lower_bound
        crossed = np.where(above | below)[0] + self.nobs - len(self.statistic_path)
        return [int(i) for i in crossed]

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
        lines.append(f"Max |CUSUM| / bound:      {self.max_ratio:>10.4f}")
        lines.append("  (the test rejects if the ratio exceeds 1)")

        key = _match_level(self.significance, _CUSUM_CRITICAL_VALUES)
        if key is not None:
            cv = _CUSUM_CRITICAL_VALUES[key]
            lines.append(f"Boundary constant (a):    {cv:>10.4f}")

        decision = "REJECT" if self.reject else "Do not reject"
        lines.append(f"\nDecision:  {decision} the null of parameter stability")

        if self.reject and self.first_crossing is not None:
            lines.append(
                f"First boundary crossing at observation {self.first_crossing}"
            )
            lines.append("  (not an estimate of the date of the change)")

        lines.append("=" * 78)
        return "\n".join(lines)


@dataclass
class CUSUMSQResults(BreakTestResultsBase):
    """Results from the CUSUM-of-squares test for variance instability.

    The CUSUM-SQ test computes the cumulative sum of squared recursive
    residuals, normalized to [0, 1]. Under the null, this path tracks
    the diagonal, within bounds from the table of Durbin (1969) (see the
    module docstring).

    Attributes
    ----------
    test_name : str
        "CUSUM-SQ".
    nobs : int
        Number of observations.
    n_breaks : int
        Always 0: the test does not estimate a break date.
    break_indices : Sequence[int]
        Always empty; see ``first_crossing``.
    first_crossing : int | None
        Observation (0-based, in the sample the test was given) at which
        the path first leaves the bounds, or None. It is not an estimate of
        the date at which the variance changed.
    observations : NDArray
        Observation number of each point of the path: ``m, ..., T - 1``,
        where ``m`` is the number of observations that start the recursion
        (the number of regressors ``k``, unless the regressors are collinear
        at the start of the sample).
    critical_value : float
        Half-width ``c_0`` of the bounds.
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
    first_crossing: int | None = None
    critical_value: float = np.nan
    _test: CUSUMSQTest | None = field(default=None, repr=False)

    @property
    def is_stable(self) -> bool:
        """Whether the null of parameter stability is not rejected."""
        return not self.reject

    @property
    def observations(self) -> NDArray[np.intp]:
        """Observation number of each point of the path."""
        start = self.nobs - len(self.statistic_path)
        return np.arange(start, self.nobs)

    @property
    def max_deviation(self) -> float:
        """Maximum absolute deviation of the statistic from the expected path."""
        if len(self.statistic_path) == 0:
            return 0.0
        return float(np.max(np.abs(self.statistic_path - self.expected_path)))

    @property
    def crossing_indices(self) -> list[int]:
        """Observations at which the statistic is outside the bounds."""
        if len(self.statistic_path) == 0:
            return []
        above = self.statistic_path > self.upper_bound
        below = self.statistic_path < self.lower_bound
        crossed = np.where(above | below)[0] + self.nobs - len(self.statistic_path)
        return [int(i) for i in crossed]

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

        lines.append(f"Critical value (c_0):     {self.critical_value:>10.4f}")

        decision = "REJECT" if self.reject else "Do not reject"
        lines.append(f"\nDecision:  {decision} the null of variance stability")

        if self.reject and self.first_crossing is not None:
            lines.append(
                f"First boundary crossing at observation {self.first_crossing}"
            )
            lines.append("  (not an estimate of the date of the change)")

        lines.append("=" * 78)
        return "\n".join(lines)


class CUSUMTest(BreakTestBase):
    """CUSUM test for parameter instability (Brown, Durbin, Evans, 1975).

    Tests the null hypothesis of parameter stability by examining the
    cumulative sum of recursive residuals. Under H0, the cumulated sum
    behaves like a Brownian motion and stays within a pair of diverging
    straight lines with probability one minus the significance level. If
    the path crosses a line, stability is rejected.

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
    standard deviation (uncentered: ``sqrt(sum w_t^2 / (T - k))``). The
    critical boundaries are:

        ±a * sqrt(T - k) ± 2a * (r - k) / sqrt(T - k)

    where a is 1.143, 0.948 or 0.850 for a significance level of 1, 5 or 10
    percent. The test is somewhat conservative.

    The first crossing of a boundary is reported as ``first_crossing``. It
    is not an estimate of the date of the change, and the test reports no
    break date.

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
        level = _match_level(significance, _CUSUM_CRITICAL_VALUES)
        if level is None:
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

        # Critical boundaries (a pair of diverging straight lines)
        a = _CUSUM_CRITICAL_VALUES[level]
        r = np.arange(1, n + 1)  # r = 1, 2, ..., T-k
        sqrt_n = np.sqrt(n)
        upper_bound = a * sqrt_n + 2 * a * r / sqrt_n
        lower_bound = -(a * sqrt_n + 2 * a * r / sqrt_n)

        # Check for boundary crossings
        above = statistic_path > upper_bound
        below = statistic_path < lower_bound
        crossed = np.where(above | below)[0]
        reject = len(crossed) > 0

        # First crossing, as an observation of the sample. It is not an
        # estimate of a break date, so no break is reported.
        first_crossing = int(crossed[0]) + (T - n) if reject else None

        return CUSUMResults(
            test_name="CUSUM",
            nobs=T,
            n_breaks=0,
            break_indices=[],
            first_crossing=first_crossing,
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
    normalized path tracks the diagonal line, within bounds taken from the
    table of Durbin (1969) through the approximation of Edgerton and Wells
    (1994); see the module docstring.

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

        E[S_r] +/- c_0

    where c_0 is the critical value of the modified Kolmogorov-Smirnov
    statistic of Durbin (1969) for ``n = (T - k) / 2 - 1``, not for
    ``T - k``: bounds for ``T - k`` are too narrow by a factor that tends to
    ``sqrt(2)``.

    The first crossing of a boundary is reported as ``first_crossing``. It
    is not an estimate of the date of the change, and the test reports no
    break date.

    References
    ----------
    Brown, R. L., Durbin, J., & Evans, J. M. (1975). Techniques for testing
        the constancy of regression relationships over time. JRSS-B, 37(2).

    Durbin, J. (1969). Tests for serial correlation in regression analysis
        based on the periodogram of least-squares residuals. Biometrika,
        56(1), 1-15.

    Edgerton, D., & Wells, C. (1994). Critical values for the CUSUMSQ
        statistic in medium and large sized samples. Oxford Bulletin of
        Economics and Statistics, 56(3), 355-365.
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
            Significance level: 0.20, 0.10, 0.05 (default), 0.02 or 0.01,
            the levels for which the bounds are tabulated.

        Returns
        -------
        CUSUMSQResults
            Results object with test statistic path, boundaries,
            and rejection decision.

        Raises
        ------
        ValueError
            If significance is not one of the tabulated levels.
        """
        if _match_level(significance, _CUSUMSQ_COEFFICIENTS) is None:
            raise ValueError(
                "significance must be one of "
                f"{sorted(_CUSUMSQ_COEFFICIENTS)}, got {significance}"
            )

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

        # Critical value: Durbin (1969) through Edgerton and Wells (1994)
        c_alpha = cusum_sq_critical_value(significance, n)

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

        # First crossing, as an observation of the sample. It is not an
        # estimate of a break date, so no break is reported.
        first_crossing = int(crossed[0]) + (T - n) if reject else None

        return CUSUMSQResults(
            test_name="CUSUM-SQ",
            nobs=T,
            n_breaks=0,
            break_indices=[],
            first_crossing=first_crossing,
            critical_value=c_alpha,
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
