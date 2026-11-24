"""Andrews-Ploberger tests for a single structural break at an unknown date.

This module implements the SupF test of Andrews (1993) and the ExpF and
AveF tests of Andrews and Ploberger (1994). All three are functions of one
sequence of statistics, one for every candidate break date ``tau`` in a
trimmed range:

    F(tau) = (SSR_0 - SSR_1(tau)) / (SSR_1(tau) / (T - 2q - p))

where ``SSR_0`` is the sum of squared residuals without a break,
``SSR_1(tau)`` the sum with a break at ``tau`` in the ``q`` breaking
coefficients, and ``p`` the number of coefficients that do not break.
``F(tau)`` is ``q`` times the classical Chow F statistic: it is "scaled by
q", not divided by it, and is asymptotically chi-squared with ``q`` degrees
of freedom at a given date. That is the scale on which the three statistics
are defined in the literature and tabulated:

- **SupF**: ``max F(tau)``
- **ExpF**: ``ln[ mean( exp(F(tau) / 2) ) ]``
- **AveF**: ``mean F(tau)``

P-values are the approximate asymptotic p-values of Hansen (1997); critical
values are obtained by inverting them. They are available for up to 40
breaking coefficients and for any trimming. The same statistic and scaling
are used by ``BaiPerronTest`` for one break (with ``hetvar=False``).

The statistics assume homoskedastic, serially uncorrelated errors.

References
----------
Andrews, D. W. K. (1993). Tests for parameter instability and structural
    change with unknown change point. Econometrica, 61(4), 821-856.
    (Corrigendum: Econometrica, 71(1), 2003, 395-397.)

Andrews, D. W. K. & Ploberger, W. (1994). Optimal tests when a nuisance
    parameter is present only under the alternative. Econometrica, 62(6),
    1383-1414.

Hansen, B. E. (1997). Approximate asymptotic p values for structural-change
    tests. Journal of Business and Economic Statistics, 15(1), 60-67.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from regimes.tests._bp_core import independent_columns
from regimes.tests._hansen_pvalues import (
    MAX_Q,
    SMALLEST_PI0,
    hansen_critical_value,
    hansen_pvalue,
    pi0_from_range,
)
from regimes.tests.base import BreakTestBase, BreakTestResultsBase, split_constant

# A sum of squared residuals this small relative to y'y is rounding error:
# the model fits exactly and the statistic is not defined. (The threshold
# is far below the residual variance of any series with noise, also when
# its mean is large.)
_EXACT_FIT = 1e-24

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from regimes.models.adl import ADL
    from regimes.models.ar import AR
    from regimes.models.ols import OLS

_TABULATED_LEVELS = (0.10, 0.05, 0.01)


# ---------------------------------------------------------------------------
# Results dataclass
# ---------------------------------------------------------------------------


@dataclass
class AndrewsPlobergerResults(BreakTestResultsBase):
    """Results from the Andrews-Ploberger structural break tests.

    Contains three test statistics (SupF, ExpF, AveF) computed from the
    same sequence ``F(tau)``, with the p-values of Hansen (1997) and the
    critical values implied by them.

    Attributes
    ----------
    test_name : str
        Name of the test ("Andrews-Ploberger").
    nobs : int
        Number of observations.
    n_breaks : int
        1 if the SupF test rejects, otherwise 0.
    break_indices : Sequence[int]
        The date that maximizes ``F(tau)`` if the SupF test rejects,
        otherwise empty. (The date belongs to the SupF test: ExpF and AveF
        are averages over dates and do not point to one.)
    sup_f : float
        SupF statistic (maximum of the sequence).
    exp_f : float
        ExpF statistic (log of the average of ``exp(F / 2)``).
    ave_f : float
        AveF statistic (average of the sequence).
    sup_f_critical : dict[float, float]
        Critical values for SupF at 10%, 5%, 1% and at
        ``significance_level``.
    exp_f_critical : dict[float, float]
        Critical values for ExpF.
    ave_f_critical : dict[float, float]
        Critical values for AveF.
    sup_f_pvalue : float
        Asymptotic p-value for SupF (Hansen, 1997).
    exp_f_pvalue : float
        Asymptotic p-value for ExpF.
    ave_f_pvalue : float
        Asymptotic p-value for AveF.
    f_sequence : NDArray[np.floating]
        ``F(tau)`` at each candidate break date, scaled by ``q`` (see the
        module docstring).
    candidate_indices : NDArray[np.intp]
        Candidate break dates: ``tau`` is the first observation of the
        second regime. For a test built from an AR or ADL model these are
        observation numbers of the model's data.
    sup_f_break_index : int
        The date that maximizes ``F(tau)``, whether or not the test
        rejects.
    trimming : float
        Trimming fraction requested.
    pi0 : float
        Andrews' ``pi_0`` for the range of candidate dates actually used.
    q : int
        Number of breaking parameters.
    p : int
        Number of non-breaking parameters.
    significance_level : float
        Significance level used for rejection decisions.
    _test : AndrewsPlobergerTest | None
        Reference back to the test object (not shown in repr).
    """

    sup_f: float = np.nan
    exp_f: float = np.nan
    ave_f: float = np.nan
    sup_f_critical: dict[float, float] = field(default_factory=dict)
    exp_f_critical: dict[float, float] = field(default_factory=dict)
    ave_f_critical: dict[float, float] = field(default_factory=dict)
    sup_f_pvalue: float = np.nan
    exp_f_pvalue: float = np.nan
    ave_f_pvalue: float = np.nan
    f_sequence: np.ndarray = field(default_factory=lambda: np.array([]))
    candidate_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    sup_f_break_index: int = 0
    trimming: float = 0.15
    pi0: float = np.nan
    q: int = 0
    p: int = 0
    significance_level: float = 0.05
    _test: AndrewsPlobergerTest | None = field(default=None, repr=False)

    @property
    def rejected(self) -> dict[str, bool]:
        """Whether H0 was rejected for each test statistic."""
        alpha = self.significance_level
        return {
            "SupF": bool(self.sup_f_pvalue < alpha),
            "ExpF": bool(self.exp_f_pvalue < alpha),
            "AveF": bool(self.ave_f_pvalue < alpha),
        }

    def plot(
        self,
        ax: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Plot the F-statistic sequence.

        Convenience method delegating to
        :func:`regimes.visualization.andrews_ploberger.plot_f_sequence`.

        Parameters
        ----------
        ax : Axes | None
            Matplotlib axes to plot on.
        **kwargs
            Additional keyword arguments passed to ``plot_f_sequence``.

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib figure and axes.
        """
        from regimes.visualization.andrews_ploberger import plot_f_sequence

        return plot_f_sequence(self, ax=ax, **kwargs)

    def summary(self) -> str:
        """Generate a text summary of Andrews-Ploberger test results.

        Returns
        -------
        str
            Formatted summary including all three statistics, critical
            values, p-values, and rejection decisions.
        """
        alpha = self.significance_level
        lines = []
        lines.append("=" * 78)
        lines.append(f"{'Andrews-Ploberger Structural Break Tests':^78}")
        lines.append("=" * 78)
        lines.append(f"Number of observations:   {self.nobs:>10}")
        lines.append(f"Breaking parameters (q):  {self.q:>10}")
        lines.append(f"Non-breaking params (p):  {self.p:>10}")
        lines.append(f"Trimming:                 {self.trimming:>10.2f}")
        lines.append(f"Significance level:       {self.significance_level:>10.3f}")
        lines.append(f"Date of the largest F:    {self.sup_f_break_index:>10}")
        lines.append("-" * 78)

        # Header row
        lines.append(
            f"\n{'Statistic':>12} {'Value':>12} "
            f"{'CV(10%)':>10} {'CV(5%)':>10} {'CV(1%)':>10} "
            f"{'p-value':>10} {'Reject':>8}"
        )
        lines.append("-" * 78)

        rows = (
            ("SupF", self.sup_f, self.sup_f_critical, self.sup_f_pvalue),
            ("ExpF", self.exp_f, self.exp_f_critical, self.exp_f_pvalue),
            ("AveF", self.ave_f, self.ave_f_critical, self.ave_f_pvalue),
        )
        for name, value, critical, pvalue in rows:
            cv10 = critical.get(0.10, np.nan)
            cv05 = critical.get(0.05, np.nan)
            cv01 = critical.get(0.01, np.nan)
            reject = "Yes" if pvalue < alpha else "No"
            lines.append(
                f"{name:>12} {value:>12.4f} "
                f"{cv10:>10.2f} {cv05:>10.2f} {cv01:>10.2f} "
                f"{_format_pvalue(pvalue):>10} {reject:>8}"
            )

        lines.append("-" * 78)
        lines.append("Statistics are scaled by q; p-values from Hansen (1997).")
        if self.n_breaks > 0:
            lines.append(
                f"SupF rejects at the {alpha:.1%} level: break estimated at "
                f"{self.sup_f_break_index}"
            )
        else:
            lines.append(f"SupF does not reject at the {alpha:.1%} level")
        lines.append("=" * 78)
        return "\n".join(lines)


def _format_pvalue(pv: float) -> str:
    """Format a p-value for display."""
    if np.isnan(pv):
        return "n/a"
    if pv < 0.0001:
        return "< 0.0001"
    return f"{pv:.4f}"


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class AndrewsPlobergerTest(BreakTestBase):
    """Andrews-Ploberger tests for a single structural break at an unknown date.

    Tests whether regression coefficients change at an unknown break point
    by computing a statistic at every candidate break date in a trimmed
    range. Reports three summary statistics: SupF, ExpF, and AveF, each with
    its own p-value. See the module docstring for the definitions.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    exog : ArrayLike | None
        Exogenous regressors whose coefficients do NOT break. A column that
        is a linear combination of the breaking regressors (a constant,
        when the constant breaks) is left out.
    exog_break : ArrayLike | None
        Regressors whose coefficients may break. If None, defaults to
        a constant (mean-shift model).

    Examples
    --------
    >>> import numpy as np
    >>> from regimes import AndrewsPlobergerTest
    >>> rng = np.random.default_rng(42)
    >>> y = np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 100)])
    >>> test = AndrewsPlobergerTest(y)
    >>> results = test.fit()
    >>> print(results.summary())

    Notes
    -----
    The statistics are valid asymptotically under homoskedastic, serially
    uncorrelated errors. No robust version is implemented.

    The trimming parameter controls how close to the sample boundaries
    candidate break dates can be: with ``h = floor(trimming * T)`` (the rule
    ``BaiPerronTest`` uses), each regime has at least ``h`` observations.
    Andrews (1993) recommends 0.15 for SupF.

    The three tests are separate tests. Rejecting when any one of them
    rejects is a different test with a larger size.

    References
    ----------
    Andrews, D. W. K. (1993). Tests for parameter instability and structural
        change with unknown change point. Econometrica, 61(4), 821-856.

    Andrews, D. W. K. & Ploberger, W. (1994). Optimal tests when a nuisance
        parameter is present only under the alternative. Econometrica, 62(6),
        1383-1414.

    Hansen, B. E. (1997). Approximate asymptotic p values for
        structural-change tests. Journal of Business and Economic
        Statistics, 15(1), 60-67.
    """

    def __init__(
        self,
        endog: ArrayLike,
        exog: ArrayLike | None = None,
        exog_break: ArrayLike | None = None,
    ) -> None:
        """Initialize the Andrews-Ploberger test."""
        if exog_break is None:
            exog_break = np.ones((len(np.asarray(endog)), 1))

        super().__init__(endog, exog, exog_break)

        # A fixed regressor that the breaking regressors span (a constant
        # when the constant breaks) is not identified: leave it out.
        fixed = self.exog
        if fixed is not None and self.exog_break is not None:
            keep = independent_columns(self.exog_break, fixed)
            self.exog = fixed[:, keep] if keep else None

    @classmethod
    def from_model(
        cls,
        model: OLS | AR | ADL,
        break_vars: Literal["all", "const"] = "all",
    ) -> AndrewsPlobergerTest:
        """Create AndrewsPlobergerTest from an OLS, AR, or ADL model.

        Parameters
        ----------
        model : OLS | AR | ADL
            Model to test for structural breaks.
        break_vars : "all" | "const"
            Which variables can have breaks:

            - "all": All regressors can break (default)
            - "const": Only the intercept can break (mean-shift model); the
              model must have a constant.

        Returns
        -------
        AndrewsPlobergerTest
            Test instance ready for .fit()

        Notes
        -----
        For AR and ADL models, the test is computed on the effective
        sample (after dropping the initial observations for lags).
        Break dates and other observation numbers in the results are
        those of the model's data (see ``index_offset``).
        """
        from regimes.models.adl import ADL as ADLModel
        from regimes.models.ar import AR as ARModel
        from regimes.models.ols import OLS as OLSModel

        if isinstance(model, (ARModel, ADLModel)):
            y, X, _ = model._build_design_matrix()
            offset = int(model.maxlag)
            endog = y
            exog_all = X
        elif isinstance(model, OLSModel):
            offset = 0
            endog = model.endog
            exog_all = model.exog
        else:
            raise TypeError(
                f"model must be OLS, AR, or ADL, got {type(model).__name__}"
            )

        if exog_all is None:
            exog_all = np.ones((len(endog), 1))

        if break_vars == "all":
            return cls._create(offset, endog, exog_break=exog_all)
        elif break_vars == "const":
            return cls._create(
                offset,
                endog,
                exog=split_constant(np.asarray(exog_all, dtype=np.float64)),
                exog_break=np.ones((len(endog), 1)),
            )
        else:
            raise ValueError(f"break_vars must be 'all' or 'const', got {break_vars!r}")

    @property
    def n_break_params(self) -> int:
        """Number of breaking regressors (q)."""
        if self.exog_break is None:
            return 0
        return self.exog_break.shape[1]

    @property
    def n_nonbreak_params(self) -> int:
        """Number of non-breaking regressors (p)."""
        if self.exog is None:
            return 0
        return self.exog.shape[1]

    def _compute_ssr(
        self,
        y: NDArray[np.floating[Any]],
        X: NDArray[np.floating[Any]],
    ) -> float:
        """Compute sum of squared residuals from OLS.

        Parameters
        ----------
        y : NDArray[np.floating]
            Dependent variable.
        X : NDArray[np.floating]
            Regressor matrix.

        Returns
        -------
        float
            Sum of squared residuals.
        """
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return float(resid @ resid)

    def _build_regressor_matrix(
        self,
        start: int,
        end: int,
    ) -> NDArray[np.floating[Any]]:
        """Build the full regressor matrix for a segment [start, end).

        Parameters
        ----------
        start : int
            Start index (inclusive).
        end : int
            End index (exclusive).

        Returns
        -------
        NDArray[np.floating]
            Regressor matrix for the segment.
        """
        parts = []
        if self.exog_break is not None:
            parts.append(self.exog_break[start:end])
        if self.exog is not None:
            parts.append(self.exog[start:end])

        if not parts:
            return np.ones((end - start, 1))

        return np.column_stack(parts)

    def _segment_length(self, trimming: float) -> int:
        """Minimum number of observations in each regime."""
        q = self.n_break_params
        p = self.n_nonbreak_params
        return max(int(np.floor(self.nobs * trimming + 1e-9)), q + p + 1)

    def _compute_f_sequence(
        self,
        trimming: float,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.intp]]:
        """Compute ``F(tau)`` (scaled by q) at every candidate break date.

        Parameters
        ----------
        trimming : float
            Fraction of observations to trim from each end.

        Returns
        -------
        tuple[NDArray[np.floating], NDArray[np.intp]]
            (f_sequence, candidate_indices)
        """
        T = self.nobs
        q = self.n_break_params
        p = self.n_nonbreak_params

        # Determine trimmed range
        min_obs = self._segment_length(trimming)
        start = min_obs
        end = T - min_obs

        df_denom = T - 2 * q - p
        if start > end or df_denom <= 0:
            return np.array([]), np.array([], dtype=int)

        candidate_indices = np.arange(start, end + 1)

        # Precompute full-sample SSR
        X_full = self._build_regressor_matrix(0, T)
        ssr_full = self._compute_ssr(self.endog, X_full)

        f_sequence = np.zeros(len(candidate_indices))
        for i, tau in enumerate(candidate_indices):
            # Compute unrestricted SSR for this break date
            if self.exog is not None and self.exog_break is not None:
                Z1_break = np.zeros((T, q))
                Z1_break[:tau] = self.exog_break[:tau]
                Z2_break = np.zeros((T, q))
                Z2_break[tau:] = self.exog_break[tau:]

                X_unrestricted = np.column_stack([Z1_break, Z2_break, self.exog])
                ssr_unrestricted = self._compute_ssr(self.endog, X_unrestricted)
            else:
                X1 = self._build_regressor_matrix(0, tau)
                X2 = self._build_regressor_matrix(tau, T)
                ssr1 = self._compute_ssr(self.endog[:tau], X1)
                ssr2 = self._compute_ssr(self.endog[tau:], X2)
                ssr_unrestricted = ssr1 + ssr2

            if ssr_unrestricted <= _EXACT_FIT * float(self.endog @ self.endog):
                raise ValueError(
                    "the model fits the data exactly with a break at "
                    f"observation {tau + self.index_offset}; the test "
                    "statistic is not defined"
                )
            # q times the classical F statistic (Wald scale)
            f_val = (ssr_full - ssr_unrestricted) / (ssr_unrestricted / df_denom)
            f_sequence[i] = max(0.0, f_val)

        return f_sequence, candidate_indices

    def fit(
        self,
        trimming: float = 0.15,
        significance: float = 0.05,
        **kwargs: Any,
    ) -> AndrewsPlobergerResults:
        """Perform the Andrews-Ploberger tests.

        Parameters
        ----------
        trimming : float
            Fraction of observations trimmed from each end. Default is
            0.15 (Andrews's recommendation for SupF).
        significance : float
            Significance level for rejection decisions. Default is 0.05.
        **kwargs
            Additional arguments (reserved for future use).

        Returns
        -------
        AndrewsPlobergerResults
            Results with SupF, ExpF, AveF statistics, p-values, critical
            values, and rejection decisions.

        Raises
        ------
        ValueError
            If trimming is not in (0, 0.5), if significance is not in
            (0, 1), if there are no candidate break dates after trimming,
            or if there are more than 40 breaking coefficients (the range
            of Hansen's p-values).
        """
        if not 0 < trimming < 0.5:
            raise ValueError(f"trimming must be in (0, 0.5), got {trimming}")
        if not 0 < significance < 1:
            raise ValueError(f"significance must be in (0, 1), got {significance}")

        q = self.n_break_params
        p = self.n_nonbreak_params
        if q > MAX_Q:
            raise ValueError(
                f"p-values are available for up to {MAX_Q} breaking "
                f"coefficients, got q = {q}"
            )

        # Compute F-sequence
        f_sequence, candidate_indices = self._compute_f_sequence(trimming)

        T_star = len(f_sequence)
        if T_star == 0:
            raise ValueError(
                "No candidate break dates after trimming. "
                "Increase sample size or decrease trimming."
            )

        # Compute the three statistics
        sup_f = float(np.max(f_sequence))
        ave_f = float(np.mean(f_sequence))

        # ExpF: use log-sum-exp trick for numerical stability
        half_f = 0.5 * f_sequence
        max_half_f = np.max(half_f)
        exp_f = float(max_half_f + np.log(np.mean(np.exp(half_f - max_half_f))))

        # Break location
        sup_f_break_index = int(candidate_indices[np.argmax(f_sequence)])

        # P-values and critical values (Hansen, 1997), for the range of
        # candidate dates actually used
        T = self.nobs
        pi0 = pi0_from_range(candidate_indices[0] / T, candidate_indices[-1] / T)
        if pi0 < SMALLEST_PI0 - 1e-12:
            warnings.warn(
                f"the candidate dates leave pi_0 = {pi0:.4f}, below 0.01, the "
                "smallest value Hansen (1997) tabulates (trimming of less "
                "than about 1 percent); the p-values and critical values are "
                "those for 0.01, so the p-values are too small",
                UserWarning,
                stacklevel=2,
            )
        sup_f_pv = hansen_pvalue(sup_f, q, pi0, "sup")
        exp_f_pv = hansen_pvalue(exp_f, q, pi0, "exp")
        ave_f_pv = hansen_pvalue(ave_f, q, pi0, "ave")

        levels = sorted({*_TABULATED_LEVELS, float(significance)}, reverse=True)
        sup_f_cv = {a: hansen_critical_value(a, q, pi0, "sup") for a in levels}
        exp_f_cv = {a: hansen_critical_value(a, q, pi0, "exp") for a in levels}
        ave_f_cv = {a: hansen_critical_value(a, q, pi0, "ave") for a in levels}

        # Observation numbers of the underlying data
        candidate_indices = candidate_indices + self.index_offset
        sup_f_break_index += self.index_offset

        # A break date is reported when the SupF test rejects
        sup_rejects = bool(sup_f_pv < significance)
        break_indices: list[int] = [sup_f_break_index] if sup_rejects else []
        n_breaks = 1 if sup_rejects else 0

        return AndrewsPlobergerResults(
            test_name="Andrews-Ploberger",
            nobs=self.nobs,
            n_breaks=n_breaks,
            break_indices=break_indices,
            sup_f=sup_f,
            exp_f=exp_f,
            ave_f=ave_f,
            sup_f_critical=sup_f_cv,
            exp_f_critical=exp_f_cv,
            ave_f_critical=ave_f_cv,
            sup_f_pvalue=sup_f_pv,
            exp_f_pvalue=exp_f_pv,
            ave_f_pvalue=ave_f_pv,
            f_sequence=f_sequence,
            candidate_indices=candidate_indices,
            sup_f_break_index=sup_f_break_index,
            trimming=trimming,
            pi0=pi0,
            q=q,
            p=p,
            significance_level=significance,
            index_offset=self.index_offset,
            _test=self,
        )
