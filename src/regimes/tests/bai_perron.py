"""Bai-Perron test for multiple structural breaks.

This module implements the Bai-Perron (1998, 2003) procedure for testing
and estimating multiple structural breaks in linear regression models. The
algorithms live in :mod:`regimes.tests._bp_core`, a port of the OxRegimes
implementation that was cross-validated against Pierre Perron's program; the
critical values are the published tables (:mod:`regimes.tests._bp_critical_values`).

References
----------
Bai, J. (1997). Estimation of a change point in multiple regression models.
    Review of Economics and Statistics, 79(4), 551-563.

Bai, J., & Perron, P. (1998). Estimating and testing linear models with
    multiple structural changes. Econometrica, 66(1), 47-78.

Bai, J., & Perron, P. (2003). Computation and analysis of multiple
    structural change models. Journal of Applied Econometrics, 18(1), 1-22.

Bai, J., & Perron, P. (2003b). Critical values for multiple structural change
    tests. Econometrics Journal, 6(1), 72-78.

Liu, J., Wu, S., & Zidek, J. V. (1997). On segmented multivariate
    regression. Statistica Sinica, 7(2), 497-525.

Yao, Y.-C. (1988). Estimating the number of change-points via Schwarz'
    criterion. Statistics & Probability Letters, 6(3), 181-189.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from regimes.results.base import _obs_label
from regimes.tests import _bp_core as _bp
from regimes.tests import _bp_critical_values as _cv
from regimes.tests._bp_core import BPOptions
from regimes.tests.base import BreakTestBase, BreakTestResultsBase

# condition number of the regressors (columns scaled to unit length) above
# which fit() warns that tests and intervals may be inaccurate
_COND_WARN = 1e6
# an SSR below this fraction of y'y is treated as an exact fit
_EXACT_FIT = 1e-22

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    from regimes.models.adl import ADL
    from regimes.models.ar import AR
    from regimes.models.base import CovType
    from regimes.models.ols import OLS, OLSResults
    from regimes.results.base import IndexLike


def _fmt(value: float, spec: str) -> str:
    """Format a number, showing a dash for NaN."""
    width = int(spec.split(".")[0])
    if value is None or np.isnan(value):
        return f"{'-':>{width}}"
    return f"{value:>{spec}}"


@dataclass
class BaiPerronResults(BreakTestResultsBase):
    """Results from the Bai-Perron multiple structural break procedure.

    All F statistics are "scaled by q" (not divided by the number of breaking
    regressors), which is the scaling of the published critical values.

    Attributes
    ----------
    test_name : str
        Name of the test ("Bai-Perron").
    nobs : int
        Number of observations.
    n_breaks : int
        Number of breaks selected by ``selection_method``.
    break_indices : Sequence[int]
        Selected break dates: index of the first observation of each new
        regime.
    supf_stats : dict[int, float]
        sup F(k|0) for k = 1..max_breaks, at the global SSR minimizers.
    supf_critical : dict[int, float]
        Critical values of sup F(k|0) at ``level``.
    udmax, udmax_critical : float
        Unweighted double maximum statistic and its critical value at
        ``level``.
    wdmax, wdmax_critical : float
        Weighted double maximum statistic (weights c(1)/c(k) at ``level``)
        and its critical value.
    wdmax_stats : dict[float, float]
        WDmax at each tabulated significance level.
    seqf_stats : dict[int, float]
        ``seqf_stats[l]`` is sup F(l+1|l), computed on the global l-break
        partition; ``seqf_stats[0]`` is sup F(1|0). NaN when no regime is long
        enough to be split.
    seqf_critical : dict[int, float]
        Critical values of sup F(l+1|l) at ``level``.
    seqf_break : dict[int, int]
        Break date the (l+1)-th break would have (-1 if none).
    bic, lwz : dict[int, float]
        Information criteria for m = 0..max_breaks, per observation; the
        penalty counts the m break dates as estimated parameters.
    ssr : dict[int, float]
        Sum of squared residuals for m = 0..max_breaks.
    breaks_by_m : dict[int, Sequence[int]]
        Global SSR minimizers for each number of breaks.
    sequential_n_breaks : int | None
        Number of breaks found by the sequential procedure at ``level`` (None
        if critical values are not tabulated for the trimming).
    sequential_breaks : Sequence[int]
        Break dates estimated by the sequential procedure.
    break_ci, break_ci_90 : dict[int, tuple[int, int]]
        95% and 90% confidence intervals for the selected break dates
        (Bai 1997), keyed by break date.
    critical_values : dict[str, dict[float, Any]]
        Critical values at all tabulated levels, under the keys ``"supf"``,
        ``"seqf"``, ``"udmax"`` and ``"wdmax"``.
    trimming : float
        Trimming parameter.
    h : int
        Minimum regime length used.
    max_breaks : int
        Maximum number of breaks considered.
    selection_method : str
        Method used to select the number of breaks.
    level : float
        Significance level of the tests and the sequential procedure.
    q, p : int
        Number of regressors with breaking and with fixed coefficients.
    options : BPOptions
        Assumptions on errors and regressors used by tests and intervals.
    """

    supf_stats: dict[int, float] = field(default_factory=dict)
    supf_critical: dict[int, float] = field(default_factory=dict)
    udmax: float = np.nan
    udmax_critical: float = np.nan
    wdmax: float = np.nan
    wdmax_critical: float = np.nan
    wdmax_stats: dict[float, float] = field(default_factory=dict)
    seqf_stats: dict[int, float] = field(default_factory=dict)
    seqf_critical: dict[int, float] = field(default_factory=dict)
    seqf_break: dict[int, int] = field(default_factory=dict)
    bic: dict[int, float] = field(default_factory=dict)
    lwz: dict[int, float] = field(default_factory=dict)
    ssr: dict[int, float] = field(default_factory=dict)
    breaks_by_m: dict[int, Sequence[int]] = field(default_factory=dict)
    sequential_n_breaks: int | None = None
    sequential_breaks: Sequence[int] = field(default_factory=list)
    trimming: float = 0.15
    h: int = 0
    max_breaks: int = 5
    selection_method: str = "bic"
    level: float = 0.05
    q: int = 1  # Number of breaking regressors
    p: int = 0  # Number of fixed regressors
    break_ci: dict[int, tuple[int, int]] = field(default_factory=dict)
    break_ci_90: dict[int, tuple[int, int]] = field(default_factory=dict)
    critical_values: dict[str, dict[float, Any]] = field(default_factory=dict)
    options: BPOptions = field(default_factory=BPOptions)
    repartition: bool = False

    # Private field to store reference back to the test for to_ols()
    _test: BaiPerronTest | None = field(default=None, repr=False)

    def to_ols(self, cov_type: CovType = "nonrobust") -> OLSResults:
        """Convert to OLSResults with detected breaks.

        Returns a fitted OLS model using the break dates selected
        by this test.

        Parameters
        ----------
        cov_type : CovType
            Covariance type for the OLS estimation. Options are:
            - "nonrobust": Standard OLS covariance
            - "HC0", "HC1", "HC2", "HC3": Heteroskedasticity-robust
            - "HAC": Heteroskedasticity and autocorrelation consistent

        Returns
        -------
        OLSResults
            Fitted OLS results with regime-specific parameters.

        Raises
        ------
        ValueError
            If test reference is not stored (test was not created via
            from_model or the reference was not preserved).

        Examples
        --------
        >>> import numpy as np
        >>> from regimes import OLS, BaiPerronTest
        >>> np.random.seed(42)
        >>> n = 200
        >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
        >>> y = np.zeros(n)
        >>> y[:100] = 1 + 0.5 * X[:100, 1] + np.random.randn(100) * 0.5
        >>> y[100:] = 3 + 1.5 * X[100:, 1] + np.random.randn(100) * 0.5
        >>> model = OLS(y, X, has_constant=False)
        >>> bp_results = model.bai_perron()
        >>> ols_with_breaks = bp_results.to_ols()
        >>> print(ols_with_breaks.summary())
        """
        if self._test is None:
            raise ValueError(
                "Cannot convert to OLS: test reference not stored. "
                "Use BaiPerronTest.from_model() or model.bai_perron() to create "
                "the test, or pass the test explicitly."
            )

        # Get data from test
        endog = self._test.endog

        # Build exog and variable_breaks
        exog_list = []
        variable_breaks = {}
        col_idx = 0

        # Non-breaking regressors: the columns fit() used (columns of exog that
        # are linear combinations of the breaking regressors are left out)
        _, _, fixed = self._test._data()
        if fixed is not None:
            exog_list.append(fixed)
            col_idx += fixed.shape[1]

        # Breaking regressors
        if self._test.exog_break is not None:
            exog_list.append(self._test.exog_break)
            n_break_cols = self._test.exog_break.shape[1]
            # Add breaks for these columns
            for i in range(n_break_cols):
                variable_breaks[col_idx + i] = [
                    int(b) - self.index_offset for b in self.break_indices
                ]
            col_idx += n_break_cols

        if exog_list:
            exog = np.column_stack(exog_list)
        else:
            exog = np.ones((len(endog), 1))

        # Create and fit OLS with detected breaks
        from regimes.models.ols import OLS

        ols_model = OLS(
            endog,
            exog,
            variable_breaks=variable_breaks if self.n_breaks > 0 else None,
            has_constant=False,
        )
        return ols_model.fit(cov_type=cov_type)

    def summary(self, index: IndexLike | None = None) -> str:
        """Generate a text summary of the Bai-Perron results.

        Parameters
        ----------
        index : IndexLike or None
            Optional time index (e.g. ``pd.PeriodIndex``). When provided,
            break locations and confidence intervals are shown as dates
            instead of integer observation numbers.

        Returns
        -------
        str
            Formatted summary including test statistics and break dates.
        """
        levels = _cv.LEVELS
        cv = self.critical_values
        opt = self.options
        width = 78

        def dates(breaks: Sequence[int]) -> str:
            return ", ".join(_obs_label(int(b), index) for b in breaks)

        lines = []
        lines.append("=" * width)
        lines.append(f"{'Bai-Perron Multiple Structural Break Test':^{width}}")
        lines.append("=" * width)
        lines.append(f"Number of observations:   {self.nobs:>10}")
        if self.index_offset:
            first = _obs_label(self.index_offset, index)
            last = _obs_label(self.index_offset + self.nobs - 1, index)
            lines.append(f"Sample:                   {first} to {last}")
        lines.append(
            f"Trimming parameter:       {self.trimming:>10.2f}  (h = {self.h})"
        )
        lines.append(f"Maximum breaks tested:    {self.max_breaks:>10}")
        lines.append(f"Breaking regressors (q):  {self.q:>10}")
        if self.p > 0:
            lines.append(f"Fixed regressors (p):     {self.p:>10}")
        errors = "serial correlation allowed (HAC" + (
            ", prewhitened)" if opt.prewhiten else ")"
        )
        lines.append(
            "Errors: "
            + (errors if opt.robust else "serially uncorrelated")
            + (
                ", variances differ across regimes"
                if opt.hetvar
                else ", common variance"
            )
        )
        lines.append("-" * width)

        lines.append("\nGlobal minimizers (break date = first obs. of new regime):")
        lines.append(f"{'m':>5} {'SSR':>15}   Break dates")
        for m in sorted(self.ssr):
            lines.append(
                f"{m:>5} {self.ssr[m]:>15.6g}   {dates(self.breaks_by_m.get(m, []))}"
            )

        head = "".join(f"{f'{100 * lev:g}%':>10}" for lev in levels)
        lines.append("\nSup-F Tests (H0: 0 breaks vs H1: k breaks), scaled by q:")
        lines.append(f"{'k':>5} {'Sup-F':>12}{head}")
        for k in sorted(self.supf_stats):
            row = "".join(
                _fmt(cv.get("supf", {}).get(lev, {}).get(k, np.nan), "10.2f")
                for lev in levels
            )
            lines.append(f"{k:>5} {self.supf_stats[k]:>12.4f}{row}")
        row = "".join(
            _fmt(cv.get("udmax", {}).get(lev, np.nan), "10.2f") for lev in levels
        )
        lines.append(f"{'UDmax':>5} {self.udmax:>12.4f}{row}")
        row = "".join(
            _fmt(cv.get("wdmax", {}).get(lev, np.nan), "10.2f") for lev in levels
        )
        stats_w = " / ".join(
            _fmt(self.wdmax_stats.get(lev, np.nan), "1.4f").strip() for lev in levels
        )
        lines.append(f"{'WDmax':>5} {'':>12}{row}")
        lines.append(f"      WDmax statistic at each level: {stats_w}")

        if len(self.seqf_stats) > 1:
            lines.append(
                "\nSup-F(l+1|l) Tests (H0: l breaks vs H1: l+1 breaks), "
                "global partitions:"
            )
            lines.append(f"{'l+1|l':>7} {'Sup-F':>12} {'New break':>12}{head}")
            for ell in sorted(self.seqf_stats):
                if ell == 0:
                    continue
                row = "".join(
                    _fmt(cv.get("seqf", {}).get(lev, {}).get(ell, np.nan), "10.2f")
                    for lev in levels
                )
                new = self.seqf_break.get(ell, -1)
                label = _obs_label(new, index) if new >= 0 else "-"
                lines.append(
                    f"{f'{ell + 1}|{ell}':>7} "
                    f"{_fmt(self.seqf_stats[ell], '12.4f')} {label:>12}{row}"
                )

        lines.append("\nInformation Criteria (break dates counted as parameters):")
        lines.append(f"{'m':>5} {'BIC':>15} {'LWZ':>15}")
        best_bic = min(self.bic, key=lambda m: self.bic[m]) if self.bic else -1
        best_lwz = min(self.lwz, key=lambda m: self.lwz[m]) if self.lwz else -1
        for m in sorted(self.bic):
            mark = ("   <- BIC" if m == best_bic else "") + (
                "   <- LWZ" if m == best_lwz else ""
            )
            lines.append(
                f"{m:>5} {self.bic[m]:>15.4f} {self.lwz.get(m, np.nan):>15.4f}{mark}"
            )

        if self.sequential_n_breaks is not None:
            where = (
                f" at {dates(self.sequential_breaks)}"
                if self.sequential_n_breaks > 0
                else ""
            )
            lines.append(
                f"\nSequential procedure at {100 * self.level:g}%: "
                f"{self.sequential_n_breaks} break(s){where}"
                + (
                    " (repartition applied)"
                    if self.repartition and (self.sequential_n_breaks or 0) > 1
                    else ""
                )
            )

        lines.append("-" * width)
        lines.append(
            f"\nSelected number of breaks ({self.selection_method.upper()}): "
            f"{self.n_breaks}"
        )

        if self.n_breaks > 0:
            break_labels = [_obs_label(b, index) for b in self.break_indices]
            lines.append(f"Break dates: {break_labels}")

            if self.break_ci:
                lines.append("\n95% [90%] Confidence Intervals for Break Dates:")
                for break_idx in self.break_indices:
                    if break_idx in self.break_ci:
                        lower, upper = self.break_ci[break_idx]
                        bl = _obs_label(break_idx, index)
                        lo = _obs_label(lower, index)
                        hi = _obs_label(upper, index)
                        line = f"  Break at {bl}: [{lo}, {hi}]"
                        if break_idx in self.break_ci_90:
                            lo90, hi90 = self.break_ci_90[break_idx]
                            line += (
                                f"   [{_obs_label(lo90, index)}, "
                                f"{_obs_label(hi90, index)}]"
                            )
                        lines.append(line)

        lines.append("=" * width)
        return "\n".join(lines)


def _shift_break_dates(results: BaiPerronResults, offset: int) -> BaiPerronResults:
    """Express every break date of the results in the model's observations.

    A test built from an AR or ADL model is computed on the sample that
    remains after the initial observations for the lags; its break dates are
    reported as observation numbers of the model's data, ``offset`` later.
    """
    if offset == 0:
        return results
    results.index_offset = offset
    results.break_indices = [int(b) + offset for b in results.break_indices]
    results.sequential_breaks = [int(b) + offset for b in results.sequential_breaks]
    results.breaks_by_m = {
        m: [int(b) + offset for b in dates] for m, dates in results.breaks_by_m.items()
    }
    results.seqf_break = {
        ell: (int(b) + offset if b >= 0 else int(b))
        for ell, b in results.seqf_break.items()
    }
    results.break_ci = {
        int(b) + offset: (lo + offset, hi + offset)
        for b, (lo, hi) in results.break_ci.items()
    }
    results.break_ci_90 = {
        int(b) + offset: (lo + offset, hi + offset)
        for b, (lo, hi) in results.break_ci_90.items()
    }
    return results


class BaiPerronTest(BreakTestBase):
    """Bai-Perron procedure for multiple structural breaks.

    Implements the Bai-Perron (1998, 2003) procedure for testing and
    estimating multiple structural breaks in the linear regression

        y_t = z_t' delta_j + x_t' beta + u_t,    t in regime j,

    where the coefficients on ``z`` (``exog_break``) change across regimes
    and those on ``x`` (``exog``) do not.

    The procedure provides:

    1. Global minimizers of the sum of squared residuals for each number of
       breaks (dynamic programming)
    2. sup F(k|0) tests of no break against k breaks
    3. UDmax and WDmax tests of no break against an unknown number of breaks
    4. sup F(l+1|l) tests and the sequential procedure for the number of
       breaks
    5. Information criteria (BIC, LWZ) for the number of breaks
    6. Confidence intervals for the break dates (Bai 1997)

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    exog : ArrayLike | None
        Exogenous regressors whose coefficients do NOT break. A column that
        is collinear with the breaking regressors (for example a constant,
        when the constant is the breaking regressor) is dropped from the
        fixed regressors; ``BaiPerronResults.p`` reports the number used.
    exog_break : ArrayLike | None
        Regressors whose coefficients may break. If None, defaults to
        a constant (mean shift model).

    Examples
    --------
    >>> import numpy as np
    >>> from regimes import BaiPerronTest
    >>> np.random.seed(42)
    >>> # Simulate data with one break
    >>> n = 200
    >>> y = np.concatenate([
    ...     np.random.randn(100) + 0,  # regime 1
    ...     np.random.randn(100) + 2,  # regime 2
    ... ])
    >>> test = BaiPerronTest(y)
    >>> results = test.fit(max_breaks=3)
    >>> print(results.summary())

    Notes
    -----
    Critical values are the published tables of Bai and Perron (1998, 2003b),
    available for trimming 0.05, 0.10, 0.15, 0.20 and 0.25 and significance
    levels 10%, 5%, 2.5% and 1%. They apply to the F statistics "scaled by q"
    (not divided by the number of breaking regressors), which is how the
    statistics are reported here.

    The minimum regime length is ``h = floor(trimming * T)``, as in Bai and
    Perron (2003) and Perron's program, and at least ``q + 1`` (one more than
    the number of breaking regressors).

    The UDmax and WDmax critical values are tabulated for an upper bound of
    M = 5 breaks (trimming 0.05, 0.10, 0.15), M = 3 (0.20) and M = 2 (0.25).
    Bai and Perron (2003, Section 5.2) note that they vary little for larger
    M; with ``max_breaks`` below M the tests are conservative.

    A fixed regressor without variation inside a regime (an impulse or step
    dummy in ``exog``) is left out of the sup F(l+1|l) test within that
    regime. A model whose break dates make a fixed regressor a linear
    combination of the regime-specific ones (a step dummy in ``exog`` at an
    estimated break date) is not identified and is not used.
    """

    def __init__(
        self,
        endog: ArrayLike,
        exog: ArrayLike | None = None,
        exog_break: ArrayLike | None = None,
    ) -> None:
        """Initialize the Bai-Perron test."""
        # Default to constant (mean shift) if no breaking regressors specified
        if exog_break is None:
            exog_break = np.ones((len(endog), 1))

        super().__init__(endog, exog, exog_break)

    @classmethod
    def from_model(
        cls,
        model: OLS | AR | ADL,
        break_vars: Literal["all", "const"] = "all",
    ) -> BaiPerronTest:
        """Create BaiPerronTest from an OLS, AR, or ADL model.
        This class method provides a convenient way to create a Bai-Perron
        test directly from an existing model object, extracting the
        appropriate data for break testing.

        Parameters
        ----------
        model : OLS | AR | ADL
            Model to test for structural breaks.
        break_vars : "all" | "const"
            Which variables can have breaks:
            - "all": All regressors can break (default)
            - "const": Only intercept can break (mean-shift model)

        Returns
        -------
        BaiPerronTest
            Test instance ready for .fit()

        Examples
        --------
        >>> import numpy as np
        >>> from regimes import OLS, BaiPerronTest
        >>> np.random.seed(42)
        >>> n = 200
        >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
        >>> y = 1 + 0.5 * X[:, 1] + np.random.randn(n)
        >>> model = OLS(y, X, has_constant=False)
        >>> bp_results = BaiPerronTest.from_model(model).fit()
        >>> print(f"Detected breaks: {bp_results.break_indices}")

        Notes
        -----
        For AR and ADL models, the test uses the effective sample (after
        dropping initial observations for lags) and tests for breaks in all
        coefficients including the constant, AR parameters, and distributed
        lag parameters.
        """
        from regimes.models.adl import ADL as ADLModel
        from regimes.models.ar import AR
        from regimes.models.ols import OLS as OLSModel

        # Extract endog and exog from model
        if isinstance(model, AR):
            # For AR, build the design matrix to get effective sample
            y, X, _ = model._build_design_matrix()
            offset = int(model.maxlag)
            endog = y
            exog_all = X
        elif isinstance(model, ADLModel):
            # For ADL, build the design matrix to get effective sample
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
            # All regressors can break
            return cls._create(offset, endog, exog_break=exog_all)
        elif break_vars == "const":
            # Only constant breaks (mean-shift model)
            # exog contains all regressors (non-breaking)
            # exog_break defaults to constant

            # We need to separate the constant from other regressors
            # Check if constant exists in exog_all
            # Find ALL constant columns (there may be duplicates)
            const_indices: list[int] = []

            for i in range(exog_all.shape[1]):
                # Check for constant column (all ones)
                # We use a loose tolerance because of potential floating point issues
                if np.allclose(exog_all[:, i], 1.0, atol=1e-5):
                    const_indices.append(i)

            if const_indices:
                # Remove all constant columns from exog (non-breaking)
                non_breaking_indices = [
                    i for i in range(exog_all.shape[1]) if i not in const_indices
                ]
                if non_breaking_indices:
                    exog_non_break = exog_all[:, non_breaking_indices]
                else:
                    exog_non_break = None

                # exog_break will be a single constant column
                exog_break = np.ones((len(endog), 1))

                return cls._create(
                    offset, endog, exog=exog_non_break, exog_break=exog_break
                )
            else:
                # No constant found, so we just use exog_all as non-breaking
                # and let __init__ add a constant as exog_break
                return cls._create(offset, endog, exog=exog_all)
        else:
            raise ValueError(f"break_vars must be 'all' or 'const', got {break_vars!r}")

    @property
    def q(self) -> int:
        """Number of breaking regressors."""
        if self.exog_break is None:
            return 0
        return self.exog_break.shape[1]

    @property
    def p(self) -> int:
        """Number of non-breaking regressors used.

        Columns of ``exog`` that are linear combinations of the breaking
        regressors are not counted (see :meth:`fit`).
        """
        return len(self._fixed_columns())

    def _fixed_columns(self) -> list[int]:
        """Indices of the columns of ``exog`` that enter as fixed regressors."""
        assert self.exog_break is not None
        if self.exog is None or self.exog.shape[1] == 0:
            return []
        return _bp.independent_columns(self.exog_break, self.exog)

    def _data(
        self,
    ) -> tuple[
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]] | None,
    ]:
        """Return ``(y, z, x)``: dependent variable, breaking and fixed regressors.

        Columns of ``exog`` that are linear combinations of the breaking
        regressors (and of the fixed regressors kept so far) are left out of
        ``x``: such a regressor cannot have a fixed coefficient next to a
        breaking one. The usual case is a constant included in ``exog`` while
        the constant is also the breaking regressor. The decision does not
        depend on the units of the regressors.
        """
        assert self.exog_break is not None
        z = self.exog_break
        for label, arr in (
            ("endog", self.endog),
            ("exog_break", z),
            ("exog", self.exog),
        ):
            if arr is not None and not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"{label} contains missing or infinite values; the "
                    f"Bai-Perron procedure needs a sample without gaps"
                )
        cols = self._fixed_columns()
        x = self.exog[:, cols] if cols and self.exog is not None else None
        return self.endog, z, x

    def fit(  # type: ignore[override]  # explicit options instead of **kwargs
        self,
        max_breaks: int = 5,
        trimming: float = 0.15,
        selection: Literal["bic", "lwz", "sequential"] = "bic",
        *,
        level: float = 0.05,
        robust: bool = False,
        prewhiten: bool = False,
        hetdat: bool = True,
        hetvar: bool = True,
        hetomega: bool = True,
        hetq: bool = True,
        repartition: bool = False,
        h: int | None = None,
    ) -> BaiPerronResults:
        """Run the Bai-Perron procedure.

        Parameters
        ----------
        max_breaks : int
            Maximum number of breaks to consider. Default is 5. Reduced, with
            a warning, to the largest number for which critical values are
            tabulated at the chosen trimming (9, 8, 5, 3, 2 for trimming
            0.05, 0.10, 0.15, 0.20, 0.25).
        trimming : float
            Trimming parameter: the minimum regime length is
            ``h = max(floor(trimming * T), q + 1)``. Default is 0.15. Critical values are
            tabulated for 0.05, 0.10, 0.15, 0.20 and 0.25; with any other
            value the break dates and information criteria are still computed
            but critical values are NaN.
        selection : str
            Method for selecting the number of breaks:

            - "bic": Bayesian Information Criterion (default)
            - "lwz": Liu-Wu-Zidek modified Schwarz criterion
            - "sequential": sequential application of sup F(l+1|l) at
              ``level``. The break dates are then the sequentially estimated
              ones, not the global minimizers.
        level : float
            Significance level of the reported critical values and of the
            sequential procedure: 0.10, 0.05 (default), 0.025 or 0.01.
        robust : bool
            Allow serial correlation in the errors (HAC covariance:
            quadratic spectral kernel, Andrews (1991) bandwidth).
        prewhiten : bool
            VAR(1) prewhitening of the HAC estimator. Only used when
            ``robust`` is True.
        hetdat : bool
            Allow the regressors to have different distributions across
            regimes in the tests. Default True.
        hetvar : bool
            Allow the error variance to differ across regimes in the tests.
            Default True.
        hetomega : bool
            Allow the error variance to differ across regimes in the
            break-date confidence intervals. Default True.
        hetq : bool
            Allow the regressor moments to differ across regimes in the
            break-date confidence intervals. Default True.
        repartition : bool
            Apply the repartition step of Bai (1997) to the sequentially
            estimated break dates. Default False.
        h : int | None
            Minimum regime length (more than ``q``). Overrides the value
            implied by ``trimming``;
            ``trimming`` still selects the critical-value table.

        Returns
        -------
        BaiPerronResults
            Results object containing test statistics, break dates, and
            information criteria.

        Notes
        -----
        The defaults for the error assumptions (regime-specific variances, no
        serial correlation) are those of the OxRegimes package. Bai and
        Perron (2003, Section 5.5) recommend a trimming of at least 0.15 when
        heterogeneity is allowed and 0.20 when serial correlation is allowed,
        unless the sample is large.
        """
        nobs = self.nobs
        q = self.q
        if max_breaks < 1:
            raise ValueError(f"max_breaks must be at least 1, got {max_breaks}")
        y, z, x = self._data()
        # fixed regressors actually used (collinear columns of exog are dropped)
        p = 0 if x is None else x.shape[1]

        # Near-collinear regressors (after scaling each to unit length): the
        # break dates and SSRs are computed from an orthonormal basis, but the
        # covariance matrices behind the tests and intervals are not
        w_all = z if x is None else np.column_stack([z, x])
        norms = np.sqrt(np.einsum("ij,ij->j", w_all, w_all))
        norms[norms == 0] = 1.0
        cond = float(np.linalg.cond(w_all / norms))
        if cond > _COND_WARN:
            warnings.warn(
                f"The regressors are nearly collinear (condition number "
                f"{cond:.1e} after scaling): test statistics and confidence "
                f"intervals may be inaccurate. A regressor that varies little "
                f"around a large mean is the usual cause; centering it helps.",
                stacklevel=2,
            )

        if selection not in ("bic", "lwz", "sequential"):
            raise ValueError(f"Unknown selection method: {selection}")
        level_tab = _cv.match_level(level)
        if level_tab is None:
            raise ValueError(
                f"level must be one of {_cv.LEVELS} (tabulated significance "
                f"levels), got {level}"
            )
        opt = BPOptions(
            robust=robust,
            prewhiten=prewhiten,
            hetdat=hetdat,
            hetvar=hetvar,
            hetomega=hetomega,
            hetq=hetq,
        )
        if not hetdat and (robust or p > 0):
            raise ValueError(
                "hetdat=False is only available without serial correlation "
                "(robust=False) and without fixed regressors (exog)"
            )

        # Minimum regime length
        if h is None:
            h = _bp.segment_length(trimming, nobs, q)
        elif h <= q:
            raise ValueError(f"h must exceed the number of breaking regressors ({q})")

        # Maximum feasible breaks
        max_feasible = nobs // h - 1
        max_breaks = min(max_breaks, max_feasible)

        if max_breaks < 1:
            raise ValueError(
                f"Cannot test for breaks: trimming too large or T too small. "
                f"Need at least {2 * h} observations, have {nobs}."
            )

        trim_tab = _cv.match_trimming(trimming)
        if trim_tab is None:
            warnings.warn(
                f"Critical values are tabulated for trimming {_cv.TRIMMINGS} "
                f"only; with trimming={trimming} break dates and information "
                f"criteria are computed but critical values are not available.",
                stacklevel=2,
            )
            if selection == "sequential":
                raise ValueError(
                    "selection='sequential' needs tabulated critical values: "
                    f"trimming must be one of {_cv.TRIMMINGS}"
                )
        else:
            max_tab = _cv.MAX_BREAKS[trim_tab]
            if max_breaks > max_tab:
                warnings.warn(
                    f"At most {max_tab} breaks are tabulated for trimming "
                    f"{trim_tab}; max_breaks reduced from {max_breaks}.",
                    stacklevel=2,
                )
                max_breaks = max_tab
        if q > _cv.MAX_Q and trim_tab is not None:
            warnings.warn(
                f"Critical values are tabulated up to q = {_cv.MAX_Q} breaking "
                f"regressors; the values for q = {_cv.MAX_Q} are used.",
                stacklevel=2,
            )
        qi = min(q, _cv.MAX_Q) - 1

        # Global minimizers for 1..max_breaks breaks
        if x is not None:
            parts, ssr_m, ssr0 = _bp.partial_partition_all(y, z, x, max_breaks, h)
        else:
            parts, ssr_m, ssr0 = _bp.partition_all(
                _bp.ssr_triangle(y, z, h), max_breaks, h
            )
        n_ok = 0
        while n_ok < max_breaks and parts[n_ok].size == n_ok + 1:
            n_ok += 1
        if n_ok < 1:
            raise ValueError(
                "No admissible partition: the breaking regressors have no "
                "variation within regimes of the minimum length."
            )
        if n_ok < max_breaks:
            warnings.warn(
                f"No admissible partition with more than {n_ok} break(s); "
                f"max_breaks reduced from {max_breaks}.",
                stacklevel=2,
            )
        max_breaks = n_ok

        # An exact fit leaves nothing to test (and the statistics are 0/0)
        yy = float(y @ y)
        if not ssr0 > _EXACT_FIT * yy:
            raise ValueError(
                "The model without breaks fits the data exactly; there is "
                "nothing to test."
            )
        for m in range(1, max_breaks + 1):
            if not float(ssr_m[m - 1]) > _EXACT_FIT * yy:
                raise ValueError(
                    f"The model with {m} break(s) fits the data exactly (zero "
                    f"residual variance); the tests are not defined."
                )

        ssr_vals: dict[int, float] = {0: ssr0}
        breaks_by_m: dict[int, Sequence[int]] = {0: []}
        for m in range(1, max_breaks + 1):
            ssr_vals[m] = float(ssr_m[m - 1])
            breaks_by_m[m] = [int(b) + 1 for b in parts[m - 1]]

        # sup F(k|0) at the global minimizers
        supf_stats = {
            k: _bp.sup_f(y, z, x, parts[k - 1], opt) for k in range(1, max_breaks + 1)
        }
        udmax = max(supf_stats.values())

        # Critical values at all tabulated levels; WDmax (weights c(1)/c(k))
        cv_supf: dict[float, dict[int, float]] = {}
        cv_seqf: dict[float, dict[int, float]] = {}
        cv_ud: dict[float, float] = {}
        cv_wd: dict[float, float] = {}
        wdmax_stats: dict[float, float] = {}
        for lev in _cv.LEVELS:
            t_sup = _cv.supf_table(lev, trimming)
            t_seq = _cv.seqf_table(lev, trimming)
            t_dmax = _cv.dmax_table(lev, trimming)
            if t_sup is None or t_seq is None or t_dmax is None:
                cv_supf[lev] = dict.fromkeys(range(1, max_breaks + 1), np.nan)
                cv_seqf[lev] = dict.fromkeys(range(max_breaks), np.nan)
                cv_ud[lev] = cv_wd[lev] = wdmax_stats[lev] = np.nan
                continue
            cv_supf[lev] = {
                k: float(t_sup[qi, k - 1]) for k in range(1, max_breaks + 1)
            }
            cv_seqf[lev] = {
                ell: float(t_seq[qi, ell]) if ell < t_seq.shape[1] else np.nan
                for ell in range(max_breaks)
            }
            cv_ud[lev], cv_wd[lev] = float(t_dmax[qi, 0]), float(t_dmax[qi, 1])
            wdmax_stats[lev] = max(
                t_sup[qi, 0] / t_sup[qi, k - 1] * supf_stats[k]
                for k in range(1, max_breaks + 1)
            )

        # sup F(l+1|l) on the global l-break partitions; l = 0 is sup F(1|0)
        seqf_stats: dict[int, float] = {}
        seqf_break: dict[int, int] = {}
        stat0, at0 = _bp.one_break_sup_f(y, z, x, 0, nobs - 1, h, opt)
        seqf_stats[0], seqf_break[0] = float(stat0), int(at0) + 1 if at0 >= 0 else -1
        for ell in range(1, max_breaks):
            stat, at = _bp.seq_f(y, z, x, parts[ell - 1], h, opt)
            seqf_stats[ell] = float(stat)
            seqf_break[ell] = int(at) + 1 if at >= 0 else -1

        # Information criteria
        bic_arr, lwz_arr = _bp.info_criteria(
            np.array([ssr_vals[m] for m in range(max_breaks + 1)]), nobs, q, p
        )
        bic_vals = {m: float(bic_arr[m]) for m in range(max_breaks + 1)}
        lwz_vals = {m: float(lwz_arr[m]) for m in range(max_breaks + 1)}

        # Sequential procedure
        seq_n: int | None = None
        seq_vb: NDArray[np.integer[Any]] = np.empty(0, dtype=np.int64)
        if trim_tab is not None:
            seq_n, seq_vb = _bp.sequential(
                y, z, x, max_breaks, h, trimming, level_tab, opt, repartition
            )

        # Select number of breaks
        vb: NDArray[np.integer[Any]]
        if selection == "bic":
            n_breaks = int(np.nanargmin(bic_arr))
            vb = parts[n_breaks - 1] if n_breaks > 0 else seq_vb[:0]
        elif selection == "lwz":
            n_breaks = int(np.nanargmin(lwz_arr))
            vb = parts[n_breaks - 1] if n_breaks > 0 else seq_vb[:0]
        else:
            assert seq_n is not None
            n_breaks, vb = seq_n, seq_vb

        break_indices = [int(b) + 1 for b in vb]

        # Confidence intervals for the selected break dates
        break_ci: dict[int, tuple[int, int]] = {}
        break_ci_90: dict[int, tuple[int, int]] = {}
        if n_breaks > 0:
            try:
                ci = _bp.break_ci(y, z, x, vb, opt)
            except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
                warnings.warn(
                    f"Break-date confidence intervals could not be computed: {exc}",
                    stacklevel=2,
                )
            else:
                for i, b_idx in enumerate(break_indices):
                    lo95, hi95, lo90, hi90 = (
                        int(np.clip(c + 1, 0, nobs - 1)) for c in ci[i]
                    )
                    break_ci[b_idx] = (lo95, hi95)
                    break_ci_90[b_idx] = (lo90, hi90)

        results = BaiPerronResults(
            test_name="Bai-Perron",
            nobs=nobs,
            n_breaks=n_breaks,
            break_indices=break_indices,
            supf_stats=supf_stats,
            supf_critical=cv_supf[level_tab],
            udmax=udmax,
            udmax_critical=cv_ud[level_tab],
            wdmax=wdmax_stats[level_tab],
            wdmax_critical=cv_wd[level_tab],
            wdmax_stats=wdmax_stats,
            seqf_stats=seqf_stats,
            seqf_critical=cv_seqf[level_tab],
            seqf_break=seqf_break,
            bic=bic_vals,
            lwz=lwz_vals,
            ssr=ssr_vals,
            breaks_by_m=breaks_by_m,
            sequential_n_breaks=seq_n,
            sequential_breaks=[int(b) + 1 for b in seq_vb],
            trimming=trimming,
            h=h,
            max_breaks=max_breaks,
            selection_method=selection,
            level=level_tab,
            q=q,
            p=p,
            break_ci=break_ci,
            break_ci_90=break_ci_90,
            critical_values={
                "supf": cv_supf,
                "seqf": cv_seqf,
                "udmax": cv_ud,
                "wdmax": cv_wd,
            },
            options=opt,
            repartition=repartition,
            _test=self,
        )
        return _shift_break_dates(results, self.index_offset)

    def get_regime_estimates(
        self, break_indices: Sequence[int]
    ) -> list[tuple[NDArray[np.floating[Any]], float]]:
        """Get parameter estimates for each regime.

        The model is estimated jointly given the break dates, so that with
        fixed regressors (``exog``) their coefficients are common to all
        regimes.

        Parameters
        ----------
        break_indices : Sequence[int]
            Break dates (first observation of each new regime).

        Returns
        -------
        list[tuple[NDArray[np.floating], float]]
            One ``(coefficients, ssr)`` tuple per regime: the regime's
            coefficients on the breaking regressors and the sum of squared
            residuals within the regime.
        """
        y, z, x = self._data()
        q = self.q
        vb = np.asarray(list(break_indices), dtype=np.int64) - 1
        delta, _, resid, _, _ = _bp.estimate(y, z, x, vb)
        bd = _bp.bounds(vb, self.nobs)
        results = []
        for i in range(len(bd) - 1):
            u = resid[bd[i] : bd[i + 1]]
            results.append((delta[i * q : (i + 1) * q], float(u @ u)))
        return results
