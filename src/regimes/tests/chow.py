"""Chow test for structural breaks at known break points.

This module implements the Chow (1960) test for testing whether regression
coefficients change at a known break point. Both the standard and predictive
variants are supported.

References
----------
Chow, G. C. (1960). Tests of equality between sets of coefficients in two
    linear regressions. Econometrica, 28(3), 591-605.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import stats

from regimes.tests.base import BreakTestBase, BreakTestResultsBase

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    from regimes.models.adl import ADL
    from regimes.models.ar import AR
    from regimes.models.ols import OLS


@dataclass
class ChowTestResults(BreakTestResultsBase):
    """Results from Chow structural break test.

    Each candidate break point is tested individually. The ``break_indices``
    property (inherited from base class) contains only those break points
    where the null hypothesis of no break was rejected.

    Attributes
    ----------
    test_name : str
        Name of the test ("Chow").
    nobs : int
        Number of observations.
    n_breaks : int
        Number of breaks where H0 was rejected.
    break_indices : Sequence[int]
        Break points where H0 was rejected.
    f_stats : dict[int, float]
        F-statistic for each tested break point.
    p_values : dict[int, float]
        P-value for each tested break point.
    df_num : dict[int, int]
        Numerator degrees of freedom for each test.
    df_denom : dict[int, int]
        Denominator degrees of freedom for each test.
    ssr_full : float
        Sum of squared residuals from the restricted (full-sample) model.
    ssr_unrestricted : dict[int, float]
        Sum of squared residuals from the unrestricted model (SSR_1 + SSR_2)
        for each tested break point.
    is_predictive : dict[int, bool]
        Whether the predictive variant was used for each break point.
    n_params : int
        Number of parameters (breaking regressors).
    significance_level : float
        Significance level used for rejection decisions.
    _test : ChowTest | None
        Reference back to the test object (not shown in repr).
    """

    f_stats: dict[int, float] = field(default_factory=dict)
    p_values: dict[int, float] = field(default_factory=dict)
    df_num: dict[int, int] = field(default_factory=dict)
    df_denom: dict[int, int] = field(default_factory=dict)
    ssr_full: float = np.nan
    ssr_unrestricted: dict[int, float] = field(default_factory=dict)
    is_predictive: dict[int, bool] = field(default_factory=dict)
    n_params: int = 0
    significance_level: float = 0.05
    _test: ChowTest | None = field(default=None, repr=False)

    @property
    def rejected(self) -> dict[int, bool]:
        """Whether H0 was rejected at each tested break point."""
        return {bp: self.p_values[bp] < self.significance_level for bp in self.p_values}

    @property
    def tested_break_points(self) -> list[int]:
        """All break points that were tested (regardless of rejection)."""
        return sorted(self.f_stats.keys())

    def summary(self) -> str:
        """Generate a text summary of Chow test results.

        Returns
        -------
        str
            Formatted summary including F-statistics, p-values, and
            rejection decisions for each tested break point.
        """
        lines = []
        lines.append("=" * 78)
        lines.append(f"{'Chow Structural Break Test':^78}")
        lines.append("=" * 78)
        lines.append(f"Number of observations:   {self.nobs:>10}")
        lines.append(f"Number of parameters:     {self.n_params:>10}")
        lines.append(f"Significance level:       {self.significance_level:>10.3f}")
        lines.append(f"SSR (full sample):        {self.ssr_full:>10.4f}")
        lines.append("-" * 78)

        lines.append(
            f"\n{'Break':>8} {'F-stat':>12} {'df1':>6} {'df2':>6} "
            f"{'p-value':>12} {'Reject':>8} {'Type':>12}"
        )
        lines.append("-" * 66)

        for bp in sorted(self.f_stats.keys()):
            f_stat = self.f_stats[bp]
            p_val = self.p_values[bp]
            d1 = self.df_num[bp]
            d2 = self.df_denom[bp]
            reject = "Yes" if p_val < self.significance_level else "No"
            variant = "Predictive" if self.is_predictive[bp] else "Standard"
            lines.append(
                f"{bp:>8} {f_stat:>12.4f} {d1:>6} {d2:>6} "
                f"{p_val:>12.6f} {reject:>8} {variant:>12}"
            )

        lines.append("-" * 78)
        n_rejected = sum(1 for v in self.rejected.values() if v)
        n_tested = len(self.f_stats)
        lines.append(
            f"\nRejected H0 at {n_rejected} of {n_tested} tested break point(s)"
        )

        if self.break_indices:
            lines.append(f"Significant break(s) at: {list(self.break_indices)}")

        lines.append("=" * 78)
        return "\n".join(lines)


class ChowTest(BreakTestBase):
    """Chow test for structural breaks at known break points.

    Tests whether regression coefficients change at one or more specified
    break points. Each break point is tested individually with its own
    F-test.

    The standard Chow test requires both sub-samples to have at least k
    observations (where k is the number of parameters). When a sub-sample
    has fewer than k observations, the predictive variant is used
    automatically.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    exog : ArrayLike | None
        Exogenous regressors whose coefficients do NOT break.
    exog_break : ArrayLike | None
        Regressors whose coefficients may break. If None, defaults to
        a constant (mean-shift model).

    Examples
    --------
    >>> import numpy as np
    >>> from regimes import ChowTest
    >>> rng = np.random.default_rng(42)
    >>> y = np.concatenate([rng.normal(0, 1, 100), rng.normal(2, 1, 100)])
    >>> test = ChowTest(y)
    >>> results = test.fit(break_points=100)
    >>> print(results.summary())

    Notes
    -----
    The Chow test assumes:
    - Normally distributed errors (for exact F distribution)
    - Homoskedastic errors across regimes
    - Independent errors

    The test is valid asymptotically under weaker assumptions.

    References
    ----------
    Chow, G. C. (1960). Tests of equality between sets of coefficients in
        two linear regressions. Econometrica, 28(3), 591-605.
    """

    def __init__(
        self,
        endog: ArrayLike,
        exog: ArrayLike | None = None,
        exog_break: ArrayLike | None = None,
    ) -> None:
        """Initialize the Chow test."""
        # Default to constant (mean shift) if no breaking regressors specified
        if exog_break is None:
            exog_break = np.ones((len(np.asarray(endog)), 1))

        super().__init__(endog, exog, exog_break)

    @classmethod
    def from_model(
        cls,
        model: OLS | AR | ADL,
        break_vars: Literal["all", "const"] = "all",
    ) -> ChowTest:
        """Create ChowTest from an OLS, AR, or ADL model.

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
        ChowTest
            Test instance ready for .fit()

        Notes
        -----
        For AR and ADL models, the test uses the effective sample (after
        dropping initial observations for lags).
        """
        from regimes.models.adl import ADL as ADLModel
        from regimes.models.ar import AR
        from regimes.models.ols import OLS as OLSModel

        # Extract endog and exog from model
        if isinstance(model, (AR, ADLModel)):
            y, X, _ = model._build_design_matrix()
            endog = y
            exog_all = X
        elif isinstance(model, OLSModel):
            endog = model.endog
            exog_all = model.exog
        else:
            raise TypeError(
                f"model must be OLS, AR, or ADL, got {type(model).__name__}"
            )

        if exog_all is None:
            exog_all = np.ones((len(endog), 1))

        if break_vars == "all":
            return cls(endog, exog_break=exog_all)
        elif break_vars == "const":
            return cls(endog, exog=exog_all)
        else:
            raise ValueError(f"break_vars must be 'all' or 'const', got {break_vars!r}")

    @property
    def n_break_params(self) -> int:
        """Number of breaking regressors (k)."""
        if self.exog_break is None:
            return 0
        return self.exog_break.shape[1]

    @property
    def n_nonbreak_params(self) -> int:
        """Number of non-breaking regressors."""
        if self.exog is None:
            return 0
        return self.exog.shape[1]

    def _build_regressor_matrix(
        self,
        start: int,
        end: int,
    ) -> NDArray[np.floating[Any]]:
        """Build the full regressor matrix for a segment [start, end).

        For the unrestricted model, only the breaking regressors are
        segment-specific. Non-breaking regressors are included as-is.

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
        try:
            beta, residuals, _rank, _s = np.linalg.lstsq(X, y, rcond=None)
            if len(residuals) > 0:
                return float(residuals[0])
            else:
                return float(np.sum((y - X @ beta) ** 2))
        except np.linalg.LinAlgError:
            return np.inf

    def _test_single_break(
        self,
        break_point: int,
    ) -> dict[str, Any]:
        """Test a single break point.

        Parameters
        ----------
        break_point : int
            The index at which to test for a break. Observations
            [0, break_point) form the first sub-sample, and
            [break_point, T) form the second.

        Returns
        -------
        dict
            Dictionary with keys: f_stat, p_value, df_num, df_denom,
            ssr_unrestricted, is_predictive.
        """
        T = self.nobs
        k = self.n_break_params  # number of breaking regressors
        p = self.n_nonbreak_params  # number of non-breaking regressors

        # Total parameters per regime: k breaking + p non-breaking
        # But for the restricted model, we have k + p parameters
        # For the unrestricted model with partial breaks:
        #   - p non-breaking params (common)
        #   - k breaking params per sub-sample (2k total)

        n1 = break_point
        n2 = T - break_point

        # Build full-sample regressor matrix and compute SSR_full
        X_full = self._build_regressor_matrix(0, T)
        y_full = self.endog
        ssr_full = self._compute_ssr(y_full, X_full)

        # Determine if we need the predictive variant
        total_params_per_sub = k + p  # each sub-sample needs k + p params
        use_predictive = n1 < total_params_per_sub or n2 < total_params_per_sub

        if use_predictive:
            # Predictive Chow test
            # Determine which sub-sample is "too small"
            if n1 >= total_params_per_sub:
                # Sub-sample 1 is large enough, sub-sample 2 is small
                X1 = self._build_regressor_matrix(0, break_point)
                y1 = self.endog[:break_point]
                ssr1 = self._compute_ssr(y1, X1)
                n_small = n2
                n_large = n1
                ssr_large = ssr1
            else:
                # Sub-sample 2 is large enough, sub-sample 1 is small
                X2 = self._build_regressor_matrix(break_point, T)
                y2 = self.endog[break_point:]
                ssr2 = self._compute_ssr(y2, X2)
                n_small = n1
                n_large = n2
                ssr_large = ssr2

            # F = ((SSR_full - SSR_large) / n_small) / (SSR_large / (n_large - k - p))
            df_num = n_small
            df_denom = n_large - total_params_per_sub

            if df_denom <= 0 or ssr_large <= 0:
                return {
                    "f_stat": 0.0,
                    "p_value": 1.0,
                    "df_num": df_num,
                    "df_denom": max(df_denom, 1),
                    "ssr_unrestricted": ssr_large,
                    "is_predictive": True,
                }

            f_stat = ((ssr_full - ssr_large) / df_num) / (ssr_large / df_denom)
            f_stat = max(0.0, f_stat)
            p_value = 1.0 - stats.f.cdf(f_stat, df_num, df_denom)

            return {
                "f_stat": f_stat,
                "p_value": float(p_value),
                "df_num": df_num,
                "df_denom": df_denom,
                "ssr_unrestricted": ssr_large,
                "is_predictive": True,
            }

        else:
            # Standard Chow test
            # Compute SSR for each sub-sample
            if self.exog is not None and self.exog_break is not None:
                # Partial coefficient testing: non-breaking regressors
                # constrained to be equal across regimes
                # Build block-diagonal design matrix for unrestricted model
                X1_break = self.exog_break[:break_point]
                X2_break = self.exog_break[break_point:]
                X_nonbreak = self.exog

                # Unrestricted model:
                # y = [X_nonbreak] * gamma + [X1_break, 0; 0, X2_break] * [beta1; beta2]
                Z1_break = np.zeros((T, k))
                Z1_break[:break_point] = X1_break
                Z2_break = np.zeros((T, k))
                Z2_break[break_point:] = X2_break

                X_unrestricted = np.column_stack([Z1_break, Z2_break, X_nonbreak])
                ssr_unrestricted = self._compute_ssr(y_full, X_unrestricted)
            else:
                # All coefficients break (no non-breaking regressors)
                X1 = self._build_regressor_matrix(0, break_point)
                X2 = self._build_regressor_matrix(break_point, T)
                y1 = self.endog[:break_point]
                y2 = self.endog[break_point:]

                ssr1 = self._compute_ssr(y1, X1)
                ssr2 = self._compute_ssr(y2, X2)
                ssr_unrestricted = ssr1 + ssr2

            # F = ((SSR_full - SSR_unrestricted) / k) / (SSR_unrestricted / (T - 2k - p))
            df_num = k
            df_denom = T - 2 * k - p

            if df_denom <= 0 or ssr_unrestricted <= 0:
                return {
                    "f_stat": 0.0,
                    "p_value": 1.0,
                    "df_num": df_num,
                    "df_denom": max(df_denom, 1),
                    "ssr_unrestricted": ssr_unrestricted,
                    "is_predictive": False,
                }

            f_stat = ((ssr_full - ssr_unrestricted) / df_num) / (
                ssr_unrestricted / df_denom
            )
            f_stat = max(0.0, f_stat)
            p_value = 1.0 - stats.f.cdf(f_stat, df_num, df_denom)

            return {
                "f_stat": f_stat,
                "p_value": float(p_value),
                "df_num": df_num,
                "df_denom": df_denom,
                "ssr_unrestricted": ssr_unrestricted,
                "is_predictive": False,
            }

    def fit(
        self,
        break_points: int | Sequence[int],
        significance: float = 0.05,
        **kwargs: Any,
    ) -> ChowTestResults:
        """Perform the Chow test at specified break points.

        Each break point is tested individually. The standard Chow test is
        used when both sub-samples have enough observations; the predictive
        variant is used automatically otherwise.

        Parameters
        ----------
        break_points : int | Sequence[int]
            One or more break point indices to test. Each index τ splits
            the sample into [0, τ) and [τ, T).
        significance : float
            Significance level for rejection decisions. Default is 0.05.
        **kwargs
            Additional arguments (reserved for future use).

        Returns
        -------
        ChowTestResults
            Results object with F-statistics, p-values, and rejection
            decisions for each tested break point.

        Raises
        ------
        ValueError
            If a break point is outside the valid range [1, T-1].
        """
        # Normalize to list
        if isinstance(break_points, (int, np.integer)):
            break_points = [int(break_points)]
        else:
            break_points = [int(bp) for bp in break_points]

        T = self.nobs

        # Validate break points
        for bp in break_points:
            if bp < 1 or bp >= T:
                raise ValueError(
                    f"Break point {bp} is outside the valid range [1, {T - 1}]"
                )

        # Compute full-sample SSR
        X_full = self._build_regressor_matrix(0, T)
        ssr_full = self._compute_ssr(self.endog, X_full)

        # Test each break point
        f_stats: dict[int, float] = {}
        p_values: dict[int, float] = {}
        df_num: dict[int, int] = {}
        df_denom: dict[int, int] = {}
        ssr_unrestricted: dict[int, float] = {}
        is_predictive: dict[int, bool] = {}

        for bp in break_points:
            result = self._test_single_break(bp)
            f_stats[bp] = result["f_stat"]
            p_values[bp] = result["p_value"]
            df_num[bp] = result["df_num"]
            df_denom[bp] = result["df_denom"]
            ssr_unrestricted[bp] = result["ssr_unrestricted"]
            is_predictive[bp] = result["is_predictive"]

        # Determine which breaks are significant
        significant_breaks = sorted(
            bp for bp in break_points if p_values[bp] < significance
        )

        return ChowTestResults(
            test_name="Chow",
            nobs=T,
            n_breaks=len(significant_breaks),
            break_indices=significant_breaks,
            f_stats=f_stats,
            p_values=p_values,
            df_num=df_num,
            df_denom=df_denom,
            ssr_full=ssr_full,
            ssr_unrestricted=ssr_unrestricted,
            is_predictive=is_predictive,
            n_params=self.n_break_params,
            significance_level=significance,
            _test=self,
        )
