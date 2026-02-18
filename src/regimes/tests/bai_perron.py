"""Bai-Perron test for multiple structural breaks.

This module implements the Bai-Perron (1998, 2003) procedure for testing
and estimating multiple structural breaks in linear regression models.

References
----------
Bai, J., & Perron, P. (1998). Estimating and testing linear models with
    multiple structural changes. Econometrica, 66(1), 47-78.

Bai, J., & Perron, P. (2003). Computation and analysis of multiple
    structural change models. Journal of Applied Econometrics, 18(1), 1-22.
"""

from __future__ import annotations

import warnings
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
    from regimes.models.base import CovType
    from regimes.models.ols import OLS, OLSResults


# Critical values for Bai-Perron tests (approximations from tables)
# These are for 5% significance level, trimming = 0.15
# Keys are (q, m) where q = number of breaking regressors, m = number of breaks
_SUPF_CRITICAL_VALUES: dict[tuple[int, int], float] = {
    # q=1
    (1, 1): 8.58,
    (1, 2): 7.22,
    (1, 3): 5.96,
    (1, 4): 4.99,
    (1, 5): 4.09,
    # q=2
    (2, 1): 10.13,
    (2, 2): 8.51,
    (2, 3): 7.42,
    (2, 4): 6.38,
    (2, 5): 5.60,
    # q=3
    (3, 1): 11.47,
    (3, 2): 9.75,
    (3, 3): 8.36,
    (3, 4): 7.42,
    (3, 5): 6.57,
    # q=4
    (4, 1): 12.77,
    (4, 2): 10.58,
    (4, 3): 9.35,
    (4, 4): 8.19,
    (4, 5): 7.45,
    # q=5
    (5, 1): 13.96,
    (5, 2): 11.65,
    (5, 3): 10.18,
    (5, 4): 9.12,
    (5, 5): 8.26,
}

# UDmax critical values at 5% (from Bai-Perron tables)
_UDMAX_CRITICAL_VALUES: dict[int, float] = {
    1: 8.88,
    2: 10.55,
    3: 11.70,
    4: 12.81,
    5: 13.83,
}


@dataclass
class BaiPerronResults(BreakTestResultsBase):
    """Results from Bai-Perron structural break test.

    Attributes
    ----------
    test_name : str
        Name of the test ("Bai-Perron").
    nobs : int
        Number of observations.
    n_breaks : int
        Number of breaks selected (by specified criterion).
    break_indices : Sequence[int]
        Estimated break point indices.
    supf_stats : dict[int, float]
        Sup-F statistics for testing m breaks vs 0 breaks.
    supf_pvalues : dict[int, float]
        P-values for Sup-F tests.
    udmax : float
        UDmax statistic (unweighted double maximum).
    wdmax : float
        WDmax statistic (weighted double maximum).
    seqf_stats : dict[int, float]
        Sequential Sup-F statistics for testing m+1 vs m breaks.
    bic : dict[int, float]
        BIC values for models with different numbers of breaks.
    lwz : dict[int, float]
        LWZ (modified Schwarz) values for different numbers of breaks.
    ssr : dict[int, float]
        Sum of squared residuals for different numbers of breaks.
    breaks_by_m : dict[int, Sequence[int]]
        Optimal break locations for each number of breaks m.
    trimming : float
        Trimming parameter used.
    max_breaks : int
        Maximum number of breaks considered.
    selection_method : str
        Method used to select number of breaks.
    """

    supf_stats: dict[int, float] = field(default_factory=dict)
    supf_pvalues: dict[int, float] = field(default_factory=dict)
    supf_critical: dict[int, float] = field(default_factory=dict)
    udmax: float = np.nan
    udmax_critical: float = np.nan
    wdmax: float = np.nan
    seqf_stats: dict[int, float] = field(default_factory=dict)
    seqf_critical: dict[int, float] = field(default_factory=dict)
    bic: dict[int, float] = field(default_factory=dict)
    lwz: dict[int, float] = field(default_factory=dict)
    ssr: dict[int, float] = field(default_factory=dict)
    breaks_by_m: dict[int, Sequence[int]] = field(default_factory=dict)
    trimming: float = 0.15
    max_breaks: int = 5
    selection_method: str = "bic"
    q: int = 1  # Number of breaking regressors
    break_ci: dict[int, tuple[int, int]] = field(default_factory=dict)

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

        # Non-breaking regressors
        if self._test.exog is not None:
            exog_list.append(self._test.exog)
            col_idx += self._test.exog.shape[1]

        # Breaking regressors
        if self._test.exog_break is not None:
            exog_list.append(self._test.exog_break)
            n_break_cols = self._test.exog_break.shape[1]
            # Add breaks for these columns
            for i in range(n_break_cols):
                variable_breaks[col_idx + i] = list(self.break_indices)
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

    def summary(self) -> str:
        """Generate a text summary of Bai-Perron test results.

        Returns
        -------
        str
            Formatted summary including test statistics and break dates.
        """
        lines = []
        lines.append("=" * 78)
        lines.append(f"{'Bai-Perron Multiple Structural Break Test':^78}")
        lines.append("=" * 78)
        lines.append(f"Number of observations:   {self.nobs:>10}")
        lines.append(f"Trimming parameter:       {self.trimming:>10.2f}")
        lines.append(f"Maximum breaks tested:    {self.max_breaks:>10}")
        lines.append(f"Breaking regressors (q):  {self.q:>10}")
        lines.append("-" * 78)

        # Sup-F tests
        lines.append("\nSup-F Tests (H0: 0 breaks vs H1: m breaks):")
        lines.append(
            f"{'m':>5} {'Sup-F':>12} {'Critical':>12} {'p-value':>12} {'Reject':>10}"
        )
        lines.append("-" * 51)
        for m in sorted(self.supf_stats.keys()):
            stat = self.supf_stats[m]
            crit = self.supf_critical.get(m, np.nan)
            pval = self.supf_pvalues.get(m, np.nan)
            reject = "Yes" if stat > crit else "No"
            pval_str = f"{pval:.4f}" if not np.isnan(pval) else "N/A"
            lines.append(
                f"{m:>5} {stat:>12.3f} {crit:>12.3f} {pval_str:>12} {reject:>10}"
            )

        # UDmax test
        lines.append(
            f"\nUDmax statistic: {self.udmax:.3f} (5% critical: {self.udmax_critical:.3f})"
        )
        if self.udmax > self.udmax_critical:
            lines.append("  => Reject null of no breaks")
        else:
            lines.append("  => Cannot reject null of no breaks")

        # Sequential tests
        if self.seqf_stats:
            lines.append("\nSequential Sup-F Tests (H0: m breaks vs H1: m+1 breaks):")
            lines.append(
                f"{'m':>5} {'m+1':>5} {'Seq-F':>12} {'Critical':>12} {'Reject':>10}"
            )
            lines.append("-" * 46)
            for m in sorted(self.seqf_stats.keys()):
                stat = self.seqf_stats[m]
                crit = self.seqf_critical.get(m, np.nan)
                reject = "Yes" if stat > crit else "No"
                lines.append(
                    f"{m:>5} {m + 1:>5} {stat:>12.3f} {crit:>12.3f} {reject:>10}"
                )

        # Information criteria
        lines.append("\nInformation Criteria:")
        lines.append(f"{'m':>5} {'SSR':>15} {'BIC':>15} {'LWZ':>15}")
        lines.append("-" * 52)
        for m in sorted(self.bic.keys()):
            ssr_val = self.ssr.get(m, np.nan)
            bic_val = self.bic[m]
            lwz_val = self.lwz.get(m, np.nan)
            lines.append(f"{m:>5} {ssr_val:>15.4f} {bic_val:>15.4f} {lwz_val:>15.4f}")

        # Selected breaks
        lines.append("-" * 78)
        lines.append(
            f"\nSelected number of breaks ({self.selection_method.upper()}): {self.n_breaks}"
        )

        if self.n_breaks > 0:
            lines.append(f"Break dates: {list(self.break_indices)}")

            if self.break_ci:
                lines.append("\n95% Confidence Intervals for Break Dates:")
                for break_idx in self.break_indices:
                    if break_idx in self.break_ci:
                        lower, upper = self.break_ci[break_idx]
                        lines.append(f"  Break at {break_idx}: [{lower}, {upper}]")

            lines.append("\nBreak locations by number of breaks:")
            for m, breaks in sorted(self.breaks_by_m.items()):
                if m > 0:
                    lines.append(f"  m={m}: {list(breaks)}")

        lines.append("=" * 78)
        return "\n".join(lines)


class BaiPerronTest(BreakTestBase):
    """Bai-Perron test for multiple structural breaks.

    Implements the Bai-Perron (1998, 2003) procedure for testing and
    estimating multiple structural breaks in linear regression models.

    The test provides:
    1. Sup-F tests for m breaks vs 0 breaks
    2. UDmax and WDmax tests for presence of any breaks
    3. Sequential Sup-F tests for determining the number of breaks
    4. Information criteria (BIC, LWZ) for model selection
    5. Optimal break locations via dynamic programming

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    exog : ArrayLike | None
        Exogenous regressors whose coefficients do NOT break.
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
    The implementation uses dynamic programming (Bai-Perron algorithm)
    to efficiently compute optimal break locations. The complexity is
    O(T^2) for a given number of breaks.

    Critical values are approximations from the original Bai-Perron
    tables. For more precise inference, bootstrap methods can be used.
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
        """
        from regimes.models.adl import ADL as ADLModel
        from regimes.models.ar import AR
        from regimes.models.ols import OLS as OLSModel

        # Extract endog and exog from model
        if isinstance(model, AR):
            # For AR, build the design matrix to get effective sample
            y, X, _ = model._build_design_matrix()
            endog = y
            exog_all = X
        elif isinstance(model, ADLModel):
            # For ADL, build the design matrix to get effective sample
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
            # All regressors can break
            return cls(endog, exog_break=exog_all)
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
                
                return cls(endog, exog=exog_non_break, exog_break=exog_break)
            else:
                # No constant found, so we just use exog_all as non-breaking
                # and let __init__ add a constant as exog_break
                return cls(endog, exog=exog_all)
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
        """Number of non-breaking regressors."""
        if self.exog is None:
            return 0
        return self.exog.shape[1]

    def _compute_ssr_segment(
        self,
        start: int,
        end: int,
        y: NDArray[np.floating[Any]] | None = None,
        x: NDArray[np.floating[Any]] | None = None,
    ) -> tuple[float, NDArray[np.floating[Any]]]:
        """Compute SSR for a segment [start, end).

        Parameters
        ----------
        start : int
            Start index (inclusive).
        end : int
            End index (exclusive).
        y : NDArray | None
            Dependent variable override. If None, uses self.endog.
        x : NDArray | None
            Regressor matrix override. If None, uses self.exog_break
            and self.exog combined.

        Returns
        -------
        tuple[float, NDArray[np.floating]]
            Sum of squared residuals and OLS coefficients.
        """
        if y is None:
            y_seg = self.endog[start:end]
        else:
            y_seg = y[start:end]

        if x is None:
            # Build regressor matrix from self.exog_break and self.exog
            X_list = []
            if self.exog_break is not None:
                X_list.append(self.exog_break[start:end])
            if self.exog is not None:
                X_list.append(self.exog[start:end])
            
            if not X_list:
                X_seg = None
            else:
                X_seg = np.column_stack(X_list)
        else:
            X_seg = x[start:end]

        if X_seg is None or X_seg.shape[1] == 0:
            # No regressors - SSR is total sum of squares
            ssr = float(np.sum((y_seg - np.mean(y_seg)) ** 2))
            return ssr, np.array([np.mean(y_seg)])

        # OLS estimation
        try:
            beta, residuals, _rank, _s = np.linalg.lstsq(X_seg, y_seg, rcond=None)
            if len(residuals) > 0:
                ssr = float(residuals[0])
            else:
                ssr = float(np.sum((y_seg - X_seg @ beta) ** 2))
        except np.linalg.LinAlgError:
            ssr = np.inf
            beta = np.zeros(X_seg.shape[1])

        return ssr, beta

    def _build_ssr_matrix(
        self,
        h: int,
        y: NDArray[np.floating[Any]] | None = None,
        x: NDArray[np.floating[Any]] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Build matrix of segment SSRs.

        Computes SSR[i,j] = SSR for segment from observation i to j.
        Only computes for valid segments (length >= h).

        Parameters
        ----------
        h : int
            Minimum segment length.
        y : NDArray | None
            Dependent variable override.
        x : NDArray | None
            Regressor matrix override.

        Returns
        -------
        NDArray[np.floating]
            Matrix of shape (T, T) with SSR values.
        """
        T = self.nobs
        ssr_matrix = np.full((T, T), np.inf)

        for i in range(T):
            for j in range(i + h, T + 1):  # j is exclusive end
                ssr, _ = self._compute_ssr_segment(i, j, y=y, x=x)
                ssr_matrix[i, j - 1] = ssr  # Store at [i, j-1] for segment [i, j)

        return ssr_matrix

    def _dynamic_programming(
        self, m: int, h: int, ssr_matrix: NDArray[np.floating[Any]]
    ) -> tuple[float, list[int]]:
        """Find optimal m breaks using dynamic programming.

        This implements the Bai-Perron (2003) algorithm for computing
        optimal break locations that minimize total SSR.

        Parameters
        ----------
        m : int
            Number of breaks to find.
        h : int
            Minimum segment length.
        ssr_matrix : NDArray[np.floating]
            Pre-computed matrix of segment SSRs.

        Returns
        -------
        tuple[float, list[int]]
            Total SSR and list of break indices.
        """
        T = self.nobs

        if m == 0:
            return float(ssr_matrix[0, T - 1]), []

        # U[i, k] = minimum SSR using first i observations with k breaks
        # Break indices stored separately
        U = np.full((T + 1, m + 1), np.inf)
        break_storage: dict[tuple[int, int], list[int]] = {}

        # Base case: no breaks
        for i in range(h, T + 1):
            U[i, 0] = ssr_matrix[0, i - 1]
            break_storage[(i, 0)] = []

        # Fill DP table
        for k in range(1, m + 1):
            # Need at least (k+1)*h observations for k breaks
            for i in range((k + 1) * h, T + 1):
                best_ssr = np.inf
                best_break = -1

                # Try each possible location for the k-th break
                for j in range(k * h, i - h + 1):
                    candidate_ssr = U[j, k - 1] + ssr_matrix[j, i - 1]
                    if candidate_ssr < best_ssr:
                        best_ssr = candidate_ssr
                        best_break = j

                U[i, k] = best_ssr
                if best_break >= 0:
                    prev_breaks = break_storage.get((best_break, k - 1), [])
                    break_storage[(i, k)] = prev_breaks + [best_break]

        optimal_ssr = U[T, m]
        optimal_breaks = break_storage.get((T, m), [])

        return float(optimal_ssr), optimal_breaks

    def _compute_supf(
        self, m: int, h: int, ssr_matrix: NDArray[np.floating[Any]]
    ) -> float:
        """Compute Sup-F statistic for m breaks vs 0 breaks.

        Parameters
        ----------
        m : int
            Number of breaks under alternative.
        h : int
            Minimum segment length.
        ssr_matrix : NDArray[np.floating]
            Pre-computed matrix of segment SSRs.

        Returns
        -------
        float
            Sup-F statistic.
        """
        T = self.nobs
        q = self.q  # Number of breaking regressors
        p = self.p  # Number of non-breaking regressors

        # SSR under null (no breaks)
        ssr_0 = ssr_matrix[0, T - 1]

        # SSR under alternative (m breaks)
        ssr_m, _ = self._dynamic_programming(m, h, ssr_matrix)

        # F-statistic
        # F = ((SSR_0 - SSR_m) / (m * q)) / (SSR_m / (T - (m + 1) * q - p))
        df1 = m * q
        df2 = T - (m + 1) * q - p

        if df2 <= 0 or ssr_m <= 0:
            return 0.0

        f_stat = ((ssr_0 - ssr_m) / df1) / (ssr_m / df2)
        return max(0.0, f_stat)

    def _compute_seqf(
        self,
        m: int,
        h: int,
        ssr_matrix: NDArray[np.floating[Any]],
        breaks_m: list[int],
    ) -> float:
        """Compute sequential Sup-F statistic for m+1 vs m breaks.

        Parameters
        ----------
        m : int
            Number of breaks under null.
        h : int
            Minimum segment length.
        ssr_matrix : NDArray[np.floating]
            Pre-computed matrix of segment SSRs.
        breaks_m : list[int]
            Break locations under null (m breaks).

        Returns
        -------
        float
            Sequential Sup-F statistic.
        """
        T = self.nobs
        q = self.q

        if m == 0:
            # Test 1 vs 0 breaks - same as SupF(1)
            return self._compute_supf(1, h, ssr_matrix)

        # SSR with m breaks
        ssr_m, _ = self._dynamic_programming(m, h, ssr_matrix)

        # SSR with m+1 breaks
        ssr_m1, _ = self._dynamic_programming(m + 1, h, ssr_matrix)

        # F-statistic
        df2 = T - (m + 2) * q - self.p
        if df2 <= 0 or ssr_m1 <= 0:
            return 0.0

        f_stat = ((ssr_m - ssr_m1) / q) / (ssr_m1 / df2)
        return max(0.0, f_stat)

    def _compute_information_criteria(self, ssr: float, m: int) -> tuple[float, float]:
        """Compute BIC and LWZ for model with m breaks.

        Parameters
        ----------
        ssr : float
            Sum of squared residuals.
        m : int
            Number of breaks.

        Returns
        -------
        tuple[float, float]
            BIC and LWZ values.
        """
        T = self.nobs
        q = self.q
        p = self.p

        # Number of parameters: (m+1)*q + p + m (break locations) + 1 (variance)
        k = (m + 1) * q + p

        # Log-likelihood (Gaussian)
        if ssr <= 0:
            return np.inf, np.inf

        sigma2 = ssr / T
        llf = -T / 2 * (np.log(2 * np.pi) + np.log(sigma2) + 1)

        # BIC
        bic = -2 * llf + k * np.log(T)

        # LWZ (Liu-Wu-Zidek modified Schwarz criterion)
        lwz = -2 * llf + k * np.log(T) * (np.log(np.log(T)))

        return bic, lwz

    def fit(
        self,
        max_breaks: int = 5,
        trimming: float = 0.15,
        selection: Literal["bic", "lwz", "sequential"] = "bic",
        **kwargs: Any,
    ) -> BaiPerronResults:
        """Perform the Bai-Perron structural break test.

        Parameters
        ----------
        max_breaks : int
            Maximum number of breaks to consider. Default is 5.
        trimming : float
            Trimming parameter (minimum segment length as fraction of T).
            Default is 0.15 (15%).
        selection : str
            Method for selecting number of breaks:
            - "bic": Bayesian Information Criterion (default)
            - "lwz": Liu-Wu-Zidek modified Schwarz criterion
            - "sequential": Sequential testing procedure
        **kwargs
            Additional arguments (reserved for future use).

        Returns
        -------
        BaiPerronResults
            Results object containing test statistics, break dates, and
            information criteria.
        """
        T = self.nobs
        q = self.q

        # Minimum segment length
        h = max(int(np.ceil(trimming * T)), q + 1)

        # Maximum feasible breaks
        max_feasible = T // h - 1
        max_breaks = min(max_breaks, max_feasible)

        if max_breaks < 1:
            raise ValueError(
                f"Cannot test for breaks: trimming too large or T too small. "
                f"Need at least {2 * h} observations, have {T}."
            )

        # Pre-compute SSR matrix
        # If we have non-breaking regressors (partial structural change),
        # we need to use the iterative procedure from Bai & Perron (2003).
        #
        # Partial structural change model — Bai & Perron (2003), Section 3
        # ----------------------------------------------------------------
        # The model is:
        #     y_t = x_t' * beta + z_t' * delta_j + e_t,  t in regime j
        #
        # where beta are fixed (non-breaking) coefficients on x_t (self.exog),
        # and delta_j are regime-specific coefficients on z_t (self.exog_break).
        #
        # Estimation follows a concentrating / iterative scheme:
        #   1. Given current beta_hat, compute y* = y - X * beta_hat.
        #   2. Apply dynamic programming to y* on Z to find optimal breaks
        #      and regime-specific delta_j estimates.
        #   3. Construct the full design matrix (Z_bar | X) and re-estimate
        #      all coefficients jointly.
        #   4. Update beta_hat from step 3 and repeat until convergence.
        #
        # When p = 0 (pure structural change), no iteration is needed and the
        # standard Bai-Perron dynamic programming algorithm is used directly.
        
        if self.p > 0:
            # Partial structural change model
            # Iterative procedure per Bai & Perron (2003), Section 3.
            
            # Initial estimate of fixed coefficients (assuming no breaks)
            X_full = []
            if self.exog_break is not None:
                X_full.append(self.exog_break)
            if self.exog is not None:
                X_full.append(self.exog)
            
            X_mat = np.column_stack(X_full)
            # Use OLS to get initial beta
            beta_full = np.linalg.lstsq(X_mat, self.endog, rcond=None)[0]
            
            # Extract fixed coefficients (last p)
            beta_fixed = beta_full[-self.p:]
            
            ssr_vals = {}
            breaks_by_m = {}
            
            # m=0 case: No breaks
            # SSR is just the SSR from the full model with no breaks
            resid_0 = self.endog - X_mat @ beta_full
            ssr_vals[0] = float(np.sum(resid_0**2))
            breaks_by_m[0] = []
            
            for m in range(1, max_breaks + 1):
                # Initialize beta_fixed from m=0 (or could use previous m's result)
                # We restart from the no-break estimate to avoid getting stuck in local optima
                curr_beta_fixed = beta_fixed.copy()
                
                # Iterative procedure for specific m
                # We need to define Full_Design and full_beta outside the loop 
                # in case the loop doesn't run (though it always runs at least once)
                Full_Design = None
                full_beta = None
                max_iter = 50
                converged = False
                
                for _iter in range(max_iter):
                    # 1. Partial out fixed regressors
                    if self.exog is not None:
                        y_star = self.endog - self.exog @ curr_beta_fixed
                    else:
                        y_star = self.endog
                    
                    # 2. Find optimal breaks for y_star on exog_break
                    # We pass x=self.exog_break to override default behavior
                    ssr_matrix_m = self._build_ssr_matrix(h, y=y_star, x=self.exog_break)
                    ssr_m, breaks_m = self._dynamic_programming(m, h, ssr_matrix_m)
                    
                    # 3. Re-estimate beta_fixed given these breaks
                    # Construct full design matrix with breaks
                    
                    # Z_bar columns (breaking regressors)
                    boundaries = [0] + breaks_m + [T]
                    Z_cols = []
                    for i in range(len(boundaries) - 1):
                        start, end = boundaries[i], boundaries[i+1]
                        for col in range(self.q):
                            z_col = np.zeros(T)
                            z_col[start:end] = self.exog_break[start:end, col]
                            Z_cols.append(z_col)
                    
                    if not Z_cols:
                         # Should not happen if q > 0
                         Z_bar = np.zeros((T, 0))
                    else:
                        Z_bar = np.column_stack(Z_cols)
                        
                    Full_Design = np.column_stack([Z_bar, self.exog])
                    
                    # Estimate
                    full_beta = np.linalg.lstsq(Full_Design, self.endog, rcond=None)[0]
                    
                    # Update beta_fixed (last p coeffs)
                    new_beta_fixed = full_beta[-self.p:]
                    
                    # Check convergence
                    if np.allclose(curr_beta_fixed, new_beta_fixed, rtol=1e-4, atol=1e-6):
                        curr_beta_fixed = new_beta_fixed
                        converged = True
                        break
                    curr_beta_fixed = new_beta_fixed
                
                if not converged:
                    warnings.warn(
                        f"Partial structural change iterative procedure did not "
                        f"converge for m={m} after {max_iter} iterations. "
                        f"Results may be unreliable.",
                        stacklevel=2,
                    )
                
                # Store results for this m
                # Re-calculate SSR with final beta
                if Full_Design is not None and full_beta is not None:
                    resid = self.endog - Full_Design @ full_beta
                    ssr_vals[m] = float(np.sum(resid**2))
                    breaks_by_m[m] = breaks_m
                else:
                    # Fallback if something went wrong
                    ssr_vals[m] = np.inf
                    breaks_by_m[m] = []

            
        else:
            # Pure structural change (original code)
            ssr_matrix = self._build_ssr_matrix(h)
            
            ssr_vals = {}
            breaks_by_m = {}
            
            # No breaks case
            ssr_0 = float(ssr_matrix[0, T - 1])
            ssr_vals[0] = ssr_0
            breaks_by_m[0] = []
            
            for m in range(1, max_breaks + 1):
                ssr_m, breaks_m = self._dynamic_programming(m, h, ssr_matrix)
                ssr_vals[m] = ssr_m
                breaks_by_m[m] = breaks_m

        # Compute statistics for each number of breaks
        supf_stats: dict[int, float] = {}
        supf_pvalues: dict[int, float] = {}
        supf_critical: dict[int, float] = {}
        seqf_stats: dict[int, float] = {}
        seqf_critical: dict[int, float] = {}
        bic_vals: dict[int, float] = {}
        lwz_vals: dict[int, float] = {}
        
        # Compute Information Criteria and SupF stats
        for m in range(max_breaks + 1):
            if m == 0:
                bic_vals[0], lwz_vals[0] = self._compute_information_criteria(ssr_vals[0], 0)
                continue
                
            # Sup-F test
            # F = ((SSR_0 - SSR_m) / (m * q)) / (SSR_m / (T - (m + 1) * q - p))
            df1 = m * q
            df2 = T - (m + 1) * q - self.p
            
            ssr_m = ssr_vals[m]
            ssr_0 = ssr_vals[0]
            
            if df2 <= 0 or ssr_m <= 0:
                supf = 0.0
            else:
                supf = ((ssr_0 - ssr_m) / df1) / (ssr_m / df2)
                supf = max(0.0, supf)
            
            supf_stats[m] = supf

            # Critical value (use approximation)
            crit = _SUPF_CRITICAL_VALUES.get((min(q, 5), m), 10.0)
            supf_critical[m] = crit

            # Approximate p-value
            pval = 1 - stats.chi2.cdf(supf * m * q / 2, m * q)
            supf_pvalues[m] = min(1.0, max(0.0, pval))

            # Information criteria
            bic_vals[m], lwz_vals[m] = self._compute_information_criteria(ssr_m, m)

        # Sequential F-tests: Seq(m|m-1) tests m-1 vs m breaks.
        # Computed from pre-computed SSR values for both the pure and
        # partial structural change cases.
        # Note: Bai & Perron (1998) Table III provides separate critical
        # values for sequential tests.  As an approximation, we use the
        # Sup-F(m) critical values here; this is conservative for small m
        # and follows the convention used in the original implementation.
        for m in range(1, max_breaks + 1):
            ssr_null = ssr_vals[m - 1]
            ssr_alt = ssr_vals[m]

            df2 = T - (m + 1) * q - self.p

            if df2 <= 0 or ssr_alt <= 0:
                seqf = 0.0
            else:
                seqf = ((ssr_null - ssr_alt) / q) / (ssr_alt / df2)
                seqf = max(0.0, seqf)

            seqf_stats[m - 1] = seqf
            crit = _SUPF_CRITICAL_VALUES.get((min(q, 5), m), 10.0)
            seqf_critical[m - 1] = crit

        # UDmax and WDmax
        udmax = max(supf_stats.values()) if supf_stats else 0.0
        udmax_crit = _UDMAX_CRITICAL_VALUES.get(min(q, 5), 10.0)

        # WDmax uses weights based on critical values
        wdmax_values = [
            supf_stats[m] * supf_critical.get(m, 10.0) / supf_critical.get(1, 10.0)
            for m in supf_stats
        ]
        wdmax = max(wdmax_values) if wdmax_values else 0.0

        # Select number of breaks
        if selection == "bic":
            n_breaks = min(bic_vals, key=bic_vals.get)  # type: ignore[arg-type]
        elif selection == "lwz":
            n_breaks = min(lwz_vals, key=lwz_vals.get)  # type: ignore[arg-type]
        elif selection == "sequential":
            # Sequential procedure: start with m=0, increase while significant
            n_breaks = 0
            for m in range(max_breaks):
                if m in seqf_stats:
                    if seqf_stats[m] > seqf_critical.get(m, 10.0):
                        n_breaks = m + 1
                    else:
                        break
        else:
            raise ValueError(f"Unknown selection method: {selection}")

        break_indices = list(breaks_by_m.get(n_breaks, []))

        # Compute confidence intervals for break dates
        break_ci: dict[int, tuple[int, int]] = {}
        if n_breaks > 0 and break_indices:
            break_ci = self._compute_break_confidence_intervals(break_indices)

        return BaiPerronResults(
            test_name="Bai-Perron",
            nobs=T,
            n_breaks=n_breaks,
            break_indices=break_indices,
            supf_stats=supf_stats,
            supf_pvalues=supf_pvalues,
            supf_critical=supf_critical,
            udmax=udmax,
            udmax_critical=udmax_crit,
            wdmax=wdmax,
            seqf_stats=seqf_stats,
            seqf_critical=seqf_critical,
            bic=bic_vals,
            lwz=lwz_vals,
            ssr=ssr_vals,
            breaks_by_m=breaks_by_m,
            trimming=trimming,
            max_breaks=max_breaks,
            selection_method=selection,
            q=q,
            break_ci=break_ci,
            _test=self,
        )

    def get_regime_estimates(
        self, break_indices: Sequence[int]
    ) -> list[tuple[NDArray[np.floating[Any]], float]]:
        """Get parameter estimates for each regime.

        Parameters
        ----------
        break_indices : Sequence[int]
            Break point indices.

        Returns
        -------
        list[tuple[NDArray[np.floating], float]]
            List of (coefficients, ssr) tuples for each regime.
        """
        results = []
        breaks = [0] + list(break_indices) + [self.nobs]

        for i in range(len(breaks) - 1):
            start, end = breaks[i], breaks[i + 1]
            ssr, beta = self._compute_ssr_segment(start, end)
            results.append((beta, ssr))

        return results

    def _compute_break_confidence_intervals(
        self,
        break_indices: Sequence[int],
        alpha: float = 0.05,
    ) -> dict[int, tuple[int, int]]:
        """Compute confidence intervals for break dates.

        Uses the asymptotic distribution from Bai (1997) and Bai-Perron (1998).
        The confidence interval is computed as:

            CI = [k - c(α) * λ,  k + c(α) * λ]

        where λ = σ² / (δ'Qδ) is the shrinkage factor, σ² is the residual
        variance, δ is the parameter change at the break, and Q is the
        second moment matrix of breaking regressors.

        Parameters
        ----------
        break_indices : Sequence[int]
            Break point indices.
        alpha : float
            Significance level. Default is 0.05 (95% CI).

        Returns
        -------
        dict[int, tuple[int, int]]
            Dictionary mapping break index to (lower, upper) bounds.

        References
        ----------
        Bai, J. (1997). Estimation of a change point in multiple regression
            models. Review of Economics and Statistics, 79(4), 551-563.
        """
        if len(break_indices) == 0:
            return {}

        # Critical values from Bai (1997) asymptotic distribution
        # These are for the two-sided case
        critical_values = {
            0.10: 6.70,
            0.05: 7.78,
            0.01: 11.45,
        }
        c_alpha = critical_values.get(alpha, 7.78)

        # Get regime estimates
        regime_estimates = self.get_regime_estimates(break_indices)

        # Compute total SSR for pooled variance estimate
        total_ssr = sum(ssr for _, ssr in regime_estimates)
        total_obs = self.nobs
        n_params = (len(break_indices) + 1) * self.q + self.p
        sigma2 = total_ssr / (total_obs - n_params)

        break_ci: dict[int, tuple[int, int]] = {}

        for i, break_idx in enumerate(break_indices):
            # Get parameter estimates for regimes before and after the break
            beta_before = regime_estimates[i][0]
            beta_after = regime_estimates[i + 1][0]

            # Extract breaking regressor coefficients (first q coefficients)
            delta = beta_after[: self.q] - beta_before[: self.q]

            # Compute Q (second moment matrix of breaking regressors)
            # For mean-shift model (q=1 with constant), Q = 1
            if self.q == 1 and self.exog_break is not None:
                # Check if it's a constant
                if np.allclose(self.exog_break, 1.0):
                    Q = 1.0
                else:
                    # Use local X'X/T around the break
                    h = max(20, int(0.1 * self.nobs))  # Window around break
                    start = max(0, break_idx - h)
                    end = min(self.nobs, break_idx + h)
                    X_local = self.exog_break[start:end]
                    Q = float(np.mean(X_local**2))
            else:
                # General case: use X'X/T from full sample
                if self.exog_break is not None:
                    X = self.exog_break
                    Q_matrix = X.T @ X / self.nobs
                    # Scalar shrinkage factor: δ'Qδ
                    Q = float(delta @ Q_matrix @ delta) / float(delta @ delta + 1e-10)
                else:
                    Q = 1.0

            # Compute shrinkage factor λ = σ² / (δ'Qδ)
            delta_Q_delta = (
                float(delta @ delta) * Q
                if self.q == 1
                else float(delta @ (Q_matrix @ delta))
            )
            if delta_Q_delta < 1e-10:
                # If parameter change is tiny, CI is very wide
                lambda_factor = self.nobs  # Conservative upper bound
            else:
                lambda_factor = sigma2 / delta_Q_delta

            # Compute CI bounds
            half_width = c_alpha * lambda_factor
            lower = int(max(0, np.floor(break_idx - half_width)))
            upper = int(min(self.nobs - 1, np.ceil(break_idx + half_width)))

            break_ci[break_idx] = (lower, upper)

        return break_ci
