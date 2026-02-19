"""OLS regression with HAC standard errors and structural break support.

This module provides OLS estimation with various covariance estimators,
including heteroskedasticity and autocorrelation consistent (HAC) standard
errors using the Newey-West estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import statsmodels.api as sm
from scipy import stats

from regimes.diagnostics import DiagnosticsResults, compute_diagnostics
from regimes.models.base import CovType, RegimesModelBase
from regimes.results.base import RegressionResultsBase

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    import pandas as pd
    from numpy.typing import ArrayLike, NDArray

    from regimes.markov.results import MarkovRegressionResults
    from regimes.rolling.ols import RecursiveOLS, RollingOLS
    from regimes.tests.andrews_ploberger import AndrewsPlobergerResults
    from regimes.tests.bai_perron import BaiPerronResults
    from regimes.tests.chow import ChowTestResults
    from regimes.tests.cusum import CUSUMResults, CUSUMSQResults


@dataclass(kw_only=True)
class OLSResults(RegressionResultsBase):
    """Results from OLS estimation.

    Extends RegressionResultsBase with OLS-specific attributes and methods.

    Additional Attributes
    ---------------------
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    llf : float
        Log-likelihood of the model.
    fvalue : float
        F-statistic for overall model significance.
    f_pvalue : float
        P-value for the F-statistic.
    """

    llf: float | None = None
    scale: float | None = None

    # Private fields for diagnostics (not shown in repr)
    _exog: NDArray[np.floating[Any]] | None = field(default=None, repr=False)
    _diagnostics_cache: DiagnosticsResults | None = field(default=None, repr=False)

    # Break information (not shown in repr)
    _breaks: Sequence[int] | None = field(default=None, repr=False)
    _variable_breaks: dict[str | int, Sequence[int]] | None = field(
        default=None, repr=False
    )
    _nobs_original: int | None = field(default=None, repr=False)

    @property
    def sigma_squared(self) -> float:
        """Residual variance (sigma^2)."""
        return self.scale if self.scale is not None else np.nan

    @property
    def sigma(self) -> float:
        """Residual standard error (sigma)."""
        return np.sqrt(self.sigma_squared)

    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        if self.llf is None:
            return np.nan
        return -2 * self.llf + 2 * self.df_model

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        if self.llf is None:
            return np.nan
        return -2 * self.llf + np.log(self.nobs) * self.df_model

    @property
    def fvalue(self) -> float:
        """F-statistic for overall significance (excluding intercept)."""
        if self._tss is None or self._tss == 0:
            return np.nan
        # F = (R^2 / k) / ((1 - R^2) / (n - k - 1))
        # where k is number of regressors excluding constant
        k = self.df_model - 1  # Assuming one regressor is constant
        if k <= 0 or self.df_resid <= 0:
            return np.nan
        return (self.rsquared / k) / ((1 - self.rsquared) / self.df_resid)

    @property
    def f_pvalue(self) -> float:
        """P-value for F-statistic."""
        if np.isnan(self.fvalue):
            return np.nan
        k = self.df_model - 1
        if k <= 0:
            return np.nan
        return 1 - stats.f.cdf(self.fvalue, k, self.df_resid)

    def diagnostics(
        self, lags_autocorr: int = 2, lags_arch: int = 1
    ) -> DiagnosticsResults:
        """Compute misspecification tests (cached).

        Parameters
        ----------
        lags_autocorr : int, default 2
            Number of lags for the Breusch-Godfrey autocorrelation test.
        lags_arch : int, default 1
            Number of lags for the ARCH test.

        Returns
        -------
        DiagnosticsResults
            Collection of diagnostic test results including:
            - AR test (Breusch-Godfrey LM)
            - ARCH test (Engle's LM)
            - Normality test (Jarque-Bera)
            - Heteroskedasticity test (White)

        Raises
        ------
        ValueError
            If _exog was not stored during model fitting.

        Notes
        -----
        Results are cached on first call. Subsequent calls with different
        lag parameters will recompute the tests.
        """
        if self._exog is None:
            raise ValueError(
                "Regressors (exog) not available. Diagnostics require the "
                "original design matrix to be stored during fitting."
            )

        # Check if we need to recompute (cache invalidation by lag changes)
        if self._diagnostics_cache is not None:
            cached = self._diagnostics_cache
            # Check if lag parameters match
            if (
                cached.autocorrelation is not None
                and cached.arch is not None
                and cached.autocorrelation.df == lags_autocorr
                and cached.arch.df == lags_arch
            ):
                return cached

        # Compute diagnostics
        self._diagnostics_cache = compute_diagnostics(
            self.resid, self._exog, lags_autocorr=lags_autocorr, lags_arch=lags_arch
        )
        return self._diagnostics_cache

    def _format_break_section(self) -> list[str]:
        """Format the structural breaks section for summary output.

        Returns
        -------
        list[str]
            Lines describing the break structure.
        """
        lines = []
        nobs = self._nobs_original if self._nobs_original is not None else self.nobs

        if self._breaks:
            # Common breaks
            lines.append("-" * 81)
            lines.append("Structural Breaks")
            lines.append("-" * 81)

            for bp in self._breaks:
                lines.append(f"Break at observation {bp}")

            # Show regime information
            boundaries = [0] + list(self._breaks) + [nobs]
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1] - 1
                n_regime = boundaries[i + 1] - boundaries[i]
                lines.append(
                    f"  Regime {i + 1}: observations {start}-{end} (n={n_regime})"
                )

        elif self._variable_breaks:
            # Variable-specific breaks
            lines.append("-" * 81)
            lines.append("Variable-Specific Structural Breaks")
            lines.append("-" * 81)

            for var_name, breaks in self._variable_breaks.items():
                break_str = ", ".join(str(bp) for bp in breaks)
                if len(breaks) == 1:
                    lines.append(f"{var_name}: break at observation {break_str}")
                else:
                    lines.append(f"{var_name}: breaks at observations {break_str}")

                # Show regime ranges on same line
                boundaries = [0] + list(breaks) + [nobs]
                regime_parts = []
                for i in range(len(boundaries) - 1):
                    start = boundaries[i]
                    end = boundaries[i + 1] - 1
                    regime_parts.append(f"Regime {i + 1}: obs {start}-{end}")
                lines.append(f"  {', '.join(regime_parts)}")

        return lines

    def summary(self, diagnostics: bool = True) -> str:
        """Generate a text summary of OLS results.

        Parameters
        ----------
        diagnostics : bool, default True
            If True, include misspecification tests (autocorrelation, ARCH,
            normality, heteroskedasticity) in the output.

        Returns
        -------
        str
            Formatted summary including coefficients, standard errors,
            t-values, p-values, and optionally misspecification tests.
        """
        lines = []
        lines.append("=" * 81)
        lines.append(f"{'OLS Regression Results':^81}")
        lines.append("=" * 81)
        lines.append(
            f"Dep. Variable:           y   No. Observations:    {self.nobs:>10}"
        )
        lines.append(
            f"Model:                 OLS   Df Residuals:        {self.df_resid:>10}"
        )
        lines.append(
            f"Cov. Type:      {self.cov_type:>10}   Df Model:            {self.df_model:>10}"
        )
        lines.append(
            f"R-squared:         {self.rsquared:>7.4f}   Adj. R-squared:      {self.rsquared_adj:>10.4f}"
        )
        lines.append(
            f"Residual Std Err:  {self.sigma:>7.4f}   Residual Variance:   {self.sigma_squared:>10.4f}"
        )

        if self.llf is not None:
            lines.append(
                f"Log-Likelihood:    {self.llf:>7.2f}   AIC:                 {self.aic:>10.2f}"
            )
            lines.append(
                f"F-statistic:       {self.fvalue:>7.2f}   BIC:                 {self.bic:>10.2f}"
            )
            if not np.isnan(self.f_pvalue):
                lines.append(f"Prob (F-statistic): {self.f_pvalue:>.2e}")

        lines.append("=" * 81)

        # Add break timing section if breaks are present
        if self._breaks or self._variable_breaks:
            lines.append("")
            lines.extend(self._format_break_section())
            lines.append("")
            lines.append("=" * 81)

        # Parameter table header
        lines.append(
            f"{'':>15} {'coef':>10} {'std err':>10} {'t':>10} "
            f"{'P>|t|':>10} {'[0.025':>10} {'0.975]':>10}"
        )
        lines.append("-" * 81)

        # Parameter rows
        ci = self.conf_int()
        names = self.param_names or [f"x{i}" for i in range(len(self.params))]
        for i, name in enumerate(names):
            pval_str = (
                f"{self.pvalues[i]:.3f}"
                if self.pvalues[i] >= 0.001
                else f"{self.pvalues[i]:.2e}"
            )
            lines.append(
                f"{name:>15} {self.params[i]:>10.4f} {self.bse[i]:>10.4f} "
                f"{self.tvalues[i]:>10.3f} {pval_str:>10} "
                f"{ci[i, 0]:>10.3f} {ci[i, 1]:>10.3f}"
            )

        lines.append("=" * 81)
        if self.cov_type == "HAC":
            lines.append("Note: Standard errors are HAC (Newey-West) robust.")
        elif self.cov_type.startswith("HC"):
            lines.append(
                f"Note: Standard errors are {self.cov_type} heteroskedasticity-robust."
            )

        # Add diagnostics section if requested
        if diagnostics and self._exog is not None:
            diag = self.diagnostics()
            lines.append("")
            lines.append(diag.summary())

        return "\n".join(lines)


class OLS(RegimesModelBase):
    """Ordinary Least Squares regression with robust standard errors.

    This class provides OLS estimation with support for various covariance
    estimators including heteroskedasticity-robust (HC0-HC3) and HAC
    (Newey-West) standard errors. It can also handle known structural breaks
    by estimating separate coefficients for each regime.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    exog : ArrayLike | None
        Exogenous regressors (n_obs, k). If None, a constant is added.
    breaks : Sequence[int] | None
        Known break points (observation indices). If provided, coefficients
        are estimated separately for each regime. All variables share these
        break points.
    variable_breaks : dict[str | int, Sequence[int]] | None
        Variable-specific break points. Keys can be variable names (str) or
        column indices (int). Values are lists of break points for that
        variable. Variables not in the dict have no breaks (constant
        coefficient across the sample). Cannot be used together with `breaks`.
    has_constant : bool
        Whether to add a constant term. Default True if exog is None.

    Examples
    --------
    >>> import numpy as np
    >>> from regimes import OLS
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = X @ [1, 2] + np.random.randn(n)
    >>> model = OLS(y, X)
    >>> results = model.fit(cov_type="HAC")
    >>> print(results.summary())

    Variable-specific breaks:

    >>> # Different break points for constant and regressor
    >>> model = OLS(y, X, variable_breaks={
    ...     "const": [50],  # constant breaks at t=50
    ...     "x1": [75],     # x1 coefficient breaks at t=75
    ... })

    Notes
    -----
    When breaks are specified, the model creates regime-specific dummy
    interactions for all regressors, allowing coefficients to differ
    across regimes.

    When variable_breaks is specified, each variable can have its own
    set of break points, allowing for different timing of structural
    changes across coefficients.
    """

    def __init__(
        self,
        endog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        exog: ArrayLike | pd.Series[Any] | pd.DataFrame | None = None,
        breaks: Sequence[int] | None = None,
        variable_breaks: dict[str | int, Sequence[int]] | None = None,
        has_constant: bool | None = None,
    ) -> None:
        """Initialize OLS model."""
        super().__init__(endog, exog, breaks)

        # Validate that breaks and variable_breaks are not both specified
        if breaks is not None and variable_breaks is not None:
            raise ValueError(
                "Cannot specify both 'breaks' and 'variable_breaks'. "
                "Use 'breaks' for common breaks across all variables, or "
                "'variable_breaks' for variable-specific breaks."
            )

        self._variable_breaks = variable_breaks

        # Determine if we need to add a constant
        if has_constant is None:
            has_constant = exog is None

        self._has_constant = has_constant

        # Add constant if needed
        if has_constant and self.exog is None:
            self.exog = np.ones((self.nobs, 1))
            self._exog_names = ["const"]
        elif has_constant and self.exog is not None:
            # Check if constant already exists
            if not self._has_constant_column(self.exog):
                self.exog = np.column_stack([np.ones(self.nobs), self.exog])
                self._exog_names = ["const"] + self._exog_names

    @staticmethod
    def _has_constant_column(X: NDArray[np.floating[Any]]) -> bool:
        """Check if X already has a constant column."""
        return any(np.allclose(X[:, i], 1.0) for i in range(X.shape[1]))

    def _normalize_variable_breaks(
        self,
    ) -> dict[int, list[int]] | None:
        """Normalize variable_breaks keys to column indices.

        Returns
        -------
        dict[int, list[int]] | None
            Dictionary mapping column indices to sorted break points,
            or None if variable_breaks is not specified.
        """
        if self._variable_breaks is None:
            return None

        if self.exog is None:
            raise ValueError("exog is required when using variable_breaks")

        k = self.exog.shape[1]
        normalized: dict[int, list[int]] = {}

        for key, breaks in self._variable_breaks.items():
            # Convert key to index
            if isinstance(key, int):
                if key < 0 or key >= k:
                    raise ValueError(
                        f"Variable index {key} is out of bounds for exog with {k} columns"
                    )
                idx = key
            else:
                # key is a string, find it in _exog_names
                if key not in self._exog_names:
                    raise ValueError(
                        f"Variable name '{key}' not found. "
                        f"Available names: {self._exog_names}"
                    )
                idx = self._exog_names.index(key)

            # Validate and sort break points
            breaks_list = sorted(breaks)
            for bp in breaks_list:
                if bp <= 0 or bp >= self.nobs:
                    raise ValueError(
                        f"Break point {bp} for variable '{key}' is out of bounds. "
                        f"Must be in range (0, {self.nobs})."
                    )
            normalized[idx] = breaks_list

        return normalized

    def _get_variable_regime_indices(self, breaks: list[int]) -> list[tuple[int, int]]:
        """Get regime start/end indices for a variable's break points.

        Parameters
        ----------
        breaks : list[int]
            Sorted list of break points for this variable.

        Returns
        -------
        list[tuple[int, int]]
            List of (start, end) tuples for each regime.
        """
        if not breaks:
            return [(0, self.nobs)]

        indices = []
        prev = 0
        for bp in breaks:
            indices.append((prev, bp))
            prev = bp
        indices.append((prev, self.nobs))
        return indices

    def _create_break_design(self) -> tuple[NDArray[np.floating[Any]], list[str]]:
        """Create design matrix with regime-specific coefficients.

        Returns
        -------
        tuple[NDArray[np.floating], list[str]]
            Design matrix with regime interactions and parameter names.
        """
        if self.exog is None:
            return self.exog, self._exog_names

        # Handle variable-specific breaks
        if self._variable_breaks is not None:
            return self._create_variable_break_design()

        # Handle common breaks (original behavior)
        if not self.breaks:
            return self.exog, self._exog_names

        # Create regime indicators
        regime_indices = self.get_regime_indices()
        n_regimes = len(regime_indices)
        k = self.exog.shape[1]

        # Create expanded design matrix
        X_expanded = np.zeros((self.nobs, k * n_regimes))
        param_names = []

        for r, (start, end) in enumerate(regime_indices):
            col_start = r * k
            col_end = (r + 1) * k
            X_expanded[start:end, col_start:col_end] = self.exog[start:end]

            for name in self._exog_names:
                param_names.append(f"{name}_regime{r + 1}")

        return X_expanded, param_names

    def _create_variable_break_design(
        self,
    ) -> tuple[NDArray[np.floating[Any]], list[str]]:
        """Create design matrix with variable-specific break points.

        Returns
        -------
        tuple[NDArray[np.floating], list[str]]
            Design matrix with variable-specific regime columns and
            parameter names.
        """
        if self.exog is None:
            raise ValueError("exog is required for break design")

        normalized_breaks = self._normalize_variable_breaks()
        if normalized_breaks is None:
            return self.exog, self._exog_names

        k = self.exog.shape[1]

        # Calculate total number of columns needed
        # For each variable: if it has breaks, num_regimes = len(breaks) + 1
        # Otherwise, just 1 column (no breaks)
        columns = []
        param_names = []

        for var_idx in range(k):
            var_name = self._exog_names[var_idx]
            var_data = self.exog[:, var_idx]

            if var_idx in normalized_breaks:
                # This variable has breaks
                breaks = normalized_breaks[var_idx]
                regime_indices = self._get_variable_regime_indices(breaks)

                for r, (start, end) in enumerate(regime_indices):
                    col = np.zeros(self.nobs)
                    col[start:end] = var_data[start:end]
                    columns.append(col)
                    param_names.append(f"{var_name}_regime{r + 1}")
            else:
                # No breaks for this variable - constant coefficient
                columns.append(var_data)
                param_names.append(var_name)

        X_expanded = np.column_stack(columns)
        return X_expanded, param_names

    def fit(
        self,
        cov_type: CovType = "nonrobust",
        cov_kwds: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> OLSResults:
        """Fit the OLS model.

        Parameters
        ----------
        cov_type : CovType
            Type of covariance estimator:
            - "nonrobust": Standard OLS covariance
            - "HC0", "HC1", "HC2", "HC3": Heteroskedasticity-robust
            - "HAC": Heteroskedasticity and autocorrelation consistent
        cov_kwds : dict | None
            Additional keywords for covariance estimation. For HAC, can
            include 'maxlags' (default uses Newey-West automatic selection).
        **kwargs
            Additional arguments (reserved for future use).

        Returns
        -------
        OLSResults
            Results object containing estimates and inference.
        """
        if self.exog is None:
            raise ValueError("exog is required for OLS estimation")

        cov_kwds = cov_kwds or {}

        # Create design matrix (with regime interactions if breaks specified)
        X, param_names = self._create_break_design()

        # Use statsmodels for estimation
        sm_model = sm.OLS(self.endog, X)

        # Map cov_type to statsmodels format
        if cov_type == "nonrobust":
            sm_results = sm_model.fit()
        elif cov_type.startswith("HC"):
            sm_results = sm_model.fit(cov_type=cov_type)
        elif cov_type == "HAC":
            maxlags = cov_kwds.get("maxlags")
            if maxlags is None:
                # Newey-West automatic bandwidth selection
                maxlags = int(np.floor(4 * (self.nobs / 100) ** (2 / 9)))
            sm_results = sm_model.fit(
                cov_type="HAC",
                cov_kwds={"maxlags": maxlags, "use_correction": True},
            )
        else:
            raise ValueError(f"Unknown cov_type: {cov_type}")

        # Extract results
        params = np.asarray(sm_results.params)
        bse = np.asarray(sm_results.bse)
        resid = np.asarray(sm_results.resid)
        fittedvalues = np.asarray(sm_results.fittedvalues)
        cov_params = np.asarray(sm_results.cov_params())

        # Compute TSS for R-squared
        y_mean = np.mean(self.endog)
        tss = float(np.sum((self.endog - y_mean) ** 2))

        return OLSResults(
            params=params,
            bse=bse,
            resid=resid,
            fittedvalues=fittedvalues,
            cov_params_matrix=cov_params,
            nobs=self.nobs,
            cov_type=cov_type,
            param_names=param_names,
            model_name="OLS",
            llf=float(sm_results.llf),
            scale=float(sm_results.scale),
            _tss=tss,
            _exog=X,  # Store for diagnostic tests
            _breaks=list(self.breaks) if self.breaks else None,
            _variable_breaks=self._variable_breaks,
            _nobs_original=self.nobs,
        )

    def fit_by_regime(
        self,
        cov_type: CovType = "nonrobust",
        cov_kwds: dict[str, Any] | None = None,
    ) -> list[OLSResults]:
        """Fit separate OLS models for each regime.

        This is an alternative to the pooled estimation with regime
        interactions. It fits completely separate models for each regime.

        Parameters
        ----------
        cov_type : CovType
            Type of covariance estimator.
        cov_kwds : dict | None
            Additional keywords for covariance estimation.

        Returns
        -------
        list[OLSResults]
            List of results, one per regime.
        """
        if not self.breaks:
            return [self.fit(cov_type=cov_type, cov_kwds=cov_kwds)]

        results = []
        for regime in range(self.n_regimes):
            y_r, X_r = self.get_regime_data(regime)
            if X_r is None:
                raise ValueError("exog is required for OLS estimation")

            regime_model = OLS(y_r, X_r, has_constant=False)
            regime_results = regime_model.fit(cov_type=cov_type, cov_kwds=cov_kwds)
            results.append(regime_results)

        return results

    def bai_perron(
        self,
        break_vars: Literal["all", "const"] = "all",
        max_breaks: int = 5,
        trimming: float = 0.15,
        selection: Literal["bic", "lwz", "sequential"] = "bic",
    ) -> BaiPerronResults:
        """Test for structural breaks using Bai-Perron procedure.

        Convenience method that creates a BaiPerronTest from this model
        and runs it. The result can be converted back to an OLSResults
        object with detected breaks using .to_ols().

        Parameters
        ----------
        break_vars : "all" | "const"
            Which variables can have breaks:
            - "all": All regressors can break (default)
            - "const": Only intercept can break (mean-shift model)
        max_breaks : int
            Maximum number of breaks to test. Default is 5.
        trimming : float
            Minimum segment length as fraction of sample. Default is 0.15.
        selection : str
            Break selection criterion:
            - "bic": Bayesian Information Criterion (default)
            - "lwz": Liu-Wu-Zidek modified Schwarz criterion
            - "sequential": Sequential testing procedure

        Returns
        -------
        BaiPerronResults
            Test results. Use .to_ols() to get fitted model with breaks.

        Examples
        --------
        >>> import numpy as np
        >>> from regimes import OLS
        >>> np.random.seed(42)
        >>> n = 200
        >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
        >>> y = np.zeros(n)
        >>> y[:100] = 1 + 0.5 * X[:100, 1] + np.random.randn(100) * 0.5
        >>> y[100:] = 3 + 1.5 * X[100:, 1] + np.random.randn(100) * 0.5
        >>> model = OLS(y, X, has_constant=False)
        >>> bp_results = model.bai_perron()
        >>> print(f"Detected {bp_results.n_breaks} breaks at {bp_results.break_indices}")
        >>> ols_with_breaks = bp_results.to_ols()
        >>> print(ols_with_breaks.summary())

        See Also
        --------
        BaiPerronTest : The underlying test class.
        BaiPerronResults.to_ols : Convert results to OLS with breaks.
        """
        from regimes.tests.bai_perron import BaiPerronTest

        test = BaiPerronTest.from_model(self, break_vars=break_vars)
        return test.fit(max_breaks=max_breaks, trimming=trimming, selection=selection)

    def chow_test(
        self,
        break_points: int | Sequence[int],
        break_vars: Literal["all", "const"] = "all",
        significance: float = 0.05,
    ) -> ChowTestResults:
        """Test for structural breaks at known break points using the Chow test.

        Convenience method that creates a ChowTest from this model and
        runs it. Each break point is tested individually.

        Parameters
        ----------
        break_points : int | Sequence[int]
            One or more break point indices to test.
        break_vars : "all" | "const"
            Which variables can have breaks:
            - "all": All regressors can break (default)
            - "const": Only intercept can break (mean-shift model)
        significance : float
            Significance level for rejection decisions. Default is 0.05.

        Returns
        -------
        ChowTestResults
            Test results with F-statistics and p-values.

        Examples
        --------
        >>> import numpy as np
        >>> from regimes import OLS
        >>> np.random.seed(42)
        >>> n = 200
        >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
        >>> y = np.zeros(n)
        >>> y[:100] = 1 + 0.5 * X[:100, 1] + np.random.randn(100) * 0.5
        >>> y[100:] = 3 + 1.5 * X[100:, 1] + np.random.randn(100) * 0.5
        >>> model = OLS(y, X, has_constant=False)
        >>> results = model.chow_test(break_points=100)
        >>> print(results.summary())

        See Also
        --------
        ChowTest : The underlying test class.
        bai_perron : Test for breaks at unknown break points.
        """
        from regimes.tests.chow import ChowTest

        test = ChowTest.from_model(self, break_vars=break_vars)
        return test.fit(break_points=break_points, significance=significance)

    def cusum_test(self, significance: float = 0.05) -> CUSUMResults:
        """Test for parameter instability using the CUSUM test.

        Convenience method that creates a CUSUMTest from this model
        and runs it. The CUSUM test examines the cumulative sum of
        recursive residuals for departures from parameter stability.

        Parameters
        ----------
        significance : float
            Significance level. Must be one of 0.01, 0.05, 0.10.

        Returns
        -------
        CUSUMResults
            Test results with statistic path, boundaries, and decision.

        See Also
        --------
        CUSUMTest : The underlying test class.
        cusum_sq_test : Test for variance instability.
        """
        from regimes.tests.cusum import CUSUMTest

        test = CUSUMTest.from_model(self)
        return test.fit(significance=significance)

    def cusum_sq_test(self, significance: float = 0.05) -> CUSUMSQResults:
        """Test for variance instability using the CUSUM-SQ test.

        Convenience method that creates a CUSUMSQTest from this model
        and runs it. The CUSUM-SQ test examines the cumulative sum of
        squared recursive residuals for departures from variance stability.

        Parameters
        ----------
        significance : float
            Significance level for KS critical value. Any value in (0, 1).

        Returns
        -------
        CUSUMSQResults
            Test results with statistic path, boundaries, and decision.

        See Also
        --------
        CUSUMSQTest : The underlying test class.
        cusum_test : Test for parameter instability.
        """
        from regimes.tests.cusum import CUSUMSQTest

        test = CUSUMSQTest.from_model(self)
        return test.fit(significance=significance)

    def andrews_ploberger(
        self,
        break_vars: Literal["all", "const"] = "all",
        trimming: float = 0.15,
        significance: float = 0.05,
    ) -> AndrewsPlobergerResults:
        """Test for a structural break at unknown date using Andrews-Ploberger.

        Convenience method that creates an AndrewsPlobergerTest from this
        model and runs it. Reports SupF, ExpF, and AveF statistics.

        Parameters
        ----------
        break_vars : "all" | "const"
            Which variables can have breaks:
            - "all": All regressors can break (default)
            - "const": Only intercept can break (mean-shift model)
        trimming : float
            Fraction of observations trimmed from each end. Default is 0.15.
        significance : float
            Significance level for rejection decisions. Default is 0.05.

        Returns
        -------
        AndrewsPlobergerResults
            Test results with SupF, ExpF, AveF statistics and decisions.

        See Also
        --------
        AndrewsPlobergerTest : The underlying test class.
        chow_test : Test at known break points.
        bai_perron : Test for multiple breaks.
        """
        from regimes.tests.andrews_ploberger import AndrewsPlobergerTest

        test = AndrewsPlobergerTest.from_model(self, break_vars=break_vars)
        return test.fit(trimming=trimming, significance=significance)

    def rolling(self, window: int) -> RollingOLS:
        """Create a rolling OLS estimator from this model.

        Parameters
        ----------
        window : int
            Window size for rolling estimation. Must be at least k + 1
            where k is the number of parameters.

        Returns
        -------
        RollingOLS
            Rolling OLS estimator ready to be fitted.

        Examples
        --------
        >>> import numpy as np
        >>> from regimes import OLS
        >>> np.random.seed(42)
        >>> n = 200
        >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
        >>> y = X @ [1, 2] + np.random.randn(n)
        >>> model = OLS(y, X, has_constant=False)
        >>> rolling_results = model.rolling(window=60).fit()
        >>> print(rolling_results.summary())

        See Also
        --------
        RollingOLS : The rolling OLS estimator class.
        OLS.recursive : Recursive (expanding window) estimation.
        """
        from regimes.rolling.ols import RollingOLS

        return RollingOLS.from_model(self, window=window)

    def recursive(self, min_nobs: int | None = None) -> RecursiveOLS:
        """Create a recursive (expanding window) OLS estimator from this model.

        Parameters
        ----------
        min_nobs : int | None
            Minimum number of observations to start estimation. Defaults
            to k + 1 where k is the number of parameters.

        Returns
        -------
        RecursiveOLS
            Recursive OLS estimator ready to be fitted.

        Examples
        --------
        >>> import numpy as np
        >>> from regimes import OLS
        >>> np.random.seed(42)
        >>> n = 200
        >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
        >>> y = X @ [1, 2] + np.random.randn(n)
        >>> model = OLS(y, X, has_constant=False)
        >>> recursive_results = model.recursive(min_nobs=30).fit()
        >>> print(recursive_results.summary())

        See Also
        --------
        RecursiveOLS : The recursive OLS estimator class.
        OLS.rolling : Fixed window rolling estimation.
        """
        from regimes.rolling.ols import RecursiveOLS

        return RecursiveOLS.from_model(self, min_nobs=min_nobs)

    def markov_switching(
        self,
        k_regimes: int = 2,
        **kwargs: Any,
    ) -> MarkovRegressionResults:
        """Fit a Markov regime-switching version of this OLS model.

        Creates a MarkovRegression from this model's specification and
        fits it. This is a convenience method for one-way mapping from
        OLS to Markov switching.

        Parameters
        ----------
        k_regimes : int
            Number of regimes. Default is 2.
        **kwargs
            Additional keyword arguments passed to MarkovRegression and
            its fit() method. Model-level kwargs (e.g., switching_trend,
            switching_exog) are forwarded to MarkovRegression; fit-level
            kwargs (e.g., method, maxiter, em_iter, search_reps) are
            forwarded to fit().

        Returns
        -------
        MarkovRegressionResults
            Fitted Markov switching regression results.

        Examples
        --------
        >>> import numpy as np
        >>> from regimes import OLS
        >>> np.random.seed(42)
        >>> y = np.concatenate([np.random.randn(100), np.random.randn(100) + 3])
        >>> X = np.column_stack([np.ones(200), np.random.randn(200)])
        >>> model = OLS(y, X, has_constant=False)
        >>> ms_results = model.markov_switching(k_regimes=2)
        >>> print(ms_results.summary())

        See Also
        --------
        regimes.markov.MarkovRegression : The underlying MS model class.
        """
        from regimes.markov import MarkovRegression

        # Separate model-level kwargs from fit-level kwargs
        fit_kwargs_names = {"method", "maxiter", "em_iter", "search_reps"}
        model_kwargs = {k: v for k, v in kwargs.items() if k not in fit_kwargs_names}
        fit_kwargs = {k: v for k, v in kwargs.items() if k in fit_kwargs_names}

        ms_model = MarkovRegression.from_model(
            self, k_regimes=k_regimes, **model_kwargs
        )
        return ms_model.fit(**fit_kwargs)


def summary_by_regime(
    results_list: list[OLSResults],
    breaks: Sequence[int] | None = None,
    nobs_total: int | None = None,
) -> str:
    """Generate combined summary for regime-specific OLS results.

    Creates a formatted summary combining multiple OLS results from separate
    regime estimations, typically from `OLS.fit_by_regime()`.

    Parameters
    ----------
    results_list : list[OLSResults]
        Results from OLS.fit_by_regime(), one per regime.
    breaks : Sequence[int] | None
        Break points for labeling regime boundaries. If None, regimes are
        numbered without specific boundary information.
    nobs_total : int | None
        Total observations across all regimes. If None, computed as sum
        of observations in each result.

    Returns
    -------
    str
        Combined summary string with all regimes, showing coefficients
        and fit statistics for each regime separately.

    Examples
    --------
    >>> import numpy as np
    >>> from regimes import OLS, summary_by_regime
    >>> np.random.seed(42)
    >>> n = 200
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = np.random.randn(n)
    >>> model = OLS(y, X, breaks=[100])
    >>> regime_results = model.fit_by_regime()
    >>> print(summary_by_regime(regime_results, breaks=[100], nobs_total=200))
    """
    if not results_list:
        return "No results to summarize."

    lines = []
    lines.append("=" * 81)
    lines.append(f"{'OLS Regression Results by Regime':^81}")
    lines.append("=" * 81)

    # Show break information
    if breaks:
        break_str = ", ".join(str(bp) for bp in breaks)
        lines.append(f"Breaks at observations: {break_str}")
        lines.append("")

    # Compute nobs_total if not provided
    if nobs_total is None:
        nobs_total = sum(r.nobs for r in results_list)

    # Compute regime boundaries
    if breaks:
        boundaries = [0] + list(breaks) + [nobs_total]
    else:
        # Infer boundaries from results
        boundaries = [0]
        for r in results_list:
            boundaries.append(boundaries[-1] + r.nobs)

    # Add each regime's summary
    for i, result in enumerate(results_list):
        start = boundaries[i]
        end = boundaries[i + 1] - 1

        lines.append("-" * 81)
        lines.append(f"{'Regime ' + str(i + 1) + f' (obs {start}-{end})':^81}")
        lines.append("-" * 81)

        # Add fit statistics
        lines.append(
            f"No. Observations: {result.nobs:>6}   R-squared:     {result.rsquared:>10.4f}"
        )
        lines.append(
            f"Cov. Type:    {result.cov_type:>10}   Adj. R-squared: {result.rsquared_adj:>9.4f}"
        )
        lines.append(
            f"Residual Std Err: {result.sigma:>6.4f}   Residual Var:  {result.sigma_squared:>10.4f}"
        )
        lines.append("")

        # Parameter table header
        lines.append(
            f"{'':>15} {'coef':>10} {'std err':>10} {'t':>10} "
            f"{'P>|t|':>10} {'[0.025':>10} {'0.975]':>10}"
        )
        lines.append("-" * 81)

        # Parameter rows
        ci = result.conf_int()
        names = result.param_names or [f"x{j}" for j in range(len(result.params))]
        for j, name in enumerate(names):
            pval = result.pvalues[j]
            pval_str = f"{pval:.3f}" if pval >= 0.001 else f"{pval:.2e}"
            lines.append(
                f"{name:>15} {result.params[j]:>10.4f} {result.bse[j]:>10.4f} "
                f"{result.tvalues[j]:>10.3f} {pval_str:>10} "
                f"{ci[j, 0]:>10.3f} {ci[j, 1]:>10.3f}"
            )

        lines.append("")

        # Add note about covariance type
        if result.cov_type == "HAC":
            lines.append("Note: Standard errors are HAC (Newey-West) robust.")
        elif result.cov_type.startswith("HC"):
            lines.append(
                f"Note: Standard errors are {result.cov_type} heteroskedasticity-robust."
            )

        lines.append("")

    lines.append("=" * 81)
    return "\n".join(lines)
