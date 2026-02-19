"""Autoregressive models with structural break support.

This module provides AR(p) estimation with support for known structural
breaks and various covariance estimators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import statsmodels.api as sm

from regimes.diagnostics import DiagnosticsResults, compute_diagnostics
from regimes.models.base import CovType, TimeSeriesModelBase
from regimes.results.base import RegressionResultsBase

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    import pandas as pd
    from numpy.typing import ArrayLike, NDArray

    from regimes.markov.results import MarkovARResults
    from regimes.rolling.ar import RecursiveAR, RollingAR
    from regimes.tests.andrews_ploberger import AndrewsPlobergerResults
    from regimes.tests.bai_perron import BaiPerronResults
    from regimes.tests.chow import ChowTestResults
    from regimes.tests.cusum import CUSUMResults, CUSUMSQResults


@dataclass(kw_only=True)
class ARResults(RegressionResultsBase):
    """Results from AR model estimation.

    Extends RegressionResultsBase with AR-specific attributes.

    Additional Attributes
    ---------------------
    lags : list[int]
        Lag indices used in the model.
    ar_params : NDArray[np.floating]
        Autoregressive coefficients only (excluding constant/exog).
    roots : NDArray[np.complexfloating]
        Roots of the AR polynomial.
    """

    lags: list[int] = field(default_factory=list)
    ar_params: NDArray[np.floating[Any]] | None = None
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
    def roots(self) -> NDArray[np.complexfloating[Any, Any]]:
        """Roots of the AR characteristic polynomial.

        Returns
        -------
        NDArray[np.complexfloating]
            Roots of 1 - phi_1*z - phi_2*z^2 - ... - phi_p*z^p.
            For stationarity, all roots should be outside the unit circle.
        """
        if self.ar_params is None or len(self.ar_params) == 0:
            return np.array([], dtype=np.complex128)

        # Characteristic polynomial: 1 - phi_1*z - phi_2*z^2 - ...
        # numpy.roots expects coefficients in descending order
        coeffs = np.concatenate([[1], -self.ar_params])[::-1]
        return np.roots(coeffs)

    @property
    def is_stationary(self) -> bool:
        """Check if the AR process is stationary.

        Returns True if all roots are outside the unit circle.
        """
        if len(self.roots) == 0:
            return True
        return bool(np.all(np.abs(self.roots) > 1))

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
    def sigma_squared(self) -> float:
        """Residual variance (sigma^2)."""
        return self.scale if self.scale is not None else np.nan

    @property
    def sigma(self) -> float:
        """Residual standard error (sigma)."""
        return np.sqrt(self.sigma_squared)

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

        Notes
        -----
        Break indices are shown in original sample coordinates (before lag
        adjustment). This matches user input and makes interpretation easier.
        """
        lines = []
        # Use original nobs if available, otherwise current nobs
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
        """Generate a text summary of AR results.

        Parameters
        ----------
        diagnostics : bool, default True
            If True, include misspecification tests (autocorrelation, ARCH,
            normality, heteroskedasticity) in the output.

        Returns
        -------
        str
            Formatted summary including AR coefficients and optionally
            misspecification tests.
        """
        lines = []
        lines.append("=" * 81)
        lines.append(f"{'AR Model Results':^81}")
        lines.append("=" * 81)
        lines.append(
            f"Dep. Variable:           y   No. Observations:    {self.nobs:>10}"
        )
        model_str = f"AR({max(self.lags) if self.lags else 0})"
        lines.append(
            f"Model:          {model_str:>10}   Df Residuals:        {self.df_resid:>10}"
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
                f"                             BIC:                 {self.bic:>10.2f}"
            )

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

        # Stationarity check
        if self.ar_params is not None and len(self.ar_params) > 0:
            if self.is_stationary:
                lines.append("Roots are outside the unit circle (stationary).")
            else:
                lines.append(
                    "WARNING: Some roots are inside the unit circle (non-stationary)."
                )

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


class AR(TimeSeriesModelBase):
    """Autoregressive model with structural break support.

    Estimates AR(p) models with support for known structural breaks
    and various covariance estimators including HAC standard errors.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    lags : int | Sequence[int]
        Number of lags (if int) or specific lag indices (if sequence).
        Default is 1.
    exog : ArrayLike | None
        Additional exogenous regressors (n_obs, k). Constant is always
        included.
    breaks : Sequence[int] | None
        Known break points (observation indices). If provided, AR
        coefficients are estimated separately for each regime. All variables
        share these break points.
    variable_breaks : dict[str | int, Sequence[int]] | None
        Variable-specific break points. Keys can be variable names (str) or
        column indices (int). Values are lists of break points for that
        variable. Variables not in the dict have no breaks (constant
        coefficient across the sample). Cannot be used together with `breaks`.
    trend : str
        Trend to include: "c" (constant only, default), "ct" (constant
        and trend), "n" (no deterministic terms).

    Examples
    --------
    >>> import numpy as np
    >>> from regimes import AR
    >>> np.random.seed(42)
    >>> # Simulate AR(1) process
    >>> n = 200
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = 0.8 * y[t-1] + np.random.randn()
    >>> model = AR(y, lags=1)
    >>> results = model.fit(cov_type="HAC")
    >>> print(results.summary())

    >>> # AR with known break
    >>> model_break = AR(y, lags=1, breaks=[100])
    >>> results_break = model_break.fit()
    >>> print(results_break.summary())

    Variable-specific breaks:

    >>> # Break only in constant (intercept shift), AR coefficient stable
    >>> model = AR(y, lags=1, variable_breaks={"const": [100]})
    >>> results = model.fit()
    >>> print(results.summary())

    Notes
    -----
    The model is estimated using OLS on the equation:
        y_t = c + phi_1 * y_{t-1} + ... + phi_p * y_{t-p} + X_t'*beta + e_t

    When breaks are specified, all coefficients (including the constant
    and AR parameters) are allowed to differ across regimes.

    When variable_breaks is specified, each variable can have its own
    set of break points, allowing for different timing of structural
    changes across coefficients.
    """

    def __init__(
        self,
        endog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        lags: int | Sequence[int] = 1,
        exog: ArrayLike | pd.Series[Any] | pd.DataFrame | None = None,
        breaks: Sequence[int] | None = None,
        variable_breaks: dict[str | int, Sequence[int]] | None = None,
        trend: str = "c",
    ) -> None:
        """Initialize AR model."""
        super().__init__(endog, exog, lags, breaks)

        # Validate that breaks and variable_breaks are not both specified
        if breaks is not None and variable_breaks is not None:
            raise ValueError(
                "Cannot specify both 'breaks' and 'variable_breaks'. "
                "Use 'breaks' for common breaks across all variables, or "
                "'variable_breaks' for variable-specific breaks."
            )

        self._variable_breaks = variable_breaks
        self.trend = trend

        if trend not in ("c", "ct", "n"):
            raise ValueError(f"trend must be 'c', 'ct', or 'n', got {trend}")

    def _build_design_matrix(
        self,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], list[str]]:
        """Build the design matrix for AR estimation.

        Returns
        -------
        tuple[NDArray[np.floating], NDArray[np.floating], list[str]]
            y (dependent), X (design matrix), and parameter names.
        """
        # Effective sample (after dropping initial observations for lags)
        y = self.endog[self.maxlag :]
        n_eff = len(y)

        # Build design matrix components
        components = []
        param_names: list[str] = []

        # Deterministic terms
        if self.trend in ("c", "ct"):
            components.append(np.ones((n_eff, 1)))
            param_names.append("const")

        if self.trend == "ct":
            trend_var = np.arange(self.maxlag + 1, self.nobs + 1).reshape(-1, 1)
            components.append(trend_var)
            param_names.append("trend")

        # Lagged dependent variable
        lag_matrix = self._create_lag_matrix(self.endog)
        components.append(lag_matrix)
        param_names.extend([f"y.L{lag}" for lag in self.lags])

        # Exogenous variables
        if self.exog is not None:
            X_eff = self.exog[self.maxlag :]
            components.append(X_eff)
            param_names.extend(self._exog_names)

        X = np.column_stack(components)
        return y, X, param_names

    def _normalize_variable_breaks(
        self, param_names: list[str]
    ) -> dict[int, list[int]] | None:
        """Normalize variable_breaks keys to column indices.

        Parameters
        ----------
        param_names : list[str]
            Parameter names from the design matrix.

        Returns
        -------
        dict[int, list[int]] | None
            Dictionary mapping column indices to sorted break points
            (adjusted for effective sample), or None if variable_breaks
            is not specified.

        Raises
        ------
        ValueError
            If variable name or index is invalid, or break points are
            out of bounds.
        """
        if self._variable_breaks is None:
            return None

        k = len(param_names)
        normalized: dict[int, list[int]] = {}

        for key, breaks in self._variable_breaks.items():
            # Convert key to index
            if isinstance(key, int):
                if key < 0 or key >= k:
                    raise ValueError(
                        f"Variable index {key} is out of bounds for design "
                        f"matrix with {k} columns"
                    )
                idx = key
            else:
                # key is a string, find it in param_names
                if key not in param_names:
                    raise ValueError(
                        f"Variable name '{key}' not found. "
                        f"Available names: {param_names}"
                    )
                idx = param_names.index(key)

            # Validate and sort break points (in original sample coordinates)
            breaks_list = sorted(breaks)
            for bp in breaks_list:
                if bp <= 0 or bp >= self.nobs:
                    raise ValueError(
                        f"Break point {bp} for variable '{key}' is out of bounds. "
                        f"Must be in range (0, {self.nobs})."
                    )

            # Adjust for effective sample (subtract maxlag)
            adjusted_breaks = [
                bp - self.maxlag for bp in breaks_list if bp > self.maxlag
            ]
            if adjusted_breaks:
                normalized[idx] = adjusted_breaks

        return normalized if normalized else None

    def _get_variable_regime_indices(
        self, breaks: list[int], n_eff: int
    ) -> list[tuple[int, int]]:
        """Get regime start/end indices for a variable's break points.

        Parameters
        ----------
        breaks : list[int]
            Sorted list of break points (in effective sample coordinates).
        n_eff : int
            Effective sample size.

        Returns
        -------
        list[tuple[int, int]]
            List of (start, end) tuples for each regime.
        """
        if not breaks:
            return [(0, n_eff)]

        indices = []
        prev = 0
        for bp in breaks:
            indices.append((prev, bp))
            prev = bp
        indices.append((prev, n_eff))
        return indices

    def _create_variable_break_design(
        self,
        y: NDArray[np.floating[Any]],
        X: NDArray[np.floating[Any]],
        param_names: list[str],
    ) -> tuple[NDArray[np.floating[Any]], list[str]]:
        """Create design matrix with variable-specific break points.

        Parameters
        ----------
        y : NDArray[np.floating]
            Effective dependent variable.
        X : NDArray[np.floating]
            Design matrix without regime interactions.
        param_names : list[str]
            Parameter names without regime suffixes.

        Returns
        -------
        tuple[NDArray[np.floating], list[str]]
            Design matrix with variable-specific regime columns and
            parameter names.
        """
        normalized_breaks = self._normalize_variable_breaks(param_names)
        if normalized_breaks is None:
            return X, param_names

        n_eff = len(y)
        k = X.shape[1]

        # Build columns for each variable
        columns = []
        new_param_names = []

        for var_idx in range(k):
            var_name = param_names[var_idx]
            var_data = X[:, var_idx]

            if var_idx in normalized_breaks:
                # This variable has breaks
                breaks = normalized_breaks[var_idx]
                regime_indices = self._get_variable_regime_indices(breaks, n_eff)

                for r, (start, end) in enumerate(regime_indices):
                    col = np.zeros(n_eff)
                    col[start:end] = var_data[start:end]
                    columns.append(col)
                    new_param_names.append(f"{var_name}_regime{r + 1}")
            else:
                # No breaks for this variable - constant coefficient
                columns.append(var_data)
                new_param_names.append(var_name)

        X_expanded = np.column_stack(columns)
        return X_expanded, new_param_names

    def _create_break_design(
        self,
        y: NDArray[np.floating[Any]],
        X: NDArray[np.floating[Any]],
        param_names: list[str],
    ) -> tuple[NDArray[np.floating[Any]], list[str]]:
        """Create design matrix with regime-specific coefficients.

        Parameters
        ----------
        y : NDArray[np.floating]
            Effective dependent variable.
        X : NDArray[np.floating]
            Design matrix without regime interactions.
        param_names : list[str]
            Parameter names without regime suffixes.

        Returns
        -------
        tuple[NDArray[np.floating], list[str]]
            Design matrix with regime interactions and updated names.
        """
        # Handle variable-specific breaks first
        if self._variable_breaks is not None:
            return self._create_variable_break_design(y, X, param_names)

        # Handle common breaks
        if not self.breaks:
            return X, param_names

        # Adjust break indices for effective sample
        adjusted_breaks = [b - self.maxlag for b in self.breaks if b >= self.maxlag]

        if not adjusted_breaks:
            return X, param_names

        # Create regime indices for effective sample
        n_eff = len(y)
        sorted_breaks = sorted(adjusted_breaks)
        regime_indices = []

        regime_indices.append((0, sorted_breaks[0]))
        for i in range(len(sorted_breaks) - 1):
            regime_indices.append((sorted_breaks[i], sorted_breaks[i + 1]))
        regime_indices.append((sorted_breaks[-1], n_eff))

        # Create expanded design matrix
        n_regimes = len(regime_indices)
        k = X.shape[1]
        X_expanded = np.zeros((n_eff, k * n_regimes))
        new_param_names = []

        for r, (start, end) in enumerate(regime_indices):
            col_start = r * k
            col_end = (r + 1) * k
            X_expanded[start:end, col_start:col_end] = X[start:end]

            for name in param_names:
                new_param_names.append(f"{name}_regime{r + 1}")

        return X_expanded, new_param_names

    def fit(
        self,
        cov_type: CovType = "nonrobust",
        cov_kwds: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ARResults:
        """Fit the AR model.

        Parameters
        ----------
        cov_type : CovType
            Type of covariance estimator:
            - "nonrobust": Standard OLS covariance
            - "HC0", "HC1", "HC2", "HC3": Heteroskedasticity-robust
            - "HAC": Heteroskedasticity and autocorrelation consistent
        cov_kwds : dict | None
            Additional keywords for covariance estimation. For HAC, can
            include 'maxlags'.
        **kwargs
            Additional arguments (reserved for future use).

        Returns
        -------
        ARResults
            Results object containing estimates and inference.
        """
        cov_kwds = cov_kwds or {}

        # Build design matrix
        y, X, param_names = self._build_design_matrix()

        # Handle breaks
        X, param_names = self._create_break_design(y, X, param_names)

        # Use statsmodels for estimation
        sm_model = sm.OLS(y, X)

        # Map cov_type to statsmodels format
        if cov_type == "nonrobust":
            sm_results = sm_model.fit()
        elif cov_type.startswith("HC"):
            sm_results = sm_model.fit(cov_type=cov_type)
        elif cov_type == "HAC":
            maxlags = cov_kwds.get("maxlags")
            if maxlags is None:
                maxlags = int(np.floor(4 * (len(y) / 100) ** (2 / 9)))
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

        # Extract AR parameters (depends on whether breaks are present)
        ar_params = self._extract_ar_params(params, param_names)

        # Compute TSS for R-squared
        y_mean = np.mean(y)
        tss = float(np.sum((y - y_mean) ** 2))

        return ARResults(
            params=params,
            bse=bse,
            resid=resid,
            fittedvalues=fittedvalues,
            cov_params_matrix=cov_params,
            nobs=len(y),
            cov_type=cov_type,
            param_names=param_names,
            model_name="AR",
            lags=self.lags,
            ar_params=ar_params,
            llf=float(sm_results.llf),
            scale=float(sm_results.scale),
            _tss=tss,
            _exog=X,  # Store for diagnostic tests
            _breaks=list(self.breaks) if self.breaks else None,
            _variable_breaks=self._variable_breaks,
            _nobs_original=self.nobs,  # Original nobs before dropping lags
        )

    def _extract_ar_params(
        self, params: NDArray[np.floating[Any]], param_names: list[str]
    ) -> NDArray[np.floating[Any]]:
        """Extract AR coefficients from full parameter vector.

        Parameters
        ----------
        params : NDArray[np.floating]
            Full parameter vector.
        param_names : list[str]
            Parameter names.

        Returns
        -------
        NDArray[np.floating]
            AR coefficients only (from first regime if breaks present).
        """
        ar_indices = [
            i
            for i, name in enumerate(param_names)
            if name.startswith("y.L") and ("regime1" in name or "regime" not in name)
        ]
        return params[ar_indices] if ar_indices else np.array([])

    def predict(
        self,
        params: NDArray[np.floating[Any]] | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Generate in-sample predictions.

        Parameters
        ----------
        params : NDArray[np.floating] | None
            Parameter values. If None, model must be fitted first.
        start : int | None
            Start index for predictions (default: maxlag).
        end : int | None
            End index for predictions (default: nobs).

        Returns
        -------
        NDArray[np.floating]
            Predicted values.

        Raises
        ------
        ValueError
            If params is None and model hasn't been fitted.
        """
        if params is None:
            raise ValueError("params must be provided")

        if start is None:
            start = self.maxlag
        if end is None:
            end = self.nobs

        y, X, _ = self._build_design_matrix()

        # Adjust for effective sample
        adj_start = max(0, start - self.maxlag)
        adj_end = min(len(y), end - self.maxlag)

        return X[adj_start:adj_end] @ params

    def fit_by_regime(
        self,
        cov_type: CovType = "nonrobust",
        cov_kwds: dict[str, Any] | None = None,
    ) -> list[ARResults]:
        """Fit separate AR models for each regime.

        This is an alternative to the pooled estimation with regime
        interactions. It fits completely separate models for each regime,
        with each regime losing its first `maxlag` observations for lag
        initialization.

        Parameters
        ----------
        cov_type : CovType
            Type of covariance estimator.
        cov_kwds : dict | None
            Additional keywords for covariance estimation.

        Returns
        -------
        list[ARResults]
            List of results, one per regime.

        Notes
        -----
        Unlike the pooled estimation with regime-interacted design matrix,
        this method estimates each regime completely independently. Each
        regime loses `maxlag` observations at its start for lag initialization.
        """
        if not self.breaks:
            return [self.fit(cov_type=cov_type, cov_kwds=cov_kwds)]

        results = []
        regime_indices = self.get_regime_indices()

        for _regime, (start, end) in enumerate(regime_indices):
            y_regime = self.endog[start:end]
            exog_regime = self.exog[start:end] if self.exog is not None else None

            regime_model = AR(
                y_regime,
                lags=self.lags,
                exog=exog_regime,
                breaks=None,
                trend=self.trend,
            )
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

        Convenience method that creates a BaiPerronTest from this AR model
        and runs it. The test uses the effective sample (after dropping
        initial observations for lags).

        Parameters
        ----------
        break_vars : "all" | "const"
            Which variables can have breaks:
            - "all": All regressors can break (default), including constant
              and AR parameters
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
            Test results. Use .to_ols() to get fitted model with breaks
            (note: this returns an OLSResults, not ARResults, as the
            underlying estimation uses the AR design matrix as regressors).

        Examples
        --------
        >>> import numpy as np
        >>> from regimes import AR
        >>> np.random.seed(42)
        >>> n = 200
        >>> y = np.zeros(n)
        >>> # Regime 1: AR(1) with phi=0.3
        >>> for t in range(1, 100):
        ...     y[t] = 0.3 * y[t-1] + np.random.randn()
        >>> # Regime 2: AR(1) with phi=0.8
        >>> for t in range(100, n):
        ...     y[t] = 0.8 * y[t-1] + np.random.randn()
        >>> model = AR(y, lags=1)
        >>> bp_results = model.bai_perron()
        >>> print(f"Detected {bp_results.n_breaks} breaks at {bp_results.break_indices}")

        See Also
        --------
        BaiPerronTest : The underlying test class.
        BaiPerronResults.to_ols : Convert results to OLS with breaks.

        Notes
        -----
        The Bai-Perron test is applied to the AR design matrix, which includes
        the constant, lagged dependent variables, and any exogenous regressors.
        Break indices are reported in effective sample coordinates (after
        dropping initial observations for lags).
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

        Convenience method that creates a ChowTest from this AR model
        and runs it. The test uses the effective sample (after dropping
        initial observations for lags).

        Parameters
        ----------
        break_points : int | Sequence[int]
            One or more break point indices to test (in effective sample
            coordinates, after lag trimming).
        break_vars : "all" | "const"
            Which variables can have breaks:
            - "all": All regressors can break (default), including constant
              and AR parameters
            - "const": Only intercept can break (mean-shift model)
        significance : float
            Significance level for rejection decisions. Default is 0.05.

        Returns
        -------
        ChowTestResults
            Test results with F-statistics and p-values.

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

        Convenience method that creates a CUSUMTest from this AR model
        and runs it. The test uses the effective sample (after dropping
        initial observations for lags).

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

        Convenience method that creates a CUSUMSQTest from this AR model
        and runs it. The test uses the effective sample (after dropping
        initial observations for lags).

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
        AR model and runs it. The test uses the effective sample (after
        dropping initial observations for lags).

        Parameters
        ----------
        break_vars : "all" | "const"
            Which variables can have breaks:
            - "all": All regressors can break (default), including constant
              and AR parameters
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

    def rolling(self, window: int) -> RollingAR:
        """Create a rolling AR estimator from this model.

        Parameters
        ----------
        window : int
            Window size for rolling estimation. Must be at least k + 1
            where k is the number of parameters.

        Returns
        -------
        RollingAR
            Rolling AR estimator ready to be fitted.

        Examples
        --------
        >>> import numpy as np
        >>> from regimes import AR
        >>> np.random.seed(42)
        >>> n = 200
        >>> y = np.zeros(n)
        >>> for t in range(1, n):
        ...     y[t] = 0.7 * y[t-1] + np.random.randn()
        >>> model = AR(y, lags=1)
        >>> rolling_results = model.rolling(window=60).fit()
        >>> print(rolling_results.summary())

        See Also
        --------
        RollingAR : The rolling AR estimator class.
        AR.recursive : Recursive (expanding window) estimation.
        """
        from regimes.rolling.ar import RollingAR

        return RollingAR.from_model(self, window=window)

    def recursive(self, min_nobs: int | None = None) -> RecursiveAR:
        """Create a recursive (expanding window) AR estimator from this model.

        Parameters
        ----------
        min_nobs : int | None
            Minimum number of observations to start estimation. Defaults
            to k + 1 where k is the number of parameters.

        Returns
        -------
        RecursiveAR
            Recursive AR estimator ready to be fitted.

        Examples
        --------
        >>> import numpy as np
        >>> from regimes import AR
        >>> np.random.seed(42)
        >>> n = 200
        >>> y = np.zeros(n)
        >>> for t in range(1, n):
        ...     y[t] = 0.7 * y[t-1] + np.random.randn()
        >>> model = AR(y, lags=1)
        >>> recursive_results = model.recursive(min_nobs=30).fit()
        >>> print(recursive_results.summary())

        See Also
        --------
        RecursiveAR : The recursive AR estimator class.
        AR.rolling : Fixed window rolling estimation.
        """
        from regimes.rolling.ar import RecursiveAR

        return RecursiveAR.from_model(self, min_nobs=min_nobs)

    def markov_switching(
        self,
        k_regimes: int = 2,
        **kwargs: Any,
    ) -> MarkovARResults:
        """Fit a Markov regime-switching version of this AR model.

        Creates a MarkovAR from this model's specification and fits it.

        Parameters
        ----------
        k_regimes : int
            Number of regimes. Default is 2.
        **kwargs
            Additional keyword arguments. Model-level kwargs (e.g.,
            switching_ar, switching_trend) forwarded to MarkovAR;
            fit-level kwargs (method, maxiter, etc.) forwarded to fit().

        Returns
        -------
        MarkovARResults
            Fitted Markov switching AR results.

        See Also
        --------
        regimes.markov.MarkovAR : The underlying MS AR model class.
        """
        from regimes.markov import MarkovAR

        fit_kwargs_names = {"method", "maxiter", "em_iter", "search_reps"}
        model_kwargs = {k: v for k, v in kwargs.items() if k not in fit_kwargs_names}
        fit_kwargs = {k: v for k, v in kwargs.items() if k in fit_kwargs_names}

        ms_model = MarkovAR.from_model(self, k_regimes=k_regimes, **model_kwargs)
        return ms_model.fit(**fit_kwargs)


def ar_summary_by_regime(
    results_list: list[ARResults],
    breaks: Sequence[int] | None = None,
    nobs_total: int | None = None,
    diagnostics: bool = True,
) -> str:
    """Generate combined summary for regime-specific AR results.

    Creates a formatted summary combining multiple AR results from separate
    regime estimations, typically from `AR.fit_by_regime()`.

    Parameters
    ----------
    results_list : list[ARResults]
        Results from AR.fit_by_regime(), one per regime.
    breaks : Sequence[int] | None
        Break points for labeling regime boundaries. If None, regimes are
        numbered without specific boundary information.
    nobs_total : int | None
        Total observations across all regimes (in original sample, before
        lag adjustments). If None, computed as sum of observations in each
        result plus the maximum lag.
    diagnostics : bool
        If True (default), include misspecification tests (autocorrelation,
        ARCH, normality, heteroskedasticity) for each regime.

    Returns
    -------
    str
        Combined summary string with all regimes, showing coefficients
        and fit statistics for each regime separately.

    Examples
    --------
    >>> import numpy as np
    >>> from regimes import AR, ar_summary_by_regime
    >>> np.random.seed(42)
    >>> n = 200
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = 0.8 * y[t-1] + np.random.randn()
    >>> model = AR(y, lags=1, breaks=[100])
    >>> regime_results = model.fit_by_regime()
    >>> print(ar_summary_by_regime(regime_results, breaks=[100], nobs_total=200))
    """
    if not results_list:
        return "No results to summarize."

    lines = []
    lines.append("=" * 81)
    lines.append(f"{'AR Model Results by Regime':^81}")
    lines.append("=" * 81)

    # Show break information
    if breaks:
        break_str = ", ".join(str(bp) for bp in breaks)
        lines.append(f"Breaks at observations: {break_str}")
        lines.append("")

    # Compute nobs_total if not provided
    # For AR models, effective nobs = original_nobs - maxlag per regime
    if nobs_total is None:
        # Each result.nobs is effective (after dropping lags)
        # We add back the maxlag for each regime
        maxlag = max(results_list[0].lags) if results_list[0].lags else 0
        nobs_total = sum(r.nobs + maxlag for r in results_list)

    # Compute regime boundaries (in original sample coordinates)
    if breaks:
        boundaries = [0] + list(breaks) + [nobs_total]
    else:
        # Infer boundaries from results (using effective nobs)
        maxlag = max(results_list[0].lags) if results_list[0].lags else 0
        boundaries = [0]
        for r in results_list:
            boundaries.append(boundaries[-1] + r.nobs + maxlag)

    # Add each regime's summary
    for i, result in enumerate(results_list):
        start = boundaries[i]
        end = boundaries[i + 1] - 1

        lines.append("-" * 81)
        lines.append(f"{'Regime ' + str(i + 1) + f' (obs {start}-{end})':^81}")
        lines.append("-" * 81)

        # Add fit statistics (matching ARResults.summary() format)
        model_str = f"AR({max(result.lags) if result.lags else 0})"
        lines.append(
            f"Model:            {model_str:>10}   No. Observations:        {result.nobs:>6}"
        )
        lines.append(
            f"Cov. Type:        {result.cov_type:>10}   Df Residuals:            {result.df_resid:>6}"
        )
        lines.append(
            f"R-squared:           {result.rsquared:>7.4f}   Adj. R-squared:      {result.rsquared_adj:>10.4f}"
        )
        lines.append(
            f"Residual Std Err:    {result.sigma:>7.4f}   Residual Variance:   {result.sigma_squared:>10.4f}"
        )
        if result.llf is not None:
            lines.append(
                f"Log-Likelihood:      {result.llf:>7.2f}   AIC:                 {result.aic:>10.2f}"
            )
            lines.append(
                f"                               BIC:                 {result.bic:>10.2f}"
            )
        lines.append("")

        # Stationarity check
        if result.ar_params is not None and len(result.ar_params) > 0:
            if result.is_stationary:
                lines.append("Roots outside unit circle (stationary).")
            else:
                lines.append("WARNING: Some roots inside unit circle (non-stationary).")
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

        # Add diagnostics if available
        if diagnostics and result._exog is not None:
            try:
                diag = result.diagnostics()
                lines.append("")
                lines.append(diag.summary())
            except Exception:
                pass  # Skip diagnostics if they fail

        lines.append("")

    lines.append("=" * 81)
    return "\n".join(lines)
