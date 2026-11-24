"""Autoregressive Distributed Lag (ADL) models with structural break support.

This module provides ADL(p,q) estimation with support for known structural
breaks and various covariance estimators.

ADL(p,q) specification:
    y_t = c + Σ(φ_i * y_{t-i}) + Σ(β_j * x_{t-j}) + ε_t
              i=1 to p              j=0 to q

Key relationships:
    - AR(p) = ADL(p,0): AR is ADL with no distributed lags on exogenous variables
    - DL(q) = ADL(0,q): Distributed lag model is ADL with no autoregressive terms
    - OLS = ADL(0,0): Static regression is ADL with no lags at all
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import statsmodels.api as sm

from regimes.diagnostics import DiagnosticsResults, compute_diagnostics
from regimes.models._dynamics import (
    ar_coefficients_by_segment,
    ar_roots,
    distributed_lag_effects_by_segment,
)
from regimes.models.base import (
    CovType,
    RegimesModelBase,
    _ensure_array,
    check_break_design,
    fit_with_covariance,
    normalize_breaks,
)
from regimes.results.base import RegressionResultsBase, _obs_label, _obs_range_label

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    import pandas as pd
    from numpy.typing import ArrayLike, NDArray

    from regimes.markov.results import MarkovADLResults
    from regimes.results.base import IndexLike
    from regimes.rolling.adl import RecursiveADL, RollingADL
    from regimes.tests.andrews_ploberger import AndrewsPlobergerResults
    from regimes.tests.bai_perron import BaiPerronResults
    from regimes.tests.chow import ChowTestResults
    from regimes.tests.cusum import CUSUMResults, CUSUMSQResults


@dataclass(kw_only=True)
class ADLResults(RegressionResultsBase):
    """Results from ADL model estimation.

    Extends RegressionResultsBase with ADL-specific attributes.

    Additional Attributes
    ---------------------
    lags : list[int]
        Autoregressive lag indices used in the model.
    exog_lags : dict[str, list[int]]
        Distributed lag structure per exogenous variable.
    ar_params : NDArray[np.floating]
        Autoregressive coefficients only (excluding constant/exog).
    dl_params : dict[str, NDArray[np.floating]]
        Distributed lag coefficients by variable.
    llf : float | None
        Log-likelihood value.
    scale : float | None
        Residual variance.
    """

    lags: list[int] = field(default_factory=list)
    exog_lags: dict[str, list[int]] = field(default_factory=dict)
    ar_params: NDArray[np.floating[Any]] | None = None
    dl_params: dict[str, NDArray[np.floating[Any]]] | None = None
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

    def _ar_segments(self) -> list[tuple[int, dict[int, float]]]:
        """Autoregressive coefficients by lag, for each stretch of the sample
        between break dates of the autoregressive coefficients."""
        names = list(self.param_names) if self.param_names is not None else []
        if not names:
            return []
        return ar_coefficients_by_segment(
            names, self.params, self._breaks, self._variable_breaks
        )

    @property
    def roots_by_regime(self) -> list[NDArray[np.complexfloating[Any, Any]]]:
        """Roots of the AR characteristic polynomial in each regime.

        One array for each stretch of the sample in which the autoregressive
        coefficients are constant (a single array without breaks). The
        polynomial is ``1 - sum_L phi_L z^L`` over the lags ``L`` in the
        model, so non-consecutive lags are placed at their own powers.
        """
        return [ar_roots(coefficients) for _start, coefficients in self._ar_segments()]

    @property
    def roots(self) -> NDArray[np.complexfloating[Any, Any]]:
        """Roots of the AR characteristic polynomial (first regime).

        Returns
        -------
        NDArray[np.complexfloating]
            Roots of ``1 - sum_L phi_L z^L``. With breaks these are the
            roots of the first regime; see ``roots_by_regime``.
        """
        by_regime = self.roots_by_regime
        if not by_regime:
            return np.array([], dtype=np.complex128)
        return by_regime[0]

    @property
    def is_stationary(self) -> bool:
        """Check if the AR component is stationary in every regime.

        Returns True if all roots of every regime are outside the unit
        circle.
        """
        return all(
            len(roots) == 0 or bool(np.all(np.abs(roots) > 1))
            for roots in self.roots_by_regime
        )

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

    @property
    def effects_by_regime(self) -> list[tuple[int, dict[str, tuple[float, float]]]]:
        """Cumulative effect and long-run multiplier in each regime.

        Returns
        -------
        list[tuple[int, dict[str, tuple[float, float]]]]
            One entry for each stretch of the sample in which the
            autoregressive and distributed-lag coefficients are constant:
            ``(start, {variable: (cumulative, long_run)})``, where ``start``
            is the first observation of the stretch (0 for the first). A
            model without breaks has one entry.
        """
        names = list(self.param_names) if self.param_names is not None else []
        if not names:
            return []
        return distributed_lag_effects_by_segment(
            names, self.params, self.exog_lags, self._breaks, self._variable_breaks
        )

    @property
    def cumulative_effect(self) -> dict[str, float]:
        """Sum of distributed lag coefficients per exogenous variable.

        The cumulative effect (or total multiplier) measures the total
        effect of a unit change in x on y, summed across all lags.

        With breaks this is the effect in the first regime; see
        ``effects_by_regime`` for every regime.

        Returns
        -------
        dict[str, float]
            Cumulative effect for each exogenous variable.
        """
        by_regime = self.effects_by_regime
        if not by_regime:
            return {}
        return {var: values[0] for var, values in by_regime[0][1].items()}

    @property
    def long_run_multiplier(self) -> dict[str, float]:
        """Long-run effect: Σβ / (1 - Σφ).

        The long-run multiplier measures the equilibrium effect of a
        permanent unit change in x on y, accounting for the autoregressive
        dynamics. It is NaN if the sum of the autoregressive coefficients is
        one or more in absolute value.

        With breaks this is the multiplier of the first regime; see
        ``effects_by_regime`` for every regime. No standard error is
        computed.

        Returns
        -------
        dict[str, float]
            Long-run multiplier for each exogenous variable.
        """
        by_regime = self.effects_by_regime
        if not by_regime:
            return {}
        return {var: values[1] for var, values in by_regime[0][1].items()}

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
            Collection of diagnostic test results.
        """
        if self._exog is None:
            raise ValueError(
                "Regressors (exog) not available. Diagnostics require the "
                "original design matrix to be stored during fitting."
            )

        # Check if we need to recompute (cache invalidation by lag changes)
        if self._diagnostics_cache is not None:
            cached = self._diagnostics_cache
            if (
                cached.autocorrelation is not None
                and cached.arch is not None
                and cached.autocorrelation.df == lags_autocorr
                and cached.arch.df == lags_arch
            ):
                return cached

        self._diagnostics_cache = compute_diagnostics(
            self.resid, self._exog, lags_autocorr=lags_autocorr, lags_arch=lags_arch
        )
        return self._diagnostics_cache

    def _format_break_section(self, index: IndexLike | None = None) -> list[str]:
        """Format the structural breaks section for summary output.

        Parameters
        ----------
        index : IndexLike or None
            Optional time index for date labels.
        """
        lines = []
        nobs = self._nobs_original if self._nobs_original is not None else self.nobs

        if self._breaks:
            lines.append("-" * 81)
            lines.append("Structural Breaks")
            lines.append("-" * 81)

            for bp in self._breaks:
                lines.append(f"Break at {_obs_label(bp, index)}")

            boundaries = [0] + list(self._breaks) + [nobs]
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1] - 1
                n_regime = boundaries[i + 1] - boundaries[i]
                range_str = _obs_range_label(start, end, index)
                lines.append(f"  Regime {i + 1}: {range_str} (n={n_regime})")

        elif self._variable_breaks:
            lines.append("-" * 81)
            lines.append("Variable-Specific Structural Breaks")
            lines.append("-" * 81)

            for var_name, unsorted_breaks in self._variable_breaks.items():
                breaks = sorted(unsorted_breaks)
                break_str = ", ".join(_obs_label(bp, index) for bp in breaks)
                if len(breaks) == 1:
                    lines.append(f"{var_name}: break at {break_str}")
                else:
                    lines.append(f"{var_name}: breaks at {break_str}")

                boundaries = [0] + list(breaks) + [nobs]
                regime_parts = []
                for i in range(len(boundaries) - 1):
                    start = boundaries[i]
                    end = boundaries[i + 1] - 1
                    range_str = _obs_range_label(start, end, index)
                    regime_parts.append(f"Regime {i + 1}: {range_str}")
                lines.append(f"  {', '.join(regime_parts)}")

        return lines

    def summary(
        self,
        diagnostics: bool = True,
        index: IndexLike | None = None,
    ) -> str:
        """Generate a text summary of ADL results.

        Parameters
        ----------
        diagnostics : bool, default True
            If True, include misspecification tests.
        index : IndexLike or None
            Optional time index (e.g. ``pd.PeriodIndex``). When provided,
            observation numbers in the break section are replaced with
            date labels.

        Returns
        -------
        str
            Formatted summary.
        """
        lines = []
        lines.append("=" * 81)
        lines.append(f"{'ADL Model Results':^81}")
        lines.append("=" * 81)
        lines.append(
            f"Dep. Variable:           y   No. Observations:    {self.nobs:>10}"
        )

        # Model specification string
        p = max(self.lags) if self.lags else 0
        if self.exog_lags:
            q_vals = [max(lags) if lags else 0 for lags in self.exog_lags.values()]
            q = max(q_vals) if q_vals else 0
        else:
            q = 0
        model_str = f"ADL({p},{q})"

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
            lines.extend(self._format_break_section(index=index))
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
            where = " in every regime" if len(self.roots_by_regime) > 1 else ""
            if self.is_stationary:
                lines.append(
                    f"AR roots are outside the unit circle{where} (stationary)."
                )
            else:
                lines.append(
                    "WARNING: Some AR roots are inside the unit circle (non-stationary)."
                )

        # Distributed lag effects
        effects = self.effects_by_regime
        if effects:
            lines.append("-" * 81)
            lines.append("Distributed Lag Effects:")
            for i, (start, by_variable) in enumerate(effects):
                if len(effects) > 1:
                    lines.append(f"  Regime {i + 1} (from {_obs_label(start, index)}):")
                indent = "    " if len(effects) > 1 else "  "
                for var, (cum_eff, lr_mult) in by_variable.items():
                    lines.append(
                        f"{indent}{var}: Cumulative = {cum_eff:.4f}, "
                        f"Long-run = {lr_mult:.4f}"
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


class ADL(RegimesModelBase):
    """Autoregressive Distributed Lag model with structural break support.

    Estimates ADL(p,q) models with support for known structural breaks
    and various covariance estimators including HAC standard errors.

    ADL(p,q) specification:
        y_t = c + Σ(φ_i * y_{t-i}) + Σ(β_j * x_{t-j}) + ε_t
                  i=1 to p              j=0 to q

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    exog : ArrayLike
        Exogenous regressors (n_obs, k). Required.
    lags : int | Sequence[int]
        Autoregressive lags (p). If int, uses lags 1 to p. If sequence,
        uses specific lag indices. Default is 1.
    exog_lags : int | dict[str | int, int | Sequence[int]]
        Distributed lags (q) for exogenous variables:
        - If int: Same lag structure for all exog variables (e.g., 2 -> L0, L1, L2)
        - If dict: Variable-specific lags (e.g., {"x1": 2, "x2": 1})
        Default is 0 (contemporaneous only).
    breaks : Sequence[int] | None
        Known break points (observation indices). All variables share
        these break points.
    variable_breaks : dict[str | int, Sequence[int]] | None
        Variable-specific break points. Cannot be used with `breaks`.
    trend : str
        Trend to include: "c" (constant only, default), "ct" (constant
        and trend), "n" (no deterministic terms).

    Examples
    --------
    >>> import numpy as np
    >>> from regimes import ADL
    >>> np.random.seed(42)
    >>> n = 200
    >>> x = np.random.randn(n)
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = 0.5 * y[t-1] + 0.3 * x[t] + 0.2 * x[t-1] + np.random.randn()
    >>> model = ADL(y, x, lags=1, exog_lags=1)
    >>> results = model.fit(cov_type="HAC")
    >>> print(results.summary())

    Notes
    -----
    The model is estimated using OLS on the effective sample (after
    dropping initial observations for maximum lag length).
    """

    def __init__(
        self,
        endog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        exog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        lags: int | Sequence[int] = 1,
        exog_lags: int | dict[str | int, int | Sequence[int]] = 0,
        breaks: Sequence[int] | None = None,
        variable_breaks: dict[str | int, Sequence[int]] | None = None,
        trend: str = "c",
    ) -> None:
        """Initialize ADL model."""
        # Convert exog to array and validate
        exog_arr = _ensure_array(exog, "exog")
        if exog_arr is None:
            raise ValueError("exog is required for ADL models")

        super().__init__(endog, exog_arr, breaks)

        # the names of a pandas object (the array above has lost them)
        columns = getattr(exog, "columns", None)
        if columns is not None:
            self._exog_names = [str(c) for c in columns]
        elif hasattr(exog, "name") and hasattr(exog, "index"):
            self._exog_names = [str(exog.name) if exog.name is not None else "x0"]

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

        # Process AR lags
        if isinstance(lags, int):
            if lags < 0:
                raise ValueError("lags must be non-negative")
            self.lags: list[int] = list(range(1, lags + 1)) if lags > 0 else []
        else:
            self.lags = sorted(list(lags))
            if any(lag < 1 for lag in self.lags):
                raise ValueError("All AR lags must be positive integers")

        # Process exog lags
        self._exog_lags_raw = exog_lags
        self._exog_lags: dict[
            int, list[int]
        ] = {}  # Will be populated in _build_design_matrix

        # Ensure exog is 2D
        if self.exog is not None and self.exog.ndim == 1:
            self.exog = self.exog.reshape(-1, 1)

    @property
    def maxlag(self) -> int:
        """Maximum lag in the model (across AR and exog lags)."""
        ar_max = max(self.lags) if self.lags else 0

        # Get max exog lag
        exog_max = 0
        if isinstance(self._exog_lags_raw, int):
            exog_max = self._exog_lags_raw
        elif isinstance(self._exog_lags_raw, dict):
            for val in self._exog_lags_raw.values():
                if isinstance(val, int):
                    exog_max = max(exog_max, val)
                else:
                    exog_max = max(exog_max, max(val) if val else 0)

        return max(ar_max, exog_max)

    @property
    def nobs_effective(self) -> int:
        """Effective number of observations (after losing lags)."""
        return self.nobs - self.maxlag

    def _process_exog_lags(self) -> dict[int, list[int]]:
        """Process exog_lags into normalized form.

        Returns
        -------
        dict[int, list[int]]
            Dictionary mapping column index to list of lag indices.
        """
        if self.exog is None:
            return {}

        k_exog = self.exog.shape[1]
        result: dict[int, list[int]] = {}

        if isinstance(self._exog_lags_raw, int):
            # Same lag structure for all variables
            lags_list = list(range(0, self._exog_lags_raw + 1))
            for i in range(k_exog):
                result[i] = lags_list
        elif isinstance(self._exog_lags_raw, dict):
            # Variable-specific lags
            for key, val in self._exog_lags_raw.items():
                # Convert key to index
                if isinstance(key, int):
                    idx = key
                else:
                    # String key - find in exog names
                    if key in self._exog_names:
                        idx = self._exog_names.index(key)
                    else:
                        raise ValueError(
                            f"Exogenous variable name '{key}' not found. "
                            f"Available names: {self._exog_names}"
                        )

                if idx < 0 or idx >= k_exog:
                    raise ValueError(
                        f"Exogenous variable index {idx} is out of bounds for "
                        f"exog with {k_exog} columns"
                    )

                # Convert val to list of lags
                if isinstance(val, int):
                    result[idx] = list(range(0, val + 1))
                else:
                    result[idx] = sorted(list(val))

            # Fill in missing variables with contemporaneous only
            for i in range(k_exog):
                if i not in result:
                    result[i] = [0]
        else:
            # Default: contemporaneous only for all
            for i in range(k_exog):
                result[i] = [0]

        return result

    def _create_lag_matrix(
        self, data: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Create a matrix of lagged values for AR terms.

        Parameters
        ----------
        data : NDArray[np.floating]
            1D array of data to create lags from.

        Returns
        -------
        NDArray[np.floating]
            Matrix of shape (nobs_effective, n_lags) with lagged values.
        """
        n = len(data)
        n_eff = n - self.maxlag

        if not self.lags:
            return np.zeros((n_eff, 0))

        lag_matrix = np.zeros((n_eff, len(self.lags)))
        for i, lag in enumerate(self.lags):
            lag_matrix[:, i] = data[self.maxlag - lag : n - lag]

        return lag_matrix

    def _create_exog_lag_matrix(
        self, exog_lags_dict: dict[int, list[int]]
    ) -> tuple[NDArray[np.floating[Any]], list[str]]:
        """Create matrix of current and lagged exogenous variables.

        Parameters
        ----------
        exog_lags_dict : dict[int, list[int]]
            Dictionary mapping column index to list of lag indices.

        Returns
        -------
        tuple[NDArray[np.floating], list[str]]
            Matrix of lagged exog values and parameter names.
        """
        if self.exog is None:
            return np.zeros((self.nobs_effective, 0)), []

        n = self.nobs
        n_eff = n - self.maxlag
        columns = []
        param_names = []

        for col_idx in sorted(exog_lags_dict.keys()):
            lag_list = exog_lags_dict[col_idx]
            var_name = (
                self._exog_names[col_idx]
                if col_idx < len(self._exog_names)
                else f"x{col_idx}"
            )

            for lag in lag_list:
                if lag == 0:
                    # Contemporaneous
                    col_data = self.exog[self.maxlag :, col_idx]
                    param_names.append(var_name)
                else:
                    # Lagged
                    col_data = self.exog[self.maxlag - lag : n - lag, col_idx]
                    param_names.append(f"{var_name}.L{lag}")

                columns.append(col_data)

        if columns:
            return np.column_stack(columns), param_names
        return np.zeros((n_eff, 0)), []

    def _build_design_matrix(
        self,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], list[str]]:
        """Build the design matrix for ADL estimation.

        Returns
        -------
        tuple[NDArray[np.floating], NDArray[np.floating], list[str]]
            y (dependent), X (design matrix), and parameter names.
        """
        # Process exog lags
        exog_lags_dict = self._process_exog_lags()
        self._exog_lags = exog_lags_dict

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

        # Lagged dependent variable (AR terms)
        if self.lags:
            lag_matrix = self._create_lag_matrix(self.endog)
            components.append(lag_matrix)
            param_names.extend([f"y.L{lag}" for lag in self.lags])

        # Current and lagged exogenous variables
        exog_matrix, exog_names = self._create_exog_lag_matrix(exog_lags_dict)
        if exog_matrix.shape[1] > 0:
            components.append(exog_matrix)
            param_names.extend(exog_names)

        X = np.column_stack(components) if components else np.ones((n_eff, 1))
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
        """
        if self._variable_breaks is None:
            return None

        k = len(param_names)
        normalized: dict[int, list[int]] = {}

        for key, breaks in self._variable_breaks.items():
            # Convert key to index
            if isinstance(key, bool):
                raise TypeError(
                    "variable_breaks keys are variable names or column "
                    f"numbers, got {key!r}"
                )
            if isinstance(key, (int, np.integer)):
                if key < 0 or key >= k:
                    raise ValueError(
                        f"Variable index {key} is out of bounds for design "
                        f"matrix with {k} columns"
                    )
                idx = int(key)
            else:
                if key not in param_names:
                    raise ValueError(
                        f"Variable name '{key}' not found. "
                        f"Available names: {param_names}"
                    )
                idx = param_names.index(key)

            # sorted, distinct integers inside the sample (original
            # observation numbers)
            breaks_list = normalize_breaks(
                breaks, self.nobs, f"Break point for variable '{key}'"
            )

            # Adjust for effective sample (subtract maxlag)
            for bp in breaks_list:
                if bp <= self.maxlag:
                    raise ValueError(
                        f"Break point {bp} for variable '{key}' leaves no "
                        f"observations before it: the first {self.maxlag} "
                        "observations are used as initial values for the lags"
                    )
            adjusted_breaks = [bp - self.maxlag for bp in breaks_list]
            if adjusted_breaks:
                normalized[idx] = adjusted_breaks

        return normalized if normalized else None

    def _get_variable_regime_indices(
        self, breaks: list[int], n_eff: int
    ) -> list[tuple[int, int]]:
        """Get regime start/end indices for a variable's break points."""
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
        """Create design matrix with variable-specific break points."""
        normalized_breaks = self._normalize_variable_breaks(param_names)
        if normalized_breaks is None:
            return X, param_names

        n_eff = len(y)
        k = X.shape[1]

        columns = []
        new_param_names = []

        for var_idx in range(k):
            var_name = param_names[var_idx]
            var_data = X[:, var_idx]

            if var_idx in normalized_breaks:
                breaks = normalized_breaks[var_idx]
                regime_indices = self._get_variable_regime_indices(breaks, n_eff)

                for r, (start, end) in enumerate(regime_indices):
                    col = np.zeros(n_eff)
                    col[start:end] = var_data[start:end]
                    columns.append(col)
                    new_param_names.append(f"{var_name}_regime{r + 1}")
            else:
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
        """Create design matrix with regime-specific coefficients."""
        # Handle variable-specific breaks first
        if self._variable_breaks is not None:
            return self._create_variable_break_design(y, X, param_names)

        # Handle common breaks
        if not self.breaks:
            return X, param_names

        # Adjust break indices for effective sample
        for b in self.breaks:
            if b <= self.maxlag:
                raise ValueError(
                    f"Break index {b} leaves no observations before it: the "
                    f"first {self.maxlag} observations are used as initial "
                    "values for the lags"
                )
        adjusted_breaks = [b - self.maxlag for b in self.breaks]

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
    ) -> ADLResults:
        """Fit the ADL model.

        Parameters
        ----------
        cov_type : CovType
            Type of covariance estimator:
            - "nonrobust": Standard OLS covariance
            - "HC0", "HC1", "HC2", "HC3": Heteroskedasticity-robust
            - "HAC": Heteroskedasticity and autocorrelation consistent
        cov_kwds : dict | None
            Additional keywords for covariance estimation, used with
            ``cov_type="HAC"`` only: ``maxlags`` (default the fixed rule
            ``floor(4 (T / 100)^(2/9))`` on the effective sample) and other
            keywords of statsmodels' HAC covariance. See ``OLS.fit`` for the
            conventions (pooled error variance with breaks, t distribution).
        **kwargs
            Additional arguments (reserved for future use).

        Returns
        -------
        ADLResults
            Results object containing estimates and inference.
        """
        cov_kwds = cov_kwds or {}

        # Build design matrix
        y, X, param_names = self._build_design_matrix()

        # Handle breaks
        X, param_names = self._create_break_design(y, X, param_names)
        if self.breaks or self._variable_breaks:
            check_break_design(X, param_names)

        # Use statsmodels for estimation
        sm_model = sm.OLS(y, X)
        sm_results = fit_with_covariance(sm_model, cov_type, cov_kwds, len(y))

        # Extract results
        params = np.asarray(sm_results.params)
        bse = np.asarray(sm_results.bse)
        resid = np.asarray(sm_results.resid)
        fittedvalues = np.asarray(sm_results.fittedvalues)
        cov_params = np.asarray(sm_results.cov_params())

        # Extract AR parameters
        ar_params = self._extract_ar_params(params, param_names)

        # Extract distributed lag parameters
        dl_params = self._extract_dl_params(params, param_names)

        # Build exog_lags dict with variable names for results
        exog_lags_result: dict[str, list[int]] = {}
        for col_idx, lag_list in self._exog_lags.items():
            var_name = (
                self._exog_names[col_idx]
                if col_idx < len(self._exog_names)
                else f"x{col_idx}"
            )
            exog_lags_result[var_name] = lag_list

        # Compute TSS for R-squared
        y_mean = np.mean(y)
        tss = float(np.sum((y - y_mean) ** 2))

        return ADLResults(
            params=params,
            bse=bse,
            resid=resid,
            fittedvalues=fittedvalues,
            cov_params_matrix=cov_params,
            nobs=len(y),
            cov_type=cov_type,
            param_names=param_names,
            model_name="ADL",
            lags=self.lags,
            exog_lags=exog_lags_result,
            ar_params=ar_params,
            dl_params=dl_params,
            llf=float(sm_results.llf),
            scale=float(sm_results.scale),
            _tss=tss,
            _exog=X,
            _breaks=list(self.breaks) if self.breaks else None,
            _variable_breaks=self._variable_breaks,
            _nobs_original=self.nobs,
        )

    def _extract_ar_params(
        self, params: NDArray[np.floating[Any]], param_names: list[str]
    ) -> NDArray[np.floating[Any]]:
        """Extract AR coefficients from full parameter vector."""
        ar_indices = [
            i
            for i, name in enumerate(param_names)
            if name.startswith("y.L") and ("regime1" in name or "regime" not in name)
        ]
        return params[ar_indices] if ar_indices else np.array([])

    def _extract_dl_params(
        self, params: NDArray[np.floating[Any]], param_names: list[str]
    ) -> dict[str, NDArray[np.floating[Any]]]:
        """Extract distributed lag coefficients by variable."""
        if self.exog is None:
            return {}

        dl_params: dict[str, NDArray[np.floating[Any]]] = {}

        for col_idx, lag_list in self._exog_lags.items():
            var_name = (
                self._exog_names[col_idx]
                if col_idx < len(self._exog_names)
                else f"x{col_idx}"
            )

            # Find indices for this variable's lags (from first regime if breaks)
            coefs = []
            for lag in lag_list:
                if lag == 0:
                    target_name = var_name
                else:
                    target_name = f"{var_name}.L{lag}"

                # Try with and without regime suffix
                for suffix in ["", "_regime1"]:
                    full_name = target_name + suffix
                    if full_name in param_names:
                        idx = param_names.index(full_name)
                        coefs.append(params[idx])
                        break

            if coefs:
                dl_params[var_name] = np.array(coefs)

        return dl_params

    def select_lags(
        self,
        max_ar_lags: int = 4,
        max_exog_lags: int = 4,
        criterion: Literal["aic", "bic"] = "bic",
    ) -> tuple[int, int]:
        """Select optimal lag structure via information criteria grid search.

        Every candidate ``(p, q)`` with ``p <= max_ar_lags`` autoregressive
        lags and ``q <= max_exog_lags`` lags of each regressor is estimated
        on the same sample, the one that remains after dropping
        ``max(max_ar_lags, max_exog_lags)`` initial observations. Criteria
        from different samples are not comparable, and a selection based on
        them changes with the units of the data.

        The candidates have the trend of the model and no breaks.

        Parameters
        ----------
        max_ar_lags : int, default 4
            Maximum AR lags to consider.
        max_exog_lags : int, default 4
            Maximum exog lags to consider.
        criterion : "aic" | "bic", default "bic"
            Information criterion to minimize.

        Returns
        -------
        tuple[int, int]
            Optimal (ar_lags, exog_lags) specification. Ties go to the
            smaller model.

        Raises
        ------
        ValueError
            For an unknown criterion or a negative maximum; if the common
            sample does not have more observations than the largest
            candidate has coefficients; if a candidate's criterion is not
            finite.

        Notes
        -----
        Every regressor gets the same number of lags, ``0, ..., q``. The
        values of the criteria are those of statsmodels' ``ARDL`` on the
        same sample; ``ardl_select_order`` searches a larger set (a lag
        length for each regressor, including leaving a regressor out), so
        its choice is the same only when the best model of that larger set
        is in this one.

        Examples
        --------
        >>> model = ADL(y, x, lags=1, exog_lags=0)
        >>> optimal_p, optimal_q = model.select_lags(max_ar_lags=4, max_exog_lags=4)
        >>> print(f"Optimal ADL({optimal_p},{optimal_q})")
        """
        if criterion not in ("aic", "bic"):
            raise ValueError(f"criterion must be 'aic' or 'bic', got {criterion!r}")
        if max_ar_lags < 0 or max_exog_lags < 0:
            raise ValueError("max_ar_lags and max_exog_lags must be non-negative")
        hold_back = max(max_ar_lags, max_exog_lags)
        if self.exog is None:
            raise ValueError("exog is required for ADL models")

        # the largest candidate must be estimable on the common sample
        n_common = self.nobs - hold_back
        k_trend = {"n": 0, "c": 1, "ct": 2}[self.trend]
        k_largest = k_trend + max_ar_lags + self.exog.shape[1] * (max_exog_lags + 1)
        if n_common <= k_largest:
            raise ValueError(
                f"the sample is too short to compare lag lengths up to "
                f"({max_ar_lags}, {max_exog_lags}): after dropping "
                f"{hold_back} initial observations {n_common} remain, for a "
                f"largest model with {k_largest} coefficients; lower "
                "max_ar_lags or max_exog_lags"
            )

        best_ic = np.inf
        best_p = 0
        best_q = 0

        for p in range(0, max_ar_lags + 1):
            for q in range(0, max_exog_lags + 1):
                # Same effective sample for every candidate: a model with
                # fewer lags starts later in the data
                drop = hold_back - max(p, q)
                temp_results = ADL(
                    endog=self.endog[drop:],
                    exog=self.exog[drop:],
                    lags=p if p > 0 else [],
                    exog_lags=q,
                    trend=self.trend,
                ).fit()
                ic = temp_results.aic if criterion == "aic" else temp_results.bic
                if not np.isfinite(ic):
                    raise ValueError(
                        f"the {criterion.upper()} of the ADL({p},{q}) candidate "
                        "is not finite (an exact fit, or collinear regressors)"
                    )
                if ic < best_ic:
                    best_ic = ic
                    best_p = p
                    best_q = q

        return best_p, best_q

    def bai_perron(
        self,
        break_vars: Literal["all", "const"] = "all",
        max_breaks: int = 5,
        trimming: float = 0.15,
        selection: Literal["bic", "lwz", "sequential"] = "bic",
        **kwargs: Any,
    ) -> BaiPerronResults:
        """Test for structural breaks using Bai-Perron procedure.

        Convenience method that creates a BaiPerronTest from this ADL model
        and runs it. The test uses the effective sample (after dropping
        initial observations for lags).

        Parameters
        ----------
        break_vars : "all" | "const"
            Which variables can have breaks:
            - "all": All regressors can break (default)
            - "const": Only intercept can break (mean-shift model)
        max_breaks : int
            Maximum number of breaks to test. Default is 5.
        trimming : float
            Minimum regime length as a fraction of the sample
            (``h = floor(trimming * T)``, at least ``q + 1``). Default is 0.15.
        selection : str
            Break selection criterion: "bic", "lwz", or "sequential".

        **kwargs
            Further options passed to :meth:`BaiPerronTest.fit`: ``level``
            (significance level of the tests and the sequential procedure),
            ``robust``, ``prewhiten``, ``hetdat``, ``hetvar`` (assumptions on
            the errors in the tests), ``hetomega``, ``hetq`` (in the
            break-date confidence intervals), ``repartition`` and ``h``.

        Returns
        -------
        BaiPerronResults
            Test results.
        """
        from regimes.tests.bai_perron import BaiPerronTest

        test = BaiPerronTest.from_model(self, break_vars=break_vars)
        return test.fit(
            max_breaks=max_breaks, trimming=trimming, selection=selection, **kwargs
        )

    def chow_test(
        self,
        break_points: int | Sequence[int],
        break_vars: Literal["all", "const"] = "all",
        significance: float = 0.05,
    ) -> ChowTestResults:
        """Test for structural breaks at known break points using the Chow test.

        Convenience method that creates a ChowTest from this ADL model
        and runs it. The test uses the effective sample (after dropping
        initial observations for lags).

        Parameters
        ----------
        break_points : int | Sequence[int]
            One or more break point indices to test (in effective sample
            coordinates, after lag trimming).
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

        Convenience method that creates a CUSUMTest from this ADL model
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

        Convenience method that creates a CUSUMSQTest from this ADL model
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
        ADL model and runs it. The test uses the effective sample (after
        dropping initial observations for lags).

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

    def rolling(self, window: int) -> RollingADL:
        """Create a rolling ADL estimator from this model.

        Parameters
        ----------
        window : int
            Window size for rolling estimation.

        Returns
        -------
        RollingADL
            Rolling ADL estimator ready to be fitted.
        """
        from regimes.rolling.adl import RollingADL

        return RollingADL.from_model(self, window=window)

    def recursive(self, min_nobs: int | None = None) -> RecursiveADL:
        """Create a recursive (expanding window) ADL estimator from this model.

        Parameters
        ----------
        min_nobs : int | None
            Minimum number of observations to start estimation.

        Returns
        -------
        RecursiveADL
            Recursive ADL estimator ready to be fitted.
        """
        from regimes.rolling.adl import RecursiveADL

        return RecursiveADL.from_model(self, min_nobs=min_nobs)

    def markov_switching(
        self,
        k_regimes: int = 2,
        **kwargs: Any,
    ) -> MarkovADLResults:
        """Fit a Markov regime-switching version of this ADL model.

        Creates a MarkovADL from this model's specification and fits it.

        Parameters
        ----------
        k_regimes : int
            Number of regimes. Default is 2.
        **kwargs
            Additional keyword arguments. Model-level kwargs forwarded
            to MarkovADL; fit-level kwargs forwarded to fit().

        Returns
        -------
        MarkovADLResults
            Fitted Markov switching ADL results.

        See Also
        --------
        regimes.markov.MarkovADL : The underlying MS ADL model class.
        """
        from regimes.markov import MarkovADL

        fit_kwargs_names = {"method", "maxiter", "em_iter", "search_reps"}
        model_kwargs = {k: v for k, v in kwargs.items() if k not in fit_kwargs_names}
        fit_kwargs = {k: v for k, v in kwargs.items() if k in fit_kwargs_names}

        ms_model = MarkovADL.from_model(self, k_regimes=k_regimes, **model_kwargs)
        return ms_model.fit(**fit_kwargs)


def _results_maxlag(results: ADLResults) -> int:
    """Initial observations an ADL fit loses: the longest lag of ``y`` or ``x``."""
    lags = list(results.lags)
    for lag_list in results.exog_lags.values():
        lags.extend(lag_list)
    return max(lags) if lags else 0


def adl_summary_by_regime(
    results_list: list[ADLResults],
    breaks: Sequence[int] | None = None,
    nobs_total: int | None = None,
    diagnostics: bool = True,
    index: IndexLike | None = None,
) -> str:
    """Generate combined summary for regime-specific ADL results.

    Creates a formatted summary combining multiple ADL results from separate
    regime estimations.

    Parameters
    ----------
    results_list : list[ADLResults]
        Results from ADL estimation per regime.
    breaks : Sequence[int] | None
        Break points for labeling regime boundaries.
    nobs_total : int | None
        Total observations across all regimes.
    diagnostics : bool
        If True (default), include misspecification tests.
    index : IndexLike or None
        Optional time index (e.g. ``pd.PeriodIndex``). When provided,
        observation numbers are replaced with date labels throughout.

    Returns
    -------
    str
        Combined summary string with all regimes.
    """
    if not results_list:
        return "No results to summarize."

    lines = []
    lines.append("=" * 81)
    lines.append(f"{'ADL Model Results by Regime':^81}")
    lines.append("=" * 81)

    if breaks:
        break_str = ", ".join(_obs_label(bp, index) for bp in breaks)
        lines.append(f"Breaks at: {break_str}")
        lines.append("")

    # Compute nobs_total if not provided
    if nobs_total is None:
        maxlag = _results_maxlag(results_list[0])
        nobs_total = sum(r.nobs + maxlag for r in results_list)

    # Compute regime boundaries
    if breaks:
        boundaries = [0] + list(breaks) + [nobs_total]
    else:
        maxlag = _results_maxlag(results_list[0])
        boundaries = [0]
        for r in results_list:
            boundaries.append(boundaries[-1] + r.nobs + maxlag)

    for i, result in enumerate(results_list):
        start = boundaries[i]
        end = boundaries[i + 1] - 1
        range_str = _obs_range_label(start, end, index)

        lines.append("-" * 81)
        lines.append(f"{'Regime ' + str(i + 1) + f' ({range_str})':^81}")
        lines.append("-" * 81)

        p = max(result.lags) if result.lags else 0
        if result.exog_lags:
            q_vals = [max(lags) if lags else 0 for lags in result.exog_lags.values()]
            q = max(q_vals) if q_vals else 0
        else:
            q = 0
        model_str = f"ADL({p},{q})"

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
                lines.append("AR roots outside unit circle (stationary).")
            else:
                lines.append(
                    "WARNING: Some AR roots inside unit circle (non-stationary)."
                )
            lines.append("")

        # Parameter table
        lines.append(
            f"{'':>15} {'coef':>10} {'std err':>10} {'t':>10} "
            f"{'P>|t|':>10} {'[0.025':>10} {'0.975]':>10}"
        )
        lines.append("-" * 81)

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

        # Distributed lag effects
        if result.dl_params:
            lines.append("Distributed Lag Effects:")
            for var, cum_eff in result.cumulative_effect.items():
                lr_mult = result.long_run_multiplier.get(var, np.nan)
                lines.append(
                    f"  {var}: Cumulative = {cum_eff:.4f}, Long-run = {lr_mult:.4f}"
                )
            lines.append("")

        if result.cov_type == "HAC":
            lines.append("Note: Standard errors are HAC (Newey-West) robust.")
        elif result.cov_type.startswith("HC"):
            lines.append(
                f"Note: Standard errors are {result.cov_type} heteroskedasticity-robust."
            )

        if diagnostics and result._exog is not None:
            try:
                diag = result.diagnostics()
                lines.append("")
                lines.append(diag.summary())
            except Exception:
                pass

        lines.append("")

    lines.append("=" * 81)
    return "\n".join(lines)
