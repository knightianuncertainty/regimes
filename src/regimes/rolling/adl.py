"""Rolling and recursive ADL estimation.

This module provides RollingADL and RecursiveADL classes for estimating
ADL models with rolling (fixed) or recursive (expanding) windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import statsmodels.api as sm

from regimes.rolling.base import (
    RollingCovType,
    RollingEstimatorBase,
    RollingResultsBase,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray

    from regimes.models.adl import ADL


@dataclass(kw_only=True)
class RollingADLResults(RollingResultsBase):
    """Results from rolling ADL estimation.

    Extends RollingResultsBase with ADL-specific attributes and methods.

    Additional Attributes
    ---------------------
    lags : list[int]
        AR lag indices used in the model.
    exog_lags : dict[str, list[int]]
        Distributed lag structure per exogenous variable.
    ssr : NDArray[np.floating]
        Sum of squared residuals at each time point.
    rsquared : NDArray[np.floating]
        R-squared at each time point.
    """

    lags: list[int] = field(default_factory=list)
    exog_lags: dict[str, list[int]] = field(default_factory=dict)
    ssr: NDArray[np.floating[Any]] | None = None
    rsquared: NDArray[np.floating[Any]] | None = None

    @property
    def ar_params(self) -> NDArray[np.floating[Any]]:
        """Extract AR coefficients only (excluding constant/exog).

        Returns
        -------
        NDArray[np.floating]
            AR coefficients, shape (nobs, n_lags).
        """
        ar_indices = [
            i for i, name in enumerate(self.param_names) if name.startswith("y.L")
        ]
        if not ar_indices:
            return np.full((self.nobs, 0), np.nan)
        return self.params[:, ar_indices]

    def plot_coefficients(
        self,
        variables: Sequence[str] | None = None,
        alpha: float = 0.05,
        figsize: tuple[float, float] = (10, 8),
        ncols: int = 1,
    ) -> tuple[Figure, Axes | NDArray[Any]]:
        """Plot coefficient estimates over time with confidence bands.

        Parameters
        ----------
        variables : Sequence[str] | None
            Which variables to plot. If None, plots all.
        alpha : float
            Significance level for confidence bands (default 0.05 = 95% CI).
        figsize : tuple[float, float]
            Figure size (width, height) in inches.
        ncols : int
            Number of columns in subplot grid.

        Returns
        -------
        tuple[Figure, Axes | NDArray[Axes]]
            Matplotlib figure and axes.
        """
        from regimes.visualization.rolling import plot_rolling_coefficients

        return plot_rolling_coefficients(
            self,
            variables=variables,
            alpha=alpha,
            figsize=figsize,
            ncols=ncols,
        )

    def summary(self) -> str:
        """Generate a text summary of rolling ADL results.

        Returns
        -------
        str
            Formatted summary of the rolling ADL estimation.
        """
        lines = []
        lines.append("=" * 70)
        est_type = "Rolling" if self.is_rolling else "Recursive"

        p = max(self.lags) if self.lags else 0
        if self.exog_lags:
            q_vals = [max(lags) if lags else 0 for lags in self.exog_lags.values()]
            q = max(q_vals) if q_vals else 0
        else:
            q = 0

        lines.append(f"{est_type} ADL({p},{q}) Results".center(70))
        lines.append("=" * 70)

        lines.append(f"No. Observations:        {self.nobs:>10}")
        lines.append(f"No. Parameters:          {self.n_params:>10}")
        lines.append(f"AR Lags:                 {self.lags}")

        if self.exog_lags:
            for var, lags_list in self.exog_lags.items():
                lines.append(f"  {var} Lags:            {lags_list}")

        if self.is_rolling:
            lines.append(f"Window Size:             {self.window:>10}")
        else:
            lines.append(f"Min Observations:        {self.min_nobs:>10}")

        lines.append(f"Valid Estimates:         {self.n_valid:>10}")
        lines.append(f"Cov. Type:               {self.cov_type:>10}")

        lines.append("-" * 70)
        lines.append("Parameter Summary (mean and std of estimates over time):")
        lines.append("-" * 70)

        lines.append(f"{'':>15} {'mean':>12} {'std':>12} {'min':>12} {'max':>12}")
        lines.append("-" * 70)

        for i, name in enumerate(self.param_names):
            valid_params = self.params[~np.isnan(self.params[:, i]), i]
            if len(valid_params) > 0:
                lines.append(
                    f"{name:>15} {np.mean(valid_params):>12.4f} "
                    f"{np.std(valid_params):>12.4f} "
                    f"{np.min(valid_params):>12.4f} "
                    f"{np.max(valid_params):>12.4f}"
                )
            else:
                lines.append(
                    f"{name:>15} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12}"
                )

        lines.append("=" * 70)
        return "\n".join(lines)


@dataclass(kw_only=True)
class RecursiveADLResults(RollingADLResults):
    """Results from recursive (expanding window) ADL estimation.

    Inherits from RollingADLResults with recursive-specific defaults.
    """

    pass


class RollingADL(RollingEstimatorBase):
    """Rolling ADL regression with fixed window size.

    Estimates ADL models using a rolling (moving) window of fixed size.
    At each time point t, the model is estimated using observations
    from t - window + 1 to t.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    exog : ArrayLike
        Exogenous regressors (n_obs, k).
    lags : int | Sequence[int]
        AR lags (if int, uses lags 1 to p).
    exog_lags : int | dict
        Distributed lags for exogenous variables.
    window : int
        Window size for rolling estimation.
    trend : str
        Trend to include: "c", "ct", or "n".

    Examples
    --------
    >>> import numpy as np
    >>> from regimes.rolling import RollingADL
    >>> np.random.seed(42)
    >>> n = 200
    >>> x = np.random.randn(n)
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = 0.5 * y[t-1] + 0.3 * x[t] + np.random.randn()
    >>> rolling = RollingADL(y, x, lags=1, exog_lags=0, window=60)
    >>> results = rolling.fit()
    >>> print(results.summary())

    See Also
    --------
    RecursiveADL : Expanding window estimation.
    RollingADL.from_model : Create from ADL model.
    """

    def __init__(
        self,
        endog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        exog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        lags: int | Sequence[int],
        exog_lags: int | dict[str | int, int | Sequence[int]],
        window: int,
        trend: str = "c",
    ) -> None:
        """Initialize RollingADL estimator."""
        # Store original data
        self._endog_full = np.asarray(endog, dtype=np.float64)
        self._exog_full = np.asarray(exog, dtype=np.float64)
        if self._exog_full.ndim == 1:
            self._exog_full = self._exog_full.reshape(-1, 1)

        # Process lags
        if isinstance(lags, int):
            self.lags: list[int] = list(range(1, lags + 1)) if lags > 0 else []
        else:
            self.lags = sorted(list(lags))

        if self.lags and any(lag < 1 for lag in self.lags):
            raise ValueError("All AR lags must be positive integers")

        # Store exog_lags for later processing
        self._exog_lags_raw = exog_lags
        self._k_exog = self._exog_full.shape[1]

        # Generate exog names
        self._exog_names = [f"x{i}" for i in range(self._k_exog)]

        # Compute maxlag
        ar_max = max(self.lags) if self.lags else 0
        exog_max = 0
        if isinstance(exog_lags, int):
            exog_max = exog_lags
        elif isinstance(exog_lags, dict):
            for val in exog_lags.values():
                if isinstance(val, int):
                    exog_max = max(exog_max, val)
                else:
                    exog_max = max(exog_max, max(val) if val else 0)

        self.maxlag = max(ar_max, exog_max)
        self.trend = trend

        if trend not in ("c", "ct", "n"):
            raise ValueError(f"trend must be 'c', 'ct', or 'n', got {trend}")

        # Build design matrix for the full sample
        y_eff, X_full, param_names, exog_lags_dict = self._build_design_matrix()

        # Store exog_lags dict for results
        self._exog_lags_result: dict[str, list[int]] = {}
        for col_idx, lag_list in exog_lags_dict.items():
            var_name = (
                self._exog_names[col_idx]
                if col_idx < len(self._exog_names)
                else f"x{col_idx}"
            )
            self._exog_lags_result[var_name] = lag_list

        # Initialize base class with effective sample
        super().__init__(
            endog=y_eff,
            exog=X_full,
            window=window,
            min_nobs=window,
            param_names=param_names,
        )

    def _process_exog_lags(self) -> dict[int, list[int]]:
        """Process exog_lags into normalized form."""
        result: dict[int, list[int]] = {}

        if isinstance(self._exog_lags_raw, int):
            lags_list = list(range(0, self._exog_lags_raw + 1))
            for i in range(self._k_exog):
                result[i] = lags_list
        elif isinstance(self._exog_lags_raw, dict):
            for key, val in self._exog_lags_raw.items():
                if isinstance(key, int):
                    idx = key
                else:
                    if key in self._exog_names:
                        idx = self._exog_names.index(key)
                    else:
                        raise ValueError(f"Exogenous variable name '{key}' not found")

                if isinstance(val, int):
                    result[idx] = list(range(0, val + 1))
                else:
                    result[idx] = sorted(list(val))

            for i in range(self._k_exog):
                if i not in result:
                    result[i] = [0]
        else:
            for i in range(self._k_exog):
                result[i] = [0]

        return result

    def _create_ar_lag_matrix(
        self, data: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Create matrix of AR lagged values."""
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
        """Create matrix of current and lagged exogenous variables."""
        n = len(self._endog_full)
        n_eff = n - self.maxlag
        columns = []
        param_names = []

        for col_idx in sorted(exog_lags_dict.keys()):
            lag_list = exog_lags_dict[col_idx]
            var_name = self._exog_names[col_idx]

            for lag in lag_list:
                if lag == 0:
                    col_data = self._exog_full[self.maxlag :, col_idx]
                    param_names.append(var_name)
                else:
                    col_data = self._exog_full[self.maxlag - lag : n - lag, col_idx]
                    param_names.append(f"{var_name}.L{lag}")

                columns.append(col_data)

        if columns:
            return np.column_stack(columns), param_names
        return np.zeros((n_eff, 0)), []

    def _build_design_matrix(
        self,
    ) -> tuple[
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        list[str],
        dict[int, list[int]],
    ]:
        """Build the design matrix for ADL estimation.

        Returns
        -------
        tuple
            y (dependent), X (design matrix), parameter names, and exog_lags dict.
        """
        exog_lags_dict = self._process_exog_lags()

        y = self._endog_full[self.maxlag :]
        n_eff = len(y)

        components = []
        param_names: list[str] = []

        # Deterministic terms
        if self.trend in ("c", "ct"):
            components.append(np.ones((n_eff, 1)))
            param_names.append("const")

        if self.trend == "ct":
            trend_var = np.arange(self.maxlag + 1, len(self._endog_full) + 1).reshape(
                -1, 1
            )
            components.append(trend_var)
            param_names.append("trend")

        # AR lags
        if self.lags:
            lag_matrix = self._create_ar_lag_matrix(self._endog_full)
            components.append(lag_matrix)
            param_names.extend([f"y.L{lag}" for lag in self.lags])

        # Exog lags
        exog_matrix, exog_names = self._create_exog_lag_matrix(exog_lags_dict)
        if exog_matrix.shape[1] > 0:
            components.append(exog_matrix)
            param_names.extend(exog_names)

        X = np.column_stack(components) if components else np.ones((n_eff, 1))
        return y, X, param_names, exog_lags_dict

    @classmethod
    def from_model(cls, model: ADL, window: int) -> RollingADL:
        """Create RollingADL estimator from an existing ADL model.

        Parameters
        ----------
        model : ADL
            An ADL model instance.
        window : int
            Window size for rolling estimation.

        Returns
        -------
        RollingADL
            Rolling ADL estimator initialized with the model's data.

        Examples
        --------
        >>> from regimes import ADL
        >>> model = ADL(y, x, lags=1, exog_lags=1)
        >>> rolling = RollingADL.from_model(model, window=60)
        >>> results = rolling.fit()
        """
        return cls(
            endog=model.endog,
            exog=model.exog,
            lags=model.lags,
            exog_lags=model._exog_lags_raw,
            window=window,
            trend=model.trend,
        )

    def fit(self, cov_type: RollingCovType = "nonrobust") -> RollingADLResults:
        """Fit rolling ADL model.

        Parameters
        ----------
        cov_type : RollingCovType
            Type of covariance estimator:
            - "nonrobust": Standard OLS covariance
            - "HC0": Heteroskedasticity-robust (White)

        Returns
        -------
        RollingADLResults
            Results object with rolling estimates.
        """
        n = self.nobs
        k = self.n_params
        window = self.window

        if window is None:
            raise ValueError("window must be specified for RollingADL")

        # Initialize output arrays
        params = np.full((n, k), np.nan)
        bse = np.full((n, k), np.nan)
        ssr = np.full(n, np.nan)
        rsquared = np.full(n, np.nan)

        # Rolling estimation
        n_valid = 0
        for t in range(window - 1, n):
            start = t - window + 1
            end = t + 1

            y_win = self.endog[start:end]
            X_win = self.exog[start:end]

            try:
                sm_model = sm.OLS(y_win, X_win)
                if cov_type == "nonrobust":
                    sm_results = sm_model.fit()
                else:
                    sm_results = sm_model.fit(cov_type="HC0")

                params[t] = sm_results.params
                bse[t] = sm_results.bse
                ssr[t] = sm_results.ssr
                rsquared[t] = sm_results.rsquared
                n_valid += 1
            except (np.linalg.LinAlgError, ValueError):
                continue

        return RollingADLResults(
            params=params,
            bse=bse,
            nobs=n,
            window=window,
            min_nobs=window,
            n_valid=n_valid,
            cov_type=cov_type,
            param_names=self.param_names,
            lags=self.lags,
            exog_lags=self._exog_lags_result,
            ssr=ssr,
            rsquared=rsquared,
        )


class RecursiveADL(RollingEstimatorBase):
    """Recursive (expanding window) ADL regression.

    Estimates ADL models using an expanding window starting from
    min_nobs observations. At each time point t, the model is estimated
    using all observations from 0 to t.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    exog : ArrayLike
        Exogenous regressors (n_obs, k).
    lags : int | Sequence[int]
        AR lags (if int, uses lags 1 to p).
    exog_lags : int | dict
        Distributed lags for exogenous variables.
    min_nobs : int | None
        Minimum number of observations to start estimation.
    trend : str
        Trend to include: "c", "ct", or "n".

    Examples
    --------
    >>> import numpy as np
    >>> from regimes.rolling import RecursiveADL
    >>> np.random.seed(42)
    >>> n = 200
    >>> x = np.random.randn(n)
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = 0.5 * y[t-1] + 0.3 * x[t] + np.random.randn()
    >>> recursive = RecursiveADL(y, x, lags=1, exog_lags=0, min_nobs=30)
    >>> results = recursive.fit()
    >>> print(results.summary())

    See Also
    --------
    RollingADL : Fixed window estimation.
    RecursiveADL.from_model : Create from ADL model.
    """

    def __init__(
        self,
        endog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        exog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        lags: int | Sequence[int],
        exog_lags: int | dict[str | int, int | Sequence[int]],
        min_nobs: int | None = None,
        trend: str = "c",
    ) -> None:
        """Initialize RecursiveADL estimator."""
        # Store original data
        self._endog_full = np.asarray(endog, dtype=np.float64)
        self._exog_full = np.asarray(exog, dtype=np.float64)
        if self._exog_full.ndim == 1:
            self._exog_full = self._exog_full.reshape(-1, 1)

        # Process lags
        if isinstance(lags, int):
            self.lags: list[int] = list(range(1, lags + 1)) if lags > 0 else []
        else:
            self.lags = sorted(list(lags))

        if self.lags and any(lag < 1 for lag in self.lags):
            raise ValueError("All AR lags must be positive integers")

        # Store exog_lags for later processing
        self._exog_lags_raw = exog_lags
        self._k_exog = self._exog_full.shape[1]

        # Generate exog names
        self._exog_names = [f"x{i}" for i in range(self._k_exog)]

        # Compute maxlag
        ar_max = max(self.lags) if self.lags else 0
        exog_max = 0
        if isinstance(exog_lags, int):
            exog_max = exog_lags
        elif isinstance(exog_lags, dict):
            for val in exog_lags.values():
                if isinstance(val, int):
                    exog_max = max(exog_max, val)
                else:
                    exog_max = max(exog_max, max(val) if val else 0)

        self.maxlag = max(ar_max, exog_max)
        self.trend = trend

        if trend not in ("c", "ct", "n"):
            raise ValueError(f"trend must be 'c', 'ct', or 'n', got {trend}")

        # Build design matrix for the full sample
        y_eff, X_full, param_names, exog_lags_dict = self._build_design_matrix()

        # Store exog_lags dict for results
        self._exog_lags_result: dict[str, list[int]] = {}
        for col_idx, lag_list in exog_lags_dict.items():
            var_name = (
                self._exog_names[col_idx]
                if col_idx < len(self._exog_names)
                else f"x{col_idx}"
            )
            self._exog_lags_result[var_name] = lag_list

        # Initialize base class with effective sample
        super().__init__(
            endog=y_eff,
            exog=X_full,
            window=None,
            min_nobs=min_nobs,
            param_names=param_names,
        )

    def _process_exog_lags(self) -> dict[int, list[int]]:
        """Process exog_lags into normalized form."""
        result: dict[int, list[int]] = {}

        if isinstance(self._exog_lags_raw, int):
            lags_list = list(range(0, self._exog_lags_raw + 1))
            for i in range(self._k_exog):
                result[i] = lags_list
        elif isinstance(self._exog_lags_raw, dict):
            for key, val in self._exog_lags_raw.items():
                if isinstance(key, int):
                    idx = key
                else:
                    if key in self._exog_names:
                        idx = self._exog_names.index(key)
                    else:
                        raise ValueError(f"Exogenous variable name '{key}' not found")

                if isinstance(val, int):
                    result[idx] = list(range(0, val + 1))
                else:
                    result[idx] = sorted(list(val))

            for i in range(self._k_exog):
                if i not in result:
                    result[i] = [0]
        else:
            for i in range(self._k_exog):
                result[i] = [0]

        return result

    def _create_ar_lag_matrix(
        self, data: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Create matrix of AR lagged values."""
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
        """Create matrix of current and lagged exogenous variables."""
        n = len(self._endog_full)
        n_eff = n - self.maxlag
        columns = []
        param_names = []

        for col_idx in sorted(exog_lags_dict.keys()):
            lag_list = exog_lags_dict[col_idx]
            var_name = self._exog_names[col_idx]

            for lag in lag_list:
                if lag == 0:
                    col_data = self._exog_full[self.maxlag :, col_idx]
                    param_names.append(var_name)
                else:
                    col_data = self._exog_full[self.maxlag - lag : n - lag, col_idx]
                    param_names.append(f"{var_name}.L{lag}")

                columns.append(col_data)

        if columns:
            return np.column_stack(columns), param_names
        return np.zeros((n_eff, 0)), []

    def _build_design_matrix(
        self,
    ) -> tuple[
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        list[str],
        dict[int, list[int]],
    ]:
        """Build the design matrix for ADL estimation.

        Returns
        -------
        tuple
            y (dependent), X (design matrix), parameter names, and exog_lags dict.
        """
        exog_lags_dict = self._process_exog_lags()

        y = self._endog_full[self.maxlag :]
        n_eff = len(y)

        components = []
        param_names: list[str] = []

        # Deterministic terms
        if self.trend in ("c", "ct"):
            components.append(np.ones((n_eff, 1)))
            param_names.append("const")

        if self.trend == "ct":
            trend_var = np.arange(self.maxlag + 1, len(self._endog_full) + 1).reshape(
                -1, 1
            )
            components.append(trend_var)
            param_names.append("trend")

        # AR lags
        if self.lags:
            lag_matrix = self._create_ar_lag_matrix(self._endog_full)
            components.append(lag_matrix)
            param_names.extend([f"y.L{lag}" for lag in self.lags])

        # Exog lags
        exog_matrix, exog_names = self._create_exog_lag_matrix(exog_lags_dict)
        if exog_matrix.shape[1] > 0:
            components.append(exog_matrix)
            param_names.extend(exog_names)

        X = np.column_stack(components) if components else np.ones((n_eff, 1))
        return y, X, param_names, exog_lags_dict

    @classmethod
    def from_model(cls, model: ADL, min_nobs: int | None = None) -> RecursiveADL:
        """Create RecursiveADL estimator from an existing ADL model.

        Parameters
        ----------
        model : ADL
            An ADL model instance.
        min_nobs : int | None
            Minimum observations to start estimation.

        Returns
        -------
        RecursiveADL
            Recursive ADL estimator initialized with the model's data.

        Examples
        --------
        >>> from regimes import ADL
        >>> model = ADL(y, x, lags=1, exog_lags=1)
        >>> recursive = RecursiveADL.from_model(model, min_nobs=30)
        >>> results = recursive.fit()
        """
        return cls(
            endog=model.endog,
            exog=model.exog,
            lags=model.lags,
            exog_lags=model._exog_lags_raw,
            min_nobs=min_nobs,
            trend=model.trend,
        )

    def fit(self, cov_type: RollingCovType = "nonrobust") -> RecursiveADLResults:
        """Fit recursive ADL model.

        Parameters
        ----------
        cov_type : RollingCovType
            Type of covariance estimator:
            - "nonrobust": Standard OLS covariance
            - "HC0": Heteroskedasticity-robust (White)

        Returns
        -------
        RecursiveADLResults
            Results object with recursive estimates.
        """
        n = self.nobs
        k = self.n_params

        # Initialize output arrays
        params = np.full((n, k), np.nan)
        bse = np.full((n, k), np.nan)
        ssr = np.full(n, np.nan)
        rsquared = np.full(n, np.nan)

        # Recursive estimation (expanding window)
        n_valid = 0
        for t in range(self.min_nobs - 1, n):
            end = t + 1

            y_win = self.endog[:end]
            X_win = self.exog[:end]

            try:
                sm_model = sm.OLS(y_win, X_win)
                if cov_type == "nonrobust":
                    sm_results = sm_model.fit()
                else:
                    sm_results = sm_model.fit(cov_type="HC0")

                params[t] = sm_results.params
                bse[t] = sm_results.bse
                ssr[t] = sm_results.ssr
                rsquared[t] = sm_results.rsquared
                n_valid += 1
            except (np.linalg.LinAlgError, ValueError):
                continue

        return RecursiveADLResults(
            params=params,
            bse=bse,
            nobs=n,
            window=None,
            min_nobs=self.min_nobs,
            n_valid=n_valid,
            cov_type=cov_type,
            param_names=self.param_names,
            lags=self.lags,
            exog_lags=self._exog_lags_result,
            ssr=ssr,
            rsquared=rsquared,
        )
