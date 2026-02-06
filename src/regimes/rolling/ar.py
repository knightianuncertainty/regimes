"""Rolling and recursive AR estimation.

This module provides RollingAR and RecursiveAR classes for estimating
AR models with rolling (fixed) or recursive (expanding) windows.
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

    from regimes.models.ar import AR


@dataclass(kw_only=True)
class RollingARResults(RollingResultsBase):
    """Results from rolling AR estimation.

    Extends RollingResultsBase with AR-specific attributes and methods.

    Additional Attributes
    ---------------------
    lags : list[int]
        Lag indices used in the model.
    ssr : NDArray[np.floating]
        Sum of squared residuals at each time point.
    rsquared : NDArray[np.floating]
        R-squared at each time point.
    """

    lags: list[int] = field(default_factory=list)
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
        # Find indices of lag parameters
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
        """Generate a text summary of rolling AR results.

        Returns
        -------
        str
            Formatted summary of the rolling AR estimation.
        """
        lines = []
        lines.append("=" * 70)
        est_type = "Rolling" if self.is_rolling else "Recursive"
        max_lag = max(self.lags) if self.lags else 0
        lines.append(f"{est_type} AR({max_lag}) Results".center(70))
        lines.append("=" * 70)

        lines.append(f"No. Observations:        {self.nobs:>10}")
        lines.append(f"No. Parameters:          {self.n_params:>10}")
        lines.append(f"Lags:                    {self.lags}")

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
class RecursiveARResults(RollingARResults):
    """Results from recursive (expanding window) AR estimation.

    Inherits from RollingARResults with recursive-specific defaults.
    """

    pass


class RollingAR(RollingEstimatorBase):
    """Rolling AR regression with fixed window size.

    Estimates AR models using a rolling (moving) window of fixed size.
    At each time point t, the model is estimated using observations
    from t - window + 1 to t.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    lags : int | Sequence[int]
        Number of lags (if int) or specific lag indices (if sequence).
    window : int
        Window size for rolling estimation.
    exog : ArrayLike | None
        Additional exogenous regressors.
    trend : str
        Trend to include: "c" (constant only), "ct" (constant and trend),
        "n" (no deterministic terms).

    Examples
    --------
    >>> import numpy as np
    >>> from regimes.rolling import RollingAR
    >>> np.random.seed(42)
    >>> n = 200
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = 0.7 * y[t-1] + np.random.randn()
    >>> rolling = RollingAR(y, lags=1, window=60)
    >>> results = rolling.fit()
    >>> print(results.summary())

    See Also
    --------
    RecursiveAR : Expanding window estimation.
    RollingAR.from_model : Create from AR model.
    """

    def __init__(
        self,
        endog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        lags: int | Sequence[int],
        window: int,
        exog: ArrayLike | pd.Series[Any] | pd.DataFrame | None = None,
        trend: str = "c",
    ) -> None:
        """Initialize RollingAR estimator."""
        # Store original endog for design matrix construction
        self._endog_full = np.asarray(endog, dtype=np.float64)
        self._exog_full = (
            np.asarray(exog, dtype=np.float64) if exog is not None else None
        )

        # Process lags
        if isinstance(lags, int):
            self.lags: list[int] = list(range(1, lags + 1))
        else:
            self.lags = sorted(list(lags))

        if not self.lags:
            raise ValueError("At least one lag must be specified")

        if any(lag < 1 for lag in self.lags):
            raise ValueError("All lags must be positive integers")

        self.maxlag = max(self.lags)
        self.trend = trend

        if trend not in ("c", "ct", "n"):
            raise ValueError(f"trend must be 'c', 'ct', or 'n', got {trend}")

        # Build design matrix for the full sample
        y_eff, X_full, param_names = self._build_design_matrix()

        # Initialize base class with effective sample
        super().__init__(
            endog=y_eff,
            exog=X_full,
            window=window,
            min_nobs=window,
            param_names=param_names,
        )

        # Store lags for results
        self._lags = self.lags

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
        y = self._endog_full[self.maxlag :]
        n_eff = len(y)

        # Build design matrix components
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

        # Lagged dependent variable
        lag_matrix = self._create_lag_matrix(self._endog_full)
        components.append(lag_matrix)
        param_names.extend([f"y.L{lag}" for lag in self.lags])

        # Exogenous variables
        if self._exog_full is not None:
            X_eff = self._exog_full[self.maxlag :]
            if X_eff.ndim == 1:
                X_eff = X_eff.reshape(-1, 1)
            components.append(X_eff)
            # Generate names for exog
            n_exog = X_eff.shape[1]
            param_names.extend([f"x{i}" for i in range(n_exog)])

        X = np.column_stack(components)
        return y, X, param_names

    def _create_lag_matrix(
        self, data: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Create a matrix of lagged values.

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

        lag_matrix = np.zeros((n_eff, len(self.lags)))
        for i, lag in enumerate(self.lags):
            lag_matrix[:, i] = data[self.maxlag - lag : n - lag]

        return lag_matrix

    @classmethod
    def from_model(cls, model: AR, window: int) -> RollingAR:
        """Create RollingAR estimator from an existing AR model.

        Parameters
        ----------
        model : AR
            An AR model instance.
        window : int
            Window size for rolling estimation.

        Returns
        -------
        RollingAR
            Rolling AR estimator initialized with the model's data.

        Examples
        --------
        >>> from regimes import AR
        >>> model = AR(y, lags=1)
        >>> rolling = RollingAR.from_model(model, window=60)
        >>> results = rolling.fit()
        """
        return cls(
            endog=model.endog,
            lags=model.lags,
            window=window,
            exog=model.exog,
            trend=model.trend,
        )

    def fit(self, cov_type: RollingCovType = "nonrobust") -> RollingARResults:
        """Fit rolling AR model.

        Parameters
        ----------
        cov_type : RollingCovType
            Type of covariance estimator:
            - "nonrobust": Standard OLS covariance
            - "HC0": Heteroskedasticity-robust (White)

        Returns
        -------
        RollingARResults
            Results object with rolling estimates.

        Notes
        -----
        Estimates are NaN-padded for observations where there are not
        enough data points for the rolling window.
        """
        n = self.nobs
        k = self.n_params
        window = self.window

        if window is None:
            raise ValueError("window must be specified for RollingAR")

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

            # Fit OLS for this window
            try:
                sm_model = sm.OLS(y_win, X_win)
                if cov_type == "nonrobust":
                    sm_results = sm_model.fit()
                else:  # HC0
                    sm_results = sm_model.fit(cov_type="HC0")

                params[t] = sm_results.params
                bse[t] = sm_results.bse
                ssr[t] = sm_results.ssr
                rsquared[t] = sm_results.rsquared
                n_valid += 1
            except (np.linalg.LinAlgError, ValueError):
                # Singular matrix or other estimation error
                continue

        return RollingARResults(
            params=params,
            bse=bse,
            nobs=n,
            window=window,
            min_nobs=window,
            n_valid=n_valid,
            cov_type=cov_type,
            param_names=self.param_names,
            lags=self._lags,
            ssr=ssr,
            rsquared=rsquared,
        )


class RecursiveAR(RollingEstimatorBase):
    """Recursive (expanding window) AR regression.

    Estimates AR models using an expanding window starting from
    min_nobs observations. At each time point t, the model is estimated
    using all observations from 0 to t.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    lags : int | Sequence[int]
        Number of lags (if int) or specific lag indices (if sequence).
    min_nobs : int | None
        Minimum number of observations to start estimation.
    exog : ArrayLike | None
        Additional exogenous regressors.
    trend : str
        Trend to include: "c" (constant only), "ct" (constant and trend),
        "n" (no deterministic terms).

    Examples
    --------
    >>> import numpy as np
    >>> from regimes.rolling import RecursiveAR
    >>> np.random.seed(42)
    >>> n = 200
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = 0.7 * y[t-1] + np.random.randn()
    >>> recursive = RecursiveAR(y, lags=1, min_nobs=30)
    >>> results = recursive.fit()
    >>> print(results.summary())

    See Also
    --------
    RollingAR : Fixed window estimation.
    RecursiveAR.from_model : Create from AR model.
    """

    def __init__(
        self,
        endog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        lags: int | Sequence[int],
        min_nobs: int | None = None,
        exog: ArrayLike | pd.Series[Any] | pd.DataFrame | None = None,
        trend: str = "c",
    ) -> None:
        """Initialize RecursiveAR estimator."""
        # Store original endog for design matrix construction
        self._endog_full = np.asarray(endog, dtype=np.float64)
        self._exog_full = (
            np.asarray(exog, dtype=np.float64) if exog is not None else None
        )

        # Process lags
        if isinstance(lags, int):
            self.lags: list[int] = list(range(1, lags + 1))
        else:
            self.lags = sorted(list(lags))

        if not self.lags:
            raise ValueError("At least one lag must be specified")

        if any(lag < 1 for lag in self.lags):
            raise ValueError("All lags must be positive integers")

        self.maxlag = max(self.lags)
        self.trend = trend

        if trend not in ("c", "ct", "n"):
            raise ValueError(f"trend must be 'c', 'ct', or 'n', got {trend}")

        # Build design matrix for the full sample
        y_eff, X_full, param_names = self._build_design_matrix()

        # Initialize base class with effective sample
        super().__init__(
            endog=y_eff,
            exog=X_full,
            window=None,  # No window for recursive
            min_nobs=min_nobs,
            param_names=param_names,
        )

        # Store lags for results
        self._lags = self.lags

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
        y = self._endog_full[self.maxlag :]
        n_eff = len(y)

        # Build design matrix components
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

        # Lagged dependent variable
        lag_matrix = self._create_lag_matrix(self._endog_full)
        components.append(lag_matrix)
        param_names.extend([f"y.L{lag}" for lag in self.lags])

        # Exogenous variables
        if self._exog_full is not None:
            X_eff = self._exog_full[self.maxlag :]
            if X_eff.ndim == 1:
                X_eff = X_eff.reshape(-1, 1)
            components.append(X_eff)
            # Generate names for exog
            n_exog = X_eff.shape[1]
            param_names.extend([f"x{i}" for i in range(n_exog)])

        X = np.column_stack(components)
        return y, X, param_names

    def _create_lag_matrix(
        self, data: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Create a matrix of lagged values.

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

        lag_matrix = np.zeros((n_eff, len(self.lags)))
        for i, lag in enumerate(self.lags):
            lag_matrix[:, i] = data[self.maxlag - lag : n - lag]

        return lag_matrix

    @classmethod
    def from_model(cls, model: AR, min_nobs: int | None = None) -> RecursiveAR:
        """Create RecursiveAR estimator from an existing AR model.

        Parameters
        ----------
        model : AR
            An AR model instance.
        min_nobs : int | None
            Minimum observations to start estimation.

        Returns
        -------
        RecursiveAR
            Recursive AR estimator initialized with the model's data.

        Examples
        --------
        >>> from regimes import AR
        >>> model = AR(y, lags=1)
        >>> recursive = RecursiveAR.from_model(model, min_nobs=30)
        >>> results = recursive.fit()
        """
        return cls(
            endog=model.endog,
            lags=model.lags,
            min_nobs=min_nobs,
            exog=model.exog,
            trend=model.trend,
        )

    def fit(self, cov_type: RollingCovType = "nonrobust") -> RecursiveARResults:
        """Fit recursive AR model.

        Parameters
        ----------
        cov_type : RollingCovType
            Type of covariance estimator:
            - "nonrobust": Standard OLS covariance
            - "HC0": Heteroskedasticity-robust (White)

        Returns
        -------
        RecursiveARResults
            Results object with recursive estimates.

        Notes
        -----
        Estimates are NaN-padded for observations where there are not
        enough data points for estimation.
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

            # Fit OLS for this window
            try:
                sm_model = sm.OLS(y_win, X_win)
                if cov_type == "nonrobust":
                    sm_results = sm_model.fit()
                else:  # HC0
                    sm_results = sm_model.fit(cov_type="HC0")

                params[t] = sm_results.params
                bse[t] = sm_results.bse
                ssr[t] = sm_results.ssr
                rsquared[t] = sm_results.rsquared
                n_valid += 1
            except (np.linalg.LinAlgError, ValueError):
                # Singular matrix or other estimation error
                continue

        return RecursiveARResults(
            params=params,
            bse=bse,
            nobs=n,
            window=None,
            min_nobs=self.min_nobs,
            n_valid=n_valid,
            cov_type=cov_type,
            param_names=self.param_names,
            lags=self._lags,
            ssr=ssr,
            rsquared=rsquared,
        )
