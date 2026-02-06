"""Rolling and recursive OLS estimation.

This module provides RollingOLS and RecursiveOLS classes for estimating
OLS regressions with rolling (fixed) or recursive (expanding) windows.
"""

from __future__ import annotations

from dataclasses import dataclass
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

    from regimes.models.ols import OLS


@dataclass(kw_only=True)
class RollingOLSResults(RollingResultsBase):
    """Results from rolling OLS estimation.

    Extends RollingResultsBase with OLS-specific attributes and methods.

    Additional Attributes
    ---------------------
    ssr : NDArray[np.floating]
        Sum of squared residuals at each time point.
    rsquared : NDArray[np.floating]
        R-squared at each time point.
    """

    ssr: NDArray[np.floating[Any]] | None = None
    rsquared: NDArray[np.floating[Any]] | None = None

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
        """Generate a text summary of rolling OLS results.

        Returns
        -------
        str
            Formatted summary of the rolling estimation.
        """
        lines = []
        lines.append("=" * 70)
        est_type = "Rolling" if self.is_rolling else "Recursive"
        lines.append(f"{est_type} OLS Regression Results".center(70))
        lines.append("=" * 70)

        lines.append(f"No. Observations:        {self.nobs:>10}")
        lines.append(f"No. Parameters:          {self.n_params:>10}")

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
class RecursiveOLSResults(RollingOLSResults):
    """Results from recursive (expanding window) OLS estimation.

    Inherits from RollingOLSResults with recursive-specific defaults.
    """

    pass


class RollingOLS(RollingEstimatorBase):
    """Rolling OLS regression with fixed window size.

    Estimates OLS regressions using a rolling (moving) window of fixed
    size. At each time point t, the model is estimated using observations
    from t - window + 1 to t.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    exog : ArrayLike
        Exogenous regressors (n_obs, k).
    window : int
        Window size for rolling estimation.
    param_names : Sequence[str] | None
        Names of the parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from regimes.rolling import RollingOLS
    >>> np.random.seed(42)
    >>> n = 200
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = X @ [1, 2] + np.random.randn(n)
    >>> rolling = RollingOLS(y, X, window=60)
    >>> results = rolling.fit()
    >>> print(results.summary())

    See Also
    --------
    RecursiveOLS : Expanding window estimation.
    RollingOLS.from_model : Create from OLS model.
    """

    def __init__(
        self,
        endog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        exog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        window: int,
        param_names: Sequence[str] | None = None,
    ) -> None:
        """Initialize RollingOLS estimator."""
        super().__init__(
            endog=endog,
            exog=exog,
            window=window,
            min_nobs=window,
            param_names=param_names,
        )

    @classmethod
    def from_model(cls, model: OLS, window: int) -> RollingOLS:
        """Create RollingOLS estimator from an existing OLS model.

        Parameters
        ----------
        model : OLS
            An OLS model instance.
        window : int
            Window size for rolling estimation.

        Returns
        -------
        RollingOLS
            Rolling OLS estimator initialized with the model's data.

        Examples
        --------
        >>> from regimes import OLS
        >>> model = OLS(y, X)
        >>> rolling = RollingOLS.from_model(model, window=60)
        >>> results = rolling.fit()
        """
        if model.exog is None:
            raise ValueError("Model must have exog for rolling estimation")

        return cls(
            endog=model.endog,
            exog=model.exog,
            window=window,
            param_names=model._exog_names,
        )

    def fit(self, cov_type: RollingCovType = "nonrobust") -> RollingOLSResults:
        """Fit rolling OLS model.

        Parameters
        ----------
        cov_type : RollingCovType
            Type of covariance estimator:
            - "nonrobust": Standard OLS covariance
            - "HC0": Heteroskedasticity-robust (White)

        Returns
        -------
        RollingOLSResults
            Results object with rolling estimates.

        Notes
        -----
        The first `window - 1` observations will have NaN estimates since
        there are not enough observations to fill the window.
        """
        n = self.nobs
        k = self.n_params
        window = self.window

        if window is None:
            raise ValueError("window must be specified for RollingOLS")

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

        return RollingOLSResults(
            params=params,
            bse=bse,
            nobs=n,
            window=window,
            min_nobs=window,
            n_valid=n_valid,
            cov_type=cov_type,
            param_names=self.param_names,
            ssr=ssr,
            rsquared=rsquared,
        )


class RecursiveOLS(RollingEstimatorBase):
    """Recursive (expanding window) OLS regression.

    Estimates OLS regressions using an expanding window starting from
    min_nobs observations. At each time point t, the model is estimated
    using all observations from 0 to t.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    exog : ArrayLike
        Exogenous regressors (n_obs, k).
    min_nobs : int | None
        Minimum number of observations to start estimation. Defaults to
        k + 1 (number of parameters plus one).
    param_names : Sequence[str] | None
        Names of the parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from regimes.rolling import RecursiveOLS
    >>> np.random.seed(42)
    >>> n = 200
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = X @ [1, 2] + np.random.randn(n)
    >>> recursive = RecursiveOLS(y, X, min_nobs=30)
    >>> results = recursive.fit()
    >>> print(results.summary())

    See Also
    --------
    RollingOLS : Fixed window estimation.
    RecursiveOLS.from_model : Create from OLS model.
    """

    def __init__(
        self,
        endog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        exog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        min_nobs: int | None = None,
        param_names: Sequence[str] | None = None,
    ) -> None:
        """Initialize RecursiveOLS estimator."""
        super().__init__(
            endog=endog,
            exog=exog,
            window=None,  # No window for recursive
            min_nobs=min_nobs,
            param_names=param_names,
        )

    @classmethod
    def from_model(cls, model: OLS, min_nobs: int | None = None) -> RecursiveOLS:
        """Create RecursiveOLS estimator from an existing OLS model.

        Parameters
        ----------
        model : OLS
            An OLS model instance.
        min_nobs : int | None
            Minimum observations to start estimation. Defaults to k + 1.

        Returns
        -------
        RecursiveOLS
            Recursive OLS estimator initialized with the model's data.

        Examples
        --------
        >>> from regimes import OLS
        >>> model = OLS(y, X)
        >>> recursive = RecursiveOLS.from_model(model, min_nobs=30)
        >>> results = recursive.fit()
        """
        if model.exog is None:
            raise ValueError("Model must have exog for recursive estimation")

        return cls(
            endog=model.endog,
            exog=model.exog,
            min_nobs=min_nobs,
            param_names=model._exog_names,
        )

    def fit(self, cov_type: RollingCovType = "nonrobust") -> RecursiveOLSResults:
        """Fit recursive OLS model.

        Parameters
        ----------
        cov_type : RollingCovType
            Type of covariance estimator:
            - "nonrobust": Standard OLS covariance
            - "HC0": Heteroskedasticity-robust (White)

        Returns
        -------
        RecursiveOLSResults
            Results object with recursive estimates.

        Notes
        -----
        The first `min_nobs - 1` observations will have NaN estimates since
        there are not enough observations for estimation.
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

        return RecursiveOLSResults(
            params=params,
            bse=bse,
            nobs=n,
            window=None,
            min_nobs=self.min_nobs,
            n_valid=n_valid,
            cov_type=cov_type,
            param_names=self.param_names,
            ssr=ssr,
            rsquared=rsquared,
        )
