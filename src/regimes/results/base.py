"""Base classes for structural break model results.

This module provides the foundational result container classes that all
model-specific results inherit from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


@dataclass(kw_only=True)
class RegimesResultsBase(ABC):
    """Base class for all structural break model results.

    This abstract base class defines the common interface for all result
    containers in the regimes package. It follows the statsmodels
    convention where fitted models return results objects containing
    parameter estimates, standard errors, and inference methods.

    Parameters
    ----------
    params : NDArray[np.floating]
        Estimated model parameters.
    nobs : int
        Number of observations used in estimation.
    model_name : str
        Name of the model that produced these results.

    Attributes
    ----------
    params : NDArray[np.floating]
        Estimated model parameters.
    nobs : int
        Number of observations.
    model_name : str
        Name of the model.
    """

    params: NDArray[np.floating[Any]]
    nobs: int
    model_name: str = "StructBreakModel"

    @property
    @abstractmethod
    def df_model(self) -> int:
        """Degrees of freedom used by the model (number of parameters)."""
        ...

    @property
    def df_resid(self) -> int:
        """Residual degrees of freedom (nobs - df_model)."""
        return self.nobs - self.df_model

    @abstractmethod
    def summary(self) -> str:
        """Generate a text summary of the results.

        Returns
        -------
        str
            Formatted summary string.
        """
        ...


@dataclass(kw_only=True)
class RegressionResultsBase(RegimesResultsBase):
    """Base class for regression-type model results.

    Extends the base results class with attributes common to regression
    models, including standard errors, residuals, and fitted values.

    Parameters
    ----------
    params : NDArray[np.floating]
        Estimated regression coefficients.
    bse : NDArray[np.floating]
        Standard errors of the coefficients.
    resid : NDArray[np.floating]
        Model residuals.
    fittedvalues : NDArray[np.floating]
        Fitted values from the model.
    nobs : int
        Number of observations.
    cov_params_matrix : NDArray[np.floating]
        Covariance matrix of the parameter estimates.
    cov_type : str
        Type of covariance estimator used (e.g., "nonrobust", "HC0", "HAC").
    model_name : str
        Name of the model.
    param_names : Sequence[str] | None
        Names of the parameters for display purposes.

    Attributes
    ----------
    tvalues : NDArray[np.floating]
        t-statistics for the parameter estimates.
    pvalues : NDArray[np.floating]
        Two-sided p-values for the t-statistics.
    rsquared : float
        R-squared (coefficient of determination).
    rsquared_adj : float
        Adjusted R-squared.
    ssr : float
        Sum of squared residuals.
    """

    bse: NDArray[np.floating[Any]]
    resid: NDArray[np.floating[Any]]
    fittedvalues: NDArray[np.floating[Any]]
    cov_params_matrix: NDArray[np.floating[Any]]
    cov_type: str = "nonrobust"
    param_names: Sequence[str] | None = None
    _ssr: float | None = field(default=None, repr=False)
    _tss: float | None = field(default=None, repr=False)

    @property
    def df_model(self) -> int:
        """Degrees of freedom used by the model."""
        return len(self.params)

    @property
    def tvalues(self) -> NDArray[np.floating[Any]]:
        """t-statistics for parameter estimates."""
        return self.params / self.bse

    @property
    def pvalues(self) -> NDArray[np.floating[Any]]:
        """Two-sided p-values for t-statistics."""
        from scipy import stats

        return 2 * (1 - stats.t.cdf(np.abs(self.tvalues), self.df_resid))

    @property
    def ssr(self) -> float:
        """Sum of squared residuals."""
        if self._ssr is None:
            return float(np.sum(self.resid**2))
        return self._ssr

    @property
    def mse_resid(self) -> float:
        """Mean squared error of residuals."""
        return self.ssr / self.df_resid

    @property
    def rsquared(self) -> float:
        """R-squared (coefficient of determination)."""
        if self._tss is None or self._tss == 0:
            return np.nan
        return 1 - self.ssr / self._tss

    @property
    def rsquared_adj(self) -> float:
        """Adjusted R-squared."""
        if self._tss is None or self._tss == 0:
            return np.nan
        return 1 - (self.ssr / self.df_resid) / (self._tss / (self.nobs - 1))

    def conf_int(self, alpha: float = 0.05) -> NDArray[np.floating[Any]]:
        """Compute confidence intervals for parameter estimates.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level. Default gives 95% confidence intervals.

        Returns
        -------
        NDArray[np.floating]
            Array of shape (n_params, 2) with lower and upper bounds.
        """
        from scipy import stats

        q = stats.t.ppf(1 - alpha / 2, self.df_resid)
        lower = self.params - q * self.bse
        upper = self.params + q * self.bse
        return np.column_stack([lower, upper])

    def cov_params(self) -> NDArray[np.floating[Any]]:
        """Return the covariance matrix of parameter estimates.

        Returns
        -------
        NDArray[np.floating]
            Covariance matrix of shape (n_params, n_params).
        """
        return self.cov_params_matrix

    def summary(self) -> str:
        """Generate a text summary of regression results.

        Returns
        -------
        str
            Formatted summary including coefficients, standard errors,
            t-values, p-values, and model diagnostics.
        """
        lines = []
        lines.append("=" * 78)
        lines.append(f"{self.model_name:^78}")
        lines.append("=" * 78)
        lines.append(
            f"Dep. Variable:           y   No. Observations:    {self.nobs:>10}"
        )
        lines.append(
            f"Model:          {self.model_name:>10}   Df Residuals:        {self.df_resid:>10}"
        )
        lines.append(
            f"Cov. Type:      {self.cov_type:>10}   Df Model:            {self.df_model:>10}"
        )
        lines.append(
            f"R-squared:         {self.rsquared:>7.4f}   Adj. R-squared:      {self.rsquared_adj:>10.4f}"
        )
        lines.append("=" * 78)

        # Parameter table header
        lines.append(
            f"{'':>15} {'coef':>10} {'std err':>10} {'t':>10} "
            f"{'P>|t|':>10} {'[0.025':>10} {'0.975]':>10}"
        )
        lines.append("-" * 78)

        # Parameter rows
        ci = self.conf_int()
        names = self.param_names or [f"x{i}" for i in range(len(self.params))]
        for i, name in enumerate(names):
            lines.append(
                f"{name:>15} {self.params[i]:>10.4f} {self.bse[i]:>10.4f} "
                f"{self.tvalues[i]:>10.3f} {self.pvalues[i]:>10.3f} "
                f"{ci[i, 0]:>10.3f} {ci[i, 1]:>10.3f}"
            )

        lines.append("=" * 78)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with parameter estimates, standard errors, t-values,
            p-values, and confidence intervals.
        """
        ci = self.conf_int()
        names = self.param_names or [f"x{i}" for i in range(len(self.params))]
        return pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.bse,
                "t": self.tvalues,
                "P>|t|": self.pvalues,
                "ci_lower": ci[:, 0],
                "ci_upper": ci[:, 1],
            },
            index=names,
        )

    def plot_diagnostics(
        self,
        nlags: int = 20,
        alpha: float = 0.05,
        figsize: tuple[float, float] = (12, 10),
    ) -> tuple[Any, NDArray[Any]]:
        """Plot PcGive-style misspecification diagnostic panel.

        Creates a 2x2 panel with:
        - Top left: Actual vs Fitted values
        - Top right: Residual Distribution histogram
        - Bottom left: Scaled Residuals (vertical index plot)
        - Bottom right: ACF and PACF (stacked)

        Parameters
        ----------
        nlags : int
            Number of lags for ACF/PACF plots.
        alpha : float
            Significance level for ACF/PACF confidence bands.
        figsize : tuple[float, float]
            Figure size (width, height) in inches.

        Returns
        -------
        tuple[Figure, NDArray[Axes]]
            Matplotlib figure and 2x2 array of axes.

        Examples
        --------
        >>> import numpy as np
        >>> import regimes as rg
        >>> np.random.seed(42)
        >>> y = np.random.randn(100)
        >>> X = np.column_stack([np.ones(100), np.random.randn(100)])
        >>> results = rg.OLS(y, X, has_constant=False).fit()
        >>> fig, axes = results.plot_diagnostics()
        """
        from regimes.visualization.diagnostics import plot_diagnostics

        return plot_diagnostics(self, nlags=nlags, alpha=alpha, figsize=figsize)


@dataclass(kw_only=True)
class BreakResultsBase(RegimesResultsBase):
    """Base class for structural break model results.

    This class extends the base results with attributes specific to models
    that estimate structural breaks in time series.

    Parameters
    ----------
    params : NDArray[np.floating]
        Estimated parameters (may vary across regimes).
    nobs : int
        Number of observations.
    break_indices : Sequence[int]
        Indices of estimated break points.
    n_breaks : int
        Number of breaks detected/estimated.
    model_name : str
        Name of the model.
    regime_params : dict[int, NDArray[np.floating]] | None
        Parameters for each regime, keyed by regime number (0, 1, ...).

    Attributes
    ----------
    n_regimes : int
        Number of regimes (n_breaks + 1).
    break_dates : Sequence[int]
        Same as break_indices (alias for convenience).
    """

    break_indices: Sequence[int]
    n_breaks: int
    regime_params: dict[int, NDArray[np.floating[Any]]] | None = None

    @property
    def n_regimes(self) -> int:
        """Number of regimes (n_breaks + 1)."""
        return self.n_breaks + 1

    @property
    def break_dates(self) -> Sequence[int]:
        """Alias for break_indices."""
        return self.break_indices

    @property
    def df_model(self) -> int:
        """Degrees of freedom used by the model."""
        return len(self.params)

    def summary(self) -> str:
        """Generate a text summary of break detection results.

        Returns
        -------
        str
            Formatted summary including break dates and regime information.
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"{self.model_name:^60}")
        lines.append("=" * 60)
        lines.append(f"Number of observations: {self.nobs}")
        lines.append(f"Number of breaks:       {self.n_breaks}")
        lines.append(f"Number of regimes:      {self.n_regimes}")
        lines.append("-" * 60)

        if self.n_breaks > 0:
            lines.append("Break points:")
            for i, idx in enumerate(self.break_indices):
                lines.append(f"  Break {i + 1}: observation {idx}")
        else:
            lines.append("No structural breaks detected.")

        lines.append("=" * 60)
        return "\n".join(lines)
