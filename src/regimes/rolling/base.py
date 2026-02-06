"""Base classes for rolling and recursive estimation.

This module provides the foundational classes for rolling and recursive
estimation that OLS and AR rolling estimators inherit from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray


RollingCovType = Literal["nonrobust", "HC0"]


def _ensure_array(
    data: ArrayLike | pd.Series[Any] | pd.DataFrame | None,
    name: str = "data",
    ndim: int | None = None,
) -> NDArray[np.floating[Any]] | None:
    """Convert input data to a numpy array.

    Parameters
    ----------
    data : ArrayLike | pd.Series | pd.DataFrame | None
        Input data to convert.
    name : str
        Name of the variable for error messages.
    ndim : int | None
        Expected number of dimensions. If None, no check is performed.

    Returns
    -------
    NDArray[np.floating] | None
        Converted array, or None if input is None.

    Raises
    ------
    ValueError
        If data has unexpected dimensions.
    """
    if data is None:
        return None

    if isinstance(data, (pd.Series, pd.DataFrame)):
        arr = data.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(data, dtype=np.float64)

    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got {arr.ndim}")

    return arr


class RollingEstimatorBase(ABC):
    """Abstract base class for rolling and recursive estimators.

    This class defines the common interface for rolling and recursive
    estimation of regression models.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    exog : ArrayLike
        Exogenous regressors (n_obs, k).
    window : int | None
        Window size for rolling estimation. If None, uses recursive
        (expanding) estimation.
    min_nobs : int | None
        Minimum number of observations for estimation. For rolling,
        defaults to window. For recursive, defaults to k + 1 (number
        of parameters plus one).
    param_names : Sequence[str] | None
        Names of the parameters for display purposes.

    Attributes
    ----------
    endog : NDArray[np.floating]
        Dependent variable array.
    exog : NDArray[np.floating]
        Exogenous regressors array.
    window : int | None
        Window size (None for recursive estimation).
    min_nobs : int
        Minimum observations for valid estimation.
    nobs : int
        Total number of observations.
    n_params : int
        Number of parameters (columns in exog).
    """

    def __init__(
        self,
        endog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        exog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        window: int | None = None,
        min_nobs: int | None = None,
        param_names: Sequence[str] | None = None,
    ) -> None:
        """Initialize the rolling estimator."""
        self._endog_orig = endog
        self._exog_orig = exog

        # Convert to arrays
        endog_arr = _ensure_array(endog, "endog")
        if endog_arr is None:
            raise ValueError("endog cannot be None")
        self.endog: NDArray[np.floating[Any]] = endog_arr

        exog_arr = _ensure_array(exog, "exog")
        if exog_arr is None:
            raise ValueError("exog cannot be None for rolling estimation")
        if exog_arr.ndim == 1:
            exog_arr = exog_arr.reshape(-1, 1)
        self.exog: NDArray[np.floating[Any]] = exog_arr

        # Validate dimensions
        if len(self.endog) != len(self.exog):
            raise ValueError(
                f"endog and exog must have same length, "
                f"got {len(self.endog)} and {len(self.exog)}"
            )

        self.window = window
        self._n_params = self.exog.shape[1]

        # Determine minimum observations
        if min_nobs is None:
            if window is not None:
                min_nobs = window
            else:
                min_nobs = self._n_params + 1
        self.min_nobs = min_nobs

        # Validate window/min_nobs
        if window is not None and window < self._n_params + 1:
            raise ValueError(
                f"window ({window}) must be at least n_params + 1 ({self._n_params + 1})"
            )
        if self.min_nobs < self._n_params + 1:
            raise ValueError(
                f"min_nobs ({self.min_nobs}) must be at least n_params + 1 ({self._n_params + 1})"
            )

        # Handle parameter names
        if param_names is not None:
            self._param_names = list(param_names)
        elif isinstance(exog, pd.DataFrame):
            self._param_names = [str(c) for c in exog.columns]
        else:
            self._param_names = [f"x{i}" for i in range(self._n_params)]

    @property
    def nobs(self) -> int:
        """Number of observations."""
        return len(self.endog)

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return self._n_params

    @property
    def param_names(self) -> list[str]:
        """Parameter names."""
        return self._param_names

    @property
    def is_rolling(self) -> bool:
        """Whether this is rolling (True) or recursive (False) estimation."""
        return self.window is not None

    @abstractmethod
    def fit(self, cov_type: RollingCovType = "nonrobust") -> RollingResultsBase:
        """Fit the rolling/recursive model.

        Parameters
        ----------
        cov_type : RollingCovType
            Type of covariance estimator:
            - "nonrobust": Standard OLS covariance
            - "HC0": Heteroskedasticity-robust

        Returns
        -------
        RollingResultsBase
            Results object containing rolling/recursive estimates.
        """
        ...

    def __repr__(self) -> str:
        """Return string representation of the estimator."""
        class_name = self.__class__.__name__
        est_type = "rolling" if self.is_rolling else "recursive"
        window_info = (
            f"window={self.window}" if self.is_rolling else f"min_nobs={self.min_nobs}"
        )
        return f"{class_name}({est_type}, {window_info}, nobs={self.nobs})"


@dataclass(kw_only=True)
class RollingResultsBase(ABC):
    """Base class for rolling/recursive estimation results.

    This abstract base class defines the common interface for results
    from rolling and recursive estimation.

    Attributes
    ----------
    params : NDArray[np.floating]
        Parameter estimates, shape (nobs, n_params). NaN-padded at the
        start until the first valid estimate.
    bse : NDArray[np.floating]
        Standard errors, shape (nobs, n_params). NaN-padded like params.
    nobs : int
        Total number of observations.
    window : int | None
        Window size (None for recursive estimation).
    min_nobs : int
        Minimum observations used for estimation.
    n_valid : int
        Number of valid estimates (non-NaN).
    cov_type : str
        Type of covariance estimator used.
    param_names : list[str]
        Names of the parameters.
    """

    params: NDArray[np.floating[Any]]
    bse: NDArray[np.floating[Any]]
    nobs: int
    window: int | None
    min_nobs: int
    n_valid: int
    cov_type: str
    param_names: list[str]

    # Private cache for computed properties
    _tvalues: NDArray[np.floating[Any]] | None = field(default=None, repr=False)
    _pvalues: NDArray[np.floating[Any]] | None = field(default=None, repr=False)

    @property
    def is_rolling(self) -> bool:
        """Whether these are rolling (True) or recursive (False) results."""
        return self.window is not None

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return self.params.shape[1]

    @property
    def tvalues(self) -> NDArray[np.floating[Any]]:
        """t-statistics for parameter estimates.

        Returns
        -------
        NDArray[np.floating]
            Shape (nobs, n_params). NaN where estimates are not valid.
        """
        if self._tvalues is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                self._tvalues = self.params / self.bse
        return self._tvalues

    @property
    def pvalues(self) -> NDArray[np.floating[Any]]:
        """Two-sided p-values for t-statistics.

        Returns
        -------
        NDArray[np.floating]
            Shape (nobs, n_params). NaN where estimates are not valid.

        Notes
        -----
        Uses window size (or current sample size for recursive) minus
        n_params as degrees of freedom for each estimate.
        """
        if self._pvalues is None:
            from scipy import stats

            # Calculate degrees of freedom for each observation
            if self.is_rolling:
                df = self.window - self.n_params if self.window else 1
                df_arr = np.full(self.nobs, df)
            else:
                # For recursive, df increases with sample size
                df_arr = np.arange(self.nobs) + 1 - self.n_params
                df_arr = np.maximum(df_arr, 1)  # Ensure at least 1 df

            # Compute p-values
            pvals = np.full_like(self.params, np.nan)
            for i in range(self.nobs):
                if not np.isnan(self.tvalues[i, 0]):
                    pvals[i] = 2 * (1 - stats.t.cdf(np.abs(self.tvalues[i]), df_arr[i]))

            self._pvalues = pvals

        return self._pvalues

    def conf_int(self, alpha: float = 0.05) -> NDArray[np.floating[Any]]:
        """Compute confidence intervals for parameter estimates.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level. Default gives 95% confidence intervals.

        Returns
        -------
        NDArray[np.floating]
            Array of shape (nobs, n_params, 2) with lower and upper bounds.
        """
        from scipy import stats

        # Calculate degrees of freedom for each observation
        if self.is_rolling:
            df = self.window - self.n_params if self.window else 1
            q = stats.t.ppf(1 - alpha / 2, df)
            lower = self.params - q * self.bse
            upper = self.params + q * self.bse
        else:
            # For recursive, df increases with sample size
            lower = np.full_like(self.params, np.nan)
            upper = np.full_like(self.params, np.nan)

            for i in range(self.nobs):
                if not np.isnan(self.params[i, 0]):
                    df = i + 1 - self.n_params
                    if df > 0:
                        q = stats.t.ppf(1 - alpha / 2, df)
                        lower[i] = self.params[i] - q * self.bse[i]
                        upper[i] = self.params[i] + q * self.bse[i]

        return np.stack([lower, upper], axis=-1)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert parameter estimates to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with parameter estimates, indexed by observation.
            Column names are parameter names.
        """
        return pd.DataFrame(
            self.params,
            columns=self.param_names,
        )

    def to_dataframe_full(self) -> pd.DataFrame:
        """Convert all estimates with standard errors to DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for each parameter's estimate and
            standard error (e.g., "const", "const_se", "x1", "x1_se").
        """
        data = {}
        for i, name in enumerate(self.param_names):
            data[name] = self.params[:, i]
            data[f"{name}_se"] = self.bse[:, i]
        return pd.DataFrame(data)

    @abstractmethod
    def plot_coefficients(
        self,
        variables: Sequence[str] | None = None,
        alpha: float = 0.05,
    ) -> Any:
        """Plot coefficient estimates over time with confidence bands.

        Parameters
        ----------
        variables : Sequence[str] | None
            Which variables to plot. If None, plots all.
        alpha : float
            Significance level for confidence bands.

        Returns
        -------
        tuple[Figure, Axes | NDArray[Axes]]
            Matplotlib figure and axes.
        """
        ...

    def __repr__(self) -> str:
        """Return string representation of results."""
        class_name = self.__class__.__name__
        est_type = "rolling" if self.is_rolling else "recursive"
        return f"{class_name}({est_type}, nobs={self.nobs}, n_valid={self.n_valid})"
