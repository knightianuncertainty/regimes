"""Base classes for structural break models.

This module provides the foundational model classes that all specific
models in the regimes package inherit from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    from regimes.results.base import RegimesResultsBase


CovType = Literal["nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC"]


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


class RegimesModelBase(ABC):
    """Abstract base class for all structural break models.

    This class defines the common interface that all models in the
    regimes package must implement. It follows the statsmodels
    convention of `Model(endog, exog).fit() -> Results`.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,) or (n_obs, n_endog).
    exog : ArrayLike | None
        Exogenous regressors (n_obs, n_exog). If None, model-specific
        defaults apply (e.g., constant only).
    breaks : Sequence[int] | None
        Known break points (observation indices). If None, breaks may be
        estimated or the model assumes no breaks.

    Attributes
    ----------
    endog : NDArray[np.floating]
        Dependent variable array.
    exog : NDArray[np.floating] | None
        Exogenous regressors array.
    breaks : Sequence[int]
        Break points (empty list if none).
    nobs : int
        Number of observations.
    n_breaks : int
        Number of break points.
    n_regimes : int
        Number of regimes (n_breaks + 1).

    Notes
    -----
    Subclasses must implement the `fit` method to perform estimation and
    return an appropriate results object.
    """

    def __init__(
        self,
        endog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        exog: ArrayLike | pd.Series[Any] | pd.DataFrame | None = None,
        breaks: Sequence[int] | None = None,
    ) -> None:
        """Initialize the model with data and optional break points."""
        self._endog_orig = endog
        self._exog_orig = exog

        # Convert to arrays
        endog_arr = _ensure_array(endog, "endog")
        if endog_arr is None:
            raise ValueError("endog cannot be None")
        self.endog: NDArray[np.floating[Any]] = endog_arr

        self.exog: NDArray[np.floating[Any]] | None = _ensure_array(exog, "exog")

        # Handle 1D endog
        if self.endog.ndim == 1:
            self._endog_names: list[str] = ["y"]
        else:
            self._endog_names = [f"y{i}" for i in range(self.endog.shape[1])]

        # Handle exog names
        if self.exog is not None:
            if self.exog.ndim == 1:
                self._exog_names: list[str] = ["x0"]
                self.exog = self.exog.reshape(-1, 1)
            else:
                self._exog_names = [f"x{i}" for i in range(self.exog.shape[1])]
        else:
            self._exog_names = []

        # Extract names from pandas objects if available
        if isinstance(endog, pd.Series):
            self._endog_names = [str(endog.name) if endog.name else "y"]
        elif isinstance(endog, pd.DataFrame):
            self._endog_names = [str(c) for c in endog.columns]

        if isinstance(exog, pd.Series):
            self._exog_names = [str(exog.name) if exog.name else "x0"]
        elif isinstance(exog, pd.DataFrame):
            self._exog_names = [str(c) for c in exog.columns]

        # Store breaks
        self.breaks: Sequence[int] = list(breaks) if breaks is not None else []

        # Validate
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate input data dimensions and consistency.

        Raises
        ------
        ValueError
            If data dimensions are inconsistent or break indices are invalid.
        """
        if self.endog.ndim > 2:
            raise ValueError("endog must be 1D or 2D")

        if self.exog is not None:
            if self.exog.ndim > 2:
                raise ValueError("exog must be 1D or 2D")
            if len(self.exog) != len(self.endog):
                raise ValueError(
                    f"endog and exog must have same length, "
                    f"got {len(self.endog)} and {len(self.exog)}"
                )

        # Validate break indices
        for b in self.breaks:
            if b < 0 or b >= self.nobs:
                raise ValueError(
                    f"Break index {b} is out of bounds [0, {self.nobs - 1}]"
                )

    @property
    def nobs(self) -> int:
        """Number of observations."""
        return len(self.endog)

    @property
    def n_breaks(self) -> int:
        """Number of break points."""
        return len(self.breaks)

    @property
    def n_regimes(self) -> int:
        """Number of regimes (n_breaks + 1)."""
        return self.n_breaks + 1

    @property
    def k_exog(self) -> int:
        """Number of exogenous variables (excluding any added constant)."""
        if self.exog is None:
            return 0
        return self.exog.shape[1] if self.exog.ndim == 2 else 1

    def get_regime_indices(self) -> list[tuple[int, int]]:
        """Get start and end indices for each regime.

        Returns
        -------
        list[tuple[int, int]]
            List of (start, end) tuples for each regime. End indices are
            exclusive (suitable for slicing).
        """
        if not self.breaks:
            return [(0, self.nobs)]

        sorted_breaks = sorted(self.breaks)
        regime_indices = []

        # First regime
        regime_indices.append((0, sorted_breaks[0]))

        # Middle regimes
        for i in range(len(sorted_breaks) - 1):
            regime_indices.append((sorted_breaks[i], sorted_breaks[i + 1]))

        # Last regime
        regime_indices.append((sorted_breaks[-1], self.nobs))

        return regime_indices

    def get_regime_data(
        self, regime: int
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]] | None]:
        """Get endog and exog data for a specific regime.

        Parameters
        ----------
        regime : int
            Regime index (0-based).

        Returns
        -------
        tuple[NDArray[np.floating], NDArray[np.floating] | None]
            Tuple of (endog, exog) arrays for the specified regime.

        Raises
        ------
        ValueError
            If regime index is out of bounds.
        """
        if regime < 0 or regime >= self.n_regimes:
            raise ValueError(
                f"Regime {regime} is out of bounds [0, {self.n_regimes - 1}]"
            )

        indices = self.get_regime_indices()
        start, end = indices[regime]

        endog_regime = self.endog[start:end]
        exog_regime = self.exog[start:end] if self.exog is not None else None

        return endog_regime, exog_regime

    @abstractmethod
    def fit(self, cov_type: CovType = "nonrobust", **kwargs: Any) -> RegimesResultsBase:
        """Fit the model and return results.

        Parameters
        ----------
        cov_type : CovType
            Type of covariance estimator. Options include:
            - "nonrobust": Standard OLS covariance
            - "HC0", "HC1", "HC2", "HC3": Heteroskedasticity-robust
            - "HAC": Heteroskedasticity and autocorrelation consistent
        **kwargs
            Additional model-specific arguments.

        Returns
        -------
        RegimesResultsBase
            Results object containing estimates and inference.
        """
        ...

    def __repr__(self) -> str:
        """Return string representation of the model."""
        class_name = self.__class__.__name__
        return (
            f"{class_name}(nobs={self.nobs}, k_exog={self.k_exog}, "
            f"n_breaks={self.n_breaks})"
        )


class TimeSeriesModelBase(RegimesModelBase):
    """Base class for time series models with lags.

    Extends the base model class with functionality for autoregressive
    and distributed lag models.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable.
    exog : ArrayLike | None
        Exogenous regressors.
    lags : int | Sequence[int]
        Number of lags or specific lag indices.
    breaks : Sequence[int] | None
        Known break points.

    Attributes
    ----------
    lags : list[int]
        List of lag indices used in the model.
    maxlag : int
        Maximum lag in the model.
    """

    def __init__(
        self,
        endog: ArrayLike | pd.Series[Any] | pd.DataFrame,
        exog: ArrayLike | pd.Series[Any] | pd.DataFrame | None = None,
        lags: int | Sequence[int] = 1,
        breaks: Sequence[int] | None = None,
    ) -> None:
        """Initialize the time series model."""
        super().__init__(endog, exog, breaks)

        # Process lags
        if isinstance(lags, int):
            self.lags: list[int] = list(range(1, lags + 1))
        else:
            self.lags = sorted(list(lags))

        if not self.lags:
            raise ValueError("At least one lag must be specified")

        if any(lag < 1 for lag in self.lags):
            raise ValueError("All lags must be positive integers")

    @property
    def maxlag(self) -> int:
        """Maximum lag in the model."""
        return max(self.lags)

    @property
    def nobs_effective(self) -> int:
        """Effective number of observations (after losing lags)."""
        return self.nobs - self.maxlag

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
