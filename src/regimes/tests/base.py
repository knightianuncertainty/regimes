"""Base classes for structural break tests.

This module provides foundational classes for structural break testing
procedures in the regimes package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray
    from typing_extensions import Self


def split_constant(
    exog_all: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]] | None:
    """The regressors of a model without its constant column(s).

    Used for ``break_vars="const"``: the constant is the breaking regressor
    and must not also be among the regressors that do not break.

    Raises
    ------
    ValueError
        If the model has no constant.
    """
    is_const = [
        bool(exog_all[0, i] != 0.0 and np.ptp(exog_all[:, i]) == 0.0)
        for i in range(exog_all.shape[1])
    ]
    if not any(is_const):
        raise ValueError(
            "break_vars='const' tests a break in the intercept, but the "
            "model has no constant"
        )
    keep = [i for i, const in enumerate(is_const) if not const]
    if not keep:
        return None
    return exog_all[:, keep]


@dataclass
class BreakTestResultsBase(ABC):
    """Base class for structural break test results.

    This abstract base class defines the common interface for all break
    test result containers.

    Parameters
    ----------
    test_name : str
        Name of the test.
    nobs : int
        Number of observations.
    n_breaks : int
        Number of breaks detected/tested.
    break_indices : Sequence[int]
        Estimated break point indices.
    index_offset : int
        Number of observations of the underlying data before the first
        observation the test was computed on (the initial observations of
        an AR or ADL model; otherwise 0). Observation numbers in the
        results already include it.
    """

    test_name: str
    nobs: int
    n_breaks: int
    break_indices: Sequence[int]
    index_offset: int = 0

    @property
    def break_dates(self) -> Sequence[int]:
        """Alias for break_indices."""
        return self.break_indices

    @property
    def n_regimes(self) -> int:
        """Number of regimes (n_breaks + 1)."""
        return self.n_breaks + 1

    @abstractmethod
    def summary(self) -> str:
        """Generate a text summary of test results.

        Returns
        -------
        str
            Formatted summary string.
        """
        ...


class BreakTestBase(ABC):
    """Abstract base class for structural break tests.

    This class defines the common interface for all structural break
    testing procedures in the regimes package.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    exog : ArrayLike | None
        Exogenous regressors that are NOT subject to breaks.
    exog_break : ArrayLike | None
        Regressors whose coefficients may break.
    """

    #: Number of observations of the underlying data that come before the
    #: first observation the test is computed on. It is zero for a test built
    #: from arrays or from an OLS model, and the number of initial
    #: observations lost to the lags for a test built from an AR or ADL model
    #: (``from_model``). Every observation number in the results (break
    #: dates, candidate dates, crossings) and every break date passed to
    #: ``fit`` is an observation number of the underlying data: the position
    #: in the arrays of the test plus this offset. One exception:
    #: ``BaiPerronResults.to_ols()`` returns the regression on the arrays of
    #: the test, with break dates as positions in them.
    index_offset: int = 0

    @classmethod
    def _create(cls, index_offset: int, *args: Any, **kwargs: Any) -> Self:
        """Construct a test and record the offset of its first observation."""
        test = cls(*args, **kwargs)
        test.index_offset = int(index_offset)
        return test

    def __init__(
        self,
        endog: ArrayLike,
        exog: ArrayLike | None = None,
        exog_break: ArrayLike | None = None,
    ) -> None:
        """Initialize the test."""
        self.endog = np.asarray(endog, dtype=np.float64)
        self.exog = np.asarray(exog, dtype=np.float64) if exog is not None else None
        self.exog_break = (
            np.asarray(exog_break, dtype=np.float64) if exog_break is not None else None
        )

        # Ensure 2D
        if self.exog is not None and self.exog.ndim == 1:
            self.exog = self.exog.reshape(-1, 1)
        if self.exog_break is not None and self.exog_break.ndim == 1:
            self.exog_break = self.exog_break.reshape(-1, 1)

        self._validate()

    def _validate(self) -> None:
        """Validate input data."""
        if self.endog.ndim != 1:
            raise ValueError("endog must be 1-dimensional")

        n = len(self.endog)
        if self.exog is not None and len(self.exog) != n:
            raise ValueError("exog must have same length as endog")
        if self.exog_break is not None and len(self.exog_break) != n:
            raise ValueError("exog_break must have same length as endog")

    @property
    def nobs(self) -> int:
        """Number of observations."""
        return len(self.endog)

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> BreakTestResultsBase:
        """Perform the structural break test.

        Returns
        -------
        BreakTestResultsBase
            Results object containing test statistics and break estimates.
        """
        ...
