"""Rolling and recursive estimation.

This module provides rolling (fixed window) and recursive (expanding window)
estimation for OLS, AR, and ADL models.

Classes
-------
RollingOLS
    Rolling OLS regression with fixed window.
RecursiveOLS
    Recursive (expanding window) OLS regression.
RollingAR
    Rolling AR model with fixed window.
RecursiveAR
    Recursive (expanding window) AR model.
RollingADL
    Rolling ADL model with fixed window.
RecursiveADL
    Recursive (expanding window) ADL model.

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
"""

from regimes.rolling.adl import (
    RecursiveADL,
    RecursiveADLResults,
    RollingADL,
    RollingADLResults,
)
from regimes.rolling.ar import (
    RecursiveAR,
    RecursiveARResults,
    RollingAR,
    RollingARResults,
)
from regimes.rolling.base import (
    RollingCovType,
    RollingEstimatorBase,
    RollingResultsBase,
)
from regimes.rolling.ols import (
    RecursiveOLS,
    RecursiveOLSResults,
    RollingOLS,
    RollingOLSResults,
)

__all__ = [
    "RecursiveADL",
    "RecursiveADLResults",
    "RecursiveAR",
    "RecursiveARResults",
    "RecursiveOLS",
    "RecursiveOLSResults",
    "RollingADL",
    "RollingADLResults",
    "RollingAR",
    "RollingARResults",
    "RollingCovType",
    "RollingEstimatorBase",
    "RollingOLS",
    "RollingOLSResults",
    "RollingResultsBase",
]
