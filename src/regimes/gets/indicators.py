"""Indicator generation for saturation analysis.

Provides functions to create step (SIS), impulse (IIS), multiplicative (MIS),
and trend (TIS) indicator matrices used in indicator saturation procedures.

All indicators use the **forward convention**: a step indicator at tau has
value 1 for all t >= tau, so coefficients estimate the *shift* at tau.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def _resolve_taus(
    n: int,
    taus: Sequence[int] | None,
    trim: float,
) -> list[int]:
    """Compute the set of candidate break dates.

    Parameters
    ----------
    n : int
        Sample size.
    taus : Sequence[int] | None
        Explicit break dates. If None, generates full set respecting trim.
    trim : float
        Fraction of observations trimmed from each end (0 to 0.5).

    Returns
    -------
    list[int]
        Sorted list of candidate break dates.
    """
    if taus is not None:
        result = sorted(set(taus))
        for tau in result:
            if tau < 0 or tau >= n:
                raise ValueError(f"tau={tau} out of range [0, {n - 1}]")
        return result

    lo = max(1, int(np.ceil(trim * n)))
    hi = n - int(np.ceil(trim * n))
    if lo > hi:
        raise ValueError(f"trim={trim} too large for n={n}: no valid break dates")
    return list(range(lo, hi + 1))


def step_indicators(
    n: int,
    taus: Sequence[int] | None = None,
    trim: float = 0.0,
) -> tuple[NDArray[np.floating[Any]], list[str]]:
    """Generate forward step indicators for SIS.

    Each column has value 1 for all t >= tau, 0 otherwise. The coefficient
    on a step indicator estimates the shift in the intercept at tau.

    Parameters
    ----------
    n : int
        Number of observations.
    taus : Sequence[int] | None
        Specific break dates. If None, generates full set respecting trim.
    trim : float
        Fraction trimmed from each end when generating the full set.

    Returns
    -------
    indicators : NDArray[np.floating]
        Indicator matrix of shape (n, len(taus)).
    names : list[str]
        Names of the form ``"step_50"``, ``"step_51"``, etc.

    Examples
    --------
    >>> S, names = step_indicators(100, taus=[25, 50, 75])
    >>> S.shape
    (100, 3)
    >>> names
    ['step_25', 'step_50', 'step_75']
    """
    resolved = _resolve_taus(n, taus, trim)
    k = len(resolved)
    S = np.zeros((n, k), dtype=np.float64)
    names: list[str] = []
    for j, tau in enumerate(resolved):
        S[tau:, j] = 1.0
        names.append(f"step_{tau}")
    return S, names


def impulse_indicators(
    n: int,
    taus: Sequence[int] | None = None,
    trim: float = 0.0,
) -> tuple[NDArray[np.floating[Any]], list[str]]:
    """Generate impulse indicators for IIS.

    Each column has value 1 at exactly t == tau, 0 elsewhere. The
    coefficient estimates an additive outlier at tau.

    Parameters
    ----------
    n : int
        Number of observations.
    taus : Sequence[int] | None
        Specific dates. If None, generates full set respecting trim.
    trim : float
        Fraction trimmed from each end when generating the full set.

    Returns
    -------
    indicators : NDArray[np.floating]
        Indicator matrix of shape (n, len(taus)).
    names : list[str]
        Names of the form ``"impulse_50"``.

    Examples
    --------
    >>> I, names = impulse_indicators(100, taus=[10, 90])
    >>> I.shape
    (100, 2)
    """
    resolved = _resolve_taus(n, taus, trim)
    k = len(resolved)
    ind = np.zeros((n, k), dtype=np.float64)
    names: list[str] = []
    for j, tau in enumerate(resolved):
        ind[tau, j] = 1.0
        names.append(f"impulse_{tau}")
    return ind, names


def multiplicative_indicators(
    exog: NDArray[np.floating[Any]],
    n: int,
    variables: Sequence[int | str] | None = None,
    taus: Sequence[int] | None = None,
    trim: float = 0.0,
    exog_names: Sequence[str] | None = None,
) -> tuple[NDArray[np.floating[Any]], list[str]]:
    """Generate multiplicative step indicators for MIS.

    Each column is ``step(t >= tau) * x_j`` — a step indicator multiplied
    by regressor j. The coefficient estimates the shift in the slope of
    x_j at tau.

    Parameters
    ----------
    exog : NDArray[np.floating]
        Regressor matrix of shape (n, k). Should **not** include a constant
        column (constant shifts are handled by SIS).
    n : int
        Number of observations (must match ``exog.shape[0]``).
    variables : Sequence[int | str] | None
        Which columns of exog to interact with step indicators. Accepts
        column indices (int) or names (str, matched against exog_names).
        If None, interacts with all columns.
    taus : Sequence[int] | None
        Specific break dates. If None, generates full set respecting trim.
    trim : float
        Fraction trimmed from each end when generating the full set.
    exog_names : Sequence[str] | None
        Names for the columns of exog. If None, uses ``"x0"``, ``"x1"``, etc.

    Returns
    -------
    indicators : NDArray[np.floating]
        Indicator matrix of shape (n, len(vars) * len(taus)).
    names : list[str]
        Names of the form ``"x0*step_50"``, ``"y.L1*step_120"``, etc.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(200, 2)
    >>> M, names = multiplicative_indicators(X, 200, taus=[50, 100, 150])
    >>> M.shape
    (200, 6)
    """
    exog = np.asarray(exog, dtype=np.float64)
    if exog.ndim == 1:
        exog = exog.reshape(-1, 1)
    if exog.shape[0] != n:
        raise ValueError(f"exog has {exog.shape[0]} rows but n={n}")

    k = exog.shape[1]
    col_names = (
        list(exog_names) if exog_names is not None else [f"x{j}" for j in range(k)]
    )
    if len(col_names) != k:
        raise ValueError(
            f"exog_names has {len(col_names)} entries but exog has {k} columns"
        )

    # Resolve which variables to interact
    if variables is None:
        var_indices = list(range(k))
    else:
        var_indices = []
        for v in variables:
            if isinstance(v, int):
                if v < 0 or v >= k:
                    raise ValueError(f"Variable index {v} out of range [0, {k - 1}]")
                var_indices.append(v)
            else:
                if v not in col_names:
                    raise ValueError(
                        f"Variable name '{v}' not found. Available: {col_names}"
                    )
                var_indices.append(col_names.index(v))

    resolved = _resolve_taus(n, taus, trim)
    n_cols = len(var_indices) * len(resolved)
    M = np.zeros((n, n_cols), dtype=np.float64)
    names: list[str] = []

    col = 0
    for vi in var_indices:
        vname = col_names[vi]
        for tau in resolved:
            M[tau:, col] = exog[tau:, vi]
            names.append(f"{vname}*step_{tau}")
            col += 1

    return M, names


def trend_indicators(
    n: int,
    taus: Sequence[int] | None = None,
    trim: float = 0.0,
) -> tuple[NDArray[np.floating[Any]], list[str]]:
    """Generate broken trend indicators for TIS.

    Each column has value ``max(0, t - tau)`` — a broken linear trend
    starting at tau. The coefficient estimates the change in trend slope
    at tau.

    Parameters
    ----------
    n : int
        Number of observations.
    taus : Sequence[int] | None
        Specific break dates. If None, generates full set respecting trim.
    trim : float
        Fraction trimmed from each end when generating the full set.

    Returns
    -------
    indicators : NDArray[np.floating]
        Indicator matrix of shape (n, len(taus)).
    names : list[str]
        Names of the form ``"trend_50"``.

    Examples
    --------
    >>> T, names = trend_indicators(100, taus=[30, 60])
    >>> T[30, 0]  # trend starts at tau
    0.0
    >>> T[31, 0]  # one period after tau
    1.0
    """
    resolved = _resolve_taus(n, taus, trim)
    k = len(resolved)
    t = np.arange(n, dtype=np.float64)
    T = np.zeros((n, k), dtype=np.float64)
    names: list[str] = []
    for j, tau in enumerate(resolved):
        T[:, j] = np.maximum(0.0, t - tau)
        names.append(f"trend_{tau}")
    return T, names
