"""Hypothesis strategies for property-based testing of regimes.

This module provides reusable data generators for property tests using
the Hypothesis library.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from hypothesis import strategies as st
from numpy.typing import NDArray


@st.composite
def regression_data(
    draw: st.DrawFn,
    min_n: int = 30,
    max_n: int = 100,
    min_k: int = 1,
    max_k: int = 5,
    with_constant: bool = True,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Generate regression data (y, X) for OLS.

    Generates data where y = X @ beta + noise with X full rank.
    Uses a seeded random generator to ensure data has variation.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.
    min_n : int
        Minimum number of observations.
    max_n : int
        Maximum number of observations.
    min_k : int
        Minimum number of regressors (excluding constant).
    max_k : int
        Maximum number of regressors (excluding constant).
    with_constant : bool
        Whether to include a constant column in X.

    Returns
    -------
    tuple[NDArray, NDArray]
        y (n,) and X (n, k) arrays.
    """
    # Draw dimensions
    k_regressors = draw(st.integers(min_value=min_k, max_value=max_k))
    n = draw(st.integers(min_value=max(min_n, k_regressors + 5), max_value=max_n))

    # Total columns including constant if requested
    k_total = k_regressors + (1 if with_constant else 0)

    # Use a seeded random generator to ensure non-degenerate data
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    # Generate X with random values from standard normal (ensures variation)
    X_data = rng.standard_normal((n, k_regressors)) * 2  # Scale for variety

    # Add constant column if requested (exact 1s)
    if with_constant:
        X = np.column_stack([np.ones(n), X_data])
    else:
        X = X_data

    # Check condition number and improve if needed
    cond = np.linalg.cond(X)
    if cond > 1e10 or not np.isfinite(cond):
        # Add small perturbation to non-constant columns only
        X[:, 1:] = X[:, 1:] + rng.standard_normal((n, k_regressors)) * 0.1

    # Generate true coefficients (ensure non-zero)
    beta = rng.uniform(-3, 3, size=k_total)
    # Make sure at least one coefficient is meaningfully different from zero
    beta[0] = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])

    # Generate noise with meaningful variance
    noise_scale = draw(st.floats(min_value=0.5, max_value=2.0))
    noise = rng.standard_normal(n) * noise_scale

    # Generate y = X @ beta + noise (ensures variation in y)
    y = X @ beta + noise

    return y, X


@st.composite
def stationary_ar_data(
    draw: st.DrawFn,
    min_n: int = 50,
    max_n: int = 150,
    max_p: int = 3,
) -> tuple[NDArray[np.floating[Any]], list[int], NDArray[np.floating[Any]]]:
    """Generate stationary AR(p) data.

    Generates AR data with coefficients that guarantee stationarity
    (sum of absolute AR coefficients < 1).

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.
    min_n : int
        Minimum number of observations.
    max_n : int
        Maximum number of observations.
    max_p : int
        Maximum AR order.

    Returns
    -------
    tuple[NDArray, list[int], NDArray]
        y (n,), lags list, and ar_params array.
    """
    # Draw AR order
    p = draw(st.integers(min_value=1, max_value=max_p))
    lags = list(range(1, p + 1))

    # Draw sample size
    n = draw(st.integers(min_value=max(min_n, p + 30), max_value=max_n))

    # Generate AR coefficients ensuring stationarity
    # Use smaller coefficients to ensure stability
    ar_params = np.array(
        draw(
            st.lists(
                st.floats(
                    min_value=-0.8 / p,
                    max_value=0.8 / p,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=p,
                max_size=p,
            )
        )
    )

    # Scale down if sum of absolute values >= 1 (to ensure stationarity)
    ar_sum_abs = np.sum(np.abs(ar_params))
    if ar_sum_abs >= 0.99:
        ar_params = ar_params * 0.95 / ar_sum_abs

    # Simulate AR(p) process
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**31 - 1)))
    y = np.zeros(n)

    # Burn-in period
    burn_in = 100
    y_full = np.zeros(n + burn_in)

    for t in range(p, n + burn_in):
        y_full[t] = np.sum(ar_params * y_full[t - p : t][::-1]) + rng.standard_normal()

    y = y_full[burn_in:]

    return y, lags, ar_params


@st.composite
def adl_data(
    draw: st.DrawFn,
    min_n: int = 50,
    max_n: int = 150,
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    int,
    int,
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
]:
    """Generate ADL(p,q) data.

    Generates ADL data with stationary AR component and distributed lags
    on a single exogenous variable.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.
    min_n : int
        Minimum number of observations.
    max_n : int
        Maximum number of observations.

    Returns
    -------
    tuple
        y (n,), x (n,), p (AR order), q (DL order), ar_params, dl_params.
    """
    # Draw specifications
    p = draw(st.integers(min_value=1, max_value=2))  # AR order
    q = draw(
        st.integers(min_value=0, max_value=2)
    )  # DL order (includes contemporaneous)
    maxlag = max(p, q)

    # Draw sample size
    n = draw(st.integers(min_value=max(min_n, maxlag + 30), max_value=max_n))

    # Generate AR coefficients ensuring stationarity
    ar_params = np.array(
        draw(
            st.lists(
                st.floats(
                    min_value=-0.7 / p,
                    max_value=0.7 / p,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=p,
                max_size=p,
            )
        )
    )

    # Ensure stationarity
    ar_sum_abs = np.sum(np.abs(ar_params))
    if ar_sum_abs >= 0.99:
        ar_params = ar_params * 0.95 / ar_sum_abs

    # Generate distributed lag coefficients (q+1 coefficients for lags 0 to q)
    n_dl_coefs = q + 1
    dl_params = np.array(
        draw(
            st.lists(
                st.floats(
                    min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
                ),
                min_size=n_dl_coefs,
                max_size=n_dl_coefs,
            )
        )
    )

    # Generate exogenous variable
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**31 - 1)))
    x = rng.standard_normal(n)

    # Simulate ADL(p,q) process
    # y_t = c + sum_i(phi_i * y_{t-i}) + sum_j(beta_j * x_{t-j}) + e_t
    burn_in = 100
    y_full = np.zeros(n + burn_in)
    x_full = np.concatenate([rng.standard_normal(burn_in), x])

    const = draw(
        st.floats(min_value=-2, max_value=2, allow_nan=False, allow_infinity=False)
    )

    for t in range(maxlag, n + burn_in):
        ar_term = np.sum(ar_params * y_full[t - p : t][::-1]) if p > 0 else 0
        dl_term = (
            np.sum(dl_params * x_full[t - q : t + 1][::-1]) if n_dl_coefs > 0 else 0
        )
        y_full[t] = const + ar_term + dl_term + rng.standard_normal() * 0.5

    y = y_full[burn_in:]

    return y, x, p, q, ar_params, dl_params


@st.composite
def rolling_regression_data(
    draw: st.DrawFn,
    min_n: int = 80,
    max_n: int = 150,
    min_k: int = 1,
    max_k: int = 3,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], int]:
    """Generate regression data for rolling/recursive estimation.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.
    min_n : int
        Minimum number of observations.
    max_n : int
        Maximum number of observations.
    min_k : int
        Minimum number of regressors (excluding constant).
    max_k : int
        Maximum number of regressors (excluding constant).

    Returns
    -------
    tuple[NDArray, NDArray, int]
        y (n,), X (n, k), and window size.
    """
    # Get regression data (note: with_constant=True means X has k+1 columns)
    y, X = draw(regression_data(min_n=min_n, max_n=max_n, min_k=min_k, max_k=max_k))
    n = len(y)
    k = X.shape[1]

    # Draw window size (must be at least k + 1, with some buffer for stability)
    min_window = k + 10  # Need reasonable window for stable estimates
    max_window = min(n // 2, 60)
    if max_window < min_window:
        max_window = min_window + 10

    window = draw(st.integers(min_value=min_window, max_value=max_window))

    return y, X, window
