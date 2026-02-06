"""Pytest configuration and fixtures for regimes tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_data(rng: np.random.Generator) -> NDArray[np.floating[Any]]:
    """Simple random data without breaks."""
    return rng.standard_normal(100)


@pytest.fixture
def data_with_mean_shift(
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating[Any]], int]:
    """Data with a single mean shift at t=100.

    Returns
    -------
    tuple[NDArray[np.floating], int]
        Data array and break location.
    """
    y1 = rng.standard_normal(100)  # mean = 0
    y2 = rng.standard_normal(100) + 2  # mean = 2
    return np.concatenate([y1, y2]), 100


@pytest.fixture
def data_with_two_breaks(
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating[Any]], list[int]]:
    """Data with two mean shifts.

    Returns
    -------
    tuple[NDArray[np.floating], list[int]]
        Data array and break locations.
    """
    y1 = rng.standard_normal(80)  # mean = 0
    y2 = rng.standard_normal(80) + 2  # mean = 2
    y3 = rng.standard_normal(80) - 1  # mean = -1
    return np.concatenate([y1, y2, y3]), [80, 160]


@pytest.fixture
def ar1_data(rng: np.random.Generator) -> NDArray[np.floating[Any]]:
    """Simulated AR(1) process with phi=0.7."""
    n = 200
    y = np.zeros(n)
    phi = 0.7
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.standard_normal()
    return y


@pytest.fixture
def ar1_data_with_break(
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating[Any]], int]:
    """AR(1) process with structural break in AR coefficient.

    First half: phi = 0.3
    Second half: phi = 0.8

    Returns
    -------
    tuple[NDArray[np.floating], int]
        Data array and break location.
    """
    n = 200
    break_point = 100
    y = np.zeros(n)

    # First regime: phi = 0.3
    for t in range(1, break_point):
        y[t] = 0.3 * y[t - 1] + rng.standard_normal()

    # Second regime: phi = 0.8
    for t in range(break_point, n):
        y[t] = 0.8 * y[t - 1] + rng.standard_normal()

    return y, break_point


@pytest.fixture
def regression_data(
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Simple regression data: y = 1 + 2*x + e.

    Returns
    -------
    tuple[NDArray[np.floating], NDArray[np.floating]]
        y and X arrays.
    """
    n = 100
    x = rng.standard_normal(n)
    e = rng.standard_normal(n)
    y = 1 + 2 * x + e
    X = np.column_stack([np.ones(n), x])
    return y, X


@pytest.fixture
def regression_data_with_break(
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], int]:
    """Regression data with structural break in slope.

    First half: y = 1 + 0.5*x + e
    Second half: y = 1 + 2*x + e

    Returns
    -------
    tuple[NDArray[np.floating], NDArray[np.floating], int]
        y, X arrays, and break location.
    """
    n = 200
    break_point = 100
    x = rng.standard_normal(n)
    e = rng.standard_normal(n) * 0.5

    y = np.zeros(n)
    y[:break_point] = 1 + 0.5 * x[:break_point] + e[:break_point]
    y[break_point:] = 1 + 2.0 * x[break_point:] + e[break_point:]

    X = np.column_stack([np.ones(n), x])
    return y, X, break_point


@pytest.fixture
def adl_data(
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Simulated ADL(1,1) process.

    y_t = 0.5 + 0.6*y_{t-1} + 0.3*x_t + 0.15*x_{t-1} + e_t

    Returns
    -------
    tuple[NDArray[np.floating], NDArray[np.floating]]
        y and x arrays.
    """
    n = 200
    x = rng.standard_normal(n)
    y = np.zeros(n)

    for t in range(1, n):
        y[t] = (
            0.5 + 0.6 * y[t - 1] + 0.3 * x[t] + 0.15 * x[t - 1] + rng.standard_normal()
        )

    return y, x


@pytest.fixture
def adl_data_with_break(
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], int]:
    """ADL process with structural break in AR coefficient.

    First half: y_t = 0.5 + 0.3*y_{t-1} + 0.4*x_t + e_t
    Second half: y_t = 0.5 + 0.8*y_{t-1} + 0.4*x_t + e_t

    Returns
    -------
    tuple[NDArray[np.floating], NDArray[np.floating], int]
        y, x arrays, and break location.
    """
    n = 200
    break_point = 100
    x = rng.standard_normal(n)
    y = np.zeros(n)

    # First regime: phi = 0.3
    for t in range(1, break_point):
        y[t] = 0.5 + 0.3 * y[t - 1] + 0.4 * x[t] + rng.standard_normal()

    # Second regime: phi = 0.8
    for t in range(break_point, n):
        y[t] = 0.5 + 0.8 * y[t - 1] + 0.4 * x[t] + rng.standard_normal()

    return y, x, break_point
