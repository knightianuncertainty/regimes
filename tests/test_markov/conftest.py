"""Fixtures for Markov switching tests."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def two_regime_data(rng: np.random.Generator) -> NDArray[np.floating[Any]]:
    """Data with two clear mean-shifted regimes (100 obs each)."""
    y1 = rng.standard_normal(100) + 1.0
    y2 = rng.standard_normal(100) + 4.0
    return np.concatenate([y1, y2])


@pytest.fixture
def three_regime_data(rng: np.random.Generator) -> NDArray[np.floating[Any]]:
    """Data with three mean-shifted regimes."""
    y1 = rng.standard_normal(80) + 0.0
    y2 = rng.standard_normal(80) + 3.0
    y3 = rng.standard_normal(80) + 6.0
    return np.concatenate([y1, y2, y3])


@pytest.fixture
def two_regime_ar_data(rng: np.random.Generator) -> NDArray[np.floating[Any]]:
    """AR(1) data with two regimes: different AR coefficients."""
    n = 200
    y = np.zeros(n)
    # Regime 1: low persistence
    for t in range(1, 100):
        y[t] = 1.0 + 0.3 * y[t - 1] + rng.standard_normal()
    # Regime 2: high persistence
    for t in range(100, n):
        y[t] = 3.0 + 0.8 * y[t - 1] + rng.standard_normal()
    return y


@pytest.fixture
def two_regime_regression_data(
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Regression data with two regimes: different slopes."""
    n = 200
    x = rng.standard_normal(n)
    y = np.zeros(n)
    # Regime 1: y = 1 + 0.5*x + e
    y[:100] = 1.0 + 0.5 * x[:100] + rng.standard_normal(100) * 0.5
    # Regime 2: y = 3 + 2.0*x + e
    y[100:] = 3.0 + 2.0 * x[100:] + rng.standard_normal(100) * 0.5
    X = np.column_stack([np.ones(n), x])
    return y, X


@pytest.fixture
def ms_regression_results(two_regime_data: NDArray[np.floating[Any]]) -> Any:
    """Fitted MarkovRegression results."""
    from regimes.markov import MarkovRegression

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MarkovRegression(two_regime_data, k_regimes=2)
        return model.fit(search_reps=3)


@pytest.fixture
def ms_ar_results(two_regime_ar_data: NDArray[np.floating[Any]]) -> Any:
    """Fitted MarkovAR results."""
    from regimes.markov import MarkovAR

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MarkovAR(two_regime_ar_data, k_regimes=2, order=1)
        return model.fit(search_reps=3)
