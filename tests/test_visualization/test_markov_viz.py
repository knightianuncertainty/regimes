"""Tests for Markov switching visualization functions."""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

matplotlib.use("Agg")

from regimes.visualization.markov import (
    plot_ic,
    plot_parameter_time_series,
    plot_regime_shading,
    plot_smoothed_probabilities,
    plot_transition_matrix,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def two_regime_data(rng: np.random.Generator) -> NDArray[np.floating[Any]]:
    y1 = rng.standard_normal(100) + 1.0
    y2 = rng.standard_normal(100) + 4.0
    return np.concatenate([y1, y2])


@pytest.fixture
def ms_results(two_regime_data: NDArray[np.floating[Any]]) -> Any:
    from regimes.markov import MarkovRegression

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MarkovRegression(two_regime_data, k_regimes=2)
        return model.fit(search_reps=3)


@pytest.fixture
def ic_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "K": [1, 2, 3],
            "AIC": [500.0, 400.0, 410.0],
            "BIC": [505.0, 415.0, 435.0],
            "HQIC": [502.0, 405.0, 420.0],
        }
    )


class TestPlotSmoothedProbabilities:
    """Test plot_smoothed_probabilities."""

    def test_basic(self, ms_results: Any) -> None:
        fig, axes = plot_smoothed_probabilities(ms_results)
        assert fig is not None
        assert len(axes) == ms_results.k_regimes
        plt.close(fig)

    def test_custom_title(self, ms_results: Any) -> None:
        fig, axes = plot_smoothed_probabilities(ms_results, title="Custom Title")
        assert fig._suptitle.get_text() == "Custom Title"
        plt.close(fig)

    def test_with_existing_axes(self, ms_results: Any) -> None:
        fig, axes = plt.subplots(2, 1, squeeze=False)
        fig2, axes2 = plot_smoothed_probabilities(ms_results, ax=axes.flatten())
        assert fig2 is fig
        plt.close(fig)

    def test_result_method(self, ms_results: Any) -> None:
        fig, axes = ms_results.plot_smoothed_probabilities()
        assert fig is not None
        plt.close(fig)


class TestPlotRegimeShading:
    """Test plot_regime_shading."""

    def test_basic(
        self,
        two_regime_data: NDArray[np.floating[Any]],
        ms_results: Any,
    ) -> None:
        fig, ax = plot_regime_shading(two_regime_data, ms_results)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_custom_title(
        self,
        two_regime_data: NDArray[np.floating[Any]],
        ms_results: Any,
    ) -> None:
        fig, ax = plot_regime_shading(two_regime_data, ms_results, title="My Title")
        assert ax.get_title() == "My Title"
        plt.close(fig)

    def test_no_legend(
        self,
        two_regime_data: NDArray[np.floating[Any]],
        ms_results: Any,
    ) -> None:
        fig, ax = plot_regime_shading(two_regime_data, ms_results, show_legend=False)
        plt.close(fig)

    def test_with_existing_ax(
        self,
        two_regime_data: NDArray[np.floating[Any]],
        ms_results: Any,
    ) -> None:
        fig, ax = plt.subplots()
        fig2, ax2 = plot_regime_shading(two_regime_data, ms_results, ax=ax)
        assert fig2 is fig
        plt.close(fig)

    def test_result_method(
        self,
        two_regime_data: NDArray[np.floating[Any]],
        ms_results: Any,
    ) -> None:
        fig, ax = ms_results.plot_regime_shading(y=two_regime_data)
        assert fig is not None
        plt.close(fig)


class TestPlotTransitionMatrix:
    """Test plot_transition_matrix."""

    def test_basic(self, ms_results: Any) -> None:
        fig, ax = plot_transition_matrix(ms_results)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_no_annotations(self, ms_results: Any) -> None:
        fig, ax = plot_transition_matrix(ms_results, annotate=False)
        plt.close(fig)

    def test_custom_cmap(self, ms_results: Any) -> None:
        fig, ax = plot_transition_matrix(ms_results, cmap="Reds")
        plt.close(fig)

    def test_result_method(self, ms_results: Any) -> None:
        fig, ax = ms_results.plot_transition_matrix()
        assert fig is not None
        plt.close(fig)


class TestPlotParameterTimeSeries:
    """Test plot_parameter_time_series."""

    def test_basic(self, ms_results: Any) -> None:
        fig, axes = plot_parameter_time_series(ms_results)
        assert fig is not None
        plt.close(fig)

    def test_weighted(self, ms_results: Any) -> None:
        fig, axes = plot_parameter_time_series(ms_results, weighted=True)
        assert fig is not None
        plt.close(fig)

    def test_specific_param(self, ms_results: Any) -> None:
        # Get a param name from the results
        all_params = set()
        for params in ms_results.regime_params.values():
            all_params.update(params.keys())
        if all_params:
            param_name = sorted(all_params)[0]
            fig, axes = plot_parameter_time_series(ms_results, param_name=param_name)
            assert fig is not None
            plt.close(fig)

    def test_no_regime_shading(self, ms_results: Any) -> None:
        fig, axes = plot_parameter_time_series(ms_results, show_regime_shading=False)
        plt.close(fig)

    def test_result_method(self, ms_results: Any) -> None:
        fig, axes = ms_results.plot_parameter_time_series()
        assert fig is not None
        plt.close(fig)


class TestPlotTransitionMatrixRestricted:
    """Test plot_transition_matrix with restricted transitions."""

    def test_with_restricted_transitions(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        """Transition matrix plot should handle restricted transitions."""
        from regimes.markov.restricted import RestrictedMarkovRegression

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = RestrictedMarkovRegression(
                two_regime_data, k_regimes=2, restrictions={(0, 1): 0.0}
            )
            results = model.fit(search_reps=5)

        fig, ax = plot_transition_matrix(results)
        assert fig is not None
        plt.close(fig)

    def test_with_existing_ax(self, ms_results: Any) -> None:
        """Passing an existing ax should reuse the figure."""
        fig, ax = plt.subplots()
        fig2, ax2 = plot_transition_matrix(ms_results, ax=ax)
        assert fig2 is fig
        plt.close(fig)


class TestPlotParameterTimeSeriesEdgeCases:
    """Test edge cases for plot_parameter_time_series."""

    def test_weighted_parameter(self, ms_results: Any) -> None:
        """Weighted=True should produce probability-weighted time series."""
        fig, axes = plot_parameter_time_series(ms_results, weighted=True)
        assert fig is not None
        plt.close(fig)

    def test_custom_figsize(self, ms_results: Any) -> None:
        """Custom figsize should be respected."""
        fig, axes = plot_parameter_time_series(ms_results, figsize=(14, 8))
        assert fig is not None
        plt.close(fig)

    def test_with_existing_axes(self, ms_results: Any) -> None:
        """Providing existing axes should reuse figure."""
        # Get number of params to create matching axes
        all_params = set()
        for params in ms_results.regime_params.values():
            all_params.update(params.keys())
        n_params = len(all_params)
        if n_params > 0:
            fig, axes = plt.subplots(n_params, 1, squeeze=False)
            fig2, axes2 = plot_parameter_time_series(ms_results, ax=axes.flatten())
            assert fig2 is fig
            plt.close(fig)


class TestPlotSmoothedProbabilitiesEdgeCases:
    """Test edge cases for plot_smoothed_probabilities."""

    def test_custom_figsize(self, ms_results: Any) -> None:
        fig, axes = plot_smoothed_probabilities(ms_results, figsize=(14, 6))
        assert fig is not None
        plt.close(fig)

    def test_custom_alpha(self, ms_results: Any) -> None:
        fig, axes = plot_smoothed_probabilities(ms_results, alpha=0.5)
        assert fig is not None
        plt.close(fig)


class TestPlotIC:
    """Test plot_ic."""

    def test_basic(self, ic_table: pd.DataFrame) -> None:
        fig, ax = plot_ic(ic_table)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_with_selected_k(self, ic_table: pd.DataFrame) -> None:
        fig, ax = plot_ic(ic_table, selected_k=2)
        plt.close(fig)

    def test_custom_criteria(self, ic_table: pd.DataFrame) -> None:
        fig, ax = plot_ic(ic_table, criteria=["BIC"])
        plt.close(fig)

    def test_custom_title(self, ic_table: pd.DataFrame) -> None:
        fig, ax = plot_ic(ic_table, title="IC Comparison")
        assert ax.get_title() == "IC Comparison"
        plt.close(fig)

    def test_with_existing_ax(self, ic_table: pd.DataFrame) -> None:
        fig, ax = plt.subplots()
        fig2, ax2 = plot_ic(ic_table, ax=ax)
        assert fig2 is fig
        plt.close(fig)
