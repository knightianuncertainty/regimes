"""Tests for regime number selection."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from regimes.markov.selection import RegimeNumberSelection, RegimeNumberSelectionResults


class TestRegimeNumberSelection:
    """Test RegimeNumberSelection."""

    def test_init(self, two_regime_data: NDArray[np.floating[Any]]) -> None:
        sel = RegimeNumberSelection(two_regime_data, k_max=3, method="bic")
        assert sel.k_max == 3
        assert sel.method == "bic"

    def test_fit_bic(self, two_regime_data: NDArray[np.floating[Any]]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = RegimeNumberSelection(two_regime_data, k_max=3, method="bic")
            results = sel.fit()

        assert isinstance(results, RegimeNumberSelectionResults)
        assert results.selected_k >= 1
        assert results.method == "bic"
        assert len(results.results_by_k) >= 2

    def test_fit_aic(self, two_regime_data: NDArray[np.floating[Any]]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = RegimeNumberSelection(two_regime_data, k_max=3, method="aic")
            results = sel.fit()

        assert results.selected_k >= 1
        assert results.method == "aic"

    def test_fit_hqic(self, two_regime_data: NDArray[np.floating[Any]]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = RegimeNumberSelection(two_regime_data, k_max=3, method="hqic")
            results = sel.fit()

        assert results.selected_k >= 1
        assert results.method == "hqic"

    def test_ic_table(self, two_regime_data: NDArray[np.floating[Any]]) -> None:
        import pandas as pd

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = RegimeNumberSelection(two_regime_data, k_max=3, method="bic")
            results = sel.fit()

        assert isinstance(results.ic_table, pd.DataFrame)
        assert "K" in results.ic_table.columns
        assert "AIC" in results.ic_table.columns
        assert "BIC" in results.ic_table.columns
        assert "HQIC" in results.ic_table.columns
        assert "llf" in results.ic_table.columns

    def test_selects_correct_k_on_well_separated(self) -> None:
        """BIC should select K=2 on data with two well-separated regimes."""
        rng = np.random.default_rng(42)
        y = np.concatenate(
            [
                rng.standard_normal(150) + 0.0,
                rng.standard_normal(150) + 6.0,
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = RegimeNumberSelection(y, k_max=3, method="bic")
            results = sel.fit()

        assert results.selected_k == 2

    def test_sequential_method(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = RegimeNumberSelection(two_regime_data, k_max=3, method="sequential")
            results = sel.fit()

        assert results.selected_k >= 1
        assert results.method == "sequential"
        assert results.sequential_tests is not None
        assert len(results.sequential_tests) > 0

    def test_summary(self, two_regime_data: NDArray[np.floating[Any]]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = RegimeNumberSelection(two_regime_data, k_max=3, method="bic")
            results = sel.fit()

        s = results.summary()
        assert "Regime Number Selection" in s
        assert "BIC" in s

    def test_ar_model_type(self, two_regime_ar_data: NDArray[np.floating[Any]]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = RegimeNumberSelection(
                two_regime_ar_data,
                k_max=3,
                method="bic",
                model_type="ar",
                order=1,
            )
            results = sel.fit()

        assert results.selected_k >= 1


class TestRegimeNumberSelectionVerbose:
    """Test verbose output and edge cases for coverage."""

    def test_fit_verbose(
        self, two_regime_data: NDArray[np.floating[Any]], capsys: Any
    ) -> None:
        """Verbose output should print fitting progress."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = RegimeNumberSelection(two_regime_data, k_max=3, method="bic")
            sel.fit(verbose=True)

        captured = capsys.readouterr()
        assert "K=1" in captured.out
        assert "K=2" in captured.out

    def test_sequential_summary(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        """Summary with sequential tests should include test results."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = RegimeNumberSelection(two_regime_data, k_max=3, method="sequential")
            results = sel.fit()

        s = results.summary()
        assert "Sequential Tests" in s
        # Should have REJECTED or NOT REJECTED
        assert "REJECTED" in s

    def test_ar_model_type_bic(
        self, two_regime_ar_data: NDArray[np.floating[Any]]
    ) -> None:
        """AR model type should work with BIC selection."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = RegimeNumberSelection(
                two_regime_ar_data,
                k_max=3,
                method="bic",
                model_type="ar",
                order=1,
            )
            results = sel.fit()

        assert results.selected_k >= 1
        assert "K" in results.ic_table.columns


class TestRegimeNumberSelectionResults:
    """Test RegimeNumberSelectionResults methods."""

    def test_plot_ic(self, two_regime_data: NDArray[np.floating[Any]]) -> None:
        import matplotlib

        matplotlib.use("Agg")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = RegimeNumberSelection(two_regime_data, k_max=3, method="bic")
            results = sel.fit()

        fig, ax = results.plot_ic()
        assert fig is not None
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close(fig)
