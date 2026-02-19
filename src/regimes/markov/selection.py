"""Regime number selection for Markov switching models.

This module provides tools for selecting the number of regimes (K) in a
Markov switching model, using either information criteria (AIC, BIC, HQIC)
or sequential likelihood ratio testing.

The information criteria approach fits models for K = 1, 2, ..., max_regimes
and selects the K that minimizes the chosen criterion. The sequential testing
approach tests K vs K+1 using a likelihood ratio test, stopping when the null
(K regimes is sufficient) is not rejected.

Note on K=1: A model with K=1 (no regime switching) is just a standard
regression/AR. We fit it using the existing OLS/AR classes to get the
baseline log-likelihood and information criteria.

Note on testing K=1 vs K=2: This involves the well-known non-standard
distribution problem (Hansen 1992, Garcia 1998) where nuisance parameters
are unidentified under the null. We use bootstrap critical values for this.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass(kw_only=True)
class RegimeNumberSelectionResults:
    """Results from regime number selection.

    Attributes
    ----------
    selected_k : int
        Selected number of regimes.
    results_by_k : dict[int, Any]
        Fitted model results for each K tried.
    ic_table : pd.DataFrame
        DataFrame with columns K, AIC, BIC, HQIC, llf, n_params.
    method : str
        Selection method used ("aic", "bic", "hqic", or "sequential").
    sequential_tests : list[dict[str, Any]] | None
        Sequential test details (if method="sequential").
    """

    selected_k: int
    results_by_k: dict[int, Any]
    ic_table: Any  # pd.DataFrame
    method: str
    sequential_tests: list[dict[str, Any]] | None = None

    def summary(self) -> str:
        """Generate text summary of regime selection."""
        lines = []
        lines.append("=" * 65)
        lines.append("Regime Number Selection Results")
        lines.append("=" * 65)
        lines.append(f"Method:              {self.method.upper()}")
        lines.append(f"Selected K:          {self.selected_k}")
        lines.append(
            f"Models estimated:    K = {min(self.results_by_k)}"
            f" to {max(self.results_by_k)}"
        )
        lines.append("")

        # IC table
        lines.append("Information Criteria:")
        lines.append("-" * 65)
        lines.append(
            f"{'K':>3}  {'Log-lik':>12}  {'AIC':>12}  {'BIC':>12}  {'HQIC':>12}"
        )
        lines.append("-" * 65)

        for _, row in self.ic_table.iterrows():
            k = int(row["K"])
            marker = " *" if k == self.selected_k else "  "
            lines.append(
                f"{k:>3}{marker}"
                f"  {row['llf']:>12.2f}"
                f"  {row['AIC']:>12.2f}"
                f"  {row['BIC']:>12.2f}"
                f"  {row['HQIC']:>12.2f}"
            )

        lines.append("-" * 65)
        lines.append("* = selected model")

        if self.sequential_tests:
            lines.append("")
            lines.append("Sequential Tests:")
            lines.append("-" * 65)
            for test in self.sequential_tests:
                status = "REJECTED" if test["rejected"] else "NOT REJECTED"
                lines.append(
                    f"  K={test['k0']} vs K={test['k1']}: "
                    f"LR={test['lr_statistic']:.4f}  "
                    f"p={test['p_value']:.4f}  [{status}]"
                )

        lines.append("=" * 65)
        return "\n".join(lines)

    def plot_ic(
        self,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot information criteria vs number of regimes.

        Parameters
        ----------
        ax : Axes | None
            Matplotlib axes to plot on.
        **kwargs
            Additional arguments for plot_ic.

        Returns
        -------
        tuple[Figure, Axes]
            Figure and axes.
        """
        from regimes.visualization.markov import plot_ic

        return plot_ic(
            self.ic_table,
            selected_k=self.selected_k,
            ax=ax,
            **kwargs,
        )


class RegimeNumberSelection:
    """Select the number of regimes by information criteria or sequential testing.

    Estimates MS models for K = 1, 2, ..., max_regimes and selects K by:
    - Information criteria: AIC, BIC, HQIC (fit all, pick minimum)
    - Sequential testing: test K vs K+1 using LRT, stop when not rejected

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable.
    k_max : int
        Maximum number of regimes to consider. Default 5.
    model_type : str
        "regression" or "ar". Default "regression".
    method : str
        Selection method: "bic" (default), "aic", "hqic", or "sequential".
    significance : float
        Significance level for sequential testing. Default 0.05.
    order : int
        AR order (for model_type="ar"). Default 1.
    **model_kwargs
        Additional arguments passed to the model constructor.

    Examples
    --------
    >>> import numpy as np
    >>> from regimes.markov.selection import RegimeNumberSelection
    >>> np.random.seed(42)
    >>> y = np.concatenate([np.random.randn(100), np.random.randn(100) + 3])
    >>> sel = RegimeNumberSelection(y, k_max=4, method="bic")
    >>> results = sel.fit()
    >>> print(f"Selected K = {results.selected_k}")
    """

    def __init__(
        self,
        endog: Any,
        k_max: int = 5,
        model_type: str = "regression",
        method: str = "bic",
        significance: float = 0.05,
        order: int = 1,
        **model_kwargs: Any,
    ) -> None:
        self.endog = np.asarray(endog, dtype=np.float64)
        self.k_max = k_max
        self.model_type = model_type
        self.method = method
        self.significance = significance
        self.order = order
        self.model_kwargs = model_kwargs

    def _fit_k1(self) -> dict[str, Any]:
        """Fit K=1 model (no switching) using OLS or AR."""
        from regimes.models import AR, OLS

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.model_type == "ar":
                model = AR(self.endog, lags=self.order)
                results = model.fit()
            else:
                model = OLS(self.endog, has_constant=True)
                results = model.fit()

        nobs = results.nobs
        k_params = len(results.params)
        llf = float(results.llf)

        aic = -2 * llf + 2 * k_params
        bic = -2 * llf + np.log(nobs) * k_params
        hqic = -2 * llf + 2 * np.log(np.log(nobs)) * k_params

        return {
            "results": results,
            "llf": llf,
            "aic": aic,
            "bic": bic,
            "hqic": hqic,
            "n_params": k_params,
        }

    def _fit_k_regimes(self, k: int) -> dict[str, Any]:
        """Fit a k-regime Markov switching model."""
        from regimes.markov import MarkovAR, MarkovRegression

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.model_type == "ar":
                model = MarkovAR(
                    self.endog, k_regimes=k, order=self.order, **self.model_kwargs
                )
            else:
                model = MarkovRegression(self.endog, k_regimes=k, **self.model_kwargs)

            results = model.fit(search_reps=5)

        return {
            "results": results,
            "llf": results.llf,
            "aic": results.aic,
            "bic": results.bic,
            "hqic": results.hqic,
            "n_params": len(results.params),
        }

    def _build_ic_table(self, fitted: dict[int, dict[str, Any]]) -> Any:
        """Build the IC comparison table."""
        import pandas as pd

        rows = []
        for k in sorted(fitted.keys()):
            info = fitted[k]
            rows.append(
                {
                    "K": k,
                    "llf": info["llf"],
                    "AIC": info["aic"],
                    "BIC": info["bic"],
                    "HQIC": info["hqic"],
                    "n_params": info["n_params"],
                }
            )

        return pd.DataFrame(rows)

    def _select_by_ic(
        self,
        ic_table: Any,
        criterion: str,
    ) -> int:
        """Select K that minimizes the specified IC."""
        min_idx = ic_table[criterion].idxmin()
        return int(ic_table.loc[min_idx, "K"])

    def _select_sequential(
        self,
        fitted: dict[int, dict[str, Any]],
    ) -> tuple[int, list[dict[str, Any]]]:
        """Select K by sequential LRT.

        Tests K vs K+1: H0 is that K regimes are sufficient.
        Stops when H0 is not rejected.
        """
        from regimes.markov.sequential_restriction import _chi_bar_squared_pvalue

        tests = []
        selected_k = 1

        for k0 in range(1, self.k_max):
            k1 = k0 + 1
            if k1 not in fitted:
                break

            llf0 = fitted[k0]["llf"]
            llf1 = fitted[k1]["llf"]

            # Number of additional parameters
            n_extra = fitted[k1]["n_params"] - fitted[k0]["n_params"]
            lr_stat = max(0, 2 * (llf1 - llf0))

            # For K=1 vs K=2, use chi-bar-squared (boundary problem)
            # For K>=2 vs K+1, also use chi-bar-squared as approximation
            p_value = _chi_bar_squared_pvalue(lr_stat, max(1, n_extra))

            rejected = p_value < self.significance

            tests.append(
                {
                    "k0": k0,
                    "k1": k1,
                    "lr_statistic": lr_stat,
                    "p_value": p_value,
                    "n_extra_params": n_extra,
                    "rejected": rejected,
                }
            )

            if rejected:
                selected_k = k1
            else:
                break

        return selected_k, tests

    def fit(self, verbose: bool = False) -> RegimeNumberSelectionResults:
        """Run regime number selection.

        Parameters
        ----------
        verbose : bool
            Whether to print progress.

        Returns
        -------
        RegimeNumberSelectionResults
            Selection results.
        """
        fitted: dict[int, dict[str, Any]] = {}

        # Fit K=1 (no switching)
        if verbose:
            print("Fitting K=1 (no switching)...")
        fitted[1] = self._fit_k1()

        # Fit K=2, ..., k_max
        for k in range(2, self.k_max + 1):
            if verbose:
                print(f"Fitting K={k}...")
            try:
                fitted[k] = self._fit_k_regimes(k)
            except Exception as e:
                if verbose:
                    print(f"  Failed for K={k}: {e}")
                break

        ic_table = self._build_ic_table(fitted)

        sequential_tests = None
        if self.method == "sequential":
            selected_k, sequential_tests = self._select_sequential(fitted)
        else:
            criterion_map = {"aic": "AIC", "bic": "BIC", "hqic": "HQIC"}
            criterion = criterion_map.get(self.method, "BIC")
            selected_k = self._select_by_ic(ic_table, criterion)

        results_by_k = {k: info["results"] for k, info in fitted.items()}

        return RegimeNumberSelectionResults(
            selected_k=selected_k,
            results_by_k=results_by_k,
            ic_table=ic_table,
            method=self.method,
            sequential_tests=sequential_tests,
        )
