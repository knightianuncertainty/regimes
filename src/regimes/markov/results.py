"""Result classes for Markov regime-switching models.

This module provides result containers for Markov switching models.
MarkovSwitchingResultsBase inherits directly from RegimesResultsBase
(not RegressionResultsBase) because MS models use MLE with normal-distribution
inference rather than OLS with Student's t inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from regimes.results.base import RegimesResultsBase

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


@dataclass(kw_only=True)
class MarkovSwitchingResultsBase(RegimesResultsBase):
    """Base class for all Markov regime-switching model results.

    This class extends RegimesResultsBase with attributes specific to
    Markov switching models: transition matrices, smoothed probabilities,
    and regime-specific parameters.

    Parameters
    ----------
    params : NDArray[np.floating]
        Full parameter vector from MLE estimation.
    nobs : int
        Number of observations.
    k_regimes : int
        Number of regimes.
    regime_transition : NDArray[np.floating]
        Transition probability matrix of shape (k_regimes, k_regimes).
        Entry (i, j) is the probability of transitioning from regime j
        to regime i: P(S_t = i | S_{t-1} = j).
    smoothed_marginal_probabilities : NDArray[np.floating]
        Smoothed probabilities P(S_t = j | Y_1,...,Y_T), shape (nobs, k_regimes).
    filtered_marginal_probabilities : NDArray[np.floating]
        Filtered probabilities P(S_t = j | Y_1,...,Y_t), shape (nobs, k_regimes).
    predicted_marginal_probabilities : NDArray[np.floating]
        Predicted probabilities P(S_t = j | Y_1,...,Y_{t-1}), shape (nobs, k_regimes).
    bse : NDArray[np.floating]
        Standard errors of the parameter estimates.
    llf : float
        Log-likelihood at the optimum.
    resid : NDArray[np.floating]
        Model residuals (probability-weighted).
    fittedvalues : NDArray[np.floating]
        Fitted values (probability-weighted).
    param_names : list[str]
        Names of the parameters.
    regime_params : dict[int, dict[str, float]]
        Per-regime parameters: {0: {"const": 1.5, "sigma2": 0.9}, ...}.
    """

    # Regime structure
    k_regimes: int
    regime_transition: NDArray[np.floating[Any]]

    # Probabilities — all (nobs, k_regimes)
    smoothed_marginal_probabilities: NDArray[np.floating[Any]]
    filtered_marginal_probabilities: NDArray[np.floating[Any]]
    predicted_marginal_probabilities: NDArray[np.floating[Any]]

    # Standard inference
    bse: NDArray[np.floating[Any]]
    llf: float
    resid: NDArray[np.floating[Any]]
    fittedvalues: NDArray[np.floating[Any]]
    param_names: list[str]

    # Per-regime parameters: {0: {"const": 1.5, "sigma2": 0.9}, ...}
    regime_params: dict[int, dict[str, float]]

    # Estimation metadata
    cov_type: str = "approx"
    converged: bool = True
    n_iterations: int = 0

    # Restrictions (None if unrestricted)
    restricted_transitions: dict[tuple[int, int], float] | None = None

    # Back-reference to statsmodels (private)
    _sm_results: Any = field(default=None, repr=False)

    @property
    def df_model(self) -> int:
        """Number of free parameters."""
        return len(self.params)

    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        return -2 * self.llf + 2 * self.df_model

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        return -2 * self.llf + np.log(self.nobs) * self.df_model

    @property
    def hqic(self) -> float:
        """Hannan-Quinn Information Criterion."""
        return -2 * self.llf + 2 * np.log(np.log(self.nobs)) * self.df_model

    @property
    def tvalues(self) -> NDArray[np.floating[Any]]:
        """z-statistics for parameter estimates (normal distribution)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.params / self.bse

    @property
    def pvalues(self) -> NDArray[np.floating[Any]]:
        """Two-sided p-values using normal distribution (MLE inference)."""
        return 2 * (1 - stats.norm.cdf(np.abs(self.tvalues)))

    @property
    def expected_durations(self) -> NDArray[np.floating[Any]]:
        """Expected duration in each regime: 1 / (1 - p_ii).

        Returns
        -------
        NDArray[np.floating]
            Array of shape (k_regimes,) with expected durations.
        """
        diag = np.diag(self.regime_transition)
        with np.errstate(divide="ignore"):
            return 1.0 / (1.0 - diag)

    @property
    def most_likely_regime(self) -> NDArray[np.intp]:
        """Most likely regime at each time point based on smoothed probabilities.

        Returns
        -------
        NDArray[np.intp]
            Array of shape (nobs,) with regime indices (0-based).
        """
        return np.argmax(self.smoothed_marginal_probabilities, axis=1)

    @property
    def regime_assignments(self) -> NDArray[np.intp]:
        """Alias for most_likely_regime."""
        return self.most_likely_regime

    def regime_periods(self) -> list[tuple[int, int, int]]:
        """Identify contiguous regime periods from most likely states.

        Returns
        -------
        list[tuple[int, int, int]]
            List of (regime, start, end) tuples. End is exclusive.
        """
        assignments = self.most_likely_regime
        periods = []
        current_regime = int(assignments[0])
        start = 0

        for t in range(1, len(assignments)):
            if assignments[t] != current_regime:
                periods.append((current_regime, start, t))
                current_regime = int(assignments[t])
                start = t

        periods.append((current_regime, start, len(assignments)))
        return periods

    def conf_int(self, alpha: float = 0.05) -> NDArray[np.floating[Any]]:
        """Confidence intervals using normal distribution.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level. Default gives 95% confidence intervals.

        Returns
        -------
        NDArray[np.floating]
            Array of shape (n_params, 2) with lower and upper bounds.
        """
        q = stats.norm.ppf(1 - alpha / 2)
        lower = self.params - q * self.bse
        upper = self.params + q * self.bse
        return np.column_stack([lower, upper])

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with parameter estimates, standard errors, z-values,
            p-values, and confidence intervals.
        """
        ci = self.conf_int()
        return pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.bse,
                "z": self.tvalues,
                "P>|z|": self.pvalues,
                "ci_lower": ci[:, 0],
                "ci_upper": ci[:, 1],
            },
            index=self.param_names,
        )

    def summary(self) -> str:
        """Generate a text summary of Markov switching results.

        Returns
        -------
        str
            Formatted summary including regime parameters, transition
            matrix, and inference.
        """
        lines = []
        lines.append("=" * 81)
        lines.append(f"{self.model_name:^81}")
        lines.append("=" * 81)
        lines.append(
            f"No. Observations:    {self.nobs:>10}   "
            f"No. Regimes:         {self.k_regimes:>10}"
        )
        lines.append(
            f"Log-Likelihood:    {self.llf:>10.2f}   "
            f"AIC:                 {self.aic:>10.2f}"
        )
        lines.append(
            f"BIC:               {self.bic:>10.2f}   "
            f"HQIC:                {self.hqic:>10.2f}"
        )
        lines.append(
            f"Converged:         {'Yes' if self.converged else 'No':>10}   "
            f"Iterations:          {self.n_iterations:>10}"
        )
        lines.append("=" * 81)

        # Transition matrix
        lines.append("")
        lines.append("Regime Transition Matrix")
        lines.append("  P(S_t = row | S_{t-1} = col)")
        lines.append("-" * 81)

        # Column headers
        header = "         " + "".join(
            f"  Regime {j:>2}" for j in range(self.k_regimes)
        )
        lines.append(header)
        for i in range(self.k_regimes):
            row = f"Regime {i:>2}"
            for j in range(self.k_regimes):
                row += f"  {self.regime_transition[i, j]:>9.4f}"
            lines.append(row)

        # Expected durations
        lines.append("")
        dur_str = ", ".join(
            f"Regime {i}: {d:.1f}" for i, d in enumerate(self.expected_durations)
        )
        lines.append(f"Expected durations: {dur_str}")
        lines.append("")

        # Regime-specific parameters
        lines.append("Regime Parameters")
        lines.append("-" * 81)
        for regime_idx in sorted(self.regime_params.keys()):
            params = self.regime_params[regime_idx]
            lines.append(f"  Regime {regime_idx}:")
            for name, val in params.items():
                lines.append(f"    {name:>20}: {val:>10.4f}")
        lines.append("")

        # Parameter table
        lines.append("=" * 81)
        lines.append(
            f"{'':>25} {'coef':>10} {'std err':>10} {'z':>10} "
            f"{'P>|z|':>10} {'[0.025':>10} {'0.975]':>10}"
        )
        lines.append("-" * 81)

        ci = self.conf_int()
        for i, name in enumerate(self.param_names):
            pval = self.pvalues[i]
            pval_str = f"{pval:.3f}" if pval >= 0.001 else f"{pval:.2e}"
            lines.append(
                f"{name:>25} {self.params[i]:>10.4f} {self.bse[i]:>10.4f} "
                f"{self.tvalues[i]:>10.3f} {pval_str:>10} "
                f"{ci[i, 0]:>10.3f} {ci[i, 1]:>10.3f}"
            )

        lines.append("=" * 81)

        if self.restricted_transitions:
            lines.append("")
            lines.append("Restricted transitions:")
            for (i, j), val in self.restricted_transitions.items():
                lines.append(f"  P({i},{j}) = {val:.4f}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Plot shortcuts (delegate to visualization module)
    # ------------------------------------------------------------------

    def plot_smoothed_probabilities(self, **kwargs: Any) -> tuple[Figure, Any]:
        """Plot smoothed regime probabilities.

        See ``regimes.visualization.markov.plot_smoothed_probabilities``.
        """
        from regimes.visualization.markov import plot_smoothed_probabilities

        return plot_smoothed_probabilities(self, **kwargs)

    def plot_regime_shading(
        self, y: NDArray[np.floating[Any]] | None = None, **kwargs: Any
    ) -> tuple[Figure, Axes]:
        """Plot time series with regime-colored background shading.

        See ``regimes.visualization.markov.plot_regime_shading``.
        """
        from regimes.visualization.markov import plot_regime_shading

        if y is None:
            y = self.fittedvalues + self.resid  # reconstruct endog
        return plot_regime_shading(y, self, **kwargs)

    def plot_transition_matrix(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Plot heatmap of the transition probability matrix.

        See ``regimes.visualization.markov.plot_transition_matrix``.
        """
        from regimes.visualization.markov import plot_transition_matrix

        return plot_transition_matrix(self, **kwargs)

    def plot_parameter_time_series(
        self, param_name: str | None = None, **kwargs: Any
    ) -> tuple[Figure, Any]:
        """Plot regime-dependent parameter over time.

        See ``regimes.visualization.markov.plot_parameter_time_series``.
        """
        from regimes.visualization.markov import plot_parameter_time_series

        return plot_parameter_time_series(self, param_name=param_name, **kwargs)


@dataclass(kw_only=True)
class MarkovRegressionResults(MarkovSwitchingResultsBase):
    """Results from Markov regime-switching regression.

    Additional Attributes
    ---------------------
    switching_trend : bool
        Whether trend parameters switch across regimes.
    switching_exog : bool
        Whether exogenous variable coefficients switch.
    switching_variance : bool
        Whether the error variance switches.
    """

    switching_trend: bool = True
    switching_exog: bool = True
    switching_variance: bool = False
    model_name: str = "Markov Switching Regression"


@dataclass(kw_only=True)
class MarkovARResults(MarkovSwitchingResultsBase):
    """Results from Markov regime-switching autoregression.

    Additional Attributes
    ---------------------
    order : int
        AR order.
    switching_ar : bool
        Whether AR parameters switch across regimes.
    switching_trend : bool
        Whether trend parameters switch.
    switching_variance : bool
        Whether the error variance switches.
    ar_params : dict[int, NDArray[np.floating]]
        AR coefficients per regime.
    """

    order: int = 1
    switching_ar: bool = True
    switching_trend: bool = True
    switching_variance: bool = False
    ar_params: dict[int, NDArray[np.floating[Any]]] | None = None
    model_name: str = "Markov Switching AR"


@dataclass(kw_only=True)
class MarkovADLResults(MarkovSwitchingResultsBase):
    """Results from Markov regime-switching ADL model.

    Additional Attributes
    ---------------------
    ar_order : int
        AR order (p).
    exog_lags : dict[str, list[int]]
        Distributed lag structure per exogenous variable.
    switching_ar : bool
        Whether AR parameters switch.
    switching_exog : bool
        Whether exogenous coefficients switch.
    switching_variance : bool
        Whether the error variance switches.
    """

    ar_order: int = 1
    exog_lags: dict[str, list[int]] = field(default_factory=dict)
    switching_ar: bool = True
    switching_exog: bool = True
    switching_variance: bool = False
    model_name: str = "Markov Switching ADL"
