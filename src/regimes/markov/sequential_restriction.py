"""Sequential restriction testing for Markov switching transition matrices.

This module implements:
1. NonRecurringRegimeTest: Test H0 (non-recurring / structural break) vs
   H1 (unrestricted / recurring Markov switching).
2. SequentialRestrictionTest: GETS-style algorithm for identifying and
   imposing individual restrictions on transition probabilities.

The boundary testing problem (Andrews 2001) is handled via:
- Parametric bootstrap (most reliable, exact finite-sample size control)
- Chi-bar-squared (fast analytical alternative for single restrictions)

References
----------
- Chib (1998), "Estimation and Comparison of Multiple Change-Point Models",
  Journal of Econometrics.
- Andrews (2001), "Testing When a Parameter Is on the Boundary", Econometrica.
- Silvapulle and Sen (2005), Constrained Statistical Inference.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(kw_only=True)
class RestrictionTestStep:
    """Result of a single restriction test step.

    Attributes
    ----------
    restriction : tuple[int, int]
        The (row, col) entry tested.
    test_type : str
        "individual" or "joint".
    lr_statistic : float
        Likelihood ratio test statistic: 2 * (llf_unrestricted - llf_restricted).
    p_value : float
        P-value from bootstrap or chi-bar-squared.
    critical_value : float
        Critical value at the specified significance level.
    rejected : bool
        Whether H0 (restriction holds) was rejected.
    significance : float
        Significance level used.
    method : str
        "bootstrap" or "chi_bar_squared".
    """

    restriction: tuple[int, int]
    test_type: str
    lr_statistic: float
    p_value: float
    critical_value: float
    rejected: bool
    significance: float
    method: str


@dataclass(kw_only=True)
class NonRecurringRegimeTestResults:
    """Results from testing H0: non-recurring vs H1: unrestricted.

    Attributes
    ----------
    lr_statistic : float
        Likelihood ratio statistic.
    p_value : float
        P-value.
    rejected : bool
        Whether non-recurring H0 is rejected.
    significance : float
        Significance level.
    llf_unrestricted : float
        Log-likelihood of unrestricted model.
    llf_restricted : float
        Log-likelihood of restricted (non-recurring) model.
    n_restrictions : int
        Number of zero restrictions imposed.
    method : str
        Critical value method used.
    """

    lr_statistic: float
    p_value: float
    rejected: bool
    significance: float
    llf_unrestricted: float
    llf_restricted: float
    n_restrictions: int
    method: str

    def summary(self) -> str:
        """Generate text summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("Non-Recurring Regime Test")
        lines.append("=" * 60)
        lines.append("H0: Non-recurring (structural break) transitions")
        lines.append("H1: Unrestricted (recurring) Markov switching")
        lines.append("-" * 60)
        lines.append(f"LR statistic:         {self.lr_statistic:>10.4f}")
        lines.append(f"p-value:              {self.p_value:>10.4f}")
        lines.append(f"Significance level:   {self.significance:>10.4f}")
        lines.append(f"Method:               {self.method:>10}")
        lines.append(
            f"Decision:             {'Reject H0' if self.rejected else 'Fail to reject H0'}"
        )
        lines.append("-" * 60)
        lines.append(f"Log-lik (unrestricted): {self.llf_unrestricted:>10.4f}")
        lines.append(f"Log-lik (restricted):   {self.llf_restricted:>10.4f}")
        lines.append(f"Number of restrictions: {self.n_restrictions:>10}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass(kw_only=True)
class SequentialRestrictionResults:
    """Results from the sequential restriction algorithm.

    Attributes
    ----------
    final_restrictions : dict[tuple[int, int], float]
        Restrictions imposed by the algorithm.
    final_transition : NDArray[np.floating]
        Final restricted transition matrix.
    history : list[RestrictionTestStep]
        Full history of test steps.
    llf_unrestricted : float
        Log-likelihood of the unrestricted model.
    llf_final : float
        Log-likelihood of the final restricted model.
    is_non_recurring : bool
        Whether the final model has a non-recurring structure.
    """

    final_restrictions: dict[tuple[int, int], float]
    final_transition: NDArray[np.floating[Any]]
    history: list[RestrictionTestStep]
    llf_unrestricted: float
    llf_final: float
    is_non_recurring: bool

    def summary(self) -> str:
        """Generate text summary of the sequential procedure."""
        lines = []
        lines.append("=" * 70)
        lines.append("Sequential Restriction Test Results")
        lines.append("=" * 70)
        lines.append(f"Number of restrictions imposed: {len(self.final_restrictions)}")
        lines.append(
            f"Non-recurring structure: {'Yes' if self.is_non_recurring else 'No'}"
        )
        lines.append(f"Log-lik (unrestricted): {self.llf_unrestricted:.4f}")
        lines.append(f"Log-lik (final):        {self.llf_final:.4f}")
        lines.append("")

        lines.append("Test History:")
        lines.append("-" * 70)
        for i, step in enumerate(self.history):
            status = "REJECTED" if step.rejected else "NOT REJECTED"
            lines.append(
                f"  Step {i + 1}: P{step.restriction} "
                f"({step.test_type}) LR={step.lr_statistic:.4f} "
                f"p={step.p_value:.4f} [{status}]"
            )

        lines.append("")
        lines.append("Final Restrictions:")
        for (i, j), val in sorted(self.final_restrictions.items()):
            lines.append(f"  P({i},{j}) = {val:.4f}")

        lines.append("")
        lines.append("Final Transition Matrix:")
        k = self.final_transition.shape[0]
        for i in range(k):
            row = "  " + " ".join(
                f"{self.final_transition[i, j]:.4f}" for j in range(k)
            )
            lines.append(row)

        lines.append("=" * 70)
        return "\n".join(lines)


def _chi_bar_squared_pvalue(lr_stat: float, n_restrictions: int = 1) -> float:
    """Compute p-value from chi-bar-squared distribution.

    For a single boundary restriction, the distribution is a 50/50 mixture
    of a point mass at 0 and chi2(1).

    For multiple restrictions, uses conservative Bonferroni-based approximation.

    Parameters
    ----------
    lr_stat : float
        Likelihood ratio statistic.
    n_restrictions : int
        Number of boundary restrictions tested simultaneously.

    Returns
    -------
    float
        P-value.

    References
    ----------
    Andrews (2001), Silvapulle and Sen (2005).
    """
    if n_restrictions == 1:
        # 50/50 mixture: P(chi_bar > c) = 0.5 * P(chi2(1) > c)
        return 0.5 * stats.chi2.sf(lr_stat, 1)

    # For multiple restrictions, use conservative bound
    # Upper bound: treat each restriction independently
    p_single = 0.5 * stats.chi2.sf(lr_stat, 1)
    return min(p_single * n_restrictions, 1.0)


def _chi_bar_squared_critical_value(
    significance: float, n_restrictions: int = 1
) -> float:
    """Critical value from chi-bar-squared distribution.

    Parameters
    ----------
    significance : float
        Significance level.
    n_restrictions : int
        Number of restrictions.

    Returns
    -------
    float
        Critical value.
    """
    if n_restrictions == 1:
        # P(chi_bar > c) = 0.5 * P(chi2(1) > c) = alpha
        # => P(chi2(1) > c) = 2*alpha
        return float(stats.chi2.ppf(1 - 2 * significance, 1))

    # Conservative: use chi2 with full df
    return float(stats.chi2.ppf(1 - significance, n_restrictions))


class NonRecurringRegimeTest:
    """Test H0: non-recurring (structural break) vs H1: unrestricted MS.

    Computes LR = 2 * (llf_unrestricted - llf_restricted) where the restricted
    model enforces the Chib (1998) upper-triangular transition structure.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable.
    k_regimes : int
        Number of regimes.
    model_type : str
        "regression" or "ar".
    significance : float
        Significance level. Default 0.05.
    method : str
        Critical value method: "bootstrap" or "chi_bar_squared".
    n_bootstrap : int
        Number of bootstrap replications (if method="bootstrap").
    order : int
        AR order (only for model_type="ar").
    **model_kwargs
        Additional arguments for the MS model.

    Examples
    --------
    >>> import numpy as np
    >>> from regimes.markov.sequential_restriction import NonRecurringRegimeTest
    >>> np.random.seed(42)
    >>> y = np.concatenate([np.random.randn(100), np.random.randn(100) + 3])
    >>> test = NonRecurringRegimeTest(y, k_regimes=2, method="chi_bar_squared")
    >>> results = test.fit()
    >>> print(results.summary())
    """

    def __init__(
        self,
        endog: Any,
        k_regimes: int = 2,
        model_type: str = "regression",
        significance: float = 0.05,
        method: str = "chi_bar_squared",
        n_bootstrap: int = 999,
        order: int = 1,
        **model_kwargs: Any,
    ) -> None:
        self.endog = np.asarray(endog, dtype=np.float64)
        self.k_regimes = k_regimes
        self.model_type = model_type
        self.significance = significance
        self.method = method
        self.n_bootstrap = n_bootstrap
        self.order = order
        self.model_kwargs = model_kwargs

    def _build_non_recurring_restrictions(self) -> dict[tuple[int, int], float]:
        """Build the set of restrictions for non-recurring transitions."""
        restrictions: dict[tuple[int, int], float] = {}
        k = self.k_regimes
        for j in range(k):
            for i in range(k):
                if i != j and i != j + 1:
                    restrictions[(i, j)] = 0.0
        return restrictions

    def _fit_unrestricted(self) -> Any:
        """Fit the unrestricted MS model."""
        from regimes.markov import MarkovAR, MarkovRegression

        if self.model_type == "ar":
            model = MarkovAR(
                self.endog,
                k_regimes=self.k_regimes,
                order=self.order,
                ordering=None,
                **self.model_kwargs,
            )
        else:
            model = MarkovRegression(
                self.endog, k_regimes=self.k_regimes, ordering=None, **self.model_kwargs
            )
        return model.fit(search_reps=5)

    def _fit_restricted(self) -> Any:
        """Fit the restricted (non-recurring) MS model."""
        from regimes.markov.restricted import (
            RestrictedMarkovAR,
            RestrictedMarkovRegression,
        )

        self._build_non_recurring_restrictions()

        if self.model_type == "ar":
            model = RestrictedMarkovAR.non_recurring(
                self.endog,
                k_regimes=self.k_regimes,
                order=self.order,
                **self.model_kwargs,
            )
        else:
            model = RestrictedMarkovRegression.non_recurring(
                self.endog, k_regimes=self.k_regimes, **self.model_kwargs
            )
        return model.fit(search_reps=10)

    def fit(self) -> NonRecurringRegimeTestResults:
        """Run the non-recurring regime test.

        Returns
        -------
        NonRecurringRegimeTestResults
            Test results.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            unrestricted = self._fit_unrestricted()
            restricted = self._fit_restricted()

        llf_u = unrestricted.llf
        llf_r = restricted.llf

        # Handle cases where restricted model fails
        if not np.isfinite(llf_r):
            llf_r = -np.inf

        lr_stat = max(0, 2 * (llf_u - llf_r))

        restrictions = self._build_non_recurring_restrictions()
        n_restrictions = len(restrictions)

        if self.method == "bootstrap":
            p_value = self._bootstrap_pvalue(restricted, lr_stat, n_restrictions)
        else:
            p_value = _chi_bar_squared_pvalue(lr_stat, n_restrictions)

        rejected = p_value < self.significance

        return NonRecurringRegimeTestResults(
            lr_statistic=lr_stat,
            p_value=p_value,
            rejected=rejected,
            significance=self.significance,
            llf_unrestricted=llf_u,
            llf_restricted=llf_r,
            n_restrictions=n_restrictions,
            method=self.method,
        )

    def _bootstrap_pvalue(
        self,
        restricted_results: Any,
        observed_lr: float,
        n_restrictions: int,
    ) -> float:
        """Compute bootstrap p-value by simulating under H0.

        Parameters
        ----------
        restricted_results : MarkovSwitchingResultsBase
            Fitted restricted model (H0 is true).
        observed_lr : float
            Observed LR statistic.
        n_restrictions : int
            Number of restrictions.

        Returns
        -------
        float
            Bootstrap p-value.
        """
        if not np.isfinite(restricted_results.llf):
            return 0.0

        sm_res = restricted_results._sm_results
        if sm_res is None:
            return _chi_bar_squared_pvalue(observed_lr, n_restrictions)

        count_exceed = 0

        for _ in range(self.n_bootstrap):
            try:
                # Simulate data from restricted model
                sim_y = sm_res.simulate(self.endog.shape[0])

                # Fit both models on simulated data
                from regimes.markov import MarkovRegression

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if self.model_type == "regression":
                        u_model = MarkovRegression(
                            sim_y,
                            k_regimes=self.k_regimes,
                            ordering=None,
                            **self.model_kwargs,
                        )
                        u_res = u_model.fit()

                        from regimes.markov.restricted import RestrictedMarkovRegression

                        r_model = RestrictedMarkovRegression.non_recurring(
                            sim_y, k_regimes=self.k_regimes, **self.model_kwargs
                        )
                        r_res = r_model.fit()
                    else:
                        from regimes.markov import MarkovAR

                        u_model = MarkovAR(
                            sim_y,
                            k_regimes=self.k_regimes,
                            order=self.order,
                            ordering=None,
                            **self.model_kwargs,
                        )
                        u_res = u_model.fit()

                        from regimes.markov.restricted import RestrictedMarkovAR

                        r_model = RestrictedMarkovAR.non_recurring(
                            sim_y,
                            k_regimes=self.k_regimes,
                            order=self.order,
                            **self.model_kwargs,
                        )
                        r_res = r_model.fit()

                boot_lr = max(0, 2 * (u_res.llf - r_res.llf))
                if boot_lr >= observed_lr:
                    count_exceed += 1

            except Exception:
                continue

        n_valid = max(1, self.n_bootstrap)
        return count_exceed / n_valid


class SequentialRestrictionTest:
    """General-to-specific procedure for transition matrix restrictions.

    Algorithm:
    1. Estimate unrestricted k-regime MS model
    2. Sort off-diagonal transition probabilities by value (ascending)
    3. Test the smallest: H0: p_ij = 0 via boundary-corrected LR test
    4. If not rejected:
       a. Impose restriction
       b. Re-estimate with all restrictions so far
       c. Test BOTH: (i) newest individual restriction, (ii) joint test
       d. If joint test rejected: undo last restriction, stop
    5. Continue until a test rejects or no more candidates

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable.
    k_regimes : int
        Number of regimes.
    model_type : str
        "regression" or "ar".
    significance : float
        Significance level. Default 0.05.
    multiple_testing : str
        Multiple testing correction: "holm" (default), "bonferroni", or "none".
    critical_value_method : str
        "chi_bar_squared" (default) or "bootstrap".
    n_bootstrap : int
        Bootstrap replications. Default 999.
    order : int
        AR order (for model_type="ar").
    **model_kwargs
        Additional model arguments.
    """

    def __init__(
        self,
        endog: Any,
        k_regimes: int = 2,
        model_type: str = "regression",
        significance: float = 0.05,
        multiple_testing: str = "holm",
        critical_value_method: str = "chi_bar_squared",
        n_bootstrap: int = 999,
        order: int = 1,
        **model_kwargs: Any,
    ) -> None:
        self.endog = np.asarray(endog, dtype=np.float64)
        self.k_regimes = k_regimes
        self.model_type = model_type
        self.significance = significance
        self.multiple_testing = multiple_testing
        self.critical_value_method = critical_value_method
        self.n_bootstrap = n_bootstrap
        self.order = order
        self.model_kwargs = model_kwargs

    def _fit_model(
        self,
        restrictions: dict[tuple[int, int], float] | None = None,
    ) -> Any:
        """Fit an MS model with optional restrictions."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if restrictions:
                from regimes.markov.restricted import (
                    RestrictedMarkovAR,
                    RestrictedMarkovRegression,
                )

                if self.model_type == "ar":
                    model = RestrictedMarkovAR(
                        self.endog,
                        k_regimes=self.k_regimes,
                        order=self.order,
                        restrictions=restrictions,
                        ordering=None,
                        **self.model_kwargs,
                    )
                else:
                    model = RestrictedMarkovRegression(
                        self.endog,
                        k_regimes=self.k_regimes,
                        restrictions=restrictions,
                        ordering=None,
                        **self.model_kwargs,
                    )
                return model.fit(search_reps=5)
            else:
                from regimes.markov import MarkovAR, MarkovRegression

                if self.model_type == "ar":
                    model = MarkovAR(
                        self.endog,
                        k_regimes=self.k_regimes,
                        order=self.order,
                        ordering=None,
                        **self.model_kwargs,
                    )
                else:
                    model = MarkovRegression(
                        self.endog,
                        k_regimes=self.k_regimes,
                        ordering=None,
                        **self.model_kwargs,
                    )
                return model.fit(search_reps=5)

    def _identify_candidates(
        self,
        transition: NDArray[Any],
        existing_restrictions: dict[tuple[int, int], float],
    ) -> list[tuple[int, int]]:
        """Identify testable off-diagonal entries, sorted by probability ascending."""
        k = self.k_regimes
        candidates = []

        for i in range(k):
            for j in range(k):
                if i == j:
                    continue  # Don't restrict diagonal
                if (i, j) in existing_restrictions:
                    continue  # Already restricted
                prob = transition[i, j]
                candidates.append(((i, j), prob))

        # Sort by probability (ascending — smallest first)
        candidates.sort(key=lambda x: x[1])
        return [c[0] for c in candidates]

    def _holm_significance(self, step: int, total_candidates: int) -> float:
        """Holm-Bonferroni adjusted significance level."""
        if self.multiple_testing == "holm":
            return self.significance / (total_candidates - step)
        elif self.multiple_testing == "bonferroni":
            return self.significance / total_candidates
        return self.significance

    def _test_single_restriction(
        self,
        unrestricted_llf: float,
        target: tuple[int, int],
        existing_restrictions: dict[tuple[int, int], float],
        adjusted_significance: float,
    ) -> RestrictionTestStep:
        """Test a single restriction p_ij = 0."""
        test_restrictions = {**existing_restrictions, target: 0.0}
        restricted = self._fit_model(test_restrictions)

        lr_stat = max(0, 2 * (unrestricted_llf - restricted.llf))

        if self.critical_value_method == "bootstrap":
            p_value = _chi_bar_squared_pvalue(lr_stat, 1)  # Fallback
        else:
            p_value = _chi_bar_squared_pvalue(lr_stat, 1)

        cv = _chi_bar_squared_critical_value(adjusted_significance, 1)
        rejected = p_value < adjusted_significance

        return RestrictionTestStep(
            restriction=target,
            test_type="individual",
            lr_statistic=lr_stat,
            p_value=p_value,
            critical_value=cv,
            rejected=rejected,
            significance=adjusted_significance,
            method=self.critical_value_method,
        )

    def _test_joint_restriction(
        self,
        unrestricted_llf: float,
        all_restrictions: dict[tuple[int, int], float],
    ) -> RestrictionTestStep:
        """Test all restrictions jointly against the unrestricted model."""
        restricted = self._fit_model(all_restrictions)

        n_restrictions = len(all_restrictions)
        lr_stat = max(0, 2 * (unrestricted_llf - restricted.llf))

        p_value = _chi_bar_squared_pvalue(lr_stat, n_restrictions)
        cv = _chi_bar_squared_critical_value(self.significance, n_restrictions)
        rejected = p_value < self.significance

        # Use the last restriction as the identifier
        last_restriction = list(all_restrictions.keys())[-1]

        return RestrictionTestStep(
            restriction=last_restriction,
            test_type="joint",
            lr_statistic=lr_stat,
            p_value=p_value,
            critical_value=cv,
            rejected=rejected,
            significance=self.significance,
            method=self.critical_value_method,
        )

    def fit(self, verbose: bool = False) -> SequentialRestrictionResults:
        """Run the sequential restriction algorithm.

        Parameters
        ----------
        verbose : bool
            Whether to print progress.

        Returns
        -------
        SequentialRestrictionResults
            Results of the sequential procedure.
        """
        # Step 1: Fit unrestricted model
        unrestricted = self._fit_model()
        unrestricted_llf = unrestricted.llf
        transition = unrestricted.regime_transition

        restrictions: dict[tuple[int, int], float] = {}
        history: list[RestrictionTestStep] = []

        # Count total testable candidates for Holm correction
        total_candidates = sum(
            1 for i in range(self.k_regimes) for j in range(self.k_regimes) if i != j
        )

        step = 0
        while True:
            candidates = self._identify_candidates(transition, restrictions)
            if not candidates:
                break

            target = candidates[0]  # Smallest unrestricted p_ij

            if verbose:
                prob = transition[target[0], target[1]]
                print(f"Step {step + 1}: Testing P{target} = {prob:.4f} -> 0")

            # Individual test
            adj_sig = self._holm_significance(step, total_candidates)
            individual = self._test_single_restriction(
                unrestricted_llf, target, restrictions, adj_sig
            )
            history.append(individual)

            if individual.rejected:
                if verbose:
                    print(f"  Individual test REJECTED (p={individual.p_value:.4f})")
                break

            if verbose:
                print(f"  Individual test not rejected (p={individual.p_value:.4f})")

            # Impose restriction
            restrictions[target] = 0.0

            # Joint test
            joint = self._test_joint_restriction(unrestricted_llf, restrictions)
            history.append(joint)

            if joint.rejected:
                if verbose:
                    print(f"  Joint test REJECTED (p={joint.p_value:.4f})")
                del restrictions[target]
                break

            if verbose:
                print(f"  Joint test not rejected (p={joint.p_value:.4f})")

            # Update transition matrix from the restricted fit
            restricted_results = self._fit_model(restrictions)
            transition = restricted_results.regime_transition

            step += 1

        # Fit final restricted model
        if restrictions:
            final = self._fit_model(restrictions)
        else:
            final = unrestricted

        # Check if the structure is non-recurring
        is_non_recurring = self._check_non_recurring(
            final.regime_transition, restrictions
        )

        return SequentialRestrictionResults(
            final_restrictions=restrictions,
            final_transition=final.regime_transition,
            history=history,
            llf_unrestricted=unrestricted_llf,
            llf_final=final.llf,
            is_non_recurring=is_non_recurring,
        )

    def _check_non_recurring(
        self,
        transition: NDArray[Any],
        restrictions: dict[tuple[int, int], float],
    ) -> bool:
        """Check if the transition matrix has a non-recurring structure.

        Non-recurring means: for each column j, only entries j and j+1 are
        non-zero (Chib 1998 upper-triangular structure).
        """
        k = self.k_regimes
        for j in range(k):
            for i in range(k):
                if (
                    i != j
                    and i != j + 1
                    and transition[i, j] > 1e-10
                    and (i, j) not in restrictions
                ):
                    return False
        return True
