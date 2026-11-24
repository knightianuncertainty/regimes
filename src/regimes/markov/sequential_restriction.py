"""Tests of restrictions on the transition matrix of a Markov switching model.

This module implements:

1. ``NonRecurringRegimeTest``: H0 non-recurring regimes (a structural break
   model in the sense of Chib, 1998) against H1 an unrestricted transition
   matrix.
2. ``SequentialRestrictionTest``: a general-to-specific search that imposes
   zero restrictions on transition probabilities one at a time.

Both use likelihood ratio statistics. A zero transition probability lies on
the boundary of the parameter space, so the statistics are not chi-squared.
Two reference distributions are offered:

* ``"chi_bar_squared"``: for one restriction the mixture
  ``0.5 chi2(0) + 0.5 chi2(1)``; for ``n`` restrictions the mixture
  ``0.5 chi2(n - 1) + 0.5 chi2(n)``, whose tail probability is an upper
  bound on that of every chi-bar-squared distribution with ``n`` inequality
  restrictions (Perlman, 1969; Kodde and Palm, 1986).
* ``"bootstrap"``: a parametric bootstrap. Samples are drawn from the model
  estimated under the null hypothesis, both models are estimated on each
  sample by the procedure used on the data, and the p-value is
  ``(1 + #{simulated >= observed}) / (B + 1)``.

Status of the theory. The boundary results (Andrews, 2001) are derived for
models in which the information about every parameter grows with the
sample. That is not obviously so here: in a non-recurring model each regime
is left once, so a transition probability is estimated from a single event.
No published result is known to us that gives the distribution of these
statistics, so it is not established that they are chi-bar-squared, or that
the bootstrap is consistent. In the sequential search the statistic that is
tested is, in addition, the smallest among the candidates, which neither
reference distribution accounts for, and the multiple-testing levels have no
proven error-rate property. The tests are best read as descriptive: a small
statistic says that the restriction costs little likelihood.

Estimation. A Markov switching likelihood has local maxima, and a model
that contains another must reach at least its likelihood. Every model is
therefore estimated from several starting values, among them the estimates
of the models nested in it, so that the statistics are not negative.

Regressors are held fixed in the bootstrap, which is appropriate only if
they are strictly exogenous (Qu and Zhuo, 2021, Section 7).

References
----------
- Andrews (2001), "Testing When a Parameter Is on the Boundary of the
  Maintained Hypothesis", Econometrica 69, 683-734.
- Chib (1998), "Estimation and Comparison of Multiple Change-Point Models",
  Journal of Econometrics 86, 221-241.
- Dufour (2006), "Monte Carlo Tests with Nuisance Parameters", Journal of
  Econometrics 133, 443-477.
- Hochberg (1988), "A Sharper Bonferroni Procedure for Multiple Tests of
  Significance", Biometrika 75, 800-802.
- Kodde and Palm (1986), "Wald Criteria for Jointly Testing Equality and
  Inequality Restrictions", Econometrica 54, 1243-1248.
- Perlman (1969), "One-Sided Testing Problems in Multivariate Analysis",
  Annals of Mathematical Statistics 40, 549-567.
- Qu and Zhuo (2021), "Likelihood Ratio-Based Tests for Markov Regime
  Switching", Review of Economic Studies 88, 937-968.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import optimize, stats

from regimes.markov._simulate import (
    MonteCarloPValue,
    RandomState,
    as_generator,
    can_reject,
    monte_carlo_pvalue,
    rejects,
    replications_needed,
    simulate_markov_switching,
    smallest_pvalue,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

_METHODS = ("chi_bar_squared", "bootstrap")
_MODEL_TYPES = ("regression", "ar")
_MULTIPLE_TESTING = ("holm", "bonferroni", "none")
# Two fits of the same optimum differ by rounding in the optimizer. A larger
# model whose log-likelihood is more than this below that of a model nested in
# it was not maximized.
_NESTING_TOLERANCE = 1e-4


@dataclass(kw_only=True)
class RestrictionTestStep:
    """Result of a single restriction test step.

    Attributes
    ----------
    restriction : tuple[int, int]
        The entry ``(i, j)`` of the transition matrix (destination ``i``,
        origin ``j``) that the step is about.
    test_type : str
        "individual": the restriction against the model with the
        restrictions imposed so far. "joint": all restrictions, including
        this one, against the unrestricted model.
    lr_statistic : float
        Likelihood ratio statistic.
    p_value : float
        P-value from the bootstrap or the chi-bar-squared distribution.
    critical_value : float
        Critical value at ``significance``.
    rejected : bool
        Whether H0 (the restrictions hold) was rejected.
    significance : float
        Significance level used, after any adjustment for multiple testing.
    method : str
        "bootstrap" or "chi_bar_squared".
    n_restrictions : int
        Number of restrictions tested.
    """

    restriction: tuple[int, int]
    test_type: str
    lr_statistic: float
    p_value: float
    critical_value: float
    rejected: bool
    significance: float
    method: str
    n_restrictions: int = 1


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
        "chi_bar_squared" or "bootstrap".
    critical_value : float
        The value the statistic must exceed to be rejected at
        ``significance``.
    n_bootstrap : int | None
        Number of bootstrap replications the p-value rests on (None unless
        ``method="bootstrap"``).
    n_bootstrap_failed : int
        Number of bootstrap replications in which a model could not be
        estimated.
    """

    lr_statistic: float
    p_value: float
    rejected: bool
    significance: float
    llf_unrestricted: float
    llf_restricted: float
    n_restrictions: int
    method: str
    critical_value: float = float("nan")
    n_bootstrap: int | None = None
    n_bootstrap_failed: int = 0

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
        lines.append(f"Critical value:       {self.critical_value:>10.4f}")
        lines.append(f"Significance level:   {self.significance:>10.4f}")
        lines.append(f"Method:               {self.method:>10}")
        if self.n_bootstrap is not None:
            lines.append(f"Bootstrap replications: {self.n_bootstrap:>8}")
            if self.n_bootstrap_failed:
                lines.append(f"  not computed:         {self.n_bootstrap_failed:>8}")
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
        Whether the final transition matrix describes non-recurring
        regimes: the regimes can be ordered so that each is left only for
        the next one and the last is never left. The check does not depend
        on how the regimes are labeled.
    initial_probabilities : NDArray[np.floating] | None
        Distribution of the regime before the sample, common to every model
        of the search.
    """

    final_restrictions: dict[tuple[int, int], float]
    final_transition: NDArray[np.floating[Any]]
    history: list[RestrictionTestStep]
    llf_unrestricted: float
    llf_final: float
    is_non_recurring: bool
    initial_probabilities: NDArray[np.floating[Any]] | None = None

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
                f"p={step.p_value:.4f} cv={step.critical_value:.4f} "
                f"at {step.significance:.4f} [{status}]"
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
    """Tail probability ``P(X >= lr_stat)`` of the reference distribution.

    For one restriction on the boundary the distribution is the mixture
    ``0.5 chi2(0) + 0.5 chi2(1)``, where ``chi2(0)`` is a point mass at
    zero; a statistic of zero therefore has p-value one.

    For ``n`` restrictions the weights of the chi-bar-squared mixture
    depend on the information matrix. The value returned is that of
    ``0.5 chi2(n - 1) + 0.5 chi2(n)``, an upper bound on the tail
    probability whatever the weights (Perlman, 1969; Kodde and Palm, 1986).
    If the statistic is chi-bar-squared, the p-value is therefore
    conservative.

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
    """
    if n_restrictions < 1:
        raise ValueError("n_restrictions must be at least 1")
    if lr_stat <= 0.0:
        return 1.0
    if n_restrictions == 1:
        return float(0.5 * stats.chi2.sf(lr_stat, 1))
    return float(
        0.5 * stats.chi2.sf(lr_stat, n_restrictions - 1)
        + 0.5 * stats.chi2.sf(lr_stat, n_restrictions)
    )


def _chi_bar_squared_critical_value(
    significance: float, n_restrictions: int = 1
) -> float:
    """Critical value of the distribution used by ``_chi_bar_squared_pvalue``.

    Parameters
    ----------
    significance : float
        Significance level.
    n_restrictions : int
        Number of restrictions.

    Returns
    -------
    float
        The value ``c`` with tail probability ``significance`` beyond it
        (zero if the significance level is 0.5 or more and ``n = 1``).
    """
    if n_restrictions < 1:
        raise ValueError("n_restrictions must be at least 1")
    if n_restrictions == 1:
        if significance >= 0.5:
            return 0.0
        return float(stats.chi2.ppf(1 - 2 * significance, 1))

    def gap(c: float) -> float:
        return float(
            0.5 * stats.chi2.sf(c, n_restrictions - 1)
            + 0.5 * stats.chi2.sf(c, n_restrictions)
            - significance
        )

    upper = float(stats.chi2.ppf(1 - significance, n_restrictions))
    return float(optimize.brentq(gap, 0.0, upper))


def _warm_start(source: Any) -> NDArray[np.floating[Any]]:
    """Parameters of a fitted model, as a starting value for another model.

    The model started from them has the same parameter layout and differs
    only in its restrictions. Transition probabilities of zero are moved
    just inside the unit interval, because the unconstrained parameters are
    their logarithms; a restricted model sets its own zeros again.
    """
    sm_results = source._sm_results
    model = sm_results.model
    params = np.array(sm_results.params, dtype=np.float64)
    k = int(model.k_regimes)
    matrix = np.asarray(model.regime_transition_matrix(params), dtype=np.float64)
    if matrix.ndim == 3:
        matrix = matrix[..., 0]
    matrix = np.clip(matrix, 1e-6, None)
    matrix /= matrix.sum(axis=0, keepdims=True)
    index = np.arange(model.k_params)[model.parameters["regime_transition"]]
    params[index.reshape(k - 1, k)] = matrix[: k - 1, :]
    return params


def _fit_restricted_model(
    endog: NDArray[Any],
    k_regimes: int,
    model_type: str,
    order: int,
    restrictions: dict[tuple[int, int], float],
    initial_probabilities: NDArray[Any],
    model_kwargs: dict[str, Any],
    search_reps: int,
    rng: np.random.Generator | None = None,
    warm: Sequence[Any] = (),
    time_ordered: bool = False,
) -> Any:
    """Fit a model with the given restrictions and starting distribution.

    The same class is used with and without restrictions, so that two
    models that differ only in their restrictions are nested. ``warm`` holds
    fitted models nested in this one: it is also estimated from their
    estimates, so that its likelihood is at least theirs. The random
    starting values of the search are drawn from ``rng``.
    """
    from regimes.markov.restricted import (
        RestrictedMarkovAR,
        RestrictedMarkovRegression,
    )

    model: Any
    if model_type == "ar":
        model = RestrictedMarkovAR(
            endog,
            k_regimes=k_regimes,
            order=order,
            restrictions=dict(restrictions),
            initial_probabilities=initial_probabilities,
            **model_kwargs,
        )
    else:
        model = RestrictedMarkovRegression(
            endog,
            k_regimes=k_regimes,
            restrictions=dict(restrictions),
            initial_probabilities=initial_probabilities,
            **model_kwargs,
        )
    model._time_ordered = time_ordered
    model._skip_inference = True  # only the likelihood is used
    model._search_rng = rng
    model._extra_starts = [_warm_start(source) for source in warm]
    return model.fit(search_reps=search_reps)


def _lr_statistic(llf_alternative: float, llf_null: float, what: str) -> float:
    """``2 (llf_alternative - llf_null)``.

    The alternative contains the null, so the difference cannot be negative
    when both are maximized. A shortfall beyond rounding means the
    alternative was not maximized; it is reported and the statistic is set
    to zero.
    """
    if not np.isfinite(llf_alternative) or not np.isfinite(llf_null):
        raise RuntimeError(f"{what}: a model could not be estimated")
    difference = 2.0 * (llf_alternative - llf_null)
    if difference < -2.0 * _NESTING_TOLERANCE:
        warnings.warn(
            f"{what}: the larger model has a lower log-likelihood than the "
            f"model nested in it ({llf_alternative:.6f} against "
            f"{llf_null:.6f}); it was not maximized. The statistic is set "
            "to zero",
            UserWarning,
            stacklevel=3,
        )
    return float(max(0.0, difference))


def _check_arguments(
    model_type: str,
    significance: float,
    n_bootstrap: int,
    model_kwargs: dict[str, Any],
) -> None:
    if model_type not in _MODEL_TYPES:
        raise ValueError(
            f"model_type must be one of {_MODEL_TYPES}, got {model_type!r}"
        )
    if not 0.0 < significance < 1.0:
        raise ValueError("significance must be between 0 and 1")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    for name in ("initial_probabilities", "restrictions", "ordering"):
        if name in model_kwargs:
            raise ValueError(f"{name} is set by the test and cannot be passed to it")


def _bootstrap_outcome(outcome: MonteCarloPValue, what: str) -> MonteCarloPValue:
    """Raise if no bootstrap replication could be computed."""
    if outcome.n_valid == 0:
        raise RuntimeError(
            f"{what}: the statistic could not be computed in any bootstrap replication"
        ) from outcome.last_error
    return outcome


class NonRecurringRegimeTest:
    """Test H0: non-recurring (structural break) vs H1: unrestricted MS.

    Computes LR = 2 * (llf_unrestricted - llf_restricted) where the restricted
    model enforces the Chib (1998) structure: from each regime the chain can
    only stay or move to the next one. Both models start in regime 0, so
    that they are nested, and the unrestricted model is also estimated from
    the estimates of the restricted one.

    No published result gives the distribution of this statistic; see the
    module docstring. With ``method="chi_bar_squared"`` the statistic is
    referred to ``0.5 chi2(n - 1) + 0.5 chi2(n)`` for the ``n``
    restrictions; with ``method="bootstrap"`` to its distribution in samples
    drawn from the estimated non-recurring model. When the null is true the
    statistic is typically zero or close to it, and neither version rejects.

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
        Reference distribution: "chi_bar_squared" (default) or "bootstrap".
    n_bootstrap : int
        Number of bootstrap replications (if method="bootstrap"). Default
        199. It must be at least ``1 / significance - 1`` (19 for a test
        at 5 percent), or the test could never reject; the test rejects
        when the p-value is at most ``significance``, which makes its level
        exact when ``significance * (n_bootstrap + 1)`` is an integer
        (Dufour, 2006). Each replication estimates both models, so the
        bootstrap is slow. If the statistic is exactly zero the p-value is
        one whatever samples are drawn, and no replications are run.
    order : int
        AR order (only for model_type="ar").
    random_state : int | numpy.random.Generator | None
        Seed or generator. It governs the bootstrap samples and the random
        starting values of every estimation, so that a given seed
        reproduces the result. With None the result varies from run to run.
    **model_kwargs
        Additional arguments for the MS model. Regressors passed as ``exog``
        are held fixed in the bootstrap and must be strictly exogenous.

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

    # random restarts of the two estimations (lowered in the test suite)
    _search_reps_unrestricted = 5
    _search_reps_restricted = 10

    def __init__(
        self,
        endog: Any,
        k_regimes: int = 2,
        model_type: str = "regression",
        significance: float = 0.05,
        method: str = "chi_bar_squared",
        n_bootstrap: int = 199,
        order: int = 1,
        random_state: RandomState = None,
        **model_kwargs: Any,
    ) -> None:
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
        _check_arguments(model_type, significance, n_bootstrap, model_kwargs)
        if method == "bootstrap" and not can_reject(n_bootstrap, significance):
            raise ValueError(
                f"with n_bootstrap={n_bootstrap} the smallest possible p-value "
                f"is {smallest_pvalue(n_bootstrap):.4f}, so the test could "
                f"never reject at significance={significance}; use at least "
                f"{replications_needed(significance)} replications"
            )
        self.endog = np.asarray(endog, dtype=np.float64)
        self.k_regimes = k_regimes
        self.model_type = model_type
        self.significance = significance
        self.method = method
        self.n_bootstrap = n_bootstrap
        self.order = order
        self.random_state = random_state
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

    def _fit_restricted(
        self,
        endog: NDArray[Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> Any:
        """Fit the restricted (non-recurring) MS model."""
        return _fit_restricted_model(
            self.endog if endog is None else endog,
            self.k_regimes,
            self.model_type,
            self.order,
            self._build_non_recurring_restrictions(),
            np.eye(self.k_regimes)[0],
            self.model_kwargs,
            search_reps=self._search_reps_restricted,
            rng=rng,
            time_ordered=True,
        )

    def _fit_unrestricted(
        self,
        endog: NDArray[Any] | None = None,
        rng: np.random.Generator | None = None,
        restricted: Any = None,
    ) -> Any:
        """Fit the unrestricted MS model.

        The chain starts in regime 0, as in the non-recurring model, so that
        the two models are nested and differ only in the transition matrix.
        (With the stationary starting distribution the unrestricted model
        would not contain the non-recurring one.) If the fitted restricted
        model is given, the estimation is also started from its estimates.
        """
        return _fit_restricted_model(
            self.endog if endog is None else endog,
            self.k_regimes,
            self.model_type,
            self.order,
            {},
            np.eye(self.k_regimes)[0],
            self.model_kwargs,
            search_reps=self._search_reps_unrestricted,
            rng=rng,
            warm=() if restricted is None else (restricted,),
        )

    def _fit_both(
        self, endog: NDArray[Any] | None, rng: np.random.Generator | None
    ) -> tuple[Any, Any]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            restricted = self._fit_restricted(endog, rng)
            unrestricted = self._fit_unrestricted(endog, rng, restricted)
        return unrestricted, restricted

    def _statistic(
        self, endog: NDArray[Any], rng: np.random.Generator | None = None
    ) -> float:
        """The LR statistic on a sample, computed as on the data."""
        unrestricted, restricted = self._fit_both(endog, rng)
        return _lr_statistic(
            float(unrestricted.llf), float(restricted.llf), "non-recurring test"
        )

    def fit(self) -> NonRecurringRegimeTestResults:
        """Run the non-recurring regime test.

        Returns
        -------
        NonRecurringRegimeTestResults
            Test results.

        Raises
        ------
        RuntimeError
            If a model cannot be estimated on the data, or the statistic
            cannot be computed in any bootstrap replication.
        """
        rng = None if self.random_state is None else as_generator(self.random_state)
        unrestricted, restricted = self._fit_both(None, rng)

        llf_u = float(unrestricted.llf)
        llf_r = float(restricted.llf)
        lr_stat = _lr_statistic(llf_u, llf_r, "non-recurring test")

        n_restrictions = len(self._build_non_recurring_restrictions())

        n_bootstrap: int | None = None
        n_failed = 0
        if self.method == "bootstrap":
            if lr_stat == 0.0:
                # Every simulated statistic is at least zero, so the p-value
                # is one whatever samples are drawn: no bootstrap is needed.
                p_value, critical_value, n_bootstrap = 1.0, float("nan"), 0
            else:
                outcome = _bootstrap_outcome(
                    self._bootstrap(restricted, lr_stat, rng), "non-recurring test"
                )
                p_value = outcome.p_value
                critical_value = outcome.critical_value(self.significance)
                n_bootstrap = outcome.n_valid
                n_failed = outcome.n_failed
        else:
            p_value = _chi_bar_squared_pvalue(lr_stat, n_restrictions)
            critical_value = _chi_bar_squared_critical_value(
                self.significance, n_restrictions
            )

        rejected = rejects(p_value, self.significance)

        return NonRecurringRegimeTestResults(
            lr_statistic=lr_stat,
            p_value=p_value,
            rejected=rejected,
            significance=self.significance,
            llf_unrestricted=llf_u,
            llf_restricted=llf_r,
            n_restrictions=n_restrictions,
            method=self.method,
            critical_value=critical_value,
            n_bootstrap=n_bootstrap,
            n_bootstrap_failed=n_failed,
        )

    def _bootstrap(
        self,
        restricted_results: Any,
        observed_lr: float,
        rng: np.random.Generator | None,
    ) -> MonteCarloPValue:
        """Parametric bootstrap of the LR statistic under H0.

        Samples are drawn from the estimated non-recurring model, with the
        regressors and the starting regime of that model, and both models
        are estimated on each sample by the procedure used on the data.
        """
        sm_results = restricted_results._sm_results
        if sm_results is None:
            raise RuntimeError("the fitted model does not hold its statsmodels results")
        sm_model = sm_results.model
        params = np.asarray(sm_results.params, dtype=np.float64)
        generator = as_generator(rng)

        def simulate(draws: np.random.Generator) -> NDArray[np.floating[Any]]:
            return simulate_markov_switching(sm_model, params, draws)

        def statistic(sample: NDArray[np.floating[Any]]) -> float:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self._statistic(sample, generator)

        return monte_carlo_pvalue(
            observed_lr, simulate, statistic, self.n_bootstrap, generator
        )


class SequentialRestrictionTest:
    """General-to-specific search for zero restrictions on the transition matrix.

    Algorithm:

    1. Estimate the unrestricted k-regime model.
    2. For every off-diagonal transition probability not yet restricted,
       estimate the model with that probability set to zero in addition to
       the restrictions imposed so far, and compute the LR statistic against
       the current model.
    3. Take the candidate with the smallest statistic (the least
       significant) and test it: one restriction against the current model.
       If it is rejected, stop: every other candidate has a larger
       statistic.
    4. If two or more restrictions would then be imposed, test all of them
       jointly against the unrestricted model at the unadjusted significance
       level. If that is rejected, stop without imposing the candidate.
       (With one restriction the joint statistic is the individual one, and
       the step is skipped.)
    5. Impose the restriction and return to step 2.

    Every model of the search has the same distribution of the regime before
    the sample, so that each is nested in the previous one: the smoothed
    distribution of the regime of the first observation in a preliminary
    unrestricted fit (which starts from the stationary distribution). Each
    model is also estimated from the estimates of the model it is compared
    with, so that the statistics are not negative.

    No published result gives the distribution of these statistics, and the
    statistic tested at each step is the smallest among the candidates,
    which the reference distributions do not account for; see the module
    docstring. The outcome is best read as a description of which
    transitions the data do not need.

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
        Level of the individual tests. With ``m`` off-diagonal
        probabilities, the test at step ``s = 1, 2, ...`` uses

        - "holm" (default): ``significance / s``. The search runs from the
          least significant restriction upwards; in that direction these
          are the constants of the step-up procedure of Hochberg (1988),
          which are those of Holm. Because the p-values are recomputed at
          each step for the candidate with the smallest statistic, the
          error-rate property of that procedure is not established here.
        - "bonferroni": ``significance / m`` at every step.
        - "none": ``significance`` at every step.
    critical_value_method : str
        "chi_bar_squared" (default) or "bootstrap". With "bootstrap" the
        candidates are still ranked by their LR statistics, and the
        bootstrap is run for the chosen candidate and for the joint test.
        A warning is issued when ``n_bootstrap`` is too small for a test at
        an adjusted level to reject.
    n_bootstrap : int
        Bootstrap replications. Default 199.
    order : int
        AR order (for model_type="ar").
    random_state : int | numpy.random.Generator | None
        Seed or generator. It governs the bootstrap samples and the random
        starting values of every estimation, so that a given seed
        reproduces the result.
    **model_kwargs
        Additional model arguments. Regressors passed as ``exog`` are held
        fixed in the bootstrap and must be strictly exogenous.
    """

    # random restarts of each estimation (lowered in the test suite)
    _search_reps = 5

    def __init__(
        self,
        endog: Any,
        k_regimes: int = 2,
        model_type: str = "regression",
        significance: float = 0.05,
        multiple_testing: str | None = "holm",
        critical_value_method: str = "chi_bar_squared",
        n_bootstrap: int = 199,
        order: int = 1,
        random_state: RandomState = None,
        **model_kwargs: Any,
    ) -> None:
        if multiple_testing is None:
            multiple_testing = "none"
        if multiple_testing not in _MULTIPLE_TESTING:
            raise ValueError(
                f"multiple_testing must be one of {_MULTIPLE_TESTING}, "
                f"got {multiple_testing!r}"
            )
        if critical_value_method not in _METHODS:
            raise ValueError(
                f"critical_value_method must be one of {_METHODS}, "
                f"got {critical_value_method!r}"
            )
        _check_arguments(model_type, significance, n_bootstrap, model_kwargs)
        self.endog = np.asarray(endog, dtype=np.float64)
        self.k_regimes = k_regimes
        self.model_type = model_type
        self.significance = significance
        self.multiple_testing = multiple_testing
        self.critical_value_method = critical_value_method
        self.n_bootstrap = n_bootstrap
        self.order = order
        self.random_state = random_state
        self.model_kwargs = model_kwargs
        self._rng: np.random.Generator | None = None
        self._initial: NDArray[np.floating[Any]] = np.full(k_regimes, 1.0 / k_regimes)

    def _fit_model(
        self,
        restrictions: dict[tuple[int, int], float] | None = None,
        endog: NDArray[Any] | None = None,
        warm: Sequence[Any] = (),
    ) -> Any:
        """Fit the model with the given restrictions (none: unrestricted)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _fit_restricted_model(
                self.endog if endog is None else endog,
                self.k_regimes,
                self.model_type,
                self.order,
                restrictions or {},
                self._initial,
                self.model_kwargs,
                search_reps=self._search_reps,
                rng=self._rng,
                warm=warm,
            )

    def _starting_distribution(self) -> tuple[NDArray[np.floating[Any]], Any]:
        """Distribution of the regime before the sample, and the fit it is from.

        A preliminary unrestricted model is estimated with the stationary
        starting distribution; the smoothed distribution of the regime of
        the first observation is then used, as a fixed vector, in every
        model of the search.
        """
        from regimes.markov.restricted import (
            RestrictedMarkovAR,
            RestrictedMarkovRegression,
        )

        model: Any
        if self.model_type == "ar":
            model = RestrictedMarkovAR(
                self.endog,
                k_regimes=self.k_regimes,
                order=self.order,
                restrictions={},
                **self.model_kwargs,
            )
        else:
            model = RestrictedMarkovRegression(
                self.endog,
                k_regimes=self.k_regimes,
                restrictions={},
                **self.model_kwargs,
            )
        model._search_rng = self._rng
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preliminary = model.fit(search_reps=self._search_reps)
        smoothed = np.asarray(
            preliminary.smoothed_marginal_probabilities, dtype=np.float64
        )
        if smoothed.shape[0] == self.k_regimes and smoothed.shape[1] != self.k_regimes:
            smoothed = smoothed.T
        start = np.clip(smoothed[0], 1e-8, None)
        return start / start.sum(), preliminary

    def _identify_candidates(
        self,
        transition: NDArray[Any],
        existing_restrictions: dict[tuple[int, int], float],
    ) -> list[tuple[int, int]]:
        """Testable off-diagonal entries, sorted by probability ascending."""
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
        """Significance level of the individual test at step ``step`` (from 0).

        The search tests the least significant restriction first. In that
        direction the Holm constants are ``significance / (step + 1)``: the
        largest p-value is compared with ``significance``, the next with
        ``significance / 2``, and so on (Hochberg, 1988).
        """
        if self.multiple_testing == "holm":
            return self.significance / (step + 1)
        if self.multiple_testing == "bonferroni":
            return self.significance / total_candidates
        return self.significance

    def _reference(
        self,
        lr_stat: float,
        n_restrictions: int,
        significance: float,
        null_results: Any,
        null_restrictions: dict[tuple[int, int], float],
        alternative_restrictions: dict[tuple[int, int], float],
    ) -> tuple[float, float]:
        """P-value and critical value of an LR statistic."""
        if self.critical_value_method != "bootstrap":
            return (
                _chi_bar_squared_pvalue(lr_stat, n_restrictions),
                _chi_bar_squared_critical_value(significance, n_restrictions),
            )
        if not np.isfinite(lr_stat):
            return 0.0, float("nan")
        if lr_stat == 0.0:
            # Every simulated statistic is at least zero: the p-value is one.
            return 1.0, float("nan")
        if not can_reject(self.n_bootstrap, significance):
            warnings.warn(
                f"a test at level {significance:.4f} cannot reject with "
                f"n_bootstrap={self.n_bootstrap}: the smallest possible "
                f"p-value is {smallest_pvalue(self.n_bootstrap):.4f}",
                UserWarning,
                stacklevel=3,
            )

        sm_results = null_results._sm_results
        sm_model = sm_results.model
        params = np.asarray(sm_results.params, dtype=np.float64)

        def simulate(rng: np.random.Generator) -> NDArray[np.floating[Any]]:
            return simulate_markov_switching(sm_model, params, rng)

        def statistic(sample: NDArray[np.floating[Any]]) -> float:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                null = self._fit_model(null_restrictions, sample)
                alternative = self._fit_model(
                    alternative_restrictions, sample, warm=(null,)
                )
                return _lr_statistic(
                    float(alternative.llf), float(null.llf), "restriction test"
                )

        outcome = _bootstrap_outcome(
            monte_carlo_pvalue(
                lr_stat, simulate, statistic, self.n_bootstrap, as_generator(self._rng)
            ),
            "restriction test",
        )
        return outcome.p_value, outcome.critical_value(significance)

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
        self._rng = (
            None if self.random_state is None else as_generator(self.random_state)
        )

        # Step 1: the starting distribution, then the unrestricted model
        self._initial, preliminary = self._starting_distribution()
        unrestricted = self._fit_model(warm=(preliminary,))
        if not np.isfinite(unrestricted.llf):
            raise RuntimeError("the unrestricted model could not be estimated")

        restrictions: dict[tuple[int, int], float] = {}
        history: list[RestrictionTestStep] = []
        current = unrestricted

        total_candidates = self.k_regimes * (self.k_regimes - 1)

        step = 0
        while True:
            candidates = self._identify_candidates(
                current.regime_transition, restrictions
            )
            if not candidates:
                break

            # Each candidate is estimated from the current estimates as well
            # as from the search. `candidates` is sorted by estimated
            # probability, which breaks ties between equal statistics.
            trials: list[tuple[tuple[int, int], Any]] = []
            for candidate in candidates:
                try:
                    trial = self._fit_model(
                        {**restrictions, candidate: 0.0}, warm=(current,)
                    )
                except Exception:
                    trial = None
                if trial is not None and not np.isfinite(trial.llf):
                    trial = None
                trials.append((candidate, trial))
            fitted = [(c, t) for c, t in trials if t is not None]
            if not fitted:
                break

            # A model that contains another must reach its likelihood: if a
            # restricted fit is above the model it is nested in, that model
            # is estimated again from the restricted estimates.
            best_trial = max(fitted, key=lambda item: float(item[1].llf))[1]
            if float(best_trial.llf) > float(current.llf) + _NESTING_TOLERANCE:
                current = self._fit_model(restrictions, warm=(best_trial, current))
            if float(current.llf) > float(unrestricted.llf) + _NESTING_TOLERANCE:
                unrestricted = self._fit_model(warm=(current, unrestricted))

            target, restricted = min(
                fitted, key=lambda item: float(current.llf) - float(item[1].llf)
            )
            lr_stat = _lr_statistic(
                float(current.llf), float(restricted.llf), "restriction test"
            )
            new_restrictions = {**restrictions, target: 0.0}

            if verbose:
                prob = current.regime_transition[target[0], target[1]]
                print(f"Step {step + 1}: Testing P{target} = {prob:.4f} -> 0")

            # Individual test: one restriction against the current model
            adj_sig = self._holm_significance(step, total_candidates)
            p_value, cv = self._reference(
                lr_stat, 1, adj_sig, restricted, new_restrictions, restrictions
            )
            individual = RestrictionTestStep(
                restriction=target,
                test_type="individual",
                lr_statistic=lr_stat,
                p_value=p_value,
                critical_value=cv,
                rejected=rejects(p_value, adj_sig),
                significance=adj_sig,
                method=self.critical_value_method,
                n_restrictions=1,
            )
            history.append(individual)

            if individual.rejected:
                if verbose:
                    print(f"  Individual test REJECTED (p={individual.p_value:.4f})")
                break

            if verbose:
                print(f"  Individual test not rejected (p={individual.p_value:.4f})")

            # Joint test of all restrictions against the unrestricted model.
            # With a single restriction it is the same statistic.
            n_restrictions = len(new_restrictions)
            if n_restrictions > 1:
                joint_lr = _lr_statistic(
                    float(unrestricted.llf), float(restricted.llf), "joint test"
                )
                joint_p, joint_cv = self._reference(
                    joint_lr,
                    n_restrictions,
                    self.significance,
                    restricted,
                    new_restrictions,
                    {},
                )
                joint = RestrictionTestStep(
                    restriction=target,
                    test_type="joint",
                    lr_statistic=joint_lr,
                    p_value=joint_p,
                    critical_value=joint_cv,
                    rejected=rejects(joint_p, self.significance),
                    significance=self.significance,
                    method=self.critical_value_method,
                    n_restrictions=n_restrictions,
                )
                history.append(joint)

                if joint.rejected:
                    if verbose:
                        print(f"  Joint test REJECTED (p={joint.p_value:.4f})")
                    break

                if verbose:
                    print(f"  Joint test not rejected (p={joint.p_value:.4f})")

            restrictions = new_restrictions
            current = restricted
            step += 1

        final = current

        return SequentialRestrictionResults(
            final_restrictions=restrictions,
            final_transition=final.regime_transition,
            history=history,
            llf_unrestricted=float(unrestricted.llf),
            llf_final=float(final.llf),
            is_non_recurring=self._check_non_recurring(
                final.regime_transition, restrictions
            ),
            initial_probabilities=self._initial,
        )

    def _check_non_recurring(
        self,
        transition: NDArray[Any],
        restrictions: dict[tuple[int, int], float] | None = None,
    ) -> bool:
        """Whether a transition matrix describes non-recurring regimes.

        True if the regimes can be ordered so that each one is left only for
        the next and the last is never left (Chib, 1998): the moves with
        positive probability form a single path through all regimes. The
        answer does not depend on the labels of the regimes. An entry counts
        as zero if it is restricted to zero or is below 1e-10.
        """
        k = self.k_regimes
        moves = np.asarray(transition, dtype=np.float64) > 1e-10
        for (i, j), value in (restrictions or {}).items():
            if value == 0.0:
                moves[i, j] = False
        np.fill_diagonal(moves, False)
        # moves[i, j]: the chain can move from j to i
        leaving = moves.sum(axis=0)
        entering = moves.sum(axis=1)
        if moves.sum() != k - 1 or np.any(leaving > 1) or np.any(entering > 1):
            return False
        starts = np.flatnonzero(entering == 0)
        if len(starts) != 1:
            return False
        visited = [int(starts[0])]
        while leaving[visited[-1]] == 1:
            successor = int(np.flatnonzero(moves[:, visited[-1]])[0])
            if successor in visited:
                return False
            visited.append(successor)
        return len(visited) == k
