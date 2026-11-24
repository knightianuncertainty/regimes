"""Selection of the number of regimes in a Markov switching model.

Two approaches are offered.

Information criteria (AIC, BIC, HQIC): the models with ``K = 1, ..., k_max``
regimes are estimated and the one with the smallest criterion is chosen.
The consistency of these criteria for the number of regimes has not been
established (Kasahara and Shimotsu, 2018, Section 10.2), and their behavior
depends on the structure of the model (Qu and Zhuo, 2021, Section 7).

Sequential testing: ``K`` regimes are tested against ``K + 1`` with a
likelihood ratio test, for ``K = 1, 2, ...``, until the test does not
reject. The likelihood ratio statistic is not chi-squared: under the null
hypothesis the transition probabilities of the extra regime are not
identified, the score is zero, and parameters lie on the boundary of the
parameter space. Its limiting distribution has been derived (Qu and Zhuo,
2021, for one regime against two; Kasahara and Shimotsu, 2018, for ``K``
against ``K + 1``), but it depends on the model, on which parameters switch
and on the regressors, so it cannot be tabulated. The test here is the
parametric bootstrap of Kasahara and Shimotsu (2018, Section 9):

1. Estimate the models with ``K`` and ``K + 1`` regimes and compute the
   likelihood ratio statistic.
2. Draw samples from the estimated ``K``-regime model, conditional on the
   regressors and the initial observations.
3. On each sample, estimate the models with ``1, ..., K + 1`` regimes by
   the procedure used on the data and compute the statistic.
4. The p-value is ``(1 + #{simulated >= observed}) / (B + 1)`` and the test
   rejects when it is at most the level: the Monte Carlo test of Dufour
   (2006), which Rodriguez-Rondon and Dufour (2025) use for this
   statistic. (Kasahara and Shimotsu use ``#{simulated > observed} / B``.)

What supports the test, and where the support ends:

* Theory. Kasahara and Shimotsu prove that the bootstrap critical values
  are consistent for one regime against two (their Proposition 21) and
  state that the argument extends to more regimes. Their framework covers
  models in which the distribution of ``y_t`` depends on the current
  regime only, lagged dependent variables entering as regressors with a
  switching intercept. It excludes models in which the distribution
  depends on lagged regimes as well (their Assumption 1 and footnote 5),
  which is the case of the mean-adjusted autoregression that
  ``model_type="ar"`` estimates. Qu and Zhuo derive the limiting
  distribution for one regime against two; they do not prove a bootstrap
  result.
* Simulations. Kasahara and Shimotsu (Table 1; n = 200 and 500; B = 199;
  an autoregression of order one with a switching intercept, with and
  without a switching variance) report rejection rates of 4.0 to 5.3
  percent at a nominal 5 for one regime against two, and 3.9 to 7.2
  percent for two against three. For the mean-adjusted autoregression the
  evidence is the simulations of Rodriguez-Rondon and Dufour, where the
  Monte Carlo version of the test has rejection rates close to the nominal
  level (switching mean, common autoregressive coefficient, no regressors).
  No simulation in these papers has switching autoregressive coefficients
  or exogenous regressors.
* Regime-specific variances. The likelihood is unbounded as a regime
  variance tends to zero. As Kasahara and Shimotsu do in their simulations
  (Section 10.1), the models are estimated with the penalty
  ``-a_n sum_j {s2 / s2_j + log(s2_j / s2) - 1}``, ``a_n = 20 / sqrt(n)``,
  where ``s2`` is the error variance of the one-regime model, and the
  statistic is computed from the log-likelihood without the penalty. The
  penalty also keeps the variances far above their lower bound of
  ``(0.01)^2 s2``.
* Transition probabilities. The theory requires them to be bounded away
  from zero and one. No bound is imposed here; a regime that is (almost)
  never left is a sign that the condition fails.
* Regressors are held fixed, which is appropriate only if they are
  strictly exogenous. With a fixed regressor that depends on past values
  of the dependent variable the bootstrap is inconsistent: Qu and Zhuo
  (2021, Section 7) report 10 percent rejections at a nominal 5. The lags
  of ``model_type="ar"`` are generated recursively in the bootstrap, not
  held fixed.
* The search. The likelihood of a model with three or more regimes has
  several local maxima, and the statistic is the result of a finite
  search: the default starting value, ``search_reps`` random ones, and
  for each regime of the smaller model its estimates with that regime
  split in two (once exactly, so that the larger model cannot end below
  the smaller one, and once with the halves moved apart). The same search
  is run on the data and on every bootstrap sample. For one regime against
  two the result is stable. For two against three it can depend on
  ``random_state``: on the reference series of the test suite the
  statistic is 2.5 or 2.9 depending on the seed. Raise ``search_reps`` if
  the result changes with the seed.
* Cost. Each replication estimates every model up to ``K + 1`` regimes
  from all starting values. With 200 observations and the defaults, one
  regime against two takes on the order of 15 minutes and two against
  three a few hours.

References
----------
- Dufour (2006), "Monte Carlo Tests with Nuisance Parameters", Journal of
  Econometrics 133, 443-477.
- Kasahara and Shimotsu (2018), "Testing the Number of Regimes in Markov
  Regime Switching Models", arXiv:1801.06862.
- Qu and Zhuo (2021), "Likelihood Ratio-Based Tests for Markov Regime
  Switching", Review of Economic Studies 88, 937-968.
- Rodriguez-Rondon and Dufour (2025), "Monte Carlo Likelihood Ratio Tests
  for Markov Switching Models", working paper (Bank of Canada Staff
  Working Paper 2026-23; implemented in the R package MSTest, which
  differs in details: one starting value per sample, simulation with a
  burn-in period).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import optimize

from regimes.markov._simulate import (
    RandomState,
    as_generator,
    can_reject,
    monte_carlo_pvalue,
    rejects,
    replications_needed,
    simulate_linear,
    simulate_markov_switching,
    smallest_pvalue,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

_METHODS = ("aic", "bic", "hqic", "sequential")
_MODEL_TYPES = ("regression", "ar")


@dataclass(kw_only=True)
class OneRegimeResults:
    """Estimates of the model without regime switching (``K = 1``).

    The model is the one-regime version of the switching models it is
    compared with: the same trend and regressors and, for an
    autoregression, the same mean-adjusted form
    ``y_t - x_t'b = sum_i phi_i (y_{t-i} - x_{t-i}'b) + e_t``, estimated by
    maximum likelihood conditional on the first ``order`` observations.

    Attributes
    ----------
    params : NDArray
        Coefficients of the trend terms and regressors.
    param_names : list[str]
        Names of ``params``.
    ar_params : NDArray
        Autoregressive coefficients (empty for a regression).
    sigma2 : float
        Error variance (sum of squared residuals over ``nobs``).
    llf : float
        Log-likelihood.
    nobs : int
        Number of observations in the likelihood (the sample less the first
        ``order`` observations).
    df_model : int
        Number of estimated parameters, the variance included.
    """

    params: NDArray[np.floating[Any]]
    param_names: list[str]
    ar_params: NDArray[np.floating[Any]]
    sigma2: float
    llf: float
    nobs: int
    df_model: int
    k_regimes: int = 1
    _exog: NDArray[np.floating[Any]] | None = field(default=None, repr=False)
    _endog: NDArray[np.floating[Any]] | None = field(default=None, repr=False)

    @property
    def aic(self) -> float:
        """Akaike information criterion."""
        return float(-2 * self.llf + 2 * self.df_model)

    @property
    def bic(self) -> float:
        """Bayesian information criterion."""
        return float(-2 * self.llf + np.log(self.nobs) * self.df_model)

    @property
    def hqic(self) -> float:
        """Hannan-Quinn information criterion."""
        return float(-2 * self.llf + 2 * np.log(np.log(self.nobs)) * self.df_model)


def _lr(
    llf_larger: float, llf_smaller: float, k0: int, penalized: bool = False
) -> float:
    """``2 (llf_larger - llf_smaller)``, at least zero.

    The larger model contains the smaller one, so a shortfall beyond
    rounding means that it was not maximized; that is reported. With a
    variance penalty (``penalized``) the estimates maximize the penalized
    likelihood, the plain log-likelihoods need not be ordered, and a
    shortfall is not reported.
    """
    difference = 2.0 * (llf_larger - llf_smaller)
    if difference < -2e-4 and not penalized:
        warnings.warn(
            f"the model with {k0 + 1} regimes has a lower log-likelihood "
            f"({llf_larger:.6f}) than the model with {k0} ({llf_smaller:.6f}), "
            "which it contains: it was not maximized. The statistic is set to "
            "zero",
            UserWarning,
            stacklevel=3,
        )
    return float(max(0.0, difference))


# Kasahara and Shimotsu (2018, Section 10.1): the lower bound on a regime's
# standard deviation relative to that of the one-regime model, and the
# constant of the penalty a_n = 20 / sqrt(n).
_SIGMA_BOUND = 0.01
_PENALTY_CONSTANT = 20.0


def _variance_penalty(sm_model: Any, sigma2_one_regime: float) -> Any:
    """Penalty on small regime variances, or None with a common variance.

    ``-a_n sum_j {s2 / s2_j + log(s2_j / s2) - 1}`` with ``a_n = 20 /
    sqrt(n)`` and ``s2`` the error variance of the one-regime model
    (Kasahara and Shimotsu, 2018, Section 10.1). It is zero when every
    regime has the variance of the one-regime model and tends to minus
    infinity as a regime variance tends to zero, where the likelihood of a
    model with regime-specific variances is unbounded. A variance below the
    paper's lower bound, ``(0.01)^2 s2``, costs about ``10^4 a_n``.
    """
    positions = np.atleast_1d(
        np.arange(sm_model.k_params)[sm_model.parameters["variance"]]
    )
    if len(positions) < 2:
        return None
    a_n = _PENALTY_CONSTANT / np.sqrt(float(sm_model.nobs))
    s2 = float(sigma2_one_regime)

    def penalty(params: Any) -> Any:
        ratio = s2 / params[positions]
        return -a_n * np.sum(ratio - np.log(ratio) - 1.0)

    return penalty


def _gaussian_llf(ssr: float, nobs: int) -> float:
    """Gaussian log-likelihood at ``sigma2 = ssr / nobs``."""
    return float(-0.5 * nobs * (np.log(2 * np.pi * ssr / nobs) + 1.0))


def _ar_residuals(
    y: NDArray[Any],
    exog: NDArray[Any] | None,
    beta: NDArray[Any],
    phi: NDArray[Any],
) -> NDArray[Any]:
    """Residuals of the mean-adjusted autoregression."""
    order = len(phi)
    nobs = len(y)
    u = y - exog @ beta if exog is not None else y
    resid = u[order:].copy()
    for i in range(1, order + 1):
        resid -= phi[i - 1] * u[order - i : nobs - i]
    return np.asarray(resid)


def _fit_one_regime(
    y: NDArray[Any],
    exog: NDArray[Any] | None,
    order: int,
    param_names: list[str],
) -> OneRegimeResults:
    """Maximum likelihood estimates of the one-regime model.

    For ``order = 0`` this is least squares. For an autoregression the sum
    of squared residuals is minimized over the mean coefficients and the
    autoregressive coefficients jointly: the two linear regressions are
    alternated until the sum of squares stops falling, and the result is
    polished by a Gauss-Newton step.
    """
    nobs = len(y)
    k_exog = 0 if exog is None else exog.shape[1]
    if order == 0:
        if exog is None:
            beta = np.zeros(0)
            resid = y
        else:
            beta, *_ = np.linalg.lstsq(exog, y, rcond=None)
            resid = y - exog @ beta
        ssr = float(resid @ resid)
        return OneRegimeResults(
            params=beta,
            param_names=param_names,
            ar_params=np.zeros(0),
            sigma2=ssr / nobs,
            llf=_gaussian_llf(ssr, nobs),
            nobs=nobs,
            df_model=k_exog + 1,
            _exog=exog,
            _endog=y,
        )

    n_eff = nobs - order
    if n_eff <= k_exog + order:
        raise ValueError("too few observations for the one-regime autoregression")
    if exog is None:
        beta = np.zeros(0)
    else:
        beta, *_ = np.linalg.lstsq(exog, y, rcond=None)
    phi = np.zeros(order)
    ssr_old = np.inf
    for _ in range(1000):
        u = y - exog @ beta if exog is not None else y
        lags = np.column_stack([u[order - i : nobs - i] for i in range(1, order + 1)])
        phi, *_ = np.linalg.lstsq(lags, u[order:], rcond=None)
        if exog is not None:
            y_star = y[order:].copy()
            x_star = exog[order:].copy()
            for i in range(1, order + 1):
                y_star -= phi[i - 1] * y[order - i : nobs - i]
                x_star -= phi[i - 1] * exog[order - i : nobs - i]
            beta, *_ = np.linalg.lstsq(x_star, y_star, rcond=None)
        resid = _ar_residuals(y, exog, beta, phi)
        ssr = float(resid @ resid)
        if ssr_old - ssr <= 1e-13 * max(1.0, ssr):
            break
        ssr_old = ssr

    if exog is not None:
        start = np.concatenate([beta, phi])
        polished = optimize.least_squares(
            lambda theta: _ar_residuals(y, exog, theta[:k_exog], theta[k_exog:]),
            start,
            method="lm",
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
        )
        if polished.success and 2 * polished.cost <= ssr:
            beta, phi = polished.x[:k_exog], polished.x[k_exog:]
            ssr = float(2 * polished.cost)

    return OneRegimeResults(
        params=np.asarray(beta, dtype=np.float64),
        param_names=param_names,
        ar_params=np.asarray(phi, dtype=np.float64),
        sigma2=ssr / n_eff,
        llf=_gaussian_llf(ssr, n_eff),
        nobs=n_eff,
        df_model=k_exog + order + 1,
        _exog=exog,
        _endog=y,
    )


def _switching_positions(sm_model: Any, name: str) -> NDArray[np.intp]:
    """Positions, within a regime's block of ``name`` parameters, that switch."""
    first = np.atleast_1d(np.arange(sm_model.k_params)[sm_model.parameters[0, name]])
    second = np.atleast_1d(np.arange(sm_model.k_params)[sm_model.parameters[1, name]])
    return np.flatnonzero(first != second)


def _split_starts(sm_model: Any, lower: Any) -> list[NDArray[np.floating[Any]]]:
    """Starting values for ``K + 1`` regimes from the estimates with ``K``.

    Splitting one regime of the smaller model into two regimes with the same
    parameters gives a model with ``K + 1`` regimes and the same likelihood.
    For each regime of the smaller model two starting values are returned:
    the exact split, from which the optimizer cannot end below the smaller
    model, and a split in which the two halves are moved apart (the first
    switching coefficient by half an error standard deviation in each
    direction, or the variances by a factor of two), which is not a
    stationary point.

    Parameters
    ----------
    sm_model : statsmodels model
        The model with ``K + 1`` regimes that is about to be estimated.
    lower : OneRegimeResults or a fitted Markov switching results object
        The fitted model with ``K`` regimes.
    """
    k1 = int(sm_model.k_regimes)
    k0 = k1 - 1
    order = int(getattr(sm_model, "order", 0))
    names = ["exog"] if sm_model._k_exog > 0 else []
    names.append("variance")
    if order > 0:
        names.append("autoregressive")

    def block(source_regime: int, name: str) -> NDArray[np.floating[Any]]:
        if isinstance(lower, OneRegimeResults):
            values = {
                "exog": lower.params,
                "variance": np.array([lower.sigma2]),
                "autoregressive": lower.ar_params,
            }[name]
            return np.atleast_1d(np.asarray(values, dtype=np.float64))
        model0 = lower._sm_results.model
        params0 = np.asarray(lower._sm_results.params, dtype=np.float64)
        return np.atleast_1d(params0[model0.parameters[source_regime, name]])

    if isinstance(lower, OneRegimeResults):
        transition0 = np.ones((1, 1))
    else:
        model0 = lower._sm_results.model
        transition0 = np.asarray(
            model0.regime_transition_matrix(np.asarray(lower._sm_results.params)),
            dtype=np.float64,
        )
        if transition0.ndim == 3:
            transition0 = transition0[..., 0]
        # zero probabilities cannot be mapped to the optimizer's parameters
        transition0 = np.clip(transition0, 1e-10, None)
        transition0 /= transition0.sum(axis=0, keepdims=True)

    index = np.arange(sm_model.k_params)[
        sm_model.parameters["regime_transition"]
    ].reshape(k0, k1)
    starts = []
    for split in range(k0):
        source = [*range(k0), split]  # regime of the smaller model behind each new one
        share = np.ones(k1)
        share[[split, k0]] = (
            0.5  # the probability of entering the split regime is halved
        )
        transition = np.empty((k1, k1))
        for a in range(k1):
            for b in range(k1):
                transition[a, b] = transition0[source[a], source[b]] * share[a]
        if k0 == 1:
            # one regime: any chain gives the same likelihood; take a persistent one
            transition = np.array([[0.9, 0.1], [0.1, 0.9]])

        exact = np.zeros(sm_model.k_params)
        exact[index] = transition[:k0, :]
        for a in range(k1):
            for name in names:
                exact[sm_model.parameters[a, name]] = block(source[a], name)
        starts.append(exact)

        apart = exact.copy()
        sigma = float(np.sqrt(np.mean(block(split, "variance"))))
        moved = False
        if "exog" in names:
            switching = _switching_positions(sm_model, "exog")
            if len(switching) > 0:
                for a, sign in ((split, -1.0), (k0, 1.0)):
                    position = np.atleast_1d(
                        np.arange(sm_model.k_params)[sm_model.parameters[a, "exog"]]
                    )[switching[0]]
                    apart[position] += sign * 0.5 * sigma
                moved = True
        if not moved and len(_switching_positions(sm_model, "variance")) > 0:
            for a, factor in ((split, 0.5), (k0, 2.0)):
                apart[sm_model.parameters[a, "variance"]] *= factor
            moved = True
        if moved:
            starts.append(apart)
    return starts


@dataclass(kw_only=True)
class RegimeNumberSelectionResults:
    """Results from regime number selection.

    Attributes
    ----------
    selected_k : int
        Selected number of regimes.
    results_by_k : dict[int, Any]
        Fitted model results for each K tried. ``results_by_k[1]`` is a
        ``OneRegimeResults``.
    ic_table : pd.DataFrame
        DataFrame with columns K, llf, AIC, BIC, HQIC, n_params. The
        parameter counts include the error variance(s) and, for ``K >= 2``,
        the transition probabilities.
    method : str
        Selection method used ("aic", "bic", "hqic", or "sequential").
    sequential_tests : list[dict[str, Any]] | None
        One entry per bootstrap test (if method="sequential") with the keys
        ``k0``, ``k1``, ``lr_statistic``, ``p_value``, ``critical_value``,
        ``rejected``, ``n_bootstrap`` (replications the p-value rests on),
        ``n_bootstrap_failed`` and ``n_extra_params``.
    notes : list[str]
        Anything that qualifies the selection: a model that could not be
        estimated, or a selected K equal to ``k_max``, which was not tested
        against a larger model. Printed by ``summary()``.
    """

    selected_k: int
    results_by_k: dict[int, Any]
    ic_table: Any  # pd.DataFrame
    method: str
    sequential_tests: list[dict[str, Any]] | None = None
    notes: list[str] = field(default_factory=list)

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
            lines.append("Sequential Tests (parametric bootstrap):")
            lines.append("-" * 65)
            for test in self.sequential_tests:
                status = "REJECTED" if test["rejected"] else "NOT REJECTED"
                lines.append(
                    f"  K={test['k0']} vs K={test['k1']}: "
                    f"LR={test['lr_statistic']:.4f}  "
                    f"p={test['p_value']:.4f}  "
                    f"B={test['n_bootstrap']}  [{status}]"
                )
                if test["n_bootstrap_failed"]:
                    lines.append(
                        f"    ({test['n_bootstrap_failed']} replications "
                        "could not be computed)"
                    )

        for note in self.notes:
            lines.append("")
            lines.append(f"Note: {note}")

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

    Estimates MS models for K = 1, 2, ..., k_max and selects K by:

    - Information criteria: AIC, BIC, HQIC (fit all, pick minimum).
    - Sequential testing: test K against K + 1 with a parametric bootstrap
      likelihood ratio test, and stop when the test does not reject.

    See the module docstring for the test, its sources and its conditions.

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
    n_bootstrap : int
        Bootstrap replications of each sequential test. Default 199; at
        least ``1 / significance - 1``. Each replication estimates every
        model up to ``K + 1`` regimes: expect minutes for one regime
        against two and hours for two against three with a few hundred
        observations.
    random_state : int | numpy.random.Generator | None
        Seed or generator. It governs the bootstrap samples and the random
        starting values of every estimation, so that a given seed
        reproduces the result.
    search_reps : int | None
        Random starting values of each Markov switching estimation, in
        addition to the default one and those built from the smaller model.
        Default 5. With three or more regimes the likelihood has several
        local maxima; if the result changes with ``random_state``, raise
        this number.
    **model_kwargs
        Additional arguments passed to the model constructor (``exog``,
        ``trend``, ``switching_variance``, ...). The model with one regime
        has the same trend and regressors. Regressors are held fixed in the
        bootstrap and must be strictly exogenous.

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

    # random restarts of each Markov switching estimation (lowered in tests)
    _search_reps = 5

    def __init__(
        self,
        endog: Any,
        k_max: int = 5,
        model_type: str = "regression",
        method: str = "bic",
        significance: float = 0.05,
        order: int = 1,
        n_bootstrap: int = 199,
        random_state: RandomState = None,
        search_reps: int | None = None,
        **model_kwargs: Any,
    ) -> None:
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
        if model_type not in _MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {_MODEL_TYPES}, got {model_type!r}"
            )
        if k_max < 1:
            raise ValueError("k_max must be at least 1")
        if not 0.0 < significance < 1.0:
            raise ValueError("significance must be between 0 and 1")
        if n_bootstrap < 1:
            raise ValueError("n_bootstrap must be at least 1")
        if method == "sequential" and not can_reject(n_bootstrap, significance):
            raise ValueError(
                f"with n_bootstrap={n_bootstrap} the smallest possible p-value "
                f"is {smallest_pvalue(n_bootstrap):.4f}, so the tests could "
                f"never reject at significance={significance}; use at least "
                f"{replications_needed(significance)} replications"
            )
        if model_type == "ar" and (int(order) != order or order < 1):
            raise ValueError('order must be a positive integer for model_type="ar"')
        if search_reps is not None and search_reps < 0:
            raise ValueError("search_reps must not be negative")
        if search_reps is not None:
            self._search_reps = int(search_reps)
        self.endog = np.asarray(endog, dtype=np.float64).ravel()
        self.k_max = k_max
        self.model_type = model_type
        self.method = method
        self.significance = significance
        self.order = order if model_type == "ar" else 0
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.model_kwargs = model_kwargs
        self._rng: np.random.Generator | None = None
        # a misspelled or unsupported model argument is an error here, not
        # a model that silently cannot be estimated later
        self._build_model(2, self.endog)

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def _one_regime_design(
        self, endog: NDArray[Any]
    ) -> tuple[NDArray[Any] | None, list[str]]:
        """Trend terms and regressors of the switching models, and names.

        The design is taken from the statsmodels model that the switching
        models are built on, so that the one-regime model is nested in them
        whatever the trend specification.
        """
        from statsmodels.tsa.regime_switching.markov_regression import (
            MarkovRegression as SMMarkovRegression,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm_model = SMMarkovRegression(
                endog,
                k_regimes=2,
                trend=self.model_kwargs.get("trend", "c"),
                exog=self.model_kwargs.get("exog"),
            )
        if sm_model._k_exog == 0:
            return None, []
        exog = np.asarray(sm_model.exog, dtype=np.float64).reshape(len(endog), -1)
        k_trend = int(sm_model.k_trend)
        trend_names = {0: [], 1: ["trend"], 2: ["const", "trend"]}[k_trend]
        if k_trend == 1 and np.allclose(exog[:, 0], 1.0):
            trend_names = ["const"]
        names = trend_names + [f"x{i + 1}" for i in range(exog.shape[1] - k_trend)]
        return exog, names

    def _fit_k1(self, endog: NDArray[Any] | None = None) -> dict[str, Any]:
        """Fit the model without switching (K = 1)."""
        y = self.endog if endog is None else np.asarray(endog, dtype=np.float64)
        exog, names = self._one_regime_design(y)
        results = _fit_one_regime(y, exog, self.order, names)
        return {
            "results": results,
            "llf": results.llf,
            "aic": results.aic,
            "bic": results.bic,
            "hqic": results.hqic,
            "n_params": results.df_model,
        }

    def _build_model(self, k: int, endog: NDArray[Any]) -> Any:
        """The Markov switching model with ``k`` regimes."""
        from regimes.markov import MarkovAR, MarkovRegression

        if self.model_type == "ar":
            return MarkovAR(endog, k_regimes=k, order=self.order, **self.model_kwargs)
        return MarkovRegression(endog, k_regimes=k, **self.model_kwargs)

    def _fit_k_regimes(
        self,
        k: int,
        lower: Any,
        one_regime: OneRegimeResults,
        endog: NDArray[Any] | None = None,
        inference: bool = True,
    ) -> dict[str, Any]:
        """Fit a k-regime Markov switching model.

        ``lower`` is the fitted model with ``k - 1`` regimes: the estimation
        is also started from its estimates with each regime split in two.
        ``one_regime`` is the fitted one-regime model of the same sample,
        whose variance enters the penalty of a model with regime-specific
        variances. Without ``inference`` no covariance matrix is computed
        (bootstrap samples).
        """
        y = self.endog if endog is None else endog
        penalized = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = self._build_model(k, y)
            model._search_rng = self._rng
            model._extra_starts = lambda sm_model: _split_starts(sm_model, lower)

            def objective(sm_model: Any) -> Any:
                nonlocal penalized
                penalty = _variance_penalty(sm_model, one_regime.sigma2)
                penalized = penalty is not None
                return penalty

            model._search_objective = objective
            model._search_inference = inference
            results = model.fit(search_reps=self._search_reps)

        if not np.isfinite(results.llf):
            raise RuntimeError(f"the model with {k} regimes could not be estimated")
        return {
            "results": results,
            "llf": float(results.llf),
            "aic": float(results.aic),
            "bic": float(results.bic),
            "hqic": float(results.hqic),
            "n_params": int(results.df_model),
            "penalized": penalized,
        }

    def _fit_up_to(
        self, k_top: int, endog: NDArray[Any], inference: bool
    ) -> dict[int, dict[str, Any]]:
        """Fit the models with ``1, ..., k_top`` regimes on a sample.

        The one procedure for the data and for every bootstrap sample: each
        model is estimated from the default, the random and the split
        starting values built from the model with one regime less.
        """
        fitted = {1: self._fit_k1(endog)}
        for k in range(2, k_top + 1):
            fitted[k] = self._fit_k_regimes(
                k, fitted[k - 1]["results"], fitted[1]["results"], endog, inference
            )
        return fitted

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

    # ------------------------------------------------------------------
    # Bootstrap likelihood ratio test
    # ------------------------------------------------------------------

    def _simulate(
        self, null_results: Any, rng: np.random.Generator
    ) -> NDArray[np.floating[Any]]:
        """Draw a sample from the fitted model with K regimes."""
        if isinstance(null_results, OneRegimeResults):
            return simulate_linear(
                null_results._exog,
                null_results.params,
                null_results.sigma2,
                null_results.ar_params,
                self.endog[: self.order],
                len(self.endog),
                rng,
            )
        sm_results = null_results._sm_results
        return simulate_markov_switching(
            sm_results.model, np.asarray(sm_results.params, dtype=np.float64), rng
        )

    def _lr_statistic(self, k0: int, endog: NDArray[Any]) -> float:
        """LR statistic of ``k0`` against ``k0 + 1`` regimes on a sample.

        Every model up to ``k0 + 1`` regimes is estimated on the sample by
        the procedure used on the data (``_fit_up_to``).
        """
        fitted = self._fit_up_to(k0 + 1, endog, inference=False)
        return _lr(
            fitted[k0 + 1]["llf"],
            fitted[k0]["llf"],
            k0,
            penalized=fitted[k0 + 1]["penalized"],
        )

    def _bootstrap_test(
        self, k0: int, fitted: dict[int, dict[str, Any]], verbose: bool
    ) -> dict[str, Any]:
        """Bootstrap test of ``k0`` against ``k0 + 1`` regimes."""
        k1 = k0 + 1
        lr_stat = _lr(
            fitted[k1]["llf"], fitted[k0]["llf"], k0, penalized=fitted[k1]["penalized"]
        )
        null_results = fitted[k0]["results"]

        if verbose:
            print(
                f"Testing K={k0} against K={k1}: LR = {lr_stat:.4f}, "
                f"{self.n_bootstrap} bootstrap replications..."
            )

        if lr_stat == 0.0:
            # every simulated statistic is at least zero: p-value one
            p_value, critical_value = 1.0, float("nan")
            n_valid = n_failed = 0
        else:
            outcome = monte_carlo_pvalue(
                lr_stat,
                lambda rng: self._simulate(null_results, rng),
                lambda sample: self._lr_statistic(k0, sample),
                self.n_bootstrap,
                as_generator(self._rng),
            )
            if outcome.n_valid == 0:
                raise RuntimeError(
                    f"testing {k0} against {k1} regimes: the statistic "
                    "could not be computed in any bootstrap replication"
                ) from outcome.last_error
            if not can_reject(outcome.n_valid, self.significance):
                warnings.warn(
                    f"testing {k0} against {k1} regimes: the statistic could "
                    f"be computed in {outcome.n_valid} of {self.n_bootstrap} "
                    "bootstrap replications only, too few for a test at "
                    f"level {self.significance} to reject",
                    UserWarning,
                    stacklevel=4,
                )
            p_value = outcome.p_value
            critical_value = outcome.critical_value(self.significance)
            n_valid, n_failed = outcome.n_valid, outcome.n_failed

        return {
            "k0": k0,
            "k1": k1,
            "lr_statistic": lr_stat,
            "p_value": p_value,
            "critical_value": critical_value,
            "n_extra_params": fitted[k1]["n_params"] - fitted[k0]["n_params"],
            "n_bootstrap": n_valid,
            "n_bootstrap_failed": n_failed,
            "rejected": rejects(p_value, self.significance),
        }

    def _try_fit(
        self,
        k: int,
        fitted: dict[int, dict[str, Any]],
        notes: list[str],
        verbose: bool,
    ) -> bool:
        """Fit the model with ``k`` regimes on the data; False if it fails."""
        if verbose:
            print(f"Fitting K={k}...")
        try:
            fitted[k] = self._fit_k_regimes(
                k, fitted[k - 1]["results"], fitted[1]["results"]
            )
        except Exception as error:
            note = (
                f"the model with {k} regimes could not be estimated "
                f"({type(error).__name__}: {error}); models with {k} or more "
                "regimes were not considered"
            )
            notes.append(note)
            warnings.warn(note, UserWarning, stacklevel=3)
            return False
        return True

    def fit(self, verbose: bool = False) -> RegimeNumberSelectionResults:
        """Run regime number selection.

        With an information criterion the models with ``1, ..., k_max``
        regimes are estimated. With ``method="sequential"`` a model is
        estimated only when it is needed for a test, so ``results_by_k`` and
        the table stop one regime above the selected number.

        Parameters
        ----------
        verbose : bool
            Whether to print progress.

        Returns
        -------
        RegimeNumberSelectionResults
            Selection results. ``notes`` lists a model that could not be
            estimated (a warning is issued as well) and a selected K equal
            to ``k_max``.
        """
        self._rng = (
            None if self.random_state is None else as_generator(self.random_state)
        )
        notes: list[str] = []
        fitted: dict[int, dict[str, Any]] = {}

        if verbose:
            print("Fitting K=1 (no switching)...")
        fitted[1] = self._fit_k1()

        sequential_tests = None
        if self.method == "sequential":
            sequential_tests = []
            selected_k = 1
            for k0 in range(1, self.k_max):
                if not self._try_fit(k0 + 1, fitted, notes, verbose):
                    break
                test = self._bootstrap_test(k0, fitted, verbose)
                sequential_tests.append(test)
                if not test["rejected"]:
                    break
                selected_k = k0 + 1
            if selected_k == self.k_max and self.k_max > 1:
                notes.append(
                    f"{self.k_max} regimes is k_max: it was not tested against "
                    f"{self.k_max + 1}"
                )
        else:
            for k in range(2, self.k_max + 1):
                if not self._try_fit(k, fitted, notes, verbose):
                    break

        ic_table = self._build_ic_table(fitted)
        if self.method != "sequential":
            criterion = {"aic": "AIC", "bic": "BIC", "hqic": "HQIC"}[self.method]
            selected_k = self._select_by_ic(ic_table, criterion)

        return RegimeNumberSelectionResults(
            selected_k=selected_k,
            results_by_k={k: info["results"] for k, info in fitted.items()},
            ic_table=ic_table,
            method=self.method,
            sequential_tests=sequential_tests,
            notes=notes,
        )
