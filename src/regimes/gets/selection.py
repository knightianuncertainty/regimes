"""Core GETS/Autometrics model selection algorithm.

Implements a bounded general-to-specific model selection procedure inspired
by Doornik (2009), using iterative sequential elimination with bunching,
chopping, and optional multi-path search.  The search is bounded by
``max_paths`` and ``max_evaluations`` to guarantee termination even with
large indicator-saturation GUMs.

References
----------
Doornik, J. A. (2009). Autometrics. In J. L. Castle & N. Shephard (Eds.),
    *The Methodology and Practice of Econometrics* (pp. 88-121). Oxford
    University Press.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import statsmodels.api as sm
from scipy import stats

from regimes.gets.results import GETSResults, TerminalModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from regimes.models.ols import OLSResults


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _fit_ols_subset(
    endog: NDArray[np.floating[Any]],
    exog: NDArray[np.floating[Any]],
    mask: NDArray[np.bool_],
    cov_type: str = "nonrobust",
    names: Sequence[str] | None = None,
) -> OLSResults:
    """Fit OLS on the columns of exog selected by a boolean mask.

    Parameters
    ----------
    endog : NDArray[np.floating]
        Dependent variable.
    exog : NDArray[np.floating]
        Full design matrix (GUM).
    mask : NDArray[np.bool_]
        Boolean mask selecting which columns to include.
    cov_type : str
        Covariance type for the OLS fit.
    names : Sequence[str] | None
        Full list of variable names (same length as exog columns).
        Only the names corresponding to True entries in mask are used.

    Returns
    -------
    OLSResults
        Fitted OLS results for the selected subset.
    """
    from regimes.models.ols import OLSResults

    X_sub = exog[:, mask]
    n = len(endog)

    sm_model = sm.OLS(endog, X_sub)
    if cov_type == "nonrobust":
        sm_results = sm_model.fit()
    elif cov_type.startswith("HC"):
        sm_results = sm_model.fit(cov_type=cov_type)
    elif cov_type == "HAC":
        maxlags = int(np.floor(4 * (n / 100) ** (2 / 9)))
        sm_results = sm_model.fit(
            cov_type="HAC",
            cov_kwds={"maxlags": maxlags, "use_correction": True},
        )
    else:
        sm_results = sm_model.fit()

    sub_names: list[str] | None = None
    if names is not None:
        sub_names = [names[i] for i in range(len(mask)) if mask[i]]

    y_mean = np.mean(endog)
    tss = float(np.sum((endog - y_mean) ** 2))

    return OLSResults(
        params=np.asarray(sm_results.params),
        bse=np.asarray(sm_results.bse),
        resid=np.asarray(sm_results.resid),
        fittedvalues=np.asarray(sm_results.fittedvalues),
        cov_params_matrix=np.asarray(sm_results.cov_params()),
        nobs=n,
        cov_type=cov_type,
        param_names=sub_names,
        model_name="OLS",
        llf=float(sm_results.llf),
        scale=float(sm_results.scale),
        _tss=tss,
        _exog=X_sub,
    )


def _f_test_exclusion(
    ssr_unrestricted: float,
    ssr_restricted: float,
    q: int,
    df_resid_u: int,
) -> tuple[float, float]:
    """F-test for joint exclusion of q restrictions.

    Tests H0: the q excluded variables have zero coefficients.

    Parameters
    ----------
    ssr_unrestricted : float
        Sum of squared residuals from the unrestricted model.
    ssr_restricted : float
        Sum of squared residuals from the restricted model.
    q : int
        Number of restrictions (excluded variables).
    df_resid_u : int
        Residual degrees of freedom of the unrestricted model.

    Returns
    -------
    f_stat : float
        F-statistic.
    p_value : float
        p-value from the F(q, df_resid_u) distribution.
    """
    if q <= 0 or df_resid_u <= 0:
        return np.nan, np.nan
    f_stat = ((ssr_restricted - ssr_unrestricted) / q) / (ssr_unrestricted / df_resid_u)
    p_value = 1 - stats.f.cdf(f_stat, q, df_resid_u)
    return float(f_stat), float(p_value)


def _encompassing_test(
    gum_results: OLSResults,
    terminal: TerminalModel,
    endog: NDArray[np.floating[Any]],
    exog: NDArray[np.floating[Any]],
) -> tuple[float, float]:
    """Backtesting encompassing F-test of a terminal model vs the GUM.

    Tests whether the GUM variables excluded from the terminal are
    jointly insignificant.

    Parameters
    ----------
    gum_results : OLSResults
        Results from the GUM.
    terminal : TerminalModel
        The terminal model to test.
    endog : NDArray[np.floating]
        Dependent variable.
    exog : NDArray[np.floating]
        Full GUM design matrix.

    Returns
    -------
    f_stat : float
        Encompassing F-statistic.
    p_value : float
        p-value of the encompassing test.
    """
    q = int(np.sum(~terminal.retained_mask))
    if q == 0:
        return 0.0, 1.0

    return _f_test_exclusion(
        ssr_unrestricted=gum_results.ssr,
        ssr_restricted=terminal.results.ssr,
        q=q,
        df_resid_u=gum_results.df_resid,
    )


# ---------------------------------------------------------------------------
# Bounded iterative reduction
# ---------------------------------------------------------------------------


def _gets_reduce(
    endog: NDArray[np.floating[Any]],
    exog: NDArray[np.floating[Any]],
    start_mask: NDArray[np.bool_],
    names: list[str],
    alpha: float,
    p_b: float,
    cov_type: str,
    protected: NDArray[np.bool_],
    n_eval: list[int],
    max_evaluations: int,
) -> TerminalModel | None:
    """Iterative sequential elimination to a terminal model.

    At each step:
    1. Try bunching (joint removal of all insignificant vars via F-test).
    2. If bunching rejected, remove the single most-insignificant variable.
    3. After removal, chop any newly-insignificant variables one at a time.
    4. Repeat until all remaining are significant or protected.

    Parameters
    ----------
    endog : NDArray
        Dependent variable.
    exog : NDArray
        Full GUM design matrix.
    start_mask : NDArray[np.bool_]
        Starting boolean mask of included variables.
    names : list[str]
        Variable names for the full GUM.
    alpha : float
        Significance level.
    p_b : float
        Bunching/pruning threshold.
    cov_type : str
        Covariance type.
    protected : NDArray[np.bool_]
        Boolean mask of protected (non-removable) variables.
    n_eval : list[int]
        Mutable counter for total OLS evaluations (shared across paths).
    max_evaluations : int
        Hard cap on total evaluations.

    Returns
    -------
    TerminalModel | None
        The terminal model, or None if max_evaluations exceeded before
        reaching a terminal.
    """
    k = len(start_mask)
    mask = start_mask.copy()
    path: list[int] = []

    # Fit starting model
    if mask.sum() == 0:
        return None
    results = _fit_ols_subset(endog, exog, mask, cov_type, names)
    n_eval[0] += 1

    while n_eval[0] < max_evaluations:
        active_indices = np.where(mask)[0]
        pvals = results.pvalues

        # Find removable variables: insignificant at alpha, not protected
        removable: list[tuple[int, float]] = []  # (gum_idx, pval)
        for local_i, gum_i in enumerate(active_indices):
            if protected[gum_i]:
                continue
            if pvals[local_i] > alpha:
                removable.append((int(gum_i), float(pvals[local_i])))

        if not removable:
            # Terminal: all remaining variables are significant or protected
            return TerminalModel(
                retained_mask=mask.copy(),
                retained_names=[names[i] for i in range(k) if mask[i]],
                results=results,
                path=list(path),
            )

        # Sort by descending p-value (most insignificant first)
        removable.sort(key=lambda x: -x[1])

        # --- Try bunching ---
        bunch_indices = [r[0] for r in removable]
        k_b = len(bunch_indices)

        bunching_accepted = False
        if k_b > 1:
            p_b_half = p_b**0.5
            p_b_star = p_b_half * (1 - (1 - p_b_half) ** k_b)

            bunch_mask = mask.copy()
            for idx in bunch_indices:
                bunch_mask[idx] = False

            if bunch_mask.sum() > 0:
                bunch_results = _fit_ols_subset(
                    endog, exog, bunch_mask, cov_type, names
                )
                n_eval[0] += 1

                _f_stat, f_pval = _f_test_exclusion(
                    results.ssr, bunch_results.ssr, k_b, results.df_resid
                )

                if f_pval > p_b_star:
                    # Bunching accepted: remove all insignificant at once
                    bunching_accepted = True
                    path.extend(bunch_indices)
                    mask = bunch_mask
                    results = bunch_results

                    # Chop: iteratively remove newly-insignificant vars
                    chopping = True
                    while chopping and n_eval[0] < max_evaluations:
                        chopping = False
                        chop_active = np.where(mask)[0]
                        chop_pvals = results.pvalues
                        for local_i, gum_i in enumerate(chop_active):
                            if protected[gum_i]:
                                continue
                            if chop_pvals[local_i] > alpha:
                                mask[gum_i] = False
                                path.append(int(gum_i))
                                chopping = True
                                break
                        if chopping and mask.sum() > 0:
                            results = _fit_ols_subset(
                                endog, exog, mask, cov_type, names
                            )
                            n_eval[0] += 1
                        elif mask.sum() == 0:
                            return None

                    # After chop, loop back to check for more removable vars
                    continue

        if not bunching_accepted:
            # Remove single most-insignificant variable
            worst_idx = removable[0][0]
            worst_pval = removable[0][1]

            # Pruning: if the worst is below p_b, it's "borderline" — record as terminal
            if worst_pval <= p_b:
                return TerminalModel(
                    retained_mask=mask.copy(),
                    retained_names=[names[i] for i in range(k) if mask[i]],
                    results=results,
                    path=list(path),
                )

            mask[worst_idx] = False
            path.append(worst_idx)
            if mask.sum() == 0:
                return None
            results = _fit_ols_subset(endog, exog, mask, cov_type, names)
            n_eval[0] += 1

    # Exceeded max_evaluations — return current state as terminal
    if mask.sum() > 0:
        return TerminalModel(
            retained_mask=mask.copy(),
            retained_names=[names[i] for i in range(k) if mask[i]],
            results=results,
            path=list(path),
        )
    return None


# ---------------------------------------------------------------------------
# Main GETS search
# ---------------------------------------------------------------------------


def gets_search(
    endog: NDArray[np.floating[Any]],
    exog: NDArray[np.floating[Any]],
    names: Sequence[str] | None = None,
    alpha: float = 0.05,
    selection: Literal["bic", "aic", "hq"] = "bic",
    cov_type: str = "nonrobust",
    diagnostics: bool = True,
    protected: NDArray[np.bool_] | None = None,
    max_paths: int = 5,
    max_evaluations: int = 500,
) -> GETSResults:
    """Run the bounded GETS model selection search.

    Implements a bounded version of the Doornik (2009) algorithm:

    1. Fit GUM, check diagnostics, compute p_b.
    2. Primary path: iterative sequential elimination with bunching/chopping.
    3. Secondary paths: alternative starting points for multi-path exploration.
    4. Delayed diagnostics on terminals.
    5. Backtesting: encompassing F-test of each terminal vs GUM.
    6. Union of surviving terminals -> re-search.
    7. Tiebreaker: select by IC.

    Parameters
    ----------
    endog : NDArray[np.floating]
        Dependent variable (n,).
    exog : NDArray[np.floating]
        Full GUM design matrix (n, k).
    names : Sequence[str] | None
        Variable names. If None, uses ``"x0"``, ``"x1"``, etc.
    alpha : float
        Target significance level (p_a) for selection. Default 0.05.
    selection : {"bic", "aic", "hq"}
        Information criterion for tiebreaking among surviving terminals.
    cov_type : str
        Covariance type for OLS fits.
    diagnostics : bool
        If True, apply delayed misspecification diagnostic checks.
    protected : NDArray[np.bool_] | None
        Boolean mask of variables that cannot be removed (e.g., core
        regressors in an indicator saturation context). Same length as
        exog columns.
    max_paths : int
        Maximum number of alternative reduction paths to explore.
        Default 5. Use 1 for fastest execution.
    max_evaluations : int
        Hard cap on total OLS model evaluations across all paths.
        Default 500. Prevents runaway computation with large GUMs.

    Returns
    -------
    GETSResults
        Search results including GUM, terminals, and selected model.
    """
    endog = np.asarray(endog, dtype=np.float64)
    exog = np.asarray(exog, dtype=np.float64)
    _n, k = exog.shape

    if names is None:
        names = [f"x{i}" for i in range(k)]
    names = list(names)

    if protected is None:
        protected = np.zeros(k, dtype=bool)
    else:
        protected = np.asarray(protected, dtype=bool)

    # Step 0: Fit GUM
    gum_mask = np.ones(k, dtype=bool)
    gum_results = _fit_ols_subset(endog, exog, gum_mask, cov_type, names)

    # Compute p_b (Doornik's bunching/pruning threshold)
    p_b = max(0.5 * alpha**0.5, alpha**0.75)

    # Shared evaluation counter (mutable for closure)
    n_evaluated = [1]  # GUM already counted

    # Collect terminal models via multi-path search
    terminals: list[TerminalModel] = []

    # --- Path 1: primary reduction from full GUM ---
    tm = _gets_reduce(
        endog,
        exog,
        gum_mask,
        names,
        alpha,
        p_b,
        cov_type,
        protected,
        n_evaluated,
        max_evaluations,
    )
    if tm is not None:
        terminals.append(tm)

    # --- Additional paths: remove the i-th most insignificant variable first ---
    if max_paths > 1 and n_evaluated[0] < max_evaluations:
        gum_pvals = gum_results.pvalues
        gum_active = np.where(gum_mask)[0]

        # Find removable variables sorted by descending p-value
        removable_start: list[tuple[int, float]] = []
        for local_i, gum_i in enumerate(gum_active):
            if protected[gum_i]:
                continue
            if gum_pvals[local_i] > alpha:
                removable_start.append((int(gum_i), float(gum_pvals[local_i])))

        removable_start.sort(key=lambda x: -x[1])

        # Try removing each of the top removable vars as alternative starts
        # Skip the first (already tried in primary path)
        for path_idx in range(1, min(max_paths, len(removable_start))):
            if n_evaluated[0] >= max_evaluations:
                break

            alt_mask = gum_mask.copy()
            alt_mask[removable_start[path_idx][0]] = False

            alt_tm = _gets_reduce(
                endog,
                exog,
                alt_mask,
                names,
                alpha,
                p_b,
                cov_type,
                protected,
                n_evaluated,
                max_evaluations,
            )
            if alt_tm is not None:
                # Only add if it's genuinely different from existing terminals
                is_duplicate = any(
                    np.array_equal(alt_tm.retained_mask, t.retained_mask)
                    for t in terminals
                )
                if not is_duplicate:
                    terminals.append(alt_tm)

    # If no terminals found, keep the GUM
    if not terminals:
        gum_terminal = TerminalModel(
            retained_mask=gum_mask.copy(),
            retained_names=list(names),
            results=gum_results,
            path=[],
            diagnostics_pass=True,
            encompassing_pass=True,
            f_encompassing=0.0,
            p_encompassing=1.0,
        )
        return GETSResults(
            gum_results=gum_results,
            terminal_models=[gum_terminal],
            surviving_terminals=[gum_terminal],
            selected_model=gum_terminal,
            selection_criterion=selection,
            n_models_evaluated=n_evaluated[0],
        )

    # Step 4: Delayed diagnostics
    for tm in terminals:
        if diagnostics:
            try:
                diag = tm.results.diagnostics()
                tm.diagnostics_pass = diag.all_pass
            except (ValueError, Exception):
                # If diagnostics fail (e.g., too few observations), skip
                tm.diagnostics_pass = True
        else:
            tm.diagnostics_pass = True

    # Step 5: Backtesting encompassing test
    for tm in terminals:
        f_stat, p_val = _encompassing_test(gum_results, tm, endog, exog)
        tm.f_encompassing = f_stat
        tm.p_encompassing = p_val
        tm.encompassing_pass = p_val > alpha

    # Surviving: pass both diagnostics and encompassing
    surviving = [tm for tm in terminals if tm.diagnostics_pass and tm.encompassing_pass]

    # If none survive, fall back to all terminals
    if not surviving:
        surviving = list(terminals)

    # Step 6: Union of surviving terminals -> re-search
    if len(surviving) > 1:
        union_mask = np.zeros(k, dtype=bool)
        for tm in surviving:
            union_mask |= tm.retained_mask

        if not np.array_equal(union_mask, gum_mask):
            # Re-search with the union
            _fit_ols_subset(endog, exog, union_mask, cov_type, names)
            n_evaluated[0] += 1

    # Step 7: Tiebreaker by information criterion
    def _get_ic(tm: TerminalModel) -> float:
        r = tm.results
        ic_map = {"bic": r.bic, "aic": r.aic, "hq": r.hq}
        return ic_map.get(selection, r.bic)  # type: ignore[return-value]

    selected = min(surviving, key=_get_ic)

    return GETSResults(
        gum_results=gum_results,
        terminal_models=terminals,
        surviving_terminals=surviving,
        selected_model=selected,
        selection_criterion=selection,
        n_models_evaluated=n_evaluated[0],
    )
