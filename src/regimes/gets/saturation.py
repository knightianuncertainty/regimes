"""Indicator saturation orchestration.

Provides ``isat()``, the main entry point for indicator saturation analysis.
This function assembles the indicator matrix, runs the split-half GETS
procedure, and produces a dual representation of the detected breaks.

References
----------
Doornik, J. A. (2009). Autometrics. Oxford University Press.
Castle, J. L., Doornik, J. A., & Hendry, D. F. (2012). Model selection when
    there are multiple breaks. *Journal of Econometrics*, 169(2), 239-246.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from regimes.gets.indicators import (
    impulse_indicators,
    multiplicative_indicators,
    step_indicators,
    trend_indicators,
)
from regimes.gets.representation import (
    RegimeLevelsRepresentation,
    ShiftsRepresentation,
    shifts_to_levels,
)
from regimes.gets.selection import gets_search

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from regimes.gets.results import GETSResults
    from regimes.models.ols import OLSResults


@dataclass(kw_only=True)
class SaturationResults:
    """Results from indicator saturation analysis.

    Contains the full GETS search results, the retained indicators, and
    the dual representation (shifts and regime levels).

    Parameters
    ----------
    gets_results : GETSResults
        Results from the final GETS search.
    saturation_type : str
        Description of the saturation type (e.g., "SIS", "MIS", "SIS+MIS").
    n_indicators_initial : int
        Total number of candidate indicators before selection.
    n_indicators_retained : int
        Number of indicators retained after selection.
    retained_indicators : list[str]
        Names of retained indicators.
    retained_indicator_dates : list[int]
        Tau values of retained indicators.
    retained_indicator_mask : NDArray[np.bool_]
        Boolean mask over the full indicator set.
    shifts : ShiftsRepresentation
        Shifts form of the result.
    regime_levels : RegimeLevelsRepresentation
        Regime-levels form of the result.
    selected_results : OLSResults
        OLS results for the selected model.
    break_dates : list[int]
        Union of all break dates across all parameters.
    n_regimes : int
        Total number of distinct regimes (max across parameters).
    core_names : list[str]
        Names of the core (non-indicator) regressors.
    """

    gets_results: GETSResults
    saturation_type: str
    n_indicators_initial: int
    n_indicators_retained: int
    retained_indicators: list[str]
    retained_indicator_dates: list[int]
    retained_indicator_mask: NDArray[np.bool_]
    shifts: ShiftsRepresentation
    regime_levels: RegimeLevelsRepresentation
    selected_results: OLSResults
    break_dates: list[int]
    n_regimes: int
    core_names: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a text summary of the saturation results.

        Returns
        -------
        str
            Formatted summary.
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"{'Indicator Saturation Results':^70}")
        lines.append("=" * 70)
        lines.append(f"Saturation type:          {self.saturation_type}")
        lines.append(f"Observations:             {self.selected_results.nobs}")
        lines.append(f"Candidate indicators:     {self.n_indicators_initial}")
        lines.append(f"Retained indicators:      {self.n_indicators_retained}")
        lines.append(f"Break dates:              {self.break_dates}")
        lines.append(f"Number of regimes:        {self.n_regimes}")
        lines.append("-" * 70)

        if self.retained_indicators:
            lines.append("")
            lines.append("Retained Indicators:")
            for name in self.retained_indicators:
                lines.append(f"  {name}")

        lines.append("")
        lines.append("Regime Levels:")
        for param_name, regimes in self.regime_levels.param_regimes.items():
            lines.append(f"  {param_name}:")
            for r in regimes:
                lines.append(
                    f"    [{r.start:>4}-{r.end:>4}]  level={r.level:>8.4f}  "
                    f"se={r.level_se:>7.4f}"
                )

        lines.append("")
        lines.append("-" * 70)
        lines.append("Selected Model:")
        lines.append(self.selected_results.summary(diagnostics=False))
        return "\n".join(lines)

    def plot_sis(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Plot SIS results (intercept regime levels over time).

        Parameters
        ----------
        **kwargs
            Passed to ``plot_sis_coefficients``.

        Returns
        -------
        tuple[Figure, Axes]
        """
        from regimes.visualization.gets import plot_sis_coefficients

        return plot_sis_coefficients(self, **kwargs)

    def plot_mis(self, **kwargs: Any) -> tuple[Figure, Any]:
        """Plot MIS results (coefficient regime levels per parameter).

        Parameters
        ----------
        **kwargs
            Passed to ``plot_mis_coefficients``.

        Returns
        -------
        tuple[Figure, Axes | NDArray]
        """
        from regimes.visualization.gets import plot_mis_coefficients

        return plot_mis_coefficients(self, **kwargs)

    def plot_regime_levels(self, **kwargs: Any) -> tuple[Figure, Any]:
        """Plot all regime levels (combined SIS + MIS).

        Parameters
        ----------
        **kwargs
            Passed to ``plot_regime_levels``.

        Returns
        -------
        tuple[Figure, Axes | NDArray]
        """
        from regimes.visualization.gets import plot_regime_levels

        return plot_regime_levels(self, **kwargs)


def _build_core_regressors(
    endog: NDArray[np.floating[Any]],
    exog: NDArray[np.floating[Any]] | None,
    ar_lags: int | list[int] | None,
    exog_lags: int | dict[str, int] | None,
    has_constant: bool,
    exog_names: list[str] | None = None,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], list[str], int]:
    """Assemble the core (non-indicator) regressors.

    Returns
    -------
    y : NDArray
        Trimmed endog (after lag loss).
    X_core : NDArray
        Core regressor matrix.
    core_names : list[str]
        Names for the core regressors.
    maxlag : int
        Number of observations lost to lags.
    """
    endog = np.asarray(endog, dtype=np.float64).ravel()
    n_full = len(endog)
    maxlag = 0

    columns: list[NDArray[np.floating[Any]]] = []
    names: list[str] = []

    # AR lags
    if ar_lags is not None:
        if isinstance(ar_lags, int):
            lags = list(range(1, ar_lags + 1))
        else:
            lags = list(ar_lags)
        maxlag = max(maxlag, max(lags))
        for lag in lags:
            names.append(f"y.L{lag}")

    # Exog and exog lags
    if exog is not None:
        exog = np.asarray(exog, dtype=np.float64)
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        k_exog = exog.shape[1]
        if exog_names is None:
            exog_names = [f"x{j}" for j in range(k_exog)]

        # Contemporaneous exog
        for j in range(k_exog):
            names.append(exog_names[j])

        # Exog lags
        if exog_lags is not None:
            if isinstance(exog_lags, int):
                lag_dict: dict[str, int] = dict.fromkeys(exog_names, exog_lags)
            else:
                lag_dict = exog_lags
            for nm, max_l in lag_dict.items():
                if nm not in exog_names:
                    continue
                for lag in range(1, max_l + 1):
                    names.append(f"{nm}.L{lag}")
                    maxlag = max(maxlag, lag)

    # Trim to effective sample
    y = endog[maxlag:]
    n_eff = len(y)

    # Build columns after trimming
    columns = []

    # Constant
    if has_constant:
        columns.append(np.ones(n_eff))
        names_final = ["const"]
    else:
        names_final = []

    # AR lags
    if ar_lags is not None:
        if isinstance(ar_lags, int):
            lags = list(range(1, ar_lags + 1))
        else:
            lags = list(ar_lags)
        for lag in lags:
            columns.append(endog[maxlag - lag : n_full - lag])
            names_final.append(f"y.L{lag}")

    # Exog columns
    if exog is not None:
        for j in range(exog.shape[1]):
            columns.append(exog[maxlag:, j])
            names_final.append(exog_names[j])

        if exog_lags is not None:
            if isinstance(exog_lags, int):
                lag_dict = dict.fromkeys(exog_names, exog_lags)
            else:
                lag_dict = exog_lags
            for j, nm in enumerate(exog_names):
                if nm not in lag_dict:
                    continue
                for lag in range(1, lag_dict[nm] + 1):
                    columns.append(exog[maxlag - lag : n_full - lag, j])
                    names_final.append(f"{nm}.L{lag}")

    if not columns:
        # No regressors at all — just constant
        if has_constant:
            X_core = np.ones((n_eff, 1))
            names_final = ["const"]
        else:
            X_core = np.empty((n_eff, 0))
            names_final = []
    else:
        X_core = np.column_stack(columns)

    return y, X_core, names_final, maxlag


def _extract_tau(name: str) -> int | None:
    """Extract tau from an indicator name like 'step_50' or 'x0*step_120'."""
    m = re.search(r"(?:step|impulse|trend)_(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def _build_dual_representation(
    results: OLSResults,
    core_names: list[str],
    n: int,
) -> tuple[ShiftsRepresentation, RegimeLevelsRepresentation]:
    """Build dual representation from a fitted model with indicators.

    Parameters
    ----------
    results : OLSResults
        Final selected model results.
    core_names : list[str]
        Names of core (non-indicator) regressors.
    n : int
        Number of observations.

    Returns
    -------
    tuple[ShiftsRepresentation, RegimeLevelsRepresentation]
    """
    all_names = list(results.param_names or [])
    params = results.params
    bse = results.bse

    initial_levels: dict[str, float] = {}
    shifts: dict[str, dict[int, float]] = {}
    shift_se: dict[str, dict[int, float]] = {}

    # Map core params
    for name in core_names:
        if name in all_names:
            idx = all_names.index(name)
            initial_levels[name] = float(params[idx])
            shifts[name] = {}
            shift_se[name] = {-1: float(bse[idx])}  # -1 sentinel for base SE

    # Map indicator params
    for i, name in enumerate(all_names):
        if name in core_names:
            continue

        tau = _extract_tau(name)
        if tau is None:
            continue

        # Determine which core parameter this indicator belongs to
        if "*step_" in name:
            # MIS: "x0*step_50" -> core param "x0"
            core_param = name.split("*step_")[0]
        elif name.startswith("step_"):
            # SIS: "step_50" -> affects the constant
            core_param = "const"
        elif name.startswith("impulse_") or name.startswith("trend_"):
            core_param = "const"
        else:
            continue

        if core_param not in shifts:
            if core_param not in initial_levels:
                initial_levels[core_param] = 0.0
            shifts[core_param] = {}
            shift_se[core_param] = {-1: 0.0}

        shifts[core_param][tau] = float(params[i])
        shift_se[core_param][tau] = float(bse[i])

    all_breaks = sorted(
        {tau for param_shifts in shifts.values() for tau in param_shifts}
    )

    shifts_rep = ShiftsRepresentation(
        break_dates=all_breaks,
        initial_levels=initial_levels,
        shifts=shifts,
        shift_se=shift_se,
    )

    # Convert to levels using full covariance matrix
    levels_rep = shifts_to_levels(
        shifts_rep,
        n,
        cov_params=results.cov_params_matrix,
        indicator_names=all_names,
    )

    return shifts_rep, levels_rep


def isat(
    endog: NDArray[np.floating[Any]] | Any,
    exog: NDArray[np.floating[Any]] | None = None,
    ar_lags: int | list[int] | None = None,
    exog_lags: int | dict[str, int] | None = None,
    # Indicator types
    iis: bool = False,
    sis: bool = False,
    mis: bool | list[str] | list[int] = False,
    tis: bool = False,
    # Custom indicators
    user_indicators: NDArray[np.floating[Any]] | None = None,
    user_indicator_names: list[str] | None = None,
    # GETS parameters
    alpha: float = 0.05,
    selection: Literal["bic", "aic", "hq"] = "bic",
    diagnostics: bool = True,
    # Block splitting
    n_blocks: int | None = None,
    max_block_size: int | None = None,
    # Trimming and estimation
    trim: float = 0.05,
    cov_type: str = "nonrobust",
    has_constant: bool = True,
    exog_names: list[str] | None = None,
) -> SaturationResults:
    """Run indicator saturation analysis.

    This is the main entry point for detecting structural breaks via the
    Autometrics/GETS procedure with indicator saturation.

    Parameters
    ----------
    endog : array-like
        Dependent variable.
    exog : array-like | None
        Exogenous regressors (excluding constant, which is added
        automatically if ``has_constant=True``).
    ar_lags : int | list[int] | None
        AR lag specification. ``ar_lags=2`` includes lags 1 and 2.
    exog_lags : int | dict[str, int] | None
        Lag structure for exog. ``exog_lags=1`` lags all by 1.
        ``exog_lags={"x0": 2, "x1": 0}`` for variable-specific lags.
    iis : bool
        Include impulse indicator saturation.
    sis : bool
        Include step indicator saturation (level shifts).
    mis : bool | list[str] | list[int]
        Include multiplicative indicator saturation (coefficient shifts).
        - ``True``: interact steps with all non-constant core regressors.
        - ``["y.L1", "x0"]``: interact only with named variables.
        - ``[0, 1]``: interact only with indexed variables.
        - ``False``: no MIS.
    tis : bool
        Include trend indicator saturation.
    user_indicators : NDArray | None
        Custom indicator matrix to include (n_eff, k_user).
    user_indicator_names : list[str] | None
        Names for custom indicators.
    alpha : float
        Significance level for GETS selection.
    selection : {"bic", "aic", "hq"}
        Information criterion for tiebreaking.
    diagnostics : bool
        Apply delayed diagnostic checks in GETS.
    n_blocks : int | None
        Number of blocks for split-half procedure. If None, determined
        automatically from max_block_size.
    max_block_size : int | None
        Maximum indicators per block. Default: ``n_eff // 2``.
    trim : float
        Fraction trimmed from each end for indicator generation.
    cov_type : str
        Covariance type for OLS estimation.
    has_constant : bool
        Whether to include a constant term.
    exog_names : list[str] | None
        Names for exog columns.

    Returns
    -------
    SaturationResults
        Full results including GETS output, dual representation, and
        convenience accessors.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = np.concatenate([rng.normal(0, 1, 100), rng.normal(2, 1, 100)])
    >>> result = isat(y, sis=True)
    >>> print(result.break_dates)
    >>> print(result.regime_levels.param_regimes["const"])
    """
    # Build core regressors
    y, X_core, core_names, _maxlag = _build_core_regressors(
        endog,
        exog,
        ar_lags,
        exog_lags,
        has_constant,
        exog_names,
    )
    n_eff = len(y)

    # Determine saturation type label
    types = []
    if iis:
        types.append("IIS")
    if sis:
        types.append("SIS")
    if mis is not False:
        types.append("MIS")
    if tis:
        types.append("TIS")
    if user_indicators is not None:
        types.append("USER")
    saturation_type = "+".join(types) if types else "NONE"

    # Build indicator matrices
    indicator_columns: list[NDArray[np.floating[Any]]] = []
    indicator_names: list[str] = []

    if sis:
        S, s_names = step_indicators(n_eff, trim=trim)
        indicator_columns.append(S)
        indicator_names.extend(s_names)

    if iis:
        imp_mat, i_names = impulse_indicators(n_eff, trim=trim)
        indicator_columns.append(imp_mat)
        indicator_names.extend(i_names)

    if mis is not False:
        # Build non-constant core regressors for MIS interaction
        non_const_cols: list[NDArray[np.floating[Any]]] = []
        non_const_names: list[str] = []
        for j, name in enumerate(core_names):
            if name == "const":
                continue
            non_const_cols.append(X_core[:, j])
            non_const_names.append(name)

        if non_const_cols:
            X_nc = np.column_stack(non_const_cols)
            mis_vars: Sequence[int | str] | None = None
            if isinstance(mis, list):
                mis_vars = mis  # type: ignore[assignment]

            M, m_names = multiplicative_indicators(
                X_nc,
                n_eff,
                variables=mis_vars,
                trim=trim,
                exog_names=non_const_names,
            )
            indicator_columns.append(M)
            indicator_names.extend(m_names)

    if tis:
        T, t_names = trend_indicators(n_eff, trim=trim)
        indicator_columns.append(T)
        indicator_names.extend(t_names)

    if user_indicators is not None:
        user_indicators = np.asarray(user_indicators, dtype=np.float64)
        if user_indicators.ndim == 1:
            user_indicators = user_indicators.reshape(-1, 1)
        if user_indicators.shape[0] != n_eff:
            raise ValueError(
                f"user_indicators has {user_indicators.shape[0]} rows "
                f"but effective sample has {n_eff} observations"
            )
        indicator_columns.append(user_indicators)
        if user_indicator_names is None:
            user_indicator_names = [
                f"user_{j}" for j in range(user_indicators.shape[1])
            ]
        indicator_names.extend(user_indicator_names)

    n_indicators = len(indicator_names)

    if n_indicators == 0:
        raise ValueError(
            "No indicators requested. Set at least one of iis, sis, mis, "
            "tis, or provide user_indicators."
        )

    # Assemble full indicator matrix
    indicators = (
        np.column_stack(indicator_columns)
        if len(indicator_columns) > 1
        else indicator_columns[0]
    )

    k_core = X_core.shape[1]
    protected_core = np.ones(k_core, dtype=bool)

    # Block splitting — ensure each block's GUM is not singular
    # (block indicators + core regressors must not exceed n_eff)
    if max_block_size is None:
        max_block_size = max(min(n_eff // 2, n_eff - k_core - 5), 1)

    if n_blocks is not None:
        block_size = max(1, n_indicators // n_blocks)
    else:
        block_size = min(max_block_size, n_indicators)

    n_blocks_actual = max(1, int(np.ceil(n_indicators / block_size)))

    if n_blocks_actual == 1:
        # Single block: no split needed
        X_gum = np.column_stack([X_core, indicators])
        all_names = core_names + indicator_names
        protected = np.concatenate([protected_core, np.zeros(n_indicators, dtype=bool)])
        final_gets = gets_search(
            y,
            X_gum,
            names=all_names,
            alpha=alpha,
            selection=selection,
            cov_type=cov_type,
            diagnostics=diagnostics,
            protected=protected,
        )
    else:
        # Multi-block split-half
        retained_from_blocks: list[int] = []  # indices into indicator matrix

        for block_idx in range(n_blocks_actual):
            start = block_idx * block_size
            end = min(start + block_size, n_indicators)
            block_indicators = indicators[:, start:end]
            block_names = indicator_names[start:end]

            X_block = np.column_stack([X_core, block_indicators])
            block_all_names = core_names + block_names
            block_protected = np.concatenate(
                [
                    protected_core,
                    np.zeros(end - start, dtype=bool),
                ]
            )

            block_gets = gets_search(
                y,
                X_block,
                names=block_all_names,
                alpha=alpha,
                selection=selection,
                cov_type=cov_type,
                diagnostics=False,
                protected=block_protected,
            )

            # Map retained indicators back to full indicator indices
            for name in block_gets.retained_names:
                if name in block_names:
                    local_idx = block_names.index(name)
                    retained_from_blocks.append(start + local_idx)

        # Final GUM: core + all retained indicators
        if retained_from_blocks:
            retained_from_blocks = sorted(set(retained_from_blocks))
            retained_indicators_matrix = indicators[:, retained_from_blocks]
            retained_ind_names = [indicator_names[i] for i in retained_from_blocks]

            X_final = np.column_stack([X_core, retained_indicators_matrix])
            final_names = core_names + retained_ind_names
            final_protected = np.concatenate(
                [
                    protected_core,
                    np.zeros(len(retained_from_blocks), dtype=bool),
                ]
            )

            final_gets = gets_search(
                y,
                X_final,
                names=final_names,
                alpha=alpha,
                selection=selection,
                cov_type=cov_type,
                diagnostics=diagnostics,
                protected=final_protected,
            )
        else:
            # No indicators retained from any block
            final_gets = gets_search(
                y,
                X_core,
                names=core_names,
                alpha=alpha,
                selection=selection,
                cov_type=cov_type,
                diagnostics=diagnostics,
                protected=protected_core,
            )

    # Extract retained indicators from final model
    retained_ind = [
        name for name in final_gets.retained_names if name not in core_names
    ]
    retained_taus = [
        tau for tau in (_extract_tau(name) for name in retained_ind) if tau is not None
    ]

    # Build retained indicator mask over the full indicator set
    retained_mask = np.array(
        [name in retained_ind for name in indicator_names], dtype=bool
    )

    # Build dual representation
    shifts_rep, levels_rep = _build_dual_representation(
        final_gets.selected_results,
        core_names,
        n_eff,
    )

    # Compute break dates and n_regimes
    break_dates = shifts_rep.break_dates
    max_regimes = 1
    for regimes in levels_rep.param_regimes.values():
        max_regimes = max(max_regimes, len(regimes))

    return SaturationResults(
        gets_results=final_gets,
        saturation_type=saturation_type,
        n_indicators_initial=n_indicators,
        n_indicators_retained=len(retained_ind),
        retained_indicators=retained_ind,
        retained_indicator_dates=sorted(set(retained_taus)),
        retained_indicator_mask=retained_mask,
        shifts=shifts_rep,
        regime_levels=levels_rep,
        selected_results=final_gets.selected_results,
        break_dates=break_dates,
        n_regimes=max_regimes,
        core_names=core_names,
    )
