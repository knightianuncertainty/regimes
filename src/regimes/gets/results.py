"""Result containers for GETS model selection.

Provides dataclasses for terminal models and the overall GETS search results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from regimes.models.ols import OLSResults


@dataclass(kw_only=True)
class TerminalModel:
    """A terminal model from the GETS tree search.

    A terminal model is reached when all remaining variables are significant
    at the selection threshold, and no further reductions can be made.

    Parameters
    ----------
    retained_mask : NDArray[np.bool_]
        Boolean mask over GUM columns indicating which are retained.
    retained_names : list[str]
        Names of the retained variables.
    results : OLSResults
        Full OLS estimation results for this terminal.
    path : list[int]
        Indices (in GUM) of variables removed to reach this terminal.
    diagnostics_pass : bool
        Whether the terminal passes delayed diagnostic checks.
    encompassing_pass : bool
        Whether the terminal passes the backtesting encompassing test vs GUM.
    f_encompassing : float
        F-statistic from the encompassing test.
    p_encompassing : float
        p-value from the encompassing test.
    """

    retained_mask: NDArray[np.bool_]
    retained_names: list[str]
    results: OLSResults
    path: list[int] = field(default_factory=list)
    diagnostics_pass: bool = True
    encompassing_pass: bool = True
    f_encompassing: float = np.nan
    p_encompassing: float = np.nan


@dataclass(kw_only=True)
class GETSResults:
    """Results from a GETS tree search.

    Contains the GUM results, all terminal models found, the subset that
    survives diagnostics and backtesting, and the final selected model.

    Parameters
    ----------
    gum_results : OLSResults
        Results from the General Unrestricted Model.
    terminal_models : list[TerminalModel]
        All terminal models found by tree search.
    surviving_terminals : list[TerminalModel]
        Terminals that pass diagnostics and encompassing tests.
    selected_model : TerminalModel
        Best model among surviving terminals (by IC).
    selection_criterion : str
        Which information criterion was used for tiebreaking.
    n_models_evaluated : int
        Number of OLS fits performed during the search.
    """

    gum_results: OLSResults
    terminal_models: list[TerminalModel]
    surviving_terminals: list[TerminalModel]
    selected_model: TerminalModel
    selection_criterion: str = "bic"
    n_models_evaluated: int = 0

    @property
    def retained_mask(self) -> NDArray[np.bool_]:
        """Boolean mask of retained GUM columns in the selected model."""
        return self.selected_model.retained_mask

    @property
    def retained_names(self) -> list[str]:
        """Names of retained variables in the selected model."""
        return self.selected_model.retained_names

    @property
    def selected_results(self) -> OLSResults:
        """OLS results for the selected terminal model."""
        return self.selected_model.results

    def summary(self) -> str:
        """Generate a text summary of the GETS search.

        Returns
        -------
        str
            Formatted summary of GUM, search statistics, and selected model.
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"{'GETS Model Selection Results':^70}")
        lines.append("=" * 70)
        lines.append(f"GUM variables:            {self.gum_results.df_model:>6}")
        lines.append(f"Terminal models found:     {len(self.terminal_models):>6}")
        lines.append(f"Surviving terminals:      {len(self.surviving_terminals):>6}")
        lines.append(
            f"Selected model variables:  {len(self.selected_model.retained_names):>6}"
        )
        lines.append(f"Selection criterion:       {self.selection_criterion:>6}")
        lines.append(f"Models evaluated:         {self.n_models_evaluated:>6}")
        lines.append("-" * 70)
        lines.append("")
        lines.append("Selected Model:")
        lines.append(self.selected_model.results.summary(diagnostics=False))
        return "\n".join(lines)
