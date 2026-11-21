"""Dual representation: shifts and regime levels.

Every indicator saturation result can be expressed in two equivalent forms:

1. **Shifts representation** — what the SIS/MIS regression directly estimates:
   initial level plus step changes at each break date.

2. **Regime levels representation** — cumulated from shifts: the level of each
   parameter within each regime. Each parameter has its **own** regime
   schedule (breaks rarely coincide across parameters).

This module provides dataclasses for both forms and conversion between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ParameterRegime:
    """A single regime for one parameter.

    Parameters
    ----------
    start : int
        First observation in this regime (inclusive).
    end : int
        Last observation in this regime (inclusive).
    level : float
        Estimated parameter level in this regime.
    level_se : float
        Standard error of the level estimate.
    """

    start: int
    end: int
    level: float
    level_se: float


@dataclass
class ShiftsRepresentation:
    """Shifts form of a saturation result.

    This is what SIS/MIS directly estimates: an initial level and step changes.

    Parameters
    ----------
    break_dates : list[int]
        Union of all break dates across all parameters.
    initial_levels : dict[str, float]
        Pre-break level of each parameter (the base coefficient).
    shifts : dict[str, dict[int, float]]
        Step-change magnitudes. Outer key is parameter name, inner key
        is break date, value is the shift magnitude.
    shift_se : dict[str, dict[int, float]]
        Standard errors of the shift estimates.
    """

    break_dates: list[int]
    initial_levels: dict[str, float]
    shifts: dict[str, dict[int, float]]
    shift_se: dict[str, dict[int, float]]


@dataclass
class RegimeLevelsRepresentation:
    """Regime-levels form of a saturation result.

    Each parameter has its own regime schedule — its level as a step function
    over time. Breaks rarely coincide across parameters, so each parameter
    has independent regimes.

    Parameters
    ----------
    param_regimes : dict[str, list[ParameterRegime]]
        Per-parameter list of regimes, sorted by start date.
    """

    param_regimes: dict[str, list[ParameterRegime]]


def shifts_to_levels(
    shifts_rep: ShiftsRepresentation,
    n: int,
    cov_params: NDArray[np.floating[Any]] | None = None,
    param_names: list[str] | None = None,
    indicator_names: list[str] | None = None,
) -> RegimeLevelsRepresentation:
    """Convert shifts representation to regime levels.

    For each parameter, cumulates the shifts to get the level in each regime.
    Standard errors are propagated via the delta method using the cumulation
    matrix.

    Parameters
    ----------
    shifts_rep : ShiftsRepresentation
        The shifts representation to convert.
    n : int
        Total number of observations (for regime end bounds).
    cov_params : NDArray[np.floating] | None
        Full covariance matrix of the estimated model. If provided, used
        for SE propagation. Otherwise, SEs are taken from shift_se directly.
    param_names : list[str] | None
        Names of the base parameters in the model (e.g., ["const", "y.L1"]).
    indicator_names : list[str] | None
        Names of all parameters in the estimated model (base + indicators),
        used to locate entries in cov_params.

    Returns
    -------
    RegimeLevelsRepresentation
        Regime-levels form with per-parameter regime schedules.
    """
    result: dict[str, list[ParameterRegime]] = {}

    for param_name, param_shifts in shifts_rep.shifts.items():
        initial = shifts_rep.initial_levels.get(param_name, 0.0)
        se_dict = shifts_rep.shift_se.get(param_name, {})

        if not param_shifts:
            # No breaks for this parameter — single regime
            initial_se = se_dict.get(-1, 0.0)  # sentinel
            # Try to get SE from the model directly
            if (
                cov_params is not None
                and indicator_names is not None
                and param_name in indicator_names
            ):
                idx = indicator_names.index(param_name)
                initial_se = float(np.sqrt(cov_params[idx, idx]))
            result[param_name] = [
                ParameterRegime(start=0, end=n - 1, level=initial, level_se=initial_se)
            ]
            continue

        # Sort break dates for this parameter
        sorted_dates = sorted(param_shifts.keys())
        # Build regime boundaries
        boundaries = [0] + sorted_dates + [n]
        regimes: list[ParameterRegime] = []

        # Cumulate shifts to get levels
        cumulative = initial
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1] - 1

            if i > 0:
                cumulative += param_shifts[sorted_dates[i - 1]]

            # SE propagation
            level_se = _propagate_se(
                param_name,
                i,
                sorted_dates,
                initial,
                se_dict,
                cov_params,
                indicator_names,
                param_names,
            )

            regimes.append(
                ParameterRegime(
                    start=start,
                    end=end,
                    level=cumulative,
                    level_se=level_se,
                )
            )

        result[param_name] = regimes

    # Add parameters with no shifts (just the initial level)
    for param_name, initial in shifts_rep.initial_levels.items():
        if param_name not in result:
            initial_se = 0.0
            if (
                cov_params is not None
                and indicator_names is not None
                and param_name in indicator_names
            ):
                idx = indicator_names.index(param_name)
                initial_se = float(np.sqrt(cov_params[idx, idx]))
            result[param_name] = [
                ParameterRegime(start=0, end=n - 1, level=initial, level_se=initial_se)
            ]

    return RegimeLevelsRepresentation(param_regimes=result)


def _propagate_se(
    param_name: str,
    regime_idx: int,
    sorted_dates: list[int],
    initial: float,
    se_dict: dict[int, float],
    cov_params: NDArray[np.floating[Any]] | None,
    indicator_names: list[str] | None,
    param_names: list[str] | None,
) -> float:
    """Propagate standard errors for a regime level via delta method.

    For regime j (0-indexed), the level is:
        level_0 = beta_base
        level_j = beta_base + sum(shift_1, ..., shift_j)

    The variance is L @ Sigma_sub @ L.T where L is the cumulation row
    [1, 1, ..., 1, 0, ..., 0].
    """
    if regime_idx == 0:
        # First regime: just the base parameter SE
        if (
            cov_params is not None
            and indicator_names is not None
            and param_name in indicator_names
        ):
            idx = indicator_names.index(param_name)
            return float(np.sqrt(max(0, cov_params[idx, idx])))
        return se_dict.get(-1, 0.0)

    if cov_params is None or indicator_names is None:
        # Approximate: sqrt(sum of squared SEs) - ignoring covariances
        var_sum = 0.0
        # Add base parameter variance
        base_se = se_dict.get(-1, 0.0)
        var_sum += base_se**2
        # Add shift variances
        for j in range(regime_idx):
            tau = sorted_dates[j]
            shift_se = se_dict.get(tau, 0.0)
            var_sum += shift_se**2
        return float(np.sqrt(var_sum))

    # Delta method with full covariance matrix
    # Identify the indices in indicator_names for: base param + shifts 1..regime_idx
    indices = []
    if param_name in indicator_names:
        indices.append(indicator_names.index(param_name))

    for j in range(regime_idx):
        tau = sorted_dates[j]
        # Find the indicator name: could be "step_TAU" (SIS) or "PARAM*step_TAU" (MIS)
        step_name = f"step_{tau}"
        mis_name = f"{param_name}*step_{tau}"
        if mis_name in indicator_names:
            indices.append(indicator_names.index(mis_name))
        elif step_name in indicator_names:
            indices.append(indicator_names.index(step_name))

    if not indices:
        return 0.0

    # Cumulation vector: all ones (level = base + sum of shifts)
    L = np.ones(len(indices))
    sub_cov = cov_params[np.ix_(indices, indices)]
    var = float(L @ sub_cov @ L)
    return float(np.sqrt(max(0, var)))


def levels_to_shifts(
    levels_rep: RegimeLevelsRepresentation,
) -> ShiftsRepresentation:
    """Convert regime levels representation back to shifts.

    Parameters
    ----------
    levels_rep : RegimeLevelsRepresentation
        The regime-levels representation to convert.

    Returns
    -------
    ShiftsRepresentation
        Equivalent shifts representation.
    """
    all_breaks: set[int] = set()
    initial_levels: dict[str, float] = {}
    shifts: dict[str, dict[int, float]] = {}
    shift_se: dict[str, dict[int, float]] = {}

    for param_name, regimes in levels_rep.param_regimes.items():
        if not regimes:
            continue

        initial_levels[param_name] = regimes[0].level
        param_shifts: dict[int, float] = {}
        param_se: dict[int, float] = {}
        param_se[-1] = regimes[0].level_se  # base SE sentinel

        for i in range(1, len(regimes)):
            tau = regimes[i].start
            shift = regimes[i].level - regimes[i - 1].level
            param_shifts[tau] = shift
            # Approximate SE: sqrt(var_i + var_{i-1}) ignoring covariance
            param_se[tau] = float(
                np.sqrt(regimes[i].level_se ** 2 + regimes[i - 1].level_se ** 2)
            )
            all_breaks.add(tau)

        shifts[param_name] = param_shifts
        shift_se[param_name] = param_se

    return ShiftsRepresentation(
        break_dates=sorted(all_breaks),
        initial_levels=initial_levels,
        shifts=shifts,
        shift_se=shift_se,
    )
