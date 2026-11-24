"""Helpers for the dynamics of AR and ADL results with breaks.

The autoregressive coefficients of a fitted model are found by name
(``y.L2`` or, with breaks, ``y.L2_regime1``). A model with breaks has one
set of coefficients for each stretch of the sample between break dates, and
the roots of the autoregressive polynomial, the stationarity check and the
long-run multiplier are defined stretch by stretch.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

_REGIME_SUFFIX = re.compile(r"^(?P<base>.*)_regime(?P<regime>\d+)$")
_AR_NAME = re.compile(r"^y\.L(?P<lag>\d+)$")


def split_regime_name(name: str) -> tuple[str, int | None]:
    """``"x_regime2"`` -> ``("x", 2)``; ``"x"`` -> ``("x", None)``."""
    match = _REGIME_SUFFIX.match(name)
    if match is None:
        return name, None
    return match.group("base"), int(match.group("regime"))


def coefficients_by_name(
    param_names: Sequence[str], params: NDArray[np.floating[Any]]
) -> dict[str, list[float]]:
    """Coefficients of each variable, in the order of its regimes.

    A variable without breaks has a list of length one.
    """
    collected: dict[str, dict[int, float]] = {}
    for name, value in zip(param_names, params, strict=True):
        base, regime = split_regime_name(name)
        collected.setdefault(base, {})[regime if regime is not None else 1] = float(
            value
        )
    return {
        base: [by_regime[r] for r in sorted(by_regime)]
        for base, by_regime in collected.items()
    }


def break_dates_by_name(
    base_names: Sequence[str],
    breaks: Sequence[int] | None,
    variable_breaks: dict[str | int, Sequence[int]] | None,
) -> dict[str, list[int]]:
    """Break dates that apply to each variable (empty list: none)."""
    if breaks:
        return {name: sorted(breaks) for name in base_names}
    out: dict[str, list[int]] = {name: [] for name in base_names}
    if variable_breaks:
        for key, dates in variable_breaks.items():
            name = base_names[key] if isinstance(key, int) else key
            if name in out:
                out[name] = sorted(dates)
    return out


def segments(dates_by_name: dict[str, list[int]], names: Sequence[str]) -> list[int]:
    """Start of each stretch of the sample in which the coefficients of
    ``names`` are constant: 0 and every break date of one of them."""
    starts = {0}
    for name in names:
        starts.update(dates_by_name.get(name, []))
    return sorted(starts)


def coefficient_in_segment(
    values: list[float], dates: list[int], segment_start: int
) -> float:
    """The coefficient of a variable in the stretch that starts at
    ``segment_start``."""
    regime = sum(1 for date in dates if date <= segment_start)
    return values[min(regime, len(values) - 1)]


def ar_lag_names(base_names: Sequence[str]) -> dict[int, str]:
    """Lag -> name for the lagged dependent variable terms."""
    out: dict[int, str] = {}
    for name in base_names:
        match = _AR_NAME.match(name)
        if match is not None:
            out[int(match.group("lag"))] = name
    return out


def ar_roots(coefficients: dict[int, float]) -> NDArray[np.complexfloating[Any, Any]]:
    """Roots of ``1 - sum_L phi_L z^L`` for coefficients given by lag.

    Lags that are not in the model have a zero coefficient, so a model with
    lags 1 and 3 has a polynomial of degree 3.
    """
    if not coefficients:
        return np.array([], dtype=np.complex128)
    degree = max(coefficients)
    ascending = np.zeros(degree + 1)
    ascending[0] = 1.0
    for lag, phi in coefficients.items():
        ascending[lag] = -phi
    # trailing zero coefficients do not add roots
    while len(ascending) > 1 and ascending[-1] == 0.0:
        ascending = ascending[:-1]
    if len(ascending) == 1:
        return np.array([], dtype=np.complex128)
    return np.asarray(np.roots(ascending[::-1]), dtype=np.complex128)


def ar_coefficients_by_segment(
    param_names: Sequence[str],
    params: NDArray[np.floating[Any]],
    breaks: Sequence[int] | None,
    variable_breaks: dict[str | int, Sequence[int]] | None,
) -> list[tuple[int, dict[int, float]]]:
    """Autoregressive coefficients, by lag, for each stretch of the sample.

    Returns
    -------
    list[tuple[int, dict[int, float]]]
        Pairs ``(start, {lag: coefficient})``; ``start`` is the first
        observation (in the coordinates of the break dates) of a stretch in
        which the autoregressive coefficients are constant.
    """
    values = coefficients_by_name(param_names, params)
    base_names = list(values)
    lags = ar_lag_names(base_names)
    if not lags:
        return []
    dates = break_dates_by_name(base_names, breaks, variable_breaks)
    out = []
    for start in segments(dates, list(lags.values())):
        out.append(
            (
                start,
                {
                    lag: coefficient_in_segment(values[name], dates[name], start)
                    for lag, name in lags.items()
                },
            )
        )
    return out


def distributed_lag_effects_by_segment(
    param_names: Sequence[str],
    params: NDArray[np.floating[Any]],
    exog_lags: dict[str, list[int]],
    breaks: Sequence[int] | None,
    variable_breaks: dict[str | int, Sequence[int]] | None,
) -> list[tuple[int, dict[str, tuple[float, float]]]]:
    """Cumulative effect and long-run multiplier for each stretch of the sample.

    Returns
    -------
    list[tuple[int, dict[str, tuple[float, float]]]]
        Pairs ``(start, {variable: (cumulative, long_run)})``. The
        cumulative effect is the sum of the distributed-lag coefficients of
        the variable; the long-run multiplier divides it by one minus the
        sum of the autoregressive coefficients, and is NaN if that sum is
        one or more in absolute value.
    """
    values = coefficients_by_name(param_names, params)
    base_names = list(values)
    lags = ar_lag_names(base_names)
    dates = break_dates_by_name(base_names, breaks, variable_breaks)

    dl_names: dict[str, list[str]] = {}
    for variable, lag_list in exog_lags.items():
        names = [variable if lag == 0 else f"{variable}.L{lag}" for lag in lag_list]
        names = [name for name in names if name in values]
        if names:
            dl_names[variable] = names
    if not dl_names:
        return []

    relevant = list(lags.values()) + [n for names in dl_names.values() for n in names]
    out = []
    for start in segments(dates, relevant):
        ar_sum = sum(
            coefficient_in_segment(values[name], dates[name], start)
            for name in lags.values()
        )
        effects = {}
        for variable, names in dl_names.items():
            cumulative = sum(
                coefficient_in_segment(values[name], dates[name], start)
                for name in names
            )
            long_run = float("nan") if abs(ar_sum) >= 1 else cumulative / (1 - ar_sum)
            effects[variable] = (float(cumulative), float(long_run))
        out.append((start, effects))
    return out
