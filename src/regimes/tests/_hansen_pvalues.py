"""Asymptotic p-values of the Sup, Exp and Ave tests for a structural break.

The statistics of Andrews (1993) and Andrews and Ploberger (1994) have
limiting distributions that depend on the number of coefficients tested,
``q``, and on the range of candidate break dates. Hansen (1997) approximates
the p-value function by

    p(x) = 1 - G(theta_0 + theta_1 x + ... | eta)

where ``G(. | eta)`` is the chi-squared distribution function with ``eta``
degrees of freedom, and tabulates the coefficients for ``q = 1, ..., 40`` and
``pi_0 = 0.01, 0.03, ..., 0.49``. This module evaluates that approximation
exactly as Hansen's own program does (linear interpolation in ``pi_0``
between the two nearest tabulated values) and inverts it for critical
values. The coefficients are in ``regimes.tests._hansen_tables``.

Three deviations from Hansen's program, each where it is not meant to be
used: for ``pi_0`` between 0.49 and 0.5 (almost a single candidate date) the
ExpF statistic is doubled before it is referred to chi-squared, since it is
half the statistic of that date; far in the tail, where the polynomials for
``q <= 3`` turn back, the p-value is held at its minimum; and a warning is
issued for ``pi_0`` below 0.01, the smallest tabulated value, where the
p-value for 0.01 is returned.

The range of candidate dates enters through

    lambda = pi_2 (1 - pi_1) / (pi_1 (1 - pi_2)),   pi_0 = 1 / (1 + sqrt(lambda))

where the candidate break fractions run from ``pi_1`` to ``pi_2``. With
symmetric trimming, ``pi_2 = 1 - pi_1``, this gives ``pi_0 = pi_1``.

The statistics must be on the chi-squared (Wald, LM or LR) scale: built from
a sequence that is asymptotically chi-squared with ``q`` degrees of freedom
at each date, not from the F statistic divided by ``q``.

References
----------
- Andrews (1993), "Tests for Parameter Instability and Structural Change
  with Unknown Change Point", Econometrica 61, 821-856.
- Andrews and Ploberger (1994), "Optimal Tests When a Nuisance Parameter Is
  Present Only under the Alternative", Econometrica 62, 1383-1414.
- Hansen (1997), "Approximate Asymptotic P Values for Structural-Change
  Tests", Journal of Business and Economic Statistics 15, 60-67.
"""

from __future__ import annotations

import warnings
from functools import cache
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import optimize, stats

from regimes.tests import _hansen_tables

if TYPE_CHECKING:
    from numpy.typing import NDArray

Kind = Literal["sup", "exp", "ave"]

MAX_Q = _hansen_tables.MAX_Q
_TABLES = {
    "sup": _hansen_tables.SUP,
    "exp": _hansen_tables.EXP,
    "ave": _hansen_tables.AVE,
}


def pi0_from_range(pi_1: float, pi_2: float) -> float:
    """Andrews' ``pi_0`` for candidate break fractions from ``pi_1`` to ``pi_2``."""
    if not 0.0 < pi_1 <= pi_2 < 1.0:
        raise ValueError("the break fractions must satisfy 0 < pi_1 <= pi_2 < 1")
    lam = pi_2 * (1.0 - pi_1) / (pi_1 * (1.0 - pi_2))
    return float(1.0 / (1.0 + np.sqrt(lam)))


@cache
def _turning_points(kind: str, q: int) -> tuple[tuple[tuple[float, float], ...], ...]:
    """For each row of a table, the local maxima of its polynomial.

    Returns, per row, pairs ``(x, value)`` with ``x > 0`` at which the
    polynomial in the statistic has a local maximum.
    """
    beta = np.asarray(_TABLES[kind][q], dtype=np.float64)
    n_terms = beta.shape[1] - 1
    rows = []
    for coefs in beta[:, :n_terms]:
        first = np.polynomial.polynomial.polyder(coefs)
        second = np.polynomial.polynomial.polyder(first)
        points = []
        for root in np.atleast_1d(np.polynomial.polynomial.polyroots(first)):
            if abs(root.imag) > 1e-12 or root.real <= 0:
                continue
            x = float(root.real)
            if np.polynomial.polynomial.polyval(x, second) < 0:
                points.append((x, float(np.polynomial.polynomial.polyval(x, coefs))))
        rows.append(tuple(points))
    return tuple(rows)


def _running_maximum(kind: str, q: int, statistic: float) -> NDArray[np.floating[Any]]:
    """Largest value each row's polynomial has reached at or before ``statistic``.

    For ``q <= 3`` Hansen's polynomials are quadratic or cubic and turn down
    far in the tail: for ExpF with two or three coefficients and AveF with
    one or two, at statistics whose p-value is below about 0.001. Beyond that
    point
    the published formula (and programs that use it) return p-values that
    rise towards one. Here the argument of the chi-squared distribution is
    held at its maximum, so that the p-value never increases with the
    statistic.
    """
    out = np.full(len(_TABLES[kind][q]), -np.inf)
    for i, points in enumerate(_turning_points(kind, q)):
        for x, value in points:
            if x <= statistic:
                out[i] = max(out[i], value)
    return out


def hansen_pvalue(statistic: float, q: int, pi0: float, kind: Kind) -> float:
    """Approximate asymptotic p-value of Hansen (1997).

    The value is that of Hansen's own program, except far in the tail for
    ``q <= 3``, where his polynomial turns down and the p-value is held at
    its minimum (see ``_running_maximum``).

    Parameters
    ----------
    statistic : float
        Value of the Sup, Exp or Ave statistic (chi-squared scale).
    q : int
        Number of coefficients tested for constancy, 1 to 40.
    pi0 : float
        Andrews' ``pi_0``, in (0, 0.5]; see ``pi0_from_range``.
    kind : "sup" | "exp" | "ave"
        Which statistic.

    Returns
    -------
    float
        P-value. NaN if the statistic is NaN.
    """
    if kind not in _TABLES:
        raise ValueError(f"kind must be 'sup', 'exp' or 'ave', got {kind!r}")
    if not 1 <= q <= MAX_Q:
        raise ValueError(
            f"p-values are tabulated for 1 to {MAX_Q} coefficients, got q = {q}"
        )
    if not 0.0 < pi0 <= 0.5:
        raise ValueError(f"pi0 must be in (0, 0.5], got {pi0}")
    if np.isnan(statistic):
        return float("nan")

    # With a single candidate date (pi_0 = 0.5) SupF and AveF are the
    # chi-squared statistic of that date and ExpF is half of it. Hansen's
    # program refers all three to chi2(q) there, which is wrong for ExpF by
    # the factor two; here ExpF is doubled first.
    at_single_date = 2.0 * statistic if kind == "exp" else statistic
    chi2_tail = float(stats.chi2.sf(at_single_date, q))
    if pi0 == 0.5:
        return chi2_tail

    beta = np.asarray(_TABLES[kind][q], dtype=np.float64)
    n_terms = beta.shape[1] - 1
    argument = beta[:, :n_terms] @ (float(statistic) ** np.arange(n_terms))
    if n_terms > 2:
        argument = np.maximum(argument, _running_maximum(kind, q, float(statistic)))
    argument = np.maximum(argument, 0.0)
    tails = stats.chi2.sf(argument, beta[:, n_terms])

    if pi0 < 0.01 - 1e-12:
        warnings.warn(
            f"pi_0 = {pi0:.4f} is below 0.01, the smallest value Hansen "
            "(1997) tabulates; the p-value is that for pi_0 = 0.01 and is too "
            "small (trimming of less than about 1 percent)",
            UserWarning,
            stacklevel=3,
        )
    if pi0 <= 0.01:
        return float(tails[-1])
    if pi0 >= 0.49:
        return float(((0.5 - pi0) * tails[0] + (pi0 - 0.49) * chi2_tail) * 100.0)
    # rows are pi_0 = 0.49, 0.47, ..., 0.01; position on that grid, from 1
    position = (0.5 - pi0 + 0.01) * 50.0
    lower = int(np.floor(position + 1e-9))
    lower = min(max(lower, 1), len(tails) - 1)
    weight = position - lower
    return float((1.0 - weight) * tails[lower - 1] + weight * tails[lower])


def hansen_critical_value(significance: float, q: int, pi0: float, kind: Kind) -> float:
    """Critical value: the statistic whose Hansen p-value is ``significance``.

    NaN if no statistic has that p-value: a level above the p-value of a
    statistic of zero, or (for ExpF with two or three coefficients and AveF
    with one or two) a level below the floor at which the p-value is held,
    about 0.0005 or less.
    """
    if not 0.0 < significance < 1.0:
        raise ValueError("significance must be between 0 and 1")

    def gap(x: float) -> float:
        return hansen_pvalue(x, q, pi0, kind) - significance

    lower = 0.0
    if gap(lower) <= 0.0:
        # the approximation is not meant for p-values this large
        return float("nan")
    upper = float(q) + 10.0
    for _ in range(60):
        if gap(upper) < 0.0:
            break
        upper *= 2.0
    else:
        return float("nan")
    return float(optimize.brentq(gap, lower, upper, xtol=1e-10, rtol=1e-12))
