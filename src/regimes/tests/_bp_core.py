"""Bai-Perron multiple structural change algorithms on arrays.

The model is

    y_t = z_t' delta_j + x_t' beta + u_t,    t in regime j = 1, ..., m + 1,

where ``z`` (T x q) has coefficients that change across regimes and ``x``
(T x p) has coefficients that do not (partial structural change; ``x`` may be
absent). This module is a function-for-function port of the ``RgBP`` class of
the OxRegimes package (``src/bpcore.ox``), which was written against, and
cross-validated with, Pierre Perron's Bai-Perron program (GAUSS, October 2004,
distributed in the Journal of Applied Econometrics data archive with Bai and
Perron 2003). Where this module departs from that program it says so. The
two packages give the same numbers; a change here has to be made there too.

Conventions
-----------
* A partition with ``m`` breaks is the integer vector ``vb`` holding the LAST
  observation index (0-based) of regimes 1..m, as in Perron's program. The
  package-level break index (first observation of the new regime) is
  ``vb + 1``.
* All F statistics are "scaled by q": they are not divided by ``q``, exactly
  as computed by Perron's program, because that is the scaling of the
  published critical values:

      F = (T - (k+1)q - p) / (T k) * delta' R' (R V R')^{-1} R delta.

  With a common error variance this equals ``q`` times the classical F test.
* sup F(l+1|l) is the maximum over the l + 1 regimes of the one-break test
  inside each regime, where the candidate date is the SSR minimizer within
  the regime and the statistic uses that regime's data only (Perron's
  ``spflp1``/``sequa``). Bai and Perron (2003, Section 5.3) describe the test
  as "the application of (l + 1) tests of the null hypothesis of no
  structural change versus the alternative hypothesis of a single change",
  one per regime; they describe the selected date as the one giving the
  largest reduction in the overall sum of squared residuals, whereas the
  program (and this module) takes the regime with the largest F statistic.
* The minimum regime length is ``h = floor(trimming * T)``, as in the
  program (``h=int(eps1*bigt)``) and in Bai and Perron (2003, Table I), and
  at least ``q + 1``.

Departures from Perron's program
--------------------------------
* HAC prewhitening recolors with ``(I - A)^{-1}``, row i of ``A`` holding
  equation i of the VAR(1); the program uses the transpose (:func:`long_run`).
* Partial structural change: one break is found by direct search, and for
  two or more breaks the iteration is run from two starting values, the
  program's and the no-break regression, keeping the lower SSR
  (:func:`partial_partition_all`).
* Fixed regressors without variation inside a sub-sample (an impulse or step
  dummy) are left out of the tests in that sub-sample, where the program
  stops with a singular matrix (:func:`independent_columns`).
* The segment SSRs are computed from an orthonormal basis of the regressors,
  so that they do not depend on the units or the level of the regressors
  (:func:`ssr_triangle`).
* The search for the quantiles of the break-date distribution is widened
  when a quantile lies outside [-2000, 2000] (:func:`quantiles_argmax`).

References
----------
Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent
    covariance matrix estimation. Econometrica, 59(3), 817-858.

Andrews, D. W. K., & Monahan, J. C. (1992). An improved heteroskedasticity
    and autocorrelation consistent covariance matrix estimator.
    Econometrica, 60(4), 953-966.

Bai, J. (1997). Estimation of a change point in multiple regression models.
    Review of Economics and Statistics, 79(4), 551-563.

Bai, J., & Perron, P. (1998). Estimating and testing linear models with
    multiple structural changes. Econometrica, 66(1), 47-78.

Bai, J., & Perron, P. (2003). Computation and analysis of multiple
    structural change models. Journal of Applied Econometrics, 18(1), 1-22.

Liu, J., Wu, S., & Zidek, J. V. (1997). On segmented multivariate
    regression. Statistica Sinica, 7(2), 497-525.

Yao, Y.-C. (1988). Estimating the number of change-points via Schwarz'
    criterion. Statistics & Probability Letters, 6(3), 181-189.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

from regimes.tests import _bp_critical_values as _cv

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]

# Constants of the LWZ criterion (Liu, Wu and Zidek 1997)
LWZ_C0 = 0.299
LWZ_DELTA0 = 0.1
# smallest relative singular value for which the regressors are replaced by an
# orthonormal basis of their column space in the SSR computations
ORTHO_TOL = 1e-12
# relative length of the projection residual below which a fixed regressor is
# treated as a linear combination of the other regressors in a sub-sample
REDUNDANT_TOL = 1e-8
# the search interval for the quantiles of the break-date distribution starts
# at [-2000, 2000] (Perron's cvg) and is widened up to this bound
QUANTILE_BOUND = 1e9


@dataclass(frozen=True)
class BPOptions:
    """Assumptions on the errors and regressors used by tests and intervals.

    Parameters
    ----------
    robust : bool
        Allow serial correlation in the errors (HAC, quadratic spectral kernel
        with the Andrews (1991) AR(1) bandwidth).
    prewhiten : bool
        VAR(1) prewhitening of the HAC estimator (Andrews and Monahan 1992).
        Only used when ``robust`` is True.
    hetdat : bool
        Allow the distribution of the regressors to differ across regimes in
        the covariance of the coefficient estimates.
    hetvar : bool
        Allow the error variance to differ across regimes in the tests.
    hetomega : bool
        Allow the (long-run) error variance to differ across regimes in the
        break-date confidence intervals.
    hetq : bool
        Allow the regressor moment matrix to differ across regimes in the
        break-date confidence intervals.
    """

    robust: bool = False
    prewhiten: bool = False
    hetdat: bool = True
    hetvar: bool = True
    hetomega: bool = True
    hetq: bool = True


def segment_length(trimming: float, nobs: int, q: int) -> int:
    """Minimum regime length ``h = max(floor(trimming * T), q + 1)``.

    Rounding down follows Bai and Perron (2003, Table I: T = 103, trimming
    0.15, h = 15) and Perron's program. The small tolerance guards products
    such as ``0.1 * 30 = 3.0000000000000004`` and
    ``0.29 * 100 = 28.999999999999996``.
    """
    return max(math.floor(trimming * nobs + 1e-9), q + 1)


def bounds(vb: IntArray | list[int], nobs: int) -> IntArray:
    """Regime start indices ``[0, b_1 + 1, ..., b_m + 1, T]``."""
    b = np.asarray(vb, dtype=np.int64).ravel()
    out = np.empty(len(b) + 2, dtype=np.int64)
    out[0] = 0
    out[1:-1] = b + 1
    out[-1] = nobs
    return out


def _has_x(x: FloatArray | None) -> bool:
    return x is not None and x.size > 0


def _unit_columns(a: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Columns of ``a`` scaled to unit length, and the lengths (1 for a zero column)."""
    norms = np.sqrt(np.einsum("ij,ij->j", a, a))
    norms[norms == 0] = 1.0
    return a / norms, norms


def _orthonormal(z: FloatArray) -> FloatArray:
    """An orthonormal basis of the column space of ``z``.

    The SSR of a regression on ``z`` over any set of rows is unchanged when
    ``z`` is replaced by ``z A`` with ``A`` invertible, so the segment SSRs
    can be computed from this basis. With regressors of very different scale,
    or a regressor that varies little around a large mean, that keeps the
    moment matrices of the segments well conditioned. ``z`` is returned
    unchanged when it does not have full column rank.
    """
    if z.shape[0] < z.shape[1]:
        return z
    # columns scaled to unit length first, so that the rank decision does not
    # depend on the units of the regressors
    u, sv, _ = np.linalg.svd(_unit_columns(z)[0], full_matrices=False)
    if sv[-1] <= ORTHO_TOL * sv[0]:
        return z
    basis: FloatArray = u
    return basis


def independent_columns(
    z: FloatArray, x: FloatArray | None, tol: float = REDUNDANT_TOL
) -> list[int]:
    """Indices of the columns of ``x`` that are not linear combinations of
    ``z`` and of the columns of ``x`` before them, in the sample given.

    Inside a sub-sample a fixed regressor can be redundant although it is not
    in the full sample: an impulse dummy is zero in every regime but one, and
    a step dummy is constant on each side of its date. Its coefficient is then
    not identified in that sub-sample, and leaving it out changes neither the
    fitted values nor the test of a break in the coefficients of ``z``. A
    column is redundant when the residual of its projection on the others is
    less than ``tol`` times its own length.

    The decision does not depend on the units of the regressors.
    """
    if not _has_x(x):
        return []
    assert x is not None
    base, _ = _unit_columns(z)
    keep: list[int] = []
    for j in range(x.shape[1]):
        length = float(np.sqrt(x[:, j] @ x[:, j]))
        if length == 0.0:
            continue
        col = x[:, j] / length
        resid = col - base @ np.linalg.lstsq(base, col, rcond=None)[0]
        if float(np.sqrt(resid @ resid)) > tol:
            base = np.column_stack([base, col])
            keep.append(j)
    return keep


def independent_fixed(z: FloatArray, x: FloatArray | None) -> FloatArray | None:
    """The columns of ``x`` selected by :func:`independent_columns`, or None
    when none is left."""
    keep = independent_columns(z, x)
    if not keep:
        return None
    assert x is not None
    kept: FloatArray = x[:, keep]
    return kept


def _initial_fits(
    y: FloatArray, z: FloatArray, h: int
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray]:
    """OLS on the first admissible window of every start date.

    For each start ``i`` with ``i + h <= T`` the window is ``i..i+h-1``,
    extended one observation at a time while the design is rank deficient.

    Returns ``(P, beta, ssr, end)`` with ``P[i] = (Z'Z)^{-1}``, the
    coefficients, the SSR and the last index of the window; ``end[i] = T``
    marks a start for which no window has full column rank.
    """
    nobs, q = z.shape
    n_start = nobs - h + 1
    zw = np.lib.stride_tricks.sliding_window_view(z, h, axis=0).transpose(0, 2, 1)
    yw = np.lib.stride_tricks.sliding_window_view(y, h)
    gram = np.einsum("ihq,ihr->iqr", zw, zw)
    full_rank = np.linalg.matrix_rank(gram, hermitian=True) == q
    p_mat = np.zeros((n_start, q, q))
    beta = np.zeros((n_start, q))
    ssr = np.zeros(n_start)
    last = np.full(n_start, nobs, dtype=np.int64)
    if full_rank.any():
        p_ok = np.linalg.inv(gram[full_rank])
        b_ok = np.einsum(
            "iqr,ir->iq", p_ok, np.einsum("ihq,ih->iq", zw[full_rank], yw[full_rank])
        )
        resid = yw[full_rank] - np.einsum("ihq,iq->ih", zw[full_rank], b_ok)
        p_mat[full_rank], beta[full_rank] = p_ok, b_ok
        ssr[full_rank] = np.einsum("ih,ih->i", resid, resid)
        last[full_rank] = np.flatnonzero(full_rank) + h - 1
    for i in np.flatnonzero(~full_rank):
        stop = i + h
        while stop < nobs and np.linalg.matrix_rank(z[i : stop + 1]) < q:
            stop += 1
        if stop >= nobs:
            continue
        z0, y0 = z[i : stop + 1], y[i : stop + 1]
        p_i = np.linalg.inv(z0.T @ z0)
        b_i = p_i @ (z0.T @ y0)
        r_i = y0 - z0 @ b_i
        p_mat[i], beta[i], ssr[i], last[i] = p_i, b_i, float(r_i @ r_i), stop
    return p_mat, beta, ssr, last


def ssr_triangle(y: FloatArray, z: FloatArray, h: int) -> FloatArray:
    """Sums of squared residuals of all admissible segments.

    ``S[i, j]`` is the SSR of the OLS regression of ``y`` on ``z`` over the
    observations ``i..j`` (inclusive) for ``j - i + 1 >= h``, NaN elsewhere.
    Computed with recursive residuals (Brown, Durbin and Evans 1975), as in
    Bai and Perron (2003, Sections 3.1-3.2): each added observation updates
    the SSR by ``v^2 / f`` with ``v`` the one-step prediction error and
    ``f = 1 + z'(Z'Z)^{-1} z``.

    A segment whose first ``h`` rows give a singular design is extended until
    the design has full column rank. The regressors are first replaced by an
    orthonormal basis of their column space (see :func:`_orthonormal`), which
    leaves every SSR unchanged.
    """
    nobs = z.shape[0]
    s = np.full((nobs, nobs), np.nan)
    if nobs - h + 1 <= 0:
        return s
    z = _orthonormal(z)
    p_mat, beta, cur, first_end = _initial_fits(y, z, h)
    ok = np.flatnonzero(first_end < nobs)
    s[ok, first_end[ok]] = cur[ok]

    for j in range(h, nobs):
        act = np.flatnonzero(first_end < j)
        if act.size == 0:
            continue
        zj = z[j]
        pz = p_mat[act] @ zj  # (n_act, q): P z'
        f = 1.0 + pz @ zj
        v = y[j] - beta[act] @ zj
        gain = pz / f[:, None]
        beta[act] += gain * v[:, None]
        p_mat[act] -= gain[:, :, None] * pz[:, None, :]
        cur[act] += v * v / f
        s[act, j] = cur[act]
    return s


def ssr_expanding(y: FloatArray, z: FloatArray, h: int) -> FloatArray:
    """SSR of the regressions over ``0..j`` for every ``j`` (NaN if ``j < h - 1``).

    The first row of :func:`ssr_triangle`, computed alone.
    """
    nobs, q = z.shape
    out = np.full(nobs, np.nan)
    if nobs < h:
        return out
    z = _orthonormal(z)
    stop = h - 1
    while stop < nobs and np.linalg.matrix_rank(z[: stop + 1]) < q:
        stop += 1
    if stop >= nobs:
        return out
    z0, y0 = z[: stop + 1], y[: stop + 1]
    p_mat = np.linalg.inv(z0.T @ z0)
    beta = p_mat @ (z0.T @ y0)
    resid = y0 - z0 @ beta
    cur = float(resid @ resid)
    out[stop] = cur
    for j in range(stop + 1, nobs):
        zj = z[j]
        pz = p_mat @ zj
        f = 1.0 + float(pz @ zj)
        v = float(y[j] - beta @ zj)
        beta = beta + pz * (v / f)
        p_mat = p_mat - np.outer(pz, pz) / f
        cur += v * v / f
        out[j] = cur
    return out


def partition_all(
    s: FloatArray, max_breaks: int, h: int
) -> tuple[list[IntArray], FloatArray, float]:
    """Global SSR minimizers with 1..M breaks by dynamic programming.

    Implements the algorithm of Bai and Perron (2003, Section 3.3, eq. 4) on
    the matrix of segment SSRs from :func:`ssr_triangle`.

    Returns
    -------
    breaks : list of ndarray
        ``breaks[m - 1]`` is the last-index vector of the optimal m-break
        partition (empty if no admissible partition exists).
    ssr : ndarray
        ``ssr[m - 1]`` is its sum of squared residuals (NaN if none exists).
    ssr0 : float
        The SSR without breaks.
    """
    nobs = s.shape[0]
    u = np.full((max_breaks + 1, nobs), np.nan)
    arg = np.full((max_breaks + 1, nobs), -1, dtype=np.int64)
    u[0] = s[0]
    for k in range(1, max_breaks + 1):
        for j in range((k + 1) * h - 1, nobs):
            lo, hi = k * h - 1, j - h
            if hi < lo:
                continue
            cand = u[k - 1, lo : hi + 1] + s[lo + 1 : hi + 2, j]
            if np.all(np.isnan(cand)):
                continue
            best = int(np.nanargmin(cand))
            u[k, j] = cand[best]
            arg[k, j] = lo + best

    breaks: list[IntArray] = []
    ssr = np.full(max_breaks, np.nan)
    for m in range(1, max_breaks + 1):
        if np.isnan(u[m, nobs - 1]):
            breaks.append(np.empty(0, dtype=np.int64))
            continue
        vb = np.zeros(m, dtype=np.int64)
        jj = nobs - 1
        for k in range(m, 0, -1):
            vb[k - 1] = arg[k, jj]
            jj = int(arg[k, jj])
        breaks.append(vb)
        ssr[m - 1] = u[m, nobs - 1]
    return breaks, ssr, float(s[0, nobs - 1])


def zbar(z: FloatArray, vb: IntArray | list[int]) -> FloatArray:
    """Regime-blocked regressors: T x (m+1)q, block i nonzero in regime i."""
    nobs, q = z.shape
    bd = bounds(vb, nobs)
    m = len(bd) - 2
    zb = np.zeros((nobs, (m + 1) * q))
    for i in range(m + 1):
        zb[bd[i] : bd[i + 1], i * q : (i + 1) * q] = z[bd[i] : bd[i + 1]]
    return zb


def estimate(
    y: FloatArray, z: FloatArray, x: FloatArray | None, vb: IntArray | list[int]
) -> tuple[FloatArray, FloatArray, FloatArray, float, FloatArray]:
    """OLS of ``y`` on the regime-blocked ``z`` and the fixed ``x``.

    Returns
    -------
    delta : ndarray, ((m+1)q,)
        Regime coefficients, regime by regime.
    beta : ndarray, (p,)
        Coefficients on the fixed regressors (empty if none).
    resid : ndarray, (T,)
    ssr : float
    zb : ndarray
        The regime-blocked ``z``.
    """
    zb = zbar(z, vb)
    w = np.column_stack([zb, x]) if x is not None and x.size > 0 else zb
    # unit-length columns: the rank decision does not depend on the units
    wn, norms = _unit_columns(w)
    par_n, _, rank, _ = np.linalg.lstsq(wn, y, rcond=None)
    if rank < w.shape[1]:
        raise np.linalg.LinAlgError(
            "singular design: a regressor with breaking coefficients has no "
            "variation within one of the regimes, or a fixed regressor is a "
            "linear combination of the regime-specific ones"
        )
    par = par_n / norms
    resid = y - w @ par
    n_delta = zb.shape[1]
    return par[:n_delta], par[n_delta:], resid, float(resid @ resid), zb


def partial_partition_all(
    y: FloatArray,
    z: FloatArray,
    x: FloatArray,
    max_breaks: int,
    h: int,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> tuple[list[IntArray], FloatArray, float]:
    """Global minimizers for a partial structural change model.

    Iterative scheme of Bai and Perron (2003, Section 3.4): given ``beta``,
    apply the dynamic programming algorithm to ``y - x beta``; given the break
    dates, re-estimate all coefficients jointly; repeat until both the SSR and
    ``beta`` have converged. Convergence to the global minimum is not
    guaranteed (their footnote 3).

    One break is found by direct search over the admissible dates
    (:func:`one_break_date`), which is exact. For two or more breaks the
    iteration is run from two starting values of ``beta`` and the result with
    the lower SSR is kept:

    - the joint regression without breaks;
    - the pure structural change model in which all coefficients are allowed
      to change, as suggested by Bai and Perron (2003, Section 3.5) and done
      in Perron's ``nldat``.

    A model whose estimated partition makes a fixed regressor a linear
    combination of the regime-specific ones (a step dummy in ``x`` at one of
    the break dates) is not identified; it is reported as not admissible
    (empty break vector, NaN SSR) with a warning.

    Returns the same triple as :func:`partition_all`.
    """
    q = z.shape[1]
    w0 = np.column_stack([z, x])
    w0n, w0norms = _unit_columns(w0)
    par0 = np.linalg.lstsq(w0n, y, rcond=None)[0] / w0norms
    resid0 = y - w0 @ par0
    ssr0 = float(resid0 @ resid0)
    empty: IntArray = np.empty(0, dtype=np.int64)

    # break dates of the pure structural change model (second starting value)
    pure_parts: list[IntArray] = []
    if max_breaks >= 2:
        pure_parts = partition_all(ssr_triangle(y, w0, h), max_breaks, h)[0]

    breaks: list[IntArray] = []
    ssr = np.full(max_breaks, np.nan)
    for m in range(1, max_breaks + 1):
        if m == 1:
            b1, ssr1 = one_break_date(y, z, x, h)
            if b1 < 0:
                breaks.append(empty)
            else:
                breaks.append(np.array([b1], dtype=np.int64))
                ssr[0] = ssr1
            continue
        starts = [par0[q:].copy()]
        vb0 = pure_parts[m - 1]
        if vb0.size == m:
            # Perron's nldat: all coefficients regime-specific at the pure-change
            # dates, then beta from the regression of y - zbar delta on x
            zb0, xb0 = zbar(z, vb0), zbar(x, vb0)
            wn, norms = _unit_columns(np.column_stack([zb0, xb0]))
            teta = np.linalg.lstsq(wn, y, rcond=None)[0] / norms
            xn, xnorms = _unit_columns(x)
            starts.append(
                np.linalg.lstsq(xn, y - zb0 @ teta[: zb0.shape[1]], rcond=None)[0]
                / xnorms
            )
        best_vb, best_ssr, best_conv = empty, np.nan, False
        for beta0 in starts:
            vb, ssr_m, conv = _partial_iterate(y, z, x, m, h, beta0, tol, max_iter)
            if vb.size == m and (np.isnan(best_ssr) or ssr_m < best_ssr):
                best_vb, best_ssr, best_conv = vb, ssr_m, conv
        if best_vb.size == 0:
            warnings.warn(
                f"No identified partial structural change model with m={m} "
                f"breaks: no admissible partition, or a fixed regressor is a "
                f"linear combination of the regime-specific ones at the "
                f"estimated break dates.",
                stacklevel=3,
            )
            breaks.append(empty)
            continue
        if not best_conv:
            warnings.warn(
                f"Partial structural change iterative procedure did not "
                f"converge for m={m} after {max_iter} iterations. "
                f"Results may be unreliable.",
                stacklevel=3,
            )
        breaks.append(best_vb)
        ssr[m - 1] = best_ssr
    return breaks, ssr, ssr0


def _partial_iterate(
    y: FloatArray,
    z: FloatArray,
    x: FloatArray,
    m: int,
    h: int,
    beta0: FloatArray,
    tol: float,
    max_iter: int,
) -> tuple[IntArray, float, bool]:
    """Iterate between the break dates and ``beta`` from the start ``beta0``.

    Returns the break vector, the SSR and whether the iteration converged;
    an empty vector and NaN when no admissible partition exists or the design
    is singular at the dates reached.
    """
    empty: IntArray = np.empty(0, dtype=np.int64)
    beta = np.asarray(beta0, dtype=np.float64).copy()
    vb = empty
    last = np.nan
    converged = False
    for _ in range(max_iter):
        ystar = y - x @ beta
        vb = partition_all(ssr_triangle(ystar, z, h), m, h)[0][m - 1]
        if vb.size == 0:
            return empty, np.nan, False
        try:
            _, beta_new, _, ssr_m, _ = estimate(y, z, x, vb)
        except np.linalg.LinAlgError:
            return empty, np.nan, False
        converged = bool(
            not np.isnan(last)
            and abs(ssr_m - last) <= tol * (1 + abs(ssr_m))
            and np.max(np.abs(beta_new - beta))
            <= math.sqrt(tol) * (1 + np.max(np.abs(beta)))
        )
        beta = beta_new
        last = ssr_m
        if converged:
            break
    return vb, float(last), converged


def one_break_date(
    y: FloatArray, z: FloatArray, x: FloatArray | None, h: int
) -> tuple[int, float]:
    """SSR-minimizing single break in the sample ``(y, z, x)``.

    Returns the last index of the first regime and the minimized SSR, or
    ``(-1, nan)`` when the sample is shorter than ``2h``. With fixed
    regressors the minimizer is found by direct search over the admissible
    dates (Perron's ``onebp``), which is exact; without them from the SSRs of
    the expanding samples run forward and backward (Perron's ``partione``).
    """
    n = len(y)
    if n < 2 * h:
        return -1, np.nan
    if not _has_x(x):
        # SSR(0..b) + SSR(b+1..n-1) for b = h-1, ..., n-h-1 (Perron's partione)
        forward = ssr_expanding(y, z, h)
        backward = ssr_expanding(y[::-1], z[::-1], h)[::-1]
        total = forward[h - 1 : n - h] + backward[h : n - h + 1]
        if np.all(np.isnan(total)):
            return -1, np.nan
        best = int(np.nanargmin(total))
        return h - 1 + best, float(total[best])
    assert x is not None
    # unit-length columns: the SSR is unchanged and the rank decision does not
    # depend on the units of the regressors
    zn, xn = _unit_columns(z)[0], _unit_columns(x)[0]
    best_b, best_ssr = -1, np.nan
    for b in range(h - 1, n - h):
        w = np.column_stack([zbar(zn, [b]), xn])
        par, _, rank, _ = np.linalg.lstsq(w, y, rcond=None)
        if rank < w.shape[1]:
            # a regressor has no variation in one of the two regimes, or a
            # fixed regressor is a step at this date: not an admissible date
            continue
        resid = y - w @ par
        ssr_b = float(resid @ resid)
        if np.isnan(best_ssr) or ssr_b < best_ssr:
            best_b, best_ssr = b, ssr_b
    return best_b, float(best_ssr)


# ---------------------------------------------------------------------------
# Long-run covariance (quadratic spectral kernel, Andrews 1991 bandwidth)
# ---------------------------------------------------------------------------


def kernel_qs(xv: FloatArray) -> FloatArray:
    """Quadratic spectral kernel (Andrews 1991)."""
    xv = np.asarray(xv, dtype=np.float64)
    out = np.ones_like(xv)
    nz = xv != 0
    d = 6.0 * np.pi * xv[nz] / 5.0
    out[nz] = 3.0 * (np.sin(d) / d - np.cos(d)) / (d * d)
    return out


def bandwidth_andrews_qs(v: FloatArray) -> float:
    """Andrews (1991) automatic bandwidth for the QS kernel.

    AR(1) approximation for each column, unit weights:
    ``S_T = 1.3221 (alpha(2) T)^{1/5}``, as in Perron's ``bandw``.
    """
    nobs, d = v.shape
    num = den = 0.0
    for k in range(d):
        v0, v1 = v[:-1, k], v[1:, k]
        rho = float(v0 @ v1) / float(v0 @ v0)
        e = v1 - rho * v0
        sig2 = float(e @ e) / (nobs - 1)
        num += 4.0 * rho**2 * sig2**2 / (1.0 - rho) ** 8
        den += sig2**2 / (1.0 - rho) ** 4
    return float(1.3221 * (num / den * nobs) ** 0.2)


def long_run(v: FloatArray, prewhiten: bool) -> FloatArray:
    """Long-run covariance of the columns of ``v``, as in Perron's program.

    Quadratic spectral kernel with the Andrews (1991) bandwidth based on an
    AR(1) approximation, all T - 1 autocovariances, small-sample factor
    ``1/(T - d)``, and optional VAR(1) prewhitening with recoloring (Andrews
    and Monahan 1992); see Bai and Perron (2003, Section 4.1).

    Notes
    -----
    With prewhitening, ``v_t = A v_{t-1} + e_t`` is estimated equation by
    equation and the estimate is recolored as
    ``(I - A)^{-1} J (I - A)^{-1}'``, where row i of ``A`` holds the
    coefficients of equation i. Perron's GAUSS program (``correct``) stores
    those coefficients in column i (``bmat[.,i]=b``), i.e. it recolors with
    the transpose of ``A``. The two coincide for a single series (d = 1). For
    d > 1 the transposed form is the long-run covariance implied by the
    VAR(1) only when ``A`` is symmetric, and it can be far off when the
    series have different scales; the form used here is the one in the R port
    of the program (``mbreaks``).
    """
    d = v.shape[1]
    vs = v
    a_mat = np.zeros((d, d))
    if prewhiten:
        # v_t = A v_{t-1} + e_t: row i of A holds the coefficients of equation i
        lag, lead = v[:-1], v[1:]
        a_mat = np.linalg.lstsq(lag, lead, rcond=None)[0].T
        vs = lead - lag @ a_mat.T
    n = vs.shape[0]
    bw = bandwidth_andrews_qs(vs)
    w = kernel_qs(np.arange(1, n) / bw)
    jhat: FloatArray = vs.T @ vs
    for j in range(1, n):
        g = vs[j:].T @ vs[: n - j]
        jhat += w[j - 1] * (g + g.T)
    jhat /= n - d
    if prewhiten:
        inv = np.linalg.inv(np.eye(d) - a_mat)
        jhat = inv @ jhat @ inv.T
    return jhat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def cov_delta(
    y: FloatArray,
    z: FloatArray,
    x: FloatArray | None,
    vb: IntArray | list[int],
    opt: BPOptions,
) -> FloatArray:
    """Covariance matrix of the regime coefficients ``delta``.

    The cases follow Bai and Perron (2003, Section 5.1) and Perron's
    ``pvdel``:

    ==========  =======  =======  ==========================================
    robust      hetdat   hetvar   estimate
    ==========  =======  =======  ==========================================
    False       True     False    ``sigma^2 (Zbar'Zbar)^{-1}``
    False       True     True     regime variances, regime moment matrices
    False       False    False    ``sigma^2 (Lambda x Z'Z)^{-1}`` (p = 0)
    False       False    True     regime variances, common ``Z'Z`` (p = 0)
    True        True     True     HAC regime by regime (p = 0)
    True        True     False    one HAC estimate, scaled by regime share
    ==========  =======  =======  ==========================================

    With fixed regressors (``p > 0``) the regime-blocked ``z`` is first
    projected off ``x``, and ``hetdat=False`` is not available.
    """
    nobs, q = z.shape
    b = np.asarray(vb, dtype=np.int64).ravel()
    m = len(b)
    has_x = _has_x(x)
    _, _, resid, _, zb = estimate(y, z, x, b)
    reg: FloatArray = zb
    if has_x:
        assert x is not None
        reg = zb - x @ np.linalg.solve(x.T @ x, x.T @ zb)
    bd = bounds(b, nobs)
    sig = np.zeros(m + 1)
    lam = np.zeros(m + 1)
    for i in range(m + 1):
        ui = resid[bd[i] : bd[i + 1]]
        sig[i] = float(ui @ ui) / len(ui)
        lam[i] = len(ui) / nobs
    rr: FloatArray = np.linalg.inv(reg.T @ reg)
    sig_kron = np.kron(np.diag(sig), np.eye(q))

    if not opt.robust:
        s2 = float(resid @ resid) / nobs
        common_var: FloatArray = s2 * rr
        if not has_x:
            if opt.hetdat and not opt.hetvar:
                return common_var
            if opt.hetdat and opt.hetvar:
                return sig_kron @ rr
            common: FloatArray = np.linalg.inv(np.kron(np.diag(lam), z.T @ z))
            if not opt.hetvar:
                scaled: FloatArray = s2 * common
                return scaled
            return sig_kron @ common
        if not opt.hetdat:
            raise ValueError(
                "hetdat=False is not available with fixed regressors (exog)"
            )
        if not opt.hetvar:
            return common_var
        g = np.zeros_like(rr)
        for i in range(m + 1):
            wi = reg[bd[i] : bd[i + 1]]
            g += sig[i] * (wi.T @ wi)
        return rr @ g @ rr

    if not opt.hetdat:
        raise ValueError("hetdat=False is not available with robust=True")
    if not has_x:
        if opt.hetvar:
            hac = np.zeros(((m + 1) * q, (m + 1) * q))
            for i in range(m + 1):
                r1, r2 = bd[i], bd[i + 1]
                block = slice(i * q, (i + 1) * q)
                hac[block, block] = (r2 - r1) * long_run(
                    z[r1:r2] * resid[r1:r2, None], opt.prewhiten
                )
            return rr @ hac @ rr
        lr = long_run(z * resid[:, None], opt.prewhiten)
        out_lr: FloatArray = nobs * rr @ np.kron(np.diag(lam), lr) @ rr
        return out_lr
    out: FloatArray = nobs * rr @ long_run(reg * resid[:, None], opt.prewhiten) @ rr
    return out


def sup_f(
    y: FloatArray,
    z: FloatArray,
    x: FloatArray | None,
    vb: IntArray | list[int],
    opt: BPOptions,
) -> float:
    """F test of no break against the k breaks in ``vb``, scaled by q.

    Bai and Perron (2003, eq. 10) without the division by ``q``:

        F = (T - (k+1)q - p) / (T k) * delta' R' (R V R')^{-1} R delta,

    where ``(R delta)' = (delta_1' - delta_2', ..., delta_k' - delta_{k+1}')``.
    Evaluated at the global SSR minimizer it is the sup F(k|0) statistic.
    """
    nobs = len(y)
    q = z.shape[1]
    k = len(np.asarray(vb).ravel())
    p = x.shape[1] if _has_x(x) and x is not None else 0
    rsub = np.zeros((k, k + 1))
    for j in range(k):
        rsub[j, j], rsub[j, j + 1] = -1.0, 1.0
    r = np.kron(rsub, np.eye(q))
    delta = estimate(y, z, x, vb)[0]
    v = cov_delta(y, z, x, vb, opt)
    rd = r @ delta
    fstar = float(rd @ np.linalg.solve(r @ v @ r.T, rd))
    return float((nobs - (k + 1) * q - p) * fstar / (nobs * k))


def one_break_sup_f(
    y: FloatArray,
    z: FloatArray,
    x: FloatArray | None,
    start: int,
    end: int,
    h: int,
    opt: BPOptions,
) -> tuple[float, int]:
    """One-break test inside the segment ``start..end`` (inclusive).

    The break date is the SSR minimizer within the segment and the F statistic
    is evaluated there, with the segment's data only. Returns the statistic
    and the last index (in full-sample units) of the first sub-regime, or
    ``(nan, -1)`` if the segment is shorter than ``2h`` or has no admissible
    break date. Fixed regressors that are redundant within the segment are
    left out (:func:`independent_fixed`).
    """
    ys, zs = y[start : end + 1], z[start : end + 1]
    xs = x[start : end + 1] if _has_x(x) and x is not None else None
    xs = independent_fixed(zs, xs)
    b, _ = one_break_date(ys, zs, xs, h)
    if b < 0:
        return np.nan, -1
    return sup_f(ys, zs, xs, [b], opt), start + b


def seq_f(
    y: FloatArray,
    z: FloatArray,
    x: FloatArray | None,
    vb: IntArray | list[int],
    h: int,
    opt: BPOptions,
) -> tuple[float, int]:
    """sup F(l+1|l) given the l-break partition ``vb``.

    Bai and Perron (1998; 2003, Section 5.3): the maximum over the l + 1
    regimes of the one-break test inside each regime. Returns the
    statistic and the last index of the regime the additional break would
    end, or ``(nan, -1)`` when no regime is long enough to be split.
    """
    bd = bounds(vb, len(y))
    best, at = np.nan, -1
    for i in range(len(bd) - 1):
        stat, where = one_break_sup_f(y, z, x, int(bd[i]), int(bd[i + 1]) - 1, h, opt)
        if not np.isnan(stat) and (np.isnan(best) or stat > best):
            best, at = stat, where
    return best, at


def repartition(
    y: FloatArray,
    z: FloatArray,
    x: FloatArray | None,
    vb: IntArray | list[int],
    h: int,
) -> IntArray:
    """Repartition of sequentially estimated breaks (Bai 1997).

    Each break is re-estimated as a single break in the sample running from
    the start of its regime to the end of the next one, using the original
    (not yet updated) neighboring dates.
    """
    b = np.asarray(vb, dtype=np.int64).ravel()
    bd = bounds(b, len(y))
    new: IntArray = b.copy()
    for i in range(len(b)):
        start, end = int(bd[i]), int(bd[i + 2]) - 1
        xs = x[start : end + 1] if _has_x(x) and x is not None else None
        xs = independent_fixed(z[start : end + 1], xs)
        nb, _ = one_break_date(y[start : end + 1], z[start : end + 1], xs, h)
        if nb >= 0:
            new[i] = start + nb
    return new


def sequential(
    y: FloatArray,
    z: FloatArray,
    x: FloatArray | None,
    max_breaks: int,
    h: int,
    trimming: float,
    level: float,
    opt: BPOptions,
    do_repartition: bool = False,
) -> tuple[int, IntArray]:
    """Sequential procedure for the number of breaks.

    Bai and Perron (1998; 2003, Sections 5.3-5.4) and Perron's ``sequa``:
    test 0 against 1 break; while sup F(l+1|l), computed on the
    breaks estimated so far (one at a time, not global minimizers), exceeds
    its critical value, add the break of the regime with the largest
    statistic. Stops at the first non-rejection or at ``max_breaks``.

    Returns the number of breaks and their last-index vector.
    """
    q = z.shape[1]
    cv_seq = _cv.seqf_table(level, trimming)
    if cv_seq is None:
        raise ValueError(
            "the sequential procedure needs tabulated critical values: "
            f"trimming must be one of {_cv.TRIMMINGS} and level one of "
            f"{_cv.LEVELS}"
        )
    row = cv_seq[min(q, _cv.MAX_Q) - 1]
    empty: IntArray = np.empty(0, dtype=np.int64)
    stat, where = one_break_sup_f(y, z, x, 0, len(y) - 1, h, opt)
    if np.isnan(stat) or stat < row[0]:
        return 0, empty
    vb: IntArray = np.array([where], dtype=np.int64)
    n_breaks = 1
    while n_breaks < max_breaks:
        stat, where = seq_f(y, z, x, vb, h, opt)
        if np.isnan(stat) or n_breaks >= len(row) or stat < row[n_breaks]:
            break
        vb = np.sort(np.append(vb, where))
        n_breaks += 1
    if do_repartition and n_breaks > 1:
        vb = repartition(y, z, x, vb, h)
    return n_breaks, vb


def info_criteria(
    ssr: FloatArray, nobs: int, q: int, p: int
) -> tuple[FloatArray, FloatArray]:
    """BIC and LWZ for ``m = 0, ..., M`` breaks.

    With ``p* = (m + 1) q + m + p`` (regime coefficients, the ``m`` break
    dates, fixed coefficients):

        BIC(m) = ln(SSR_m / T) + p* ln(T) / T                    (Yao 1988)
        LWZ(m) = ln(SSR_m / (T - p*)) + p* c0 ln(T)^(2 + d0) / T (Liu, Wu and
                                                                  Zidek 1997)

    with ``c0 = 0.299`` and ``d0 = 0.1``. The break dates are estimated
    parameters and are counted: Perron's ``order`` uses
    ``ln(SSR_m/T) + ln(T) m (q + 1)/T`` and
    ``ln(SSR_m/(T - (m+1)q - m)) + m (q + 1) c0 ln(T)^2.1 / T``, which select
    the same ``m`` as the formulas above when there are no fixed regressors.
    With fixed regressors (p > 0) the LWZ denominator here is smaller by
    ``p``, since the fixed coefficients are estimated too; the program does
    not subtract them.
    """
    ssr = np.asarray(ssr, dtype=np.float64)
    m = np.arange(len(ssr))
    pstar = (m + 1) * q + m + p
    log_t = math.log(nobs)
    with np.errstate(divide="ignore", invalid="ignore"):
        bic = np.log(ssr / nobs) + pstar * log_t / nobs
        lwz = (
            np.log(ssr / (nobs - pstar))
            + pstar * LWZ_C0 * log_t ** (2 + LWZ_DELTA0) / nobs
        )
    return bic, lwz


# ---------------------------------------------------------------------------
# Break-date confidence intervals (Bai 1997; Bai and Perron 2003, Section 4.2)
# ---------------------------------------------------------------------------


def cdf_argmax(
    xv: float, bet: float, alph: float, b: float, deld: float, gam: float
) -> float:
    """Distribution function of the argmax process derived in Bai (1997).

    This is the distribution of ``argmax_s V(s)`` in Bai and Perron (2003,
    eq. 7). Port of Perron's ``funcg``.
    """
    if xv <= 0:
        xb = bet * math.sqrt(abs(xv))
        tail = ndtr(-math.sqrt(abs(xv)) / 2)
        head = -math.sqrt(-xv / (2 * math.pi)) * math.exp(xv / 8)
        last = ((2 * bet * bet / alph) - 2 - xv / 2) * tail
        if abs(xb) <= 30:
            return float(
                head
                - (bet / alph) * math.exp(-alph * xv) * ndtr(-bet * math.sqrt(abs(xv)))
                + last
            )
        aa = (
            math.log(bet / alph)
            - alph * xv
            - xb**2 / 2
            - math.log(math.sqrt(2 * math.pi))
            - math.log(xb)
        )
        return float(head - math.exp(aa) * tail + last)
    xb = deld * math.sqrt(xv)
    head = 1 + (b / math.sqrt(2 * math.pi)) * math.sqrt(xv) * math.exp(-b * b * xv / 8)
    last = (2 - b * b * xv / 2 - 2 * deld * deld / gam) * ndtr(-b * math.sqrt(xv) / 2)
    if abs(xb) <= 30:
        return float(
            head
            + (b * deld / gam) * math.exp(gam * xv) * ndtr(-deld * math.sqrt(xv))
            + last
        )
    aa = (
        math.log(b * deld / gam)
        + gam * xv
        - xb**2 / 2
        - math.log(math.sqrt(2 * math.pi))
        - math.log(xb)
    )
    return float(head + math.exp(aa) + last)


def quantiles_argmax(eta: float, phi1: float, phi2: float) -> FloatArray:
    """Quantiles (2.5%, 5%, 95%, 97.5%) of the Bai (1997) distribution.

    Port of Perron's ``cvg`` (bisection on :func:`cdf_argmax`). The program
    searches in [-2000, 2000] and gives up with a warning when a quantile
    lies outside; here the interval is widened until it brackets the
    quantile, which changes nothing when it already does.
    """
    a = phi1 / phi2
    gam = ((phi2 / phi1) + 1) * eta / 2
    b = math.sqrt(phi1 * eta / phi2)
    deld = math.sqrt(phi2 * eta / phi1) + b / 2
    alph = a * (1 + a) / 2
    bet = (1 + 2 * a) / 2
    out = np.zeros(4)
    for i, prob in enumerate((0.025, 0.05, 0.95, 0.975)):
        lo, hi, xx = -2000.0, 2000.0, 0.0
        while lo > -QUANTILE_BOUND and cdf_argmax(lo, bet, alph, b, deld, gam) > prob:
            lo *= 4
        while hi < QUANTILE_BOUND and cdf_argmax(hi, bet, alph, b, deld, gam) < prob:
            hi *= 4
        for _ in range(200):
            xx = lo + (hi - lo) / 2
            crit = cdf_argmax(xx, bet, alph, b, deld, gam) - prob
            if abs(crit) < 1e-6:
                break
            if crit <= 0:
                lo = xx
            else:
                hi = xx
        out[i] = xx
    return out


def break_ci(
    y: FloatArray,
    z: FloatArray,
    x: FloatArray | None,
    vb: IntArray | list[int],
    opt: BPOptions,
) -> IntArray:
    """Confidence intervals for the break dates.

    Bai (1997) and Bai and Perron (2003, Section 4.2, eq. 7), as in Perron's
    ``interval``. For break i with coefficient change ``d``, regime moment
    matrices ``Q_i``, ``Q_{i+1}`` and error variances (or long-run variances
    when ``robust``), the interval inverts the asymmetric limit distribution
    of the break-date estimator. The ``hetq`` and ``hetomega`` options impose
    equal ``Q`` and equal variances across regimes (the special cases listed
    in that section).

    The bounds are integers: "the highest smaller integer for the lower
    bound and the smallest higher integer for the upper bound" (Bai and
    Perron 2003, p. 11), computed as ``floor`` and ``floor + 1`` like the
    program's ``int(.)`` and ``int(.) + 1``.

    Returns an (m, 4) integer array of last-index bounds
    ``[95% lower, 95% upper, 90% lower, 90% upper]``.
    """
    nobs, q = z.shape
    b = np.asarray(vb, dtype=np.int64).ravel()
    m = len(b)
    delta, _, resid, _, _ = estimate(y, z, x, b)
    bd = bounds(b, nobs)
    out = np.zeros((m, 4), dtype=np.int64)
    zu = z * resid[:, None]
    for i in range(1, m + 1):
        d = delta[i * q : (i + 1) * q] - delta[(i - 1) * q : i * q]
        r0, r1, r2 = int(bd[i - 1]), int(bd[i]), int(bd[i + 1])  # [r0, r1), [r1, r2)
        if opt.hetq:
            q1 = z[r0:r1].T @ z[r0:r1] / (r1 - r0)
            q2 = z[r1:r2].T @ z[r1:r2] / (r2 - r1)
        else:
            q1 = q2 = z.T @ z / nobs
        dqd = float(d @ q1 @ d)
        eta = float(d @ q2 @ d) / dqd
        if not opt.robust:
            if opt.hetomega:
                phi1 = float(resid[r0:r1] @ resid[r0:r1]) / (r1 - r0)
                phi2 = float(resid[r1:r2] @ resid[r1:r2]) / (r2 - r1)
            else:
                phi1 = phi2 = float(resid @ resid) / nobs
            scale = dqd / phi1
        else:
            if opt.hetomega:
                om1 = long_run(zu[r0:r1], opt.prewhiten)
                om2 = long_run(zu[r1:r2], opt.prewhiten)
            else:
                om1 = om2 = long_run(zu, opt.prewhiten)
            # phi_1^2 = d'Omega_i d / d'Q_i d, phi_2^2 = d'Omega_{i+1} d / d'Q_{i+1} d
            # (Bai and Perron 2003, Section 4.2; Perron's interval)
            phi1 = float(d @ om1 @ d) / dqd
            phi2 = float(d @ om2 @ d) / float(d @ q2 @ d)
            scale = dqd**2 / float(d @ om1 @ d)
        quant = quantiles_argmax(eta, phi1, phi2)
        out[i - 1, 0] = math.floor(b[i - 1] - quant[3] / scale)
        out[i - 1, 1] = math.floor(b[i - 1] - quant[0] / scale) + 1
        out[i - 1, 2] = math.floor(b[i - 1] - quant[2] / scale)
        out[i - 1, 3] = math.floor(b[i - 1] - quant[1] / scale) + 1
    return out
