"""Numerical parity of ``BaiPerronTest`` with the OxRegimes package.

``tests/reference_data/bp_oxregimes_reference.json`` holds the output of the
OxRegimes Bai-Perron core (``RgBP``) on two public data sets for a range of
options; it is produced by ``generate_bai_perron_ox_reference.ox`` in that
folder. Every statistic, break date, information criterion and confidence
interval is compared here.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from regimes import BaiPerronTest

_REF_DIR = Path(__file__).resolve().parent.parent / "reference_data"
_RTOL = 1e-6


def _load(name: str) -> dict[str, np.ndarray]:
    data = np.genfromtxt(_REF_DIR / name, delimiter=",", names=True)
    return {key: np.asarray(data[key], dtype=float) for key in data.dtype.names}


with (_REF_DIR / "bp_oxregimes_reference.json").open() as _f:
    _REFERENCE: dict[str, dict[str, Any]] = json.load(_f)


def _data(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (y, z, x) for a reference case."""
    if name.startswith("exdata"):
        d = _load("bp_exdata.csv")
        if name.startswith("exdata_const_partial"):
            return d["Y"], d["Constant"], d["X1"]
        x = d["Constant"] if name == "exdata_partial" else None
        return d["Y"], d["X1"], x
    d = _load("bp_simulated_consumption.csv")
    one = np.ones(len(d["dc"]))
    # impulse dummy at the outlier of the simulated data (observation 31)
    impulse = np.zeros(len(one))
    impulse[31] = 1.0
    if name.startswith("consumption_outlier_dummy"):
        return d["dc"], one, np.column_stack([d["dy"], impulse])
    if name == "consumption_slope_outlier_dummy":
        return d["dc"], d["dy"], np.column_stack([one, impulse])
    if name == "consumption_slope_partial":
        return d["dc"], d["dy"], one
    if name.startswith("consumption_partial"):
        return d["dc"], one, d["dy"]
    return d["dc"], np.column_stack([one, d["dy"]]), None


def _arr(values: list[Any]) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


@pytest.mark.parametrize("name", sorted(_REFERENCE))
def test_matches_oxregimes(name: str) -> None:
    ref = _REFERENCE[name]
    y, z, x = _data(name)
    test = BaiPerronTest(y, exog=x, exog_break=z)
    kwargs: dict[str, Any] = {
        "max_breaks": ref["M"],
        "trimming": ref["trimming"],
        "h": ref["h"],
        "robust": bool(ref["robust"]),
        "prewhiten": bool(ref["prewhiten"]),
        "hetdat": bool(ref["hetdat"]),
        "hetvar": bool(ref["hetvar"]),
        "hetomega": bool(ref["hetomega"]),
        "hetq": bool(ref["hetq"]),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = test.fit(selection="bic", **kwargs)
        seq = test.fit(selection="sequential", **kwargs)
        rep = test.fit(selection="sequential", repartition=True, **kwargs)
        seq10 = test.fit(selection="sequential", level=0.10, **kwargs)

    m_max = ref["M"]
    assert res.max_breaks == m_max
    assert res.h == ref["h"]
    assert (res.nobs, res.q, res.p) == (ref["T"], ref["q"], ref["p"])

    np.testing.assert_allclose(
        [res.ssr[m] for m in range(m_max + 1)], _arr(ref["ssr"]), rtol=_RTOL
    )
    for m in range(1, m_max + 1):
        assert list(res.breaks_by_m[m]) == [int(b) for b in ref["breaks_by_m"][m - 1]]
    np.testing.assert_allclose(
        [res.supf_stats[k] for k in range(1, m_max + 1)],
        _arr(ref["supf"]),
        rtol=_RTOL,
    )
    np.testing.assert_allclose(
        [res.seqf_stats[ell] for ell in range(1, m_max)],
        _arr(ref["seqf"]),
        rtol=_RTOL,
    )
    assert [res.seqf_break[ell] for ell in range(1, m_max)] == [
        int(b) for b in ref["seqf_break"]
    ]
    np.testing.assert_allclose(
        [res.bic[m] for m in range(m_max + 1)], _arr(ref["bic"]), rtol=_RTOL
    )
    np.testing.assert_allclose(
        [res.lwz[m] for m in range(m_max + 1)], _arr(ref["lwz"]), rtol=_RTOL
    )

    # selection by BIC, with confidence intervals
    assert res.n_breaks == ref["bic_n"]
    if ref["bic_n"] > 0:
        assert [res.break_ci[b] for b in res.break_indices] == [
            (int(lo), int(hi)) for lo, hi in zip(ref["ci95_lo"], ref["ci95_hi"])
        ]
        assert [res.break_ci_90[b] for b in res.break_indices] == [
            (int(lo), int(hi)) for lo, hi in zip(ref["ci90_lo"], ref["ci90_hi"])
        ]

    # sequential procedure at 5%, with repartition, and at 10%
    assert seq.n_breaks == ref["seq_n"]
    assert list(seq.break_indices) == [int(b) for b in ref["seq_breaks"]]
    assert rep.n_breaks == ref["seq_rep_n"]
    assert list(rep.break_indices) == [int(b) for b in ref["seq_rep_breaks"]]
    assert seq10.n_breaks == ref["seq10_n"]
    assert list(seq10.break_indices) == [int(b) for b in ref["seq10_breaks"]]
    # the sequential result is reported whatever the selection method
    assert res.sequential_n_breaks == ref["seq_n"]
    assert list(res.sequential_breaks) == [int(b) for b in ref["seq_breaks"]]


def test_reference_covers_the_option_space() -> None:
    """The reference includes robust, prewhitened, common-variance and partial cases."""
    refs = _REFERENCE.values()
    assert any(r["robust"] and r["prewhiten"] and r["q"] > 1 for r in refs)
    assert any(r["robust"] and not r["prewhiten"] for r in refs)
    assert any(not r["hetdat"] for r in refs)
    assert any(not r["hetvar"] for r in refs)
    assert any(r["p"] > 0 and r["robust"] for r in refs)
    assert any(r["trimming"] != 0.15 for r in refs)
