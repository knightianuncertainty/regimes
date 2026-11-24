"""Convert the coefficient tables of Hansen (1997) into a Python module.

Source: the R programs that accompany B. E. Hansen, "Approximate Asymptotic
P Values for Structural-Change Tests", Journal of Business and Economic
Statistics 15 (1997), 60-67: files pv_sup.R, pv_exp.R and pv_ave.R in
https://users.ssc.wisc.edu/~bhansen/progs/jbes_97r.zip (downloaded
2026-11-24). The R files are read as text; nothing in them is executed.

Usage: python make_hansen_tables.py <folder with the R files> <output.py>

The output is formatted with ``ruff format`` before it is committed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

src, out = sys.argv[1], sys.argv[2]
number = r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?"
block = re.compile(
    r"if \(k == (\d+)\) beta <- matrix\(\s*c\((.*?)\),\s*(\d+)\s*,\s*(\d+)\s*,byrow=TRUE\)",
    re.S,
)

tables: dict[str, dict[int, list[list[str]]]] = {}
for kind in ("sup", "exp", "ave"):
    text = Path(src, f"pv_{kind}.R").read_text(encoding="latin-1")
    found = {}
    for match in block.finditer(text):
        k, body, nrow, ncol = (
            int(match.group(1)),
            match.group(2),
            int(match.group(3)),
            int(match.group(4)),
        )
        values = re.findall(number, body)
        assert len(values) == nrow * ncol, (kind, k, len(values), nrow, ncol)
        assert nrow == 25, (kind, k, nrow)
        found[k] = [values[r * ncol : (r + 1) * ncol] for r in range(nrow)]
    assert sorted(found) == list(range(1, 41)), (kind, sorted(found))
    tables[kind] = found

lines = [
    '"""Coefficients of Hansen (1997) for p-values of structural-change tests.',
    "",
    "GENERATED FILE - do not edit by hand.",
    "",
    "The numbers are the coefficient tables in the R programs that accompany",
    'B. E. Hansen, "Approximate Asymptotic P Values for Structural-Change',
    'Tests", Journal of Business and Economic Statistics 15 (1997), 60-67',
    "(files ``pv_sup.R``, ``pv_exp.R`` and ``pv_ave.R``, dated April 2011, from",
    "https://users.ssc.wisc.edu/~bhansen/progs/jbes_97.html, downloaded",
    "2026-11-24). They were converted by the script",
    "``tests/reference_data/make_hansen_tables.py``, which reads the R files as",
    "text.",
    "",
    "Each table is keyed by the number of coefficients tested for constancy,",
    "``q = 1, ..., 40``, and holds 25 rows, one for each value of ``pi_0`` from",
    "0.49 down to 0.01 in steps of 0.02. A row holds the coefficients of a",
    "polynomial in the statistic followed by the degrees of freedom of the",
    "chi-squared distribution the polynomial is referred to; see",
    "``regimes.tests._hansen_pvalues``.",
    '"""',
    "",
    "from __future__ import annotations",
    "",
    "MAX_Q = 40",
    "N_PI0 = 25",
    "",
]
for kind in ("sup", "exp", "ave"):
    lines.append(f"{kind.upper()}: dict[int, tuple[tuple[float, ...], ...]] = {{")
    for k in range(1, 41):
        lines.append(f"    {k}: (")
        lines.extend("        (" + ", ".join(row) + ")," for row in tables[kind][k])
        lines.append("    ),")
    lines.append("}")
    lines.append("")
Path(out).write_text("\n".join(lines), encoding="utf-8", newline="\n")
print({kind: {k: len(tables[kind][k][0]) for k in (1, 2, 3, 4, 40)} for kind in tables})
