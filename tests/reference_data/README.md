# Reference Data for Cross-Validation

This directory contains datasets and an R script used to validate the
Python `BaiPerronTest` implementation against R's `strucchange` package
(Zeileis et al., 2002). The Python test
`tests/test_tests/test_bai_perron_reference.py` loads these CSV files and
asserts that break locations, SSR values, and BIC model selection match the
R reference within `rtol=1e-6`.

## Contents

| File | Description |
|------|-------------|
| `generate_bai_perron_reference.R` | R script that generates the CSV datasets and prints reference values |
| `dgp1_mean_shift.csv` | DGP 1 — simple mean shift (`y ~ 1`, n=200, 1 true break at t=100) |
| `dgp2_regression_break.csv` | DGP 2 — regression break (`y ~ x`, n=200, intercept+slope break at t=100) |
| `dgp3_two_breaks.csv` | DGP 3 — two mean shifts (`y ~ 1`, n=240, breaks at t=80 and t=160) |

## Bai-Perron: Perron's program, OxRegimes and the published results

| File | Description |
|------|-------------|
| `bp_exdata.csv` | Example data distributed with Pierre Perron's Bai-Perron program (T = 101; columns `Constant`, `Y`, `X1`). The expected values in `test_bp_core.py::TestPerronProgram` were produced by that program |
| `bp_simulated_consumption.csv` | Simulated data of the OxRegimes examples (T = 132; columns `dc`, `dy`) |
| `generate_bai_perron_ox_reference.ox` | Ox script that runs the OxRegimes Bai-Perron core on the two files above for 16 option sets |
| `bp_oxregimes_reference.json` | Its output: break dates, SSR, sup F, sup F(l+1\|l), BIC, LWZ, sequential results and confidence intervals, compared in `test_bai_perron_ox_parity.py` at `rtol=1e-6` |
| `bp2003_real_interest_rate.csv` | US ex-post real interest rate, 1961:1-1986:3 (`real.dat` from the Journal of Applied Econometrics data archive entry of Bai and Perron 2003), used by `test_bai_perron_published.py` to reproduce Table I of that paper |

To regenerate the OxRegimes reference (needs Ox and the OxRegimes package; `<packages>` is the folder containing `OxRegimes`):

```bash
cd tests/reference_data
oxl -i<packages> generate_bai_perron_ox_reference.ox
```

The critical values in `src/regimes/tests/_bp_critical_values.py` were dumped from the OxRegimes tables and agree entry by entry with those in the R package `mbreaks`.

## Data Generating Processes

### DGP 1 — Mean Shift
```
y_t = 1 + e_t,  t = 1,...,100
y_t = 4 + e_t,  t = 101,...,200
e_t ~ N(0, 0.25),  seed = 42
```

### DGP 2 — Regression Break
```
y_t = 1 + 0.5·x_t + e_t,  t = 1,...,100
y_t = 3 + 1.5·x_t + e_t,  t = 101,...,200
x_t ~ N(0,1), e_t ~ N(0, 0.09),  seed = 123
```

### DGP 3 — Two Mean Shifts
```
y_t = 0 + e_t,  t = 1,...,80
y_t = 3 + e_t,  t = 81,...,160
y_t = 1 + e_t,  t = 161,...,240
e_t ~ N(0, 0.25),  seed = 7
```

## How to Reproduce

1. Install R (≥ 4.0) and the `strucchange` package:
   ```r
   install.packages("strucchange")
   ```

2. From the repo root, run the R script:
   ```bash
   Rscript tests/reference_data/generate_bai_perron_reference.R
   ```
   This regenerates the CSV files and prints break locations, RSS tables,
   BIC values, and segment coefficients.

3. Run the Python cross-validation tests:
   ```bash
   pytest tests/test_tests/test_bai_perron_reference.py -v
   ```

## Why CSV Files?

R and Python use different random number generators, so the same seed
produces different data. The R script generates the data once and saves it
to CSV. Both R (`strucchange::breakpoints`) and Python (`BaiPerronTest`)
then operate on the **identical** dataset, making the comparison
deterministic.

## References

- Zeileis, A., Leisch, F., Hornik, K., & Kleiber, C. (2002).
  strucchange: An R package for testing for structural change in linear
  regression models. *Journal of Statistical Software*, 7(2), 1–38.
- Bai, J., & Perron, P. (2003). Computation and analysis of multiple
  structural change models. *Journal of Applied Econometrics*, 18(1), 1–22.
