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
