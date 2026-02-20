# regimes

[![CI](https://github.com/knightianuncertainty/regimes/actions/workflows/ci.yml/badge.svg)](https://github.com/knightianuncertainty/regimes/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/regimes.svg)](https://badge.fury.io/py/regimes)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for structural break detection and estimation in time-series econometrics, extending [statsmodels](https://www.statsmodels.org/) with robust methods for analyzing regime changes.

## Features

- **Structural Break Tests**: Bai-Perron test for multiple structural breaks with Sup-F, UDmax, and sequential testing procedures
- **Chow Test**: Test for structural breaks at known break points with standard and predictive variants
- **CUSUM Tests**: CUSUM and CUSUM-SQ tests for parameter and variance instability (Brown, Durbin, Evans, 1975)
- **Andrews-Ploberger Test**: SupF, ExpF, and AveF tests for a structural break at unknown date (Andrews, 1993; Andrews & Ploberger, 1994)
- **Markov Regime-Switching**: Markov switching regression, AR, and ADL models with regime number selection and restricted transitions
- **Time-Series Models**: AR, ADL, OLS with HAC standard errors and known break support
- **ADL Models**: Autoregressive Distributed Lag models with flexible lag specification and distributed lag diagnostics
- **Rolling & Recursive Estimation**: Track parameter evolution with fixed or expanding windows
- **Model Selection**: BIC, LWZ criteria for selecting the number of breaks; AIC/BIC lag selection for ADL
- **Visualization**: Plot time series with break lines, regime means, rolling coefficients, and confidence intervals
- **statsmodels Integration**: Familiar `Model.fit() -> Results` API pattern

## Installation

```bash
pip install regimes
```

For development:

```bash
pip install regimes[dev,test]
```

## Quick Start

### Testing for Structural Breaks

```python
import numpy as np
import regimes as rg

# Simulate data with a mean shift
np.random.seed(42)
y = np.concatenate([
    np.random.randn(100),      # regime 1: mean = 0
    np.random.randn(100) + 2,  # regime 2: mean = 2
])

# Test for breaks using Bai-Perron
test = rg.BaiPerronTest(y)
results = test.fit(max_breaks=3)

print(results.summary())
print(f"Detected {results.n_breaks} break(s) at: {results.break_indices}")
```

### Integrated Workflow: Model → Break Detection → Model with Breaks

The recommended workflow for break detection starts with a model, runs the Bai-Perron test, and converts the results back to a model with detected breaks:

```python
import numpy as np
import regimes as rg

# Simulate regression data with a break
np.random.seed(42)
n = 200
X = np.column_stack([np.ones(n), np.random.randn(n)])
y = np.zeros(n)
y[:100] = 1 + 0.5 * X[:100, 1] + np.random.randn(100) * 0.5
y[100:] = 3 + 1.5 * X[100:, 1] + np.random.randn(100) * 0.5

# Define model and fit without breaks
model = rg.OLS(y, X, has_constant=False)
constant_results = model.fit()

# Run Bai-Perron test directly from model
bp_results = model.bai_perron()
print(f"Detected {bp_results.n_breaks} break(s) at: {bp_results.break_indices}")

# Convert to OLS with detected breaks
ols_with_breaks = bp_results.to_ols()
print(ols_with_breaks.summary())

# Compare models visually
rg.plot_params_over_time({
    "Constant parameters": constant_results,
    "Bai-Perron breaks": ols_with_breaks,
})
```

You can also use the explicit class method approach:

```python
# Equivalent to model.bai_perron()
bp_results = rg.BaiPerronTest.from_model(model).fit()
```

### Testing Known Break Points (Chow Test)

```python
import numpy as np
import regimes as rg

# Simulate data with a mean shift at t=100
np.random.seed(42)
y = np.concatenate([
    np.random.randn(100),      # regime 1: mean = 0
    np.random.randn(100) + 2,  # regime 2: mean = 2
])

# Test for a break at the hypothesized date
test = rg.ChowTest(y)
results = test.fit(break_points=100)
print(results.summary())

# Or use the convenience method on a model
model = rg.OLS(y, np.ones((200, 1)), has_constant=False)
results = model.chow_test(break_points=[80, 100])
print(results.summary())
```

### Testing for Parameter Instability (CUSUM)

The **CUSUM test** (Brown, Durbin, Evans, 1975) detects parameter instability without requiring a known break date. The **CUSUM-SQ** variant detects changes in variance:

```python
import numpy as np
import regimes as rg

# Simulate data with a mean shift
np.random.seed(42)
y = np.concatenate([
    np.random.randn(100),      # regime 1: mean = 0
    np.random.randn(100) + 2,  # regime 2: mean = 2
])

# CUSUM test for parameter instability
cusum_test = rg.CUSUMTest(y)
cusum_results = cusum_test.fit(significance=0.05)
print(cusum_results.summary())
fig, ax = cusum_results.plot()
```

CUSUM-SQ for variance instability:

```python
# Simulate data with a variance change
rng = np.random.default_rng(42)
y_var = np.concatenate([rng.normal(0, 1, 100), rng.normal(0, 3, 100)])

# CUSUM-SQ test for variance instability
cusumsq_results = rg.CUSUMSQTest(y_var).fit(significance=0.05)
print(cusumsq_results.summary())
fig, ax = cusumsq_results.plot()
```

Or use the convenience method on any model:

```python
model = rg.OLS(y, np.ones((200, 1)), has_constant=False)
cusum_from_model = model.cusum_test()
print(cusum_from_model.summary())
fig, ax = cusum_from_model.plot()
```

### Testing for a Break at Unknown Date (Andrews-Ploberger)

The **Andrews-Ploberger test** tests for a single structural break at an unknown date, reporting three statistics (SupF, ExpF, AveF) with different power properties:

```python
import numpy as np
import regimes as rg

# Simulate data with a mean shift
rng = np.random.default_rng(42)
y = np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 100)])

# Andrews-Ploberger test
results = rg.AndrewsPlobergerTest(y).fit()
print(results.summary())
fig, ax = results.plot()
```

Or use the convenience method on any model:

```python
model = rg.OLS(y, np.ones((200, 1)), has_constant=False)
ap_results = model.andrews_ploberger()
print(ap_results.summary())
```

### Markov Regime-Switching Models

Markov switching models allow parameters to change across unobserved regimes governed by a Markov chain:

```python
import numpy as np
import regimes as rg

# Simulate data with regime-switching mean
rng = np.random.default_rng(42)
y = np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 100), rng.normal(0, 1, 100)])

# Fit Markov switching regression
model = rg.MarkovRegression(y, k_regimes=2)
results = model.fit()
print(results.summary())

# Visualize smoothed probabilities and regime shading
results.plot_smoothed_probabilities()
results.plot_regime_shading(y=y)
```

Use the convenience method on any existing model:

```python
# One-step conversion from OLS to Markov switching
ols_model = rg.OLS(y, np.ones((300, 1)), has_constant=False)
ms_results = ols_model.markov_switching(k_regimes=2)

# Also works with AR and ADL models
ar_model = rg.AR(y, lags=1)
ms_ar = ar_model.markov_switching(k_regimes=2)
```

Select the number of regimes using information criteria or sequential likelihood ratio tests:

```python
# Regime number selection
selection = rg.RegimeNumberSelection(y, k_max=4, method="bic")
sel_results = selection.fit()
print(sel_results.summary())
sel_results.plot_ic()
```

Test whether transitions are non-recurring (structural break) or allow recurrence:

```python
# Non-recurring regime test (Chib 1998 structure)
nr_test = rg.NonRecurringRegimeTest(y, k_regimes=2)
nr_results = nr_test.fit()
print(nr_results.summary())

# Fit with restricted (non-recurring) transitions directly
restricted = rg.RestrictedMarkovRegression.non_recurring(y, k_regimes=3)
restricted_results = restricted.fit()
results.plot_transition_matrix()
```

### OLS with HAC Standard Errors

```python
import numpy as np
import regimes as rg

# Generate data
n = 200
X = np.column_stack([np.ones(n), np.random.randn(n)])
y = X @ [1, 2] + np.random.randn(n)

# Fit OLS with HAC standard errors
model = rg.OLS(y, X)
results = model.fit(cov_type="HAC")
print(results.summary())
```

### OLS with Variable-Specific Breaks

```python
import numpy as np
import regimes as rg

# Different coefficients can break at different times
n = 200
X = np.random.randn(n)

# True DGP: intercept breaks at t=50, slope breaks at t=100
y = np.zeros(n)
y[:50] = 1.0 + 0.5 * X[:50]
y[50:100] = 2.0 + 0.5 * X[50:100]   # intercept changed
y[100:] = 2.0 + 1.5 * X[100:]       # slope changed
y += 0.3 * np.random.randn(n)

# Estimate with variable-specific breaks
model = rg.OLS(y, X.reshape(-1, 1), has_constant=True, variable_breaks={
    "const": [50],   # intercept breaks at t=50
    "x0": [100],     # slope breaks at t=100
})
results = model.fit(cov_type="HAC")
print(results.summary())
```

### AR Model with Known Breaks

```python
import numpy as np
import regimes as rg

# Simulate AR(1) data
n = 200
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.7 * y[t-1] + np.random.randn()

# Fit AR model with known structural break
model = rg.AR(y, lags=1, breaks=[100])
results = model.fit(cov_type="HAC")
print(results.summary())
```

### ADL (Autoregressive Distributed Lag) Model

```python
import numpy as np
import regimes as rg

# Simulate ADL(1,1) data
np.random.seed(42)
n = 200
x = np.random.randn(n)
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.5 * y[t-1] + 0.3 * x[t] + 0.15 * x[t-1] + np.random.randn()

# Fit ADL model
model = rg.ADL(y, x, lags=1, exog_lags=1)
results = model.fit(cov_type="HAC")
print(results.summary())

# Distributed lag diagnostics
print(f"Cumulative effect: {results.cumulative_effect}")
print(f"Long-run multiplier: {results.long_run_multiplier}")

# Lag selection via information criteria
optimal_p, optimal_q = model.select_lags(max_ar_lags=4, max_exog_lags=4, criterion="bic")
print(f"Optimal specification: ADL({optimal_p},{optimal_q})")
```

ADL with multiple exogenous variables and different lag structures:

```python
# x1: 2 lags (contemporary + L1 + L2), x2: contemporary only
model = rg.ADL(y, X, lags=1, exog_lags={"x1": 2, "x2": 0})
results = model.fit()
```

### Rolling and Recursive Estimation

Track how parameter estimates evolve over time to identify potential instability:

```python
import numpy as np
import regimes as rg

# Generate data
n = 200
X = np.column_stack([np.ones(n), np.random.randn(n)])
y = X @ [1, 2] + np.random.randn(n)

# Define model
model = rg.OLS(y, X, has_constant=False)

# Rolling estimation (fixed window)
rolling_results = model.rolling(window=60).fit()
print(rolling_results.summary())

# Recursive estimation (expanding window)
recursive_results = model.recursive(min_nobs=30).fit()
print(recursive_results.summary())

# Visualize coefficient evolution
fig, axes = rolling_results.plot_coefficients()
```

Rolling and recursive estimation also work with AR models:

```python
ar_model = rg.AR(y, lags=1)
ar_rolling = ar_model.rolling(window=60).fit()
ar_recursive = ar_model.recursive(min_nobs=30).fit()
```

### Parameter Naming with Breaks

When structural breaks are specified, parameter names include regime suffixes using **1-indexed** numbering:

```python
model = rg.AR(y, lags=1, breaks=[100])
results = model.fit()
print(results.param_names)
# ['const_regime1', 'y.L1_regime1', 'const_regime2', 'y.L1_regime2']
```

- `const_regime1`, `y.L1_regime1`: Parameters for the first regime (before the break)
- `const_regime2`, `y.L1_regime2`: Parameters for the second regime (after the break)

### Visualization

```python
import regimes as rg

# Plot data with detected breaks
fig, ax = rg.plot_breaks(y, breaks=results.break_indices, shade_regimes=True)

# Plot with regime means
fig, ax = rg.plot_regime_means(y, breaks=results.break_indices)
```

### Plotting Style

All plots use a consistent Economist/FT-inspired style with no top/right spines, horizontal gridlines only, and a professional color palette:

```python
from regimes.visualization import set_style, use_style, REGIMES_COLORS

# Apply style globally
set_style()

# Or use as context manager
with use_style():
    fig, ax = rg.plot_breaks(y, breaks=[100])

# Access the color palette
print(REGIMES_COLORS['blue'])   # '#006BA2' - primary series
print(REGIMES_COLORS['red'])    # '#DB444B' - significance/rejection
print(REGIMES_COLORS['grey'])   # '#758D99' - break dates, context
```

### Misspecification Diagnostics

PcGive-style diagnostic plots for assessing model fit:

```python
# 2×2 panel: Actual vs Fitted, Residual Distribution, Scaled Residuals, ACF/PACF
fig, axes = results.plot_diagnostics()

# Individual plots for customization
fig, ax = rg.plot_actual_fitted(results)
fig, ax = rg.plot_scaled_residuals(results)
fig, ax = rg.plot_residual_distribution(results)
fig, axes = rg.plot_residual_acf(results, nlags=15)
```

## API Reference

### Models

| Class | Description |
|-------|-------------|
| `OLS` | Ordinary Least Squares with robust standard errors and variable-specific breaks |
| `AR` | Autoregressive model with break support |
| `ADL` | Autoregressive Distributed Lag model with flexible lag specification |
| `MarkovRegression` | Markov regime-switching regression (wraps statsmodels) |
| `MarkovAR` | Markov regime-switching autoregression |
| `MarkovADL` | Markov regime-switching ADL model |
| `RestrictedMarkovRegression` | Markov regression with restricted transition probabilities |
| `RestrictedMarkovAR` | Markov AR with restricted transition probabilities |

All models (`OLS`, `AR`, `ADL`) have:
- `.bai_perron()` method for integrated break detection
- `.chow_test(break_points)` method for testing breaks at known dates
- `.cusum_test()` method for CUSUM parameter instability test
- `.cusum_sq_test()` method for CUSUM-SQ variance instability test
- `.andrews_ploberger()` method for testing breaks at unknown date
- `.markov_switching(k_regimes)` method for one-step Markov switching estimation
- `.rolling(window)` method for rolling window estimation
- `.recursive(min_nobs)` method for recursive (expanding window) estimation

`ADL` additionally has:
- `.select_lags()` method for AIC/BIC lag selection
- `cumulative_effect` and `long_run_multiplier` properties on results

### Rolling/Recursive Estimation

| Class | Description |
|-------|-------------|
| `RollingOLS` | Rolling OLS with fixed window |
| `RecursiveOLS` | Recursive OLS with expanding window |
| `RollingAR` | Rolling AR with fixed window |
| `RecursiveAR` | Recursive AR with expanding window |
| `RollingADL` | Rolling ADL with fixed window |
| `RecursiveADL` | Recursive ADL with expanding window |

**Key methods:**
- `results.summary()` - Text summary with parameter statistics over time
- `results.plot_coefficients()` - Plot coefficient evolution with confidence bands

### Tests

| Class | Description |
|-------|-------------|
| `BaiPerronTest` | Bai-Perron test for multiple structural breaks |
| `ChowTest` | Chow test for breaks at known break points (standard and predictive) |
| `CUSUMTest` | CUSUM test for parameter instability via recursive residuals |
| `CUSUMSQTest` | CUSUM-of-squares test for variance instability |
| `AndrewsPlobergerTest` | Andrews-Ploberger SupF/ExpF/AveF test for break at unknown date |
| `NonRecurringRegimeTest` | LR test: non-recurring (structural break) vs unrestricted Markov transitions |
| `SequentialRestrictionTest` | GETS-style sequential restriction of transition probabilities |
| `RegimeNumberSelection` | Select number of regimes by IC (AIC/BIC/HQIC) or sequential LRT |

**Key methods:**
- `BaiPerronTest.from_model(model)` - Create test from OLS or AR model
- `BaiPerronResults.to_ols()` - Convert results to OLS with detected breaks
- `ChowTest.from_model(model)` - Create test from OLS, AR, or ADL model
- `.chow_test(break_points)` - Convenience method on all model classes
- `.cusum_test()` / `.cusum_sq_test()` - Convenience methods on all model classes
- `AndrewsPlobergerTest.from_model(model)` - Create test from OLS, AR, or ADL model
- `.andrews_ploberger()` - Convenience method on all model classes

### Visualization

| Function | Description |
|----------|-------------|
| `plot_breaks` | Plot time series with break lines |
| `plot_regime_means` | Plot time series with regime-specific means |
| `plot_break_confidence` | Plot breaks with confidence intervals |
| `plot_params_over_time` | Compare parameter estimates across models/regimes |
| `plot_rolling_coefficients` | Plot rolling/recursive coefficient estimates |
| `plot_diagnostics` | PcGive-style 2×2 diagnostic panel (also available as `results.plot_diagnostics()`) |
| `plot_actual_fitted` | Actual vs fitted values over time |
| `plot_scaled_residuals` | Vertical index plot of residuals/σ |
| `plot_residual_distribution` | Histogram with N(0,1) overlay |
| `plot_residual_acf` | ACF and PACF bar plots |
| `plot_cusum` | CUSUM statistic with critical boundaries |
| `plot_cusum_sq` | CUSUM-SQ statistic with critical boundaries |
| `plot_f_sequence` | Andrews-Ploberger F-statistic sequence with critical value line |
| `plot_smoothed_probabilities` | Smoothed regime probabilities (one subplot per regime) |
| `plot_regime_shading` | Time series with regime-colored background shading |
| `plot_transition_matrix` | Heatmap of transition probability matrix |
| `plot_parameter_time_series` | Regime-dependent parameter as step function over time |
| `plot_ic` | Information criteria vs number of regimes |

### Style Utilities

| Function/Constant | Description |
|-------------------|-------------|
| `REGIMES_COLORS` | Dictionary of Economist-inspired colors (blue, red, teal, green, gold, grey, mauve) |
| `REGIMES_COLOR_CYCLE` | List of colors for matplotlib's prop_cycle |
| `set_style()` | Apply regimes style globally |
| `use_style()` | Context manager for temporary style application |
| `add_break_dates()` | Helper to add vertical break lines to any plot |
| `add_confidence_band()` | Helper to add translucent confidence bands |
| `shade_regimes()` | Helper to shade alternating regimes |
| `label_line_end()` | Helper to label lines at their endpoints (replaces legend) |
| `add_source()` | Helper to add source annotation at bottom-left |

## Covariance Types

All regression models support multiple covariance estimators:

- `"nonrobust"`: Standard OLS covariance
- `"HC0"`, `"HC1"`, `"HC2"`, `"HC3"`: Heteroskedasticity-robust (White)
- `"HAC"`: Heteroskedasticity and autocorrelation consistent (Newey-West)

## Testing

The package includes a comprehensive test suite with 880 tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=regimes

# Run property-based tests only
pytest tests/test_property/
```

### Property-Based Testing

regimes uses [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing, which verifies that mathematical invariants hold for *any* valid input:

| Test Module | Properties Verified |
|-------------|---------------------|
| `test_ols_properties.py` | Residuals sum to zero, R² bounds, TSS decomposition, CI contains estimates, AIC/BIC formulas |
| `test_ar_properties.py` | Root count = AR order, stationarity ↔ roots outside unit circle, effective sample size |
| `test_adl_properties.py` | Cumulative effect = Σβ, long-run multiplier = Σβ/(1-Σφ), LRM undefined when non-stationary |
| `test_rolling_properties.py` | Shape alignment, NaN alignment, recursive monotonicity |

## References

- Bai, J., & Perron, P. (1998). Estimating and testing linear models with multiple structural changes. *Econometrica*, 66(1), 47-78.
- Bai, J., & Perron, P. (2003). Computation and analysis of multiple structural change models. *Journal of Applied Econometrics*, 18(1), 1-22.
- Brown, R. L., Durbin, J., & Evans, J. M. (1975). Techniques for testing the constancy of regression relationships over time. *Journal of the Royal Statistical Society: Series B*, 37(2), 149-192.
- Chow, G. C. (1960). Tests of equality between sets of coefficients in two linear regressions. *Econometrica*, 28(3), 591-605.
- Andrews, D. W. K. (1993). Tests for parameter instability and structural change with unknown change point. *Econometrica*, 61(4), 821-856.
- Andrews, D. W. K. & Ploberger, W. (1994). Optimal tests when a nuisance parameter is present only under the alternative. *Econometrica*, 62(6), 1383-1414.
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384.
- Chib, S. (1998). Estimation and comparison of multiple change-point models. *Journal of Econometrics*, 86(2), 221-241.
- Andrews, D. W. K. (2001). Testing when a parameter is on the boundary of the maintained hypothesis. *Econometrica*, 69(3), 683-734.

## License

MIT License. See [LICENSE](LICENSE) for details.
