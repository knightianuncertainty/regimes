# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **GitHub repository**: Published at [github.com/knightianuncertainty/regimes](https://github.com/knightianuncertainty/regimes)
  - CI pipeline: ruff lint/format, mypy (advisory), tests on Python 3.10/3.11/3.12, package build
  - Git tag `v0.2.0` for hatch-vcs versioning
- **Chow test for structural breaks at known break points**
  - `ChowTest` class with standard and predictive variants
  - `ChowTestResults` dataclass with F-statistics, p-values, degrees of freedom
  - Auto-detection of predictive variant when sub-samples are too small
  - `ChowTest.from_model()` class method for OLS, AR, and ADL models
  - `.chow_test()` convenience method on OLS, AR, and ADL models
  - Partial coefficient testing via `exog` vs `exog_break` split
  - 48 new tests across 11 test classes
- **CUSUM and CUSUM-SQ tests for parameter and variance instability**
  - `CUSUMTest` class: cumulative sum of recursive residuals (Brown, Durbin, Evans, 1975)
  - `CUSUMSQTest` class: cumulative sum of squared recursive residuals for variance instability
  - `CUSUMResults` / `CUSUMSQResults` dataclasses with statistic path, boundaries, summary
  - Recursive residual computation via Sherman-Morrison-Woodbury updating
  - `plot_cusum()` and `plot_cusum_sq()` visualization functions
  - `.cusum_test()` and `.cusum_sq_test()` convenience methods on OLS, AR, and ADL models
  - `.plot()` convenience method on results objects
  - 74 new tests (662 total)
- **Andrews-Ploberger test for structural breaks at unknown date**
  - `AndrewsPlobergerTest` class with SupF, ExpF, and AveF statistics (Andrews, 1993; Andrews & Ploberger, 1994)
  - `AndrewsPlobergerResults` dataclass with critical values, coarse p-values, and F-sequence
  - Critical value tables for q=1–10, trimming 0.05–0.20
  - `plot_f_sequence()` visualization function for F-statistic path
  - `AndrewsPlobergerTest.from_model()` class method for OLS, AR, and ADL models
  - `.andrews_ploberger()` convenience method on OLS, AR, and ADL models
  - 56 new tests (718 total)
- **Markov regime-switching models**
  - `MarkovRegression` class wrapping `statsmodels.tsa.regime_switching.MarkovRegression` with regime ordering, multi-start optimization, and regimes-style results
  - `MarkovAR` class wrapping `statsmodels.tsa.regime_switching.MarkovAutoregression` with switching AR coefficients
  - `MarkovADL` class: Markov switching ADL model (builds ADL design matrix and passes to statsmodels `MarkovRegression`)
  - `MarkovSwitchingResultsBase`, `MarkovRegressionResults`, `MarkovARResults`, `MarkovADLResults` dataclasses with transition matrix, smoothed/filtered/predicted probabilities, expected durations, regime assignments, and information criteria
  - `RestrictedMarkovRegression` and `RestrictedMarkovAR` for imposing fixed transition probability entries via softmax redistribution
  - `RestrictedMarkovRegression.non_recurring()` / `RestrictedMarkovAR.non_recurring()` factory methods for Chib (1998) upper-triangular non-recurring transition structure
  - `NonRecurringRegimeTest`: LR test of H0 (non-recurring structural break) vs H1 (unrestricted Markov switching) with chi-bar-squared or bootstrap p-values
  - `SequentialRestrictionTest`: GETS-style algorithm for sequentially restricting transition probabilities with Holm-Bonferroni multiple testing correction
  - `RegimeNumberSelection`: select number of regimes K by information criteria (AIC/BIC/HQIC) or sequential LRT, with K=1 using standard OLS/AR
  - 5 Markov visualization functions: `plot_smoothed_probabilities`, `plot_regime_shading`, `plot_transition_matrix`, `plot_parameter_time_series`, `plot_ic`
  - `.markov_switching(k_regimes)` convenience method on `OLS`, `AR`, and `ADL` for one-step conversion to Markov switching
  - `MarkovRegression.from_model()`, `MarkovAR.from_model()`, `MarkovADL.from_model()` class methods for explicit construction from existing models
  - 98 new tests (816 total)

### Changed

- Fixed all ruff lint errors and applied ruff format across codebase
- Removed non-existent `types-matplotlib` from dev dependencies
- Removed deprecated `numpy.typing.mypy_plugin` from mypy config
- Excluded auto-generated `_version.py` from ruff checks

## [0.2.0] - 2025-02-05

### Added

- **Plotting Style System**: Consistent Economist/FT-inspired visual identity for all plots (PLOTTING_STYLE.md v1.0)
  - New `visualization/style.py` module with centralized style configuration
  - `REGIMES_COLORS` palette: blue, red, teal, green, gold, grey, mauve + supplementary colors
  - `set_style()` function for global style application
  - `use_style()` context manager for temporary style (used internally by all plot functions)
  - Helper functions: `add_break_dates()`, `add_confidence_band()`, `shade_regimes()`, `label_line_end()`, `add_source()`
  - All 10 plotting functions updated to use the style system
  - Default figure size changed to (10, 5) for 2:1 aspect ratio
  - No top/right spines, horizontal gridlines only, frameless legends
  - 30 new tests in `test_style.py`

- **PcGive-Style Diagnostic Plots**: Misspecification diagnostics mimicking OxMetrics/PcGive
  - `plot_diagnostics()` function and `results.plot_diagnostics()` method for 2×2 diagnostic panel
  - `plot_actual_fitted()`: Time series of actual vs fitted values
  - `plot_scaled_residuals()`: Vertical index plot (resid/σ) for spotting autocorrelation
  - `plot_residual_distribution()`: Histogram with N(0,1) overlay for normality assessment
  - `plot_residual_acf()`: ACF and PACF bar plots with confidence bands
  - Works with OLS, AR, and ADL results
  - Example notebook: Part 5 sections 5.8-5.10

- **ADL (Autoregressive Distributed Lag) Model**: Full-featured ADL(p,q) estimation
  - `ADL` class with flexible lag specification: `lags` for AR, `exog_lags` for distributed lags
  - `exog_lags` accepts int (same for all variables) or dict (variable-specific)
  - Distributed lag properties: `cumulative_effect`, `long_run_multiplier`
  - Lag selection via `select_lags()` method with AIC/BIC grid search
  - Full break support (common and variable-specific breaks)
  - `RollingADL` / `RecursiveADL`: Rolling and expanding window ADL estimation
  - Bai-Perron integration via `.bai_perron()` method and `BaiPerronTest.from_model()`
  - Comprehensive test suite

- **Rolling and Recursive Estimation**: Track parameter evolution over time
  - `RollingOLS` / `RecursiveOLS`: Rolling and expanding window OLS estimation
  - `RollingAR` / `RecursiveAR`: Rolling and expanding window AR estimation
  - `RollingADL` / `RecursiveADL`: Rolling and expanding window ADL estimation
  - Model integration via `.rolling(window)` and `.recursive(min_nobs)` methods on OLS, AR, and ADL
  - `RollingOLSResults` / `RollingARResults` / `RollingADLResults` with `.summary()` and `.plot_coefficients()` methods
  - New `plot_rolling_coefficients()` visualization function
  - Example notebook Part 6: Rolling and Recursive Estimation

- **Model-to-BaiPerron workflow integration**: Seamless workflow from model definition to break detection to model with breaks
  - New `BaiPerronTest.from_model(model)` class method to create test directly from OLS, AR, or ADL models
  - New `BaiPerronResults.to_ols()` method to convert detected breaks back to a fitted OLS model
  - New `OLS.bai_perron()` convenience method for one-step break detection
  - New `AR.bai_perron()` convenience method for AR models
  - New `ADL.bai_perron()` convenience method for ADL models
  - Supports `break_vars` parameter: `"all"` (all regressors break) or `"const"` (mean-shift only)
  - Results from `to_ols()` work seamlessly with `plot_params_over_time()`

- **95% confidence intervals for break dates** in `BaiPerronTest`
  - New `break_ci` field in `BaiPerronResults` mapping break indices to (lower, upper) bounds
  - Based on Bai (1997) asymptotic distribution with critical value 7.78 for 95% CI
  - Displayed in `summary()` output after break dates

- **Variable-specific breaks** in OLS
  - `variable_breaks` parameter allows different coefficients to break at different times
  - Keys can be variable names or column indices
  - Summary output shows variable-specific regime information

- **Property-based testing** with Hypothesis
  - Hypothesis strategies for regression, AR, ADL, and rolling data
  - OLS invariants (residuals, R², TSS decomposition, CI, AIC/BIC)
  - AR invariants (roots, stationarity, effective sample)
  - ADL invariants (cumulative effect, long-run multiplier)
  - Rolling/recursive invariants (shape, NaN alignment, monotonicity)

### Changed

- Test coverage increased from 17.88% to 88% (540 tests total)

## [0.1.0] - 2025-02-03

### Added

- Initial release
- `OLS` model with HAC standard errors and structural break support
- `AR` autoregressive model with known breaks
- `BaiPerronTest` for detecting multiple structural breaks
  - Sup-F tests for m breaks vs 0 breaks
  - UDmax and WDmax statistics
  - Sequential testing procedures
  - BIC and LWZ model selection criteria
  - Dynamic programming for optimal break locations
- Visualization functions:
  - `plot_breaks`: Plot time series with break lines
  - `plot_regime_means`: Plot with regime-specific means
  - `plot_break_confidence`: Plot with confidence intervals
- Full type annotations (PEP 561 compliant)
- Comprehensive test suite
- CI/CD with GitHub Actions
