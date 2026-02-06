# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **GitHub repository**: Published at [github.com/knightianuncertainty/regimes](https://github.com/knightianuncertainty/regimes)
  - CI pipeline: ruff lint/format, mypy (advisory), tests on Python 3.10/3.11/3.12, package build
  - Git tag `v0.2.0` for hatch-vcs versioning

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
