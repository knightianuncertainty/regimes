# regimes Development Roadmap

## Current Status

**Phase 2: v0.2.0 - Estimation Tools** Complete — [github.com/knightianuncertainty/regimes](https://github.com/knightianuncertainty/regimes)

## Completed Phases

### Phase 1: v0.1.0 - Foundation (Complete)

Core package structure, OLS/AR models, Bai-Perron test, basic visualization, CI/CD.

### Phase 2: v0.2.0 - Estimation Tools (Complete)

- [x] RollingOLS / RecursiveOLS classes
- [x] RollingAR / RecursiveAR classes
- [x] Model integration (`.rolling()`, `.recursive()` methods)
- [x] Rolling results visualization (`plot_coefficients()`, `plot_rolling_coefficients()`)
- [x] Example notebook: Part 6 (Rolling and Recursive Estimation)
- [x] Enhanced Bai-Perron features:
  - [x] Confidence intervals for break dates
  - [x] Sequential testing procedure
  - [x] Information criteria (BIC/LWZ)
  - [x] Model integration (`from_model()`, `to_ols()`, `model.bai_perron()`)
- [x] ADL (Autoregressive Distributed Lag) model
  - [x] ADL class with flexible lag specification (int or dict per variable)
  - [x] Distributed lag properties (cumulative effect, long-run multiplier)
  - [x] Lag selection via AIC/BIC grid search
  - [x] RollingADL / RecursiveADL classes
  - [x] Bai-Perron integration
  - [x] Example notebook: Part 7 (ADL Models)
- [x] Property-based tests with Hypothesis
  - [x] Hypothesis strategies for regression, AR, ADL, and rolling data
  - [x] OLS invariants (residuals, R², TSS decomposition, CI, AIC/BIC)
  - [x] AR invariants (roots, stationarity, effective sample)
  - [x] ADL invariants (cumulative effect, long-run multiplier)
  - [x] Rolling/recursive invariants (shape, NaN alignment, monotonicity)
- [x] PcGive-style diagnostic plots
  - [x] `plot_diagnostics()` function and `results.plot_diagnostics()` method
  - [x] Individual plots: actual/fitted, scaled residuals, distribution, ACF/PACF
  - [x] Example notebook: Part 5 sections 5.8-5.10
- [x] Plotting style system (PLOTTING_STYLE.md v1.0)
  - [x] `style.py` module with Economist-inspired color palette
  - [x] `set_style()` global setter and `use_style()` context manager
  - [x] Helper functions: `add_break_dates()`, `add_confidence_band()`, `shade_regimes()`, `label_line_end()`, `add_source()`
  - [x] All 10 plotting functions updated to use centralized style
  - [x] 30 tests for style module
- [x] Target: 80%+ test coverage (achieved: 88% with 540 tests)

## Planned Phases

### Phase 3: v0.3.0 - Advanced Models

- [ ] VAR (Vector Autoregression) model with breaks
- [ ] Cointegration model
- [ ] Comprehensive example notebooks
- [ ] Full Sphinx documentation on Read the Docs
- [ ] Target: 85%+ test coverage

### Phase 4: v0.4.0 - Additional Tests

- [ ] Chow test
- [ ] CUSUM/CUSUM-SQ tests
- [ ] Andrews-Ploberger test
- [ ] Real data examples (US interest rates, inflation)
- [ ] PyPI stable release
- [ ] Target: 90%+ test coverage

### Phase 5: v0.5.0+ - Future Extensions

- [ ] Time-varying parameter models
- [ ] Markov-switching models
- [ ] Panel data extensions
- [ ] Bootstrap inference methods
- [ ] Ox code translation utilities (as needed)

### Future Additions (Not Planned)

These features are deferred and may be added in future versions based on need:

- Full covariance support (HC1-HC3, HAC) for rolling estimation
- Rolling residual diagnostics
- Recursive CUSUM statistics
- Rolling forecasts
- Forecast-error regression class (moved to separate paper-specific package)

## Version History

| Version | Status | Description |
|---------|--------|-------------|
| 0.1.0 | Complete | Foundation: package structure, base classes, OLS/AR models, Bai-Perron test, basic visualization, CI/CD |
| 0.2.0 | Complete | Estimation tools: rolling/recursive estimation, ADL model, diagnostic plots, style system, 88% test coverage. GitHub repo live. |
| 0.3.0 | Planned | Advanced models: VAR, cointegration, full documentation |
| 0.4.0 | Planned | Additional tests: Chow, CUSUM, Andrews-Ploberger, PyPI release |
| 0.5.0+ | Future | Extensions: TVP, Markov-switching, panel data, bootstrap |
