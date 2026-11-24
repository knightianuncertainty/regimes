# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Markov switching: the parameter table used different regime labels from every other result.** After the regimes were relabeled (`ordering="first_appearance"`, the default, or `"intercept"`), `regime_params`, the probabilities and the transition matrix followed the new labels, while `params`, `bse` and `param_names` (and so `summary()`, `to_dataframe()`, `tvalues`, `pvalues`, `conf_int()`) kept the order of the estimation: `const[0]` could be the constant of the regime reported everywhere else as regime 1. The table now follows the same labels in `MarkovRegression`, `MarkovAR` and `MarkovADL`. Code that paired `regime_params[j]` with `bse` through names such as `const[j]` gave the wrong standard errors whenever the regimes had been relabeled.
- **`ordering="intercept"`** sorted the regimes on the first transition probabilities, not on the constants. It now sorts on the regime-specific constant, and warns and leaves the order unchanged when the model has none.
- **Coefficients that do not switch** (`switching_exog=False`, `switching_trend=False`) were missing from `regime_params` for all regimes but the last, because statsmodels gives them the last regime's suffix (`x1[1]`). They now appear in every regime and are named without a suffix in the table.
- `MarkovARResults.ar_params` follows the relabeled regimes.
- **Models with restricted transition probabilities** (`RestrictedMarkovRegression`, `RestrictedMarkovAR`, including `non_recurring()`) were relabeled by first appearance like unrestricted ones, although their restrictions refer to the regime labels; the probabilities and `regime_params` could then describe other regimes than the restrictions. They now keep the labels of the estimation, and an `ordering` argument is ignored with a warning.
- **Restricted transition matrices were mis-indexed for three or more regimes.** The restrictions were applied as if statsmodels stored the transition probabilities origin by origin; it stores them destination by destination. The two layouts coincide only for two regimes. A three-regime `non_recurring()` model freed forbidden transitions, forced an allowed one to zero and could produce a negative "probability". The reported matrix was then overwritten with the restriction values, which hid this; it is now the matrix the filter used.
- **Non-recurring models started in the wrong regime.** The chain started from the stationary distribution, which for a chain with an absorbing regime puts all mass on the last regime. `non_recurring()` now starts in regime 0 (Chib 1998); other restricted models take an optional `initial_probabilities`.
- **`NonRecurringRegimeTest` rejected a true null.** With the two defects above the statistic was about 60 to 100 for a series with a single break. The unrestricted model now starts in regime 0 as well, so that the two models are nested.
- **Starting values of non-recurring models** follow the order of the regimes in time (regressions on consecutive stretches of the sample). With statsmodels' defaults, which order the regimes by the size of their coefficients, a series that starts in its high regime gave a degenerate fit.
- **EM steps of restricted models.** statsmodels' EM update divides the expected number of moves out of a regime by the expected time spent in it over the sample, which leaves out the starting period. When a regime is occupied only briefly at the start, the probability of staying comes out as one and the model can never leave its first regime. Restricted models use the expected number of moves divided by the expected number of departures, with the restrictions imposed.
- **Standard errors and parameter counts of restricted models.** Standard errors are computed in the parameters that are free under the restrictions; a transition probability that the restrictions determine has a NaN standard error. `df_model`, and with it AIC, BIC and HQIC, counts only the free parameters (6, not 10, for a three-regime non-recurring model with a switching constant).
- Restricted entries of the transition matrix are exactly their fixed values (a rounding error of 1e-16 in the implied last row was enough to let probability leak through a forbidden transition).
- Restrictions and `initial_probabilities` that do not describe a probability distribution raise `ValueError`.
- **`ADL.select_lags` compared information criteria across different samples.** Each candidate dropped its own number of initial observations, so the criteria were not comparable and the choice changed with the units of the data (for the same data multiplied by 1, 100 and 10000 the BIC choice was (0,0), (1,0) and (4,0)). Every candidate is now estimated on the sample that remains after dropping `max(max_ar_lags, max_exog_lags)` observations; the choice equals that of statsmodels' `ardl_select_order`.
- **A break at `maxlag` created an empty regime.** `AR(y, lags=2, breaks=[2]).fit()` returned six coefficients, three of them 0, without warning; a break before `maxlag` was dropped silently; `breaks=[3]` fitted three coefficients on one observation. Break dates must now be distinct and lie in `1, ..., nobs - 1`; in AR and ADL models they must come after the initial observations used for the lags; and a model in which a regime has fewer observations than coefficients (or collinear regressors within the regime) raises `ValueError`. Break dates are sorted on input, so summaries no longer print regimes such as "90-59 (n=-30)".
- **Rolling and recursive estimators returned HC0 standard errors for any `cov_type` other than `"nonrobust"`**, and recorded the name that was asked for (`fit(cov_type="HAC")` gave HC0 labeled "HAC"). Only `"nonrobust"` and `"HC0"` are implemented; other values raise `ValueError`.
- **`summary()` raised `ValueError` for a constant-only model** (`rg.OLS(y).fit().summary()`): White's test needs a regressor besides the constant. The test is reported as not available.
- **`fvalue` was wrong for a model without a constant** (it assumed one of the regressors was a constant) and NaN with a single regressor. Without a constant it now tests that all coefficients are zero, as statsmodels does; with regime-specific constants it uses centered sums of squares.
- **Roots and stationarity with non-consecutive lags or breaks.** `roots` treated the coefficients as lags 1, 2, ... whatever the lags of the model (lags 1 and 3 gave two roots instead of three), and with breaks used the first regime only, so a second regime with a root inside the unit circle was reported as stationary. The polynomial now has each coefficient at its own lag, `is_stationary` checks every regime, and the new `roots_by_regime` gives the roots of each. `roots` remains the first regime.
- **The long-run multiplier of an ADL model with breaks was that of the first regime, unlabeled.** `ADLResults.effects_by_regime` gives the cumulative effect and the long-run multiplier of every regime, and the summary prints them by regime. `cumulative_effect` and `long_run_multiplier` remain the first regime and say so.
- `fit_by_regime()` with `variable_breaks` returned the pooled fit; it raises `ValueError` (there is no single set of regimes).
- `adl_summary_by_regime` inferred the regime boundaries from the autoregressive lags only; it uses the longest lag of `y` or `x`.

### Changed

- `MarkovADL` names its coefficients `y.L1`, `x0`, `x0.L1`, ... (and the column names of a DataFrame) in `param_names` and `regime_params`, instead of statsmodels' `x1`, `x2`, ....
- Markov results report the transition probabilities `p[j->i]` for the destinations `i = 0, ..., k-2` in the relabeled regimes. Where relabeling changes which destination statsmodels left out, the reported probability is one minus the estimated ones, with the standard error from their covariance matrix.
- Rolling and recursive estimators built from a model with breaks (`model.rolling(...)`, `from_model`) warn that the breaks are ignored.
- `cov_kwds` is passed to statsmodels' HAC covariance (`maxlags`, `use_correction`, ...); with another `cov_type` it has no effect and a warning says so. An unknown `cov_type` raises in every model.
- Documentation of conventions: the default HAC lag `floor(4 (T/100)^(2/9))` is a fixed rule, not "Newey-West automatic bandwidth selection"; the misspecification tests are standard LM tests and not the statistics PcGive prints; with breaks the error variance is pooled across regimes; inference under robust covariances uses t(n - k); R-squared is centered also without a constant.

## [0.4.0] - 2026-11-20

### Fixed

- **Bai-Perron procedure rewritten** as a port of the OxRegimes core (`regimes.tests._bp_core`), which is cross-validated against Pierre Perron's program. The previous implementation differed from Bai and Perron (1998, 2003) in the following ways, all corrected:
  - **BIC** did not count the `m` break dates as estimated parameters and so selected too many breaks. The penalty is now `p* ln(T)/T` with `p* = (m+1)q + m + p`.
  - **LWZ** was not the Liu-Wu-Zidek criterion. It is now `ln(SSR/(T - p*)) + p* 0.299 ln(T)^2.1 / T`.
  - **The sequential test** compared the global `m`- and `(m-1)`-break sums of squared residuals. It is now sup F(l+1|l): the largest one-break test within the `l + 1` regimes. `selection="sequential"` applies it to break dates estimated one at a time and stops at the first non-rejection.
  - **Sequential critical values** were taken from the sup F(k|0) table. They now come from the sup F(l+1|l) table.
  - **sup F(k|0)** was divided by the number of breaking regressors `q`, while the published critical values are for the statistic not divided by `q`; with `q >= 2` the test under-rejected. Statistics are now reported "scaled by q", as in Perron's program.
  - **Critical values** for `q >= 2` did not match the published tables, and a single table (5%, trimming 0.15) was used for every trimming. The full tables of Bai and Perron (1998, 2003b) are now included: trimming 0.05, 0.10, 0.15, 0.20, 0.25; levels 10%, 5%, 2.5%, 1%; `q = 1..10`.
  - **WDmax** used inverted weights and had no critical value.
  - **sup F p-values** had no basis and are removed (`supf_pvalues`).
  - **Break-date confidence intervals** did not follow Bai (1997). They now invert the asymmetric limit distribution, with regime-specific moment matrices and variances, and treat fixed regressors correctly.
- `get_regime_estimates()` keeps the coefficients on fixed regressors (`exog`) common to all regimes, and returns the `q` coefficients on the breaking regressors for each regime (it returned `q + p` values when `exog` was given).
- `BaiPerronResults.to_ols()` uses the same fixed regressors as `fit()`; with a collinear column in `exog` its design was rank deficient.

### Changed

- **Minimum regime length** is `h = floor(trimming * T)` (was `ceil`), as in Bai and Perron (2003, Table I) and Perron's program. Break dates can change when `trimming * T` is not an integer.
- `BaiPerronTest.fit()` has new keyword options: `level`, `robust`, `prewhiten`, `hetdat`, `hetvar`, `hetomega`, `hetq`, `repartition`, `h`. By default the tests allow the error variance to differ across regimes (`hetvar=True`), as in OxRegimes; `hetvar=False` gives the common-variance statistic (`q` times the classical F).
- `max_breaks` is reduced, with a warning, to the largest number of breaks tabulated for the trimming (9, 8, 5, 3, 2). A trimming outside the tabulated values gives a warning and NaN critical values.
- `BaiPerronResults`: `seqf_stats[l]` is sup F(l+1|l) on the global `l`-break partition (`seqf_stats[0]` is sup F(1|0)); `bic` and `lwz` are per-observation values; new fields `level`, `h`, `p`, `wdmax_critical`, `wdmax_stats`, `seqf_break`, `sequential_n_breaks`, `sequential_breaks`, `break_ci_90`, `critical_values`, `options`.
- `OLS.bai_perron()`, `AR.bai_perron()` and `ADL.bai_perron()` pass further keyword options to `BaiPerronTest.fit()`.
- A column of `exog` that is collinear with the breaking regressors (for example a constant, when the constant breaks) is dropped from the fixed regressors, so the degrees of freedom and the information criteria count it once; `BaiPerronResults.p` reports the number of fixed regressors used.
- The segment sums of squared residuals are computed with recursive residuals, which is faster, from an orthonormal basis of the regressors, so the break dates do not depend on the units or the level of the regressors. `fit()` warns when the regressors are nearly collinear.
- **Partial structural change** (`exog` given): one break is found by direct search over the dates; for two or more breaks the iteration of Bai and Perron (2003, Section 3.4) is run from two starting values (the no-break regression and the pure structural change model) and the lower sum of squared residuals is kept.
- A fixed regressor without variation inside a regime (an impulse or step dummy in `exog`) is left out of the sup F(l+1|l) test within that regime. A model whose break dates make a fixed regressor a linear combination of the regime-specific ones is not identified; `max_breaks` is then reduced with a warning.
- `fit()` raises a `ValueError` for missing or infinite values, for `max_breaks < 1`, for an untabulated `level`, for `selection="sequential"` with an untabulated trimming, and when a model fits the data exactly. It no longer accepts arbitrary keyword arguments.
- `summary()` has a new layout: critical values at the four levels instead of reject and p-value columns, both 90% and 95% intervals.

### Added

- Tests against Perron's program on its example data, against OxRegimes on 21 option sets (`tests/reference_data/bp_oxregimes_reference.json`), and a replication of Bai and Perron (2003), Table I.
- **Date-aware summaries**: Optional `index` parameter on all summary methods
  - `OLSResults.summary(index=)`, `ARResults.summary(index=)`, `ADLResults.summary(index=)`
  - `BaiPerronResults.summary(index=)` with date-labeled break points and confidence intervals
  - `summary_by_regime(index=)`, `ar_summary_by_regime(index=)`, `adl_summary_by_regime(index=)`
  - Accepts `pd.PeriodIndex`, `pd.DatetimeIndex`, or `Sequence[str]`
  - Shared helpers `_obs_label()` and `_obs_range_label()` in `results/base.py`
  - 21 new tests in `test_date_aware_summary.py` (924 total)

## [0.3.2] - 2026-02-20

### Fixed

- **Bai-Perron partial structural change model failing when `exog_break` is a subset of `exog`**: Duplicate columns caused multicollinearity in the regime design matrix. PR #4.
- **CI lint errors** from PR #4 code (ruff format/check, issue #7)

### Added

- **R cross-validation reference tests** for Bai-Perron (903 tests total, 86% coverage)

### Changed

- Test PyPI publish step made non-blocking (`continue-on-error`) in release workflow

## [0.3.1] - 2026-02-19

### Fixed

- **Smoothed probability leakage in restricted Markov models**: statsmodels' Hamilton filter and Kim smoother convert zero transition probabilities to `log(max(0, 1e-20)) ≈ -46`, which leaks ~1e-20 probability per step. Over many time steps with strong data, this accumulates and overwhelms transition restrictions, causing non-recurring models to show impossible regime reversals. Added `_RestrictedFilterMixin` that overrides `_filter`/`_smooth` in both restricted statsmodels subclasses with `_LOG_ZERO = -1000` to eliminate leakage. 3 new tests verify smoothed probability monotonicity (880 tests total).

## [0.3.0] - 2026-02-19

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
- **Example notebooks**: Replaced monolithic `exploration.ipynb` with 4 focused notebooks
  - `01_models_and_visualization.ipynb` — OLS, AR, visualization, diagnostics
  - `02_structural_break_tests.ipynb` — Bai-Perron, Chow, CUSUM, Andrews-Ploberger
  - `03_dynamic_estimation.ipynb` — Rolling/recursive estimation, ADL models
  - `04_markov_switching.ipynb` — Full Markov switching API demonstration
- **Expanded test coverage**: Additional edge case and integration tests (877 tests total, 86% coverage)
  - Markov regression, AR, utils, restricted, selection, sequential restriction tests
  - Results base class tests
  - Markov visualization tests

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
