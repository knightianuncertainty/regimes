# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Summaries of AR and ADL models with breaks counted the initial observations, which are used for the lags, in the first regime ("Regime 1: 0-99 (n=100)" with 199 observations in the regression). The first regime now starts at the first observation of the estimation sample, and the counts add up to the number of observations.
- A regressor whose name is one the model gives to its own terms (a DataFrame column called `const`, `trend` or `y.L1` in an AR or ADL model) gave two parameters the same name, so that `variable_breaks` by name could not tell them apart. It raises `ValueError`.

## [0.5.0] - 2026-11-24

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
- **Andrews-Ploberger: the statistics and the critical values were on different scales.** The statistics were built from the classical F statistic (divided by the number of breaking coefficients `q`), while the limiting distributions, and every published table, are for the statistic that is not divided by `q`. Under a true null the SupF test rejected about 2 percent of the time at a nominal 5 for one coefficient and essentially never for two or more. `sup_f`, `exp_f`, `ave_f` and `f_sequence` are now on the chi-squared scale (`q` times the Chow F statistic), which is also what makes ExpF the statistic of Andrews and Ploberger (1994). `sup_f` equals the sup F statistic of `BaiPerronTest` for one break and the statistic of R's `strucchange`.
- **Andrews-Ploberger: the critical values matched no published table**, and p-values were one of four fixed numbers. P-values are now the approximate asymptotic p-values of Hansen (1997), coded from his own program and tables, for trimming of 1 percent or more and up to 40 breaking coefficients (a warning is issued below 1 percent, the limit of his tables); critical values (10, 5 and 1 percent, and the level asked for) are obtained by inverting them. Far in the tail, where Hansen's formula turns back and returns p-values that rise towards one (as R's `strucchange` then does), the p-value is held at its minimum. With a single candidate date the ExpF statistic is half the chi-squared statistic of that date and is treated as such (Hansen's program refers it to chi-squared without doubling it).
- **Andrews-Ploberger: a break was declared when any of the three tests rejected**, which is a test with a larger size than each. A break date is now reported when the SupF test rejects; the three tests are reported separately.
- **CUSUM-of-squares bounds were too narrow**: they were the Kolmogorov-Smirnov critical values for `T - k` observations, where the test of Brown, Durbin and Evans (1975) uses the table of Durbin (1969) for `(T - k) / 2 - 1`. The test rejected a true null 22 to 26 percent of the time at a nominal 5. The bounds now use the approximation of Edgerton and Wells (1994), as statsmodels does.
- **`break_vars="const"` in `ChowTest.from_model` and `AndrewsPlobergerTest.from_model` put the constant in the design twice**, which cost one denominator degree of freedom (F(1, 35) = 9.64 where F(1, 36) = 9.92 is right). The same happened when a constant was passed among the regressors that do not break. A fixed regressor that the breaking regressors span is now left out.
- **`ChowTest` returned F = 0 and p = 1 without warning** when the sample was too small to compute the test, or the model fitted exactly. It now raises `ValueError` (one such break point stops a call with several). It also raises when the model with a break is not identified: a regressor that breaks is zero or collinear in one of the sub-samples, the regressors that do not break already contain the break that is tested (a step dummy at the same date), or, in the predictive variant, a regressor fits the predicted observations (an impulse dummy among them). These cases returned F = 0, or an F statistic with more degrees of freedom than coefficients that can change. Break points must be integers (`TypeError` for 40.7). `AndrewsPlobergerTest` raises when the model fits exactly at a candidate date.
- **Recursive residuals** (CUSUM and CUSUM-of-squares) were computed by updating the inverse moment matrix of the first `k` observations: a bare `LinAlgError` when those were collinear (an AR model of a series that starts with equal values, a dummy variable that is zero at the start), and inaccurate numbers when they were nearly so. They are now computed on an orthonormal basis of the regressors by updating a QR decomposition. An observation at which the rank of the regressors rises is fitted exactly and has no residual; usually these are the first `k`, but with an impulse or step dummy later in the sample it is the date at which the dummy first varies, and every other observation keeps its residual. There are always `T - k` residuals and their sum of squares is the residual sum of squares of the regression. `observations` (new) gives the observation of each point of a CUSUM path. Regressors that are collinear in the whole sample raise `ValueError`.
- **`ADL.select_lags` compared information criteria across different samples.** Each candidate dropped its own number of initial observations, so the criteria were not comparable and the choice changed with the units of the data (for the same data multiplied by 1, 100 and 10000 the BIC choice was (0,0), (1,0) and (4,0)). Every candidate is now estimated on the sample that remains after dropping `max(max_ar_lags, max_exog_lags)` observations, and the candidates keep the trend of the model. The criteria equal those of statsmodels' `ARDL` on that sample; the choice equals that of `ardl_select_order` when the best model of its larger search (a lag length for each regressor, regressors left out) is one with the same lags for every regressor, as it always is with a single regressor that stays in the model. An unknown criterion, a negative maximum and a sample too short for the largest candidate raise `ValueError` (candidates that failed were skipped silently).
- **A break at `maxlag` created an empty regime.** `AR(y, lags=2, breaks=[2]).fit()` returned six coefficients, three of them 0, without warning; a break before `maxlag` was dropped silently; `breaks=[3]` fitted three coefficients on one observation. Break dates must now be distinct and lie in `1, ..., nobs - 1`; in AR and ADL models they must come after the initial observations used for the lags; and a model in which a regime has fewer observations than coefficients (or collinear regressors within the regime) raises `ValueError`. This includes a regressor that is zero throughout a regime, such as an impulse or step dummy in a model with common breaks: give it its own break dates, or none, with `variable_breaks`. Break dates are sorted on input, so summaries no longer print regimes such as "90-59 (n=-30)"; they must be integers (`TypeError` for any float, also 100.0, for `True` and for `"100"`) in a list (`breaks=50` says to write `[50]`), and the dates in `variable_breaks` are checked in the same way as common ones.
- **Rolling and recursive estimators returned HC0 standard errors for any `cov_type` other than `"nonrobust"`**, and recorded the name that was asked for (`fit(cov_type="HAC")` gave HC0 labeled "HAC"). Only `"nonrobust"` and `"HC0"` are implemented; other values raise `ValueError`.
- **`summary()` raised `ValueError` for a constant-only model** (`rg.OLS(y).fit().summary()`): White's test needs a regressor besides the constant. The test is reported as not available, and `DiagnosticsResults.all_pass` ignores tests that are not available. A constant column that is not a column of ones is recognized as the constant.
- **`fvalue` was wrong for a model without a constant** (it assumed one of the regressors was a constant) and NaN with a single regressor. Without a constant it now tests that all coefficients are zero; with regime-specific constants it uses centered sums of squares. Whether the regressors contain a constant is decided by statsmodels' rule. `fvalue` is the classical F statistic whatever the covariance type: it equals statsmodels' `fvalue` for `cov_type="nonrobust"`, and with robust standard errors it is not the robust Wald statistic statsmodels reports (the summary says so). `f_pvalue` is computed from the upper tail, so very small p-values are no longer rounded to zero.
- **Roots and stationarity with non-consecutive lags or breaks.** `roots` treated the coefficients as lags 1, 2, ... whatever the lags of the model (lags 1 and 3 gave two roots instead of three), and with breaks used the first regime only, so a second regime with a root inside the unit circle was reported as stationary. The polynomial now has each coefficient at its own lag, `is_stationary` checks every regime, and the new `roots_by_regime` gives the roots of each. `roots` remains the first regime, and is always a complex array.
- **The long-run multiplier of an ADL model with breaks was that of the first regime, unlabeled.** `ADLResults.effects_by_regime` gives the cumulative effect and the long-run multiplier of every regime, and the summary prints them by regime. `cumulative_effect` and `long_run_multiplier` remain the first regime and say so.
- `fit_by_regime()` with `variable_breaks` returned the pooled fit; it raises `ValueError` (there is no single set of regimes). It also raises when a regime does not have more observations than coefficients, or collinear regressors.
- **`ADL` lost the column names of a DataFrame** (the regressors were named `x0`, `x1`, ...), so `exog_lags={"income": 2}` and `variable_breaks` by name failed. Regressors are named after the columns, as in `OLS`, also in the rolling, recursive and Markov switching estimators built from the model and in `RollingADL` and `RecursiveADL` given a DataFrame; parameter names of models built from pandas objects change accordingly. numpy integers are accepted for `lags` and `exog_lags` (`exog_lags=np.int64(2)` silently fitted no lags of x).
- `adl_summary_by_regime` inferred the regime boundaries from the autoregressive lags only; it uses the longest lag of `y` or `x`.
- **Break dates meant different things in AR/ADL models and in the tests built from them.** `AR(..., breaks=[k])` reads `k` as an observation of the data, while `model.bai_perron()`, `chow_test`, `andrews_ploberger` and the CUSUM tests worked in the sample that remains after the initial observations for the lags: a break found by a test and passed back to the model landed `maxlag` observations late, and a date-aware Bai-Perron summary printed dates `maxlag` early. Tests built from an AR or ADL model now report every observation number (break dates, confidence intervals, candidate dates, crossings) in the observations of the model's data, and `chow_test(break_points=...)` takes them in the same numbers. Results change by `maxlag` for these models; tests built from arrays or from an OLS model are unchanged. The offset is available as `index_offset` on the test and on its results. `get_regime_estimates` takes dates in the same numbers, and error messages name them. `summary(index=...)` of such a test needs the index of the model's data and says so when given another length (an index of the estimation sample used to give the right dates). `to_ols()` is unchanged: it returns the regression on the estimation sample, with break dates as positions in that sample.
- **The bootstrap p-value of `NonRecurringRegimeTest` was always 0.** `method="bootstrap"` called a statsmodels method that does not exist; the error was swallowed and `0 / n_bootstrap` returned, so the test rejected every time. The bootstrap now draws samples from the estimated non-recurring model (new internal module `markov/_simulate.py`), estimates both models on each sample by the procedure used on the data, and reports `(1 + #{simulated >= observed}) / (B + 1)`. The test rejects when this p-value is at most the level (Dufour, 2006), which makes the level exact when `significance * (B + 1)` is an integer: 19 replications suffice for a test at 5 percent, and the default 199 is exact at 5 and at 1 percent. Replications in which a model cannot be estimated are counted (`n_bootstrap_failed`) and a warning is issued above 10 percent; if none can be computed, or a model cannot be estimated on the data, the test raises instead of reporting a decision. A statistic of zero has p-value one without simulation; a difference in log-likelihood within the optimizer's rounding (1e-4) counts as zero, in the data and in every bootstrap sample.
- **The models of the restriction tests were not nested in practice.** A model that contains another must reach at least its likelihood, but the estimations were independent searches that often stopped at different local optima: the statistic was then negative and silently set to zero, or large for a restriction that costs nothing. Every model is now also estimated from the estimates of the model it is compared with, and a larger model that a nested one beats is estimated again from the nested estimates. A shortfall that remains is reported with a warning.
- **Tests of several restrictions used a p-value and a critical value from different distributions** (a Bonferroni multiple of a one-restriction tail against a chi-squared critical value with `n` degrees of freedom). Both now come from the mixture `0.5 chi2(n-1) + 0.5 chi2(n)`, whose tail probability is an upper bound for every chi-bar-squared distribution with `n` inequality restrictions (Kodde and Palm, 1986). A statistic of zero has p-value one (it was 0.5).
- **`SequentialRestrictionTest`:** the "individual" test was the joint test under another name; it is now the test of the one new restriction against the model with the restrictions imposed so far. Candidates are ranked by their LR statistic, least significant first (they were ranked by the size of the estimated probability). `critical_value_method="bootstrap"` runs the bootstrap (it silently returned the chi-bar-squared p-value). All models of the search have the same distribution of the regime before the sample, so that each is nested in the previous one (the unrestricted model used the stationary distribution, the restricted ones did not): the smoothed distribution of the first observation's regime in a preliminary unrestricted fit. A candidate that cannot be estimated no longer aborts the search. `is_non_recurring` no longer depends on how the regimes are labeled (a single break from the high-mean to the low-mean regime was reported as recurring).
- **`MarkovAR` with `order >= 2` and `switching_variance=True` used the variance of the wrong regime.** This is a defect in statsmodels (0.14 and 0.15): the variance of period `t` followed the regime of period `t - order + 1`. The package now uses its own subclass, in which the variance follows the regime of period `t`. Estimates of such models change. Models of order 1, and models without switching variance, are not affected.
- **`RegimeNumberSelection`: the model with one regime ignored the regressors and the trend** of the switching models, so the comparison of one regime with two was a comparison of different models (for `y = 1 + 2x + e` it selected two regimes, with a likelihood ratio of 345 where the value is now below 10). The one-regime model now has the same trend and regressors and, for `model_type="ar"`, the same mean-adjusted form as `MarkovAR`; `results_by_k[1]` is a `OneRegimeResults`.
- **`RegimeNumberSelection`: parameter counts were not comparable across K** (the error variance was counted for two or more regimes only), which understated the criteria of the one-regime model by one parameter. The counts include the variance(s) for every K.
- **`RegimeNumberSelection(method="sequential")` rejected a true null about 20 percent of the time at a nominal 5** (in the simulation of the audit). The likelihood ratio statistic was referred to a Bonferroni multiple of a chi-bar-squared tail, which no result supports for this null hypothesis: the statistic is not chi-squared (unidentified parameters, a zero score, parameters on the boundary), and its limiting distribution depends on the model. The test is now the parametric bootstrap of Kasahara and Shimotsu (2018): samples are drawn from the estimated model with K regimes, the models are estimated on each sample by the procedure used on the data, and the test is the Monte Carlo test of Dufour (2006), with p-value `(1 + #{simulated >= observed}) / (B + 1)` and rejection when it is at most the level. New arguments `n_bootstrap` (default 199), `random_state` and `search_reps`. The module docstring states what supports the test and where the support ends: the consistency proof is for one regime against two and for models that depend on the current regime only, which excludes the mean-adjusted autoregression of `model_type="ar"` (simulation evidence only); regressors must be strictly exogenous; transition probabilities are not bounded away from 0 and 1.
- **The model with K + 1 regimes could have a lower likelihood than the model with K**, which it contains, when the search stopped at a local maximum. The larger model is now also estimated from the estimates of the smaller one with each regime split in two. With three or more regimes the result of the search can still depend on the seed (on the reference series of the tests the statistic for two against three regimes is 2.5 or 2.9); `search_reps` sets the number of random starting values.
- **Models with regime-specific variances had an unbounded likelihood**, and the search ended at a variance near zero on a handful of observations, a different one for every seed (statistics from 3.9 to 9.7 on one series). Inside `RegimeNumberSelection` such models are estimated with the variance penalty of Kasahara and Shimotsu (2018, Section 10.1), `a_n = 20 / sqrt(n)`, and the statistic and the criteria use the log-likelihood without the penalty, as in the paper. `MarkovRegression(switching_variance=True)` itself is unchanged.
- **A model that could not be estimated was dropped without a word**, and the answer was a smaller number of regimes; a misspelled model argument therefore selected one regime. Model arguments are now checked when the object is created, a model that cannot be estimated raises a warning, and the results carry `notes` (also printed by `summary()`), which include a model whose optimizer did not converge and a selected K equal to `k_max`. The `converged` flag of the fitted models is that of the optimizer; an optimization that stops at its iteration limit is continued once.

### Changed

- `MarkovADL` names its coefficients `y.L1`, `x0`, `x0.L1`, ... (and the column names of a DataFrame) in `param_names` and `regime_params`, instead of statsmodels' `x1`, `x2`, ....
- Markov results report the transition probabilities `p[j->i]` for the destinations `i = 0, ..., k-2` in the relabeled regimes. Where relabeling changes which destination statsmodels left out, the reported probability is one minus the estimated ones, with the standard error from their covariance matrix.
- Rolling and recursive estimators built from a model with breaks (`model.rolling(...)`, `from_model`) warn that the breaks are ignored. This warning and the one about `cov_kwds` point at the line of the caller's code.
- `cov_kwds` is passed to statsmodels' HAC covariance (`maxlags`, `use_correction`, ...); with another `cov_type` it has no effect and a warning says so. An unknown `cov_type` raises in every model.
- Documentation of conventions: the default HAC lag `floor(4 (T/100)^(2/9))` is a fixed rule, not "Newey-West automatic bandwidth selection"; the misspecification tests are standard LM tests and not the statistics PcGive prints; with breaks the error variance is pooled across regimes; inference under robust covariances uses t(n - k); R-squared is centered also without a constant.
- **CUSUM and CUSUM-of-squares no longer report a break date.** The first crossing of a boundary shows when the evidence against stability became significant, not when the parameters changed (for a variance change the CUSUM-of-squares path can cross about a hundred observations before the change). `n_breaks` is 0 and `break_indices` is empty; the new attribute `first_crossing` holds the crossing as an observation of the sample (it was a position among the recursive residuals: sample index minus `k`). `crossing_indices` and the x-axis of the plots are in sample observations too, and the plots mark the first crossing only.
- `CUSUMSQTest.fit(significance=...)` accepts 0.20, 0.10, 0.05, 0.02 and 0.01, the levels for which the bounds are tabulated (it accepted any level, with the wrong bounds), and needs at least 20 recursive residuals, below which the approximation is poor. `CUSUMSQResults.critical_value` holds the half-width of the bounds. New properties `observations` (the observation of each point of a CUSUM path) and `CUSUMResults.max_ratio`.
- `AndrewsPlobergerTest`: trimming is `floor(trimming * T)` observations at each end, as in `BaiPerronTest` (it was `ceil`); critical values are no longer silently taken from the nearest tabulated trimming or from `q = 10`; more than 40 breaking coefficients raise `ValueError`; `break_vars="const"` on a model without a constant raises (it used to add an intercept and test it); a single candidate date is accepted. New results attribute `pi0`; the summary line "Estimated break date" is now "Date of the largest F". `ChowTest` and `AndrewsPlobergerTest` leave out regressors among those that do not break that are zero or collinear with the others.
- Docstrings: the CUSUM bounds are a pair of lines for a Brownian motion, not from "the Brownian bridge distribution"; the CUSUM-of-squares bounds are Durbin's, not "Kolmogorov-Smirnov". The CUSUM summary prints the ratio of the path to its bound instead of the maximum of the path beside the constant `a`, which are not comparable.
- The Bai-Perron summary of a test built from an AR or ADL model prints the sample the test was computed on.
- `NonRecurringRegimeTest` and `SequentialRestrictionTest`: a new `random_state` argument seeds the bootstrap samples and the random starting values of every estimation, so that a given seed reproduces the result (statsmodels draws its random starting values from numpy's global generator in 0.14 and from fresh entropy in 0.15). `n_bootstrap` defaults to 199 (was 999; each replication estimates two models); `NonRecurringRegimeTest` raises if `n_bootstrap` is too small for the test to reject at the chosen level, and the sequential search warns in that case. Unknown values of `method`, `critical_value_method`, `multiple_testing` and `model_type` raise `ValueError`, as do `initial_probabilities`, `restrictions` and `ordering` passed through to the model. The bootstrap critical value is the value the statistic must exceed for the test to reject.
- `SequentialRestrictionTest(multiple_testing="holm")`: the individual test at step `s = 1, 2, ...` uses the level `significance / s`. The search runs from the least significant restriction upwards, and in that direction these are the constants of Hochberg's (1988) step-up procedure, which are Holm's. The levels were applied in the reverse order before (`significance / m` first). New results attribute `initial_probabilities`.
- The docstrings of both tests state what is and is not known about them: no published result gives the distribution of the statistics (in a non-recurring model each regime is left once, so the usual boundary theory is not obviously applicable), the statistic tested in the search is the smallest among the candidates, which the reference distributions do not account for, and the multiple-testing levels have no proven error-rate property here.
- `RegimeNumberSelection`: an unknown `method` or `model_type` raises `ValueError` (an unknown method silently meant BIC); `method="sequential"` raises if `n_bootstrap` is too small for a test to reject at the chosen level, and estimates a model only when a test needs it, so `results_by_k` and `ic_table` stop one regime above the selected number. The sequential test is slow: each replication estimates every model up to K + 1 regimes (on the order of 15 minutes for one regime against two with 200 observations and the defaults, hours for two against three). The docstring notes that the consistency of AIC and BIC for the number of regimes is not established.

### Documentation

- API reference for every public class and function (it covered `OLS`, `AR` and `BaiPerronTest` only).
- New page **Conventions**: what a break date is in models and in tests, how the statistics are scaled, which covariance estimators and misspecification tests are used, the timing and labeling conventions of the Markov switching models.
- New page **Validation**: for every method, what it was checked against (statsmodels, R `strucchange`, R `MSTest`, Perron's program, OxRegimes, published tables), with the numbers, the simulated rejection rates, and what is not validated.
- The installation instructions said `pip install regimes`; the package is not on PyPI. They now give the installation from GitHub.
- Three code examples of the README did not run (undefined variables, a regressor name that does not exist); all 26 run now.
- Bullet lists in 36 docstrings rendered as broken block quotes in the reference; they are lists now.


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
