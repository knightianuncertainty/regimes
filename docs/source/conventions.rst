Conventions
===========

The definitions that the package uses throughout. Where a convention differs
from statsmodels, PcGive/OxMetrics or the R packages, the difference is
stated.

Break dates
-----------

A break date is the index of the **first observation of the new regime**
(0-based). A break at ``k`` splits the sample into observations
``0, ..., k - 1`` and ``k, ..., T - 1``.

* Break dates are given as a list of distinct integers in
  ``1, ..., T - 1``; they are sorted on input. A date that is not an
  integer (any float, ``True``) is an error and is not rounded.
* **AR and ADL models.** Break dates are observation numbers of the data
  passed to the model, not of the sample that remains after the initial
  observations for the lags. A break must come after those initial
  observations.
* **Tests built from a model.** A test built from an AR or ADL model
  (``model.bai_perron()``, ``chow_test``, ``andrews_ploberger``,
  ``cusum_test``, ``cusum_sq_test``, or ``from_model``) is computed on the
  sample that remains after the initial observations. Everything it reports
  (break dates, confidence intervals, candidate dates, boundary crossings)
  and the ``break_points`` it accepts are observation numbers of the model's
  data, so a date found by a test can be passed back to the model as
  ``breaks``. The number of initial observations is available as
  ``index_offset``. A test built directly from arrays reports positions in
  those arrays. ``summary(index=...)`` of a test built from a model takes
  the index of the model's data. One exception: ``to_ols()`` of a
  Bai-Perron test returns the regression on the estimation sample, with
  break dates as positions in that sample.
* Each regime needs at least as many observations as it has coefficients,
  and its regressors must not be collinear within the regime; otherwise the
  model raises. A regressor that is zero throughout a regime (an impulse or
  step dummy) cannot have a coefficient there: give it its own break dates,
  or none, with ``variable_breaks``.
* Regimes are numbered from 1 in parameter names (``const_regime1``) and in
  summaries.

Rolling and recursive estimates of AR and ADL models are indexed on the
sample after the initial observations: row ``t`` of ``params`` belongs to
observation ``t + maxlag`` of the data.

Regression models
-----------------

* **Breaks and the error variance.** A model with breaks is one regression
  with regime-specific columns. The error variance is pooled across regimes,
  so the standard errors differ from those of separate regressions on each
  regime; ``fit_by_regime()`` gives the latter. OxRegimes pools in the same
  way.
* **Inference.** p-values and confidence intervals use the t distribution
  with ``n - k`` degrees of freedom for every covariance type. statsmodels
  uses the normal distribution when the covariance is robust.
* **R-squared** is centered, also for a model without a constant, where
  statsmodels reports the uncentered version; it can then be negative.
* **F statistic.** ``fvalue`` is the classical F statistic, computed from
  sums of squares. With a constant in the column space of the regressors
  (a constant, or regime-specific constants) the null is that every
  coefficient except a common constant is zero; without one, that every
  coefficient is zero. With breaks the statistic therefore also tests
  equality of the regime constants. It does not use the covariance
  estimator: it equals statsmodels' ``fvalue`` for ``cov_type="nonrobust"``,
  while with a robust covariance statsmodels reports a robust Wald
  statistic and this package still reports the classical one.
* **Names.** Regressors are named after the columns of a DataFrame (or
  ``x0``, ``x1``, ... for arrays) in ``OLS``, ``ADL`` and the estimators
  built from them. In ``exog_lags`` an integer key is a column of the
  regressors; in ``variable_breaks`` it is a column of the design matrix
  (constant and lags included), so names are the safer choice there.
* **Roots.** The autoregressive polynomial is ``1 - sum_L phi_L z^L`` over
  the lags in the model. With breaks there is one polynomial per regime
  (``roots_by_regime``); ``roots`` is the first regime and
  ``is_stationary`` requires every regime to be stationary.
* **Long-run multiplier.** ``sum(beta) / (1 - sum(phi))``, per regime
  (``effects_by_regime``); no standard error is computed.

Covariance estimators
---------------------

``cov_type`` is one of ``"nonrobust"``, ``"HC0"`` to ``"HC3"`` and
``"HAC"``.

* **HAC** is the Newey-West estimator with the Bartlett kernel and the
  small-sample correction ``n / (n - k)`` (statsmodels'
  ``use_correction=True``). The default number of lags is the fixed rule
  ``floor(4 (T / 100)^(2/9))``. It is not a data-dependent bandwidth: in
  Newey and West (1994) the same expression is the first-step lag from
  which the bandwidth is estimated. Set the lags with
  ``cov_kwds={"maxlags": ...}``.
* OxRegimes computes HAC standard errors without the small-sample
  correction; the two packages differ by the factor ``sqrt(n / (n - k))``.
* With breaks, HAC covariances between coefficients of different regimes
  use autocovariances across the break.
* Rolling and recursive estimation supports ``"nonrobust"`` and ``"HC0"``.

Misspecification tests
----------------------

The four tests in ``summary()`` are standard tests in their LM
(chi-squared) forms: Breusch-Godfrey for autocorrelation (the first
``nlags`` observations of the auxiliary regression are dropped), Engle's
ARCH test, Jarque-Bera, and White's test with cross products. They address
the same questions as PcGive's diagnostic tests but are not the statistics
PcGive prints (F forms, and the Doornik-Hansen normality test); the numbers
can differ noticeably in a given sample.

Bai-Perron
----------

* F statistics are **scaled by q**: they are not divided by the number of
  breaking coefficients, which is the scaling of the published critical
  values.
* The minimum regime length is ``h = floor(trimming * T)``, at least
  ``q + 1``.
* BIC and LWZ count the break dates as estimated parameters and are
  per-observation values.
* Critical values exist for trimming 0.05, 0.10, 0.15, 0.20 and 0.25 and
  for the levels 10, 5, 2.5 and 1 percent.
* The numbers equal those of OxRegimes and of Perron's GAUSS program on its
  example data.

Andrews-Ploberger
-----------------

* The statistics are built from ``F(tau)``, ``q`` times the Chow F statistic
  at each candidate date: the chi-squared scale on which SupF, ExpF and AveF
  are defined in the literature. ``SupF`` equals the Bai-Perron sup F
  statistic for one break (with a common error variance).
* Trimming is ``floor(trimming * T)`` observations at each end.
* p-values are the approximate asymptotic p-values of Hansen (1997), for
  up to 40 breaking coefficients and trimming of 1 percent or more;
  critical values are obtained by inverting them. Far in the tail, where
  Hansen's polynomial turns back, the p-value is held at its minimum. With
  a single candidate date the tests are the chi-squared test of that date.
* The three tests are separate tests. A break date is reported when SupF
  rejects.

Chow test
---------

* With enough observations on both sides of the break point the test is
  the standard F test of equal coefficients; otherwise the predictive
  variant is used (``is_predictive``).
* With ``exog_break`` only those regressors change at the break point; the
  regressors in ``exog`` have common coefficients.
* A design in which the break is not identified raises: a breaking
  regressor that is zero or collinear in a sub-sample (an impulse or step
  dummy), or a step dummy at the tested date among the regressors that do
  not break. A dummy variable belongs in ``exog``, with a coefficient that
  does not break.

CUSUM tests
-----------

* Recursive residuals: an observation at which the rank of the regressors
  rises is fitted exactly and has no residual. Usually these are the first
  ``k`` observations; with regressors that are collinear at the start (a
  series that starts with equal values, a dummy variable that first varies
  later) they come later, and every other observation keeps its residual.
  There are always ``T - k`` residuals, and their sum of squares is the
  residual sum of squares of the regression. ``observations`` gives the
  observation of each point of a path. Whether an observation adds to the
  rank is decided from the linear dependencies that hold among the columns
  in all earlier observations (a column that is zero so far, two columns
  that are equal so far), so that it does not depend on the units of the
  regressors or on how much a regressor grows over the sample. Differences
  between observations of a relative 1e-9 or smaller are at the limit of
  what can be told from rounding error: they may count as no difference,
  or the residuals may not be computable (an error).
* The recursive residuals are checked: their sum of squares must equal the
  residual sum of squares of the regression. When it does not to a relative
  1e-9, they are computed again by two other methods (the last is that of
  version 0.5.0), and the computation that comes closest is used. If none
  comes within a relative 1e-6, an error is raised; this happens for a
  cubic trend in monthly calendar years and a quartic trend in quarterly
  calendar years. A regression that fits the data exactly, and data with
  NaN or infinite values, also raise an error.
* The check does not show that each residual is accurate to many digits.
  A polynomial trend in calendar time gives about eight digits (quadratic,
  quarterly calendar years), four (cubic) or three (quadratic in daily date
  ordinals near 730000); the same trend measured from the start of the sample gives
  twelve or more. Measure trends from the start of the sample.
* The Chow and Andrews-Ploberger tests compute their sums of squared
  residuals from regressors scaled to unit length, so the statistics do not
  depend on the units of the regressors.
* The CUSUM scale estimate is ``sqrt(sum w^2 / (T - k))`` (uncentered).
* CUSUM bounds: the lines of Brown, Durbin and Evans (1975) with
  ``a = 1.143, 0.948, 0.850`` for 1, 5 and 10 percent.
* CUSUM-of-squares bounds: Durbin's (1969) critical values for
  ``n = (T - k) / 2 - 1`` through the approximation of Edgerton and Wells
  (1994); levels 20, 10, 5, 2 and 1 percent, and at least 20 recursive
  residuals (the approximation is poor below that).
* A boundary crossing rejects stability. It is reported as
  ``first_crossing`` and is not an estimate of a break date; the tests
  report no break date.

Markov switching
----------------

* **Transition matrix.** ``regime_transition[i, j]`` is the probability of
  moving from regime ``j`` to regime ``i``; columns sum to one.
* **Regime labels.** By default the regimes are numbered in the order in
  which they first appear in the sample; every result (parameter table,
  ``regime_params``, probabilities, transition matrix) uses the same labels.
  Models with restricted transition probabilities are not relabeled.
* **MarkovAR and MarkovADL.** ``MarkovAR`` is in Hamilton's mean-adjusted
  form, ``y_t - mu(S_t) = sum_i phi_i (y_{t-i} - mu(S_{t-i})) + e_t``, so its
  ``const`` is a regime mean. ``MarkovADL`` enters lagged ``y`` as a
  regressor, so its ``const`` is an intercept. Within a regime that lasts,
  the intercept is the mean times ``1 - sum(phi)``, but the two are
  different models: after a change of regime the mean-adjusted form moves
  to the new mean at once, the intercept form gradually.
* **Start of the chain.** The initial distribution is that of the regime
  before the sample: two periods before the first observation in a
  regression and one period before it in an autoregression (the timing of
  statsmodels' filter). Unrestricted models use the stationary distribution;
  a non-recurring model starts in regime 0.
* **Restricted models.** Standard errors and ``df_model`` count only the
  parameters that are free under the restrictions.
* **Switching variance with two or more lags.** statsmodels (0.14, 0.15)
  assigns the variance of period ``t`` to the regime of period
  ``t - order + 1`` in ``MarkovAutoregression``; this package assigns it to
  the regime of period ``t``.
* **Estimation.** The likelihood has local maxima. ``fit(search_reps=n)``
  adds ``n`` random starting values; the default is none for the models and
  several inside the tests. Results of a search with random starting values
  vary from run to run unless a seed is given (``random_state`` in the
  tests).
* **Number of regimes.** ``RegimeNumberSelection`` compares models with
  ``K = 1, ..., k_max`` regimes on one sample; the one-regime model has the
  regressors, trend and (for ``model_type="ar"``) the mean-adjusted form of
  the switching models. Parameter counts include the error variance(s) and
  the ``K (K - 1)`` free transition probabilities. The consistency of AIC,
  BIC and HQIC for the number of regimes is not established.
* **Bootstrap tests.** The likelihood ratio statistic for ``K`` against
  ``K + 1`` regimes has no standard distribution (parameters are not
  identified under the null, and transition probabilities are on the
  boundary). ``method="sequential"`` uses the parametric bootstrap of
  Kasahara and Shimotsu (2018): samples are drawn from the estimated model
  with ``K`` regimes, with the regressors held fixed, and the models are
  estimated on each sample by the procedure used on the data. The
  regressors must be strictly exogenous: holding fixed a regressor that
  depends on past values of the dependent variable is not valid (Qu and
  Zhuo, 2021, report a rejection rate of 10 percent at a nominal 5 for
  that scheme). The consistency proof covers one regime against two in
  models that depend on the current regime only; for the mean-adjusted
  autoregression of ``model_type="ar"`` the support is simulation evidence
  (Rodriguez-Rondon and Dufour, 2025). The docstring of
  ``regimes.markov.selection`` sets this out in full.
* **Regime-specific variances in the selection.** The likelihood is
  unbounded as a regime variance tends to zero. Inside
  ``RegimeNumberSelection`` models with ``switching_variance=True`` are
  estimated with the penalty of Kasahara and Shimotsu (2018, Section
  10.1), ``-a_n sum_j {s2 / s2_j + log(s2_j / s2) - 1}`` with
  ``a_n = 20 / sqrt(n)`` and ``s2`` the variance of the one-regime model;
  the log-likelihood reported, the statistic and the information criteria
  are without the penalty. ``MarkovRegression`` and ``MarkovAR`` themselves
  are estimated without it.
* **Search.** With three or more regimes the likelihood has several local
  maxima; the result of the search can depend on ``random_state``, and
  ``search_reps`` sets the number of random starting values. ``converged``
  reports whether the optimizer converged.
* **Monte Carlo p-values.** A bootstrap p-value is
  ``(1 + #{simulated >= observed}) / (B + 1)``, and the test rejects when
  it is at most the level (Dufour, 2006). The level is exact when
  ``significance * (B + 1)`` is an integer: 19 replications suffice at 5
  percent, and the default 199 is exact at 5 and at 1 percent. A
  difference in log-likelihood within the optimizer's rounding (1e-4) is
  a statistic of zero, with p-value one.
* **Restriction tests.** For ``n`` transition probabilities restricted to
  zero, the chi-bar-squared reference is the bound
  ``0.5 chi2(n - 1) + 0.5 chi2(n)`` (Kodde and Palm, 1986). No published
  result gives the distribution of the statistic for the null of
  non-recurring regimes; the evidence for these tests is simulation only
  (see the validation page).
