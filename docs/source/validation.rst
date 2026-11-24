Validation
==========

What each method was checked against. A check listed as a *test* runs in the
package's test suite on every change (the file is given); the others were
run once and their scripts are kept with the package's audit records.
Reference values from R and Ox are stored in ``tests/reference_data``
together with the scripts that generated them; the test suite itself needs
neither R nor Ox.

Simulated rejection rates are for the null hypothesis with independent
normal errors, at nominal levels of 10, 5 and 1 percent unless stated.

Each correction of November 2026 was read by an independent reviewer before
it was proposed for merging: a fresh reader who was given the code, not
the author's reasoning, and who ran the code. What the reviews found is
fixed or listed under the limits below.

Regression models
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Checked against
   * - OLS coefficients, standard errors (nonrobust, HC0 to HC3, HAC),
       log-likelihood, AIC, BIC
     - statsmodels ``OLS``: equal to 1e-10 or better (test:
       ``test_models/test_model_corrections.py``)
   * - Regime coefficients with breaks
     - OLS on each sub-sample (test, same file)
   * - AR
     - statsmodels ``AutoReg`` coefficients (test, same file)
   * - ADL
     - statsmodels ``ARDL`` coefficients (test, same file)
   * - ``ADL.select_lags``
     - a grid of statsmodels ``ARDL`` fits on the common sample (two
       regressors; trends none, constant, constant and trend; AIC and BIC)
       and ``ardl_select_order`` with one regressor; the choice does not
       change with the units of the data (test, same file)
   * - F statistic (classical), with and without a constant, with a
       constant that is not one, a full set of dummies, regime constants
     - statsmodels ``OLS.fvalue`` with ``cov_type="nonrobust"`` (test, same
       file)
   * - Default HAC lag of AR and ADL
     - statsmodels ``AutoReg`` and ``ARDL`` with the same lag (test, same
       file)
   * - Every code example of the README and of these pages
     - run from top to bottom against the released code (once, at this
       documentation pass)
   * - Rolling OLS
     - statsmodels ``RollingOLS`` (test: ``test_rolling/test_rolling_ols.py``)
   * - Recursive OLS
     - statsmodels ``RecursiveLS`` (test:
       ``test_models/test_model_corrections.py``)
   * - HC0 to HC3 standard errors
     - OxRegimes ``RgCovar`` (once, 7e-15)

Bai-Perron
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Checked against
   * - Break dates, SSR, sup F, sup F(l+1|l), sequential procedure; pure and
       partial structural change
     - Perron's GAUSS program on its example data (test:
       ``test_tests/test_bp_core.py``)
   * - All statistics, selection rules, break dates, intervals
     - OxRegimes on 21 option sets, to a relative 1e-6 (test:
       ``test_tests/test_bai_perron_ox_parity.py``)
   * - Tests, selection rules, break dates and regime means of Bai and
       Perron (2003), Table I
     - the published table, from the data in the journal's archive (test:
       ``test_tests/test_bai_perron_published.py``). The published
       confidence intervals are not reproduced.
   * - Break dates and SSR on three simulated data sets
     - R ``strucchange`` (test: ``test_tests/test_bai_perron_reference.py``)
   * - Critical values
     - the tables of Perron's program; they agree entry by entry with those
       of the R package ``mbreaks``

Chow, CUSUM and Andrews-Ploberger
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Checked against
   * - Chow test (break-point, partial break)
     - F tests of statsmodels ``OLS`` (test:
       ``test_tests/test_break_test_references.py``)
   * - Break dates of tests built from AR and ADL models
     - a known break in simulated data is recovered at its own date by
       each of the five tests and, passed back to the model, reproduces the
       sum of squared residuals of the test (test:
       ``test_tests/test_model_coordinates.py``)
   * - Recursive residuals, CUSUM path
     - statsmodels ``RecursiveLS``, and a brute-force recursive least
       squares for regressors that are collinear at the start or contain a
       dummy variable: differences of 1e-14; the sum of squared recursive
       residuals equals the residual sum of squares, also for regressors
       that grow by nine orders of magnitude over the sample; the
       observations without a residual are those at which the rank rises
       in exact rational arithmetic, for about two hundred designs with
       exact linear dependencies in part of the sample, first observations
       that are almost equal, kinked regressors, and one very large value
       (tests, same file: ``TestDependenciesAgainstExactArithmetic``,
       ``TestDesignsFromTheSecondReview``); agreement with the computation
       of version 0.5.0 on ordinary designs to 1e-11; a cubic in calendar
       years against the same trend measured from the start of the sample:
       differences below 2e-3
   * - CUSUM-of-squares path and bounds
     - statsmodels ``RecursiveLS`` (test, same file)
   * - Sequence of F statistics, SupF, AveF, ExpF and their p-values
     - R ``strucchange`` (``Fstats``, ``sctest``) on seven cases: equal to
       1e-9 (test, same file)
   * - Hansen (1997) p-values
     - Hansen's own program on 1,680 points: equal to 1e-9 (test, same
       file)
   * - Critical values, one coefficient, trimming 0.15, 5 percent
     - published: SupF 8.68 (Andrews, 2003), ExpF 2.06 and AveF 2.88
       (Andrews and Ploberger, 1994); the package gives 8.61, 2.04 and 2.87
       (test, same file)
   * - SupF
     - ``BaiPerronTest`` with one break and a common variance (test, same
       file)

Simulated rejection rates (2,000 replications, percent at 10/5/1):

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - Design
     - SupF
     - ExpF
     - AveF
   * - T = 100, q = 1
     - 9.2 / 4.2 / 1.1
     - 10.8 / 4.9 / 1.5
     - 10.2 / 5.1 / 1.1
   * - T = 100, q = 2
     - 9.8 / 5.2 / 1.5
     - 11.8 / 6.6 / 1.8
     - 10.9 / 5.3 / 1.1
   * - T = 100, q = 5
     - 10.8 / 6.3 / 1.4
     - 13.2 / 8.1 / 2.1
     - 11.1 / 6.4 / 1.8
   * - T = 200, q = 1
     - 9.0 / 4.8 / 0.7
     - 10.5 / 4.9 / 0.9
     - 10.3 / 5.0 / 0.9
   * - T = 200, q = 2
     - 9.9 / 4.8 / 0.8
     - 11.7 / 5.6 / 0.8
     - 11.2 / 5.2 / 0.7
   * - T = 200, q = 5
     - 10.8 / 5.1 / 1.5
     - 12.1 / 6.2 / 1.9
     - 10.6 / 5.4 / 1.5

Rejecting when any one of the three tests rejects at 5 percent has a
rejection rate of 6.5 to 9.6 percent in these designs: the three are
separate tests.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Sample
     - CUSUM
     - CUSUM of squares
   * - T = 50
     - 7.2 / 3.4 / 0.3
     - 10.9 / 5.3 / 1.4
   * - T = 100
     - 7.8 / 4.2 / 0.5
     - 11.8 / 5.4 / 1.1
   * - T = 200
     - 8.8 / 4.2 / 0.8
     - 9.8 / 4.7 / 1.3

The CUSUM test is conservative, as expected for its boundary. ExpF is
somewhat liberal with five breaking coefficients in the smaller sample.

Markov switching
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Checked against
   * - Likelihood, and the simulator of a fitted model
     - the likelihood computed by brute force over all regime paths, with
       the timing of the simulator, for regressions and autoregressions of
       order one and two, with a switching variance and with restricted
       transitions (test: ``test_markov/test_simulate.py``). The same test
       established the timing of the initial distribution and exposed the
       statsmodels defect described under :doc:`conventions`.
   * - Simulated series
     - re-estimation recovers the parameters they were drawn from (test,
       same file)
   * - One-regime model of ``RegimeNumberSelection``
     - least squares; statsmodels ``AutoReg``; a brute-force maximum of the
       mean-adjusted likelihood with regressors (test:
       ``test_markov/test_number_of_regimes.py``)
   * - Likelihood function with two and three regimes
     - the R package MSTest: at MSTest's estimates the likelihood equals
       the value MSTest reports to 1e-6; the one-regime log-likelihood is
       identical (test, same file). The two-regime maximum found here,
       -459.6015, is slightly above the value at which MSTest's EM
       algorithm stops, -459.6104.
   * - Chi-bar-squared critical values of the restriction tests
     - Kodde and Palm (1986), Table 1: 2.706, 5.138, 7.045, 8.761 at 5
       percent (test: ``test_markov/test_sequential_restriction.py``)
   * - Monte Carlo decision rule
     - the level ``floor(alpha (B + 1)) / (B + 1)`` by simulation (test:
       ``test_markov/test_simulate.py``)

Simulated rejection rates of ``NonRecurringRegimeTest`` at a nominal 5
percent (the bootstrap version with 19 replications, which is an exact test
if the bootstrap distribution is right):

.. list-table::
   :header-rows: 1
   :widths: 46 14 20 20

   * - Design
     - Samples
     - Chi-bar-squared
     - Bootstrap
   * - Null: one mean shift of 2 at 50, T = 100
     - 98
     - 0
     - 0
   * - Null: one mean shift of 1.5 at 60, T = 200
     - 99
     - 0
     - 0
   * - Null: one mean shift of 3 at 100, T = 200
     - 97
     - 0
     - 0
   * - Null: two mean shifts of 3, three regimes, T = 240
     - 12
     - 0
     - not run
   * - Alternative: two recurring regimes (probability 0.95 of staying,
       means 3 apart), T = 200
     - 40 / 8
     - 40 of 40
     - 8 of 8

Under the null the statistic is zero in 99 percent of the samples (the
unrestricted estimate is itself non-recurring), so both versions are valid
and very conservative in these designs. Power against alternatives close to
the null was not measured.

.. regime-number-simulation-start

Simulated rejections of the bootstrap test of one regime against two in
``RegimeNumberSelection`` (T = 100, default settings, 19 bootstrap
replications, which give an exact test at 5 percent if the bootstrap
distribution is right):

.. list-table::
   :header-rows: 1
   :widths: 52 16 16 16

   * - Design
     - Samples
     - At 5 percent
     - At 10 percent
   * - Null: one regime, independent normal data
     - 100
     - 5
     - 6
   * - Null: one regime, ``y = 1 + 2x + e`` with a fixed regressor
     - 60
     - 4
     - 5
   * - Alternative: two recurring regimes (probabilities 0.95 and 0.90 of
       staying, means 2 apart)
     - 30
     - 27
     - 28

With 100 samples the standard error of a rejection rate of 5 percent is
about 2 percentage points (3 with 60), so the counts are consistent with
the nominal levels and rule out a gross error (the test this replaces
rejected a true null about 20 percent of the time); they do not establish
the size to the last percentage point. Two regimes against three,
autoregressions and regime-specific variances were not simulated here.

.. regime-number-simulation-end

Not checked against another implementation: the bootstrap test for the
number of regimes as a whole (MSTest's ``LMCLRTest`` simulates with a
burn-in period and uses one starting value per sample, so its p-values are
not comparable draw by draw), and the restriction tests, for which no other
implementation or published distribution exists.

Not validated
-------------

* Markov switching models with four or more regimes.
* The size of the restriction tests in models with regressors or
  autoregressive terms, and their power against alternatives close to the
  null.
* The size of the bootstrap test for the number of regimes beyond the
  designs above: two regimes against three, autoregressions, switching
  variances and models with regressors rest on the simulations in the
  cited papers, which do not cover switching autoregressive coefficients
  or exogenous regressors.
* The search for the maximum of the likelihood with three or more regimes,
  which can end at different local maxima for different seeds.
* Rolling and recursive estimates of AR and ADL models against another
  package (they equal a direct fit on each window by construction).
* Break-date confidence intervals of Bai and Perron (2003), Table I, which
  the package does not reproduce.
* Indicator saturation is not part of this release.
