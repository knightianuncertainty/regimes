"""regimes: Structural break detection and estimation for time-series econometrics.

A Python package extending statsmodels with structural break detection
and estimation capabilities for econometric time-series analysis.

Example
-------
>>> import regimes as rg
>>> import numpy as np
>>>
>>> # Simulate data with one break
>>> np.random.seed(42)
>>> y = np.concatenate([np.random.randn(100), np.random.randn(100) + 2])
>>>
>>> # Test for breaks using Bai-Perron
>>> test = rg.BaiPerronTest(y)
>>> results = test.fit(max_breaks=3)
>>> print(f"Detected {results.n_breaks} break(s) at: {results.break_indices}")
>>>
>>> # Fit AR model with known break
>>> ar_model = rg.AR(y, lags=1, breaks=[100])
>>> ar_results = ar_model.fit(cov_type="HAC")
>>> print(ar_results.summary())
>>>
>>> # Fit ADL model
>>> x = np.random.randn(len(y))
>>> adl_model = rg.ADL(y, x, lags=1, exog_lags=1)
>>> adl_results = adl_model.fit()
>>> print(adl_results.summary())
"""

from regimes._version import __version__
from regimes.api import (
    ADL,
    AR,
    OLS,
    ADLResults,
    ARResults,
    BaiPerronResults,
    BaiPerronTest,
    ChowTest,
    ChowTestResults,
    BreakResultsBase,
    CovType,
    DiagnosticsResults,
    DiagnosticTestResult,
    OLSResults,
    RecursiveADL,
    RecursiveADLResults,
    RecursiveAR,
    RecursiveARResults,
    RecursiveOLS,
    RecursiveOLSResults,
    RegimesModelBase,
    RegimesResultsBase,
    RegressionResultsBase,
    RollingADL,
    RollingADLResults,
    RollingAR,
    RollingARResults,
    RollingCovType,
    RollingEstimatorBase,
    RollingOLS,
    RollingOLSResults,
    RollingResultsBase,
    TimeSeriesModelBase,
    adl_summary_by_regime,
    ar_summary_by_regime,
    plot_actual_fitted,
    plot_break_confidence,
    plot_breaks,
    plot_diagnostics,
    plot_params_over_time,
    plot_regime_means,
    plot_residual_acf,
    plot_residual_distribution,
    plot_rolling_coefficients,
    plot_scaled_residuals,
    summary_by_regime,
)

__all__ = [
    "ADL",
    "AR",
    "OLS",
    "ADLResults",
    "ARResults",
    "BaiPerronResults",
    "BaiPerronTest",
    "ChowTest",
    "ChowTestResults",
    "BreakResultsBase",
    "CovType",
    "DiagnosticTestResult",
    "DiagnosticsResults",
    "OLSResults",
    "RecursiveADL",
    "RecursiveADLResults",
    "RecursiveAR",
    "RecursiveARResults",
    "RecursiveOLS",
    "RecursiveOLSResults",
    "RegimesModelBase",
    "RegimesResultsBase",
    "RegressionResultsBase",
    "RollingADL",
    "RollingADLResults",
    "RollingAR",
    "RollingARResults",
    "RollingCovType",
    "RollingEstimatorBase",
    "RollingOLS",
    "RollingOLSResults",
    "RollingResultsBase",
    "TimeSeriesModelBase",
    "__version__",
    "adl_summary_by_regime",
    "ar_summary_by_regime",
    "plot_actual_fitted",
    "plot_break_confidence",
    "plot_breaks",
    "plot_diagnostics",
    "plot_params_over_time",
    "plot_regime_means",
    "plot_residual_acf",
    "plot_residual_distribution",
    "plot_rolling_coefficients",
    "plot_scaled_residuals",
    "summary_by_regime",
]
