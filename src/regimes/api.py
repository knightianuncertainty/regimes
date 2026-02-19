"""Public API for regimes package.

This module provides a clean namespace for the most commonly used
classes and functions in the regimes package.
"""

# Models
# Diagnostics
from regimes.diagnostics import DiagnosticsResults, DiagnosticTestResult

# Markov switching models
from regimes.markov import (
    MarkovADL,
    MarkovADLResults,
    MarkovAR,
    MarkovARResults,
    MarkovRegression,
    MarkovRegressionResults,
    MarkovSwitchingResultsBase,
    NonRecurringRegimeTest,
    NonRecurringRegimeTestResults,
    RegimeNumberSelection,
    RegimeNumberSelectionResults,
    RestrictedMarkovAR,
    RestrictedMarkovRegression,
    SequentialRestrictionResults,
    SequentialRestrictionTest,
)

# Model base classes (for extension)
from regimes.models import (
    ADL,
    AR,
    OLS,
    ADLResults,
    ARResults,
    CovType,
    OLSResults,
    RegimesModelBase,
    TimeSeriesModelBase,
    adl_summary_by_regime,
    ar_summary_by_regime,
    summary_by_regime,
)

# Results base classes (for type checking)
from regimes.results import (
    BreakResultsBase,
    RegimesResultsBase,
    RegressionResultsBase,
)

# Rolling/Recursive estimation
from regimes.rolling import (
    RecursiveADL,
    RecursiveADLResults,
    RecursiveAR,
    RecursiveARResults,
    RecursiveOLS,
    RecursiveOLSResults,
    RollingADL,
    RollingADLResults,
    RollingAR,
    RollingARResults,
    RollingCovType,
    RollingEstimatorBase,
    RollingOLS,
    RollingOLSResults,
    RollingResultsBase,
)

# Tests
from regimes.tests import (
    AndrewsPlobergerResults,
    AndrewsPlobergerTest,
    BaiPerronResults,
    BaiPerronTest,
    ChowTest,
    ChowTestResults,
    CUSUMResults,
    CUSUMSQResults,
    CUSUMSQTest,
    CUSUMTest,
)

# Visualization
from regimes.visualization import (
    plot_actual_fitted,
    plot_break_confidence,
    plot_breaks,
    plot_cusum,
    plot_cusum_sq,
    plot_diagnostics,
    plot_f_sequence,
    plot_ic,
    plot_parameter_time_series,
    plot_params_over_time,
    plot_regime_means,
    plot_regime_shading,
    plot_residual_acf,
    plot_residual_distribution,
    plot_rolling_coefficients,
    plot_scaled_residuals,
    plot_smoothed_probabilities,
    plot_transition_matrix,
)

__all__ = [
    "ADL",
    "AR",
    "OLS",
    "ADLResults",
    "ARResults",
    "AndrewsPlobergerResults",
    "AndrewsPlobergerTest",
    "BaiPerronResults",
    "BaiPerronTest",
    "BreakResultsBase",
    "CUSUMResults",
    "CUSUMSQResults",
    "CUSUMSQTest",
    "CUSUMTest",
    "ChowTest",
    "ChowTestResults",
    "CovType",
    "DiagnosticTestResult",
    "DiagnosticsResults",
    "MarkovADL",
    "MarkovADLResults",
    "MarkovAR",
    "MarkovARResults",
    "MarkovRegression",
    "MarkovRegressionResults",
    "MarkovSwitchingResultsBase",
    "NonRecurringRegimeTest",
    "NonRecurringRegimeTestResults",
    "OLSResults",
    "RecursiveADL",
    "RecursiveADLResults",
    "RecursiveAR",
    "RecursiveARResults",
    "RecursiveOLS",
    "RecursiveOLSResults",
    "RegimeNumberSelection",
    "RegimeNumberSelectionResults",
    "RegimesModelBase",
    "RegimesResultsBase",
    "RegressionResultsBase",
    "RestrictedMarkovAR",
    "RestrictedMarkovRegression",
    "RollingADL",
    "RollingADLResults",
    "RollingAR",
    "RollingARResults",
    "RollingCovType",
    "RollingEstimatorBase",
    "RollingOLS",
    "RollingOLSResults",
    "RollingResultsBase",
    "SequentialRestrictionResults",
    "SequentialRestrictionTest",
    "TimeSeriesModelBase",
    "adl_summary_by_regime",
    "ar_summary_by_regime",
    "plot_actual_fitted",
    "plot_break_confidence",
    "plot_breaks",
    "plot_cusum",
    "plot_cusum_sq",
    "plot_diagnostics",
    "plot_f_sequence",
    "plot_ic",
    "plot_parameter_time_series",
    "plot_params_over_time",
    "plot_regime_means",
    "plot_regime_shading",
    "plot_residual_acf",
    "plot_residual_distribution",
    "plot_rolling_coefficients",
    "plot_scaled_residuals",
    "plot_smoothed_probabilities",
    "plot_transition_matrix",
    "summary_by_regime",
]
