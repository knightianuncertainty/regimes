"""Public API for regimes package.

This module provides a clean namespace for the most commonly used
classes and functions in the regimes package.
"""

# Models
# Diagnostics
from regimes.diagnostics import DiagnosticsResults, DiagnosticTestResult

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
from regimes.tests import BaiPerronResults, BaiPerronTest, ChowTest, ChowTestResults

# Visualization
from regimes.visualization import (
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
)

__all__ = [
    "ADL",
    "AR",
    # Models
    "OLS",
    "ADLResults",
    "ARResults",
    "BaiPerronResults",
    # Tests
    "BaiPerronTest",
    "ChowTest",
    "ChowTestResults",
    "BreakResultsBase",
    "CovType",
    # Diagnostics
    "DiagnosticTestResult",
    "DiagnosticsResults",
    "OLSResults",
    "RecursiveADL",
    "RecursiveADLResults",
    "RecursiveAR",
    "RecursiveARResults",
    "RecursiveOLS",
    "RecursiveOLSResults",
    # Base classes
    "RegimesModelBase",
    "RegimesResultsBase",
    "RegressionResultsBase",
    "RollingADL",
    "RollingADLResults",
    "RollingAR",
    "RollingARResults",
    "RollingCovType",
    "RollingEstimatorBase",
    # Rolling/Recursive estimation
    "RollingOLS",
    "RollingOLSResults",
    "RollingResultsBase",
    "TimeSeriesModelBase",
    "adl_summary_by_regime",
    "ar_summary_by_regime",
    "plot_actual_fitted",
    "plot_break_confidence",
    # Visualization
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
