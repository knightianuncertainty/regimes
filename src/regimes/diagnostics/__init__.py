"""Misspecification tests and model diagnostics.

This module provides PcGive-style diagnostic tests for regression models:
- Autocorrelation test (Breusch-Godfrey LM)
- ARCH test (Engle's ARCH-LM)
- Normality test (Jarque-Bera)
- Heteroskedasticity test (White's test)
"""

from regimes.diagnostics.misspec import (
    DiagnosticsResults,
    DiagnosticTestResult,
    arch_test,
    autocorrelation_test,
    compute_diagnostics,
    heteroskedasticity_test,
    normality_test,
)

__all__ = [
    "DiagnosticTestResult",
    "DiagnosticsResults",
    "arch_test",
    "autocorrelation_test",
    "compute_diagnostics",
    "heteroskedasticity_test",
    "normality_test",
]
