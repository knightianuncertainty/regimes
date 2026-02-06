"""PcGive-style misspecification tests for regression residuals.

This module implements four standard diagnostic tests:
1. Autocorrelation test - Breusch-Godfrey LM test
2. ARCH test - Engle's ARCH-LM test
3. Normality test - Jarque-Bera test
4. Heteroskedasticity test - White's test
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_white

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class DiagnosticTestResult:
    """Result from a single diagnostic test.

    Attributes
    ----------
    test_name : str
        Name of the test (e.g., "AR 1-2 test (LM)").
    null_hypothesis : str
        Description of the null hypothesis.
    statistic : float
        Test statistic value.
    pvalue : float
        P-value of the test.
    df : int | tuple[int, int] | None
        Degrees of freedom (int for chi-square, tuple for F).
    """

    test_name: str
    null_hypothesis: str
    statistic: float
    pvalue: float
    df: int | tuple[int, int] | None = None

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"{self.test_name}: statistic={self.statistic:.4f}, "
            f"p-value={self.pvalue:.4f}"
        )


@dataclass
class DiagnosticsResults:
    """Collection of diagnostic test results.

    Attributes
    ----------
    autocorrelation : DiagnosticTestResult | None
        Breusch-Godfrey LM test for autocorrelation.
    arch : DiagnosticTestResult | None
        Engle's ARCH-LM test for conditional heteroskedasticity.
    normality : DiagnosticTestResult | None
        Jarque-Bera test for normality.
    heteroskedasticity : DiagnosticTestResult | None
        White's test for heteroskedasticity.
    """

    autocorrelation: DiagnosticTestResult | None = None
    arch: DiagnosticTestResult | None = None
    normality: DiagnosticTestResult | None = None
    heteroskedasticity: DiagnosticTestResult | None = None

    @property
    def all_pass(self) -> bool:
        """Check if all tests pass at 5% significance level."""
        tests = [
            self.autocorrelation,
            self.arch,
            self.normality,
            self.heteroskedasticity,
        ]
        return all(t is None or t.pvalue >= 0.05 for t in tests)

    def summary(self) -> str:
        """Generate a text summary of diagnostic tests.

        Returns
        -------
        str
            Formatted summary table of test results.
        """
        lines = []
        lines.append("=" * 81)
        lines.append(f"{'Misspecification Tests':^81}")
        lines.append("-" * 81)
        lines.append(f"{'Test':<35} {'Statistic':>12} {'P-value':>12}")
        lines.append("-" * 81)

        tests = [
            self.autocorrelation,
            self.arch,
            self.normality,
            self.heteroskedasticity,
        ]

        for test in tests:
            if test is not None:
                lines.append(
                    f"{test.test_name:<35} {test.statistic:>12.4f} {test.pvalue:>12.4f}"
                )

        lines.append("=" * 81)
        return "\n".join(lines)


def normality_test(resid: NDArray[np.floating[Any]]) -> DiagnosticTestResult:
    """Jarque-Bera test for normality of residuals.

    Tests the null hypothesis that residuals are normally distributed
    based on their skewness and kurtosis.

    Parameters
    ----------
    resid : NDArray
        Regression residuals.

    Returns
    -------
    DiagnosticTestResult
        Test result with statistic, p-value, and degrees of freedom.

    Notes
    -----
    The test statistic follows a chi-square distribution with 2 degrees
    of freedom under the null hypothesis of normality.
    """
    statistic, pvalue = stats.jarque_bera(resid)
    return DiagnosticTestResult(
        test_name="Normality test (JB)",
        null_hypothesis="Residuals are normally distributed",
        statistic=float(statistic),
        pvalue=float(pvalue),
        df=2,
    )


def arch_test(resid: NDArray[np.floating[Any]], nlags: int = 1) -> DiagnosticTestResult:
    """Engle's ARCH-LM test for conditional heteroskedasticity.

    Tests the null hypothesis that there are no ARCH effects (no
    autoregressive conditional heteroskedasticity) in the residuals.

    Parameters
    ----------
    resid : NDArray
        Regression residuals.
    nlags : int, default 1
        Number of lags to include in the test.

    Returns
    -------
    DiagnosticTestResult
        Test result with LM statistic and p-value.

    Notes
    -----
    The test regresses squared residuals on their own lags. The LM
    statistic is n*R^2 from this auxiliary regression, which follows
    a chi-square distribution with nlags degrees of freedom.
    """
    n = len(resid)
    resid_sq = resid**2

    # Create lag matrix for squared residuals
    X = np.zeros((n - nlags, nlags))
    for lag in range(1, nlags + 1):
        X[:, lag - 1] = resid_sq[nlags - lag : n - lag]

    y = resid_sq[nlags:]
    X_const = sm.add_constant(X)

    # Auxiliary regression
    aux_model = sm.OLS(y, X_const)
    aux_results = aux_model.fit()

    # LM statistic = n * R^2
    lm_stat = len(y) * aux_results.rsquared
    pvalue = 1 - stats.chi2.cdf(lm_stat, nlags)

    return DiagnosticTestResult(
        test_name=f"ARCH 1-{nlags} test (LM)",
        null_hypothesis="No ARCH effects",
        statistic=float(lm_stat),
        pvalue=float(pvalue),
        df=nlags,
    )


def heteroskedasticity_test(
    resid: NDArray[np.floating[Any]], exog: NDArray[np.floating[Any]]
) -> DiagnosticTestResult:
    """White's test for heteroskedasticity.

    Tests the null hypothesis of homoskedasticity against the alternative
    of heteroskedasticity related to the regressors.

    Parameters
    ----------
    resid : NDArray
        Regression residuals.
    exog : NDArray
        Original regressors (including constant if present).

    Returns
    -------
    DiagnosticTestResult
        Test result with LM statistic and p-value.

    Notes
    -----
    Uses statsmodels' implementation which includes original regressors,
    their squares, and cross-products in the auxiliary regression.

    When the design matrix has regime-specific columns (from structural breaks),
    the constant columns contain zeros outside their regime. In this case,
    a constant column is prepended for the auxiliary regression.
    """
    # Check if exog has a constant column (all values ≈ 1.0)
    has_constant = any(np.allclose(exog[:, i], 1.0) for i in range(exog.shape[1]))

    # Add constant if needed for White's test auxiliary regression
    if has_constant:
        exog_for_test = exog
    else:
        exog_for_test = np.column_stack([np.ones(len(resid)), exog])

    lm_stat, lm_pvalue, _f_stat, _f_pvalue = het_white(resid, exog_for_test)
    return DiagnosticTestResult(
        test_name="Hetero test (White)",
        null_hypothesis="Homoskedasticity",
        statistic=float(lm_stat),
        pvalue=float(lm_pvalue),
        df=None,  # Degrees of freedom depends on number of auxiliary regressors
    )


def autocorrelation_test(
    resid: NDArray[np.floating[Any]],
    exog: NDArray[np.floating[Any]],
    nlags: int = 2,
) -> DiagnosticTestResult:
    """Breusch-Godfrey LM test for residual autocorrelation.

    Tests the null hypothesis of no autocorrelation up to the specified
    number of lags.

    Parameters
    ----------
    resid : NDArray
        Regression residuals.
    exog : NDArray
        Original regressors (including constant if present).
    nlags : int, default 2
        Number of lags to test for autocorrelation.

    Returns
    -------
    DiagnosticTestResult
        Test result with LM statistic and p-value.

    Notes
    -----
    The test augments the original regression with lagged residuals
    and tests their joint significance. The LM statistic is n*R^2
    from the auxiliary regression.
    """
    n = len(resid)

    # Create lagged residuals
    resid_lags = np.zeros((n, nlags))
    for lag in range(1, nlags + 1):
        resid_lags[lag:, lag - 1] = resid[:-lag]

    # Auxiliary regression: residuals on original regressors + lagged residuals
    # Start from observation nlags to avoid initial zeros
    y_aux = resid[nlags:]
    X_aux = np.column_stack([exog[nlags:], resid_lags[nlags:]])

    aux_model = sm.OLS(y_aux, X_aux)
    aux_results = aux_model.fit()

    # LM statistic = (n - nlags) * R^2
    lm_stat = len(y_aux) * aux_results.rsquared
    pvalue = 1 - stats.chi2.cdf(lm_stat, nlags)

    return DiagnosticTestResult(
        test_name=f"AR 1-{nlags} test (LM)",
        null_hypothesis="No autocorrelation",
        statistic=float(lm_stat),
        pvalue=float(pvalue),
        df=nlags,
    )


def compute_diagnostics(
    resid: NDArray[np.floating[Any]],
    exog: NDArray[np.floating[Any]],
    lags_autocorr: int = 2,
    lags_arch: int = 1,
) -> DiagnosticsResults:
    """Compute all misspecification tests.

    Parameters
    ----------
    resid : NDArray
        Regression residuals.
    exog : NDArray
        Original regressors (including constant if present).
    lags_autocorr : int, default 2
        Number of lags for the autocorrelation test.
    lags_arch : int, default 1
        Number of lags for the ARCH test.

    Returns
    -------
    DiagnosticsResults
        Collection of all diagnostic test results.
    """
    return DiagnosticsResults(
        autocorrelation=autocorrelation_test(resid, exog, nlags=lags_autocorr),
        arch=arch_test(resid, nlags=lags_arch),
        normality=normality_test(resid),
        heteroskedasticity=heteroskedasticity_test(resid, exog),
    )
