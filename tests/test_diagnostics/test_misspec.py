"""Tests for misspecification diagnostic tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import regimes as rg
from regimes.diagnostics import (
    DiagnosticsResults,
    DiagnosticTestResult,
    arch_test,
    autocorrelation_test,
    compute_diagnostics,
    heteroskedasticity_test,
    normality_test,
)

# =============================================================================
# Fixtures for generating test data
# =============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def well_specified_residuals(rng: np.random.Generator) -> NDArray[np.floating[Any]]:
    """IID normal residuals (should pass all tests)."""
    return rng.standard_normal(200)


@pytest.fixture
def ar_residuals(rng: np.random.Generator) -> NDArray[np.floating[Any]]:
    """AR(1) residuals with phi=0.6 (should fail autocorrelation test)."""
    n = 200
    e = np.zeros(n)
    phi = 0.6
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.standard_normal()
    return e


@pytest.fixture
def arch_residuals(rng: np.random.Generator) -> NDArray[np.floating[Any]]:
    """ARCH(1) residuals (should fail ARCH test)."""
    n = 200
    e = np.zeros(n)
    alpha0 = 0.1
    alpha1 = 0.7  # Strong ARCH effect
    for t in range(1, n):
        sigma2 = alpha0 + alpha1 * e[t - 1] ** 2
        e[t] = np.sqrt(sigma2) * rng.standard_normal()
    return e


@pytest.fixture
def non_normal_residuals(rng: np.random.Generator) -> NDArray[np.floating[Any]]:
    """Chi-square residuals (should fail normality test)."""
    # Chi-square is right-skewed, non-normal
    return rng.chisquare(df=3, size=200) - 3  # Center at 0


@pytest.fixture
def simple_exog(rng: np.random.Generator) -> NDArray[np.floating[Any]]:
    """Simple design matrix with constant and one regressor."""
    n = 200
    return np.column_stack([np.ones(n), rng.standard_normal(n)])


# =============================================================================
# Unit tests for individual diagnostic functions
# =============================================================================


class TestNormality:
    """Tests for Jarque-Bera normality test."""

    def test_returns_diagnostic_result(
        self, well_specified_residuals: NDArray[np.floating[Any]]
    ) -> None:
        """Test that function returns correct type."""
        result = normality_test(well_specified_residuals)
        assert isinstance(result, DiagnosticTestResult)
        assert result.test_name == "Normality test (JB)"
        assert result.df == 2

    def test_passes_for_normal_residuals(
        self, well_specified_residuals: NDArray[np.floating[Any]]
    ) -> None:
        """Normal residuals should pass the test (p > 0.05)."""
        result = normality_test(well_specified_residuals)
        assert result.pvalue > 0.05

    def test_fails_for_non_normal_residuals(
        self, non_normal_residuals: NDArray[np.floating[Any]]
    ) -> None:
        """Non-normal residuals should fail the test (p < 0.05)."""
        result = normality_test(non_normal_residuals)
        assert result.pvalue < 0.05


class TestARCH:
    """Tests for ARCH-LM test."""

    def test_returns_diagnostic_result(
        self, well_specified_residuals: NDArray[np.floating[Any]]
    ) -> None:
        """Test that function returns correct type."""
        result = arch_test(well_specified_residuals, nlags=1)
        assert isinstance(result, DiagnosticTestResult)
        assert "ARCH" in result.test_name
        assert result.df == 1

    def test_passes_for_iid_residuals(
        self, well_specified_residuals: NDArray[np.floating[Any]]
    ) -> None:
        """IID residuals should pass the test (p > 0.05)."""
        result = arch_test(well_specified_residuals, nlags=1)
        assert result.pvalue > 0.05

    def test_fails_for_arch_residuals(
        self, arch_residuals: NDArray[np.floating[Any]]
    ) -> None:
        """ARCH residuals should fail the test (p < 0.05)."""
        result = arch_test(arch_residuals, nlags=1)
        assert result.pvalue < 0.05

    def test_multiple_lags(
        self, well_specified_residuals: NDArray[np.floating[Any]]
    ) -> None:
        """Test with multiple lags."""
        result = arch_test(well_specified_residuals, nlags=3)
        assert result.df == 3
        assert "1-3" in result.test_name


class TestAutocorrelation:
    """Tests for Breusch-Godfrey LM test."""

    def test_returns_diagnostic_result(
        self,
        well_specified_residuals: NDArray[np.floating[Any]],
        simple_exog: NDArray[np.floating[Any]],
    ) -> None:
        """Test that function returns correct type."""
        result = autocorrelation_test(well_specified_residuals, simple_exog, nlags=2)
        assert isinstance(result, DiagnosticTestResult)
        assert "AR" in result.test_name
        assert result.df == 2

    def test_passes_for_iid_residuals(
        self,
        well_specified_residuals: NDArray[np.floating[Any]],
        simple_exog: NDArray[np.floating[Any]],
    ) -> None:
        """IID residuals should pass the test (p > 0.05)."""
        result = autocorrelation_test(well_specified_residuals, simple_exog, nlags=2)
        assert result.pvalue > 0.05

    def test_fails_for_ar_residuals(
        self,
        ar_residuals: NDArray[np.floating[Any]],
        simple_exog: NDArray[np.floating[Any]],
    ) -> None:
        """AR residuals should fail the test (p < 0.05)."""
        result = autocorrelation_test(ar_residuals, simple_exog, nlags=2)
        assert result.pvalue < 0.05


class TestHeteroskedasticity:
    """Tests for White's test."""

    def test_returns_diagnostic_result(
        self,
        well_specified_residuals: NDArray[np.floating[Any]],
        simple_exog: NDArray[np.floating[Any]],
    ) -> None:
        """Test that function returns correct type."""
        result = heteroskedasticity_test(well_specified_residuals, simple_exog)
        assert isinstance(result, DiagnosticTestResult)
        assert "White" in result.test_name

    def test_passes_for_homoskedastic_residuals(
        self,
        well_specified_residuals: NDArray[np.floating[Any]],
        simple_exog: NDArray[np.floating[Any]],
    ) -> None:
        """Homoskedastic residuals should pass the test (p > 0.05)."""
        result = heteroskedasticity_test(well_specified_residuals, simple_exog)
        assert result.pvalue > 0.05

    def test_fails_for_heteroskedastic_residuals(
        self, rng: np.random.Generator
    ) -> None:
        """Heteroskedastic residuals should fail the test (p < 0.05)."""
        n = 200
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])
        # Residuals with variance proportional to x^2
        resid = rng.standard_normal(n) * (1 + np.abs(x))
        result = heteroskedasticity_test(resid, X)
        assert result.pvalue < 0.05


# =============================================================================
# Tests for compute_diagnostics
# =============================================================================


class TestComputeDiagnostics:
    """Tests for the compute_diagnostics convenience function."""

    def test_returns_diagnostics_results(
        self,
        well_specified_residuals: NDArray[np.floating[Any]],
        simple_exog: NDArray[np.floating[Any]],
    ) -> None:
        """Test that function returns correct type with all tests."""
        results = compute_diagnostics(well_specified_residuals, simple_exog)
        assert isinstance(results, DiagnosticsResults)
        assert results.autocorrelation is not None
        assert results.arch is not None
        assert results.normality is not None
        assert results.heteroskedasticity is not None

    def test_all_pass_for_well_specified(
        self,
        well_specified_residuals: NDArray[np.floating[Any]],
        simple_exog: NDArray[np.floating[Any]],
    ) -> None:
        """Well-specified residuals should pass all tests."""
        results = compute_diagnostics(well_specified_residuals, simple_exog)
        assert results.all_pass

    def test_custom_lags(
        self,
        well_specified_residuals: NDArray[np.floating[Any]],
        simple_exog: NDArray[np.floating[Any]],
    ) -> None:
        """Test with custom lag parameters."""
        results = compute_diagnostics(
            well_specified_residuals, simple_exog, lags_autocorr=4, lags_arch=2
        )
        assert results.autocorrelation is not None
        assert results.autocorrelation.df == 4
        assert results.arch is not None
        assert results.arch.df == 2


# =============================================================================
# Tests for DiagnosticsResults
# =============================================================================


class TestDiagnosticsResults:
    """Tests for DiagnosticsResults dataclass."""

    def test_summary_output(
        self,
        well_specified_residuals: NDArray[np.floating[Any]],
        simple_exog: NDArray[np.floating[Any]],
    ) -> None:
        """Test that summary produces formatted output."""
        results = compute_diagnostics(well_specified_residuals, simple_exog)
        summary = results.summary()
        assert "Misspecification Tests" in summary
        assert "AR 1-2 test" in summary
        assert "ARCH" in summary
        assert "Normality" in summary
        assert "White" in summary

    def test_all_pass_false_when_test_fails(
        self,
        ar_residuals: NDArray[np.floating[Any]],
        simple_exog: NDArray[np.floating[Any]],
    ) -> None:
        """all_pass should be False when any test fails."""
        results = compute_diagnostics(ar_residuals, simple_exog)
        assert not results.all_pass


# =============================================================================
# Integration tests with OLS
# =============================================================================


class TestOLSDiagnostics:
    """Integration tests for OLSResults.diagnostics()."""

    def test_diagnostics_method_exists(self, rng: np.random.Generator) -> None:
        """Test that OLSResults has diagnostics method."""
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [1, 2] + rng.standard_normal(n)

        model = rg.OLS(y, X)
        results = model.fit()
        assert hasattr(results, "diagnostics")

    def test_diagnostics_returns_results(self, rng: np.random.Generator) -> None:
        """Test that diagnostics() returns DiagnosticsResults."""
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [1, 2] + rng.standard_normal(n)

        model = rg.OLS(y, X)
        results = model.fit()
        diag = results.diagnostics()
        assert isinstance(diag, DiagnosticsResults)

    def test_diagnostics_caching(self, rng: np.random.Generator) -> None:
        """Test that diagnostics are cached."""
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [1, 2] + rng.standard_normal(n)

        model = rg.OLS(y, X)
        results = model.fit()

        diag1 = results.diagnostics()
        diag2 = results.diagnostics()
        assert diag1 is diag2  # Same object (cached)

    def test_diagnostics_recompute_on_lag_change(
        self, rng: np.random.Generator
    ) -> None:
        """Test that changing lags recomputes diagnostics."""
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [1, 2] + rng.standard_normal(n)

        model = rg.OLS(y, X)
        results = model.fit()

        diag1 = results.diagnostics(lags_autocorr=2, lags_arch=1)
        diag2 = results.diagnostics(lags_autocorr=4, lags_arch=2)
        assert diag1 is not diag2  # Different objects (recomputed)

    def test_summary_with_diagnostics(self, rng: np.random.Generator) -> None:
        """Test that summary(diagnostics=True) includes tests."""
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [1, 2] + rng.standard_normal(n)

        model = rg.OLS(y, X)
        results = model.fit()
        summary = results.summary(diagnostics=True)

        assert "Misspecification Tests" in summary
        assert "AR 1-2 test" in summary

    def test_summary_without_diagnostics(self, rng: np.random.Generator) -> None:
        """Test that summary(diagnostics=False) excludes tests."""
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [1, 2] + rng.standard_normal(n)

        model = rg.OLS(y, X)
        results = model.fit()
        summary = results.summary(diagnostics=False)

        assert "Misspecification Tests" not in summary

    def test_well_specified_model_passes(self, rng: np.random.Generator) -> None:
        """A well-specified OLS model should pass all diagnostics."""
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [1, 2] + rng.standard_normal(n)

        model = rg.OLS(y, X)
        results = model.fit()
        diag = results.diagnostics()

        # Should pass all tests (well-specified)
        assert diag.all_pass

    def test_ar_model_fails_autocorrelation(self, rng: np.random.Generator) -> None:
        """Model with AR errors should fail autocorrelation test."""
        n = 200
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])

        # Generate y with AR(1) errors
        e = np.zeros(n)
        phi = 0.7
        for t in range(1, n):
            e[t] = phi * e[t - 1] + rng.standard_normal()
        y = X @ [1, 2] + e

        model = rg.OLS(y, X)
        results = model.fit()
        diag = results.diagnostics()

        assert diag.autocorrelation is not None
        assert diag.autocorrelation.pvalue < 0.05


# =============================================================================
# API export tests
# =============================================================================


class TestAPIExports:
    """Test that diagnostic classes are exported correctly."""

    def test_diagnostic_test_result_exported(self) -> None:
        """DiagnosticTestResult should be in public API."""
        assert hasattr(rg, "DiagnosticTestResult")

    def test_diagnostics_results_exported(self) -> None:
        """DiagnosticsResults should be in public API."""
        assert hasattr(rg, "DiagnosticsResults")
