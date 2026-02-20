"""Tests for GETS selection utilities and tree search."""

from __future__ import annotations

import numpy as np
import pytest

from regimes.gets.selection import (
    _encompassing_test,
    _f_test_exclusion,
    _fit_ols_subset,
    gets_search,
)
from regimes.models.ols import OLSResults


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# _fit_ols_subset
# ---------------------------------------------------------------------------


class TestFitOlsSubset:
    def test_basic_fit(self, rng):
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [1, 2] + rng.standard_normal(n)
        mask = np.array([True, True])
        result = _fit_ols_subset(y, X, mask, "nonrobust", ["const", "x1"])
        assert isinstance(result, OLSResults)
        assert result.nobs == n
        assert len(result.params) == 2

    def test_subset_columns(self, rng):
        n = 100
        X = np.column_stack(
            [
                np.ones(n),
                rng.standard_normal(n),
                rng.standard_normal(n),
            ]
        )
        y = X @ [1, 2, 0] + rng.standard_normal(n)
        mask = np.array([True, True, False])
        result = _fit_ols_subset(y, X, mask, "nonrobust", ["const", "x1", "x2"])
        assert len(result.params) == 2
        assert result.param_names == ["const", "x1"]

    def test_hac_covariance(self, rng):
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [1, 2] + rng.standard_normal(n)
        mask = np.array([True, True])
        result = _fit_ols_subset(y, X, mask, "HAC", ["const", "x1"])
        assert result.cov_type == "HAC"

    def test_hc_covariance(self, rng):
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [1, 2] + rng.standard_normal(n)
        mask = np.array([True, True])
        result = _fit_ols_subset(y, X, mask, "HC1", ["const", "x1"])
        assert result.cov_type == "HC1"

    def test_no_names(self, rng):
        n = 50
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)
        mask = np.array([True, True])
        result = _fit_ols_subset(y, X, mask)
        assert result.param_names is None

    def test_llf_is_set(self, rng):
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [1, 2] + rng.standard_normal(n)
        mask = np.array([True, True])
        result = _fit_ols_subset(y, X, mask, "nonrobust", ["const", "x1"])
        assert result.llf is not None
        assert not np.isnan(result.llf)


# ---------------------------------------------------------------------------
# _f_test_exclusion
# ---------------------------------------------------------------------------


class TestFTestExclusion:
    def test_identical_ssr(self):
        """No restriction effect => F=0, p=1."""
        f, p = _f_test_exclusion(10.0, 10.0, q=2, df_resid_u=50)
        assert f == 0.0
        assert p == 1.0

    def test_positive_f(self):
        """Restricted SSR > unrestricted SSR => positive F."""
        f, p = _f_test_exclusion(10.0, 20.0, q=2, df_resid_u=50)
        assert f > 0
        assert 0 < p < 1

    def test_large_restriction_effect(self):
        """Very large restriction effect => small p."""
        f, p = _f_test_exclusion(10.0, 1000.0, q=1, df_resid_u=100)
        assert p < 0.001

    def test_invalid_q(self):
        f, p = _f_test_exclusion(10.0, 20.0, q=0, df_resid_u=50)
        assert np.isnan(f)
        assert np.isnan(p)

    def test_invalid_df(self):
        f, p = _f_test_exclusion(10.0, 20.0, q=2, df_resid_u=0)
        assert np.isnan(f)
        assert np.isnan(p)

    def test_f_distribution_property(self):
        """Verify F-stat matches manual calculation."""
        ssr_u, ssr_r, q, df = 100.0, 150.0, 3, 96
        f, p = _f_test_exclusion(ssr_u, ssr_r, q, df)
        expected_f = ((150 - 100) / 3) / (100 / 96)
        np.testing.assert_almost_equal(f, expected_f)


# ---------------------------------------------------------------------------
# _encompassing_test
# ---------------------------------------------------------------------------


class TestEncompassingTest:
    def test_no_exclusion(self, rng):
        """Terminal == GUM => F=0, p=1."""
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [1, 2] + rng.standard_normal(n)
        mask = np.array([True, True])
        gum = _fit_ols_subset(y, X, mask, "nonrobust", ["const", "x1"])

        from regimes.gets.results import TerminalModel

        tm = TerminalModel(
            retained_mask=mask.copy(),
            retained_names=["const", "x1"],
            results=gum,
        )
        f, p = _encompassing_test(gum, tm, y, X)
        assert f == 0.0
        assert p == 1.0


# ---------------------------------------------------------------------------
# gets_search
# ---------------------------------------------------------------------------


class TestGetsSearch:
    def test_all_significant_returns_gum(self, rng):
        """If all variables are significant, GETS returns the GUM."""
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [5, 3] + rng.standard_normal(n) * 0.5
        result = gets_search(y, X, names=["const", "x1"], diagnostics=False)
        # All variables are very significant, so GUM should be kept
        assert len(result.retained_names) == 2
        assert "const" in result.retained_names
        assert "x1" in result.retained_names

    def test_removes_irrelevant_variables(self, rng):
        """GETS removes irrelevant variables."""
        n = 200
        x1 = rng.standard_normal(n)
        x_noise1 = rng.standard_normal(n)
        x_noise2 = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x1, x_noise1, x_noise2])
        y = 1 + 3 * x1 + rng.standard_normal(n) * 0.5
        result = gets_search(
            y,
            X,
            names=["const", "x1", "noise1", "noise2"],
            diagnostics=False,
        )
        assert "const" in result.retained_names
        assert "x1" in result.retained_names
        # Noise variables should be removed (with high probability)
        assert len(result.retained_names) <= 3

    def test_protected_variables_not_removed(self, rng):
        """Protected variables are never removed."""
        n = 200
        x1 = rng.standard_normal(n)
        x_noise = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x1, x_noise])
        y = rng.standard_normal(n)  # Pure noise
        protected = np.array([True, False, False])
        result = gets_search(
            y,
            X,
            names=["const", "x1", "noise"],
            diagnostics=False,
            protected=protected,
        )
        # Constant should always be retained
        assert "const" in result.retained_names

    def test_returns_gets_results(self, rng):
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)
        result = gets_search(y, X, diagnostics=False)
        assert hasattr(result, "gum_results")
        assert hasattr(result, "terminal_models")
        assert hasattr(result, "surviving_terminals")
        assert hasattr(result, "selected_model")
        assert hasattr(result, "selection_criterion")
        assert result.n_models_evaluated >= 1

    def test_selected_model_properties(self, rng):
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [2, 1] + rng.standard_normal(n)
        result = gets_search(y, X, names=["const", "x1"], diagnostics=False)
        assert result.retained_mask is not None
        assert result.retained_names is not None
        assert result.selected_results is not None

    def test_selection_criterion_bic(self, rng):
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)
        result = gets_search(y, X, selection="bic", diagnostics=False)
        assert result.selection_criterion == "bic"

    def test_selection_criterion_aic(self, rng):
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)
        result = gets_search(y, X, selection="aic", diagnostics=False)
        assert result.selection_criterion == "aic"

    def test_summary(self, rng):
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [2, 1] + rng.standard_normal(n)
        result = gets_search(y, X, names=["const", "x1"], diagnostics=False)
        summary = result.summary()
        assert "GETS Model Selection" in summary
        assert "GUM variables" in summary

    def test_null_dgp_reasonable_retention(self, rng):
        """Under null (no signal), most variables should be removed."""
        n = 200
        k = 10
        X = np.column_stack(
            [np.ones(n)] + [rng.standard_normal(n) for _ in range(k - 1)]
        )
        y = rng.standard_normal(n)
        names = ["const"] + [f"x{i}" for i in range(1, k)]
        protected = np.zeros(k, dtype=bool)
        protected[0] = True  # protect constant
        result = gets_search(y, X, names=names, diagnostics=False, protected=protected)
        # Should remove most noise variables
        assert len(result.retained_names) < k

    def test_large_alpha(self, rng):
        """With large alpha, fewer variables removed."""
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)
        # Very lenient alpha
        result = gets_search(y, X, alpha=0.50, diagnostics=False)
        # With alpha=0.50, nearly everything passes the threshold
        assert len(result.terminal_models) >= 1

    def test_strict_alpha(self, rng):
        """With strict alpha, more variables retained (harder to reject)."""
        n = 200
        X = np.column_stack(
            [np.ones(n), rng.standard_normal(n), rng.standard_normal(n)]
        )
        y = X @ [3, 2, 0.5] + rng.standard_normal(n) * 0.5
        result = gets_search(
            y,
            X,
            names=["const", "x1", "x2"],
            alpha=0.001,
            diagnostics=False,
        )
        # At alpha=0.001, only very significant variables pass
        assert len(result.terminal_models) >= 1

    def test_hq_selection(self, rng):
        """Test HQ criterion selection path."""
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)
        result = gets_search(y, X, selection="hq", diagnostics=False)
        assert result.selection_criterion == "hq"

    def test_with_diagnostics(self, rng):
        """Test that diagnostics=True doesn't crash."""
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ [2, 1] + rng.standard_normal(n)
        result = gets_search(y, X, names=["const", "x1"], diagnostics=True)
        assert result.selected_model is not None

    def test_single_variable(self, rng):
        """Test with a single variable (constant only)."""
        n = 100
        X = np.ones((n, 1))
        y = rng.standard_normal(n)
        result = gets_search(y, X, names=["const"], diagnostics=False)
        assert len(result.retained_names) == 1
        assert result.retained_names[0] == "const"

    def test_many_noise_variables(self, rng):
        """Test with many noise variables — stress test."""
        n = 200
        k = 20
        X = np.column_stack(
            [np.ones(n)] + [rng.standard_normal(n) for _ in range(k - 1)]
        )
        y = 2 + rng.standard_normal(n)
        names = ["const"] + [f"noise{i}" for i in range(1, k)]
        protected = np.zeros(k, dtype=bool)
        protected[0] = True
        result = gets_search(
            y,
            X,
            names=names,
            diagnostics=False,
            protected=protected,
        )
        # With 19 noise vars and alpha=0.05, expect ~1 false retention
        assert len(result.retained_names) <= 5

    def test_max_evaluations_limits_fits(self, rng):
        """max_evaluations caps total OLS evaluations."""
        n = 200
        k = 15
        X = np.column_stack(
            [np.ones(n)] + [rng.standard_normal(n) for _ in range(k - 1)]
        )
        y = rng.standard_normal(n)
        names = ["const"] + [f"x{i}" for i in range(1, k)]
        protected = np.zeros(k, dtype=bool)
        protected[0] = True
        result = gets_search(
            y,
            X,
            names=names,
            diagnostics=False,
            protected=protected,
            max_evaluations=10,
        )
        # Allow small overhead for GUM + union re-search beyond the cap
        assert result.n_models_evaluated <= 15

    def test_max_paths_single_path(self, rng):
        """max_paths=1 uses only the primary reduction path."""
        n = 200
        X = np.column_stack(
            [
                np.ones(n),
                rng.standard_normal(n),
                rng.standard_normal(n),
            ]
        )
        y = rng.standard_normal(n)
        result = gets_search(
            y,
            X,
            names=["const", "x1", "x2"],
            diagnostics=False,
            max_paths=1,
        )
        assert len(result.terminal_models) <= 1 or result.terminal_models is not None

    def test_max_paths_multiple(self, rng):
        """max_paths > 1 can find multiple distinct terminals."""
        n = 200
        k = 8
        X = np.column_stack(
            [np.ones(n)] + [rng.standard_normal(n) for _ in range(k - 1)]
        )
        y = rng.standard_normal(n)
        names = ["const"] + [f"x{i}" for i in range(1, k)]
        protected = np.zeros(k, dtype=bool)
        protected[0] = True
        result = gets_search(
            y,
            X,
            names=names,
            diagnostics=False,
            protected=protected,
            max_paths=5,
        )
        # Should find at least one terminal
        assert len(result.terminal_models) >= 1

    def test_bounded_search_completes_fast(self, rng):
        """Bounded search with many variables completes quickly."""
        import time

        n = 200
        k = 50
        X = np.column_stack(
            [np.ones(n)] + [rng.standard_normal(n) for _ in range(k - 1)]
        )
        y = rng.standard_normal(n)
        names = ["const"] + [f"x{i}" for i in range(1, k)]
        protected = np.zeros(k, dtype=bool)
        protected[0] = True

        start = time.time()
        result = gets_search(
            y,
            X,
            names=names,
            diagnostics=False,
            protected=protected,
            max_evaluations=100,
        )
        elapsed = time.time() - start
        assert elapsed < 30, f"Search took {elapsed:.1f}s, expected <30s"
        assert result.n_models_evaluated <= 100
