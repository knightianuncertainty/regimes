"""Property-based tests for rolling/recursive estimation invariants.

These tests verify that mathematical properties of rolling and recursive
estimation hold for ANY valid input, using the Hypothesis library.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings

import regimes as rg

from .conftest import rolling_regression_data


class TestRollingShapeProperties:
    """Test shape alignment properties for rolling/recursive results."""

    @given(data=rolling_regression_data())
    @settings(max_examples=50, deadline=None)
    def test_params_shape(
        self,
        data: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        """params should have shape (nobs, n_params)."""
        y, X, window = data

        model = rg.OLS(y, X, has_constant=False)
        rolling_results = model.rolling(window=window).fit()

        assert rolling_results.params.shape == (len(y), X.shape[1]), (
            f"params shape {rolling_results.params.shape} != "
            f"expected ({len(y)}, {X.shape[1]})"
        )

    @given(data=rolling_regression_data())
    @settings(max_examples=50, deadline=None)
    def test_bse_shape_equals_params_shape(
        self,
        data: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        """bse shape must equal params shape.

        Standard errors have one value per parameter per observation.
        """
        y, X, window = data

        model = rg.OLS(y, X, has_constant=False)
        rolling_results = model.rolling(window=window).fit()

        assert rolling_results.bse.shape == rolling_results.params.shape, (
            f"bse shape {rolling_results.bse.shape} != "
            f"params shape {rolling_results.params.shape}"
        )

    @given(data=rolling_regression_data())
    @settings(max_examples=50, deadline=None)
    def test_tvalues_shape_equals_params_shape(
        self,
        data: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        """tvalues shape must equal params shape."""
        y, X, window = data

        model = rg.OLS(y, X, has_constant=False)
        rolling_results = model.rolling(window=window).fit()

        assert rolling_results.tvalues.shape == rolling_results.params.shape, (
            f"tvalues shape {rolling_results.tvalues.shape} != "
            f"params shape {rolling_results.params.shape}"
        )


class TestRollingNaNAlignmentProperties:
    """Test NaN alignment properties for rolling/recursive results."""

    @given(data=rolling_regression_data())
    @settings(max_examples=50, deadline=None)
    def test_nan_alignment_params_bse(
        self,
        data: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        """NaN in params iff NaN in bse at same position.

        If we can't estimate a parameter, we can't estimate its standard error.
        """
        y, X, window = data

        model = rg.OLS(y, X, has_constant=False)
        rolling_results = model.rolling(window=window).fit()

        params_nan = np.isnan(rolling_results.params)
        bse_nan = np.isnan(rolling_results.bse)

        np.testing.assert_array_equal(
            params_nan,
            bse_nan,
            err_msg="NaN positions in params don't match bse",
        )

    @given(data=rolling_regression_data())
    @settings(max_examples=50, deadline=None)
    def test_nan_alignment_all_params_same_row(
        self,
        data: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        """If one param is NaN in a row, all params in that row are NaN.

        Either we have a valid estimate (all params) or we don't (all NaN).
        """
        y, X, window = data

        model = rg.OLS(y, X, has_constant=False)
        rolling_results = model.rolling(window=window).fit()

        # For each row, either all NaN or all valid
        for i in range(len(y)):
            row_nans = np.isnan(rolling_results.params[i])
            assert np.all(row_nans) or np.all(~row_nans), (
                f"Row {i} has mixed NaN/valid values: {rolling_results.params[i]}"
            )


class TestRollingValidCountProperties:
    """Test valid observation count properties."""

    @given(data=rolling_regression_data())
    @settings(max_examples=50, deadline=None)
    def test_n_valid_count(
        self,
        data: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        """n_valid should equal count of non-NaN rows."""
        y, X, window = data

        model = rg.OLS(y, X, has_constant=False)
        rolling_results = model.rolling(window=window).fit()

        # Count rows with valid (non-NaN) estimates
        valid_rows = np.sum(~np.isnan(rolling_results.params[:, 0]))

        assert rolling_results.n_valid == valid_rows, (
            f"n_valid={rolling_results.n_valid} != valid row count={valid_rows}"
        )

    @given(data=rolling_regression_data())
    @settings(max_examples=50, deadline=None)
    def test_n_valid_relationship_to_window(
        self,
        data: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        """For rolling: n_valid = nobs - window + 1."""
        y, X, window = data

        model = rg.OLS(y, X, has_constant=False)
        rolling_results = model.rolling(window=window).fit()

        expected_n_valid = len(y) - window + 1

        assert rolling_results.n_valid == expected_n_valid, (
            f"n_valid={rolling_results.n_valid} != expected={expected_n_valid} "
            f"(nobs={len(y)}, window={window})"
        )


class TestRecursiveMonotonicityProperties:
    """Test that recursive estimation becomes valid and stays valid."""

    @given(data=rolling_regression_data())
    @settings(max_examples=50, deadline=None)
    def test_recursive_once_valid_always_valid(
        self,
        data: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        """Once recursive estimation becomes valid, it should stay valid.

        Recursive estimation uses expanding windows, so once we have enough
        observations to estimate, we always will have enough.
        """
        y, X, _ = data

        model = rg.OLS(y, X, has_constant=False)
        k = X.shape[1]
        min_nobs = k + 5

        # Only run if we have enough observations
        if len(y) <= min_nobs:
            return

        recursive_results = model.recursive(min_nobs=min_nobs).fit()

        # Find first valid row
        first_valid = None
        for i in range(len(y)):
            if not np.isnan(recursive_results.params[i, 0]):
                first_valid = i
                break

        if first_valid is not None:
            # All subsequent rows should be valid
            remaining_params = recursive_results.params[first_valid:]
            assert not np.any(np.isnan(remaining_params)), (
                "Found NaN in recursive params after first valid estimate"
            )

    @given(data=rolling_regression_data())
    @settings(max_examples=50, deadline=None)
    def test_recursive_first_valid_at_min_nobs(
        self,
        data: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        """First valid recursive estimate should be at index min_nobs - 1.

        At index i, we have i+1 observations. So the first valid estimate
        is at index min_nobs - 1.
        """
        y, X, _ = data

        model = rg.OLS(y, X, has_constant=False)
        k = X.shape[1]
        min_nobs = k + 5

        # Only run if we have enough observations
        if len(y) <= min_nobs:
            return

        recursive_results = model.recursive(min_nobs=min_nobs).fit()

        # All rows before min_nobs - 1 should be NaN
        if min_nobs > 1:
            pre_valid = recursive_results.params[: min_nobs - 1]
            assert np.all(np.isnan(pre_valid)), (
                f"Expected NaN before index {min_nobs - 1}, got valid values"
            )

        # Row at min_nobs - 1 should be valid
        assert not np.isnan(recursive_results.params[min_nobs - 1, 0]), (
            f"Expected valid at index {min_nobs - 1}, got NaN"
        )


class TestRollingConfidenceIntervalProperties:
    """Test confidence interval properties for rolling/recursive."""

    @given(data=rolling_regression_data())
    @settings(max_examples=50, deadline=None)
    def test_ci_contains_estimate(
        self,
        data: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        """Confidence interval must contain point estimate where valid."""
        y, X, window = data

        model = rg.OLS(y, X, has_constant=False)
        rolling_results = model.rolling(window=window).fit()

        ci = rolling_results.conf_int(alpha=0.05)

        # Check only valid rows
        for i in range(len(y)):
            if not np.isnan(rolling_results.params[i, 0]):
                for j in range(X.shape[1]):
                    lower = ci[i, j, 0]
                    upper = ci[i, j, 1]
                    estimate = rolling_results.params[i, j]

                    assert lower < estimate < upper, (
                        f"CI [{lower}, {upper}] doesn't contain estimate {estimate} "
                        f"at row {i}, param {j}"
                    )

    @given(data=rolling_regression_data())
    @settings(max_examples=50, deadline=None)
    def test_ci_shape(
        self,
        data: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        """CI should have shape (nobs, n_params, 2)."""
        y, X, window = data

        model = rg.OLS(y, X, has_constant=False)
        rolling_results = model.rolling(window=window).fit()

        ci = rolling_results.conf_int()
        expected_shape = (len(y), X.shape[1], 2)

        assert ci.shape == expected_shape, (
            f"CI shape {ci.shape} != expected {expected_shape}"
        )


class TestRollingDataFrameConversionProperties:
    """Test DataFrame conversion properties."""

    @given(data=rolling_regression_data())
    @settings(max_examples=50, deadline=None)
    def test_to_dataframe_shape(
        self,
        data: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        """to_dataframe should have nobs rows and n_params columns."""
        y, X, window = data

        model = rg.OLS(y, X, has_constant=False)
        rolling_results = model.rolling(window=window).fit()

        df = rolling_results.to_dataframe()

        assert len(df) == len(y), f"DataFrame rows {len(df)} != nobs {len(y)}"
        assert len(df.columns) == X.shape[1], (
            f"DataFrame columns {len(df.columns)} != n_params {X.shape[1]}"
        )

    @given(data=rolling_regression_data())
    @settings(max_examples=50, deadline=None)
    def test_to_dataframe_full_has_se_columns(
        self,
        data: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        """to_dataframe_full should have both param and SE columns."""
        y, X, window = data

        model = rg.OLS(y, X, has_constant=False)
        rolling_results = model.rolling(window=window).fit()

        df_full = rolling_results.to_dataframe_full()

        # Should have 2 columns per parameter (param + SE)
        expected_cols = X.shape[1] * 2
        assert len(df_full.columns) == expected_cols, (
            f"to_dataframe_full columns {len(df_full.columns)} != expected {expected_cols}"
        )

        # Check that SE columns exist
        for name in rolling_results.param_names:
            assert f"{name}_se" in df_full.columns, f"Missing SE column for {name}"
