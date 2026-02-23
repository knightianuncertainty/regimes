"""Tests for dual representation (shifts <-> regime levels)."""

from __future__ import annotations

import numpy as np

from regimes.gets.representation import (
    ParameterRegime,
    RegimeLevelsRepresentation,
    ShiftsRepresentation,
    levels_to_shifts,
    shifts_to_levels,
)

# ---------------------------------------------------------------------------
# shifts_to_levels
# ---------------------------------------------------------------------------


class TestShiftsToLevels:
    def test_no_breaks(self):
        """Single regime: level equals initial."""
        shifts = ShiftsRepresentation(
            break_dates=[],
            initial_levels={"const": 2.0},
            shifts={"const": {}},
            shift_se={"const": {}},
        )
        levels = shifts_to_levels(shifts, n=100)
        regimes = levels.param_regimes["const"]
        assert len(regimes) == 1
        assert regimes[0].level == 2.0
        assert regimes[0].start == 0
        assert regimes[0].end == 99

    def test_single_break(self):
        """Two regimes from one break."""
        shifts = ShiftsRepresentation(
            break_dates=[50],
            initial_levels={"const": 1.0},
            shifts={"const": {50: 2.0}},
            shift_se={"const": {-1: 0.1, 50: 0.2}},
        )
        levels = shifts_to_levels(shifts, n=100)
        regimes = levels.param_regimes["const"]
        assert len(regimes) == 2
        assert regimes[0].level == 1.0
        assert regimes[0].start == 0
        assert regimes[0].end == 49
        assert regimes[1].level == 3.0  # 1.0 + 2.0
        assert regimes[1].start == 50
        assert regimes[1].end == 99

    def test_two_breaks(self):
        """Three regimes from two breaks."""
        shifts = ShiftsRepresentation(
            break_dates=[50, 100],
            initial_levels={"const": 0.0},
            shifts={"const": {50: 1.0, 100: -0.5}},
            shift_se={"const": {-1: 0.1, 50: 0.2, 100: 0.15}},
        )
        levels = shifts_to_levels(shifts, n=200)
        regimes = levels.param_regimes["const"]
        assert len(regimes) == 3
        assert regimes[0].level == 0.0
        assert regimes[1].level == 1.0
        assert regimes[2].level == 0.5  # 0 + 1.0 - 0.5

    def test_multiple_parameters(self):
        """Two parameters with different break schedules."""
        shifts = ShiftsRepresentation(
            break_dates=[50, 100],
            initial_levels={"const": 1.0, "y.L1": 0.3},
            shifts={
                "const": {50: 2.0},
                "y.L1": {100: 0.5},
            },
            shift_se={
                "const": {-1: 0.1, 50: 0.2},
                "y.L1": {-1: 0.05, 100: 0.1},
            },
        )
        levels = shifts_to_levels(shifts, n=200)

        const_regimes = levels.param_regimes["const"]
        assert len(const_regimes) == 2
        assert const_regimes[0].level == 1.0
        assert const_regimes[1].level == 3.0

        ar_regimes = levels.param_regimes["y.L1"]
        assert len(ar_regimes) == 2
        assert ar_regimes[0].level == 0.3
        assert ar_regimes[1].level == 0.8

    def test_se_propagation_no_cov(self):
        """SE propagation without covariance matrix (diagonal approx)."""
        shifts = ShiftsRepresentation(
            break_dates=[50],
            initial_levels={"const": 1.0},
            shifts={"const": {50: 2.0}},
            shift_se={"const": {-1: 0.1, 50: 0.2}},
        )
        levels = shifts_to_levels(shifts, n=100)
        regimes = levels.param_regimes["const"]
        # First regime SE = base SE
        assert regimes[0].level_se == 0.1
        # Second regime SE = sqrt(0.1^2 + 0.2^2)
        expected_se = np.sqrt(0.01 + 0.04)
        np.testing.assert_almost_equal(regimes[1].level_se, expected_se)

    def test_se_propagation_with_cov(self):
        """SE propagation with full covariance matrix."""
        # Model: [const, step_50]
        cov = np.array([[0.01, 0.005], [0.005, 0.04]])
        shifts = ShiftsRepresentation(
            break_dates=[50],
            initial_levels={"const": 1.0},
            shifts={"const": {50: 2.0}},
            shift_se={"const": {-1: 0.1, 50: 0.2}},
        )
        levels = shifts_to_levels(
            shifts,
            n=100,
            cov_params=cov,
            indicator_names=["const", "step_50"],
        )
        regimes = levels.param_regimes["const"]
        # First regime: sqrt(cov[0,0]) = sqrt(0.01) = 0.1
        np.testing.assert_almost_equal(regimes[0].level_se, 0.1)
        # Second regime: L=[1,1], var = L @ cov @ L = 0.01 + 0.04 + 2*0.005 = 0.06
        expected_se = np.sqrt(0.06)
        np.testing.assert_almost_equal(regimes[1].level_se, expected_se)

    def test_parameter_with_no_shifts_gets_single_regime(self):
        """A parameter in initial_levels but not in shifts gets one regime."""
        shifts = ShiftsRepresentation(
            break_dates=[50],
            initial_levels={"const": 1.0, "x1": 0.5},
            shifts={"const": {50: 2.0}},
            shift_se={"const": {-1: 0.1, 50: 0.2}},
        )
        levels = shifts_to_levels(shifts, n=100)
        assert "x1" in levels.param_regimes
        assert len(levels.param_regimes["x1"]) == 1
        assert levels.param_regimes["x1"][0].level == 0.5


# ---------------------------------------------------------------------------
# levels_to_shifts
# ---------------------------------------------------------------------------


class TestLevelsToShifts:
    def test_single_regime(self):
        levels = RegimeLevelsRepresentation(
            param_regimes={"const": [ParameterRegime(0, 99, 2.0, 0.1)]}
        )
        shifts = levels_to_shifts(levels)
        assert shifts.initial_levels["const"] == 2.0
        assert shifts.shifts["const"] == {}
        assert shifts.break_dates == []

    def test_two_regimes(self):
        levels = RegimeLevelsRepresentation(
            param_regimes={
                "const": [
                    ParameterRegime(0, 49, 1.0, 0.1),
                    ParameterRegime(50, 99, 3.0, 0.15),
                ]
            }
        )
        shifts = levels_to_shifts(levels)
        assert shifts.initial_levels["const"] == 1.0
        assert shifts.shifts["const"] == {50: 2.0}
        assert shifts.break_dates == [50]

    def test_three_regimes(self):
        levels = RegimeLevelsRepresentation(
            param_regimes={
                "const": [
                    ParameterRegime(0, 49, 0.0, 0.1),
                    ParameterRegime(50, 99, 1.0, 0.12),
                    ParameterRegime(100, 199, 0.5, 0.15),
                ]
            }
        )
        shifts = levels_to_shifts(levels)
        assert shifts.initial_levels["const"] == 0.0
        assert shifts.shifts["const"] == {50: 1.0, 100: -0.5}

    def test_multiple_params(self):
        levels = RegimeLevelsRepresentation(
            param_regimes={
                "const": [
                    ParameterRegime(0, 49, 1.0, 0.1),
                    ParameterRegime(50, 199, 3.0, 0.15),
                ],
                "y.L1": [
                    ParameterRegime(0, 99, 0.3, 0.05),
                    ParameterRegime(100, 199, 0.8, 0.07),
                ],
            }
        )
        shifts = levels_to_shifts(levels)
        assert set(shifts.break_dates) == {50, 100}
        assert shifts.shifts["const"] == {50: 2.0}
        assert shifts.shifts["y.L1"] == {100: 0.5}


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_shifts_to_levels_to_shifts(self):
        """shifts -> levels -> shifts should preserve break dates and magnitudes."""
        original = ShiftsRepresentation(
            break_dates=[50, 100],
            initial_levels={"const": 1.0, "y.L1": 0.3},
            shifts={
                "const": {50: 2.0, 100: -0.5},
                "y.L1": {50: 0.1},
            },
            shift_se={
                "const": {-1: 0.1, 50: 0.2, 100: 0.15},
                "y.L1": {-1: 0.05, 50: 0.08},
            },
        )
        levels = shifts_to_levels(original, n=200)
        recovered = levels_to_shifts(levels)

        # Initial levels preserved
        np.testing.assert_almost_equal(
            recovered.initial_levels["const"], original.initial_levels["const"]
        )
        np.testing.assert_almost_equal(
            recovered.initial_levels["y.L1"], original.initial_levels["y.L1"]
        )

        # Shifts preserved
        for param in ["const", "y.L1"]:
            for tau, shift in original.shifts[param].items():
                np.testing.assert_almost_equal(recovered.shifts[param][tau], shift)

    def test_levels_to_shifts_to_levels(self):
        """levels -> shifts -> levels should preserve regime structure."""
        original = RegimeLevelsRepresentation(
            param_regimes={
                "const": [
                    ParameterRegime(0, 49, 1.0, 0.1),
                    ParameterRegime(50, 149, 3.0, 0.15),
                    ParameterRegime(150, 199, 2.5, 0.12),
                ],
            }
        )
        shifts = levels_to_shifts(original)
        recovered = shifts_to_levels(shifts, n=200)

        for i, regime in enumerate(recovered.param_regimes["const"]):
            orig_regime = original.param_regimes["const"][i]
            assert regime.start == orig_regime.start
            assert regime.end == orig_regime.end
            np.testing.assert_almost_equal(regime.level, orig_regime.level)

    def test_empty_round_trip(self):
        """No breaks round-trip."""
        original = ShiftsRepresentation(
            break_dates=[],
            initial_levels={"const": 5.0},
            shifts={"const": {}},
            shift_se={"const": {}},
        )
        levels = shifts_to_levels(original, n=100)
        recovered = levels_to_shifts(levels)
        assert recovered.initial_levels["const"] == 5.0
        assert recovered.shifts["const"] == {}
