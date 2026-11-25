"""GETS model selection and indicator saturation.

This subpackage implements the Autometrics algorithm (Doornik, 2009) for
automated general-to-specific model selection, including indicator saturation
for detecting structural breaks.
"""

from regimes.gets.indicators import (
    impulse_indicators,
    multiplicative_indicators,
    step_indicators,
    trend_indicators,
)
from regimes.gets.representation import (
    ParameterRegime,
    RegimeLevelsRepresentation,
    ShiftsRepresentation,
    levels_to_shifts,
    shifts_to_levels,
)
from regimes.gets.results import GETSResults, TerminalModel
from regimes.gets.saturation import SaturationResults, isat
from regimes.gets.selection import gets_search

__all__ = [
    "GETSResults",
    "ParameterRegime",
    "RegimeLevelsRepresentation",
    "SaturationResults",
    "ShiftsRepresentation",
    "TerminalModel",
    "gets_search",
    "impulse_indicators",
    "isat",
    "levels_to_shifts",
    "multiplicative_indicators",
    "shifts_to_levels",
    "step_indicators",
    "trend_indicators",
]
