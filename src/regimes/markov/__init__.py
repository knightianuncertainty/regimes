"""Markov regime-switching models for the regimes package.

This subpackage provides Markov switching regression, AR, and ADL models,
restricted transition matrices, sequential testing algorithms, and regime
number selection.
"""

from regimes.markov.models import MarkovADL, MarkovAR, MarkovRegression
from regimes.markov.restricted import RestrictedMarkovAR, RestrictedMarkovRegression
from regimes.markov.results import (
    MarkovADLResults,
    MarkovARResults,
    MarkovRegressionResults,
    MarkovSwitchingResultsBase,
)
from regimes.markov.selection import RegimeNumberSelection, RegimeNumberSelectionResults
from regimes.markov.sequential_restriction import (
    NonRecurringRegimeTest,
    NonRecurringRegimeTestResults,
    SequentialRestrictionResults,
    SequentialRestrictionTest,
)

__all__ = [
    "MarkovADL",
    "MarkovADLResults",
    "MarkovAR",
    "MarkovARResults",
    "MarkovRegression",
    "MarkovRegressionResults",
    "MarkovSwitchingResultsBase",
    "NonRecurringRegimeTest",
    "NonRecurringRegimeTestResults",
    "RegimeNumberSelection",
    "RegimeNumberSelectionResults",
    "RestrictedMarkovAR",
    "RestrictedMarkovRegression",
    "SequentialRestrictionResults",
    "SequentialRestrictionTest",
]
