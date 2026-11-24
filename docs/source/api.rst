API Reference
=============

Every public class and function of the package. The conventions that the
docstrings rely on (break dates, scaling of the statistics, covariance
estimators) are collected in :doc:`conventions`; what each method was checked
against is in :doc:`validation`.

Regression models
-----------------

OLS
~~~

.. autoclass:: regimes.OLS
   :members:
   :show-inheritance:

.. autoclass:: regimes.OLSResults
   :members:
   :show-inheritance:

.. autofunction:: regimes.summary_by_regime

AR
~~

.. autoclass:: regimes.AR
   :members:
   :show-inheritance:

.. autoclass:: regimes.ARResults
   :members:
   :show-inheritance:

.. autofunction:: regimes.ar_summary_by_regime

ADL
~~~

.. autoclass:: regimes.ADL
   :members:
   :show-inheritance:

.. autoclass:: regimes.ADLResults
   :members:
   :show-inheritance:

.. autofunction:: regimes.adl_summary_by_regime

Rolling and recursive estimation
--------------------------------

.. autoclass:: regimes.RollingOLS
   :members:
   :show-inheritance:

.. autoclass:: regimes.RecursiveOLS
   :members:
   :show-inheritance:

.. autoclass:: regimes.RollingAR
   :members:
   :show-inheritance:

.. autoclass:: regimes.RecursiveAR
   :members:
   :show-inheritance:

.. autoclass:: regimes.RollingADL
   :members:
   :show-inheritance:

.. autoclass:: regimes.RecursiveADL
   :members:
   :show-inheritance:

.. autoclass:: regimes.RollingOLSResults
   :members:
   :show-inheritance:

.. autoclass:: regimes.RecursiveOLSResults
   :members:
   :show-inheritance:

.. autoclass:: regimes.RollingARResults
   :members:
   :show-inheritance:

.. autoclass:: regimes.RecursiveARResults
   :members:
   :show-inheritance:

.. autoclass:: regimes.RollingADLResults
   :members:
   :show-inheritance:

.. autoclass:: regimes.RecursiveADLResults
   :members:
   :show-inheritance:

Structural break tests
----------------------

Bai-Perron
~~~~~~~~~~

.. autoclass:: regimes.BaiPerronTest
   :members:
   :show-inheritance:

.. autoclass:: regimes.BaiPerronResults
   :members:
   :show-inheritance:

Chow
~~~~

.. autoclass:: regimes.ChowTest
   :members:
   :show-inheritance:

.. autoclass:: regimes.ChowTestResults
   :members:
   :show-inheritance:

CUSUM and CUSUM of squares
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: regimes.tests.cusum
   :no-members:

.. autoclass:: regimes.CUSUMTest
   :members:
   :show-inheritance:

.. autoclass:: regimes.CUSUMResults
   :members:
   :show-inheritance:

.. autoclass:: regimes.CUSUMSQTest
   :members:
   :show-inheritance:

.. autoclass:: regimes.CUSUMSQResults
   :members:
   :show-inheritance:

Andrews-Ploberger
~~~~~~~~~~~~~~~~~

.. automodule:: regimes.tests.andrews_ploberger
   :no-members:

.. autoclass:: regimes.AndrewsPlobergerTest
   :members:
   :show-inheritance:

.. autoclass:: regimes.AndrewsPlobergerResults
   :members:
   :show-inheritance:

Markov switching
----------------

Models
~~~~~~

.. autoclass:: regimes.MarkovRegression
   :members:
   :show-inheritance:

.. autoclass:: regimes.MarkovAR
   :members:
   :show-inheritance:

.. autoclass:: regimes.MarkovADL
   :members:
   :show-inheritance:

.. autoclass:: regimes.RestrictedMarkovRegression
   :members:
   :show-inheritance:

.. autoclass:: regimes.RestrictedMarkovAR
   :members:
   :show-inheritance:

Results
~~~~~~~

.. autoclass:: regimes.MarkovSwitchingResultsBase
   :members:
   :show-inheritance:

.. autoclass:: regimes.MarkovRegressionResults
   :members:
   :show-inheritance:

.. autoclass:: regimes.MarkovARResults
   :members:
   :show-inheritance:

.. autoclass:: regimes.MarkovADLResults
   :members:
   :show-inheritance:

Number of regimes
~~~~~~~~~~~~~~~~~

.. automodule:: regimes.markov.selection
   :no-members:

.. autoclass:: regimes.RegimeNumberSelection
   :members:
   :show-inheritance:

.. autoclass:: regimes.RegimeNumberSelectionResults
   :members:
   :show-inheritance:

Restrictions on the transition matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: regimes.markov.sequential_restriction
   :no-members:

.. autoclass:: regimes.NonRecurringRegimeTest
   :members:
   :show-inheritance:

.. autoclass:: regimes.NonRecurringRegimeTestResults
   :members:
   :show-inheritance:

.. autoclass:: regimes.SequentialRestrictionTest
   :members:
   :show-inheritance:

.. autoclass:: regimes.SequentialRestrictionResults
   :members:
   :show-inheritance:

Misspecification tests
----------------------

.. automodule:: regimes.diagnostics.misspec
   :no-members:

.. autoclass:: regimes.DiagnosticsResults
   :members:
   :show-inheritance:

.. autoclass:: regimes.DiagnosticTestResult
   :members:
   :show-inheritance:

Visualization
-------------

Breaks and regimes
~~~~~~~~~~~~~~~~~~

.. autofunction:: regimes.plot_breaks

.. autofunction:: regimes.plot_regime_means

.. autofunction:: regimes.plot_break_confidence

.. autofunction:: regimes.plot_params_over_time

.. autofunction:: regimes.plot_parameter_time_series

Tests
~~~~~

.. autofunction:: regimes.plot_cusum

.. autofunction:: regimes.plot_cusum_sq

.. autofunction:: regimes.plot_f_sequence

Rolling estimates
~~~~~~~~~~~~~~~~~

.. autofunction:: regimes.plot_rolling_coefficients

Residual diagnostics
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: regimes.plot_diagnostics

.. autofunction:: regimes.plot_actual_fitted

.. autofunction:: regimes.plot_scaled_residuals

.. autofunction:: regimes.plot_residual_distribution

.. autofunction:: regimes.plot_residual_acf

Markov switching
~~~~~~~~~~~~~~~~

.. autofunction:: regimes.plot_smoothed_probabilities

.. autofunction:: regimes.plot_regime_shading

.. autofunction:: regimes.plot_transition_matrix

.. autofunction:: regimes.plot_ic

Base classes
------------

.. autoclass:: regimes.RegimesModelBase
   :members:
   :show-inheritance:

.. autoclass:: regimes.TimeSeriesModelBase
   :members:
   :show-inheritance:

.. autoclass:: regimes.RegimesResultsBase
   :members:
   :show-inheritance:

.. autoclass:: regimes.RegressionResultsBase
   :members:
   :show-inheritance:

.. autoclass:: regimes.BreakResultsBase
   :members:
   :show-inheritance:

.. autoclass:: regimes.RollingEstimatorBase
   :members:
   :show-inheritance:

.. autoclass:: regimes.RollingResultsBase
   :members:
   :show-inheritance:
