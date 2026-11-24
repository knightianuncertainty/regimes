regimes documentation
=========================

**regimes** is a Python package for structural break detection and estimation
in time-series econometrics, extending statsmodels with robust methods for
analyzing regime changes.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   conventions
   validation
   api

Features
--------

- **Structural break tests**: the Bai-Perron procedure for multiple breaks,
  the Chow test, CUSUM and CUSUM-of-squares, and the Andrews-Ploberger tests
- **Models with known breaks**: OLS, AR and ADL with regime-specific
  coefficients and robust standard errors
- **Rolling and recursive estimation** of the same models
- **Markov switching**: regressions, autoregressions and ADL models,
  restricted (non-recurring) transitions, tests of the restrictions and
  selection of the number of regimes
- **Visualization**: break dates, regime means, coefficient paths,
  diagnostics, smoothed probabilities

What a break date is, how the statistics are scaled and which estimators
are used is set out in :doc:`conventions`. What each method was checked
against, with the numbers, is in :doc:`validation`.

Installation
------------

The package is not on PyPI yet. Install it from GitHub:

.. code-block:: bash

   pip install git+https://github.com/knightianuncertainty/regimes.git

Quick Example
-------------

.. code-block:: python

   import numpy as np
   import regimes as rg

   # Simulate data with a mean shift
   y = np.concatenate([np.random.randn(100), np.random.randn(100) + 2])

   # Test for breaks
   test = rg.BaiPerronTest(y)
   results = test.fit(max_breaks=3)
   print(f"Detected {results.n_breaks} break(s) at: {results.break_indices}")


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
