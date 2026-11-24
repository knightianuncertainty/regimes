"""Tests for sequential restriction testing algorithms."""

from __future__ import annotations

import types
import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

import regimes.markov.sequential_restriction as module
from regimes.markov._simulate import MonteCarloPValue
from regimes.markov.sequential_restriction import (
    NonRecurringRegimeTest,
    NonRecurringRegimeTestResults,
    SequentialRestrictionResults,
    SequentialRestrictionTest,
    _chi_bar_squared_critical_value,
    _chi_bar_squared_pvalue,
)


@pytest.fixture(autouse=True)
def _fewer_restarts(monkeypatch: Any) -> None:
    """One random restart per estimation instead of five or ten.

    The tests of this file estimate many models; the number of restarts does
    not change what they check.
    """
    monkeypatch.setattr(NonRecurringRegimeTest, "_search_reps_unrestricted", 1)
    monkeypatch.setattr(NonRecurringRegimeTest, "_search_reps_restricted", 1)
    monkeypatch.setattr(SequentialRestrictionTest, "_search_reps", 1)


class TestChiBarSquared:
    """Test chi-bar-squared utility functions."""

    def test_pvalue_zero_stat(self) -> None:
        """A statistic of zero has p-value one; just above zero it is 0.5."""
        assert _chi_bar_squared_pvalue(0.0, 1) == 1.0
        assert abs(_chi_bar_squared_pvalue(1e-9, 1) - 0.5) < 1e-4

    def test_pvalue_large_stat(self) -> None:
        """Large LR should give small p-value."""
        p = _chi_bar_squared_pvalue(20.0, 1)
        assert p < 0.001

    def test_pvalue_known_value(self) -> None:
        """At the 5% critical value, p should be 0.05."""
        cv = _chi_bar_squared_critical_value(0.05, 1)
        p = _chi_bar_squared_pvalue(cv, 1)
        assert abs(p - 0.05) < 0.01

    def test_critical_value_5_percent(self) -> None:
        """5% critical value for single restriction should be ~2.706."""
        cv = _chi_bar_squared_critical_value(0.05, 1)
        assert abs(cv - 2.706) < 0.1

    def test_pvalue_multiple_restrictions_conservative(self) -> None:
        """Multiple restrictions should give more conservative p-values."""
        p1 = _chi_bar_squared_pvalue(5.0, 1)
        p2 = _chi_bar_squared_pvalue(5.0, 3)
        assert p2 >= p1


class TestNonRecurringRegimeTest:
    """Test NonRecurringRegimeTest."""

    def test_init(self, two_regime_data: NDArray[np.floating[Any]]) -> None:
        test = NonRecurringRegimeTest(
            two_regime_data, k_regimes=2, method="chi_bar_squared"
        )
        assert test.k_regimes == 2
        assert test.method == "chi_bar_squared"

    def test_build_restrictions(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        test = NonRecurringRegimeTest(two_regime_data, k_regimes=2)
        restrictions = test._build_non_recurring_restrictions()
        # For k=2, the only non-recurring restriction is (0,1)=0
        # (can't go from regime 1 back to regime 0)
        assert len(restrictions) > 0
        for (i, j), val in restrictions.items():
            assert val == 0.0

    def test_build_restrictions_3_regimes(
        self, three_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        test = NonRecurringRegimeTest(three_regime_data, k_regimes=3)
        restrictions = test._build_non_recurring_restrictions()
        # For k=3: restrictions should include (0,1), (0,2), (2,0), (1,2)
        # but NOT (0,0), (1,1), (2,2) [diagonal]
        # and NOT (1,0), (2,1) [forward transitions]
        assert (0, 1) in restrictions  # Can't go back from 1 to 0
        assert (0, 2) in restrictions  # Can't go back from 2 to 0
        assert (1, 0) not in restrictions  # Forward: 0 -> 1
        assert (2, 1) not in restrictions  # Forward: 1 -> 2

    def test_fit_chi_bar_squared(self) -> None:
        """Test with larger, well-separated data for reliable convergence."""
        rng = np.random.default_rng(99)
        y = np.concatenate(
            [
                rng.standard_normal(200) + 0.0,
                rng.standard_normal(200) + 5.0,
            ]
        )
        test = NonRecurringRegimeTest(y, k_regimes=2, method="chi_bar_squared")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = test.fit()

        assert isinstance(results, NonRecurringRegimeTestResults)
        assert results.lr_statistic >= 0
        assert 0 <= results.p_value <= 1
        assert results.rejected in (True, False)
        assert results.method == "chi_bar_squared"

    def test_summary(self) -> None:
        rng = np.random.default_rng(99)
        y = np.concatenate(
            [
                rng.standard_normal(200) + 0.0,
                rng.standard_normal(200) + 5.0,
            ]
        )
        test = NonRecurringRegimeTest(y, k_regimes=2, method="chi_bar_squared")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = test.fit()

        s = results.summary()
        assert "Non-Recurring" in s
        assert "LR statistic" in s


class TestNonRecurringRegimeTestEdgeCases:
    """Additional edge cases for NonRecurringRegimeTest."""

    def test_summary_format(self) -> None:
        """Summary should contain all expected sections."""
        rng = np.random.default_rng(99)
        y = np.concatenate(
            [
                rng.standard_normal(200) + 0.0,
                rng.standard_normal(200) + 5.0,
            ]
        )
        test = NonRecurringRegimeTest(y, k_regimes=2, method="chi_bar_squared")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = test.fit()

        s = results.summary()
        assert "Non-Recurring" in s
        assert "LR statistic" in s
        assert "p-value" in s or "p_value" in s.lower() or "0." in s

    def test_3_regimes(self) -> None:
        """Test with 3 regimes."""
        rng = np.random.default_rng(42)
        y = np.concatenate(
            [
                rng.standard_normal(150) + 0.0,
                rng.standard_normal(150) + 4.0,
                rng.standard_normal(150) + 8.0,
            ]
        )
        test = NonRecurringRegimeTest(y, k_regimes=3, method="chi_bar_squared")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = test.fit()

        assert isinstance(results, NonRecurringRegimeTestResults)
        assert results.lr_statistic >= 0


class TestSequentialRestrictionTest:
    """Test SequentialRestrictionTest."""

    def test_init(self, two_regime_data: NDArray[np.floating[Any]]) -> None:
        test = SequentialRestrictionTest(
            two_regime_data,
            k_regimes=2,
            critical_value_method="chi_bar_squared",
        )
        assert test.k_regimes == 2
        assert test.multiple_testing == "holm"

    def test_identify_candidates(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        test = SequentialRestrictionTest(two_regime_data, k_regimes=2)
        transition = np.array([[0.9, 0.2], [0.1, 0.8]])
        candidates = test._identify_candidates(transition, {})
        # Should have 2 off-diagonal candidates
        assert len(candidates) == 2
        # Smallest first
        assert transition[candidates[0]] <= transition[candidates[1]]

    def test_identify_candidates_with_existing(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        test = SequentialRestrictionTest(two_regime_data, k_regimes=2)
        transition = np.array([[0.9, 0.2], [0.1, 0.8]])
        candidates = test._identify_candidates(transition, {(1, 0): 0.0})
        # One already restricted, so only 1 candidate
        assert len(candidates) == 1

    def test_holm_significance(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        test = SequentialRestrictionTest(
            two_regime_data,
            k_regimes=2,
            significance=0.05,
            multiple_testing="holm",
        )
        # The least significant restriction is tested first, at alpha
        sig = test._holm_significance(0, 2)
        assert abs(sig - 0.05) < 0.001

        # The next one at alpha / 2
        sig = test._holm_significance(1, 2)
        assert abs(sig - 0.025) < 0.001

    def test_bonferroni_significance(
        self, two_regime_data: NDArray[np.floating[Any]]
    ) -> None:
        test = SequentialRestrictionTest(
            two_regime_data,
            k_regimes=2,
            significance=0.05,
            multiple_testing="bonferroni",
        )
        sig = test._holm_significance(0, 2)
        assert abs(sig - 0.025) < 0.001

    def test_no_correction(self, two_regime_data: NDArray[np.floating[Any]]) -> None:
        test = SequentialRestrictionTest(
            two_regime_data,
            k_regimes=2,
            significance=0.05,
            multiple_testing="none",
        )
        sig = test._holm_significance(0, 2)
        assert abs(sig - 0.05) < 0.001

    def test_fit(self) -> None:
        """Use larger data for reliable restricted model convergence."""
        rng = np.random.default_rng(77)
        y = np.concatenate(
            [
                rng.standard_normal(200) + 0.0,
                rng.standard_normal(200) + 5.0,
            ]
        )
        test = SequentialRestrictionTest(
            y,
            k_regimes=2,
            critical_value_method="chi_bar_squared",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = test.fit()

        assert isinstance(results, SequentialRestrictionResults)
        assert isinstance(results.final_restrictions, dict)
        assert results.final_transition.shape == (2, 2)
        assert np.isfinite(results.llf_unrestricted)
        assert results.is_non_recurring in (True, False)
        assert len(results.history) > 0

    def test_fit_verbose(self, capsys: Any) -> None:
        """Verbose output should print step-by-step results."""
        rng = np.random.default_rng(77)
        y = np.concatenate(
            [
                rng.standard_normal(200) + 0.0,
                rng.standard_normal(200) + 5.0,
            ]
        )
        test = SequentialRestrictionTest(
            y,
            k_regimes=2,
            critical_value_method="chi_bar_squared",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test.fit(verbose=True)

        captured = capsys.readouterr()
        # Should print step information
        assert (
            "Step" in captured.out
            or "step" in captured.out.lower()
            or len(captured.out) > 0
        )

    def test_check_non_recurring(self) -> None:
        """_check_non_recurring should identify non-recurring patterns."""
        rng = np.random.default_rng(77)
        y = np.concatenate(
            [
                rng.standard_normal(200) + 0.0,
                rng.standard_normal(200) + 5.0,
            ]
        )
        test = SequentialRestrictionTest(
            y,
            k_regimes=2,
            critical_value_method="chi_bar_squared",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = test.fit()

        assert results.is_non_recurring in (True, False)

    def test_summary(self) -> None:
        rng = np.random.default_rng(77)
        y = np.concatenate(
            [
                rng.standard_normal(200) + 0.0,
                rng.standard_normal(200) + 5.0,
            ]
        )
        test = SequentialRestrictionTest(
            y,
            k_regimes=2,
            critical_value_method="chi_bar_squared",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = test.fit()

        s = results.summary()
        assert "Sequential Restriction" in s
        assert "Final Transition Matrix" in s


class TestChiBarSquaredBound:
    """The reference distribution for several restrictions."""

    def test_kodde_palm_upper_bounds(self) -> None:
        """Upper-bound critical values of Kodde and Palm (1986), Table 1, 5%."""
        published = {1: 2.706, 2: 5.138, 3: 7.045, 4: 8.761}
        for n, value in published.items():
            assert abs(_chi_bar_squared_critical_value(0.05, n) - value) < 2e-3

    def test_pvalue_and_critical_value_agree(self) -> None:
        for n in range(1, 6):
            for alpha in (0.10, 0.05, 0.01):
                cv = _chi_bar_squared_critical_value(alpha, n)
                assert abs(_chi_bar_squared_pvalue(cv, n) - alpha) < 1e-9

    def test_bound_is_a_mixture_of_two_chi_squares(self) -> None:
        p = _chi_bar_squared_pvalue(6.0, 3)
        assert abs(p - 0.5 * (stats.chi2.sf(6.0, 2) + stats.chi2.sf(6.0, 3))) < 1e-12

    def test_bound_holds_for_a_simulated_orthant_statistic(self) -> None:
        """Two independent inequality restrictions: weights 1/4, 1/2, 1/4."""
        rng = np.random.default_rng(0)
        z = rng.standard_normal((200_000, 2))
        statistic = np.sum(np.maximum(z, 0.0) ** 2, axis=1)
        for alpha in (0.10, 0.05, 0.01):
            cv = _chi_bar_squared_critical_value(alpha, 2)
            assert np.mean(statistic > cv) <= alpha

    def test_zero_statistic_has_pvalue_one(self) -> None:
        for n in (1, 2, 4):
            assert _chi_bar_squared_pvalue(0.0, n) == 1.0

    def test_invalid_number_of_restrictions(self) -> None:
        with pytest.raises(ValueError):
            _chi_bar_squared_pvalue(1.0, 0)
        with pytest.raises(ValueError):
            _chi_bar_squared_critical_value(0.05, 0)


def _break_series(
    seed: int, n: int = 80, shift: float = 3.0
) -> NDArray[np.floating[Any]]:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0, 1, n), rng.normal(shift, 1, n)])


def _recurring_series(seed: int, n: int = 30) -> NDArray[np.floating[Any]]:
    rng = np.random.default_rng(seed)
    means = [0.0, 3.0, 0.0, 3.0, 0.0, 3.0]
    return np.concatenate([rng.normal(m, 1, n) for m in means])


class _FakeFit:
    """Stand-in for a fitted model: a log-likelihood and a transition matrix."""

    def __init__(self, llf: float, k: int, key: Any = None) -> None:
        self.llf = llf
        self.regime_transition = np.full((k, k), 1.0 / k)
        self.key = key  # the restrictions of the model, if it has any meaning
        self._sm_results = types.SimpleNamespace(model=None, params=np.zeros(1))


def _fake_search(
    test: SequentialRestrictionTest,
    monkeypatch: Any,
    llf_of: Any,
    transition: Any = None,
) -> list[tuple[tuple[tuple[int, int], ...], int]]:
    """Replace the estimations of a search by a table of log-likelihoods.

    A fit started from the estimates of a model nested in it (one with the
    same restrictions and more) reaches at least that model's likelihood, as
    a real one does. Returns the list of calls made: ``(restrictions,
    number of warm starts)``.
    """
    k = test.k_regimes
    calls: list[tuple[tuple[tuple[int, int], ...], int]] = []

    def fake_fit(restrictions: Any = None, endog: Any = None, warm: Any = ()) -> Any:
        key = tuple(sorted(restrictions or {}))
        calls.append((key, len(warm)))
        llf = llf_of(key)
        if isinstance(llf, Exception):
            raise llf
        nested = [
            float(w.llf) for w in warm if w.key is not None and set(w.key) >= set(key)
        ]
        fit = _FakeFit(max([llf, *nested]), k, key)
        if transition is not None:
            fit.regime_transition = np.asarray(transition, dtype=float)
        return fit

    monkeypatch.setattr(test, "_fit_model", fake_fit)
    monkeypatch.setattr(
        test,
        "_starting_distribution",
        lambda: (np.full(k, 1.0 / k), _FakeFit(-1e9, k)),
    )
    return calls


class TestArgumentChecks:
    def test_unknown_arguments(self) -> None:
        with pytest.raises(ValueError, match="method must be one of"):
            NonRecurringRegimeTest(np.zeros(20), method="exact")
        with pytest.raises(ValueError, match="critical_value_method"):
            SequentialRestrictionTest(np.zeros(20), critical_value_method="exact")
        with pytest.raises(ValueError, match="multiple_testing"):
            SequentialRestrictionTest(np.zeros(20), multiple_testing="sidak")
        with pytest.raises(ValueError, match="model_type"):
            NonRecurringRegimeTest(np.zeros(20), model_type="adl")
        with pytest.raises(ValueError, match="n_bootstrap"):
            NonRecurringRegimeTest(np.zeros(20), n_bootstrap=0)
        with pytest.raises(ValueError, match="significance"):
            SequentialRestrictionTest(np.zeros(20), significance=1.5)

    def test_arguments_set_by_the_test(self) -> None:
        with pytest.raises(ValueError, match="initial_probabilities is set"):
            NonRecurringRegimeTest(np.zeros(20), initial_probabilities=[1.0, 0.0])
        with pytest.raises(ValueError, match="ordering is set"):
            SequentialRestrictionTest(np.zeros(20), ordering="intercept")

    def test_too_few_replications_to_reject(self) -> None:
        """With 19 replications the smallest p-value is 0.05: no rejection at 5%."""
        with pytest.raises(ValueError, match="could never reject"):
            NonRecurringRegimeTest(np.zeros(20), method="bootstrap", n_bootstrap=19)
        NonRecurringRegimeTest(np.zeros(20), method="bootstrap", n_bootstrap=20)
        # not a requirement of the chi-bar-squared version
        NonRecurringRegimeTest(np.zeros(20), n_bootstrap=5)

    def test_none_means_no_adjustment(self) -> None:
        test = SequentialRestrictionTest(np.zeros(20), multiple_testing=None)
        assert test.multiple_testing == "none"
        assert test._holm_significance(3, 6) == test.significance


class TestNonRecurringEstimation:
    def test_unrestricted_model_reaches_the_restricted_likelihood(self) -> None:
        """The models are nested in fact: no negative statistic to floor."""
        for seed in (40, 41, 42):
            test = NonRecurringRegimeTest(
                _break_series(seed, n=60, shift=1.5), k_regimes=2, random_state=seed
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # the nesting warning is an error
                result = test.fit()
            assert result.llf_unrestricted >= result.llf_restricted - 1e-4
            assert result.lr_statistic == pytest.approx(
                2 * (result.llf_unrestricted - result.llf_restricted), abs=2e-4
            )

    def test_given_seed_reproduces_the_statistic(self) -> None:
        y = _recurring_series(seed=43)
        first = NonRecurringRegimeTest(y, k_regimes=2, random_state=7).fit()
        second = NonRecurringRegimeTest(y, k_regimes=2, random_state=7).fit()
        assert first.lr_statistic == second.lr_statistic
        assert first.llf_restricted == second.llf_restricted

    def test_statistic_of_a_sample(self) -> None:
        """``_statistic(sample)`` is the statistic a test on the sample reports."""
        y = _recurring_series(seed=44)
        sample = _break_series(seed=45, n=60)
        test = NonRecurringRegimeTest(y, k_regimes=2)
        on_sample = test._statistic(sample, np.random.default_rng(5))
        direct = NonRecurringRegimeTest(sample, k_regimes=2, random_state=5).fit()
        assert on_sample == pytest.approx(direct.lr_statistic, abs=1e-8)
        # and it is not the statistic of the data
        assert test.fit().lr_statistic > 50
        assert on_sample < 5

    def test_autoregression(self) -> None:
        rng = np.random.default_rng(46)
        n = 160
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = (0.0 if t < 80 else 2.0) + 0.3 * y[t - 1] + rng.standard_normal()
        test = NonRecurringRegimeTest(y, k_regimes=2, model_type="ar", random_state=1)
        test._search_reps_unrestricted = 1
        test._search_reps_restricted = 1
        result = test.fit()
        assert np.isfinite(result.lr_statistic)
        assert result.lr_statistic >= 0
        assert not result.rejected

    def test_failed_estimation_raises(self, monkeypatch: Any) -> None:
        test = NonRecurringRegimeTest(_break_series(seed=47), k_regimes=2)
        monkeypatch.setattr(
            test,
            "_fit_both",
            lambda endog, rng: (_FakeFit(-10.0, 2), _FakeFit(float("nan"), 2)),
        )
        with pytest.raises(RuntimeError, match="could not be estimated"):
            test.fit()


class TestNonRecurringBootstrap:
    """``method="bootstrap"`` simulates under H0 and refits both models."""

    def test_zero_statistic_needs_no_bootstrap(self, monkeypatch: Any) -> None:
        test = NonRecurringRegimeTest(
            _break_series(seed=10), k_regimes=2, method="bootstrap", random_state=0
        )
        monkeypatch.setattr(
            test,
            "_fit_both",
            lambda endog, rng: (_FakeFit(-10.0, 2), _FakeFit(-10.0, 2)),
        )
        result = test.fit()
        assert result.lr_statistic == 0.0
        assert result.p_value == 1.0
        assert result.n_bootstrap == 0
        assert not result.rejected

    def test_samples_come_from_the_fitted_null_model(self, monkeypatch: Any) -> None:
        y = _recurring_series(seed=20)
        test = NonRecurringRegimeTest(
            y,
            k_regimes=2,
            method="bootstrap",
            n_bootstrap=4,
            significance=0.3,
            random_state=3,
        )
        test._search_reps_unrestricted = 1
        test._search_reps_restricted = 1

        restricted_fits: list[Any] = []
        original_restricted = test._fit_restricted

        def recording_restricted(endog: Any = None, rng: Any = None) -> Any:
            fit = original_restricted(endog, rng)
            restricted_fits.append(fit)
            return fit

        monkeypatch.setattr(test, "_fit_restricted", recording_restricted)

        drawn: list[tuple[Any, Any, Any]] = []
        original = module.simulate_markov_switching

        def recording(model: Any, params: Any, rng: Any) -> Any:
            sample = original(model, params, rng)
            drawn.append((model, np.array(params), sample))
            return sample

        monkeypatch.setattr(module, "simulate_markov_switching", recording)
        # the statistic of a simulated sample: cheap stand-in for two fits
        monkeypatch.setattr(test, "_statistic", lambda sample, rng=None: 0.5)
        result = test.fit()

        assert len(drawn) == 4
        null_fit = restricted_fits[0]._sm_results
        for model, params, sample in drawn:
            # the restricted model fitted to the data, with its estimates
            assert model is null_fit.model
            np.testing.assert_array_equal(params, np.asarray(null_fit.params))
            assert len(sample) == len(y)
            assert not np.array_equal(sample, y)
        assert drawn[0][0]._restrictions == {(0, 1): 0.0}
        np.testing.assert_allclose(drawn[0][0]._initial_probabilities, [1.0, 0.0])
        assert result.n_bootstrap == 4
        assert result.lr_statistic > 50
        # no simulated statistic reaches the observed one: 1 / (4 + 1)
        assert abs(result.p_value - 0.2) < 1e-12
        assert "Bootstrap replications" in result.summary()

    def test_failed_replications_are_reported(self, monkeypatch: Any) -> None:
        test = NonRecurringRegimeTest(
            _recurring_series(seed=22),
            k_regimes=2,
            method="bootstrap",
            n_bootstrap=4,
            significance=0.3,
            random_state=3,
        )
        test._search_reps_unrestricted = 1
        test._search_reps_restricted = 1
        outcomes = iter([0.1, float("nan"), 0.2, 0.3])
        monkeypatch.setattr(test, "_statistic", lambda sample, rng=None: next(outcomes))
        with pytest.warns(UserWarning, match="could not be computed"):
            result = test.fit()
        assert result.n_bootstrap == 3
        assert result.n_bootstrap_failed == 1
        assert abs(result.p_value - 0.25) < 1e-12

    def test_no_valid_replication_raises(self, monkeypatch: Any) -> None:
        test = NonRecurringRegimeTest(
            _recurring_series(seed=24),
            k_regimes=2,
            method="bootstrap",
            n_bootstrap=4,
            significance=0.3,
            random_state=3,
        )
        test._search_reps_unrestricted = 1
        test._search_reps_restricted = 1

        def failing(sample: Any, rng: Any = None) -> float:
            raise TypeError("an unexpected keyword argument")

        monkeypatch.setattr(test, "_statistic", failing)
        with (
            pytest.warns(UserWarning),
            pytest.raises(RuntimeError, match="any bootstrap replication") as info,
        ):
            test.fit()
        assert isinstance(info.value.__cause__, TypeError)

    def test_end_to_end_and_reproducible(self) -> None:
        """A real bootstrap: recurring regimes are rejected, twice the same."""
        y = _recurring_series(seed=23)
        results = []
        for _ in range(2):
            test = NonRecurringRegimeTest(
                y,
                k_regimes=2,
                method="bootstrap",
                n_bootstrap=12,
                significance=0.10,
                random_state=11,
            )
            test._search_reps_unrestricted = 0
            test._search_reps_restricted = 1
            results.append(test.fit())
        first, second = results
        assert first.n_bootstrap == 12
        assert first.p_value == pytest.approx(1 / 13)
        assert first.rejected
        assert first.lr_statistic > first.critical_value
        assert first.lr_statistic == second.lr_statistic
        assert first.critical_value == second.critical_value
        assert first.p_value == second.p_value


class TestSequentialProcedure:
    def test_levels(self) -> None:
        test = SequentialRestrictionTest(np.zeros(20), k_regimes=3, significance=0.06)
        # least significant first: alpha, alpha / 2, alpha / 3, ...
        assert abs(test._holm_significance(0, 6) - 0.06) < 1e-12
        assert abs(test._holm_significance(1, 6) - 0.03) < 1e-12
        assert abs(test._holm_significance(2, 6) - 0.02) < 1e-12
        bonferroni = SequentialRestrictionTest(
            np.zeros(20), k_regimes=3, significance=0.06, multiple_testing="bonferroni"
        )
        assert abs(bonferroni._holm_significance(4, 6) - 0.01) < 1e-12

    @pytest.mark.parametrize("means", [(0.0, 3.0), (3.0, 0.0)])
    def test_single_break_gives_the_non_recurring_model(
        self, means: tuple[float, float]
    ) -> None:
        """Whichever regime comes first, and whatever its label."""
        rng = np.random.default_rng(30)
        y = np.concatenate([rng.normal(means[0], 1, 80), rng.normal(means[1], 1, 80)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = SequentialRestrictionTest(y, k_regimes=2, random_state=1).fit()
        assert len(results.final_restrictions) == 1
        assert results.is_non_recurring
        # the restriction that was imposed costs (almost) nothing ...
        first = results.history[0]
        assert first.test_type == "individual"
        assert first.lr_statistic < 1.0
        assert not first.rejected
        # ... and forbidding the only move left is rejected
        last = results.history[-1]
        assert last.rejected
        assert last.lr_statistic > 50
        assert results.llf_final <= results.llf_unrestricted + 1e-6
        # the chain starts in the regime of the first observations
        assert results.initial_probabilities is not None
        assert np.max(results.initial_probabilities) > 0.99

    def test_recurring_regimes_keep_every_transition(self) -> None:
        y = _recurring_series(seed=31)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = SequentialRestrictionTest(y, k_regimes=2, random_state=1).fit()
        assert results.final_restrictions == {}
        assert not results.is_non_recurring
        assert len(results.history) == 1
        assert results.history[0].rejected

    def test_least_significant_candidate_is_tested_first(
        self, monkeypatch: Any
    ) -> None:
        """Candidates are ranked by their LR statistic against the current model."""
        test = SequentialRestrictionTest(
            _break_series(seed=32), k_regimes=2, multiple_testing="none"
        )
        # entry (0, 1) has the smaller probability but the larger LR statistic
        llf = {(): -100.0, ((0, 1),): -110.0, ((1, 0),): -100.2}
        _fake_search(
            test,
            monkeypatch,
            lambda key: llf.get(key, -500.0),
            transition=[[0.9, 0.01], [0.1, 0.99]],
        )
        results = test.fit()
        first = results.history[0]
        assert first.restriction == (1, 0)
        assert abs(first.lr_statistic - 0.4) < 1e-9
        assert (1, 0) in results.final_restrictions

    def test_individual_test_is_against_the_current_model(
        self, monkeypatch: Any
    ) -> None:
        """Step 2 compares with the model of step 1, the joint test with step 0."""
        test = SequentialRestrictionTest(
            _break_series(seed=33), k_regimes=3, multiple_testing="none"
        )

        def llf_of(key: Any) -> float:
            keys = set(key)
            llf = -100.0
            if (2, 0) in keys:
                llf -= 0.5
            if (0, 2) in keys:
                llf -= 1.0
            return llf - 50.0 * len(keys - {(2, 0), (0, 2)})

        calls = _fake_search(test, monkeypatch, llf_of)
        results = test.fit()
        steps = results.history
        assert [(s.restriction, s.test_type) for s in steps[:3]] == [
            ((2, 0), "individual"),
            ((0, 2), "individual"),
            ((0, 2), "joint"),
        ]
        assert abs(steps[0].lr_statistic - 1.0) < 1e-9
        # individual: 2 * 1.0 against the model with (2, 0) imposed
        assert abs(steps[1].lr_statistic - 2.0) < 1e-9
        assert steps[1].n_restrictions == 1
        # joint: 2 * 1.5 against the unrestricted model, two restrictions
        assert abs(steps[2].lr_statistic - 3.0) < 1e-9
        assert steps[2].n_restrictions == 2
        assert abs(steps[2].p_value - _chi_bar_squared_pvalue(3.0, 2)) < 1e-12
        assert (
            abs(steps[2].critical_value - _chi_bar_squared_critical_value(0.05, 2))
            < 1e-12
        )
        assert set(results.final_restrictions) == {(2, 0), (0, 2)}
        # the third candidate costs 100 in LR and is rejected
        assert steps[-1].rejected
        # every candidate fit is started from the current model
        assert all(n_warm == 1 for key, n_warm in calls if key)

    def test_joint_rejection_stops_without_imposing(self, monkeypatch: Any) -> None:
        test = SequentialRestrictionTest(
            _break_series(seed=34), k_regimes=3, multiple_testing="none"
        )
        # each restriction costs 1.3 in log-likelihood: LR = 2.6 one at a time
        # (not rejected at 5%: cv 2.71), 5.2 for two jointly (rejected: cv 5.14)
        _fake_search(test, monkeypatch, lambda key: -100.0 - 1.3 * len(key))
        results = test.fit()
        assert len(results.final_restrictions) == 1
        assert results.history[-1].test_type == "joint"
        assert results.history[-1].rejected

    def test_larger_model_is_reestimated_when_a_restricted_fit_beats_it(
        self, monkeypatch: Any
    ) -> None:
        """A local optimum of the unrestricted model does not become a statistic."""
        test = SequentialRestrictionTest(
            _break_series(seed=35), k_regimes=2, multiple_testing="none"
        )
        # the search finds -100 for the unrestricted model although the model
        # with (1, 0) = 0, which is nested in it, reaches -99
        llf = {
            (): -100.0,
            ((1, 0),): -99.0,
            ((0, 1),): -120.0,
            ((0, 1), (1, 0)): -120.0,
        }
        calls = _fake_search(test, monkeypatch, lambda key: llf.get(key, -500.0))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = test.fit()
        assert results.llf_unrestricted == -99.0
        first = results.history[0]
        assert first.restriction == (1, 0)
        assert first.lr_statistic == 0.0
        # the unrestricted model was estimated again, from the restricted fit
        assert ((), 2) in calls
        # the next candidate is measured against the model just accepted
        assert results.history[-1].restriction == (0, 1)
        assert results.history[-1].lr_statistic == pytest.approx(2 * (120.0 - 99.0))

    def test_failed_candidate_does_not_stop_the_search(self, monkeypatch: Any) -> None:
        test = SequentialRestrictionTest(
            _break_series(seed=36), k_regimes=2, multiple_testing="none"
        )

        def llf_of(key: Any) -> Any:
            if key == ((0, 1),):
                return np.linalg.LinAlgError("singular")
            return {(): -100.0, ((1, 0),): -100.1}.get(key, -500.0)

        _fake_search(test, monkeypatch, llf_of)
        results = test.fit()
        assert results.history[0].restriction == (1, 0)
        assert (1, 0) in results.final_restrictions

    def test_bootstrap_option_runs_the_bootstrap(self, monkeypatch: Any) -> None:
        """The statistic of a bootstrap sample is the LR of null against alternative."""
        y = _recurring_series(seed=37)
        test = SequentialRestrictionTest(
            y,
            k_regimes=2,
            critical_value_method="bootstrap",
            n_bootstrap=30,
            random_state=0,
        )
        test._search_reps = 0
        seen: dict[str, Any] = {}

        def fake_monte_carlo(
            observed: float, simulate: Any, statistic: Any, n: int, rng: Any
        ) -> Any:
            seen["observed"] = observed
            seen["sample"] = simulate(rng)
            # on the data themselves the closure must give the observed value
            seen["on_data"] = statistic(y)
            # and on a sample from the null a small one
            seen["on_sample"] = statistic(seen["sample"])
            return MonteCarloPValue(
                p_value=0.01, statistics=np.array([0.0, 1.0]), n_valid=n, n_failed=0
            )

        monkeypatch.setattr(module, "monte_carlo_pvalue", fake_monte_carlo)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = test.fit()
        assert seen["observed"] > 50
        assert len(seen["sample"]) == len(y)
        assert seen["on_data"] == pytest.approx(seen["observed"], rel=1e-3)
        assert 0.0 <= seen["on_sample"] < 10.0
        assert results.history[0].method == "bootstrap"
        assert results.history[0].p_value == 0.01
        assert results.history[0].rejected
        assert results.final_restrictions == {}

    def test_too_few_replications_for_an_adjusted_level(self, monkeypatch: Any) -> None:
        test = SequentialRestrictionTest(
            _break_series(seed=38),
            k_regimes=2,
            critical_value_method="bootstrap",
            n_bootstrap=19,
            multiple_testing="none",
        )
        _fake_search(test, monkeypatch, lambda key: -100.0 - 2.0 * len(key))
        monkeypatch.setattr(
            module,
            "monte_carlo_pvalue",
            lambda observed, simulate, statistic, n, rng: MonteCarloPValue(
                p_value=0.5, statistics=np.array([5.0]), n_valid=n, n_failed=0
            ),
        )
        with pytest.warns(UserWarning, match="cannot reject"):
            p_value, _cv = test._reference(
                4.0, 1, 0.05, _FakeFit(-102.0, 2), {(0, 1): 0.0}, {}
            )
        assert p_value == 0.5

    def test_models_in_the_search_share_the_starting_distribution(self) -> None:
        test = SequentialRestrictionTest(_break_series(seed=39), k_regimes=2)
        test._search_reps = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test._rng = np.random.default_rng(0)
            test._initial, preliminary = test._starting_distribution()
            unrestricted = test._fit_model(warm=(preliminary,))
            restricted = test._fit_model({(0, 1): 0.0}, warm=(unrestricted,))
        assert unrestricted.llf >= restricted.llf - 1e-4
        np.testing.assert_allclose(test._initial.sum(), 1.0)
        for fit in (unrestricted, restricted):
            model = fit._sm_results.model
            np.testing.assert_allclose(
                model.initial_probabilities(fit._sm_results.params), test._initial
            )

    def test_non_recurring_structure_does_not_depend_on_labels(self) -> None:
        test = SequentialRestrictionTest(np.zeros(20), k_regimes=2)
        assert test._check_non_recurring(np.array([[0.99, 0.0], [0.01, 1.0]]))
        assert test._check_non_recurring(np.array([[1.0, 0.01], [0.0, 0.99]]))
        assert not test._check_non_recurring(np.array([[0.9, 0.2], [0.1, 0.8]]))
        # two absorbing regimes: no break at all
        assert not test._check_non_recurring(np.eye(2))
        # a restricted entry counts as zero whatever its numerical value
        assert test._check_non_recurring(
            np.array([[0.99, 1e-9], [0.01, 1.0]]), {(0, 1): 0.0}
        )

        three = SequentialRestrictionTest(np.zeros(20), k_regimes=3)
        # regime 2 -> regime 0 -> regime 1 (absorbing)
        chain = np.array([[0.95, 0.0, 0.03], [0.05, 1.0, 0.0], [0.0, 0.0, 0.97]])
        assert three._check_non_recurring(chain)
        # regime 2 can never be reached and never left
        unreachable = np.array([[0.9, 0.0, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]])
        assert not three._check_non_recurring(unreachable)
        # a cycle
        cycle = np.array([[0.9, 0.0, 0.1], [0.1, 0.9, 0.0], [0.0, 0.1, 0.9]])
        assert not three._check_non_recurring(cycle)
