import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier, is_regressor
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, RepeatedKFold

from tofi import CIT, RIT, Crosser, LeakageWarning
from tofi.utils import _child_seeds


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    # x0 matters, x3 is pure noise
    y = 2 * X[:, 0] + X[:, 1] + rng.normal(size=200)
    return X, y


@pytest.fixture
def clf_data():
    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )
    # keep the informative features first but shuffle the rows
    order = np.random.default_rng(0).permutation(len(y))
    return X[order], y[order]


def _split(X, y):
    half = len(y) // 2
    return X[:half], X[half:], y[:half], y[half:]


# ---------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------


def test_zero_one_loss_counts_errors(clf_data):
    X, y = clf_data
    X_train, X_test, y_train, y_test = _split(X, y)
    learner = LogisticRegression().fit(X_train, y_train)
    competitor = LogisticRegression().fit(np.delete(X_train, 0, 1), y_train)
    rit = RIT(learner, competitor, 0, "LOCO", loss_func="zero_one_loss").infer(
        X_test, y_test
    )
    error_rate = np.mean(learner.predict(X_test) != y_test)
    assert rit.l_losses_[0].mean() == pytest.approx(error_rate)
    # x0 is informative, so removing it must increase the error rate
    assert rit.summarize(verbose=False)["estimate"].iloc[0] > 0


def test_cit_binary_log_loss_matches_sklearn(clf_data):
    from sklearn.metrics import log_loss

    X, y = clf_data
    X_train, X_test, y_train, y_test = _split(X, y)
    learner = LogisticRegression().fit(X_train, y_train)
    cit = CIT(learner, None, 3, "CPI", random_state=0).infer(X_test, y_test)
    assert cit.l_losses_[0].mean() == pytest.approx(
        log_loss(y_test, learner.predict_proba(X_test))
    )


def test_cit_resampled_loss_uses_probabilities(clf_data):
    # With an irrelevant feature permuted, the resampled log loss must be
    # close to the original one (it was wildly off when `predict` was used).
    X, y = clf_data
    X_train, X_test, y_train, y_test = _split(X, y)
    learner = LogisticRegression().fit(X_train, y_train)
    cit = CIT(learner, None, 3, "CPI", random_state=0).infer(X_test, y_test)
    assert abs(cit.r_losses_[0].mean() - cit.l_losses_[0].mean()) < 0.1


# ---------------------------------------------------------------------
# scikit-learn compatibility
# ---------------------------------------------------------------------


def test_crosser_estimator_type():
    reg = Crosser(LinearRegression(), KFold(3))
    clf = Crosser(LogisticRegression(), KFold(3))
    assert is_regressor(reg) and not is_classifier(reg)
    assert is_classifier(clf)


def test_crosser_fit_and_summarize(reg_data):
    X, y = reg_data
    crosser = Crosser(LinearRegression(), KFold(5)).fit(X, y)
    summary = crosser.summarize(verbose=False)
    assert summary.shape[0] == 1


def test_crosser_rejects_randomstate_instance():
    with pytest.raises(ValueError):
        Crosser(
            LinearRegression(),
            KFold(3, shuffle=True, random_state=np.random.RandomState(0)),
        )


def test_equal_but_distinct_cv_accepted(reg_data):
    X, y = reg_data
    learner = Crosser(LinearRegression(), KFold(5, shuffle=True, random_state=1)).fit(
        X, y
    )
    sampler = Crosser(LinearRegression(), KFold(5, shuffle=True, random_state=1)).fit(
        np.delete(X, 0, 1), X[:, 0]
    )
    CIT(learner, sampler, 0, "RPT", n_copies=20)


def test_different_cv_rejected(reg_data):
    X, y = reg_data
    learner = Crosser(LinearRegression(), KFold(5)).fit(X, y)
    sampler = Crosser(LinearRegression(), KFold(4)).fit(np.delete(X, 0, 1), X[:, 0])
    with pytest.raises(ValueError):
        CIT(learner, sampler, 0, "RPT")


# ---------------------------------------------------------------------
# Leakage warning
# ---------------------------------------------------------------------


def test_leakage_warning_cit(reg_data):
    X, y = reg_data
    learner = Crosser(LinearRegression(), KFold(5)).fit(X, y)
    sampler = LinearRegression().fit(np.delete(X, 0, 1), X[:, 0])
    with pytest.warns(LeakageWarning):
        CIT(learner, sampler, 0, "RPT")


def test_leakage_warning_rit(reg_data):
    X, y = reg_data
    learner = Crosser(LinearRegression(), KFold(5)).fit(X, y)
    competitor = LinearRegression().fit(np.delete(X, 0, 1), y)
    with pytest.warns(LeakageWarning):
        RIT(learner, competitor, 0, "LOCO")


def test_no_leakage_warning_when_consistent(reg_data):
    X, y = reg_data
    learner = Crosser(LinearRegression(), KFold(5)).fit(X, y)
    competitor = Crosser(LinearRegression(), KFold(5)).fit(np.delete(X, 0, 1), y)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LeakageWarning)
        RIT(learner, competitor, 0, "LOCO")
        CIT(learner, None, 0, "CPI")


# ---------------------------------------------------------------------
# Randomness
# ---------------------------------------------------------------------


def test_child_seeds():
    assert _child_seeds(None, 3) == [None, None, None]
    seeds = _child_seeds(0, 4)
    assert len(set(seeds)) == 4
    assert seeds == _child_seeds(0, 4)


def test_folds_use_different_random_streams(reg_data):
    X, y = reg_data
    X = X[:100]
    y = y[:100]
    # Two folds of identical data: with a shared seed the permutations
    # would be identical.
    X2 = np.vstack([X, X])
    y2 = np.concatenate([y, y])
    learner = Crosser(LinearRegression(), KFold(2)).fit(X2, y2)
    cit = CIT(learner, None, 0, "RPT", n_copies=50, random_state=0).infer(X2, y2)
    assert not np.allclose(cit.null_values_[0], cit.null_values_[1])


def test_reproducible(reg_data):
    X, y = reg_data
    X_train, X_test, y_train, y_test = _split(X, y)
    learner = LinearRegression().fit(X_train, y_train)
    sampler = LinearRegression().fit(np.delete(X_train, 1, 1), X_train[:, 1])
    a = (
        CIT(learner, sampler, 1, "RPT", n_copies=100, random_state=3)
        .infer(X_test, y_test)
        .summarize(verbose=False)
    )
    b = (
        CIT(learner, sampler, 1, "RPT", n_copies=100, random_state=3)
        .infer(X_test, y_test)
        .summarize(verbose=False)
    )
    pd.testing.assert_frame_equal(a, b)


def test_double_split_without_seed_warns(reg_data):
    X, y = reg_data
    learner = LinearRegression().fit(X, y)
    competitor = LinearRegression().fit(np.delete(X, 0, 1), y)
    with pytest.warns(UserWarning, match="cannot be replicated"):
        RIT(learner, competitor, 0, "PIE")
    with pytest.warns(UserWarning, match="cannot be replicated"):
        RIT(learner, competitor, 0, "LOCO", double_split=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RIT(learner, competitor, 0, "PIE", random_state=0)
        RIT(learner, competitor, 0, "LOCO")


def test_pie_double_split_is_random(reg_data):
    X, y = reg_data
    X_train, X_test, y_train, y_test = _split(X, y)
    learner = LinearRegression().fit(X_train, y_train)
    competitor = LinearRegression().fit(np.delete(X_train, 0, 1), y_train)
    pie = RIT(learner, competitor, 0, "PIE", random_state=0).infer(X_test, y_test)
    n = len(y_test)
    assert len(pie.l_losses_[0]) + len(pie.r_losses_[0]) == n
    full = (learner.predict(X_test) - y_test) ** 2
    even = full[::2]
    assert not np.allclose(np.sort(pie.l_losses_[0]), np.sort(even))


# ---------------------------------------------------------------------
# Defaults and validation
# ---------------------------------------------------------------------


def test_method_defaults(reg_data):
    X, y = reg_data
    learner = LinearRegression().fit(X, y)
    competitor = LinearRegression().fit(np.delete(X, 0, 1), y)
    loco = RIT(learner, competitor, 0)
    assert loco.method is None and loco.method_ == "LOCO"
    assert loco.double_split is None and loco.double_split_ is False
    pie = RIT(learner, competitor, 0, "PIE", random_state=0)
    assert pie.double_split_ is True
    rpt = CIT(learner, None, 0)
    assert rpt.method is None and rpt.method_ == "RPT"
    assert rpt.loss_func is None and rpt.loss_func_ == "mean_squared_error"
    assert rpt.null_dist_ == "resampling" and rpt.n_copies_ == 2000


def test_rit_rejects_resampling(reg_data):
    X, y = reg_data
    learner = LinearRegression().fit(X, y)
    competitor = LinearRegression().fit(np.delete(X, 0, 1), y)
    with pytest.raises(ValueError):
        RIT(learner, competitor, 0, "LOCO", null_dist="resampling")


def test_combined_p_value_names(reg_data):
    X, y = reg_data
    cv = RepeatedKFold(n_splits=2, n_repeats=3, random_state=0)
    learner = Crosser(LinearRegression(), cv).fit(X, y)
    competitor = Crosser(LinearRegression(), cv).fit(np.delete(X, 3, 1), y)
    summary = RIT(learner, competitor, 3).infer(X, y).summarize(verbose=False)
    methods = list(summary.index.get_level_values("method"))
    assert methods == ["gmean", "median", "min", "hmean", "hommel", "cauchy"]
