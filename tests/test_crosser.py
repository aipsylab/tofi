"""Tests of `Crosser`: scikit-learn integration, fitting and prediction."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
    ShuffleSplit,
    StratifiedKFold,
)
from sklearn.utils.validation import check_is_fitted

from tofi import Crosser, NotInferredError
from tofi.utils import InvalidParameterError

CVS = {
    "kfold": KFold(4),
    "kfold_shuffle": KFold(3, shuffle=True, random_state=0),
    "repeated": RepeatedKFold(n_splits=3, n_repeats=2, random_state=1),
    "shuffle": ShuffleSplit(4, test_size=0.25, random_state=2),
}
SHAPES = {  # (n_splits, n_repeats, n_folds)
    "kfold": (4, 1, 4),
    "kfold_shuffle": (3, 1, 3),
    "repeated": (6, 2, 3),
    "shuffle": (4, 4, 1),
}


@pytest.fixture
def reg():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    return X, X @ [1.0, 0.5, 0.0] + rng.normal(size=80)


@pytest.fixture
def clf():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(90, 3))
    y = np.array(["lo", "mid", "hi"])[
        np.digitize(X[:, 0] + rng.logistic(size=90), [-1, 1])
    ]
    return X, y


# ---------------------------------------------------------------------
# scikit-learn integration
# ---------------------------------------------------------------------


def test_get_params_clone_and_repr():
    crosser = Crosser(LinearRegression(), KFold(3), scoring="r2")
    params = crosser.get_params(deep=False)
    assert set(params) == {"estimator", "cv", "scoring"}
    cloned = clone(crosser)
    assert isinstance(cloned, Crosser) and cloned is not crosser
    assert repr(cloned.cv) == repr(crosser.cv)
    assert "Crosser" in repr(crosser)


def test_parameters_are_not_modified():
    crosser = Crosser(LinearRegression(), KFold(3))
    assert crosser.scoring is None
    assert crosser._scoring == "r2"
    assert Crosser(LinearDiscriminantAnalysis(), KFold(3))._scoring == "accuracy"


def test_estimator_type_and_tags():
    assert is_regressor(Crosser(LinearRegression(), KFold(3)))
    assert is_classifier(Crosser(LinearDiscriminantAnalysis(), KFold(3)))
    grid = GridSearchCV(LinearDiscriminantAnalysis(), {"tol": [1e-4]})
    assert is_classifier(Crosser(grid, KFold(3)))


def test_check_is_fitted(reg):
    X, y = reg
    crosser = Crosser(LinearRegression(), KFold(3))
    with pytest.raises(Exception, match="not fitted"):
        check_is_fitted(crosser)
    crosser.fit(X, y)
    check_is_fitted(crosser)


@pytest.mark.parametrize("kind", list(CVS))
def test_split_shapes(kind, reg):
    X, y = reg
    crosser = Crosser(LinearRegression(), CVS[kind]).fit(X, y)
    n_splits, n_repeats, n_folds = SHAPES[kind]
    assert crosser._n_splits == n_splits == len(crosser.estimators_)
    assert crosser._n_repeats == n_repeats
    assert crosser._n_folds == n_folds
    assert len(crosser._test_indices(X, y)) == n_splits


def test_splits_are_stable_across_calls(reg):
    X, y = reg
    crosser = Crosser(LinearRegression(), CVS["repeated"])
    first = crosser._test_indices(X, y)
    second = crosser._test_indices(X, y)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------
# Fitting and scores
# ---------------------------------------------------------------------


@pytest.mark.parametrize("frame", [False, True])
def test_estimators_fitted_on_training_folds(frame, reg):
    X, y = reg
    if frame:
        X, y = pd.DataFrame(X, columns=list("abc")), pd.Series(y)
    crosser = Crosser(LinearRegression(), CVS["kfold_shuffle"]).fit(X, y)
    for split, (train, test) in enumerate(crosser._split_indices(X, y)):
        X_train = X.iloc[train] if frame else X[train]
        y_train = y.iloc[train] if frame else y[train]
        expected = LinearRegression().fit(X_train, y_train)
        np.testing.assert_allclose(crosser.estimators_[split].coef_, expected.coef_)
        X_test = X.iloc[test] if frame else X[test]
        y_test = y.iloc[test] if frame else y[test]
        assert crosser.test_scores_[split] == pytest.approx(
            r2_score(y_test, expected.predict(X_test))
        )


def test_val_scores_from_best_score(reg):
    X, y = reg
    grid = GridSearchCV(LinearRegression(), {"fit_intercept": [True, False]}, cv=3)
    crosser = Crosser(grid, KFold(3)).fit(X, y)
    assert np.all(np.isfinite(crosser.val_scores_))
    plain = Crosser(LinearRegression(), KFold(3)).fit(X, y)
    assert np.all(np.isnan(plain.val_scores_))


def test_callable_scoring(reg, capsys):
    X, y = reg

    def neg_mae(estimator, X, y):
        return -np.mean(np.abs(estimator.predict(X) - y))

    crosser = Crosser(LinearRegression(), KFold(3), scoring=neg_mae).fit(X, y)
    assert np.all(crosser.test_scores_ < 0)
    crosser.summarize()
    assert "Train/Test Scorer: Callable" in capsys.readouterr().out


def test_classifier_attributes(clf):
    X, y = clf
    crosser = Crosser(LinearDiscriminantAnalysis(), KFold(3)).fit(X, y)
    assert crosser.classes_ == ["hi", "lo", "mid"]
    assert list(crosser.label_binarizer_.classes_) == ["hi", "lo", "mid"]


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(CVS))
def test_summarize_shapes(kind, reg):
    X, y = reg
    crosser = Crosser(LinearRegression(), CVS[kind]).fit(X, y)
    n_splits, n_repeats, n_folds = SHAPES[kind]
    split_level = crosser.summarize(cross_fit=False, combine=False, verbose=False)
    assert len(split_level) == n_splits
    assert list(split_level.index.names) == ["split", "repeat", "fold"]
    repeat_level = crosser.summarize(cross_fit=True, combine=False, verbose=False)
    assert len(repeat_level) == n_repeats
    np.testing.assert_allclose(
        repeat_level["test_score"],
        crosser.test_scores_.reshape(n_repeats, n_folds).mean(axis=1),
    )
    combined = crosser.summarize(cross_fit=False, combine=True, verbose=False)
    assert list(combined.index) == ["mean", "std"]
    assert combined.loc["mean", "test_score"] == pytest.approx(
        crosser.test_scores_.mean()
    )
    assert combined.loc["std", "test_score"] == pytest.approx(
        crosser.test_scores_.std()
    )


def test_summarize_reverse(reg):
    X, y = reg
    crosser = Crosser(LinearRegression(), KFold(3)).fit(X, y)
    a = crosser.summarize(verbose=False)
    b = crosser.summarize(reverse=True, verbose=False)
    np.testing.assert_allclose(a["test_score"], -b["test_score"])


def test_summarize_verbose_text(reg, capsys):
    X, y = reg
    Crosser(LinearRegression(), KFold(4)).fit(X, y).summarize()
    assert capsys.readouterr().out == (
        "Crosser Summary (cross_fit=True, combine=False)\n"
        " + Estimator: LinearRegression\n"
        " + Cross-Validator: KFold (n_folds=4, n_repeats=1)\n"
        " + Train/Test Scorer: R2 (reverse=False)\n"
    )


def test_summarize_requires_fit():
    with pytest.raises(Exception, match="not fitted"):
        Crosser(LinearRegression(), KFold(3)).summarize()


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------


def test_predict_split_and_average(reg):
    X, y = reg
    crosser = Crosser(LinearRegression(), KFold(3)).fit(X, y)
    for split, estimator in enumerate(crosser.estimators_):
        np.testing.assert_array_equal(
            crosser.predict(X, split=split), estimator.predict(X)
        )
    np.testing.assert_allclose(
        crosser.predict(X),
        np.mean([e.predict(X) for e in crosser.estimators_], axis=0),
    )


@pytest.mark.parametrize(
    "method", ["predict_proba", "predict_log_proba", "decision_function"]
)
def test_probability_methods(method, clf):
    X, y = clf
    crosser = Crosser(LinearDiscriminantAnalysis(), KFold(3)).fit(X, y)
    np.testing.assert_allclose(
        getattr(crosser, method)(X),
        np.mean([getattr(e, method)(X) for e in crosser.estimators_], axis=0),
    )
    np.testing.assert_array_equal(
        getattr(crosser, method)(X, split=1),
        getattr(crosser.estimators_[1], method)(X),
    )


def test_majority_vote_with_string_labels(clf):
    X, y = clf
    crosser = Crosser(LinearDiscriminantAnalysis(), KFold(5)).fit(X, y)
    votes = np.array([e.predict(X) for e in crosser.estimators_])
    pred = crosser.predict(X)
    for column, label in zip(votes.T, pred):
        values, counts = np.unique(column, return_counts=True)
        assert label == values[np.argmax(counts)]


def test_majority_vote_matches_bincount_for_integer_labels():
    from tofi.crosser import _majority

    rng = np.random.default_rng(0)
    for _ in range(200):
        labels = rng.integers(0, 4, size=rng.integers(1, 8))
        assert _majority(labels) == np.argmax(np.bincount(labels))


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "cv",
    [
        StratifiedKFold(3),
        KFold(3, shuffle=True),
        RepeatedKFold(n_splits=3, n_repeats=2),
        ShuffleSplit(3),
        KFold(3, shuffle=True, random_state=np.random.RandomState(0)),
    ],
)
def test_invalid_cv(cv):
    with pytest.raises(InvalidParameterError):
        Crosser(LinearRegression(), cv)


@pytest.mark.parametrize("scoring", ["not_a_scorer", 3])
def test_invalid_scoring(scoring):
    with pytest.raises(ValueError):
        Crosser(LinearRegression(), KFold(3), scoring=scoring)


def test_invalid_parameter_error_is_value_and_type_error():
    assert issubclass(InvalidParameterError, ValueError)
    assert issubclass(InvalidParameterError, TypeError)
    assert issubclass(NotInferredError, ValueError)
