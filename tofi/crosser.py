"""Repeated cross-fitting of scikit-learn estimators."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.metrics import get_scorer, get_scorer_names
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted

from .utils import InvalidParameterError, _is_int, _take_rows

_ALLOWED_CVS = ("KFold", "RepeatedKFold", "ShuffleSplit")


class Crosser(BaseEstimator):
    """
    An object for repeated cross-fitting

    Parameters
    ----------
    estimator : estimator object
        The base estimator for repeated cross-fitting.

    cv : cross-validation generator
        The cross-validation generator for repeated cross-fitting.
        It must be a class member of `KFold`, `RepeatedKFold`, or
        `ShuffleSplit`. Its `random_state` must be an `int` (or `None`
        for `KFold` without shuffling) so that every call to `split`
        yields the same folds.

    scoring : str or callable, default=None
        Strategy to evaluate the performance of the cross-validated
        model when using `summarize`. To see availble scoring methods
        via `str`, use `sklearn.get_scorer_names()`.

            - For regression tasks, `None` means "r2".
            - For classification tasks, `None` means "accuracy".

    Attributes
    ----------
    estimators_ : list of estimators
        The fitted estimators, one per split.

    classes_ : list
        Class labels (classifiers only).

    label_binarizer_ : LabelBinarizer
        Binarizer fitted on all class labels (classifiers only).

    val_scores_, train_scores_, test_scores_ : ndarray of shape (n_splits,)
        Validation (`best_score_` of the estimator, if any), training and
        test scores of each split.
    """

    def __init__(
        self,
        estimator: Any,
        cv: Any,
        scoring: str | Callable | None = None,
    ):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self._check()

    # -----------------------------------------------------------------
    # Fitting and summarizing
    # -----------------------------------------------------------------

    def fit(self, X: Any, y: Any) -> Crosser:
        """
        Fit by repeated cross-fitting

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Training array, where `n_samples` is the sample size and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target relative to X for classification or regression.

        Returns
        -------
        self : object
            Instance of fitted crosser.
        """
        estimators = []
        for train_index in self._train_indices(X, y):
            estimator = clone(self.estimator)
            estimator.fit(_take_rows(X, train_index), _take_rows(y, train_index))
            estimators.append(estimator)
        self.estimators_ = estimators

        if is_classifier(self):
            self.label_binarizer_ = LabelBinarizer().fit(y)
            self.classes_ = sorted(set(y))

        val_scores, train_scores, test_scores = self._scores(X, y)
        self.val_scores_ = val_scores
        self.train_scores_ = train_scores
        self.test_scores_ = test_scores
        return self

    def summarize(
        self,
        cross_fit: bool | None = None,
        combine: bool | None = None,
        reverse: bool = False,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Summarize repeated cross-fitting results

        Parameters
        ----------

        cross_fit : bool, default=None
            If `True`, score values will be averaged across folds.
            It is applicable if `n_folds` > 1. If applicable, `None`
            means `True`; Otherwise, `None` means `False`.

        combine : bool, default=None
            If `True`, score values will be averaged across folds
            and repeats. It is applicable if `n_splits` > 1. If
            `n_repeats` > 1, `None` means `True`; Otherwise, `None`
            means `False`.

        reverse : bool, default=False
            If `True`, negative score values will be reported.
            Note that by default a larger score value means better
            in `scikit-learn`.

        verbose : bool, default=True
            Controls the verbosity.

        Returns
        -------
        summary : DataFrame
            A summary for validation, train, and test scores (`val_score`,
            `train_score`, and `test_score`). The validation scores
            are only presented if `estimator.best_score_` is available.
        """
        check_is_fitted(self)
        n_repeats = self._n_repeats
        n_folds = self._n_folds

        if combine is None:
            combine = n_repeats > 1
        if cross_fit is None:
            cross_fit = n_folds > 1

        scores = {
            "val_score": self.val_scores_,
            "train_score": self.train_scores_,
            "test_score": self.test_scores_,
        }
        if reverse is True:
            scores = {name: -value for name, value in scores.items()}
        if cross_fit:
            scores = {
                name: value.reshape((n_repeats, n_folds)).mean(axis=1)
                for name, value in scores.items()
            }

        if not combine:
            if cross_fit:
                index = pd.Index(range(n_repeats), name="repeat")
            else:
                index = _split_index(self._n_splits, n_folds)
            summary = pd.DataFrame(scores, index=index)
        else:
            index = pd.Index(["mean", "std"], name="method")
            summary = pd.DataFrame(
                {
                    name: [np.mean(value), np.std(value)]
                    for name, value in scores.items()
                },
                index=index,
            )

        if verbose:
            print(f"Crosser Summary (cross_fit={cross_fit}, combine={combine})")
            print(f" + Estimator: {self.estimator.__class__.__name__}")
            print(
                f" + Cross-Validator: {self.cv.__class__.__name__} "
                f"(n_folds={n_folds}, n_repeats={n_repeats})"
            )
            scoring = self._scoring
            if callable(scoring):
                name = "Callable"
            else:
                name = scoring.replace("_", " ").title()
            print(f" + Train/Test Scorer: {name} (reverse={reverse})")
        return summary

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------

    def predict(self, X: Any, *, split: int | None = None) -> np.ndarray:
        """Predict with one split's estimator, or aggregate all splits.

        With `split=None`, regression predictions are averaged and
        classification predictions are combined by majority vote (ties
        go to the smallest label).
        """
        check_is_fitted(self)
        if split is not None:
            return self.estimators_[split].predict(X)
        preds = np.array([estimator.predict(X) for estimator in self.estimators_])
        if is_classifier(self):
            return np.array([_majority(column) for column in preds.T])
        return preds.mean(axis=0)

    def predict_proba(self, X: Any, *, split: int | None = None) -> np.ndarray:
        """Class probabilities of one split, or their average."""
        return self._predict_average("predict_proba", X, split)

    def predict_log_proba(self, X: Any, *, split: int | None = None) -> np.ndarray:
        """Log-probabilities of one split, or their average."""
        return self._predict_average("predict_log_proba", X, split)

    def decision_function(self, X: Any, *, split: int | None = None) -> np.ndarray:
        """Decision function of one split, or its average."""
        return self._predict_average("decision_function", X, split)

    def _predict_average(self, method: str, X: Any, split: int | None) -> np.ndarray:
        check_is_fitted(self)
        if split is not None:
            return getattr(self.estimators_[split], method)(X)
        preds = np.array(
            [getattr(estimator, method)(X) for estimator in self.estimators_]
        )
        return preds.mean(axis=0)

    # -----------------------------------------------------------------
    # Internal interface used by the inferers
    # -----------------------------------------------------------------

    def _split_indices(self, X: Any, y: Any) -> list[tuple[np.ndarray, np.ndarray]]:
        """(train, test) row indices of every split."""
        return list(self.cv.split(X, y))

    def _train_indices(self, X: Any, y: Any) -> list[np.ndarray]:
        return [train for train, _ in self._split_indices(X, y)]

    def _test_indices(self, X: Any, y: Any) -> list[np.ndarray]:
        return [test for _, test in self._split_indices(X, y)]

    @property
    def _n_splits(self) -> int:
        return self.cv.get_n_splits()

    @property
    def _n_repeats(self) -> int:
        if isinstance(self.cv, model_selection.ShuffleSplit):
            return self.cv.get_n_splits()
        return getattr(self.cv, "n_repeats", 1)

    @property
    def _n_folds(self) -> int:
        return self._n_splits // self._n_repeats

    @property
    def _scoring(self) -> str | Callable:
        """`scoring` with `None` resolved to the default for the task."""
        if self.scoring is not None:
            return self.scoring
        return "accuracy" if is_classifier(self.estimator) else "r2"

    @property
    def _scorer(self) -> Callable:
        scoring = self._scoring
        return get_scorer(scoring) if isinstance(scoring, str) else scoring

    def _scores(self, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scorer = self._scorer
        val_scores, train_scores, test_scores = [], [], []
        for estimator, (train_index, test_index) in zip(
            self.estimators_, self._split_indices(X, y)
        ):
            train_scores.append(
                scorer(
                    estimator, _take_rows(X, train_index), _take_rows(y, train_index)
                )
            )
            test_scores.append(
                scorer(estimator, _take_rows(X, test_index), _take_rows(y, test_index))
            )
            val_scores.append(getattr(estimator, "best_score_", np.nan))
        return np.array(val_scores), np.array(train_scores), np.array(test_scores)

    def _check(self) -> None:
        cv = self.cv
        scoring = self.scoring

        allowed = tuple(getattr(model_selection, name) for name in _ALLOWED_CVS)
        if not isinstance(cv, allowed):
            raise InvalidParameterError(
                f"Supported `cv` types are {list(_ALLOWED_CVS)}."
            )
        if isinstance(cv, model_selection.KFold):
            if cv.shuffle and cv.random_state is None:
                raise InvalidParameterError(
                    "When `cv` is `KFold` and `cv.shuffle` is `True`, "
                    "`cv.random_state` cannot be `None`."
                )
        elif cv.random_state is None:
            raise InvalidParameterError(
                "When `cv` is `RepeatedKFold` or `ShuffleSplit`, "
                "`cv.random_state` cannot be `None`."
            )
        if not (cv.random_state is None or _is_int(cv.random_state)):
            raise InvalidParameterError(
                "`cv.random_state` must be an `int`; a `RandomState` "
                "instance would yield different folds on each call "
                "to `split`."
            )

        if scoring is not None:
            if not (isinstance(scoring, str) or callable(scoring)):
                raise InvalidParameterError(
                    "Supported `scoring` types are `str`, `callable`, or `None`."
                )
            if isinstance(scoring, str) and scoring not in get_scorer_names():
                raise InvalidParameterError(
                    f"Supported `scoring` names are {get_scorer_names()}."
                )

    # -----------------------------------------------------------------
    # scikit-learn integration
    # -----------------------------------------------------------------

    @property
    def _estimator_type(self) -> str:
        # Used by scikit-learn < 1.6 in `is_classifier`/`is_regressor`.
        if is_classifier(self.estimator):
            return "classifier"
        return getattr(self.estimator, "_estimator_type", "regressor")

    def __sklearn_tags__(self):
        # Used by scikit-learn >= 1.6, where `_estimator_type` was
        # replaced by estimator tags (and removed in 1.8).
        return self.estimator.__sklearn_tags__()

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "estimators_")


def _majority(labels: np.ndarray) -> Any:
    """Most frequent label; ties go to the smallest label."""
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]


def _split_index(n_splits: int, n_folds: int) -> pd.MultiIndex:
    splits = list(range(n_splits))
    return pd.MultiIndex.from_tuples(
        [(split, split // n_folds, split % n_folds) for split in splits],
        names=["split", "repeat", "fold"],
    )
