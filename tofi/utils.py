"""Shared helpers: exceptions, validation, data access and randomness."""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.preprocessing import LabelBinarizer

# ---------------------------------------------------------------------
# Exceptions and warnings
# ---------------------------------------------------------------------


class NotInferredError(ValueError):
    """Exception raised if an inferer is summarized before `infer`."""


class InvalidParameterError(ValueError, TypeError):
    """Exception raised for an invalid argument.

    It subclasses both `ValueError` and `TypeError` (as scikit-learn's
    `InvalidParameterError` does), so code written against tofi 0.0.1,
    which raised `ValueError` for every invalid argument, keeps working.
    """


class LeakageWarning(UserWarning):
    """Warning raised when the inference setup may leak test information.

    It is emitted when `learner` is a `Crosser` but `sampler` (CIT) or
    `competitor` (RIT) is an ordinary estimator. If that estimator was
    fitted on data that includes the test folds of `learner`, the
    resulting test is invalid. Use `warnings.simplefilter("error",
    LeakageWarning)` to turn it into an error.
    """


# ---------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------

_INT_TYPES = (numbers.Integral,)


def _is_int(value: Any) -> bool:
    return isinstance(value, _INT_TYPES) and not isinstance(value, bool)


def _check_option(name: str, value: Any, options: Sequence[Any]) -> None:
    """Raise if `value` is not `None` and not one of `options`."""
    if value is not None and value not in options:
        raise InvalidParameterError(
            f"Supported values for `{name}` are {list(options)}; got {value!r}."
        )


def _check_type(name: str, value: Any, kind: str) -> None:
    """Raise if `value` is not `None` and not of the given kind.

    `kind` is one of "int", "number", "bool" or "str".
    """
    if value is None:
        return
    valid = {
        "int": _is_int(value),
        "number": isinstance(value, numbers.Real) and not isinstance(value, bool),
        "bool": isinstance(value, (bool, np.bool_)),
        "str": isinstance(value, str),
    }[kind]
    if not valid:
        raise InvalidParameterError(
            f"`{name}` must be of type {kind}; got {type(value).__name__}."
        )


# ---------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------


def _take_rows(data: Any, index: np.ndarray | None) -> Any:
    """Rows of an array/DataFrame/Series; `None` means all rows."""
    if index is None:
        return data
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[index]
    return data[index]


def _split_columns(X: Any, removal: Any) -> tuple[Any, Any]:
    """Return `X` without `removal` and the removed column(s)."""
    if isinstance(X, pd.DataFrame):
        return X.drop(removal, axis=1), X[removal]
    return np.delete(X, removal, axis=1), X[:, removal]


def _values(y: Any) -> Any:
    """The underlying array of a Series (other inputs are unchanged)."""
    return y.values if isinstance(y, pd.Series) else y


def _two_columns(target: np.ndarray) -> np.ndarray:
    """Expand a one-column binary indicator to two columns.

    `LabelBinarizer` returns one column for binary problems whereas
    `predict_proba` returns two.
    """
    if target.shape[1] == 1:
        target = np.append(1 - target, target, axis=1)
    return target


def _class_binarizer(model: Any, y: Any) -> LabelBinarizer:
    """A `LabelBinarizer` fitted on the classes known to `model`."""
    binarizer = getattr(model, "label_binarizer_", None)
    if binarizer is not None:
        return binarizer
    binarizer = LabelBinarizer()
    if hasattr(model, "classes_"):
        binarizer.fit(model.classes_)
    else:
        binarizer.fit(sorted(set(y)))
    return binarizer


def _encode_target(model: Any, y: Any, binarize: bool) -> Any:
    """Target in the form expected by the loss for `model`'s output.

    With `binarize`, classifier targets are one-hot encoded (two columns
    for binary problems) to match `predict_proba`.
    """
    y = _values(y)
    if binarize and is_classifier(model):
        return _two_columns(_class_binarizer(model, y).transform(y))
    return y


# ---------------------------------------------------------------------
# Randomness
# ---------------------------------------------------------------------


def _child_seeds(random_state: int | None, n: int) -> list[int | None]:
    """Derive `n` independent integer seeds from `random_state`.

    If `random_state` is `None`, a list of `None` is returned so that
    every stream is freshly (and non-reproducibly) seeded.
    """
    if random_state is None:
        return [None] * n
    states = np.random.SeedSequence(int(random_state)).generate_state(n)
    return [int(state) for state in states]


def _same_cv(cv_a: Any, cv_b: Any) -> bool:
    """Whether two cross-validators generate identical splits.

    scikit-learn splitters do not implement `__eq__`, so two splitters
    constructed separately with the same parameters are compared via
    their type and `repr` (which lists all parameters).
    """
    if cv_a is cv_b:
        return True
    return type(cv_a) is type(cv_b) and repr(cv_a) == repr(cv_b)


def _sample(
    estimator: Any,
    X: Any,
    y: Any,
    n_copies: int | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Draw copies of `y` (a feature) given `X` (the other features).

    - `estimator` is `None`: marginal permutation of `y`.
    - classifier: draws from `predict_proba`.
    - regressor: prediction plus permuted residuals (RPT).
    """
    rng = np.random.default_rng(random_state)
    if estimator is None:
        if n_copies is None:
            return rng.permutation(y)
        return np.array([rng.permutation(y) for _ in range(n_copies)])

    if is_classifier(estimator):
        pred = estimator.predict_proba(X)
        binarizer = LabelBinarizer().fit(estimator.classes_)
        if n_copies is None:
            return _inverse_binarize(binarizer, rng.multinomial(1, pred))
        draws = rng.multinomial(1, pred, (n_copies, len(pred)))
        return np.array([_inverse_binarize(binarizer, draw) for draw in draws])

    pred = estimator.predict(X)
    residual = np.asarray(y) - pred
    if n_copies is None:
        return pred + rng.permutation(residual)
    return pred + np.array([rng.permutation(residual) for _ in range(n_copies)])


def _inverse_binarize(binarizer: LabelBinarizer, onehot: np.ndarray) -> np.ndarray:
    # For binary problems `LabelBinarizer` expects a single column,
    # whereas `multinomial` draws from `predict_proba` give two.
    if len(binarizer.classes_) == 2 and onehot.shape[1] == 2:
        onehot = onehot[:, [1]]
    return binarizer.inverse_transform(onehot)


# ---------------------------------------------------------------------
# Null distributions
# ---------------------------------------------------------------------


# Permutations are generated in blocks of at most this many elements so
# that memory stays bounded for large test sets.
_BLOCK_ELEMENTS = 2**20


def _blocks(n_permutations: int, n_obs: int):
    """Yield (start, stop) ranges of permutations processed together."""
    block = max(1, _BLOCK_ELEMENTS // max(n_obs, 1))
    for start in range(0, n_permutations, block):
        yield start, min(start + block, n_permutations)


def _paired_permutation_null(
    l_loss: np.ndarray,
    r_loss: np.ndarray,
    n_permutations: int,
    random_state: int | None,
) -> np.ndarray:
    """Null values from randomly swapping the paired losses.

    Each null value is mean(first) - mean(second), where every pair
    (l_i, r_i) is swapped with probability 1/2. The two means are
    computed separately (rather than as the mean of signed differences)
    so that ties with the observed estimate are resolved exactly as for
    the estimate itself, which matters for discrete losses.
    """
    rng = np.random.default_rng(random_state)
    l_loss = np.asarray(l_loss, dtype=float)
    r_loss = np.asarray(r_loss, dtype=float)
    null_values = np.empty(n_permutations)
    for start, stop in _blocks(n_permutations, len(l_loss)):
        swap = rng.random((stop - start, len(l_loss))) < 0.5
        first = np.where(swap, r_loss, l_loss)
        second = np.where(swap, l_loss, r_loss)
        null_values[start:stop] = first.mean(axis=1) - second.mean(axis=1)
    return null_values


def _two_sample_permutation_null(
    l_loss: np.ndarray,
    r_loss: np.ndarray,
    n_permutations: int,
    random_state: int | None,
) -> np.ndarray:
    """Null values from permuting the pooled losses of two samples.

    Each null value is the difference between the mean of the first
    `len(l_loss)` pooled losses and the mean of the rest, after a random
    permutation of the pooled losses.
    """
    rng = np.random.default_rng(random_state)
    pooled_loss = np.concatenate([l_loss, r_loss]).astype(float)
    size = len(l_loss)
    null_values = np.empty(n_permutations)
    for start, stop in _blocks(n_permutations, len(pooled_loss)):
        permuted = rng.permuted(np.tile(pooled_loss, (stop - start, 1)), axis=1)
        null_values[start:stop] = permuted[:, :size].mean(axis=1) - permuted[
            :, size:
        ].mean(axis=1)
    return null_values
