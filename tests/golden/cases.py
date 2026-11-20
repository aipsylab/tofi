"""Fixed inference scenarios whose outputs are locked in `golden.json`.

Only closed-form models are used: `LinearRegression` and a small
nearest-centroid classifier defined here. scikit-learn's own classifiers
(even `LinearDiscriminantAnalysis`) give slightly different probabilities
across versions, which would make the locked numbers version-dependent.

Regenerate the golden file (only after an intended change of results)::

    python -m tests.golden.generate
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, RepeatedKFold, ShuffleSplit


class CentroidClassifier(ClassifierMixin, BaseEstimator):
    """Softmax of negative squared distances to the class centroids."""

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        dist = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        logits = -0.5 * dist
        logits -= logits.max(axis=1, keepdims=True)
        proba = np.exp(logits)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


TASKS = ["reg", "bin", "multi"]
LEARNERS = ["holdout", "kfold", "repeated", "shuffle"]
VARIANTS = [
    # (test, method, aux, removal, keyword arguments)
    ("CIT", "RPT", "plain", 0, {"n_copies": 40}),
    ("CIT", "CPI", "plain", 4, {"null_dist": "permutation", "n_permutations": 300}),
    ("CIT", "HRT", "none", 2, {"n_copies": 25}),
    ("CIT", "CPI", "plain", 1, {}),
    ("RIT", "LOCO", "plain", 0, {}),
    ("RIT", "LOCO", "plain", 1, {"null_dist": "permutation", "n_permutations": 300}),
    ("RIT", "PIE", "plain", 0, {}),
    ("RIT", "LOCO", "plain", "group", {"perturb_size": 0.05}),
    ("RIT", "PIE", "plain", 3, {"null_dist": "permutation", "n_permutations": 300}),
]
SUMMARIZE_OPTIONS = [
    (None, None),
    (False, False),
    (True, False),
    (False, True),
    (True, True),
]


def case_names() -> list[str]:
    """48 cases: every task x learner, cycling through the variants."""
    names = []
    for k, (task, learner) in enumerate(itertools.product(TASKS, LEARNERS)):
        for j in range(4):
            variant = (4 * k + j) % len(VARIANTS)
            names.append(f"{task}-{learner}-v{variant}")
    return names


def _data(task: str, seed: int, frame: bool):
    rng = np.random.default_rng(seed)
    n = 90
    X = rng.normal(size=(n, 5))
    X[:, 4] = (X[:, 4] > 0).astype(float)
    eta = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.6 * X[:, 4]
    if task == "reg":
        y = eta + rng.normal(size=n)
    elif task == "bin":
        y = (eta + rng.logistic(size=n) > 0).astype(int)
    else:
        y = np.digitize(eta + rng.logistic(size=n), [-0.8, 0.8])
    if frame:
        X = pd.DataFrame(X, columns=list("abcde"))
        y = pd.Series(y, name="y")
    return X, y


def _cv(kind: str):
    return {
        "kfold": KFold(4, shuffle=True, random_state=0),
        "repeated": RepeatedKFold(n_splits=3, n_repeats=2, random_state=1),
        "shuffle": ShuffleSplit(3, test_size=0.3, random_state=2),
    }[kind]


def build(name: str, tofi):
    """Return (inferer, X_test, y_test) for case `name` using package `tofi`."""
    task, learner_kind, variant = name.split("-")
    test, method, aux_kind, removal, kws = VARIANTS[int(variant[1:])]
    seed = sum(ord(ch) for ch in name)
    frame = seed % 2 == 0
    X, y = _data(task, seed, frame)

    def model(classifier: bool):
        return CentroidClassifier() if classifier else LinearRegression()

    if removal == "group":
        removal = ["a", "d"] if frame else [0, 3]
    elif frame:
        removal = "abcde"[removal]

    def drop(data):
        if isinstance(data, pd.DataFrame):
            return data.drop(removal, axis=1), data[removal]
        return np.delete(data, removal, axis=1), data[:, removal]

    holdout = learner_kind == "holdout"
    if holdout:
        half = len(y) // 2
        X_fit, X_test = X[:half], X[half:]
        y_fit, y_test = y[:half], y[half:]
        learner = model(task != "reg").fit(X_fit, y_fit)
    else:
        X_fit, y_fit, X_test, y_test = X, y, X, y
        learner = tofi.Crosser(model(task != "reg"), _cv(learner_kind)).fit(X, y)

    def fit_aux(target, classifier):
        X_removed = drop(X_fit)[0]
        if holdout:
            return model(classifier).fit(X_removed, target)
        return tofi.Crosser(model(classifier), _cv(learner_kind)).fit(X_removed, target)

    kws = dict(kws, random_state=seed % 1000)
    if test == "CIT":
        binary_feature = removal in (4, "e")
        sampler = None
        if aux_kind == "plain":
            sampler = fit_aux(drop(X_fit)[1], binary_feature)
        inferer = tofi.CIT(learner, sampler, removal, method, **kws)
    else:
        competitor = fit_aux(y_fit, task != "reg")
        inferer = tofi.RIT(learner, competitor, removal, method, **kws)
    return inferer, X_test, y_test


def outputs(name: str, tofi) -> dict:
    """All numbers produced by case `name`, as JSON-serialisable lists."""
    inferer, X_test, y_test = build(name, tofi)
    inferer.infer(X_test, y_test)
    result = {
        "l_losses": [np.asarray(v, dtype=float).tolist() for v in inferer.l_losses_],
        "r_losses": [np.asarray(v, dtype=float).tolist() for v in inferer.r_losses_],
        "summaries": {},
    }
    for cross_fit, combine in SUMMARIZE_OPTIONS:
        key = f"cross_fit={cross_fit},combine={combine}"
        try:
            summary = inferer.summarize(
                cross_fit=cross_fit, combine=combine, verbose=False
            )
        except Exception as error:  # noqa: BLE001 - record failures too
            result["summaries"][key] = {"error": type(error).__name__}
            continue
        result["summaries"][key] = {
            "index": [list(map(str, np.atleast_1d(i))) for i in summary.index],
            "values": summary.to_numpy(dtype=float).tolist(),
        }
    return result
