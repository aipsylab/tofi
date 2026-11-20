"""Conditional independence tests (CIT) and risk invariance tests (RIT)."""

from __future__ import annotations

import math
import warnings
from itertools import repeat
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy.stats
from scipy.special import xlogy
from sklearn.base import is_classifier

from .crosser import Crosser, _split_index
from .utils import (
    InvalidParameterError,
    LeakageWarning,
    NotInferredError,
    _check_option,
    _check_type,
    _child_seeds,
    _encode_target,
    _is_int,
    _paired_permutation_null,
    _same_cv,
    _sample,
    _split_columns,
    _take_rows,
    _two_sample_permutation_null,
    _values,
)

# ---------------------------------------------------------------------
# Loss functions (per-observation losses)
# ---------------------------------------------------------------------


def _log_loss(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    eps = np.finfo(pred.dtype).eps
    pred = np.clip(pred, eps, 1 - eps)
    return -xlogy(target, pred).sum(axis=1)


def _zero_one_loss(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return 1 * (target != pred)


def _mean_squared_error(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return (target - pred) ** 2


def _mean_absolute_error(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.abs(target - pred)


# name -> (function, one-hot encode targets?, prediction method)
_LOSSES: dict[str, tuple[Callable, bool, str]] = {
    "mean_squared_error": (_mean_squared_error, False, "predict"),
    "mean_absolute_error": (_mean_absolute_error, False, "predict"),
    "zero_one_loss": (_zero_one_loss, False, "predict"),
    "log_loss": (_log_loss, True, "predict_proba"),
}

_NULL_DISTS = ["resampling", "permutation", "normality"]
_CIT_METHODS = ["HRT", "RPT", "CPI"]
_RIT_METHODS = ["LOCO", "PIE"]

# ---------------------------------------------------------------------
# One interface for `Crosser` and ordinary (holdout) estimators: an
# ordinary estimator is treated as a crosser with a single split whose
# test set is all of the given data.
# ---------------------------------------------------------------------


def _test_indices(learner: Any, X: Any, y: Any) -> list[np.ndarray | None]:
    if isinstance(learner, Crosser):
        return learner._test_indices(X, y)
    return [None]


def _model_for_split(model: Any, split: int) -> Any:
    if isinstance(model, Crosser):
        return model.estimators_[split]
    return model


class Inferer:
    """
    An inferer for making statistical inference

    Parameters
    ----------
    learner : estimator or crosser object
        The (fitted) learner being inferred.

    sampler : estimator or crosser object
        A sampler that predicts $x_j$ by $x_{-j}$. When `sampler` is
        a crosser, its cross-validator (splitter) must be identical
        to the learner. If `sampler` doesn't have a sample method,
        the residual permutation approach will be used for resampling.

    competitor : estimator or crosser object
        A competitor that predicts $y$ by $x_{-j}$. If competitor is
        a crosser, its cross-validator (splitter) must be identical
        to the learner.

    removal : int, str, or list
        A label for removing $x_j$.
            - When `learner` is trained with numpy array, `removal`
              must be column index(es) of $x_j$.
            - When `learner` is trained with `DataFrame`, `removal`
              must be column name(s) of $x_j$.

    method : {"HRT", "RPT", "CPI", "LOCO", "PIE"}
        The method for making inference. It provides a control
        for implementing a specific procedure.

    loss_func : {"mean_squared_error", "mean_absolute_error",
                "zero_one_loss", "log_loss"}, default=None
        The loss function for measuring the difference between
        $y$ and its prediction.

            - When `learner` is a regressor, `None` means
              "mean_squared_error".
            - When `learner` is a classifier, `None` means
              "log_loss".

    null_dist : {"resampling", "normality", "permutation"}, \
                 default=None
        The way for constructing a null distribution of test statistic.

            - When `method` is "HRT", "RPT", `None` means "resampling".
            - When `method` is "CPI", "LOCO" or "PIE", `None` means
              "normality".

    double_split : bool, default=None
        An indicator for double splitting. If applicable,

            - When `method` is "LOCO", `None` means `False`
            - When `method` is "PIE", `None` means `True`.

        With double splitting, each test set is randomly divided into
        two halves; the learner's loss is evaluated on one half and
        the competitor's loss on the other.

    perturb_size : non-negative float, default=None
        The standard deviation of random error for pertubation.
        If applicable, `None` means 0.

    n_copies : int, default=None
        The number of copies for resampling $x_j$ given $x_{-j}$.
        If applicable,

            - When `method` is "HRT", "RPT", `None` means 2000.
            - When `method` is "CPI", `None` means 1.

    n_permutations : int, default=None
        The number of permutations. If applicable, `None` means 2000.

    random_state : int or None, default=None
        Controls the randomness of sampling, permutation, perturbation
        and double splitting. Independent random streams are derived
        from it for every split (fold) and every source of randomness.
        Unlike `random_state` in scikit-learn, `RandomState` is not
        allowed here.

    Attributes
    ----------
    method_, loss_func_, null_dist_, double_split_, n_copies_, \
    n_permutations_ :
        The settings actually used, i.e., the corresponding arguments
        with `None` replaced by their defaults. The arguments themselves
        are kept unchanged.

    l_losses_ : list of ndarray
        Per-observation losses of `learner`, one array per split.

    r_losses_ : list of ndarray
        Per-observation losses after removing $x_j$ (resampled $x_j$ for
        CITs, `competitor` for RITs), one array per split.

    null_values_ : list of ndarray or None
        Null distribution of the test statistic for each split
        (`None` when `null_dist` is "normality").
    """

    def __init__(
        self,
        learner: Any,
        sampler: Any = None,
        competitor: Any = None,
        removal: int | str | list | None = None,
        method: str | None = None,
        loss_func: str | None = None,
        null_dist: str | None = None,
        double_split: bool | None = None,
        perturb_size: float | None = None,
        n_copies: int | None = None,
        n_permutations: int | None = None,
        random_state: int | None = None,
    ):
        self.learner = learner
        self.sampler = sampler
        self.competitor = competitor
        self.removal = removal
        self.method = method
        self.loss_func = loss_func
        self.null_dist = null_dist
        self.double_split = double_split
        self.perturb_size = perturb_size
        self.n_copies = n_copies
        self.n_permutations = n_permutations
        self.random_state = random_state

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def summarize(
        self,
        cross_fit: bool | None = None,
        combine: bool | None = None,
        reverse: bool = False,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Summarize inference results

        Parameters
        ----------

        cross_fit : bool, default=None
            If `True`, inference results will be integrated across folds
            by cross-fitting strategy. It is applicable only if `learner`
            is a `Crosser` with `n_folds` > 1. If applicable, `None` means
            `True`; Otherwise, `None` means `False`.

        combine : bool, default=None
            If `True`, inference results will be combined across
            folds and repeats. For `estimates` and `std_error`, simple
            average is used for combination. For `p_value`, the
            following p-value combination methods are reported (with
            $m$ p-values and order statistics $p_{(1)} \\le ... \\le
            p_{(m)}$):

                - "gmean": $e$ times the geometric mean.
                - "median": twice the median.
                - "min": $m p_{(1)}$ (Bonferroni).
                - "hmean": $e \\log(m)$ times the harmonic mean.
                - "hommel": Hommel's combination.
                - "cauchy": Cauchy combination test.

            It is applicable only if `learner` is a `Crosser` with
            `n_splits` > 1. If `n_repeats` > 1, `None` means `True`;
            Otherwise, `None` means `False`.

        reverse : bool, default=False
            If `True`, negative values of `estimate` will be reported.

        verbose : bool, default=True
            Controls the verbosity.

        Returns
        -------
        summary : DataFrame
            A summary for inference results.

        """
        if not self._is_inferred:
            raise NotInferredError("This inferer is not inferred yet.")
        is_crosser = self._is_crosser

        if combine is None:
            combine = is_crosser and self._n_repeats > 1
        if cross_fit is None:
            cross_fit = is_crosser and self._n_folds > 1

        sizes, estimates, std_errors, p_values = self._stats()
        if cross_fit:
            sizes, estimates, std_errors, p_values = self._cross_fit(
                sizes, estimates, std_errors
            )
        if reverse is True:
            estimates = -estimates

        removal = self.removal
        if isinstance(removal, list):
            removal = " + ".join(str(removal_i) for removal_i in removal)

        columns = {
            "size": sizes,
            "estimate": estimates,
            "std_error": std_errors,
            "p_value": p_values,
        }
        if (not is_crosser) or (len(p_values) == 1):
            index = pd.Index([removal], name="removal")
        elif not combine:
            if cross_fit:
                index = pd.MultiIndex.from_tuples(
                    list(zip(repeat(removal), range(self._n_repeats))),
                    names=["removal", "repeat"],
                )
            else:
                splits = _split_index(self._n_splits, self._n_folds)
                index = pd.MultiIndex.from_tuples(
                    [(removal, *split) for split in splits],
                    names=["removal", "split", "repeat", "fold"],
                )
        else:
            combined = self._combined_p_values(p_values)
            n_rows = len(combined)
            index = pd.MultiIndex.from_tuples(
                list(zip(repeat(removal), combined.keys())),
                names=["removal", "method"],
            )
            columns = {
                "size": [np.mean(sizes)] * n_rows,
                "estimate": [np.mean(estimates)] * n_rows,
                "std_error": [np.mean(std_errors)] * n_rows,
                "p_value": list(combined.values()),
            }
        summary = pd.DataFrame(columns, index=index)

        if verbose:
            self._print_settings(cross_fit, combine, reverse)
        return summary

    def _print_settings(self, cross_fit: bool, combine: bool, reverse: bool) -> None:
        print(f"Inferer Summary (cross_fit={cross_fit}, combine={combine})")
        print(
            f" + Method: {self.method_} "
            f"(double_split={self.double_split_}, perturb_size={self.perturb_size})"
        )
        print(
            f" + Null Distribution: {self.null_dist_.title()} "
            f"(n_copies={self.n_copies_}, n_permutations={self.n_permutations_})"
        )
        loss_name = self.loss_func_.replace("_", " ").title()
        print(f" + Loss Function: {loss_name} (reverse={reverse})")

    def _stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Size, estimate, standard error and p-value of every split."""
        l_losses = self.l_losses_
        r_losses = self.r_losses_
        null_values = self.null_values_

        if self.double_split_:
            sizes = np.array(
                [
                    len(l_loss) + len(r_loss)
                    for l_loss, r_loss in zip(l_losses, r_losses)
                ]
            )
        else:
            sizes = np.array([len(l_loss) for l_loss in l_losses])

        estimates = np.array(
            [
                r_loss.mean() - l_loss.mean()
                for l_loss, r_loss in zip(l_losses, r_losses)
            ]
        )

        if self.null_dist_ in ("permutation", "resampling"):
            std_errors = np.array([null_value.std() for null_value in null_values])
            p_values = np.array(
                [
                    (null_value > estimate).mean()
                    for estimate, null_value in zip(estimates, null_values)
                ]
            )
        else:
            if self.double_split_:
                std_errors = np.array(
                    [
                        np.sqrt(r_loss.var() / len(r_loss) + l_loss.var() / len(l_loss))
                        for l_loss, r_loss in zip(l_losses, r_losses)
                    ]
                )
            else:
                std_errors = np.array(
                    [
                        np.sqrt((r_loss - l_loss).var() / len(l_loss))
                        for l_loss, r_loss in zip(l_losses, r_losses)
                    ]
                )
            p_values = _normal_p_values(estimates, std_errors)
        return sizes, estimates, std_errors, p_values

    def _cross_fit(
        self, sizes: np.ndarray, estimates: np.ndarray, std_errors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Integrate the split-level results over the folds of each repeat."""
        shape = (self._n_repeats, self._n_folds)
        n_folds = self._n_folds

        sizes = sizes.reshape(shape).sum(axis=1)
        if self.double_split_:
            std_errors = std_errors.reshape(shape).mean(axis=1) * np.sqrt(1 / n_folds)
        else:
            within = np.square(std_errors.reshape(shape)).mean(axis=1)
            between = estimates.reshape(shape).var(axis=1) / np.round(sizes / n_folds)
            std_errors = np.sqrt((within + between) / n_folds)
        estimates = estimates.reshape(shape).mean(axis=1)

        if self.null_dist_ in ("permutation", "resampling"):
            null_values = (
                np.array(self.null_values_)
                .reshape((self._n_repeats, n_folds, -1))
                .mean(axis=1)
            )
            p_values = np.array(
                [
                    (null_value > estimate).mean()
                    for estimate, null_value in zip(estimates, null_values)
                ]
            )
        else:
            p_values = _normal_p_values(estimates, std_errors)
        return sizes, estimates, std_errors, p_values

    def _combined_p_values(self, p_values: np.ndarray) -> dict[str, float]:
        p_values = np.asarray(p_values)
        m = len(p_values)
        const = np.sum(1 / (np.arange(m) + 1))
        order_const = const * (m / (np.arange(m) + 1))
        t0 = np.mean(np.tan((0.5 - p_values) * np.pi))

        combined = {
            "gmean": np.e * scipy.stats.gmean(p_values, 0),
            "median": 2 * np.median(p_values, 0),
            "min": m * np.min(p_values, 0),
            "hmean": np.e * np.log(m) * scipy.stats.hmean(p_values, 0),
            "hommel": np.min(np.sort(p_values) * order_const),
            "cauchy": 0.5 - np.arctan(t0) / np.pi,
        }
        return {key: np.minimum(value, 1) for key, value in combined.items()}

    # -----------------------------------------------------------------
    # Shared pieces of `infer`
    # -----------------------------------------------------------------

    def _learner_loss(
        self, split: int, index: np.ndarray | None, X: Any, y: Any
    ) -> tuple[Any, Any, np.ndarray]:
        """Test features, encoded target and losses of `learner`."""
        features = _take_rows(X, index)
        target = _encode_target(
            self.learner, _take_rows(_values(y), index), self._binarize
        )
        model = _model_for_split(self.learner, split)
        pred = getattr(model, self._response_method)(features)
        return features, target, self._loss(target, pred)

    # -----------------------------------------------------------------
    # Validation and defaults
    # -----------------------------------------------------------------

    def _check(self) -> None:
        if self.learner is None:
            raise InvalidParameterError("Argument `learner` must be given.")
        if self.removal is None:
            raise InvalidParameterError("Argument `removal` must be given.")
        if not (_is_int(self.removal) or isinstance(self.removal, (str, list))):
            raise InvalidParameterError(
                "`removal` must be of type int, str or list; "
                f"got {type(self.removal).__name__}."
            )
        _check_type("method", self.method, "str")
        _check_option("loss_func", self.loss_func, list(_LOSSES))
        _check_option("null_dist", self.null_dist, _NULL_DISTS)
        _check_type("double_split", self.double_split, "bool")
        _check_type("perturb_size", self.perturb_size, "number")
        _check_type("n_copies", self.n_copies, "int")
        _check_type("n_permutations", self.n_permutations, "int")
        _check_type("random_state", self.random_state, "int")

    def _check_auxiliary(self, name: str) -> None:
        """Validate a sampler/competitor against the learner."""
        auxiliary = getattr(self, name)
        if isinstance(self.learner, Crosser):
            if isinstance(auxiliary, Crosser):
                if not _same_cv(self.learner.cv, auxiliary.cv):
                    raise InvalidParameterError(
                        f"When `learner` and `{name}` are both `Crosser`, "
                        "they must share the same cross-validator."
                    )
            elif auxiliary is not None:
                warnings.warn(
                    f"`learner` is a `Crosser` but `{name}` is an ordinary "
                    f"estimator. If `{name}` was fitted on data overlapping "
                    "the test folds of `learner` (e.g., the full data), "
                    f"test information leaks into `{name}` and the test is "
                    "invalid. Use a `Crosser` with the same `cv` for "
                    f"`{name}` instead.",
                    LeakageWarning,
                    stacklevel=4,
                )
        elif isinstance(auxiliary, Crosser):
            raise InvalidParameterError(
                f"When `learner` is not `Crosser`, `{name}` must be usual `Estimator`."
            )

    def _setup(self) -> None:
        loss_func = self.loss_func
        if loss_func is None:
            loss_func = (
                "log_loss" if is_classifier(self.learner) else "mean_squared_error"
            )
        self.loss_func_ = loss_func
        self._loss, self._binarize, self._response_method = _LOSSES[loss_func]

        self._is_inferred = False
        self._is_crosser = isinstance(self.learner, Crosser)
        if self._is_crosser:
            self._n_splits = self.learner._n_splits
            self._n_repeats = self.learner._n_repeats
            self._n_folds = self.learner._n_folds
        else:
            self._n_splits = self._n_repeats = self._n_folds = 1


class CIT(Inferer):
    """
    Conditional independence tests (HRT, RPT, CPI)

    Parameters
    ----------
    learner : estimator or crosser object
        The fitted learner being inferred.

    sampler : estimator, crosser object, or None
        A fitted model of $x_j$ given $x_{-j}$. `None` means $x_j$ is
        permuted marginally. If `learner` is a `Crosser`, `sampler`
        should be a `Crosser` with the same `cv`; otherwise a
        `LeakageWarning` is issued.

    removal : int or str
        Column index or name of $x_j$.

    method : {"HRT", "RPT", "CPI"}, default="RPT"
        The testing procedure.

    See `Inferer` for the remaining keyword arguments and attributes.
    """

    def __init__(
        self,
        learner: Any,
        sampler: Any,
        removal: int | str,
        method: str | None = None,
        *,
        loss_func: str | None = None,
        null_dist: str | None = None,
        n_copies: int | None = None,
        n_permutations: int | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            learner=learner,
            sampler=sampler,
            removal=removal,
            method=method,
            loss_func=loss_func,
            null_dist=null_dist,
            n_copies=n_copies,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        self._check()
        self._setup()

    def infer(self, X: Any, y: Any) -> CIT:
        """
        Conduct the test on (test) data `X` and `y`

        When `learner` is a `Crosser`, `X` and `y` must be the full data
        used to fit it; the test folds are extracted internally.

        Returns
        -------
        self : object
        """
        X, y = X.copy(), y.copy()
        X_removed, x_removed = _split_columns(X, self.removal)
        test_indices = _test_indices(self.learner, X, y)
        n_splits = len(test_indices)
        seeds = _child_seeds(self.random_state, 2 * n_splits)
        sample_seeds, permutation_seeds = seeds[:n_splits], seeds[n_splits:]

        l_losses, r_losses, null_values = [], [], []
        for split, index in enumerate(test_indices):
            features, target, l_loss = self._learner_loss(split, index, X, y)
            draws = self._draw(
                split,
                _take_rows(X_removed, index),
                _take_rows(x_removed, index),
                sample_seeds[split],
            )
            # losses with x_j replaced by each draw: (n_draws, n_obs)
            r_loss = self._resampled_losses(split, features.copy(), target, draws)

            if self.null_dist_ == "resampling":
                null_values.append(r_loss.mean() - r_loss.mean(axis=1))
            r_loss = r_loss.mean(axis=0)
            if self.null_dist_ == "permutation":
                null_values.append(
                    _paired_permutation_null(
                        l_loss, r_loss, self.n_permutations_, permutation_seeds[split]
                    )
                )
            l_losses.append(l_loss)
            r_losses.append(r_loss)

        self.l_losses_ = l_losses
        self.r_losses_ = r_losses
        self.null_values_ = null_values if self.null_dist_ != "normality" else None
        self._is_inferred = True
        return self

    def _draw(
        self, split: int, features: Any, feature: Any, seed: int | None
    ) -> np.ndarray:
        """Draws of x_j given x_{-j} on one test set: (n_draws, n_obs)."""
        sampler = self.sampler
        model = None if sampler is None else _model_for_split(sampler, split)
        if hasattr(model, "sample"):
            draws = model.sample(
                features, feature, n_copies=self.n_copies_, random_state=seed
            )
        else:
            draws = _sample(
                model, features, feature, n_copies=self.n_copies_, random_state=seed
            )
        draws = np.asarray(draws)
        return draws[np.newaxis, :] if draws.ndim == 1 else draws

    def _resampled_losses(
        self, split: int, features: Any, target: Any, draws: np.ndarray
    ) -> np.ndarray:
        model = _model_for_split(self.learner, split)
        predict = getattr(model, self._response_method)
        removal = self.removal
        losses = []
        for draw in draws:
            if isinstance(features, pd.DataFrame):
                features.loc[:, removal] = draw
            else:
                features[:, removal] = draw
            losses.append(self._loss(target, predict(features)))
        return np.array(losses)

    def _check(self) -> None:
        super()._check()
        self._check_auxiliary("sampler")
        if isinstance(self.removal, list):
            raise InvalidParameterError(
                "When implementing `CIT`, `removal` must be `int` or `str`."
            )
        _check_option("method", self.method, _CIT_METHODS)

    def _setup(self) -> None:
        super()._setup()
        method = self.method or "RPT"
        null_dist = self.null_dist
        n_copies = self.n_copies
        n_permutations = self.n_permutations

        if null_dist is None:
            null_dist = "resampling" if method in ("HRT", "RPT") else "normality"
        if null_dist == "resampling":
            if n_copies is None:
                n_copies = 2000
        else:
            if n_copies is None:
                n_copies = 1
            if null_dist == "permutation" and n_permutations is None:
                n_permutations = 2000
        if null_dist in ("normality", "resampling") and n_permutations is not None:
            warnings.warn(
                "When `null_dist` is 'normality' or 'resampling', "
                "`n_permutations` must be `None`.",
                UserWarning,
                stacklevel=3,
            )
            n_permutations = None

        self.method_ = method
        self.null_dist_ = null_dist
        self.double_split_ = None
        self.n_copies_ = n_copies
        self.n_permutations_ = n_permutations


class RIT(Inferer):
    """
    Risk invariance tests (LOCO, PIE)

    Parameters
    ----------
    learner : estimator or crosser object
        The fitted learner being inferred.

    competitor : estimator or crosser object
        A fitted model of $y$ given $x_{-j}$. If `learner` is a
        `Crosser`, `competitor` should be a `Crosser` with the same
        `cv`; otherwise a `LeakageWarning` is issued.

    removal : int, str, or list
        Column index(es) or name(s) of the removed feature(s).

    method : {"LOCO", "PIE"}, default="LOCO"
        The testing procedure. "LOCO" is the default, following the
        recommendation of Huang (2025); "PIE" (with double splitting)
        is available for comparison.

    See `Inferer` for the remaining keyword arguments and attributes.
    """

    def __init__(
        self,
        learner: Any,
        competitor: Any,
        removal: int | str | list,
        method: str | None = None,
        *,
        loss_func: str | None = None,
        null_dist: str | None = None,
        double_split: bool | None = None,
        perturb_size: float | None = None,
        n_permutations: int | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            learner=learner,
            competitor=competitor,
            removal=removal,
            method=method,
            loss_func=loss_func,
            null_dist=null_dist,
            double_split=double_split,
            perturb_size=perturb_size,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        self._check()
        self._setup()

    def infer(self, X: Any, y: Any) -> RIT:
        """
        Conduct the test on (test) data `X` and `y`

        When `learner` is a `Crosser`, `X` and `y` must be the full data
        used to fit it; the test folds are extracted internally.

        Returns
        -------
        self : object
        """
        X, y = X.copy(), y.copy()
        X_removed, _ = _split_columns(X, self.removal)
        test_indices = _test_indices(self.learner, X, y)
        n_splits = len(test_indices)
        seeds = _child_seeds(self.random_state, 3 * n_splits)
        split_seeds = seeds[:n_splits]
        perturb_seeds = seeds[n_splits : 2 * n_splits]
        permutation_seeds = seeds[2 * n_splits :]

        l_losses, r_losses, null_values = [], [], []
        for split, index in enumerate(test_indices):
            _, _, l_loss = self._learner_loss(split, index, X, y)
            r_loss = self._competitor_loss(split, index, X_removed, y)

            if self.double_split_:
                # Randomly halve the test set: the learner is evaluated
                # on one half and the competitor on the other.
                order = np.random.default_rng(split_seeds[split]).permutation(
                    len(l_loss)
                )
                half = math.ceil(len(l_loss) / 2)
                l_loss, r_loss = l_loss[order[:half]], r_loss[order[half:]]

            if self.perturb_size is not None:
                r_loss = r_loss + np.random.default_rng(perturb_seeds[split]).normal(
                    scale=self.perturb_size, size=len(r_loss)
                )

            if self.null_dist_ == "permutation":
                null = (
                    _two_sample_permutation_null
                    if self.double_split_
                    else _paired_permutation_null
                )
                null_values.append(
                    null(l_loss, r_loss, self.n_permutations_, permutation_seeds[split])
                )
            l_losses.append(l_loss)
            r_losses.append(r_loss)

        self.l_losses_ = l_losses
        self.r_losses_ = r_losses
        self.null_values_ = null_values if self.null_dist_ == "permutation" else None
        self._is_inferred = True
        return self

    def _competitor_loss(
        self, split: int, index: np.ndarray | None, X_removed: Any, y: Any
    ) -> np.ndarray:
        features = _take_rows(X_removed, index)
        target = _encode_target(
            self.competitor, _take_rows(_values(y), index), self._binarize
        )
        model = _model_for_split(self.competitor, split)
        pred = getattr(model, self._response_method)(features)
        return self._loss(target, pred)

    def _check(self) -> None:
        super()._check()
        if self.competitor is None:
            raise InvalidParameterError(
                "When implementing `RIT`, `competitor` must be given."
            )
        self._check_auxiliary("competitor")
        if isinstance(self.removal, list):
            for removal_i in self.removal:
                if not (_is_int(removal_i) or isinstance(removal_i, str)):
                    raise InvalidParameterError(
                        "When `removal` is `list`, its element must be `int` or `str`."
                    )
        _check_option("method", self.method, _RIT_METHODS)

    def _setup(self) -> None:
        super()._setup()
        method = self.method or "LOCO"
        null_dist = self.null_dist
        double_split = self.double_split
        n_permutations = self.n_permutations

        if null_dist is None:
            null_dist = "normality"
        elif null_dist == "permutation":
            if n_permutations is None:
                n_permutations = 2000
        elif null_dist == "resampling":
            raise InvalidParameterError(
                "When `method` is 'LOCO' or 'PIE', "
                "`null_dist` must be 'normality' or 'permutation'."
            )
        if double_split is None:
            double_split = method == "PIE"
        if double_split and self.random_state is None:
            warnings.warn(
                "Double splitting divides each test set into two random "
                "halves. With `random_state=None`, the split (and hence "
                "`estimate`, `std_error` and `p_value`) changes on every "
                "run, so the results cannot be replicated. Set "
                "`random_state` to an `int` for reproducible results.",
                UserWarning,
                stacklevel=3,
            )
        if null_dist == "normality" and n_permutations is not None:
            warnings.warn(
                "When `null_dist` is 'normality', `n_permutations` must be `None`.",
                UserWarning,
                stacklevel=3,
            )
            n_permutations = None

        self.method_ = method
        self.null_dist_ = null_dist
        self.double_split_ = double_split
        self.n_copies_ = None
        self.n_permutations_ = n_permutations


def _normal_p_values(estimates: np.ndarray, std_errors: np.ndarray) -> np.ndarray:
    """One-sided p-values from the normal approximation."""
    return np.array(
        [
            1 - scipy.stats.norm.cdf(estimate / std_error)
            for estimate, std_error in zip(estimates, std_errors)
        ]
    )
