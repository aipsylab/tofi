"""Tests of `CIT` / `RIT`: validation, losses, equivalences and summaries."""

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, RepeatedKFold

from tofi import CIT, RIT, Crosser, LeakageWarning, NotInferredError
from tofi.inferer import _LOSSES
from tofi.utils import InvalidParameterError

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _reg(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = 1.5 * X[:, 0] - X[:, 1] + rng.normal(size=n)
    return X, y


def _clf(n=150, seed=0, n_classes=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    eta = 1.5 * X[:, 0] - X[:, 1] + rng.logistic(size=n)
    cuts = [0.0] if n_classes == 2 else [-1.0, 1.0]
    return X, np.digitize(eta, cuts)


def _holdout(X, y, model):
    half = len(y) // 2
    learner = model().fit(X[:half], y[:half])
    return learner, X[:half], y[:half], X[half:], y[half:]


@pytest.fixture
def holdout_reg():
    X, y = _reg()
    learner, X_fit, y_fit, X_test, y_test = _holdout(X, y, LinearRegression)
    competitor = LinearRegression().fit(np.delete(X_fit, 0, 1), y_fit)
    sampler = LinearRegression().fit(np.delete(X_fit, 0, 1), X_fit[:, 0])
    return learner, sampler, competitor, X_test, y_test


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"removal": None},
        {"removal": 1.5},
        {"method": 3},
        {"method": "LOCO"},
        {"removal": [0, 1]},
        {"loss_func": "hinge"},
        {"null_dist": "bootstrap"},
        {"n_copies": 2.0},
        {"n_permutations": "100"},
        {"random_state": np.random.RandomState(0)},
        {"random_state": 1.0},
    ],
)
def test_cit_invalid_arguments(kwargs, holdout_reg):
    learner, sampler, _, _, _ = holdout_reg
    args = {"removal": 0, "method": "RPT"} | kwargs
    with pytest.raises(InvalidParameterError):
        CIT(learner, sampler, args.pop("removal"), args.pop("method"), **args)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "RPT"},
        {"removal": [0, 1.5]},
        {"double_split": "yes"},
        {"perturb_size": "0.1"},
        {"null_dist": "resampling"},
    ],
)
def test_rit_invalid_arguments(kwargs, holdout_reg):
    learner, _, competitor, _, _ = holdout_reg
    args = {"removal": 0, "method": "LOCO"} | kwargs
    with pytest.raises(InvalidParameterError):
        RIT(learner, competitor, args.pop("removal"), args.pop("method"), **args)


def test_rit_requires_competitor(holdout_reg):
    learner = holdout_reg[0]
    with pytest.raises(InvalidParameterError):
        RIT(learner, None, 0)


def test_invalid_arguments_are_still_value_errors(holdout_reg):
    # tofi 0.0.1 raised ValueError; existing `except ValueError` must work.
    learner, sampler, _, _, _ = holdout_reg
    with pytest.raises(ValueError):
        CIT(learner, sampler, 0, "LOCO")
    with pytest.raises(TypeError):
        CIT(learner, sampler, 0, "RPT", n_copies="10")


def test_crosser_auxiliary_requires_crosser_learner(holdout_reg):
    learner, _, _, _, _ = holdout_reg
    X, y = _reg()
    aux = Crosser(LinearRegression(), KFold(3)).fit(np.delete(X, 0, 1), y)
    with pytest.raises(InvalidParameterError):
        RIT(learner, aux, 0)
    with pytest.raises(InvalidParameterError):
        CIT(learner, aux, 0)


def test_summarize_before_infer(holdout_reg):
    learner, sampler, _, _, _ = holdout_reg
    with pytest.raises(NotInferredError):
        CIT(learner, sampler, 0, "CPI").summarize()


def test_arguments_are_not_modified(holdout_reg):
    learner, sampler, competitor, _, _ = holdout_reg
    cit = CIT(learner, sampler, 0)
    assert (cit.method, cit.loss_func, cit.null_dist, cit.n_copies) == (
        None,
        None,
        None,
        None,
    )
    assert (cit.method_, cit.loss_func_, cit.null_dist_, cit.n_copies_) == (
        "RPT",
        "mean_squared_error",
        "resampling",
        2000,
    )
    rit = RIT(learner, competitor, 0, "PIE", random_state=0)
    assert rit.double_split is None and rit.double_split_ is True
    assert rit.null_dist is None and rit.null_dist_ == "normality"


@pytest.mark.parametrize(
    "method, null_dist, n_copies, n_permutations",
    [
        ("HRT", "resampling", 2000, None),
        ("RPT", "resampling", 2000, None),
        ("CPI", "normality", 1, None),
    ],
)
def test_cit_defaults(method, null_dist, n_copies, n_permutations, holdout_reg):
    learner, sampler, _, _, _ = holdout_reg
    cit = CIT(learner, sampler, 0, method)
    assert cit.null_dist_ == null_dist
    assert cit.n_copies_ == n_copies
    assert cit.n_permutations_ == n_permutations


def test_permutation_defaults(holdout_reg):
    learner, sampler, competitor, _, _ = holdout_reg
    cit = CIT(learner, sampler, 0, "CPI", null_dist="permutation")
    assert cit.n_copies_ == 1 and cit.n_permutations_ == 2000
    rit = RIT(learner, competitor, 0, null_dist="permutation")
    assert rit.n_permutations_ == 2000


def test_n_permutations_ignored_with_warning(holdout_reg):
    learner, sampler, competitor, _, _ = holdout_reg
    with pytest.warns(UserWarning, match="n_permutations"):
        cit = CIT(learner, sampler, 0, "RPT", n_permutations=10)
    assert cit.n_permutations == 10 and cit.n_permutations_ is None
    with pytest.warns(UserWarning, match="n_permutations"):
        rit = RIT(learner, competitor, 0, n_permutations=10)
    assert rit.n_permutations_ is None


def test_leakage_warning_points_to_caller():
    X, y = _reg()
    learner = Crosser(LinearRegression(), KFold(3)).fit(X, y)
    competitor = LinearRegression().fit(np.delete(X, 0, 1), y)
    with pytest.warns(LeakageWarning) as record:
        RIT(learner, competitor, 0)
    assert record[0].filename == __file__


# ---------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------


def test_regression_losses_match_sklearn():
    rng = np.random.default_rng(0)
    y, pred = rng.normal(size=50), rng.normal(size=50)
    mse = _LOSSES["mean_squared_error"][0]
    mae = _LOSSES["mean_absolute_error"][0]
    assert mse(y, pred).mean() == pytest.approx(metrics.mean_squared_error(y, pred))
    assert mae(y, pred).mean() == pytest.approx(metrics.mean_absolute_error(y, pred))


def test_zero_one_loss_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=60)
    pred = rng.integers(0, 3, size=60)
    loss = _LOSSES["zero_one_loss"][0]
    assert loss(y, pred).mean() == pytest.approx(metrics.zero_one_loss(y, pred))


@pytest.mark.parametrize("n_classes", [2, 3])
def test_log_loss_matches_sklearn_in_inference(n_classes):
    X, y = _clf(n_classes=n_classes)
    learner, X_fit, y_fit, X_test, y_test = _holdout(X, y, LinearDiscriminantAnalysis)
    competitor = LinearDiscriminantAnalysis().fit(np.delete(X_fit, 0, 1), y_fit)
    rit = RIT(learner, competitor, 0).infer(X_test, y_test)
    assert rit.l_losses_[0].mean() == pytest.approx(
        metrics.log_loss(y_test, learner.predict_proba(X_test))
    )
    assert rit.r_losses_[0].mean() == pytest.approx(
        metrics.log_loss(y_test, competitor.predict_proba(np.delete(X_test, 0, 1)))
    )


def test_log_loss_with_crosser_matches_sklearn():
    X, y = _clf(n_classes=3)
    cv = KFold(3, shuffle=True, random_state=0)
    learner = Crosser(LinearDiscriminantAnalysis(), cv).fit(X, y)
    competitor = Crosser(LinearDiscriminantAnalysis(), cv).fit(np.delete(X, 0, 1), y)
    rit = RIT(learner, competitor, 0).infer(X, y)
    for split, test in enumerate(learner._test_indices(X, y)):
        proba = learner.estimators_[split].predict_proba(X[test])
        assert rit.l_losses_[split].mean() == pytest.approx(
            metrics.log_loss(y[test], proba, labels=[0, 1, 2])
        )


# ---------------------------------------------------------------------
# Equivalences
# ---------------------------------------------------------------------


def _summaries(inferer, X, y):
    inferer.infer(X, y)
    return inferer.summarize(verbose=False)


@pytest.mark.parametrize("test", ["CIT", "RIT"])
def test_dataframe_and_array_inputs_agree(test):
    X, y = _reg()
    names = list("abcd")
    frame = pd.DataFrame(X, columns=names)
    cv = KFold(3, shuffle=True, random_state=0)

    def run(X_in, y_in, removal):
        learner = Crosser(LinearRegression(), cv).fit(X_in, y_in)
        if isinstance(X_in, pd.DataFrame):
            X_removed, x_j = X_in.drop(removal, axis=1), X_in[removal]
        else:
            X_removed, x_j = np.delete(X_in, removal, 1), X_in[:, removal]
        if test == "CIT":
            sampler = Crosser(LinearRegression(), cv).fit(X_removed, x_j)
            inferer = CIT(learner, sampler, removal, n_copies=30, random_state=0)
        else:
            competitor = Crosser(LinearRegression(), cv).fit(X_removed, y_in)
            inferer = RIT(learner, competitor, removal)
        return _summaries(inferer, X_in, y_in)

    a = run(X, y, 1)
    b = run(frame, pd.Series(y), "b")
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


def test_series_and_array_targets_agree():
    X, y = _clf(n_classes=3)
    learner, X_fit, y_fit, X_test, y_test = _holdout(X, y, LinearDiscriminantAnalysis)
    competitor = LinearDiscriminantAnalysis().fit(np.delete(X_fit, 2, 1), y_fit)
    a = _summaries(RIT(learner, competitor, 2), X_test, y_test)
    b = _summaries(RIT(learner, competitor, 2), X_test, pd.Series(y_test))
    pd.testing.assert_frame_equal(a, b)


def test_string_labels_agree_with_integer_labels():
    X, y = _clf(n_classes=3)
    labels = np.array(["a", "b", "c"])[y]
    out = []
    for target in (y, labels):
        learner, X_fit, y_fit, X_test, y_test = _holdout(
            X, target, LinearDiscriminantAnalysis
        )
        competitor = LinearDiscriminantAnalysis().fit(np.delete(X_fit, 0, 1), y_fit)
        out.append(_summaries(RIT(learner, competitor, 0), X_test, y_test))
    np.testing.assert_allclose(out[0].to_numpy(), out[1].to_numpy())


@pytest.mark.parametrize("test", ["CIT", "RIT"])
def test_infer_does_not_modify_inputs(test, holdout_reg):
    learner, sampler, competitor, X_test, y_test = holdout_reg
    X_before, y_before = X_test.copy(), y_test.copy()
    if test == "CIT":
        CIT(learner, sampler, 0, n_copies=20, random_state=0).infer(X_test, y_test)
    else:
        RIT(learner, competitor, 0).infer(X_test, y_test)
    np.testing.assert_array_equal(X_test, X_before)
    np.testing.assert_array_equal(y_test, y_before)


def test_infer_returns_self(holdout_reg):
    learner, sampler, competitor, X_test, y_test = holdout_reg
    cit = CIT(learner, sampler, 0, "CPI", random_state=0)
    assert cit.infer(X_test, y_test) is cit
    rit = RIT(learner, competitor, 0)
    assert rit.infer(X_test, y_test) is rit


def test_custom_sample_method_is_used(holdout_reg):
    learner, _, _, X_test, y_test = holdout_reg

    class ConstantSampler:
        def sample(self, X, y, n_copies=None, random_state=None):
            return np.zeros((n_copies, len(X)))

    cit = CIT(learner, ConstantSampler(), 0, "HRT", n_copies=3).infer(X_test, y_test)
    X_zero = X_test.copy()
    X_zero[:, 0] = 0.0
    expected = (y_test - learner.predict(X_zero)) ** 2
    np.testing.assert_allclose(cit.r_losses_[0], expected)


def test_uninformative_feature_is_not_significant():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 3))
    y = X[:, 0] + rng.normal(size=400)
    learner, X_fit, y_fit, X_test, y_test = _holdout(X, y, LinearRegression)
    sampler = LinearRegression().fit(np.delete(X_fit, 2, 1), X_fit[:, 2])
    competitor = LinearRegression().fit(np.delete(X_fit, 2, 1), y_fit)
    rpt = CIT(learner, sampler, 2, n_copies=300, random_state=0)
    loco = RIT(learner, competitor, 2)
    assert _summaries(rpt, X_test, y_test)["p_value"].iloc[0] > 0.05
    assert _summaries(loco, X_test, y_test)["p_value"].iloc[0] > 0.05
    sampler0 = LinearRegression().fit(np.delete(X_fit, 0, 1), X_fit[:, 0])
    rpt0 = CIT(learner, sampler0, 0, n_copies=300, random_state=0)
    assert _summaries(rpt0, X_test, y_test)["p_value"].iloc[0] < 0.01


def test_type_one_error_of_loco_is_controlled():
    # 200 data sets in which x_2 is irrelevant: the rejection rate at
    # alpha = 0.1 should be close to (and not far above) 0.1.
    rejections = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(200, 3))
        y = X[:, 0] + rng.normal(size=200)
        learner, X_fit, y_fit, X_test, y_test = _holdout(X, y, LinearRegression)
        competitor = LinearRegression().fit(np.delete(X_fit, 2, 1), y_fit)
        p = _summaries(RIT(learner, competitor, 2), X_test, y_test)["p_value"].iloc[0]
        rejections += p < 0.1
    assert rejections / 200 < 0.16


# ---------------------------------------------------------------------
# Summaries computed by hand
# ---------------------------------------------------------------------


def test_loco_summary_by_hand(holdout_reg):
    learner, _, competitor, X_test, y_test = holdout_reg
    summary = _summaries(RIT(learner, competitor, 0), X_test, y_test)
    l_loss = (y_test - learner.predict(X_test)) ** 2
    r_loss = (y_test - competitor.predict(np.delete(X_test, 0, 1))) ** 2
    estimate = r_loss.mean() - l_loss.mean()
    std_error = np.sqrt((r_loss - l_loss).var() / len(l_loss))
    row = summary.iloc[0]
    assert row["size"] == len(y_test)
    assert row["estimate"] == pytest.approx(estimate)
    assert row["std_error"] == pytest.approx(std_error)
    assert row["p_value"] == pytest.approx(scipy.stats.norm.sf(estimate / std_error))


def test_pie_summary_by_hand(holdout_reg):
    learner, _, competitor, X_test, y_test = holdout_reg
    pie = RIT(learner, competitor, 0, "PIE", random_state=0).infer(X_test, y_test)
    l_loss, r_loss = pie.l_losses_[0], pie.r_losses_[0]
    assert len(l_loss) == int(np.ceil(len(y_test) / 2))
    assert len(l_loss) + len(r_loss) == len(y_test)
    # the halves are disjoint subsets of the full losses
    full_l = (y_test - learner.predict(X_test)) ** 2
    full_r = (y_test - competitor.predict(np.delete(X_test, 0, 1))) ** 2
    assert set(np.round(l_loss, 12)) <= set(np.round(full_l, 12))
    assert set(np.round(r_loss, 12)) <= set(np.round(full_r, 12))
    row = pie.summarize(verbose=False).iloc[0]
    std_error = np.sqrt(r_loss.var() / len(r_loss) + l_loss.var() / len(l_loss))
    assert row["std_error"] == pytest.approx(std_error)


def test_perturbation_adds_noise_only_to_competitor(holdout_reg):
    learner, _, competitor, X_test, y_test = holdout_reg
    plain = RIT(learner, competitor, 0).infer(X_test, y_test)
    noisy = RIT(learner, competitor, 0, perturb_size=0.5, random_state=0).infer(
        X_test, y_test
    )
    np.testing.assert_array_equal(plain.l_losses_[0], noisy.l_losses_[0])
    noise = noisy.r_losses_[0] - plain.r_losses_[0]
    assert noise.std() == pytest.approx(0.5, rel=0.25)


def test_resampling_p_value_by_hand(holdout_reg):
    learner, sampler, _, X_test, y_test = holdout_reg
    rpt = CIT(learner, sampler, 0, "RPT", n_copies=50, random_state=0)
    rpt.infer(X_test, y_test)
    row = rpt.summarize(verbose=False).iloc[0]
    null = rpt.null_values_[0]
    assert len(null) == 50
    assert row["p_value"] == pytest.approx((null > row["estimate"]).mean())
    assert row["std_error"] == pytest.approx(null.std())


def test_cross_fit_by_hand():
    X, y = _reg(n=150)
    cv = RepeatedKFold(n_splits=3, n_repeats=2, random_state=0)
    learner = Crosser(LinearRegression(), cv).fit(X, y)
    competitor = Crosser(LinearRegression(), cv).fit(np.delete(X, 1, 1), y)
    rit = RIT(learner, competitor, 1).infer(X, y)
    split = rit.summarize(cross_fit=False, combine=False, verbose=False)
    fitted = rit.summarize(cross_fit=True, combine=False, verbose=False)
    for repeat in range(2):
        rows = split.xs(repeat, level="repeat")
        size = rows["size"].sum()
        estimate = rows["estimate"].mean()
        within = np.mean(rows["std_error"] ** 2)
        between = rows["estimate"].var(ddof=0) / np.round(size / 3)
        std_error = np.sqrt((within + between) / 3)
        got = fitted.xs(repeat, level="repeat").iloc[0]
        assert got["size"] == size
        assert got["estimate"] == pytest.approx(estimate)
        assert got["std_error"] == pytest.approx(std_error)
        assert got["p_value"] == pytest.approx(
            scipy.stats.norm.sf(estimate / std_error)
        )


def test_combined_p_values_by_hand():
    X, y = _reg(n=150)
    cv = RepeatedKFold(n_splits=3, n_repeats=4, random_state=0)
    learner = Crosser(LinearRegression(), cv).fit(X, y)
    competitor = Crosser(LinearRegression(), cv).fit(np.delete(X, 2, 1), y)
    rit = RIT(learner, competitor, 2).infer(X, y)
    p = rit.summarize(cross_fit=True, combine=False, verbose=False)["p_value"]
    p = p.to_numpy()
    m = len(p)
    combined = rit.summarize(verbose=False).xs(2, level="removal")["p_value"]
    harmonic = sum(1 / k for k in range(1, m + 1))
    expected = {
        "gmean": np.e * scipy.stats.gmean(p),
        "median": 2 * np.median(p),
        "min": m * p.min(),
        "hmean": np.e * np.log(m) * scipy.stats.hmean(p),
        "hommel": min(harmonic * m / k * pk for k, pk in enumerate(np.sort(p), 1)),
        "cauchy": 0.5 - np.arctan(np.mean(np.tan((0.5 - p) * np.pi))) / np.pi,
    }
    assert list(combined.index) == list(expected)
    for name, value in expected.items():
        assert combined[name] == pytest.approx(min(value, 1)), name


def test_reverse_flips_estimate_only(holdout_reg):
    learner, _, competitor, X_test, y_test = holdout_reg
    rit = RIT(learner, competitor, 0).infer(X_test, y_test)
    a = rit.summarize(verbose=False)
    b = rit.summarize(reverse=True, verbose=False)
    np.testing.assert_allclose(a["estimate"], -b["estimate"])
    np.testing.assert_allclose(a["p_value"], b["p_value"])


def test_group_removal_label(holdout_reg):
    X, y = _reg()
    frame = pd.DataFrame(X, columns=list("abcd"))
    half = 60
    learner = LinearRegression().fit(frame[:half], y[:half])
    competitor = LinearRegression().fit(frame[:half].drop(["a", "c"], axis=1), y[:half])
    summary = _summaries(RIT(learner, competitor, ["a", "c"]), frame[half:], y[half:])
    assert list(summary.index) == ["a + c"]


def test_verbose_text(holdout_reg, capsys):
    learner, _, competitor, X_test, y_test = holdout_reg
    RIT(learner, competitor, 0).infer(X_test, y_test).summarize()
    assert capsys.readouterr().out == (
        "Inferer Summary (cross_fit=False, combine=False)\n"
        " + Method: LOCO (double_split=False, perturb_size=None)\n"
        " + Null Distribution: Normality (n_copies=None, n_permutations=None)\n"
        " + Loss Function: Mean Squared Error (reverse=False)\n"
    )


def test_no_unexpected_warnings_in_normal_use(holdout_reg):
    learner, sampler, competitor, X_test, y_test = holdout_reg
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CIT(learner, sampler, 0, n_copies=20, random_state=0).infer(
            X_test, y_test
        ).summarize(verbose=False)
        RIT(learner, competitor, 0).infer(X_test, y_test).summarize(verbose=False)
