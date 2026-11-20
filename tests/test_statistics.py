"""Statistical properties of the null distributions and summaries.

The permutation nulls are checked against their exact moments and
against a straightforward (loop-based) reference implementation.
"""

import numpy as np
import pytest
import scipy.stats

from tofi.utils import (
    _paired_permutation_null,
    _sample,
    _two_sample_permutation_null,
)


def _reference_paired(l_loss, r_loss, n_permutations, seed):
    rng = np.random.default_rng(seed)
    paired = np.column_stack([l_loss, r_loss])
    out = []
    for _ in range(n_permutations):
        permuted = rng.permuted(paired, axis=1)
        out.append(permuted[:, 0].mean() - permuted[:, 1].mean())
    return np.array(out)


def _reference_two_sample(l_loss, r_loss, n_permutations, seed):
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([l_loss, r_loss])
    size = len(l_loss)
    out = []
    for _ in range(n_permutations):
        permuted = rng.permuted(pooled)
        out.append(permuted[:size].mean() - permuted[size:].mean())
    return np.array(out)


@pytest.mark.parametrize("n", [5, 37, 400])
def test_paired_null_moments(n):
    rng = np.random.default_rng(n)
    l_loss = rng.gamma(2.0, size=n)
    r_loss = l_loss + rng.normal(0.3, 1.0, size=n)
    null = _paired_permutation_null(l_loss, r_loss, 40000, random_state=1)
    diff = l_loss - r_loss
    # sign-flip mean: E = 0, Var = sum(d^2) / n^2
    sd = np.sqrt(np.sum(diff**2)) / n
    assert abs(null.mean()) < 4 * sd / np.sqrt(len(null))
    assert null.std() == pytest.approx(sd, rel=0.02)


@pytest.mark.parametrize("sizes", [(3, 4), (20, 19), (150, 150)])
def test_two_sample_null_moments(sizes):
    a, b = sizes
    rng = np.random.default_rng(a + b)
    l_loss = rng.gamma(2.0, size=a)
    r_loss = rng.gamma(2.5, size=b)
    null = _two_sample_permutation_null(l_loss, r_loss, 40000, random_state=2)
    pooled = np.concatenate([l_loss, r_loss])
    big_n = a + b
    sd = np.sqrt(pooled.var() * big_n / (big_n - 1) * (1 / a + 1 / b))
    assert abs(null.mean()) < 4 * sd / np.sqrt(len(null))
    assert null.std() == pytest.approx(sd, rel=0.02)


def test_two_sample_null_is_exact_for_tiny_samples():
    # With 2 + 2 observations all 6 splits are equally likely.
    l_loss, r_loss = np.array([0.0, 1.0]), np.array([3.0, 7.0])
    null = _two_sample_permutation_null(l_loss, r_loss, 60000, random_state=3)
    pooled = np.concatenate([l_loss, r_loss])
    exact = []
    for first in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        mask = np.zeros(4, bool)
        mask[list(first)] = True
        exact.append(pooled[mask].mean() - pooled[~mask].mean())
    values, counts = np.unique(np.round(null, 12), return_counts=True)
    np.testing.assert_allclose(np.sort(values), np.unique(np.round(exact, 12)))
    np.testing.assert_allclose(counts / counts.sum(), 1 / len(values), atol=0.01)


@pytest.mark.parametrize("seed", range(4))
def test_paired_null_matches_reference_distribution(seed):
    rng = np.random.default_rng(100 + seed)
    l_loss = rng.gamma(2.0, size=60)
    r_loss = l_loss + rng.normal(0.1, 0.5, size=60)
    new = _paired_permutation_null(l_loss, r_loss, 5000, random_state=seed)
    ref = _reference_paired(l_loss, r_loss, 5000, seed + 1000)
    assert scipy.stats.ks_2samp(new, ref).pvalue > 1e-3


@pytest.mark.parametrize("seed", range(4))
def test_two_sample_null_matches_reference_distribution(seed):
    rng = np.random.default_rng(200 + seed)
    l_loss = rng.gamma(2.0, size=31)
    r_loss = rng.gamma(2.2, size=30)
    new = _two_sample_permutation_null(l_loss, r_loss, 5000, random_state=seed)
    ref = _reference_two_sample(l_loss, r_loss, 5000, seed + 1000)
    assert scipy.stats.ks_2samp(new, ref).pvalue > 1e-3


def test_paired_null_keeps_exact_ties_for_discrete_losses():
    # zero-one losses: ties between null values and the estimate must be
    # recognised exactly (they decide `null > estimate`).
    rng = np.random.default_rng(7)
    l_loss = (rng.random(50) < 0.3).astype(int)
    r_loss = (rng.random(50) < 0.4).astype(int)
    estimate = r_loss.mean() - l_loss.mean()
    null = _paired_permutation_null(l_loss, r_loss, 20000, random_state=0)
    ref = _reference_paired(l_loss, r_loss, 20000, 1)
    assert np.mean(null == estimate) > 0.01
    assert np.mean(null == estimate) == pytest.approx(
        np.mean(ref == estimate), abs=0.01
    )
    assert np.mean(null > estimate) == pytest.approx(np.mean(ref > estimate), abs=0.015)


def test_null_is_reproducible_and_seed_dependent():
    rng = np.random.default_rng(0)
    l_loss, r_loss = rng.random(30), rng.random(30)
    a = _paired_permutation_null(l_loss, r_loss, 100, random_state=5)
    b = _paired_permutation_null(l_loss, r_loss, 100, random_state=5)
    c = _paired_permutation_null(l_loss, r_loss, 100, random_state=6)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_null_blocks_do_not_change_distribution(monkeypatch):
    # Small blocks (many chunks) must give the same distribution.
    import tofi.utils as utils

    rng = np.random.default_rng(1)
    l_loss, r_loss = rng.gamma(2, size=40), rng.gamma(2.3, size=40)
    whole = utils._paired_permutation_null(l_loss, r_loss, 4000, 0)
    monkeypatch.setattr(utils, "_BLOCK_ELEMENTS", 50)
    chunked = utils._paired_permutation_null(l_loss, r_loss, 4000, 0)
    assert len(chunked) == 4000
    assert scipy.stats.ks_2samp(whole, chunked).pvalue > 1e-3


# ---------------------------------------------------------------------
# Resampling of x_j
# ---------------------------------------------------------------------


def test_marginal_permutation_keeps_values():
    y = np.arange(20)
    draws = _sample(None, None, y, n_copies=5, random_state=0)
    assert draws.shape == (5, 20)
    for draw in draws:
        np.testing.assert_array_equal(np.sort(draw), y)


def test_residual_permutation_keeps_residuals():
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = X @ [1.0, -2.0] + rng.normal(size=50)
    model = LinearRegression().fit(X, y)
    draws = _sample(model, X, y, n_copies=3, random_state=0)
    residual = y - model.predict(X)
    for draw in draws:
        np.testing.assert_allclose(
            np.sort(draw - model.predict(X)), np.sort(residual), atol=1e-12
        )


def test_classifier_draws_follow_probabilities():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = np.where(X[:, 0] + rng.logistic(size=200) > 0, "yes", "no")
    model = LinearDiscriminantAnalysis().fit(X, y)
    draws = _sample(model, X, y, n_copies=4000, random_state=0)
    assert set(np.unique(draws)) <= {"yes", "no"}
    freq_yes = (draws == "yes").mean(axis=0)
    expected = model.predict_proba(X)[:, list(model.classes_).index("yes")]
    np.testing.assert_allclose(freq_yes, expected, atol=0.05)
