# Changelog

## 0.1.0

### Bug fixes

- `zero_one_loss` returned 1 for *correct* predictions, so tests based on it were reversed. It now returns 1 for errors.
- `CIT` evaluated the resampled data with `predict` instead of `predict_proba`, giving wrong losses for classifiers with `log_loss`.
- `CIT` with a plain (non-`Crosser`) binary classifier compared a one-column target with two-column probabilities under `log_loss`.
- `Crosser` failed with scikit-learn >= 1.8 because `_estimator_type` was removed. It now implements `__sklearn_tags__` (and keeps `_estimator_type` for older versions).
- Installing from GitHub failed because `pyproject.toml` had no `[build-system]` and package configuration.
- Two separately created but identical cross-validators (e.g. `KFold(5, shuffle=True, random_state=0)`) were rejected as different.

### Behaviour changes

- `RIT` now defaults to `method="LOCO"` (previously documented as `"PIE"`), following Huang (2025). `method` is now optional for both `CIT` (default `"RPT"`) and `RIT`.
- PIE's double splitting now divides each test set into two random halves instead of odd/even rows. A `UserWarning` is issued when double splitting is used with `random_state=None`, because the results then cannot be replicated.
- Each fold and each source of randomness (sampling, permutation, perturbation, double splitting) now receives an independent seed derived from `random_state`. Results for a given `random_state` therefore differ from 0.0.1.
- The combined p-value `"q1"` (the second smallest p-value times $m/2$) is removed from the `summarize(combine=True)` output.
- `RIT` with `null_dist="resampling"` now raises an error instead of warning and failing later.
- `Crosser` rejects a `cv` whose `random_state` is a `RandomState` instance, which would produce different folds on each call.
- `infer` returns `self`.
- The arguments of `CIT`, `RIT` and `Crosser` are no longer overwritten with their defaults. The settings actually used are available as `method_`, `loss_func_`, `null_dist_`, `double_split_`, `n_copies_` and `n_permutations_` (e.g. `RIT(...).method` is now `None` when not given, and `RIT(...).method_` is `"LOCO"`).
- Invalid arguments raise `InvalidParameterError`, which subclasses both `ValueError` (as in 0.0.1) and `TypeError`.
- Permutation null distributions are generated in vectorized blocks. They have the same distribution as before, but a given `random_state` yields different permutations (and hence slightly different permutation p-values) than earlier development versions.

### Other changes

- `LeakageWarning` when `learner` is a `Crosser` but `sampler`/`competitor` is an ordinary estimator.
- `Crosser` is now a scikit-learn `BaseEstimator` (supports `get_params`, `clone` and `repr`).
- `Crosser.predict` (majority vote over splits) now works with non-integer class labels.
- Internal refactoring: one code path for `Crosser` and ordinary estimators, shared null-distribution helpers, consistent formatting (`ruff`).
- Tests (176): unit tests, statistical checks of the null distributions against their exact moments, and regression tests that lock the numerical results of 48 scenarios (`tests/golden`). GitHub Actions CI with linting, and an API documentation site.

## 0.0.1

- Initial release.
