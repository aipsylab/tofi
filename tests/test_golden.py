"""Regression tests: outputs must match the locked values in golden.json.

These guard refactorings against any change in numerical results. If a
change of results is intended, regenerate the file with
`python -m tests.golden.generate` and explain the change in CHANGELOG.md.
"""

import json
import pathlib
import warnings

import numpy as np
import pytest

import tofi

from .golden.cases import case_names, outputs

GOLDEN = json.loads(
    (pathlib.Path(__file__).parent / "golden" / "golden.json").read_text()
)
RTOL = 1e-7
ATOL = 1e-10


def test_golden_covers_all_cases():
    assert sorted(GOLDEN) == sorted(case_names())


@pytest.mark.parametrize("name", case_names())
def test_matches_golden(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = outputs(name, tofi)
    expected = GOLDEN[name]

    for key in ("l_losses", "r_losses"):
        assert len(result[key]) == len(expected[key])
        for got, want in zip(result[key], expected[key]):
            np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)

    assert result["summaries"].keys() == expected["summaries"].keys()
    for option, want in expected["summaries"].items():
        got = result["summaries"][option]
        if "error" in want:
            assert got == want, option
            continue
        assert got["index"] == want["index"], option
        np.testing.assert_allclose(
            got["values"], want["values"], rtol=RTOL, atol=ATOL, err_msg=option
        )
