"""Regenerate `golden.json` (run from the repository root).

python -m tests.golden.generate
"""

import json
import pathlib
import warnings

import tofi

from .cases import case_names, outputs

PATH = pathlib.Path(__file__).with_name("golden.json")


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        golden = {name: outputs(name, tofi) for name in case_names()}
    PATH.write_text(json.dumps(golden, indent=0, sort_keys=True) + "\n")
    print(f"wrote {len(golden)} cases to {PATH}")


if __name__ == "__main__":
    main()
