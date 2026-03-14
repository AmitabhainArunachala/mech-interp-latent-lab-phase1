from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.persistent_patching_classification import classify_output


def test_numeric_markdown_answer_is_not_malformed() -> None:
    text = (
        "1. Calculate `2^3`: ```markdown\n"
        "2^3 = 2 x 2 x 2 = 8\n"
        "Step 2: write 8 in the answer slot and compare with 3^2 = 9.\n```"
    )
    assert classify_output(text, 0.80) != "MALFORMED"


def test_verbose_arithmetic_answer_is_not_malformed() -> None:
    text = (
        'In the arithmetic problem 3 + 5, we have added digits "3" and "5," '
        'resulting in a sum of two digits: "1" (carried over to the tens place) '
        'and "8." The carryover was not combined with any other digits at this '
        "step as there were no higher place values involved."
    )
    assert classify_output(text, 0.80) != "MALFORMED"


def test_token_salad_remains_malformed() -> None:
    text = "%%% 123 ### ::: ??? 456 // \\\\ @@@ 789 [] [] []"
    assert classify_output(text, 0.80) == "MALFORMED"


def test_repetitive_patch_style_output_remains_repetitive() -> None:
    text = "To To To To To To To Sure To To To To To To To To To To To To"
    assert classify_output(text, 0.75) == "REPETITIVE"
