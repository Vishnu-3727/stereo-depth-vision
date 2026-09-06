"""Regression protection for the EXP-016 false-conclusion defect.

EXP-016 recorded "validation improves" while its own validation EPE rose from
18.497 px at epoch 0 to 19.216 px at epoch 19. The sentence was a literal string
in `scripts/exp_train_convergence.py`, so it did not depend on the run at all
and re-running the script would have recreated the false claim.

`experiments/EXP-016/CORRECTION.md` withdrew the interpretation. These tests
stop the defect returning: the conclusion must now be derived from the measured
values, and no wording path may assert improvement that the numbers do not show.
"""

import ast
from pathlib import Path

import pytest

from src.common.conclusions import (
    classify_series_trend,
    classify_validation_change,
    demo,
    describe_training_outcome,
    describe_validation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "exp_train_convergence.py"

# The values EXP-016 actually produced.
EXP016_INITIAL_EPE = 18.497
EXP016_FINAL_EPE = 19.216
EXP016_BEST_EPE = 17.622  # epoch 15


def test_conclusions_self_check():
    demo()


# -- the three cases the audit requires ------------------------------------

def test_case_a_exp016_values_did_not_improve():
    """The exact regression: EXP-016's own numbers must not read as improvement."""
    assert (
        classify_validation_change(EXP016_INITIAL_EPE, EXP016_FINAL_EPE)
        == "did not improve"
    )


def test_case_b_lower_final_is_an_improvement():
    assert classify_validation_change(19.216, 18.497) == "improved"


def test_case_c_equal_values_are_unchanged():
    assert classify_validation_change(18.497, 18.497) == "unchanged"


# -- the classifier cannot be fooled ---------------------------------------

@pytest.mark.parametrize(
    "initial,final,expected",
    [
        (10.0, 5.0, "improved"),
        (5.0, 10.0, "did not improve"),
        (5.0, 5.0, "unchanged"),
        (0.0, 0.0, "unchanged"),
        (1e-9, 2e-9, "did not improve"),
    ],
)
def test_classifier_follows_the_numbers(initial, final, expected):
    assert classify_validation_change(initial, final) == expected


def test_metric_direction_is_explicit_not_assumed():
    """Silently assuming lower-is-better is the same class of mistake."""
    assert classify_validation_change(0.80, 0.90, lower_is_better=False) == "improved"
    assert (
        classify_validation_change(0.90, 0.80, lower_is_better=False)
        == "did not improve"
    )


def test_tolerance_must_be_chosen_by_the_caller():
    # default is exact comparison
    assert classify_validation_change(10.0, 10.0001) == "did not improve"
    # a band of indifference only applies when asked for
    assert classify_validation_change(10.0, 10.0001, tolerance=0.001) == "unchanged"
    with pytest.raises(ValueError):
        classify_validation_change(1.0, 2.0, tolerance=-0.1)


def test_series_trend_uses_first_and_last():
    """EXP-016's full series dips in the middle and ends worse. The verdict
    follows the ending, which is the criterion CORRECTION.md applies."""
    series = [18.497, 22.564, 18.827, 17.622, 19.216]
    assert classify_series_trend(series) == "did not improve"
    with pytest.raises(ValueError):
        classify_series_trend([1.0])


# -- the wording cannot contradict the data --------------------------------

def test_description_never_claims_improvement_on_exp016_values():
    text = describe_validation(
        EXP016_INITIAL_EPE, EXP016_FINAL_EPE,
        best=EXP016_BEST_EPE, best_label="epoch 15",
    )
    assert "did not improve" in text
    assert "so validation improved" not in text


def test_a_good_middle_epoch_does_not_become_an_improvement():
    """The trap that produced the original error: a better intermediate value
    is not evidence that training improved validation."""
    text = describe_validation(
        EXP016_INITIAL_EPE, EXP016_FINAL_EPE,
        best=EXP016_BEST_EPE, best_label="epoch 15",
    )
    assert "17.622" in text
    assert "does not establish improvement" in text


def test_training_outcome_reports_exp016_faithfully():
    out = describe_training_outcome(
        losses=[11.4937, 7.6137],
        grad_norms=[42.5, 38.4, 814.2],
        val_initial=EXP016_INITIAL_EPE,
        val_final=EXP016_FINAL_EPE,
        val_best=EXP016_BEST_EPE,
        val_best_label="epoch 15",
        checkpoint_written=True,
    )
    assert "loss decreased" in out
    assert "did not improve" in out
    assert "so validation improved" not in out


def test_training_outcome_reports_a_diverging_run_as_diverging():
    out = describe_training_outcome(
        losses=[1.0, 5.0],
        grad_norms=[1.0, float("nan")],
        val_initial=1.0,
        val_final=2.0,
    )
    assert "loss increased" in out
    assert "non-finite" in out
    assert "did not improve" in out


def test_training_outcome_handles_a_run_with_no_validation():
    out = describe_training_outcome(losses=[2.0, 1.0], grad_norms=[1.0])
    assert "No validation measurements were recorded." in out
    assert "improve" not in out


# -- the script itself must not re-introduce a literal claim ---------------

def test_training_script_contains_no_hardcoded_outcome_claim():
    """Guards the original defect directly: the script must not assert an
    outcome in a string literal. Comments are allowed, since the file documents
    the history of the bug."""
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = (
        "validation improves",
        "validation improved",
        "the loss decreases",
        "no non-finite values",
    )
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            low = node.value.lower()
            for phrase in banned:
                if phrase in low:
                    offenders.append((getattr(node, "lineno", "?"), phrase))
    assert not offenders, (
        "hard-coded outcome claim(s) found in string literals: " + repr(offenders)
    )


def test_training_script_derives_its_conclusion():
    """The conclusion must be built by the conclusions module, not written out."""
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert "from src.common.conclusions import" in source
    assert "describe_training_outcome(" in source
    assert "classify_validation_change(" in source


def test_exp016_correction_is_preserved():
    """The integrity trail must stay: the original record, and its withdrawal."""
    exp = REPO_ROOT / "experiments" / "EXP-016"
    correction = exp / "CORRECTION.md"
    assert (exp / "metrics.json").exists(), "original EXP-016 record must be kept"
    assert correction.exists(), "the correction must not be removed"
    text = correction.read_text(encoding="utf-8")
    assert "validation did not improve" in text.lower()
    assert "18.497" in text and "19.216" in text
