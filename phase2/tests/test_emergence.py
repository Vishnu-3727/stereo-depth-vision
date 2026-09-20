"""Checks for the pure logic of EXP-H2-EMERGENCE-001.

The probing itself needs GPU and checkpoints; these cover the two functions that
turn probe output into the verdict, which is where a silent error would change
the reported answer.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from phase2.scripts.exp_h2_emergence import (  # noqa: E402
    RIGHT_KEYS, emergence_interval, worst_penalty,
)


def ablation(*penalties):
    """One scene per penalty, all three right-image corruptions at that value."""
    return [{k: {"d1_penalty": p} for k in RIGHT_KEYS} for p in penalties]


def test_worst_penalty_is_the_minimum_over_scenes_and_corruptions():
    assert worst_penalty(ablation(80.0, -1.5, 40.0), RIGHT_KEYS) == -1.5


def test_emergence_reports_the_first_crossing_as_an_interval():
    rows = [{"epoch": e, "right_dependence_worst_pt": p} for e, p in
            [(10, -1.3), (20, 32.4), (50, 58.7), (100, 73.4)]]
    result = emergence_interval(rows)
    assert result == {"emerged": True, "first_epoch_meeting_criterion": 20,
                      "interval": [10, 20], "monotone_after": True}


def test_emergence_flags_a_later_epoch_falling_back_below_the_criterion():
    rows = [{"epoch": e, "right_dependence_worst_pt": p} for e, p in
            [(10, 25.0), (20, 5.0)]]
    assert emergence_interval(rows)["monotone_after"] is False


def test_no_crossing_reports_not_emerged():
    rows = [{"epoch": e, "right_dependence_worst_pt": 1.0} for e in (10, 20)]
    assert emergence_interval(rows) == {
        "emerged": False, "first_epoch_meeting_criterion": None,
        "interval": [20, None]}
