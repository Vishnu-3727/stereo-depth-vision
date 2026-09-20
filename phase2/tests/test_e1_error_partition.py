"""Checks for the logic E1's verdicts depend on.

The oracles themselves need GPU and checkpoints; these cover the pooling, the
verdict rule and the fit/eval split, where a silent error would change what the
experiment reports.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from phase2.scripts.exp_e1_error_partition import Pool, verdict_for  # noqa: E402
from phase2.scripts.exp_e1_heldout_oracle import split_masks  # noqa: E402


class FakeScene:
    def __init__(self, valid):
        self.gt_valid = valid


def test_pool_matches_the_kitti_d1_definition():
    gt = np.array([[10.0, 100.0, 10.0, 10.0]])
    #                exact   err 4 px (< 5 % of 100)   err 4 px (> 5 %)   err 2 px
    pred = np.array([[10.0, 104.0, 14.0, 12.0]])
    pool = Pool()
    pool.add(pred, gt, np.ones_like(gt, dtype=bool))
    result = pool.result()
    assert result["pixels"] == 4
    # Only the third pixel is an outlier: > 3 px AND > 5 % of its ground truth.
    assert result["d1"] == 25.0
    assert result["epe"] == (0.0 + 4.0 + 4.0 + 2.0) / 4


def test_pool_accumulates_across_scenes():
    a, b = Pool(), Pool()
    gt = np.array([[10.0, 10.0]])
    for pool, pred in ((a, np.array([[10.0, 20.0]])), (b, np.array([[10.0, 20.0]]))):
        pool.add(pred, gt, np.ones_like(gt, dtype=bool))
    combined = Pool()
    for _ in range(2):
        combined.add(np.array([[10.0, 20.0]]), gt, np.ones_like(gt, dtype=bool))
    assert combined.result()["pixels"] == 4
    assert combined.result()["epe"] == a.result()["epe"]


def test_verdict_rule_uses_the_frozen_thresholds():
    def scores(oracle_epe):
        return {"as trained": {"epe": 2.0, "d1": 20.0},
                "initial := a*gt+b where gt>0": {"epe": oracle_epe, "d1": 10.0}}
    assert verdict_for(scores(0.70))["verdict"] == "MATCHING-LIMITED"   # ratio 0.35
    assert verdict_for(scores(1.30))["verdict"] == "DOWNSTREAM-LIMITED"  # ratio 0.65
    assert verdict_for(scores(1.00))["verdict"] == "MIXED"               # ratio 0.50


def test_fit_and_eval_halves_are_disjoint_and_cover_the_valid_pixels():
    valid = np.zeros((16, 16), dtype=bool)
    valid[::2, ::3] = True
    fit, ev = split_masks(FakeScene(valid))
    assert not np.any(fit & ev), "a pixel may never steer the oracle and judge it"
    assert np.array_equal(fit | ev, valid)
    assert not np.any(fit & ~valid) and not np.any(ev & ~valid)
    # Both halves must be non-trivial, or the held-out measurement is meaningless.
    assert fit.sum() > 0 and ev.sum() > 0


def test_the_split_is_deterministic():
    valid = np.ones((8, 8), dtype=bool)
    first, _ = split_masks(FakeScene(valid))
    second, _ = split_masks(FakeScene(valid))
    assert np.array_equal(first, second)
