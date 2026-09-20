"""Checks for the logic E3's verdict depends on.

The ablation itself needs checkpoints; these cover the variant set, the frozen
acceptance bands, and the claim that removing a block leaves every surviving
weight untouched -- which is the whole reason the ablation is valid without
retraining.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from phase2.scripts.exp_e3_refinement_ablation import (  # noqa: E402
    DILATIONS, N_BLOCKS, ablate, band_for, variants,
)
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402


def test_variant_set_is_the_frozen_one():
    plan = variants()
    assert plan["full"] == tuple(range(N_BLOCKS))
    assert len(plan) == 1 + N_BLOCKS + N_BLOCKS + 1
    # "keep first 5" and "drop block 5" must be the same network -- the run
    # gates on their scores agreeing, which is only meaningful if the plans do.
    assert plan["keep first 5"] == plan["drop block 5 (dilation {})".format(DILATIONS[5])]
    assert plan["drop ladder (blocks 1,2,3)"] == (0, 4, 5)


def test_bands_use_the_frozen_thresholds():
    assert band_for(0.10, 1.0) == "NEGLIGIBLE"      # exactly on the boundary
    assert band_for(0.11, 1.0) == "TOLERABLE"
    assert band_for(0.10, 1.1) == "TOLERABLE"
    assert band_for(0.30, 3.0) == "TOLERABLE"
    assert band_for(0.31, 3.0) == "COSTLY"
    assert band_for(0.30, 3.1) == "COSTLY"


def test_ablation_keeps_the_surviving_weights_bit_identical():
    model = StereoNet(StereoNetConfig(cost_volume_shift="left")).eval()
    keep = (0, 4, 5)
    reduced = ablate(model, keep)
    assert len(reduced.refinement.blocks) == len(keep)
    for position, index in enumerate(keep):
        original = model.refinement.blocks[index].conv1.weight
        survivor = reduced.refinement.blocks[position].conv1.weight
        assert torch.equal(original, survivor)
    # The stage around the blocks must be untouched too.
    assert torch.equal(model.refinement.input_conv.weight,
                       reduced.refinement.input_conv.weight)
    assert torch.equal(model.refinement.output_conv.weight,
                       reduced.refinement.output_conv.weight)


def test_ablation_does_not_mutate_the_original_model():
    model = StereoNet(StereoNetConfig(cost_volume_shift="left")).eval()
    ablate(model, (0, 1))
    assert len(model.refinement.blocks) == N_BLOCKS


def test_an_ablated_refinement_still_runs_and_keeps_its_shape():
    model = StereoNet(StereoNetConfig(cost_volume_shift="left")).eval()
    reduced = ablate(model, (0, 5))
    disparity = torch.zeros(1, 1, 32, 64)
    guidance = torch.zeros(1, 3, 32, 64)
    with torch.no_grad():
        residual = reduced.refinement(disparity, guidance)
    assert residual.shape == disparity.shape
    assert torch.isfinite(residual).all()
