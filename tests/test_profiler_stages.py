"""Regression protection for the EXP-017 stage-mapping defect.

EXP-017 mapped Hailo's compiled convolutions as conv1-17 feature extraction,
conv18-22 aggregation, conv23+ refinement. That folded the whole right
feature-extractor branch and the real 3D aggregation into refinement, and
reported refinement as 95.2 % of compiled MACs.

The shapes in the profiler report settle it: conv18 is 368x1232, 3 -> 32, 5x5 --
byte-for-byte the same workload as conv1, which makes it the Siamese twin, not
an aggregation layer. conv35-conv39 are the aggregation (25x79, 448 -> 384
channels, with conv39 emitting the 12 disparity costs), and conv40 is the first
refinement layer (368x1232 with 4 input channels: disparity plus left RGB).

These tests pin the boundaries and the defusion-name handling, and check the
corrected rollup against the layer data itself.
"""

import csv
from pathlib import Path

import pytest

from src.common.profiler_stages import (
    STAGE_AGGREGATION,
    STAGE_FEATURE_LEFT,
    STAGE_FEATURE_RIGHT,
    STAGE_REFINEMENT,
    conv_index,
    demo,
    stage_of,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LAYERS_CSV = REPO_ROOT / "results" / "profiler" / "layers.csv"


def test_stage_mapping_self_check():
    demo()


# -- the boundaries the audit named -----------------------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        ("conv1", STAGE_FEATURE_LEFT),
        ("conv17", STAGE_FEATURE_LEFT),
        ("conv18", STAGE_FEATURE_RIGHT),   # the twin of conv1, not aggregation
        ("conv34", STAGE_FEATURE_RIGHT),
        ("conv35", STAGE_AGGREGATION),     # first real 3D aggregation layer
        ("conv39", STAGE_AGGREGATION),
        ("conv40", STAGE_REFINEMENT),      # first refinement layer
        ("conv53", STAGE_REFINEMENT),
    ],
)
def test_conv_range_boundaries(name, expected):
    assert stage_of(name, "conv") == expected


def test_the_specific_misclassification_that_occurred():
    """conv18-conv22 were called aggregation and conv23-conv34 refinement.
    Both were the right feature-extractor branch."""
    for i in range(18, 35):
        assert stage_of("conv{}".format(i), "conv") == STAGE_FEATURE_RIGHT


# -- defused names keep their identity --------------------------------------

@pytest.mark.parametrize(
    "name,index",
    [
        ("stereonet/conv47_sd12", 47),
        ("conv42_sdc", 42),
        ("conv48_ws", 48),
        ("conv34_sd0", 34),
        ("stereonet/conv2", 2),
        ("conv17_concat1_transpose", 17),  # trailing token must not corrupt it
    ],
)
def test_conv_index_survives_defusion_suffixes(name, index):
    assert conv_index(name) == index


@pytest.mark.parametrize(
    "name",
    [
        "ws_from_conv47_ws_to_conv47_sd0-1",
        "sh_from_conv46_to_conv48_sd0-3",
        "mux_conv17_concat3_transpose_to_concat3",
        "concat12",
        "normalization1",
        "resize1",
    ],
)
def test_routing_layers_are_not_convolutions(name):
    assert conv_index(name) is None


def test_defused_refinement_layers_stay_in_refinement():
    for name in ("conv47_sd0", "conv48_sd21", "stereonet/conv53_sd6"):
        assert stage_of(name, "conv") == STAGE_REFINEMENT


# -- the mapping must agree with the layer shapes ---------------------------

@pytest.mark.skipif(
    not LAYERS_CSV.exists(), reason="profiler layer table not extracted"
)
def test_mapping_agrees_with_the_recorded_layer_shapes():
    """Independent of the ranges: classify by shape and require the same answer.

    Feature-extractor convolutions take 3 or 32 input channels; aggregation runs
    at 25x79 with hundreds of channels; refinement runs at full resolution.
    """
    rows = [r for r in csv.DictReader(LAYERS_CSV.open(encoding="utf-8"))
            if r["layer_type"] == "conv"]
    assert rows, "no convolutions in the layer table"

    for r in rows:
        h, w = int(r["input_height"]), int(r["input_width"])
        in_c = int(r["input_channels"])
        stage = stage_of(r["layer_name"], r["layer_type"])

        if h == 25 and w == 79:
            assert stage == STAGE_AGGREGATION, (r["layer_name"], h, w, in_c)
        elif h == 368 and w == 1232 and in_c >= 4:
            # full resolution with a fused disparity+RGB or 32-channel input
            assert stage == STAGE_REFINEMENT, (r["layer_name"], h, w, in_c)
        elif h == 368 and w == 1232 and in_c == 3:
            # the two image-facing convolutions, one per branch
            assert stage in (STAGE_FEATURE_LEFT, STAGE_FEATURE_RIGHT), r["layer_name"]
        elif h == 23 and w == 77:
            assert stage in (STAGE_FEATURE_LEFT, STAGE_FEATURE_RIGHT), r["layer_name"]


@pytest.mark.skipif(
    not LAYERS_CSV.exists(), reason="profiler layer table not extracted"
)
def test_corrected_rollup_matches_the_independent_onnx_analysis():
    """The cross-check that would have caught the original error.

    EXP-001 derived refinement 90.6 %, aggregation 4.2 %, feature extraction
    5.1 % from the ONNX. The compiled report, mapped correctly, must agree. The
    incorrect mapping gave refinement 95.2 %, which disagreed by nearly five
    points and should not have been accepted.
    """
    rows = list(csv.DictReader(LAYERS_CSV.open(encoding="utf-8")))
    totals: dict[str, float] = {}
    grand = 0.0
    for r in rows:
        try:
            macs = float(r["macs"])
        except (TypeError, ValueError):
            continue
        totals[stage_of(r["layer_name"], r["layer_type"])] = (
            totals.get(stage_of(r["layer_name"], r["layer_type"]), 0.0) + macs
        )
        grand += macs

    assert grand > 0
    refinement = totals[STAGE_REFINEMENT] / grand
    aggregation = totals[STAGE_AGGREGATION] / grand
    features = (
        totals[STAGE_FEATURE_LEFT] + totals[STAGE_FEATURE_RIGHT]
    ) / grand

    assert abs(refinement - 0.9064) < 0.01, refinement
    assert abs(aggregation - 0.0423) < 0.01, aggregation
    assert abs(features - 0.0512) < 0.01, features
    # and the wrong answer must not be reachable
    assert refinement < 0.94, "rollup matches the known-bad 95.2 % figure"


def test_exp017_correction_is_preserved():
    """The original record and its withdrawal both stay."""
    exp = REPO_ROOT / "experiments" / "EXP-017"
    assert (exp / "metrics.json").exists()
    correction = exp / "CORRECTION.md"
    assert correction.exists(), "EXP-017 must carry its correction"
    text = correction.read_text(encoding="utf-8").lower()
    assert "95.2" in text and "90.6" in text
    assert "exp-018" in text
