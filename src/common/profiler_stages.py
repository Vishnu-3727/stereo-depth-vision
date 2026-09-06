"""Map Hailo compiled-layer names onto pipeline stages.

Hailo's compiled graph numbers its convolutions `conv1` upwards in execution
order, with no indication of which architectural stage each belongs to. The
mapping has to be derived from the layer shapes in the profiler report, and
getting it wrong silently corrupts every per-stage rollup built on top of it.

EXP-017 did get it wrong. It assumed `conv18`-`conv22` were the 3D aggregation
and everything from `conv23` was refinement, which folded the entire right
feature-extractor branch into refinement and reported it as 95.2 % of compiled
MACs. The correct boundaries are established by the shapes:

    conv1  .. conv17   left feature extractor
                       conv1  is 368x1232, 3 -> 32, 5x5   (the image)
                       conv5..conv17 are 23x77, 32 -> 32, 3x3 (residual stack)
    conv18 .. conv34   right feature extractor
                       conv18 is 368x1232, 3 -> 32, 5x5 -- identical to conv1,
                       which is what makes it the Siamese twin, not aggregation
    conv35 .. conv39   3D aggregation
                       25x79, 448 -> 384 channels; the compiler flattens the
                       disparity axis into channels (12 disparities x 32
                       channels = 384 out, 14 x 32 = 448 in for a 3x3x3 kernel).
                       conv39 emits 12 channels: one cost per disparity
    conv40 .. conv53   full-resolution refinement
                       conv40 is 368x1232 with 4 input channels -- disparity
                       concatenated with the left RGB image
                       conv53 emits 1 channel: the disparity map

With these boundaries the profiler rollup agrees with our independent ONNX
analysis (EXP-001) to within a rounding step, which is the cross-check that the
mapping is right rather than merely plausible.
"""

from __future__ import annotations

import re

# Inclusive upper bound of each convolution range, in compiled execution order.
FEATURE_LEFT_END = 17
FEATURE_RIGHT_END = 34
AGGREGATION_END = 39
# Everything above AGGREGATION_END is refinement.

STAGE_FEATURE_LEFT = "feature extraction (left)"
STAGE_FEATURE_RIGHT = "feature extraction (right)"
STAGE_AGGREGATION = "aggregation (3D)"
STAGE_REFINEMENT = "refinement"
STAGE_IO = "io/const"
STAGE_NORMALISATION = "normalisation"
STAGE_DATA_MOVEMENT = "data movement"
STAGE_REGRESSION = "regression / elementwise"
STAGE_OTHER = "other"

# A convolution's own name, after any defusion suffix. Anchored so that a
# routing layer such as "ws_from_conv47_ws_to_conv47_sd0-1" is not mistaken for
# a convolution, and so "conv17_concat1_transpose" cannot yield the index 171.
_CONV = re.compile(r"^conv(\d+)")

_IO_TYPES = {"const_input", "input_layer", "output_layer"}
_MOVEMENT_PREFIXES = (
    "concat", "shape_splitter", "format_conversion", "shortcut", "ws_from",
    "sh_from", "mux", "demux", "muxer", "demuxer", "slice", "external_pad",
)
_REGRESSION_PREFIXES = (
    "resize", "reduce", "softmax", "argmax", "mul", "neg", "ew_mult",
    "ew_sub", "mul_and_add", "bilinear",
)


def conv_index(layer_name: str) -> int | None:
    """The convolution index of a compiled layer, or None if it is not one.

    Strips the network prefix and any defusion suffix (`_sd0`, `_ws`, `_sdc`)
    before matching, so `stereonet/conv47_sd12` resolves to 47.
    """
    base = layer_name.split("/")[-1]
    for marker in ("_sdc", "_sd", "_ws"):
        base = base.split(marker)[0]
    match = _CONV.match(base)
    return int(match.group(1)) if match else None


def stage_of(layer_name: str, layer_type: str = "") -> str:
    """The pipeline stage a compiled layer belongs to."""
    if layer_type in _IO_TYPES:
        return STAGE_IO

    index = conv_index(layer_name)
    if index is not None:
        if index <= FEATURE_LEFT_END:
            return STAGE_FEATURE_LEFT
        if index <= FEATURE_RIGHT_END:
            return STAGE_FEATURE_RIGHT
        if index <= AGGREGATION_END:
            return STAGE_AGGREGATION
        return STAGE_REFINEMENT

    base = layer_name.split("/")[-1]
    if base.startswith("normalization"):
        return STAGE_NORMALISATION
    if base.startswith(_MOVEMENT_PREFIXES):
        return STAGE_DATA_MOVEMENT
    if base.startswith(_REGRESSION_PREFIXES):
        return STAGE_REGRESSION
    return STAGE_OTHER


def demo() -> None:
    """Self-check on the boundaries that EXP-017 got wrong."""
    # the four ranges, at every boundary
    assert stage_of("conv1", "conv") == STAGE_FEATURE_LEFT
    assert stage_of("conv17", "conv") == STAGE_FEATURE_LEFT
    assert stage_of("conv18", "conv") == STAGE_FEATURE_RIGHT
    assert stage_of("conv34", "conv") == STAGE_FEATURE_RIGHT
    assert stage_of("conv35", "conv") == STAGE_AGGREGATION
    assert stage_of("conv39", "conv") == STAGE_AGGREGATION
    assert stage_of("conv40", "conv") == STAGE_REFINEMENT
    assert stage_of("conv53", "conv") == STAGE_REFINEMENT

    # defused names keep their identity
    assert conv_index("stereonet/conv47_sd12") == 47
    assert conv_index("conv42_sdc") == 42
    assert conv_index("conv48_ws") == 48
    assert stage_of("stereonet/conv47_sd12", "conv") == STAGE_REFINEMENT
    assert stage_of("conv34_sd0", "conv") == STAGE_FEATURE_RIGHT

    # routing layers are not convolutions
    assert conv_index("ws_from_conv47_ws_to_conv47_sd0-1") is None
    assert conv_index("sh_from_conv46_to_conv48_sd0-3") is None
    assert conv_index("mux_conv17_concat3_transpose_to_concat3") is None

    # a trailing numeric token must not corrupt the index
    assert conv_index("conv17_concat1_transpose") == 17

    assert stage_of("const_input1", "const_input") == STAGE_IO
    assert stage_of("normalization1", "normalization") == STAGE_NORMALISATION
    assert stage_of("resize1", "resize") == STAGE_REGRESSION
    assert stage_of("concat12", "concat") == STAGE_DATA_MOVEMENT

    print("profiler stage mapping self-check passed")


if __name__ == "__main__":
    demo()
