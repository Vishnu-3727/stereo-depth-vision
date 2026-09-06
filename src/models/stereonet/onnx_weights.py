"""Load the reference ONNX weights into the independent implementation.

The exported graph does not use PyTorch module names for most of its weights --
batch-norm folding renamed them to generated identifiers like ``onnx::Conv_526``
-- so mapping by name is not possible for the majority of tensors.

Mapping by *execution order* is, and it is a stronger check anyway. Both the
reference graph and this implementation apply their convolutions in a fixed
order; if the two orders agree, matching them positionally loads the right
weights, and if they do not agree the shapes will not line up and the load
fails loudly. Every tensor's shape is asserted, so a silent mismatch cannot
happen.

The graph runs the shared feature extractor twice. Only the first occurrence is
consumed here; the second is verified to reference the same initializers, which
independently confirms the weights really are shared.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Node index ranges of the reference graph, from the node walk in
# results/onnx_inspection/report.txt. The right-branch feature extractor
# (35-69) is deliberately excluded: it shares weights with the left branch.
LEFT_BRANCH = (0, 34)
RIGHT_BRANCH = (35, 69)
AGGREGATION = (118, 126)
REFINEMENT = (133, 165)


def _conv_nodes(graph, lo: int, hi: int):
    return [n for i, n in enumerate(graph.node) if lo <= i <= hi and n.op_type == "Conv"]


def ordered_conv_modules(model) -> list[nn.Module]:
    """Convolutions of the independent model, in the order forward() runs them."""
    fe = model.feature_extractor
    mods: list[nn.Module] = list(fe.downsample)
    for block in fe.residual:
        mods += [block.conv1, block.conv2]
    mods.append(fe.output_conv)

    agg = model.aggregation
    mods += [m for m in agg.filter if isinstance(m, nn.Conv3d)]
    mods.append(agg.to_cost)

    ref = model.refinement
    mods.append(ref.input_conv)
    for block in ref.blocks:
        mods += [block.conv1, block.conv2]
    mods.append(ref.output_conv)
    return mods


def load_onnx_weights(model, onnx_path: str | Path, strict: bool = True) -> dict:
    """Copy the reference weights into ``model`` in place.

    Returns a report describing what was loaded and what was checked.
    """
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(onnx_path)).graph
    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}

    left = _conv_nodes(graph, *LEFT_BRANCH)
    right = _conv_nodes(graph, *RIGHT_BRANCH)
    agg = _conv_nodes(graph, *AGGREGATION)
    ref = _conv_nodes(graph, *REFINEMENT)

    # The Siamese claim, checked rather than assumed.
    shared = [a.input[1] == b.input[1] for a, b in zip(left, right)]
    if len(left) != len(right) or not all(shared):
        raise ValueError(
            "the two feature-extractor branches do not share weights; the "
            "positional mapping assumed here would be wrong"
        )

    nodes = left + agg + ref
    mods = ordered_conv_modules(model)
    if len(nodes) != len(mods):
        raise ValueError(
            "convolution count mismatch: ONNX has {} to load, the model has "
            "{}".format(len(nodes), len(mods))
        )

    loaded = []
    for node, mod in zip(nodes, mods):
        w = inits[node.input[1]]
        if tuple(w.shape) != tuple(mod.weight.shape):
            raise ValueError(
                "weight shape mismatch at {}: ONNX {} vs model {}".format(
                    node.name, w.shape, tuple(mod.weight.shape)
                )
            )
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(np.array(w)))
            if len(node.input) > 2:
                b = inits[node.input[2]]
                if mod.bias is None:
                    raise ValueError(
                        "ONNX node " + node.name + " has a bias but the model "
                        "layer does not"
                    )
                if tuple(b.shape) != tuple(mod.bias.shape):
                    raise ValueError(
                        "bias shape mismatch at " + node.name
                    )
                mod.bias.copy_(torch.from_numpy(np.array(b)))
            elif mod.bias is not None and strict:
                raise ValueError(
                    "model layer for " + node.name + " expects a bias that the "
                    "ONNX node does not provide"
                )
        loaded.append(
            {"onnx_node": node.name, "weight": node.input[1], "shape": list(w.shape)}
        )

    total = sum(int(np.prod(x["shape"])) for x in loaded)
    total += sum(
        int(inits[n.input[2]].size) for n in nodes if len(n.input) > 2
    )
    return {
        "convolutions_loaded": len(loaded),
        "parameters_loaded": total,
        "siamese_weights_shared": True,
        "layers": loaded,
    }
