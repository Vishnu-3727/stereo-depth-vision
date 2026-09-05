"""Inspect the Hailo StereoNet ONNX graph.

The ONNX is the highest-priority public artifact in this project: it is the
executable thing Hailo actually compiled, so where it disagrees with the paper
or the upstream Python source, the ONNX is what the baseline really is.

This script reports the graph without interpreting it -- operator inventory,
every node in topological order with resolved shapes, initializers ranked by
size, and where the parameter bytes actually go. Output is written to
results/onnx_inspection/ as both a readable report and JSON.

    python scripts/inspect_onnx.py [path/to/model.onnx]
"""

from __future__ import annotations

import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "reference" / "onnx" / "stereonet.onnx"
OUT_DIR = REPO_ROOT / "results" / "onnx_inspection"

ELEM_TYPE = {v: k for k, v in onnx.TensorProto.DataType.items()}


def tensor_shape(vi) -> list:
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        elif d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return dims


def main() -> None:
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(model_path))
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=False)
    except Exception as exc:  # shape inference is best-effort, never fatal
        print("shape inference failed: " + str(exc))
        inferred = model
    graph = inferred.graph

    # -- shapes for every named tensor we can resolve -----------------------
    shapes: dict[str, list] = {}
    dtypes: dict[str, str] = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        shapes[vi.name] = tensor_shape(vi)
        dtypes[vi.name] = ELEM_TYPE.get(vi.type.tensor_type.elem_type, "?")

    inits = OrderedDict()
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        inits[init.name] = arr
        shapes[init.name] = list(arr.shape)
        dtypes[init.name] = str(arr.dtype)

    # -- node walk ----------------------------------------------------------
    op_counts: Counter = Counter()
    nodes = []
    for idx, node in enumerate(graph.node):
        op_counts[node.op_type] += 1
        attrs = {}
        for a in node.attribute:
            if a.type == onnx.AttributeProto.INT:
                attrs[a.name] = a.i
            elif a.type == onnx.AttributeProto.FLOAT:
                attrs[a.name] = round(a.f, 6)
            elif a.type == onnx.AttributeProto.INTS:
                attrs[a.name] = list(a.ints)
            elif a.type == onnx.AttributeProto.FLOATS:
                attrs[a.name] = [round(f, 6) for f in a.floats]
            elif a.type == onnx.AttributeProto.STRING:
                attrs[a.name] = a.s.decode("utf-8", "replace")
            else:
                attrs[a.name] = "<" + str(a.type) + ">"
        nodes.append(
            {
                "i": idx,
                "name": node.name,
                "op": node.op_type,
                "inputs": [
                    {
                        "name": n,
                        "shape": shapes.get(n),
                        "dtype": dtypes.get(n),
                        "is_param": n in inits,
                    }
                    for n in node.input
                ],
                "outputs": [
                    {"name": n, "shape": shapes.get(n), "dtype": dtypes.get(n)}
                    for n in node.output
                ],
                "attrs": attrs,
            }
        )

    # -- where the bytes are ------------------------------------------------
    by_size = sorted(
        (
            {
                "name": n,
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "elements": int(a.size),
                "bytes": int(a.nbytes),
                "constant_value": (
                    float(a.reshape(-1)[0]) if a.size and a.dtype.kind == "f" else None
                ),
                "all_same": bool(a.size and np.all(a == a.reshape(-1)[0])),
            }
            for n, a in inits.items()
        ),
        key=lambda d: -d["bytes"],
    )
    total_elements = sum(d["elements"] for d in by_size)
    total_bytes = sum(d["bytes"] for d in by_size)

    # Learned parameters are what a "parameter count" normally means: the
    # weights and biases consumed by compute ops. Everything else in the
    # initializer list is baked-in constant data (index grids, padding blocks,
    # reshape targets) that costs file size and memory but is not learned.
    learned_ops = {
        "Conv", "ConvTranspose", "Gemm", "MatMul", "BatchNormalization",
        "PRelu", "InstanceNormalization",
    }
    learned_names = set()
    for node in graph.node:
        if node.op_type in learned_ops:
            for n in node.input[1:]:
                if n in inits:
                    learned_names.add(n)
    learned_elements = sum(int(inits[n].size) for n in learned_names)
    learned_bytes = sum(int(inits[n].nbytes) for n in learned_names)

    summary = {
        "model": str(model_path),
        "file_bytes": model_path.stat().st_size,
        "ir_version": model.ir_version,
        "producer": model.producer_name + " " + model.producer_version,
        "opset": [{"domain": o.domain, "version": o.version} for o in model.opset_import],
        "inputs": [
            {"name": i.name, "shape": tensor_shape(i),
             "dtype": ELEM_TYPE.get(i.type.tensor_type.elem_type, "?")}
            for i in graph.input if i.name not in inits
        ],
        "outputs": [
            {"name": o.name, "shape": tensor_shape(o),
             "dtype": ELEM_TYPE.get(o.type.tensor_type.elem_type, "?")}
            for o in graph.output
        ],
        "node_count": len(graph.node),
        "operator_inventory": dict(sorted(op_counts.items(), key=lambda kv: -kv[1])),
        "initializer_count": len(inits),
        "initializer_elements_total": total_elements,
        "initializer_bytes_total": total_bytes,
        "learned_parameter_count": learned_elements,
        "learned_parameter_bytes": learned_bytes,
        "non_learned_constant_elements": total_elements - learned_elements,
        "non_learned_constant_bytes": total_bytes - learned_bytes,
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "nodes.json").write_text(json.dumps(nodes, indent=2), encoding="utf-8")
    (OUT_DIR / "initializers.json").write_text(json.dumps(by_size, indent=2), encoding="utf-8")

    # -- readable report ----------------------------------------------------
    lines = []
    add = lines.append
    add("ONNX inspection: " + str(model_path))
    add("file size            : {:,} bytes".format(summary["file_bytes"]))
    add("producer             : " + summary["producer"])
    add("ir_version / opset   : {} / {}".format(
        model.ir_version, [o.version for o in model.opset_import]))
    add("nodes                : {:,}".format(summary["node_count"]))
    add("")
    add("inputs:")
    for i in summary["inputs"]:
        add("  {:<24} {:<10} {}".format(i["name"], i["dtype"], i["shape"]))
    add("outputs:")
    for o in summary["outputs"]:
        add("  {:<24} {:<10} {}".format(o["name"], o["dtype"], o["shape"]))
    add("")
    add("operator inventory:")
    for op, c in summary["operator_inventory"].items():
        add("  {:<24} {:>5}".format(op, c))
    add("")
    add("parameter accounting:")
    add("  learned parameters      : {:>12,} elements  {:>14,} bytes".format(
        learned_elements, learned_bytes))
    add("  non-learned constants   : {:>12,} elements  {:>14,} bytes".format(
        total_elements - learned_elements, total_bytes - learned_bytes))
    add("  all initializers        : {:>12,} elements  {:>14,} bytes".format(
        total_elements, total_bytes))
    add("")
    add("largest 25 initializers (bytes):")
    add("  {:<46} {:>14} {:>10} {}".format("name", "bytes", "dtype", "shape"))
    for d in by_size[:25]:
        flag = "  [constant, all elements equal]" if d["all_same"] else ""
        add("  {:<46} {:>14,} {:>10} {}{}".format(
            d["name"][:46], d["bytes"], d["dtype"], d["shape"], flag))
    add("")
    add("node walk (index | op | name | inputs -> outputs):")
    for n in nodes:
        ins = ", ".join(
            "{}{}".format(i["shape"], "*" if i["is_param"] else "") for i in n["inputs"]
        )
        outs = ", ".join(str(o["shape"]) for o in n["outputs"])
        attr = ""
        if n["op"] in {"Conv", "MaxPool", "AveragePool", "Resize", "Pad", "Slice"}:
            keep = {k: v for k, v in n["attrs"].items()
                    if k in {"kernel_shape", "strides", "pads", "dilations", "group", "mode"}}
            if keep:
                attr = "  " + json.dumps(keep, separators=(",", ":"))
        add("  {:>4} {:<22} {:<34} {}  ->  {}{}".format(
            n["i"], n["op"], (n["name"] or "")[:34], ins, outs, attr))
    add("")
    add("(* marks an input that is an initializer, i.e. stored data rather than")
    add(" a computed activation.)")

    report = "\n".join(lines)
    (OUT_DIR / "report.txt").write_text(report, encoding="utf-8")
    print(report[:8000])
    print("\n...full report written to " + str(OUT_DIR / "report.txt"))


if __name__ == "__main__":
    main()
