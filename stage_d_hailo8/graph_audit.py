#!/usr/bin/env python
"""Stage D D1 structural audit: what a Hailo compiler would be handed.

Static, read-only ONNX analysis of ARM-P-H8-V0 next to the VENDOR StereoNet,
which has a published Hailo-8 HEF and therefore demonstrates what that
toolchain accepts for this model family.

This does NOT run the Hailo Dataflow Compiler and makes no claim that the
graph compiles. It reports structure only; compiler acceptance is D2 and is
NOT ATTEMPTED until a toolchain host exists.

    python stage_d_hailo8/graph_audit.py
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import onnx
from onnx import shape_inference

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

TARGETS = {
    "armp_h8_v0": "stage_d_hailo8/v0/armp_h8_v0.onnx",
    "armp_h8_v0_static": "stage_d_hailo8/v0/armp_h8_v0_static.onnx",
    "vendor_stereonet": "reference/onnx/stereonet.onnx",
}


def audit(path: Path) -> dict:
    raw = path.read_bytes()
    m = shape_inference.infer_shapes(onnx.load(str(path)))
    init = {i.name: i for i in m.graph.initializer}

    conv_rank: collections.Counter = collections.Counter()
    for n in m.graph.node:
        if n.op_type == "Conv":
            w = init.get(n.input[1])
            conv_rank[len(w.dims) if w is not None else "unknown"] += 1

    ranks: collections.Counter = collections.Counter()
    vi_rank: dict[str, int] = {}
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        r = len(vi.type.tensor_type.shape.dim)
        ranks[r] += 1
        vi_rank[vi.name] = r
    rank5: collections.Counter = collections.Counter()
    for n in m.graph.node:
        if any(vi_rank.get(o, 0) >= 5 for o in n.output):
            rank5[n.op_type] += 1

    def shape(vi) -> list:
        return [d.dim_value or d.dim_param for d in vi.type.tensor_type.shape.dim]

    ops = collections.Counter(n.op_type for n in m.graph.node)
    return {
        "path": str(path.relative_to(REPO)).replace("\\", "/"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "opset": [(x.domain or "ai.onnx", x.version) for x in m.opset_import],
        "ir_version": m.ir_version,
        "producer": f"{m.producer_name} {m.producer_version}",
        "inputs": [(i.name, shape(i)) for i in m.graph.input],
        "outputs": [(o.name, shape(o)) for o in m.graph.output],
        "node_count": sum(ops.values()),
        "op_type_count": len(ops),
        "ops": dict(sorted(ops.items(), key=lambda kv: -kv[1])),
        "conv_weight_rank_histogram": {str(k): v for k, v in conv_rank.items()},
        "tensor_rank_histogram": {str(k): v for k, v in sorted(ranks.items())},
        "max_tensor_rank": max(ranks) if ranks else None,
        "ops_producing_rank_ge_5": dict(rank5),
    }


def main() -> None:
    report = {"note": "Structural facts only. The Hailo Dataflow Compiler was "
                      "NOT run; graph acceptance is D2 and NOT ATTEMPTED.",
              "models": {}}
    for name, rel in TARGETS.items():
        p = REPO / rel
        if not p.exists():
            report["models"][name] = {"status": "MISSING", "path": rel}
            continue
        report["models"][name] = audit(p)

    a = report["models"].get("armp_h8_v0", {})
    v = report["models"].get("vendor_stereonet", {})
    if a and v and "ops" in a and "ops" in v:
        # Compare SHAPES, not tensor names: names are arbitrary labels and
        # have no bearing on what a compiler accepts.
        a_in, v_in = [s for _, s in a["inputs"]], [s for _, s in v["inputs"]]
        a_out, v_out = [s for _, s in a["outputs"]], [s for _, s in v["outputs"]]
        report["armp_vs_vendor"] = {
            "same_io_shapes": a_in == v_in and a_out == v_out,
            "armp_input_names": [n for n, _ in a["inputs"]],
            "vendor_input_names": [n for n, _ in v["inputs"]],
            "armp_inputs": a["inputs"], "vendor_inputs": v["inputs"],
            "armp_outputs": a["outputs"], "vendor_outputs": v["outputs"],
            "conv3d_count": {"armp": a["conv_weight_rank_histogram"].get("5", 0),
                             "vendor": v["conv_weight_rank_histogram"].get("5", 0)},
            "conv2d_count": {"armp": a["conv_weight_rank_histogram"].get("4", 0),
                             "vendor": v["conv_weight_rank_histogram"].get("4", 0)},
            "node_count": {"armp": a["node_count"], "vendor": v["node_count"]},
            "op_types_only_in_armp": sorted(set(a["ops"]) - set(v["ops"])),
            "op_types_only_in_vendor": sorted(set(v["ops"]) - set(a["ops"])),
        }
    out = HERE / "reports" / "d1_graph_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for name, d in report["models"].items():
        if "ops" not in d:
            print(f"{name}: {d.get('status')}")
            continue
        print(f"{name:<18} nodes={d['node_count']:<5} "
              f"conv2d={d['conv_weight_rank_histogram'].get('4', 0)} "
              f"conv3d={d['conv_weight_rank_histogram'].get('5', 0)} "
              f"max_rank={d['max_tensor_rank']} opset={d['opset']}")
    if "armp_vs_vendor" in report:
        c = report["armp_vs_vendor"]
        print(f"same I/O shapes as vendor: {c['same_io_shapes']} "
              f"(names differ: {c['armp_input_names']} vs {c['vendor_input_names']})")
        print(f"op types only in ARM-P: {c['op_types_only_in_armp']}")
        print(f"op types only in vendor: {c['op_types_only_in_vendor']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
