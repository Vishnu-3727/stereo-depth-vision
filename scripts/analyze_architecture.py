"""Compute the per-layer cost table for the Hailo StereoNet ONNX.

Reads results/onnx_inspection/nodes.json (produced by scripts/inspect_onnx.py)
and derives, for every node, its parameter count, multiply-accumulate count and
output activation size -- then rolls those up by pipeline stage.

Everything here is arithmetic over shapes the ONNX itself declares, so the
numbers are independent of any figure Hailo publishes. Where they agree with
the published figures, that is a reproduction; where they disagree, the
disagreement is the finding.

    python scripts/analyze_architecture.py
"""

from __future__ import annotations

import json
from math import prod
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = REPO_ROOT / "results" / "onnx_inspection"
OUT_DIR = REPO_ROOT / "results" / "architecture"

BYTES_FP32 = 4

# Stage boundaries, by node index, read off the ONNX node walk. The two feature
# extractor branches are separate ranges because they are separate node runs in
# the graph even though they share weights.
STAGES = [
    ("feature extraction (left branch)", 0, 34),
    ("feature extraction (right branch)", 35, 69),
    ("cost volume construction", 70, 117),
    ("cost volume aggregation (3D)", 118, 127),
    ("upsample + soft-argmin", 128, 132),
    ("refinement (full resolution)", 133, 165),
    ("residual add + ReLU", 166, 167),
]


def numel(shape) -> int:
    if not shape or any(not isinstance(d, int) for d in shape):
        return 0
    return prod(shape)


def node_macs(node) -> int:
    """Multiply-accumulates for one node. Only ops that actually multiply are
    counted; elementwise and data-movement ops contribute 0 MACs but are still
    reported for their activation traffic."""
    op = node["op"]
    out = node["outputs"][0]["shape"]
    if op == "Conv":
        w = node["inputs"][1]["shape"]
        group = node["attrs"].get("group", 1)
        # weight is [out_c, in_c/group, *kernel]
        per_output = (w[1]) * prod(w[2:])
        return numel(out) * per_output
    if op in {"Mul", "Div"}:
        return numel(out)
    return 0


def node_flops(node) -> int:
    """Elementwise arithmetic operations, for ops that do not multiply-
    accumulate. Kept separate from MACs so neither figure is inflated."""
    op = node["op"]
    out = node["outputs"][0]["shape"]
    if op in {"Add", "Sub", "Neg", "Relu", "LeakyRelu", "Mul", "Div"}:
        return numel(out)
    if op == "Softmax":
        # exp + sum + divide over the reduced axis; a coarse but stated count
        return 3 * numel(out)
    if op == "ReduceSum":
        return numel(node["inputs"][0]["shape"])
    if op == "Resize":
        # bilinear: 4 taps and 3 lerps per output element
        return 7 * numel(out)
    return 0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes = json.loads((IN_DIR / "nodes.json").read_text())
    inits = {d["name"]: d for d in json.loads((IN_DIR / "initializers.json").read_text())}

    rows = []
    seen_params: set[str] = set()
    for n in nodes:
        param_elems_occurrence = 0
        param_elems_unique = 0
        for i in n["inputs"][1:]:
            name = i["name"]
            if name in inits and i["is_param"]:
                # Constant data that is not a learned weight (index grids,
                # padding blocks, slice bounds) is excluded from parameters and
                # reported separately.
                if n["op"] in {"Conv", "Gemm", "MatMul"} and i is n["inputs"][1] or (
                    n["op"] == "Conv" and len(n["inputs"]) > 2 and i is n["inputs"][2]
                ):
                    param_elems_occurrence += inits[name]["elements"]
                    if name not in seen_params:
                        param_elems_unique += inits[name]["elements"]
                        seen_params.add(name)

        out_shape = n["outputs"][0]["shape"]
        rows.append(
            {
                "i": n["i"],
                "name": n["name"],
                "op": n["op"],
                "input_shapes": [i["shape"] for i in n["inputs"] if not i["is_param"]],
                "weight_shape": (
                    n["inputs"][1]["shape"]
                    if n["op"] == "Conv" and len(n["inputs"]) > 1
                    else None
                ),
                "output_shape": out_shape,
                "kernel": n["attrs"].get("kernel_shape"),
                "stride": n["attrs"].get("strides"),
                "dilation": n["attrs"].get("dilations"),
                "pads": n["attrs"].get("pads"),
                "params_unique": param_elems_unique,
                "params_occurrence": param_elems_occurrence,
                "macs": node_macs(n),
                "elementwise_ops": node_flops(n),
                "activation_elements": numel(out_shape),
                "activation_bytes_fp32": numel(out_shape) * BYTES_FP32,
            }
        )

    def stage_of(i: int) -> str:
        for name, lo, hi in STAGES:
            if lo <= i <= hi:
                return name
        return "unassigned"

    for r in rows:
        r["stage"] = stage_of(r["i"])

    stage_rows = []
    for name, lo, hi in STAGES:
        sel = [r for r in rows if lo <= r["i"] <= hi]
        stage_rows.append(
            {
                "stage": name,
                "nodes": len(sel),
                "params_unique": sum(r["params_unique"] for r in sel),
                "params_occurrence": sum(r["params_occurrence"] for r in sel),
                "macs": sum(r["macs"] for r in sel),
                "elementwise_ops": sum(r["elementwise_ops"] for r in sel),
                "peak_activation_bytes": max((r["activation_bytes_fp32"] for r in sel), default=0),
                "activation_traffic_bytes": sum(r["activation_bytes_fp32"] for r in sel),
            }
        )

    totals = {
        "params_unique": sum(r["params_unique"] for r in rows),
        "params_occurrence": sum(r["params_occurrence"] for r in rows),
        "macs": sum(r["macs"] for r in rows),
        "elementwise_ops": sum(r["elementwise_ops"] for r in rows),
        "activation_traffic_bytes": sum(r["activation_bytes_fp32"] for r in rows),
        "peak_single_activation_bytes": max(r["activation_bytes_fp32"] for r in rows),
    }
    totals["ops_2x_macs"] = 2 * totals["macs"]

    (OUT_DIR / "layers.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (OUT_DIR / "stages.json").write_text(
        json.dumps({"stages": stage_rows, "totals": totals}, indent=2), encoding="utf-8"
    )

    # -- report -------------------------------------------------------------
    L = []
    L.append("Per-stage cost, derived from the ONNX shapes (batch 1, 368x1232 input).")
    L.append("")
    hdr = "{:<34} {:>5} {:>10} {:>16} {:>8} {:>14} {:>14}".format(
        "stage", "nodes", "params", "MACs", "MAC %", "peak act MiB", "act traffic MiB"
    )
    L.append(hdr)
    L.append("-" * len(hdr))
    for s in stage_rows:
        L.append(
            "{:<34} {:>5} {:>10,} {:>16,} {:>7.1f}% {:>14.1f} {:>14.1f}".format(
                s["stage"], s["nodes"], s["params_unique"], s["macs"],
                100.0 * s["macs"] / totals["macs"] if totals["macs"] else 0.0,
                s["peak_activation_bytes"] / 2**20,
                s["activation_traffic_bytes"] / 2**20,
            )
        )
    L.append("-" * len(hdr))
    L.append(
        "{:<34} {:>5} {:>10,} {:>16,} {:>8} {:>14.1f} {:>14.1f}".format(
            "TOTAL", len(rows), totals["params_unique"], totals["macs"], "",
            totals["peak_single_activation_bytes"] / 2**20,
            totals["activation_traffic_bytes"] / 2**20,
        )
    )
    L.append("")
    L.append("parameter counting:")
    L.append("  unique learned tensors            : {:,}".format(totals["params_unique"]))
    L.append("  counted once per node occurrence  : {:,}".format(totals["params_occurrence"]))
    L.append("  (the siamese feature extractor runs twice but shares one set of weights)")
    L.append("")
    L.append("operation counting:")
    L.append("  MACs                              : {:,} ({:.1f}G)".format(
        totals["macs"], totals["macs"] / 1e9))
    L.append("  2 x MACs                          : {:,} ({:.1f}G)".format(
        totals["ops_2x_macs"], totals["ops_2x_macs"] / 1e9))
    L.append("  elementwise ops (not MACs)        : {:,} ({:.2f}G)".format(
        totals["elementwise_ops"], totals["elementwise_ops"] / 1e9))
    L.append("")
    L.append("ten most expensive nodes by MACs:")
    for r in sorted(rows, key=lambda r: -r["macs"])[:10]:
        L.append("  {:>4} {:<28} {:>16,} MACs  out={}".format(
            r["i"], r["op"] + " " + (r["name"] or "").split("/")[-1][:18],
            r["macs"], r["output_shape"]))
    L.append("")
    L.append("ten largest single activations:")
    for r in sorted(rows, key=lambda r: -r["activation_bytes_fp32"])[:10]:
        L.append("  {:>4} {:<28} {:>10.2f} MiB  out={}".format(
            r["i"], r["op"] + " " + (r["name"] or "").split("/")[-1][:18],
            r["activation_bytes_fp32"] / 2**20, r["output_shape"]))

    report = "\n".join(L)
    (OUT_DIR / "report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
