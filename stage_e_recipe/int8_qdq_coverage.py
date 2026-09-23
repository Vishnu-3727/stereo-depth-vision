#!/usr/bin/env python
"""QDQ coverage audit for INT8 sensitivity (Brief 5, step 1).

Reproduces the manager's counts from baseline_int8.onnx and only_00_*.onnx:
- number of QuantizeLinear nodes in each file,
- for baseline_int8.onnx: op_type of every node that consumes a
  DequantizeLinear output (i.e. which op types stay quantized).

Read-only on the .onnx files. Writes stage_e_recipe/int8_sensitivity/qdq_coverage.json.
No training, no quantization, no environment change.
"""
from __future__ import annotations

import collections
import datetime
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORK = HERE / "int8_sensitivity"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def audit(path: Path) -> dict:
    import onnx
    m = onnx.load(str(path))
    n_ql = sum(1 for n in m.graph.node if n.op_type == "QuantizeLinear")
    n_dql = sum(1 for n in m.graph.node if n.op_type == "DequantizeLinear")
    # Map each DequantizeLinear output tensor -> consumer op types.
    dql_outputs: dict[str, str] = {}
    for n in m.graph.node:
        if n.op_type == "DequantizeLinear":
            for o in n.output:
                dql_outputs[o] = n.name
    consumer_counter: collections.Counter[str] = collections.Counter()
    consumer_edges: collections.Counter[str] = collections.Counter()
    graph_outputs = {o.name for o in m.graph.output}
    dangling = 0
    for n in m.graph.node:
        hits = sum(1 for i in n.input if i in dql_outputs)
        if hits:
            consumer_counter[n.op_type] += 1  # distinct consuming nodes
            consumer_edges[n.op_type] += hits  # input-slot edges (fan-in > 1)
    for o in dql_outputs:
        used = any(o in n.input for n in m.graph.node) or o in graph_outputs
        if not used:
            dangling += 1
    total_nodes = len(m.graph.node)
    return {
        "file": str(path.relative_to(REPO)).replace("\\", "/"),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "n_nodes": total_nodes,
        "n_quantize_linear": n_ql,
        "n_dequantize_linear": n_dql,
        "dequant_consumers_by_op_type": dict(sorted(consumer_counter.items())),
        "dequant_consumer_edges_by_op_type": dict(sorted(consumer_edges.items())),
        "dequant_consumer_total": sum(consumer_counter.values()),
        "dequant_outputs_dangling": dangling,
    }


def main() -> None:
    base = WORK / "baseline_int8.onnx"
    assert base.exists(), f"missing {base}"
    only = sorted(glob.glob(str(WORK / "only_00_*.onnx")))
    assert only, "no only_00_*.onnx found"
    only_path = Path(only[0])

    rec = {
        "script": "stage_e_recipe/int8_qdq_coverage.py",
        "purpose": ("reproduce QDQ coverage counts for the Brief-5 correction; "
                    "read-only audit, no training, no quantization"),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_head": git_head(),
        "python": sys.version.split()[0],
        "baseline": audit(base),
        "only_00": audit(only_path),
    }
    out = WORK / "qdq_coverage.json"
    out.write_text(json.dumps(rec, indent=2))
    print(json.dumps({"baseline_ql": rec["baseline"]["n_quantize_linear"],
                      "only00_ql": rec["only_00"]["n_quantize_linear"],
                      "consumers": rec["baseline"]["dequant_consumers_by_op_type"],
                      "wrote": str(out)}, indent=2))


if __name__ == "__main__":
    main()
