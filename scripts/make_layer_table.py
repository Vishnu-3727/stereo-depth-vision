"""Render the per-layer architecture table as markdown, from the ONNX analysis.

Transcribing 168 rows by hand would introduce errors, so the table in
docs/stereonet_architecture.md is generated. Rerun after any change to the
analysis and paste the output into the document.

    python scripts/make_layer_table.py > results/architecture/layer_table.md
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAYERS = REPO_ROOT / "results" / "architecture" / "layers.json"

# Nodes whose only role is data movement are collapsed into one row per run,
# so the table stays readable without hiding anything: the collapsed rows are
# summarised with their count and total activation traffic.
COLLAPSE = {"Unsqueeze"}


def fmt_shape(s) -> str:
    return "x".join(str(x) for x in s) if s else "-"


def main() -> None:
    rows = json.loads(LAYERS.read_text())

    print("| # | Stage | Operator | Output shape | Kernel | Stride | Dil. | Params | MACs | Act. MiB |")
    print("|---:|---|---|---|---|---:|---:|---:|---:|---:|")

    i = 0
    while i < len(rows):
        r = rows[i]
        if r["op"] in COLLAPSE:
            j = i
            while j < len(rows) and rows[j]["op"] == r["op"]:
                j += 1
            run = rows[i:j]
            print(
                "| {}-{} | {} | {} x{} | {} | - | - | - | 0 | {:.2f} |".format(
                    r["i"], run[-1]["i"], r["stage"], r["op"], len(run),
                    fmt_shape(r["output_shape"]),
                    sum(x["activation_bytes_fp32"] for x in run) / 2**20,
                )
            )
            i = j
            continue

        print(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.2f} |".format(
                r["i"],
                r["stage"],
                r["op"],
                fmt_shape(r["output_shape"]),
                fmt_shape(r["kernel"]) if r["kernel"] else "-",
                fmt_shape(r["stride"]) if r["stride"] else "-",
                fmt_shape(r["dilation"]) if r["dilation"] else "-",
                "{:,}".format(r["params_unique"]) if r["params_unique"] else
                ("shared" if r["op"] == "Conv" else "-"),
                "{:,}".format(r["macs"]) if r["macs"] else "0",
                r["activation_bytes_fp32"] / 2**20,
            )
        )
        i += 1


if __name__ == "__main__":
    main()
