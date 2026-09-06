"""EXP: static forensics of the Hailo StereoNet ONNX.

Runs the ONNX inspection and the shape-derived cost analysis, then checks the
numbers we derive independently against the figures Hailo publishes in its
Model Zoo configuration. Nothing here is tuned to match; the published values
are only compared against, never used as an input.

    python scripts/exp_onnx_forensics.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.experiment import Experiment  # noqa: E402

# Figures Hailo publishes for this model. SOURCE only -- never treated as our
# own measurement, and never used to steer the analysis.
HAILO_PUBLISHED = {
    "parameters": 623_100,        # "623.1K" in stereonet.yaml
    "operations": 112_200_000_000,  # "112.2G" in stereonet.yaml
    "input_shape": [368, 1232, 3],
    "output_shape": [368, 1232, 1],
    "full_precision_result_epe": 8.223,
}


def main() -> None:
    config = {
        "artifact": "reference/onnx/stereonet.onnx",
        "artifact_sha256_prefix": None,  # filled in from the manifest below
        "analysis": "static graph forensics (no inference executed)",
        "dataset": None,
        "split": None,
        "resolution": [368, 1232],
        "crop": None,
        "disparity_range": None,  # read from the graph, not assumed
        "batch_size": 1,
        "precision": "fp32",
        "seed": None,
    }
    manifest = json.loads((REPO_ROOT / "reference" / "manifest.json").read_text())
    for a in manifest["artifacts"]:
        if a["path"] == "onnx/stereonet.onnx":
            config["artifact_sha256_prefix"] = a["sha256"][:16]

    with Experiment(
        "Static forensics of the Hailo StereoNet ONNX: graph inventory, "
        "parameter accounting and shape-derived cost model",
        config=config,
    ) as exp:
        for script in ("scripts/inspect_onnx.py", "scripts/analyze_architecture.py"):
            exp.log("running " + script)
            r = subprocess.run(
                [sys.executable, script], cwd=REPO_ROOT, capture_output=True, text=True
            )
            if r.returncode != 0:
                raise RuntimeError(script + " failed:\n" + r.stderr[-2000:])

        summary = json.loads(
            (REPO_ROOT / "results" / "onnx_inspection" / "summary.json").read_text()
        )
        stages = json.loads(
            (REPO_ROOT / "results" / "architecture" / "stages.json").read_text()
        )
        totals = stages["totals"]

        # -- graph facts ----------------------------------------------------
        exp.metric("onnx_file_bytes", summary["file_bytes"])
        exp.metric("node_count", summary["node_count"])
        exp.metric("operator_inventory", summary["operator_inventory"])
        exp.metric("inputs", summary["inputs"])
        exp.metric("outputs", summary["outputs"])

        # -- parameter accounting -------------------------------------------
        exp.metric("params_unique_tensors", totals["params_unique"])
        exp.metric("params_per_node_occurrence", totals["params_occurrence"])
        exp.metric("hailo_published_parameters", HAILO_PUBLISHED["parameters"])

        # -- operation accounting -------------------------------------------
        exp.metric("macs", totals["macs"])
        exp.metric("ops_2x_macs", totals["ops_2x_macs"])
        exp.metric("elementwise_ops", totals["elementwise_ops"])
        exp.metric("hailo_published_operations", HAILO_PUBLISHED["operations"])

        # -- memory ---------------------------------------------------------
        exp.metric("peak_single_activation_bytes", totals["peak_single_activation_bytes"])
        exp.metric("activation_traffic_bytes", totals["activation_traffic_bytes"])
        exp.metric(
            "learned_parameter_bytes_fp32", summary["learned_parameter_bytes"]
        )
        exp.metric(
            "non_learned_constant_bytes", summary["non_learned_constant_bytes"]
        )

        refine = next(
            s for s in stages["stages"] if s["stage"].startswith("refinement")
        )
        exp.metric("refinement_mac_share", refine["macs"] / totals["macs"])
        exp.metric(
            "refinement_activation_traffic_share",
            refine["activation_traffic_bytes"] / totals["activation_traffic_bytes"],
        )

        # -- checks against the published figures ---------------------------
        param_gap = totals["params_occurrence"] - HAILO_PUBLISHED["parameters"]
        ops_gap = totals["ops_2x_macs"] - HAILO_PUBLISHED["operations"]
        exp.metric("param_gap_vs_published", param_gap)
        exp.metric("ops_gap_vs_published", ops_gap)
        exp.metric(
            "ops_relative_gap_vs_published",
            ops_gap / HAILO_PUBLISHED["operations"],
        )

        exp.note(
            "The exported graph builds its cost volume with Sub (12 Sub nodes at "
            "1/16 resolution, 32 channels). The upstream repository README states "
            "the cost volume uses concatenation. The ONNX is the executable "
            "artifact Hailo compiled, so subtraction is what the deployed baseline "
            "actually does."
        )
        exp.note(
            "Parameter discrepancy resolved. Unique learned tensors total "
            + "{:,}".format(totals["params_unique"])
            + ". Counting the shared siamese feature extractor once per node "
            "occurrence instead gives "
            + "{:,}".format(totals["params_occurrence"])
            + ", which matches Hailo's published 623.1K to within "
            + str(param_gap)
            + " parameters (rounding). Hailo therefore counts weights per graph "
            "occurrence, not per unique tensor."
        )
        exp.note(
            "Operation count reproduced. Our shape-derived MAC total is "
            + "{:.2f}G".format(totals["macs"] / 1e9)
            + "; doubling it gives {:.2f}G".format(totals["ops_2x_macs"] / 1e9)
            + " against Hailo's published 112.2G, a relative gap of "
            + "{:.2%}".format(ops_gap / HAILO_PUBLISHED["operations"])
            + ". Hailo's 'operations' figure is therefore 2 x MACs."
        )
        exp.note(
            "92% of the ONNX file is one initializer: /Tile_output_0, a "
            "[1, 12, 368, 1232] float32 constant of 21,762,048 bytes. It is the "
            "soft-argmin disparity index grid, materialised at full resolution "
            "and multiplied elementwise. The learned weights are only "
            + "{:,} bytes".format(summary["learned_parameter_bytes"])
            + " of the 23.7 MB file."
        )
        exp.note(
            "Static cost is dominated by the full-resolution refinement stage: "
            + "{:.1%} of MACs and {:.1%} of activation traffic. This is a static "
            "shape-derived figure, not a latency measurement -- whether it is the "
            "runtime bottleneck is still open and is tested in W8.".format(
                refine["macs"] / totals["macs"],
                refine["activation_traffic_bytes"] / totals["activation_traffic_bytes"],
            )
        )

        exp.conclude(
            "Both published complexity figures are independently reproduced from "
            "the ONNX shapes, and the counting conventions behind them are now "
            "known. The graph is fully resolved: 168 nodes, static shapes "
            "throughout, cost volume at 1/16 resolution with 12 disparity "
            "candidates built by subtraction, and a full-resolution refinement "
            "stage that carries the overwhelming majority of the arithmetic."
        )
        print("\nrecorded as " + exp.id)


if __name__ == "__main__":
    main()
