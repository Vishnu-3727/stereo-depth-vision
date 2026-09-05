"""EXP: what Hailo's own compiled profiler report says about this model.

Analyses the per-layer table extracted from
`stereonet_profiler_results_compiled_runtime_data.html` [SR-006] -- the only
public source of Hailo-side per-layer figures available without a device.

Two things this settles that were previously UNKNOWN: how the compiled model is
partitioned across device contexts, and which layers the compiler considers the
throughput bottleneck. It also lets Hailo's own parameter and operation counts be
checked against ours.

One important caveat, read off the report itself: `profiling_mode` is
`post_placement` and both the model-level `fps` and `latency` fields are `N/A`.
This is the compiler's static post-placement estimate, **not a measured run on
hardware**. Per-layer `fps` figures are therefore modelled throughput, not
measurements, and are labelled SOURCE accordingly.

    python scripts/exp_profiler_report.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.experiment import Experiment  # noqa: E402

PROFILER_DIR = REPO_ROOT / "results" / "profiler"

# Our own figures, for the cross-check.
OUR_MACS = 56_039_313_792
OUR_PARAMS_UNIQUE = 423_586
OUR_PARAMS_PER_OCCURRENCE = 623_138
HAILO_PUBLISHED_OPS = 112_200_000_000
HAILO_PUBLISHED_PARAMS = 623_100

# Map a compiled layer name onto the pipeline stage it belongs to. Derived from
# the ONNX node walk: conv1-conv17 are the two feature extractor branches,
# conv18-conv22 the 3D aggregation, conv23+ the refinement. Names carrying a
# defusion suffix keep their base identity.
def stage_of(name: str, layer_type: str) -> str:
    n = name.split("/")[-1]
    base = n.split("_sd")[0].split("_ws")[0].split("_sdc")[0]
    if layer_type in {"const_input", "input_layer", "output_layer"}:
        return "io/const"
    if base.startswith("normalization"):
        return "normalisation"
    if base.startswith(("concat", "shape_splitter", "format_conversion", "shortcut")):
        return "data movement"
    if base.startswith("conv"):
        digits = "".join(c for c in base[4:] if c.isdigit())
        if digits:
            i = int(digits)
            if i <= 17:
                return "feature extraction"
            if i <= 22:
                return "aggregation (3D)"
            return "refinement"
    if base.startswith(("ew_add", "activation", "reduce", "softmax", "resize",
                        "bilinear", "argmax", "mul", "neg", "slice")):
        return "regression / elementwise"
    return "other"


def num(v):
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return None


def main() -> None:
    layers_csv = PROFILER_DIR / "layers.csv"
    if not layers_csv.exists():
        subprocess.run(
            [sys.executable, "scripts/extract_profiler_report.py"],
            cwd=REPO_ROOT, check=True, capture_output=True,
        )
    rows = list(csv.DictReader(layers_csv.open(encoding="utf-8")))
    model = json.loads((PROFILER_DIR / "summary.json").read_text())["model"]

    config = {
        "dataset": None,
        "split": None,
        "resolution": [368, 1232],
        "crop": None,
        "disparity_range": None,
        "batch_size": 1,
        "precision": "int8 (Hailo compiled)",
        "seed": None,
        "artifact": "reference/stereonet_profiler_results_compiled_runtime_data.html",
        "profiling_mode": model.get("profiling_mode"),
        "hw_arch": model.get("hw_arch"),
    }

    with Experiment(
        "Hailo's own compiled profiler report: per-layer figures, context "
        "partitioning and a cross-check of the published counts",
        config=config,
    ) as exp:
        exp.note(
            "profiling_mode is '" + str(model.get("profiling_mode"))
            + "' and the model-level fps and latency fields are '"
            + str(model.get("fps")) + "' and '" + str(model.get("latency"))
            + "'. This is the compiler's static post-placement estimate, not a "
            "measured run on hardware. Everything below is SOURCE, and no "
            "figure here is a measurement of silicon."
        )

        # -- cross-check the published counts --------------------------------
        hailo_macs = num(model["macs_per_image"])
        hailo_ops = num(model["ops_per_image"])
        hailo_weights = num(model["weights"])
        exp.metric("hailo_weights", hailo_weights)
        exp.metric("hailo_macs_per_image", hailo_macs)
        exp.metric("hailo_ops_per_image", hailo_ops)
        exp.metric("hailo_ops_over_macs", hailo_ops / hailo_macs)
        exp.metric("our_macs", OUR_MACS)
        exp.metric("our_params_unique", OUR_PARAMS_UNIQUE)
        exp.metric("our_params_per_occurrence", OUR_PARAMS_PER_OCCURRENCE)
        exp.metric(
            "mac_relative_gap_ours_vs_hailo", (OUR_MACS - hailo_macs) / hailo_macs
        )
        exp.metric(
            "weights_exact_match_with_our_per_occurrence",
            hailo_weights == float(OUR_PARAMS_PER_OCCURRENCE),
        )

        # -- device partitioning ---------------------------------------------
        contexts = Counter(r["context"] for r in rows)
        exp.metric("number_of_contexts", model.get("number_of_contexts"))
        exp.metric("layers_per_context", dict(sorted(contexts.items())))
        exp.metric("compiled_layer_count", len(rows))

        # -- defusion ---------------------------------------------------------
        defused = [r for r in rows if r.get("defuse_name")]
        defuse_groups = defaultdict(list)
        for r in defused:
            defuse_groups[r["defuse_name"]].append(r["layer_name"])
        exp.metric("defused_layer_rows", len(defused))
        exp.metric(
            "defuse_groups",
            {k: len(v) for k, v in sorted(
                defuse_groups.items(), key=lambda kv: -len(kv[1])
            )},
        )

        # -- per-stage roll-up -------------------------------------------------
        by_stage = defaultdict(lambda: {
            "layers": 0, "macs": 0.0, "l2_data_usage": 0.0,
            "min_fps": None, "utilisations": [],
        })
        for r in rows:
            s = stage_of(r["layer_name"], r["layer_type"])
            b = by_stage[s]
            b["layers"] += 1
            b["macs"] += num(r.get("macs")) or 0.0
            b["l2_data_usage"] += num(r.get("l2_data_usage")) or 0.0
            f = num(r.get("fps"))
            if f is not None and f > 0:
                b["min_fps"] = f if b["min_fps"] is None else min(b["min_fps"], f)
            u = num(r.get("effective_mac_util"))
            if u is not None:
                b["utilisations"].append(u)

        total_macs = sum(b["macs"] for b in by_stage.values()) or 1.0
        stage_table = {}
        for s, b in by_stage.items():
            stage_table[s] = {
                "layers": b["layers"],
                "macs": b["macs"],
                "mac_share": b["macs"] / total_macs,
                "min_fps": b["min_fps"],
                "l2_data_usage_bytes": b["l2_data_usage"],
                "mean_effective_mac_util": (
                    sum(b["utilisations"]) / len(b["utilisations"])
                    if b["utilisations"] else None
                ),
            }
        exp.metric("by_stage", stage_table)

        # -- the throughput bottleneck ----------------------------------------
        fps_rows = [(num(r.get("fps")), r) for r in rows if num(r.get("fps"))]
        fps_rows = [(f, r) for f, r in fps_rows if f and f > 0]
        fps_rows.sort(key=lambda x: x[0])
        slowest = [
            {
                "layer_name": r["layer_name"],
                "layer_type": r["layer_type"],
                "stage": stage_of(r["layer_name"], r["layer_type"]),
                "fps": f,
                "macs": num(r.get("macs")),
                "context": r.get("context"),
                "effective_mac_util": num(r.get("effective_mac_util")),
                "defuse_name": r.get("defuse_name"),
            }
            for f, r in fps_rows[:15]
        ]
        exp.metric("slowest_layers_by_modelled_fps", slowest)
        exp.metric("bottleneck_fps_modelled", fps_rows[0][0] if fps_rows else None)

        # -- quantisation bit widths -------------------------------------------
        bits = Counter(
            (r.get("weights_bits"), r.get("input_activation_bits"),
             r.get("output_activation_bits"))
            for r in rows
        )
        exp.metric(
            "bit_width_combinations_weights_in_out",
            {str(k): v for k, v in bits.most_common()},
        )

        (PROFILER_DIR / "analysis.json").write_text(
            json.dumps(
                {
                    "by_stage": stage_table,
                    "slowest_layers": slowest,
                    "layers_per_context": dict(sorted(contexts.items())),
                    "defuse_groups": {k: v for k, v in defuse_groups.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # -- notes -------------------------------------------------------------
        exp.note(
            "Hailo's own compiler reports weights = {:,.0f}, which is exactly "
            "our per-occurrence parameter count of {:,}. That independently "
            "confirms the explanation in EXP-001: the published 623.1K counts "
            "the shared Siamese feature extractor once per graph occurrence, "
            "not once per unique tensor ({:,}).".format(
                hailo_weights, OUR_PARAMS_PER_OCCURRENCE, OUR_PARAMS_UNIQUE
            )
        )
        exp.note(
            "Hailo reports macs_per_image = {:,.0f} and ops_per_image = "
            "{:,.0f}, a ratio of exactly {:.1f}. That confirms the EXP-001 "
            "inference that the published operation count is 2 x MACs. Our "
            "independent MAC count of {:,} differs from Hailo's by "
            "{:+.2%}.".format(
                hailo_macs, hailo_ops, hailo_ops / hailo_macs, OUR_MACS,
                (OUR_MACS - hailo_macs) / hailo_macs,
            )
        )
        exp.note(
            "The compiled model has {} layers across {} device contexts, "
            "against 168 nodes in the ONNX. The expansion is the compiler's "
            "defusion and data-movement layers.".format(
                len(rows), model.get("number_of_contexts")
            )
        )
        exp.conclude(
            "Hailo's own compile-time report corroborates both published counts "
            "and their conventions, and supplies the per-layer and per-context "
            "structure that was previously UNKNOWN. It does not supply measured "
            "silicon latency: the report is a post-placement estimate with fps "
            "and latency reported as N/A at model level."
        )

        # readable output
        print("\nper-stage, from Hailo's compiled report:")
        print("{:<28} {:>7} {:>18} {:>8} {:>12} {:>10}".format(
            "stage", "layers", "MACs", "share", "min fps", "mac util"))
        for s, v in sorted(stage_table.items(), key=lambda kv: -kv[1]["macs"]):
            print("{:<28} {:>7} {:>18,.0f} {:>7.1%} {:>12} {:>10}".format(
                s, v["layers"], v["macs"], v["mac_share"],
                "{:.0f}".format(v["min_fps"]) if v["min_fps"] else "-",
                "{:.3f}".format(v["mean_effective_mac_util"])
                if v["mean_effective_mac_util"] is not None else "-",
            ))
        print("\nlayers per context: " + json.dumps(dict(sorted(contexts.items()))))
        print("\nrecorded as " + exp.id)


if __name__ == "__main__":
    main()
