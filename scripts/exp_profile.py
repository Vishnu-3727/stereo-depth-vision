"""EXP: where the time and memory actually go.

Profiles the independent implementation (which matches the reference to 1e-7,
EXP-011) per stage on the RTX 4060 and on CPU, and measures peak memory.

The specific assumption under test is that static MAC share predicts runtime.
EXP-001 found the full-resolution refinement stage carries 90.6 % of the MACs
and Hailo's compiler spatially defuses exactly that region [SR-003], which makes
it the obvious candidate for the runtime bottleneck. Obvious is not measured,
and the charter is explicit that "fewer FLOPs = faster" must be tested rather
than assumed. So the per-stage MAC share and the per-stage measured share are
reported side by side.

**These are our measurements on our hardware.** They say nothing about Hailo
silicon and are never to be compared with Hailo's published figures.

    python scripts/exp_profile.py [--device cuda|cpu] [--iters N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.experiment import Experiment  # noqa: E402
from src.models.stereonet import StereoNet  # noqa: E402
from src.models.stereonet.onnx_weights import load_onnx_weights  # noqa: E402

MODEL = REPO_ROOT / "reference" / "onnx" / "stereonet.onnx"
OUT_DIR = REPO_ROOT / "results" / "profile"
H, W = 368, 1232

# Static MAC shares from EXP-001, for the comparison this experiment exists to make.
STATIC_MAC_SHARE = {
    "feature_extraction": 0.0512,   # both branches
    "cost_volume": 0.0,
    "aggregation": 0.0423,
    "regression": 0.0001,
    "refinement": 0.9064,
}


class Timer:
    """Wall-clock timing that synchronises CUDA before reading the clock.

    Without the synchronise, CUDA calls return immediately and every stage but
    the last appears free.
    """

    def __init__(self, device: torch.device) -> None:
        self.cuda = device.type == "cuda"

    def __call__(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()


def profile_stages(model, left, right, timer, iters: int) -> dict:
    """Time each stage by running the pipeline explicitly, stage by stage."""
    times: dict[str, list] = {k: [] for k in
                              ["feature_extraction", "cost_volume", "aggregation",
                               "regression", "refinement", "combine", "total"]}
    size = (left.shape[-2], left.shape[-1])
    with torch.no_grad():
        for _ in range(iters):
            t0 = timer()
            lf = model.feature_extractor(left)
            rf = model.feature_extractor(right)
            t1 = timer()
            vol = model.cost_volume(lf, rf)
            t2 = timer()
            cost = model.aggregation(vol)
            t3 = timer()
            d0 = model.regression(cost, size)
            t4 = timer()
            res = model.refinement(d0, left)
            t5 = timer()
            out = torch.relu(d0 + res)
            t6 = timer()
            _ = float(out.reshape(-1)[0])  # force completion

            times["feature_extraction"].append(t1 - t0)
            times["cost_volume"].append(t2 - t1)
            times["aggregation"].append(t3 - t2)
            times["regression"].append(t4 - t3)
            times["refinement"].append(t5 - t4)
            times["combine"].append(t6 - t5)
            times["total"].append(t6 - t0)
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    if args.device == "cuda" and device.type != "cuda":
        print("CUDA requested but unavailable; falling back to CPU")

    model = StereoNet().eval()
    load_onnx_weights(model, MODEL)
    model = model.to(device)

    torch.manual_seed(0)
    left = torch.randn(1, 3, H, W, device=device)
    right = torch.randn(1, 3, H, W, device=device)
    timer = Timer(device)

    config = {
        "dataset": None,
        "split": None,
        "resolution": [H, W],
        "crop": None,
        "disparity_range": "12 candidates at 1/16",
        "batch_size": 1,
        "precision": "fp32",
        "seed": 0,
        "device": str(device),
        "iterations": args.iters,
        "warmup": args.warmup,
        "implementation": "src/models/stereonet with reference weights",
    }

    with Experiment(
        "Per-stage latency and memory of StereoNet on local hardware",
        config=config,
    ) as exp:
        exp.note(
            "Our measurement on our hardware. Not a Hailo number, not a proxy "
            "for one, and never to be compared against Hailo's published "
            "latency or FPS."
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        profile_stages(model, left, right, timer, args.warmup)  # warm up
        times = profile_stages(model, left, right, timer, args.iters)

        summary = {}
        total_median = float(np.median(times["total"]))
        for stage, t in times.items():
            arr = np.array(t) * 1000.0  # ms
            summary[stage] = {
                "median_ms": float(np.median(arr)),
                "mean_ms": float(arr.mean()),
                "std_ms": float(arr.std()),
                "min_ms": float(arr.min()),
                "share_of_total": float(np.median(t) / total_median),
            }

        exp.metric("stage_latency", summary)
        exp.metric("total_median_ms", summary["total"]["median_ms"])
        exp.metric("fps_batch1", 1000.0 / summary["total"]["median_ms"])

        if device.type == "cuda":
            exp.metric(
                "peak_gpu_memory_bytes", int(torch.cuda.max_memory_allocated())
            )
            exp.metric(
                "peak_gpu_reserved_bytes", int(torch.cuda.max_memory_reserved())
            )
            props = torch.cuda.get_device_properties(0)
            exp.metric("gpu", {"name": props.name, "total_bytes": props.total_memory})

        # -- measured share against static MAC share -------------------------
        comparison = {}
        for stage, static in STATIC_MAC_SHARE.items():
            measured = summary[stage]["share_of_total"]
            comparison[stage] = {
                "static_mac_share": static,
                "measured_time_share": measured,
                "ratio_measured_to_static": (
                    measured / static if static > 0 else None
                ),
            }
        exp.metric("mac_share_vs_time_share", comparison)

        # -- what the cost volume costs despite zero MACs ---------------------
        cv = summary["cost_volume"]
        exp.metric(
            "zero_mac_stages_time_ms",
            {
                "cost_volume": cv["median_ms"],
                "cost_volume_share": cv["share_of_total"],
            },
        )

        (OUT_DIR / ("stage_latency_" + device.type + ".json")).write_text(
            json.dumps({"summary": summary, "comparison": comparison}, indent=2),
            encoding="utf-8",
        )

        lines = [
            "Per-stage latency, {} at {}x{}, batch 1, fp32, median of {} runs".format(
                device.type, H, W, args.iters
            ),
            "",
            "{:<22} {:>10} {:>9} {:>12} {:>12} {:>9}".format(
                "stage", "median ms", "share", "MAC share", "time/MAC", "std ms"
            ),
        ]
        for stage in ["feature_extraction", "cost_volume", "aggregation",
                      "regression", "refinement", "combine"]:
            s = summary[stage]
            c = comparison.get(stage)
            static = "{:>11.1f}%".format(100 * c["static_mac_share"]) if c else " " * 12
            ratio = (
                "{:>12.2f}".format(c["ratio_measured_to_static"])
                if c and c["ratio_measured_to_static"] else "{:>12}".format("n/a")
            )
            lines.append(
                "{:<22} {:>10.3f} {:>8.1f}% {} {} {:>9.3f}".format(
                    stage, s["median_ms"], 100 * s["share_of_total"],
                    static, ratio, s["std_ms"],
                )
            )
        lines.append("-" * 78)
        lines.append(
            "{:<22} {:>10.3f} {:>8} {:>12} {:>12} {:>9.3f}".format(
                "TOTAL", summary["total"]["median_ms"], "", "", "",
                summary["total"]["std_ms"],
            )
        )
        lines.append("")
        lines.append(
            "{:.1f} FPS at batch 1 on {}".format(
                1000.0 / summary["total"]["median_ms"], device.type
            )
        )
        text = "\n".join(lines)
        (OUT_DIR / ("report_" + device.type + ".txt")).write_text(text, encoding="utf-8")

        ref = summary["refinement"]
        exp.note(
            "Refinement takes {:.3f} ms of a {:.3f} ms total, {:.1%} of the "
            "time against {:.1%} of the MACs.".format(
                ref["median_ms"], summary["total"]["median_ms"],
                ref["share_of_total"], STATIC_MAC_SHARE["refinement"],
            )
        )
        exp.note(
            "The cost volume performs zero MACs yet takes {:.3f} ms, {:.1%} of "
            "the total. A MAC count cannot see this stage at all, which is a "
            "concrete case against using FLOPs as a latency proxy.".format(
                cv["median_ms"], cv["share_of_total"]
            )
        )
        exp.conclude(
            "Per-stage latency measured and set against static MAC share. "
            "Interpretation is in docs/compute_profile.md; the model is "
            "unchanged."
        )
        print("\n" + text + "\n\nrecorded as " + exp.id)


if __name__ == "__main__":
    main()
