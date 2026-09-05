"""Check that the Phase 1 documents agree with the experiment records.

Documents are written by hand; experiment results are not. This script asserts
that the headline numbers quoted across `docs/` and `PHASE_1_FINAL_REPORT.md`
match what the corresponding `experiments/EXP-xxx/metrics.json` actually
contains, so a transcription error cannot survive into the release.

Run it before tagging.

    python scripts/verify_claims.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP = REPO_ROOT / "experiments"
DOCS = list((REPO_ROOT / "docs").glob("*.md")) + [
    REPO_ROOT / "PHASE_1_FINAL_REPORT.md"
]

failures: list[str] = []
checks = 0


def metrics(exp_id: str) -> dict:
    return json.loads((EXP / exp_id / "metrics.json").read_text())["metrics"]


def check(label: str, got, want, tol=0.0):
    global checks
    checks += 1
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        ok = abs(got - want) <= tol
    else:
        ok = got == want
    status = "ok  " if ok else "FAIL"
    print("  [{}] {:<58} got {!r}".format(status, label, got))
    if not ok:
        failures.append("{}: got {!r}, expected {!r}".format(label, got, want))


def check_in_docs(label: str, pattern: str, expect_found: bool = True):
    """Assert a claim string appears (or does not appear) in the documents."""
    global checks
    checks += 1
    hits = [d.name for d in DOCS if re.search(pattern, d.read_text(encoding="utf-8"))]
    ok = bool(hits) == expect_found
    print("  [{}] {:<58} {}".format(
        "ok  " if ok else "FAIL", label,
        ",".join(hits[:3]) if hits else "(absent)"))
    if not ok:
        failures.append(label + ": pattern " + pattern +
                        (" not found" if expect_found else " unexpectedly found"))


def main() -> None:
    print("1. The published 8.223 is a D1 outlier rate, not an EPE")
    m5 = metrics("EXP-005")
    v = m5["variants"]["hailo_exact"]
    check("EXP-005 hailo_exact headline", round(v["headline_percent"], 3), 8.224, 0.001)
    check("EXP-005 gap vs published", round(v["gap_vs_published"], 4), 0.0007, 0.0002)
    check("EXP-005 published value", m5["hailo_published"], 8.223)
    check_in_docs("docs say it is D1, not EPE",
                  r"not an? end-point error|not end-point error|D1 outlier rate")
    check_in_docs("docs name three_pixel_correct_rate", r"three_pixel_correct_rate")

    print("\n2. The actual EPE is 1.313 px")
    epe = m5["variants"]["kitti_official_d1_all"]["pooled_metrics"]["epe"]
    check("EXP-005 official-protocol EPE", round(epe, 3), 1.313, 0.001)
    m12 = metrics("EXP-012")
    check("EXP-012 EPE agrees", round(m12["disparity_metrics_all"]["epe"], 3), 1.313, 0.001)
    check_in_docs("docs quote 1.313 px", r"1\.313 px")

    print("\n3. The cost volume performs no disparity search")
    m10 = metrics("EXP-010")
    check("EXP-010 all slices identical in every scene",
          m10["all_slices_identical_in_every_scene"], True)
    check("EXP-010 padding blocks all zero", m10["all_padding_blocks_zero"], True)
    maxdiff = max(max(p["max_abs_diff_vs_slice_0"]) for p in m10["per_scene"])
    check("EXP-010 max |slice_k - slice_0| over all scenes", maxdiff, 0.0)
    m7 = metrics("EXP-007")
    check("EXP-007 largest D1 penalty from corrupting the right image",
          round(m7["largest_d1_penalty_points"], 1), 90.0, 0.2)
    check_in_docs("docs state no disparity search",
                  r"no disparity search|performs no disparity search")

    print("\n4. Parameter accounting: 423,586 unique / 623,138 per occurrence")
    m1 = metrics("EXP-001")
    check("EXP-001 unique params", m1["params_unique_tensors"], 423_586)
    check("EXP-001 per-occurrence params", m1["params_per_node_occurrence"], 623_138)
    check("EXP-001 gap vs published 623.1K", m1["param_gap_vs_published"], 38)
    m17 = metrics("EXP-017")
    check("EXP-017 Hailo's own compiler reports weights", m17["hailo_weights"], 623_138.0)
    check("EXP-017 exact match flag", m17["weights_exact_match_with_our_per_occurrence"], True)
    check_in_docs("docs quote 423,586", r"423,586")
    check_in_docs("docs quote 623,138", r"623,138")

    print("\n5. Operation accounting: 112.2G is 2 x MACs")
    check("EXP-001 our MACs", m1["macs"], 56_039_313_792)
    check("EXP-001 2 x MACs", m1["ops_2x_macs"], 112_078_627_584)
    check("EXP-001 relative gap vs published", round(m1["ops_relative_gap_vs_published"], 4),
          -0.0011, 0.0002)
    check("EXP-017 Hailo ops/macs ratio", round(m17["hailo_ops_over_macs"], 2), 1.99, 0.02)
    check_in_docs("docs state ops = 2 x MACs", r"2 ?[x×] ?MACs")

    print("\n6. Refinement is the bottleneck")
    check("EXP-001 refinement MAC share", round(m1["refinement_mac_share"], 3), 0.906, 0.001)
    m13 = metrics("EXP-013")
    ref_share = m13["stage_latency"]["refinement"]["share_of_total"]
    check("EXP-013 refinement GPU time share", round(ref_share, 3), 0.733, 0.002)
    m14 = metrics("EXP-014")
    check("EXP-014 refinement CPU time share",
          round(m14["stage_latency"]["refinement"]["share_of_total"], 3), 0.758, 0.002)
    check("EXP-017 Hailo modelled bottleneck FPS", m17["bottleneck_fps_modelled"], 43.03)
    slowest = m17["slowest_layers_by_modelled_fps"][0]
    check("EXP-017 slowest layer stage", slowest["stage"], "refinement")
    check("EXP-017 slowest layer name", slowest["layer_name"], "conv50")
    check_in_docs("docs quote 90.6% MAC share", r"90\.6 ?%")
    check_in_docs("docs quote 73.3% GPU time share", r"73\.3 ?%")

    print("\n7. The Hailo app outputs disparity, not metric depth")
    app = (REPO_ROOT / "reference" / "upstream" / "extracted" / "hailo-apps-main"
           / "hailo_apps" / "cpp" / "depth_estimation_stereo")
    if app.exists():
        src = (app / "stereo_depth_estimation.cpp").read_text(
            encoding="utf-8", errors="replace")
        hdr = (app / "common.h").read_text(encoding="utf-8", errors="replace")
        blob = src + hdr
        # Probe for the arithmetic and the parameters a depth conversion would
        # need. Word boundaries matter: an earlier version of this check used a
        # bare "fB" and matched "surfboard" in a leftover COCO class table in
        # common.h -- itself further evidence the app is an adapted
        # classification example.
        for label, pattern in [
            ("baseline", r"\bbaseline\b"),
            ("focal length", r"\bfocal\b"),
            ("calibration", r"\bcalib"),
            ("intrinsics", r"\bintrinsic"),
            ("a depth assignment", r"\bdepth\s*="),
            ("division by disparity", r"/\s*disp"),
            ("OpenCV reprojection", r"reprojectImageTo3D|\bQ\s*matrix"),
        ]:
            check("app source contains no " + label,
                  bool(re.search(pattern, blob, re.I)), False)
        check("app postprocess casts to CV_8U", "CV_8U" in src, True)
    else:
        print("  [skip] reference app source not extracted")
    check_in_docs("docs state disparity not depth",
                  r"disparity pipeline, not a depth pipeline|disparity only")

    print("\n8. Depth error amplification")
    bands = m12["by_true_depth"]
    check("EXP-012 0-10m depth MAE", round(bands["0-10m"]["depth_mae_m"], 3), 0.213, 0.001)
    check("EXP-012 50-80m depth MAE", round(bands["50-80m"]["depth_mae_m"], 3), 9.158, 0.001)
    ratio = bands["50-80m"]["depth_mae_m"] / bands["0-10m"]["depth_mae_m"]
    check("EXP-012 amplification factor", round(ratio), 43, 1)
    check_in_docs("docs quote the 43x amplification", r"43 ?[x×]")

    print("\n9. Quantisation")
    m15 = metrics("EXP-015")
    t = m15["precision_comparison"]
    check("EXP-015 int8 D1 delta vs fp32",
          round(t["int8_onnxruntime_cpu"]["d1_delta_vs_fp32_points"], 2), 2.79, 0.01)
    check("EXP-015 fp16 D1 within noise",
          abs(t["fp16_torch_cuda"]["d1_percent"] - t["fp32_torch_cuda"]["d1_percent"]) < 0.05,
          True)

    print("\n10. Independent implementation matches the reference")
    m11 = metrics("EXP-011")
    check("EXP-011 final disparity relative difference",
          m11["final_disparity_relative_mean_abs_diff"] < 1e-6, True)
    check("EXP-011 parameters match", m11["model_parameters_unique"], 423_586)

    print("\n11. Superseded experiments are marked, not deleted")
    for d, f in (("EXP-008", "SUPERSEDED_FIELD.txt"), ("EXP-016", "CORRECTION.md")):
        check(d + " carries " + f, (EXP / d / f).exists(), True)
    ids = sorted(p.name for p in EXP.glob("EXP-*"))
    check("experiment ids are contiguous",
          ids == ["EXP-{:03d}".format(i) for i in range(1, len(ids) + 1)], True)

    print("\n" + "=" * 72)
    print("{} checks, {} failures".format(checks, len(failures)))
    for f in failures:
        print("  FAIL " + f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
