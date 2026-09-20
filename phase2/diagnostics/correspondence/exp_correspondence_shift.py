"""EXP-CORRESPONDENCE-SHIFT-001 -- does the readout path perform disparity search?

Inference only. No training, no architecture change, no modification of any
historical record. The block-count campaign is CLOSED and is not reopened here.

Pre-registered in the record directory's `PREREGISTRATION.md`, frozen before any
sweep was run.

THE QUESTION
------------
The existing stereo probes corrupt the right image and score `disparity_final`,
which is `disparity_initial + refinement(disparity_initial, left)`. The
refinement is guided by the LEFT image only and supplies most of the magnitude
(`disparity_initial` is bounded by 11 candidates; GT reaches ~192 px). So those
probes establish binocular dependence, not correspondence -- and EXP-010 showed a
provably degenerate cost volume coexisting with large downstream right-image
dependence.

This diagnostic holds the left image FIXED and translates only the right image.
A monocular pathway cannot respond at all. The prediction is a signed slope, not
merely "something changed".

GEOMETRY (traced from src/models/stereonet/cost_volume.py, not assumed)
-----------------------------------------------------------------------
    shift_left(x, k) = F.pad(x, (0, k))[..., k:]      ->  shifted[u] = left[u+k]
    slice k          = left_feat[u+k] - right_feat[u]

Move right-image CONTENT right by Delta full-res px: right'[y, x] = right[y, x-Delta].
Feature stride 16, so right'_feat[u] ~= right_feat[u - delta], delta = Delta/16.
Baseline match for right-column v: left_feat[v+d] ~= right_feat[v]. With v = u-delta:
left_feat[u-delta+d] ~= right'_feat[u], so the matching slice obeys u+k = u-delta+d:

    k = d - Delta/16        ->      slope = -1 candidate / 16 px = -0.0625 cand/px

ANCHORS
-------
zero anchor           : shift="none" makes all 12 slices identical, so the
                        standardised cost is exactly 0 for every candidate,
                        softmax is uniform, and soft-argmin == 5.5 regardless of
                        input. Expected response to any translation: exactly 0.
correspondence anchor : -0.0625 candidate per pixel.

EXECUTION ORDER IS BINDING: the negative control (Stage A) runs and is recorded
before any positive checkpoint is inspected. If it shows a systematic non-zero
horizontal slope the diagnostic is invalid and the positive controls must not be
run or interpreted.

    python .../exp_correspondence_shift.py stage_a --out DIR
    python .../exp_correspondence_shift.py stage_c --out DIR
    python .../exp_correspondence_shift.py report  --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Match the determinism protocol of the campaign that produced these checkpoints.
CUBLAS_CONFIG = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_CONFIG:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_CONFIG

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

from src.models.stereonet.stereonet import StereoNet, StereoNetConfig  # noqa: E402
from src.datasets.kitti2015 import normalize                        # noqa: E402
from phase2.models import scaled_regression                         # noqa: E402
from phase2.viz import core                                         # noqa: E402

# ------------------------------------------------------------------ frozen setup
FOCUS_SCENES = [27, 0, 31, 6]          # the existing probe scene set, unchanged
SPLIT = "hailo_val"
H_SHIFTS = (-32, -16, 0, 16, 32)       # full-resolution pixels, right image only
V_SHIFTS = (-32, -16, 16, 32)
MAX_SHIFT = 32
CANDIDATES = 12
FEATURE_STRIDE = 16
UNIFORM_SOFT_ARGMIN = (CANDIDATES - 1) / 2.0            # 5.5
CORRESPONDENCE_ANCHOR = -1.0 / FEATURE_STRIDE           # -0.0625 candidate / px

# Mask, frozen before any positive result was seen.
MASK_LO, MASK_HI = 1.0, 10.0           # baseline candidate must be interior
BORDER = MAX_SHIFT                      # drop 32 px on all four edges

CHECKPOINTS = {
    "NEGATIVE_CONTROL_shift_none": {
        "path": ("phase2/factorial/shift_none_standardized/20260909T071500Z/"
                 "checkpoints/EXP-FACTORIAL-SHIFT-NONE-STANDARDISED-001-ARM-A"
                 "_checkpoint.pth"),
        "cost_volume_shift": "none", "blocks": 6, "seed": 0,
        "role": "negative control -- cost volume provably degenerate (EXP-010)",
        "stage": "A"},
    "6b_seed0": {
        "path": ("phase2/diagnostics/determinism/20260909T041500Z_baseline/"
                 "checkpoints/STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA"
                 "_checkpoint.pth"),
        "cost_volume_shift": "left", "blocks": 6, "seed": 0,
        "role": "positive", "stage": "C"},
    "6b_seed1": {
        "path": ("phase2/factorial/block_count_full/20260910T005550Z/checkpoints/"
                 "EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED1_checkpoint.pth"),
        "cost_volume_shift": "left", "blocks": 6, "seed": 1,
        "role": "positive", "stage": "C"},
    "6b_seed2": {
        "path": ("phase2/factorial/block_count_full/20260910T005550Z/checkpoints/"
                 "EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED2_checkpoint.pth"),
        "cost_volume_shift": "left", "blocks": 6, "seed": 2,
        "role": "positive", "stage": "C"},
}


# ------------------------------------------------------------------- perturbation
def translate(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate image CONTENT by (dx, dy) pixels, zero-filling what it vacates.

    ONE code path for both axes -- the horizontal sweep and the vertical null
    differ only in which argument is non-zero. Positive dx moves content right:
    out[y, x] = image[y, x - dx]. Positive dy moves content down.
    """
    out = np.zeros_like(image)
    h, w = image.shape[:2]
    xs_dst = slice(max(dx, 0), w + min(dx, 0))
    xs_src = slice(max(-dx, 0), w + min(-dx, 0))
    ys_dst = slice(max(dy, 0), h + min(dy, 0))
    ys_src = slice(max(-dy, 0), h + min(-dy, 0))
    out[ys_dst, xs_dst] = image[ys_src, xs_src]
    return out


def build_model(key: str, device: str) -> StereoNet:
    spec = CHECKPOINTS[key]
    path = REPO_ROOT / spec["path"]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    recorded = ckpt.get("config", {}).get("cost_volume_shift")
    if recorded is not None and recorded != spec["cost_volume_shift"]:
        raise ValueError("checkpoint {} records shift '{}' but {} expects '{}'"
                         .format(path.name, recorded, key,
                                 spec["cost_volume_shift"]))
    model = StereoNet(StereoNetConfig(
        cost_volume_shift=spec["cost_volume_shift"]))
    scaled_regression.apply_to(model)                  # the H2 readout, unchanged
    model.load_state_dict(ckpt["model"])
    return model.eval().to(device)


@torch.no_grad()
def disparity_initial(model: StereoNet, left: np.ndarray, right: np.ndarray,
                      device: str) -> np.ndarray:
    """`disparity_initial` only -- the candidate-space readout, never the final map."""
    _, stages = model(torch.from_numpy(normalize(left)).to(device),
                      torch.from_numpy(normalize(right)).to(device),
                      return_stages=True)
    return stages["disparity_initial"][0, 0].cpu().numpy().astype(np.float64)


def ols_slope(xs, ys) -> dict:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    slope = float(((x - xm) * (y - ym)).sum() / sxx) if sxx > 0 else float("nan")
    intercept = float(ym - slope * xm)
    resid = y - (slope * x + intercept)
    ss_tot = float(((y - ym) ** 2).sum())
    return {"slope_candidates_per_px": slope, "intercept_candidates": intercept,
            "r_squared": float(1.0 - (resid ** 2).sum() / ss_tot)
            if ss_tot > 0 else float("nan"),
            "max_abs_residual": float(np.abs(resid).max()), "n_points": int(x.size)}


def monotone(values) -> bool:
    d = np.diff(np.asarray(values, dtype=np.float64))
    return bool(np.all(d < 0) or np.all(d > 0))


# ------------------------------------------------------------------------ sweep
def sweep_checkpoint(key: str, device: str) -> dict:
    model = build_model(key, device)
    spec = CHECKPOINTS[key]
    scenes = [core.load_scene(i, split=SPLIT) for i in FOCUS_SCENES]
    rows, per_scene = [], {}

    for scene in scenes:
        base = disparity_initial(model, scene.left, scene.right, device)
        h, w = base.shape
        border = np.zeros_like(base, dtype=bool)
        border[BORDER:h - BORDER, BORDER:w - BORDER] = True
        mask = border & (base >= MASK_LO) & (base <= MASK_HI)
        retained = int(mask.sum())
        base_mean = float(base[mask].mean()) if retained else float("nan")
        base_median = float(np.median(base[mask])) if retained else float("nan")

        entries = {}
        for axis, shifts in (("horizontal", H_SHIFTS), ("vertical", V_SHIFTS)):
            for delta in shifts:
                if delta == 0:
                    cur, mean_v, median_v = base, base_mean, base_median
                else:
                    dx, dy = (delta, 0) if axis == "horizontal" else (0, delta)
                    cur = disparity_initial(
                        model, scene.left, translate(scene.right, dx, dy), device)
                    mean_v = float(cur[mask].mean()) if retained else float("nan")
                    median_v = (float(np.median(cur[mask])) if retained
                                else float("nan"))
                per_pixel = (cur[mask] - base[mask]) if retained else np.array([])
                entries["{}_{:+d}".format(axis, delta)] = {
                    "axis": axis, "shift_px": int(delta),
                    "dx": int(delta) if axis == "horizontal" else 0,
                    "dy": 0 if axis == "horizontal" else int(delta),
                    "retained_pixels": retained,
                    "disparity_initial_mean": mean_v,
                    "disparity_initial_median": median_v,
                    "response_mean": mean_v - base_mean,
                    "response_median": median_v - base_median,
                    "per_pixel_response_median": (float(np.median(per_pixel))
                                                  if retained else float("nan")),
                }
                rows.append(dict(entries["{}_{:+d}".format(axis, delta)],
                                 model=key, seed=spec["seed"], scene=scene.name))
        per_scene[scene.name] = {
            "retained_pixels": retained,
            "retained_fraction": retained / float(base.size),
            "baseline_mean": base_mean, "baseline_median": base_median,
            "baseline_min": float(base.min()), "baseline_max": float(base.max()),
            "baseline_std": float(base.std()),
            "conditions": entries,
        }

    # ---- pooled fits, weighted equally across the four scenes
    def pooled(axis, shifts, field):
        xs, ys = [], []
        for delta in shifts:
            k = "{}_{:+d}".format(axis, delta)
            vals = [per_scene[s]["conditions"][k][field] for s in per_scene]
            xs.append(delta)
            ys.append(float(np.mean(vals)))
        return xs, ys

    hx, hy_mean = pooled("horizontal", H_SHIFTS, "response_mean")
    _, hy_med = pooled("horizontal", H_SHIFTS, "response_median")
    # the vertical fit shares the Delta = 0 baseline point (response 0 by definition)
    vx, vy_mean = pooled("vertical", V_SHIFTS, "response_mean")
    _, vy_med = pooled("vertical", V_SHIFTS, "response_median")
    vx, vy_mean, vy_med = ([0] + list(vx), [0.0] + list(vy_mean),
                           [0.0] + list(vy_med))
    order = np.argsort(vx)
    vx = list(np.array(vx)[order])
    vy_mean = list(np.array(vy_mean)[order])
    vy_med = list(np.array(vy_med)[order])

    fits = {
        "horizontal_mean": ols_slope(hx, hy_mean),
        "horizontal_median": ols_slope(hx, hy_med),
        "vertical_mean": ols_slope(vx, vy_mean),
        "vertical_median": ols_slope(vx, vy_med),
    }
    h_slope = fits["horizontal_mean"]["slope_candidates_per_px"]
    v_slope = fits["vertical_mean"]["slope_candidates_per_px"]
    summary = {
        "horizontal_points": {"shift_px": hx, "response_mean": hy_mean,
                              "response_median": hy_med},
        "vertical_points": {"shift_px": vx, "response_mean": vy_mean,
                            "response_median": vy_med},
        "horizontal_monotone": monotone(hy_mean),
        "horizontal_sign_negative": bool(h_slope < 0),
        "fraction_of_correspondence_anchor":
            float(h_slope / CORRESPONDENCE_ANCHOR),
        "vertical_fraction_of_correspondence_anchor":
            float(v_slope / CORRESPONDENCE_ANCHOR),
        "horizontal_over_vertical_abs_ratio":
            float(abs(h_slope) / abs(v_slope)) if v_slope != 0 else float("inf"),
    }
    return {"model": key, "spec": spec,
            "checkpoint_sha256": core.sha256(REPO_ROOT / spec["path"]),
            "per_scene": per_scene, "fits": fits, "summary": summary,
            "rows": rows}


# ------------------------------------------------------------- reference frame
def reference_frame_note() -> dict:
    """Traced, documented, NOT corrected."""
    return {
        "cost_volume_slice": "slice k = left_feat[u+k] - right_feat[u], stored at output column u",
        "visualizer_convention": "viz/core.correspondence: x_right = x_left - d (left-referenced)",
        "observation": (
            "the cost volume is indexed by the RIGHT image's column while ground "
            "truth and the refinement guidance are left-referenced, so the stored "
            "candidate carries a reference-frame offset equal to the disparity"),
        "effect_on_this_diagnostic": (
            "affects the INTERCEPT, not the SLOPE: the matching slice obeys "
            "k = d - Delta/16 regardless of which frame the value is stored in, "
            "so the predicted slope -1/16 candidate per pixel is unchanged"),
        "status": "PRESERVED, not corrected; reported only",
        "affects_block_count_campaign": (
            "no -- all nine campaign runs share this construction identically, so "
            "it cannot have biased one arm against another"),
    }


def environment() -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None),
            "deterministic_env": {"CUBLAS_WORKSPACE_CONFIG":
                                  os.environ.get("CUBLAS_WORKSPACE_CONFIG")}}


def provenance() -> dict:
    import subprocess
    def run(*a):
        try:
            return subprocess.run(a, cwd=REPO_ROOT, capture_output=True,
                                  text=True, check=False).stdout.strip()
        except Exception as exc:                                # pragma: no cover
            return "unavailable: {}".format(exc)
    return {"head": run("git", "rev-parse", "HEAD"),
            "phase_1_frozen_commit": run("git", "rev-parse",
                                         "phase-1-frozen^{commit}"),
            "phase_1_diff_vs_frozen": run("git", "diff", "phase-1-frozen",
                                          "--", "src", "scripts"),
            "status_short": run("git", "status", "--short"),
            "phase_2_committed": False}


# --------------------------------------------------------------------- commands
def _json_default(obj):
    """numpy scalars serialise as NUMBERS, never as strings.

    Stringifying a measurement would silently turn a float into text in the
    machine-readable record, so unknown types raise instead.
    """
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError("not JSON serialisable: {}".format(type(obj).__name__))


def _write(out: Path, name: str, payload: dict) -> None:
    (out / name).write_text(json.dumps(payload, indent=2, default=_json_default),
                            encoding="utf-8")


def cmd_stage_a(args) -> int:
    out = Path(args.out).resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    res = sweep_checkpoint("NEGATIVE_CONTROL_shift_none", device)
    res["wall_clock_s"] = time.time() - t0
    res["environment"] = environment()
    res["provenance"] = provenance()
    res["reference_frame"] = reference_frame_note()

    s = res["summary"]
    h = res["fits"]["horizontal_mean"]["slope_candidates_per_px"]
    v = res["fits"]["vertical_mean"]["slope_candidates_per_px"]
    baselines = [res["per_scene"][k]["baseline_mean"] for k in res["per_scene"]]
    res["falsification"] = {
        "expected_horizontal_slope": 0.0,
        "expected_disparity_initial": UNIFORM_SOFT_ARGMIN,
        "observed_horizontal_slope": h,
        "observed_vertical_slope": v,
        "observed_baseline_means": baselines,
        "max_abs_deviation_from_5_5": float(
            max(abs(b - UNIFORM_SOFT_ARGMIN) for b in baselines)),
        "max_abs_response_any_condition": float(max(
            abs(r["response_mean"]) for r in res["rows"])),
    }
    _write(out, "stage_a_negative_control.json", res)
    print(json.dumps({"stage": "A", "model": res["model"],
                      "falsification": res["falsification"],
                      "horizontal_points": s["horizontal_points"],
                      "vertical_points": s["vertical_points"],
                      "retained": {k: res["per_scene"][k]["retained_pixels"]
                                   for k in res["per_scene"]},
                      "wall_clock_s": res["wall_clock_s"]},
                     indent=2, default=str))
    return 0


def cmd_stage_c(args) -> int:
    out = Path(args.out).resolve()
    a = json.loads((out / "stage_a_negative_control.json").read_text(
        encoding="utf-8"))
    if a["falsification"]["observed_horizontal_slope"] is None:
        print("STOP: stage A not completed.")
        return 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    t0 = time.time()
    for key, spec in CHECKPOINTS.items():
        if spec["stage"] != "C":
            continue
        r = sweep_checkpoint(key, device)
        results[key] = r
        f = r["fits"]["horizontal_mean"]
        print("{:<10} h-slope {:+.6f} cand/px  ({:.1f}% of anchor)  "
              "v-slope {:+.6f}  monotone={}  R2={:.4f}".format(
                  key, f["slope_candidates_per_px"],
                  100.0 * r["summary"]["fraction_of_correspondence_anchor"],
                  r["fits"]["vertical_mean"]["slope_candidates_per_px"],
                  r["summary"]["horizontal_monotone"], f["r_squared"]))
    payload = {"stage": "C", "results": results,
               "wall_clock_s": time.time() - t0,
               "environment": environment(), "provenance": provenance(),
               "reference_frame": reference_frame_note()}
    _write(out, "stage_c_positive.json", payload)
    return 0


def cmd_report(args) -> int:
    out = Path(args.out).resolve()
    a = json.loads((out / "stage_a_negative_control.json").read_text(encoding="utf-8"))
    c = json.loads((out / "stage_c_positive.json").read_text(encoding="utf-8"))
    neg = a["fits"]["horizontal_mean"]["slope_candidates_per_px"]

    combined = {
        "experiment": "EXP-CORRESPONDENCE-SHIFT-001",
        "kind": "inference-only diagnostic; no training; no architecture change",
        "question": ("does disparity_initial respond to controlled horizontal "
                     "translation of the right image with the geometrically "
                     "predicted candidate shift?"),
        "anchors": {"zero_anchor_slope": 0.0,
                    "zero_anchor_disparity_initial": UNIFORM_SOFT_ARGMIN,
                    "correspondence_anchor_slope": CORRESPONDENCE_ANCHOR},
        "mask": {"candidate_interior": [MASK_LO, MASK_HI], "border_px": BORDER,
                 "frozen_before_positive_results": True},
        "scenes": FOCUS_SCENES, "split": SPLIT,
        "horizontal_shifts_px": list(H_SHIFTS),
        "vertical_shifts_px": list(V_SHIFTS),
        "negative_control": {
            "model": a["model"], "checkpoint_sha256": a["checkpoint_sha256"],
            "falsification": a["falsification"],
            "fits": a["fits"], "summary": a["summary"]},
        "positive": {k: {"checkpoint_sha256": v["checkpoint_sha256"],
                         "seed": v["spec"]["seed"], "fits": v["fits"],
                         "summary": v["summary"],
                         "retained_pixels": {s: v["per_scene"][s]["retained_pixels"]
                                             for s in v["per_scene"]}}
                     for k, v in c["results"].items()},
        "negative_control_slope": neg,
        "reference_frame": a["reference_frame"],
        "environment": a["environment"], "provenance": c["provenance"],
        "warnings": [],
    }

    # ---- pre-registered pattern logic, applied mechanically
    hs = {k: v["fits"]["horizontal_mean"]["slope_candidates_per_px"]
          for k, v in c["results"].items()}
    vs = {k: v["fits"]["vertical_mean"]["slope_candidates_per_px"]
          for k, v in c["results"].items()}
    mono = {k: v["summary"]["horizontal_monotone"] for k, v in c["results"].items()}
    frac = {k: v["summary"]["fraction_of_correspondence_anchor"]
            for k, v in c["results"].items()}

    all_negative = all(s < 0 for s in hs.values())
    all_monotone = all(mono.values())
    # "substantially distinct from the vertical null" and "approximately matches
    # the degenerate control" are read against the two anchors, not a tuned cut:
    # the horizontal slope is compared with |vertical| and with |negative control|.
    distinct_from_vertical = all(abs(hs[k]) > 3.0 * abs(vs[k]) for k in hs)
    distinct_from_negative = all(abs(hs[k]) > 3.0 * abs(neg) for k in hs)
    near_anchor = all(0.5 <= frac[k] <= 1.5 for k in frac)

    if not (all_negative and distinct_from_negative):
        pattern, label = "A", "NO-CORRESPONDENCE-EVIDENCE"
    elif all_negative and all_monotone and distinct_from_vertical and near_anchor:
        pattern, label = "C", "CORRESPONDENCE-CONSISTENT-CANDIDATE-SEARCH"
    else:
        pattern, label = "B", "RIGHT-IMAGE-SENSITIVITY-WITH-HORIZONTAL-GEOMETRY"

    combined["pattern_inputs"] = {
        "horizontal_slopes": hs, "vertical_slopes": vs,
        "monotone": mono, "fraction_of_anchor": frac,
        "all_signs_negative": all_negative, "all_monotone": all_monotone,
        "distinct_from_vertical_null": distinct_from_vertical,
        "distinct_from_negative_control": distinct_from_negative,
        "consistent_with_anchor": near_anchor}
    combined["PATTERN"] = pattern
    combined["VERDICT"] = label
    _write(out, "results.json", combined)
    print(json.dumps({"PATTERN": pattern, "VERDICT": label,
                      "pattern_inputs": combined["pattern_inputs"],
                      "negative_control_slope": neg}, indent=2, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stage_a", "stage_c", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
    args = ap.parse_args()
    return {"stage_a": cmd_stage_a, "stage_c": cmd_stage_c,
            "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
