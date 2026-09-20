"""EXP-CORRESPONDENCE-INDEX-001 -- candidate-axis re-indexing, inference only.

Tests CANDIDATE-COORDINATE SENSITIVITY (level C) of the frozen aggregation /
readout path. It does NOT test geometric correspondence and does NOT test
disparity search. The words correspondence / matching / disparity search are not
synonyms for a positive result here.

No training, no optimizer, no weight change, no architecture change, no image
change, no feature change, no refinement. The closed block-count campaign stays
closed.

THE INTERVENTION
----------------
Build the cost volume once per (checkpoint, scene), cache it, then re-index its
candidate axis only:

    V'(k) = V(pi(k))        pi a bijection on {0..11},  dim=2

and read `disparity_initial` from aggregation -> regression.

WHY THIS IS DECISIVE FOR THE NEGATIVE CONTROL
---------------------------------------------
For `shift="none"`, `reference_shift` is a no-op, so V(c,k,u,v) = D(c,u,v) for
every k. Hence for ANY permutation pi, V[:,:,pi] equals V element-wise. The
degenerate model's response is zero as an ALGEBRAIC IDENTITY, not as an
empirical expectation -- unlike every image-perturbation diagnostic tried
before, which the same model could and did respond to.

MANDATORY: every arm, identity included, is built with the SAME
`index_select(dim=2, ...)` call. `build_cost_volume` returns a permuted view
with non-contiguous strides; `index_select` materialises a contiguous tensor.
Mixing the two across arms could select a different cuDNN algorithm and change
the last bits, corrupting the gate for a purely numerical reason.

    python .../exp_correspondence_index.py gate     --out DIR
    python .../exp_correspondence_index.py positive --out DIR
    python .../exp_correspondence_index.py report   --out DIR
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

CUBLAS_CONFIG = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_CONFIG:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_CONFIG

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

from src.models.stereonet.stereonet import StereoNet, StereoNetConfig  # noqa: E402
from src.datasets.kitti2015 import normalize                        # noqa: E402
from phase2.models import scaled_regression                         # noqa: E402
from phase2.viz import core                                         # noqa: E402

# ------------------------------------------------------------- frozen constants
FOCUS_SCENES = [27, 0, 31, 6]
SPLIT = "hailo_val"
CANDIDATES = 12

# V'(k) = V(pi(k)); pi_m(k) = (k - m) mod 12 moves content at j to j+m.
def pi_m(m: int) -> list[int]:
    return [(k - m) % CANDIDATES for k in range(CANDIDATES)]


# Frozen BEFORE execution: numpy.random.default_rng(20260910).permutation(12).
RANDOM_PERM = [1, 10, 9, 5, 3, 8, 11, 0, 6, 7, 2, 4]

ARMS = {
    "identity": pi_m(0),
    "m_plus_1": pi_m(+1),
    "m_plus_2": pi_m(+2),
    "m_minus_1": pi_m(-1),
    "random": list(RANDOM_PERM),
}
ARM_ORDER = ["m_minus_1", "identity", "m_plus_1", "m_plus_2", "random"]

# Interior mask, geometry-derived, from GROUND TRUTH only -- identical pixel set
# for every checkpoint and every arm. No spatial border mask: no image is moved.
GT_CAND_LO, GT_CAND_HI = 2.0, 8.0
FEATURE_STRIDE = 16

CHECKPOINTS = {
    "NEG_shift_none": {
        "path": ("phase2/factorial/shift_none_standardized/20260909T071500Z/"
                 "checkpoints/EXP-FACTORIAL-SHIFT-NONE-STANDARDISED-001-ARM-A"
                 "_checkpoint.pth"),
        "cost_volume_shift": "none", "blocks": 6, "seed": 0, "role": "negative control"},
    "6b_seed0": {
        "path": ("phase2/diagnostics/determinism/20260909T041500Z_baseline/"
                 "checkpoints/STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA_checkpoint.pth"),
        "cost_volume_shift": "left", "blocks": 6, "seed": 0, "role": "positive"},
    "6b_seed1": {
        "path": ("phase2/factorial/block_count_full/20260910T005550Z/checkpoints/"
                 "EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED1_checkpoint.pth"),
        "cost_volume_shift": "left", "blocks": 6, "seed": 1, "role": "positive"},
    "6b_seed2": {
        "path": ("phase2/factorial/block_count_full/20260910T005550Z/checkpoints/"
                 "EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED2_checkpoint.pth"),
        "cost_volume_shift": "left", "blocks": 6, "seed": 2, "role": "positive"},
}


def enable_determinism() -> dict:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_CONFIG
    return {"torch.use_deterministic_algorithms": True,
            "torch.backends.cudnn.deterministic": True,
            "torch.backends.cudnn.benchmark": False,
            "CUBLAS_WORKSPACE_CONFIG": CUBLAS_CONFIG}


def build_model(key: str, device: str) -> StereoNet:
    spec = CHECKPOINTS[key]
    path = REPO_ROOT / spec["path"]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)   # read-only
    recorded = ckpt.get("config", {}).get("cost_volume_shift")
    if recorded is not None and recorded != spec["cost_volume_shift"]:
        raise ValueError("checkpoint {} records shift '{}' but {} expects '{}'"
                         .format(path.name, recorded, key, spec["cost_volume_shift"]))
    model = StereoNet(StereoNetConfig(cost_volume_shift=spec["cost_volume_shift"]))
    scaled_regression.apply_to(model)
    model.load_state_dict(ckpt["model"])
    return model.eval().to(device)


@torch.no_grad()
def build_volume(model: StereoNet, scene, device: str):
    """Feature extraction + cost volume, ONCE. Cached by the caller."""
    left = torch.from_numpy(normalize(scene.left)).to(device)
    right = torch.from_numpy(normalize(scene.right)).to(device)
    lf = model.feature_extractor(left)
    rf = model.feature_extractor(right)
    volume = model.cost_volume(lf, rf)              # (B, C, D, H, W)
    return volume, (left.shape[-2], left.shape[-1])


@torch.no_grad()
def readout(model: StereoNet, volume: torch.Tensor, perm: list[int],
            size, device: str) -> torch.Tensor:
    """index_select on dim 2 -- the SAME op for every arm, identity included."""
    idx = torch.tensor(perm, dtype=torch.long, device=volume.device)
    permuted = volume.index_select(dim=2, index=idx)
    cost = model.aggregation(permuted)
    return model.regression(cost, size)[0, 0]        # disparity_initial, no refinement


def tensor_facts(t: torch.Tensor) -> dict:
    return {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device),
            "is_contiguous": bool(t.is_contiguous())}


def summarise(d: np.ndarray, mask: np.ndarray) -> dict:
    v = d[mask]
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std()), "min": float(v.min()), "max": float(v.max()),
            "valid_pixels": int(mask.sum())}


def run_checkpoint(key: str, device: str) -> dict:
    model = build_model(key, device)
    spec = CHECKPOINTS[key]
    scenes = [core.load_scene(i, split=SPLIT) for i in FOCUS_SCENES]
    per_scene, rows = {}, []

    for scene in scenes:
        volume, size = build_volume(model, scene, device)
        idx0 = torch.tensor(ARMS["identity"], dtype=torch.long, device=volume.device)
        identity_vol = volume.index_select(dim=2, index=idx0)
        facts = {"original_volume": tensor_facts(volume),
                 "identity_indexed_volume": tensor_facts(identity_vol),
                 "max_abs_diff_identity_vs_original":
                     float((identity_vol - volume).abs().max())}

        gt = scene.gt_disparity.astype(np.float64)
        cand = gt / FEATURE_STRIDE
        mask = (gt > 0) & (cand >= GT_CAND_LO) & (cand <= GT_CAND_HI)

        base = readout(model, volume, ARMS["identity"], size, device)
        base_np = base.double().cpu().numpy()
        arms = {}
        for arm in ARM_ORDER:
            cur = (base if arm == "identity"
                   else readout(model, volume, ARMS[arm], size, device))
            cur_np = base_np if arm == "identity" else cur.double().cpu().numpy()
            vol_perm = volume.index_select(
                dim=2, index=torch.tensor(ARMS[arm], dtype=torch.long,
                                          device=volume.device))
            entry = summarise(cur_np, mask)
            entry.update({
                "permutation": ARMS[arm],
                "volume_max_abs_diff_vs_identity":
                    float((vol_perm - identity_vol).abs().max()),
                "disparity_initial_max_abs_diff_vs_identity":
                    float(np.abs(cur_np - base_np).max()),
                "delta_mean_vs_identity": None, "delta_median_vs_identity": None,
                "per_pixel_median_delta": float(np.median(cur_np[mask] - base_np[mask])),
            })
            arms[arm] = entry
            del vol_perm
        b = arms["identity"]
        for arm in ARM_ORDER:
            arms[arm]["delta_mean_vs_identity"] = arms[arm]["mean"] - b["mean"]
            arms[arm]["delta_median_vs_identity"] = arms[arm]["median"] - b["median"]
            rows.append({"checkpoint": key, "scene": scene.name, "arm": arm,
                         "permutation": ARMS[arm],
                         "mean": arms[arm]["mean"], "median": arms[arm]["median"],
                         "std": arms[arm]["std"],
                         "delta_mean_vs_identity": arms[arm]["delta_mean_vs_identity"],
                         "delta_median_vs_identity":
                             arms[arm]["delta_median_vs_identity"],
                         "per_pixel_median_delta": arms[arm]["per_pixel_median_delta"],
                         "valid_pixels": arms[arm]["valid_pixels"]})
        per_scene[scene.name] = {"tensor_facts": facts,
                                 "mask_valid_pixels": int(mask.sum()),
                                 "mask_fraction": float(mask.mean()),
                                 "arms": arms}
        del volume, identity_vol

    pooled = {arm: {stat: float(np.mean([per_scene[s]["arms"][arm][stat]
                                         for s in per_scene]))
                    for stat in ("mean", "median", "delta_mean_vs_identity",
                                 "delta_median_vs_identity",
                                 "per_pixel_median_delta")}
              for arm in ARM_ORDER}
    pooled_maxdiff = {
        arm: {"volume": float(max(per_scene[s]["arms"][arm]
                                  ["volume_max_abs_diff_vs_identity"]
                                  for s in per_scene)),
              "disparity_initial": float(max(per_scene[s]["arms"][arm]
                                             ["disparity_initial_max_abs_diff_vs_identity"]
                                             for s in per_scene))}
        for arm in ARM_ORDER}
    return {"checkpoint": key, "spec": spec,
            "checkpoint_sha256": core.sha256(REPO_ROOT / spec["path"]),
            "per_scene": per_scene, "pooled": pooled,
            "pooled_max_abs_diff_vs_identity": pooled_maxdiff, "rows": rows}


def environment() -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def provenance() -> dict:
    import subprocess
    def run(*a):
        return subprocess.run(a, cwd=REPO_ROOT, capture_output=True, text=True,
                              check=False).stdout.strip()
    return {"head": run("git", "rev-parse", "HEAD"),
            "phase_1_frozen_commit": run("git", "rev-parse", "phase-1-frozen^{commit}"),
            "phase_1_diff_vs_frozen": run("git", "diff", "phase-1-frozen", "--",
                                          "src", "scripts"),
            "status_short": run("git", "status", "--short"),
            "phase_2_committed": False}


def _default(o):
    if hasattr(o, "item"):
        return o.item()
    raise TypeError("not JSON serialisable: " + type(o).__name__)


def _write(out: Path, name: str, payload: dict) -> None:
    (out / name).write_text(json.dumps(payload, indent=2, default=_default),
                            encoding="utf-8")


def cmd_gate(args) -> int:
    out = Path(args.out).resolve()
    controls = enable_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    res = run_checkpoint("NEG_shift_none", device)
    res.update({"controls": controls, "environment": environment(),
                "provenance": provenance(), "wall_clock_s": time.time() - t0,
                "arms": ARMS, "random_permutation_frozen": RANDOM_PERM})
    md = res["pooled_max_abs_diff_vs_identity"]
    vol_ok = all(md[a]["volume"] == 0.0 for a in ARM_ORDER)
    d_ok = all(md[a]["disparity_initial"] == 0.0 for a in ARM_ORDER)
    idfacts = [res["per_scene"][s]["tensor_facts"]
               ["max_abs_diff_identity_vs_original"] for s in res["per_scene"]]
    res["gate"] = {
        "algebraic_claim": "V(k)=D for all k, so V[:,:,pi] == V element-wise for any pi",
        "volume_invariant_under_every_permutation": vol_ok,
        "disparity_initial_invariant_under_every_permutation": d_ok,
        "identity_index_select_equals_original_max_abs_diff": max(idfacts),
        "PASS": bool(vol_ok and d_ok)}
    _write(out, "gate_negative_control.json", res)
    print(json.dumps({"gate": res["gate"],
                      "max_abs_diff_vs_identity": md,
                      "pooled_mean_disparity_initial":
                          {a: res["pooled"][a]["mean"] for a in ARM_ORDER},
                      "wall_clock_s": res["wall_clock_s"]}, indent=2, default=str))
    if not res["gate"]["PASS"]:
        print("\nGATE FAILED -- STOP. Implementation/determinism/layout failure. "
              "Positive checkpoints NOT run. Do not interpret scientifically.")
        return 1
    print("\nGATE PASS -- negative control invariant under every permutation.")
    return 0


def cmd_positive(args) -> int:
    out = Path(args.out).resolve()
    gate = json.loads((out / "gate_negative_control.json").read_text(encoding="utf-8"))
    if not gate["gate"]["PASS"]:
        print("STOP: gate did not pass.")
        return 1
    controls = enable_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0, results = time.time(), {}
    for key, spec in CHECKPOINTS.items():
        if spec["role"] != "positive":
            continue
        r = run_checkpoint(key, device)
        results[key] = r
        p = r["pooled"]
        print("{:<9} median d_init  -1 {:+.4f} | id {:.4f} | +1 {:+.4f} | "
              "+2 {:+.4f} | rand {:+.4f}   (deltas vs identity)".format(
                  key, p["m_minus_1"]["delta_median_vs_identity"],
                  p["identity"]["median"], p["m_plus_1"]["delta_median_vs_identity"],
                  p["m_plus_2"]["delta_median_vs_identity"],
                  p["random"]["delta_median_vs_identity"]))
    _write(out, "positive_checkpoints.json",
           {"results": results, "controls": controls, "arms": ARMS,
            "random_permutation_frozen": RANDOM_PERM,
            "environment": environment(), "provenance": provenance(),
            "wall_clock_s": time.time() - t0})
    return 0


def cmd_report(args) -> int:
    out = Path(args.out).resolve()
    g = json.loads((out / "gate_negative_control.json").read_text(encoding="utf-8"))
    p = json.loads((out / "positive_checkpoints.json").read_text(encoding="utf-8"))

    per_ckpt, verdicts = {}, {}
    for key, r in p["results"].items():
        pl = r["pooled"]
        dm = {a: pl[a]["delta_median_vs_identity"] for a in ARM_ORDER}
        ordered_ok = dm["m_plus_2"] > dm["m_plus_1"] > dm["identity"] > dm["m_minus_1"]
        plus1_positive = dm["m_plus_1"] > 0
        minus1_negative = dm["m_minus_1"] < 0
        plus2_larger = dm["m_plus_2"] > dm["m_plus_1"]
        # random must NOT reproduce the ordered directional pattern; judged by
        # whether it lands inside the ordered progression's step scale.
        rand_matches_plus1 = abs(dm["random"] - dm["m_plus_1"]) < 0.5 * abs(dm["m_plus_1"]) \
            if dm["m_plus_1"] != 0 else False
        per_ckpt[key] = {"seed": r["spec"]["seed"],
                         "checkpoint_sha256": r["checkpoint_sha256"],
                         "delta_median_vs_identity": dm,
                         "identity_median": pl["identity"]["median"],
                         "identity_mean": pl["identity"]["mean"],
                         "max_abs_diff_vs_identity":
                             r["pooled_max_abs_diff_vs_identity"]}
        verdicts[key] = {"ordered_monotone_d2_gt_d1_gt_id_gt_dm1": bool(ordered_ok),
                         "plus1_positive": bool(plus1_positive),
                         "minus1_negative": bool(minus1_negative),
                         "plus2_larger_than_plus1": bool(plus2_larger),
                         "random_reproduces_ordered_plus1": bool(rand_matches_plus1)}

    all_ordered = all(v["ordered_monotone_d2_gt_d1_gt_id_gt_dm1"] for v in verdicts.values())
    all_dirs = all(v["plus1_positive"] and v["minus1_negative"] for v in verdicts.values())
    rand_clean = all(not v["random_reproduces_ordered_plus1"] for v in verdicts.values())
    if not g["gate"]["PASS"]:
        verdict = "INVALID-DIAGNOSTIC"
    elif all_ordered and all_dirs and rand_clean:
        verdict = "CANDIDATE-COORDINATE-SENSITIVE"
    elif all_dirs and rand_clean:
        verdict = "CANDIDATE-COORDINATE-SENSITIVE-PARTIAL-ORDERING"
    else:
        verdict = "CANDIDATE-COORDINATE-SENSITIVITY-NOT-DEMONSTRATED"

    combined = {
        "experiment": "EXP-CORRESPONDENCE-INDEX-001",
        "kind": "inference-only; no training; no image or feature modification",
        "claim_level_targeted": "C -- candidate-coordinate sensitivity",
        "claim_levels_NOT_established": ["D geometric correspondence",
                                         "E genuine disparity search"],
        "arms": ARMS, "random_permutation_frozen": RANDOM_PERM,
        "mask": {"source": "ground truth only, model-independent",
                 "rule": "GT valid and GT/16 in [2, 8]",
                 "border_mask": "none required -- no image is translated"},
        "scenes": FOCUS_SCENES, "split": SPLIT,
        "negative_control_gate": g["gate"],
        "negative_control_pooled_mean":
            {a: g["pooled"][a]["mean"] for a in ARM_ORDER},
        "per_checkpoint": per_ckpt,
        "per_checkpoint_criteria": verdicts,
        "criteria_summary": {"all_ordered_monotone": all_ordered,
                             "all_directions_correct": all_dirs,
                             "random_does_not_reproduce_ordered": rand_clean},
        "unit_slope_claimed": False,
        "reference_frame_note": (
            "cost volume stores slice k = left_feat[u+k] - right_feat[u] at output "
            "column u (right-referenced) while GT, the visualiser convention and "
            "the refinement guidance are left-referenced. PRESERVED, not "
            "corrected. Candidate indices are NOT reinterpreted as physical "
            "disparity anywhere in this experiment."),
        "environment": p["environment"], "provenance": p["provenance"],
        "VERDICT": verdict,
    }
    _write(out, "results.json", combined)
    print(json.dumps({"VERDICT": verdict,
                      "criteria_summary": combined["criteria_summary"],
                      "per_checkpoint_criteria": verdicts,
                      "deltas": {k: per_ckpt[k]["delta_median_vs_identity"]
                                 for k in per_ckpt}}, indent=2, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gate", "positive", "report"):
        q = sub.add_parser(name)
        q.add_argument("--out", required=True)
    args = ap.parse_args()
    return {"gate": cmd_gate, "positive": cmd_positive, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
