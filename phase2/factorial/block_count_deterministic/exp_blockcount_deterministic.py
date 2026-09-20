"""EXP-BLOCKCOUNT-DETERMINISTIC-001 -- 5- and 4-block refinement, deterministic.

Re-asks E3b's question under the Stage B deterministic protocol. The six-block
control is **not retrained**: Stage B established that a fresh-process rerun of
that configuration is bit-identical, so its frozen record is used directly.

Two treatment arms, one changed variable each:

    refinement block count:  6 -> 5   (arm B, dilations 1,2,4,8,1)
    refinement block count:  6 -> 4   (arm C, dilations 1,2,4,8)

Arms are built by `exp_e3b_refinement_capacity.build_arm`, unmodified, so every
surviving weight tensor is bit-identical to the control's. Training goes through
`exp_e3b_refinement_capacity.run_training` itself; only the module's output paths
and record identity are repointed.

    python .../exp_blockcount_deterministic.py preflight  --out DIR
    python .../exp_blockcount_deterministic.py train      --out DIR --arm B
    python .../exp_blockcount_deterministic.py train      --out DIR --arm C
    python .../exp_blockcount_deterministic.py evaluate40 --out DIR --arm B
    python .../exp_blockcount_deterministic.py probes     --out DIR
    python .../exp_blockcount_deterministic.py compare    --out DIR

Writes only under --out. Phase 1, every Phase 2 scientific script, every
historical record, Stage A, Stage B and the factorial cell are read-only here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CUBLAS_CONFIG = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_CONFIG:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_CONFIG

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

import phase2.scripts.exp_e3b_refinement_capacity as e3b        # noqa: E402
from phase2.scripts.exp_h2_seed_replication import (            # noqa: E402
    FOCUS_SCENES, build_model, measure_macs, stereo_probe,
)
from phase2.scripts.exp_h2_softargmin_scale import (            # noqa: E402
    validation_stage_stats,
)
from phase2.viz import core                                     # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize   # noqa: E402
from src.evaluation.metrics import disparity_metrics            # noqa: E402

EXPERIMENT_PREFIX = "EXP-BLOCKCOUNT-DETERMINISTIC-001"
ARMS = ("B", "C")
CONTROL_DIR = (REPO_ROOT / "phase2" / "diagnostics" / "determinism"
               / "20260909T041500Z_baseline")
CONTROL_ID = "STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA"
CONTROL_REL = "phase2/diagnostics/determinism/20260909T041500Z_baseline"
CONTROL_INIT_SHA = "c4d02385e3da28a5"
PROBE_EPOCHS = (10, 50, 100, 150, 200)
FROZEN_COMMIT_REF = "phase-1-frozen^" + "{commit}"

# HISTORICAL reproducibility scales, quoted for magnitude only (section 6.4 of
# the pre-registration). Never used as a threshold.
SCALES = {
    "stage_b_controlled_same_config_variance_epe": 0.0,
    "stage_b_controlled_same_config_variance_d1": 0.0,
    "stage_b_protocol_divergence_epe": 0.2250805,
    "stage_b_protocol_divergence_d1": 1.4143253,
    "o6_uncontrolled_same_config_epe": 0.3426,
    "o6_uncontrolled_same_config_d1": 1.9701,
}


def sha(obj) -> str:
    if isinstance(obj, torch.Tensor):
        data = obj.detach().cpu().contiguous().numpy().tobytes()
    elif isinstance(obj, np.ndarray):
        data = np.ascontiguousarray(obj).tobytes()
    else:
        data = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def weight_sha(model) -> str:
    return sha(torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()]))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True,
                          text=True, check=False).stdout.strip()


def provenance() -> dict:
    return {
        "head": git("rev-parse", "HEAD"),
        "head_subject": git("log", "-1", "--format=%s"),
        "phase_1_frozen_commit": git("rev-parse", FROZEN_COMMIT_REF),
        "phase_1_diff_vs_frozen": git("diff", "phase-1-frozen", "--", "src", "scripts"),
        "status_short": git("status", "--short"),
        "phase_2_committed": git("ls-files", "phase2") != "",
        "control_record": CONTROL_REL,
    }


def environment() -> dict:
    drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                          "--format=csv,noheader"], capture_output=True,
                         text=True, check=False).stdout.strip().splitlines()
    return {
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": (torch.backends.cudnn.version()
                          if torch.backends.cudnn.is_available() else None),
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
        "nvidia_driver": drv[0] if drv else None,
        "flags": {
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn.deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn.benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "env": {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
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


def repoint(out: Path) -> None:
    out = out.resolve()
    e3b.EXPERIMENT_PREFIX = EXPERIMENT_PREFIX
    e3b.RESULT_DIR = out / "results"
    e3b.EXPERIMENTS_DIR = out / "experiments"
    e3b.OUT_DIR = out / "checkpoints"
    e3b.ARM_LABEL = ""
    for d in (e3b.RESULT_DIR, e3b.EXPERIMENTS_DIR, e3b.OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    original = e3b.arm_config

    def blockcount_arm_config(arm: str, device: str) -> dict:
        config = dict(original(arm, device))
        config.update({
            "experiment": e3b.arm_id(arm),
            "experiment_kind": ("refinement-capacity re-measurement under the "
                                "Stage B deterministic protocol -- NOT a cost "
                                "volume experiment; Stage E stays CLOSED"),
            "control": ("{} -- Stage B deterministic run A, frozen, NOT retrained "
                        "(a rerun is bit-identical, so it would add no "
                        "information)").format(CONTROL_ID),
            "control_record": CONTROL_REL,
            "deterministic_controls": {
                "torch.use_deterministic_algorithms": True,
                "torch.backends.cudnn.deterministic": True,
                "torch.backends.cudnn.benchmark": False,
                "CUBLAS_WORKSPACE_CONFIG": CUBLAS_CONFIG,
            },
            "materiality_band": ("NONE. The historical E3b/E3c band (0.2138 px / "
                                 "1.0000 pt) is invalidated and is neither reused "
                                 "nor replaced."),
            "claim_ceiling": ("Bitwise reproducibility removes EXECUTION noise, "
                              "not SEED noise. The strongest supportable claim is "
                              "'at seed 0, under this protocol, the difference is "
                              "exactly X'. No generalisation to other seeds; no "
                              "significance claimed."),
        })
        return config

    e3b.arm_config = blockcount_arm_config


# ----------------------------------------------------------------- preflight
def cmd_preflight(args) -> int:
    out = Path(args.out).resolve()
    repoint(out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    controls = enable_determinism()

    control_model = build_model(0, device)
    ref = e3b.build_arm("A", device)
    arms = {}
    identical = True
    for arm in ARMS:
        model = e3b.build_arm(arm, device)
        # surviving weights must be bit-identical to the control's
        ok = True
        for position, index in enumerate(e3b.ARMS[arm]["keep"]):
            a = ref.refinement.blocks[index].state_dict()
            b = model.refinement.blocks[position].state_dict()
            ok = ok and all(torch.equal(a[k], b[k]) for k in a)
        for name in ("feature_extractor", "aggregation"):
            a, b = getattr(ref, name).state_dict(), getattr(model, name).state_dict()
            ok = ok and all(torch.equal(a[k], b[k]) for k in a)
        for name in ("input_conv", "output_conv"):
            a = getattr(ref.refinement, name).state_dict()
            b = getattr(model.refinement, name).state_dict()
            ok = ok and all(torch.equal(a[k], b[k]) for k in a)
        identical = identical and ok
        arms[arm] = {
            "blocks": len(model.refinement.blocks),
            "blocks_expected": e3b.ARMS[arm]["blocks"],
            "dilations": [int(b.conv1.dilation[0]) for b in model.refinement.blocks],
            "dilations_expected": list(e3b.ARMS[arm]["dilations"]),
            "parameters": model.parameter_count(),
            "parameters_expected": e3b.ARMS[arm]["parameters"],
            "parameters_match": model.parameter_count() == e3b.ARMS[arm]["parameters"],
            "surviving_weights_bit_identical_to_control": ok,
            "init_weight_sha": weight_sha(model),
            "cost_volume_shift": model.cost_volume.shift,
            "num_disparities": model.cost_volume.num_disparities,
            "regression": type(model.regression).__name__,
            "macs_256x512": measure_macs(e3b.build_arm(arm, device), device),
        }
    control_entry = {
        "blocks": 6, "dilations": [1, 2, 4, 8, 1, 1],
        "parameters": control_model.parameter_count(),
        "init_weight_sha": weight_sha(control_model),
        "init_weight_sha_expected": CONTROL_INIT_SHA,
        "cost_volume_shift": control_model.cost_volume.shift,
        "num_disparities": control_model.cost_volume.num_disparities,
        "regression": type(control_model.regression).__name__,
        "macs_256x512": measure_macs(build_model(0, device), device),
        "retrained_here": False,
        "reason_not_retrained": ("Stage B measured a fresh-process rerun of this "
                                 "configuration to be bit-identical (same final "
                                 "weight sha, loss series, validation series and "
                                 "40-scene evaluation), so retraining could only "
                                 "reproduce it."),
        "record": CONTROL_REL,
        "frozen_40_scene": {"epe": 2.2955181809208267, "d1": 15.6045142562172,
                            "rmse": 6.043707506355849},
    }

    result = {
        "experiment": EXPERIMENT_PREFIX,
        "provenance": provenance(), "environment": environment(),
        "deterministic_controls": controls,
        "control": control_entry, "arms": arms,
        "reproducibility_scales_HISTORICAL": SCALES,
        "claim_ceiling": ("Bitwise reproducibility removes EXECUTION noise, not "
                          "SEED noise; results are 'at seed 0' only."),
    }
    result["checks"] = {
        "control_init_sha_matches": control_entry["init_weight_sha"] == CONTROL_INIT_SHA,
        "surviving_weights_bit_identical": bool(identical),
        "parameter_counts_match_registration": all(
            arms[a]["parameters_match"] for a in ARMS),
        "block_counts_correct": all(
            arms[a]["blocks"] == arms[a]["blocks_expected"] for a in ARMS),
        "dilations_correct": all(
            arms[a]["dilations"] == arms[a]["dilations_expected"] for a in ARMS),
        "shift_is_left": all(arms[a]["cost_volume_shift"] == "left" for a in ARMS),
        "readout_is_standardised": all(
            arms[a]["regression"] == "StandardisedDisparityRegression" for a in ARMS),
        "candidates_are_12": all(arms[a]["num_disparities"] == 12 for a in ARMS),
        "control_checkpoint_present": (
            CONTROL_DIR / "checkpoints" / (CONTROL_ID + "_checkpoint.pth")).exists(),
        "phase_1_diff_empty": result["provenance"]["phase_1_diff_vs_frozen"] == "",
    }
    result["checks"]["PASS"] = all(v for v in result["checks"].values()
                                   if isinstance(v, bool))
    (e3b.RESULT_DIR / "preflight.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"control": control_entry, "arms": arms,
                      "checks": result["checks"]}, indent=2))
    if not result["checks"]["PASS"]:
        print("\nPREFLIGHT FAILED -- STOP. Nothing was trained.")
        return 1
    print("\nPREFLIGHT PASS.")
    return 0


# --------------------------------------------------------------------- train
def cmd_train(args) -> int:
    out = Path(args.out).resolve()
    repoint(out)
    controls = enable_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arm = args.arm
    exp_id = e3b.arm_id(arm)

    model = e3b.build_arm(arm, device)
    meta = {"arm": arm, "experiment_id": exp_id, "controls": controls,
            "blocks": len(model.refinement.blocks),
            "parameters": model.parameter_count(),
            "init_weight_sha": weight_sha(model),
            "changed_variable": "refinement block count: 6 -> {}".format(
                e3b.ARMS[arm]["blocks"]),
            "provenance": provenance(), "environment": environment(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if meta["parameters"] != e3b.ARMS[arm]["parameters"]:
        (out / "ABORTED_{}.json".format(arm)).write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        print("STOP: parameter count", meta["parameters"])
        return 1
    del model

    t0 = time.time()
    e3b.run_training(arm, device)
    meta["wall_clock_s"] = time.time() - t0
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    ckpt = e3b.OUT_DIR / (exp_id + "_checkpoint.pth")
    meta["checkpoint"] = str(ckpt.relative_to(REPO_ROOT))
    meta["checkpoint_sha256"] = file_sha256(ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    tensors = state["model"] if "model" in state else state
    meta["final_weight_sha"] = sha(torch.cat(
        [v.reshape(-1).float() for v in tensors.values() if v.is_floating_point()]))
    (out / "run_meta_{}.json".format(arm)).write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print("arm {} done in {:.1f}s; final weight sha {}".format(
        arm, meta["wall_clock_s"], meta["final_weight_sha"]))
    return 0


# ------------------------------------------------------- 40-scene evaluation
def _eval40(model, device) -> dict:
    rows, preds, gts = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(40):
            scene = core.load_scene(i, split="hailo_val")
            pred = model(
                torch.from_numpy(normalize(scene.left)).to(device),
                torch.from_numpy(normalize(scene.right)).to(device),
            )[0, 0].cpu().numpy().astype(np.float64)
            gt = scene.gt_disparity.astype(np.float64)
            valid = gt > 0
            m = disparity_metrics(pred[valid], gt[valid])
            rows.append({"index": i, "scene": scene.name, "epe": m.epe, "d1": m.d1,
                         "rmse": m.rmse, "bad1": m.bad1, "bad2": m.bad2,
                         "valid_pixels": m.valid_pixels})
            preds.append(pred[valid])
            gts.append(gt[valid])
    pooled = disparity_metrics(np.concatenate(preds), np.concatenate(gts))
    epes = np.array([r["epe"] for r in rows])
    d1s = np.array([r["d1"] for r in rows])
    sub = disparity_metrics(np.concatenate(preds[:10]), np.concatenate(gts[:10]))
    return {
        "protocol": ("all 40 hailo_val scenes, full 368x1232 frames, pooled over "
                     "gt > 0 -- the existing evaluator, unchanged"),
        "pooled": {"epe": pooled.epe, "d1": pooled.d1, "rmse": pooled.rmse,
                   "bad1": pooled.bad1, "bad2": pooled.bad2,
                   "valid_pixels": pooled.valid_pixels},
        "per_scene_summary": {
            "epe": {"mean": float(epes.mean()), "median": float(np.median(epes)),
                    "sd": float(epes.std(ddof=1)), "min": float(epes.min()),
                    "max": float(epes.max())},
            "d1": {"mean": float(d1s.mean()), "median": float(np.median(d1s)),
                   "sd": float(d1s.std(ddof=1)), "min": float(d1s.min()),
                   "max": float(d1s.max())}},
        "first_10_scene_subset": {"epe": sub.epe, "d1": sub.d1,
                                  "note": "historical subset, cross-reference only"},
        "per_scene": rows,
    }


def cmd_evaluate40(args) -> int:
    out = Path(args.out).resolve()
    repoint(out)
    enable_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arm = args.arm
    exp_id = e3b.arm_id(arm)
    model = e3b.build_arm(arm, device)
    ckpt = e3b.OUT_DIR / (exp_id + "_checkpoint.pth")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    result = _eval40(model, device)
    result.update({"experiment_id": exp_id, "arm": arm,
                   "blocks": len(model.refinement.blocks),
                   "checkpoint_sha256": file_sha256(ckpt),
                   "environment": environment(), "provenance": provenance()})
    (out / "eval40_arm_{}.json".format(arm)).write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    p = result["pooled"]
    print("arm {} 40-scene  EPE {:.7f}  D1 {:.7f}  RMSE {:.7f}".format(
        arm, p["epe"], p["d1"], p["rmse"]))
    return 0


# -------------------------------------------------------------------- probes
def cmd_probes(args) -> int:
    out = Path(args.out).resolve()
    repoint(out)
    enable_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scenes = [core.load_scene(i) for i in FOCUS_SCENES]
    val_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val")

    def series(builder, ckpt_dir: Path, stem: str) -> dict:
        rows = {}
        for epoch in PROBE_EPOCHS:
            path = ckpt_dir / "{}_epoch{}.pth".format(stem, epoch)
            if not path.exists():
                rows[str(epoch)] = {"error": "snapshot missing: " + path.name}
                continue
            model = builder(device)
            state = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"] if "model" in state else state)
            model.eval()
            rng = np.random.default_rng(0)
            ab = stereo_probe(model, scenes, device, rng)
            right = min(r[k]["d1_penalty"] for r in ab
                        for k in ("right_black", "right_noise", "right_equals_left"))
            mmap = min(r[k]["d1_penalty"] for r in ab
                       for k in ("initial_constant_mean", "initial_shuffled"))
            st = validation_stage_stats(model, val_base, device)
            rows[str(epoch)] = {
                "epoch": epoch,
                "right_image_min_d1_penalty": right,
                "matching_map_min_d1_penalty": mmap,
                "softmax_entropy": st["entropy"],
                "final_disparity_std": st["final_std"],
                "disparity_initial_corr_gt": st["initial_corr_gt"],
                "final_corr_gt": st["final_corr_gt"],
            }
            print("    epoch {:>3}  right {:+8.3f}  map {:+8.3f}  entropy {:.4f}  "
                  "std {:6.3f}  init r(GT) {:+.4f}".format(
                      epoch, right, mmap, st["entropy"], st["final_std"],
                      st["initial_corr_gt"]))
        return rows

    payload = {"probe_epochs": list(PROBE_EPOCHS), "focus_scenes": FOCUS_SCENES,
               "probe_implementation": ("exp_h2_seed_replication.stereo_probe, "
                                        "imported unchanged"),
               "caveat": ("right-image and matching-map probes establish binocular "
                          "dependence, NOT correct disparity search: Phase 1 "
                          "EXP-007 measured +89.7 D1 on the reference weights "
                          "despite a provably degenerate shift."),
               "arms": {}}
    for arm in ARMS:
        print("arm {} ({} blocks):".format(arm, e3b.ARMS[arm]["blocks"]))
        payload["arms"][arm] = series(
            lambda d, a=arm: e3b.build_arm(a, d), e3b.OUT_DIR, e3b.arm_id(arm))
    print("control (6 blocks, Stage B run A, frozen):")
    payload["control"] = series(lambda d: build_model(0, d),
                                CONTROL_DIR / "checkpoints", CONTROL_ID)
    payload["environment"] = environment()
    payload["provenance"] = provenance()
    (out / "probes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", out / "probes.json")
    return 0


# ------------------------------------------------------------------- compare
def cmd_compare(args) -> int:
    out = Path(args.out).resolve()
    repoint(out)
    ctrl_eval = json.loads((CONTROL_DIR / "eval40_run_A.json").read_text(encoding="utf-8"))
    ctrl_m = json.loads((CONTROL_DIR / "experiments" / CONTROL_ID / "metrics.json")
                        .read_text(encoding="utf-8"))["metrics"]
    ctrl_meta = json.loads((CONTROL_DIR / "run_A_meta.json").read_text(encoding="utf-8"))
    pre = json.loads((e3b.RESULT_DIR / "preflight.json").read_text(encoding="utf-8"))
    probes = json.loads((out / "probes.json").read_text(encoding="utf-8"))

    res = {"experiment": EXPERIMENT_PREFIX, "provenance": provenance(),
           "control": {"id": CONTROL_ID, "record": CONTROL_REL, "blocks": 6,
                       "retrained_here": False,
                       "pooled": ctrl_eval["pooled"],
                       "late_window": ctrl_m["late_window"],
                       "stereo_verdict": ctrl_m["stereo_verdict_epoch_200"],
                       "wall_clock_s": ctrl_meta["wall_clock_s"],
                       "parameters": 423586,
                       "macs_256x512": pre["control"]["macs_256x512"]["macs"]},
           "reproducibility_scales_HISTORICAL": SCALES,
           "materiality_band": "NONE reused, NONE invented.",
           "claim_ceiling": ("at seed 0, under this deterministic protocol; "
                             "execution noise is zero, seed noise is UNKNOWN"),
           "arms": {}}

    cp = ctrl_eval["pooled"]
    for arm in ARMS:
        ev = json.loads((out / "eval40_arm_{}.json".format(arm)).read_text(encoding="utf-8"))
        m = json.loads((out / "experiments" / e3b.arm_id(arm) / "metrics.json")
                       .read_text(encoding="utf-8"))["metrics"]
        meta = json.loads((out / "run_meta_{}.json".format(arm)).read_text(encoding="utf-8"))
        per = []
        for t, c in zip(ev["per_scene"], ctrl_eval["per_scene"]):
            assert t["scene"] == c["scene"]
            per.append({"scene": t["scene"], "valid_pixels": t["valid_pixels"],
                        "control_epe": c["epe"], "arm_epe": t["epe"],
                        "delta_epe": t["epe"] - c["epe"],
                        "control_d1": c["d1"], "arm_d1": t["d1"],
                        "delta_d1": t["d1"] - c["d1"]})
        de = np.array([r["delta_epe"] for r in per])
        dd = np.array([r["delta_d1"] for r in per])
        p = ev["pooled"]
        res["arms"][arm] = {
            "blocks": e3b.ARMS[arm]["blocks"],
            "dilations": list(e3b.ARMS[arm]["dilations"]),
            "parameters": meta["parameters"],
            "macs_256x512": pre["arms"][arm]["macs_256x512"]["macs"],
            "parameter_saving_percent": 100.0 * (1 - meta["parameters"] / 423586),
            "mac_saving_percent": 100.0 * (
                1 - pre["arms"][arm]["macs_256x512"]["macs"]
                / pre["control"]["macs_256x512"]["macs"]),
            "pooled": p,
            "delta_epe": p["epe"] - cp["epe"], "delta_d1": p["d1"] - cp["d1"],
            "delta_rmse": p["rmse"] - cp["rmse"],
            "late_window": m["late_window"],
            "late_delta_epe": (m["late_window"]["epe_mean"]
                               - ctrl_m["late_window"]["epe_mean"]),
            "late_delta_d1": (m["late_window"]["d1_mean"]
                              - ctrl_m["late_window"]["d1_mean"]),
            "epochs_completed": m["epochs_completed"], "aborted": m["aborted"],
            "final_train_loss": m["history"][-1]["mean_loss"],
            "gradient_norms": m["gradient_norms"],
            "matching_gradient": m["primary_endpoint_matching_gradient"],
            "stereo_verdict": m["stereo_verdict_epoch_200"],
            "wall_clock_s": meta["wall_clock_s"],
            "final_weight_sha": meta["final_weight_sha"],
            "per_scene": per,
            "per_scene_consistency": {
                "scenes": len(per),
                "worse_on_epe": int((de > 0).sum()),
                "worse_on_d1": int((dd > 0).sum()),
                "delta_epe": {"mean": float(de.mean()), "median": float(np.median(de)),
                              "sd": float(de.std(ddof=1)), "min": float(de.min()),
                              "max": float(de.max())},
                "delta_d1": {"mean": float(dd.mean()), "median": float(np.median(dd)),
                             "sd": float(dd.std(ddof=1)), "min": float(dd.min()),
                             "max": float(dd.max())}},
            "probe_series": probes["arms"][arm],
        }
    res["control_probe_series"] = probes["control"]
    (out / "comparison.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    summary = {a: {k: res["arms"][a][k] for k in
                   ("blocks", "parameters", "mac_saving_percent", "pooled",
                    "delta_epe", "delta_d1", "delta_rmse", "late_delta_epe",
                    "late_delta_d1", "per_scene_consistency", "stereo_verdict",
                    "wall_clock_s")} for a in ARMS}
    print(json.dumps({"control_pooled": cp, "arms": summary}, indent=2)[:6000])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "train", "evaluate40", "probes", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        if name in ("train", "evaluate40"):
            p.add_argument("--arm", required=True, choices=list(ARMS))
    args = ap.parse_args()
    return {"preflight": cmd_preflight, "train": cmd_train,
            "evaluate40": cmd_evaluate40, "probes": cmd_probes,
            "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
