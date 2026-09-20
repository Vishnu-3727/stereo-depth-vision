"""STAGE B -- DIAGNOSTIC/BASELINE ONLY. Not an architecture experiment.

Re-establishes the H2 / E3b arm-A six-block baseline under the deterministic
training controls Stage A demonstrated, and checks that two fresh processes
reproduce each other bit for bit.

The ONLY protocol change from the historical E3b arm-A run is the determinism
controls. The model, seed, optimizer, schedule, loss, dataset, crop policy,
augmentation, validation protocol, checkpoint schedule, probe schedule and
epoch budget are the historical ones, reached by calling
`exp_e3b_refinement_capacity.run_training("A", ...)` itself rather than by
restating any of it. Only the module's output paths and record identity are
repointed, the same override pattern `exp_e3c_refinement_floor.py` uses.

    python phase2/diagnostics/determinism/stageb_deterministic_baseline.py preflight --out DIR
    python phase2/diagnostics/determinism/stageb_deterministic_baseline.py train --run A --out DIR
    python phase2/diagnostics/determinism/stageb_deterministic_baseline.py train --run B --out DIR
    python phase2/diagnostics/determinism/stageb_deterministic_baseline.py evaluate40 --run A --out DIR
    python phase2/diagnostics/determinism/stageb_deterministic_baseline.py compare --out DIR

Writes only under --out. Phase 1, every Phase 2 scientific script, every
historical experiment record and every preregistration are read-only here.
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

# Must be set before the CUDA context exists. Stage A demonstrated this value.
CUBLAS_CONFIG = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_CONFIG:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_CONFIG

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

import phase2.scripts.exp_e3b_refinement_capacity as e3b        # noqa: E402
from phase2.viz import core                                     # noqa: E402
from src.evaluation.metrics import disparity_metrics            # noqa: E402
from src.datasets.kitti2015 import normalize                    # noqa: E402

STAGE_B_PREFIX = "STAGEB-DETERMINISTIC-BASELINE"
# The E3b arm-A initialisation, measured in the Stage A diagnostic
# (phase2/diagnostics/determinism/20260909T022238Z/run_A/record.json).
EXPECTED_INIT_WEIGHT_SHA = "c4d02385e3da28a5"
STAGE_A_DIR = "phase2/diagnostics/determinism/20260909T022238Z"
FROZEN_COMMIT_REF = "phase-1-frozen^" + "{commit}"


def sha(obj) -> str:
    """sha256 of the object's bytes, first 16 hex characters. Stage A's method."""
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
        "phase_1_frozen_tag_object": git("rev-parse", "phase-1-frozen"),
        "phase_1_diff_vs_frozen": git("diff", "phase-1-frozen", "--", "src", "scripts"),
        "status_short": git("status", "--short"),
        "stage_a_basis": STAGE_A_DIR,
    }


def environment() -> dict:
    return {
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": (torch.backends.cudnn.version()
                          if torch.backends.cudnn.is_available() else None),
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
        "gpu_capability": (".".join(map(str, torch.cuda.get_device_capability(0)))
                           if torch.cuda.is_available() else None),
        "nvidia_driver": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False).stdout.strip().splitlines()[:1],
        "flags": {
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn.deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn.benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "env": {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
    }


def enable_determinism() -> dict:
    """The three controls Stage A demonstrated. Errors are raised, never swallowed."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_CONFIG
    return {
        "torch.use_deterministic_algorithms": True,
        "torch.backends.cudnn.deterministic": True,
        "torch.backends.cudnn.benchmark": False,
        "CUBLAS_WORKSPACE_CONFIG": CUBLAS_CONFIG,
    }


def repoint(out: Path, run: str | None) -> None:
    """Send every e3b output path into the Stage B directory. Nothing else changes.

    The paths must be ABSOLUTE: `run_training` records the checkpoint as
    `ckpt.relative_to(REPO_ROOT)`, which raises on a relative path. The first
    Stage B attempt (20260909T024500Z_baseline) passed a relative --out and died
    on that line after its 200 epochs had completed; see that directory's NOTE.md.
    """
    out = out.resolve()
    e3b.EXPERIMENT_PREFIX = STAGE_B_PREFIX
    e3b.RESULT_DIR = out / "results"
    e3b.EXPERIMENTS_DIR = out / "experiments"
    e3b.OUT_DIR = out / "checkpoints"
    e3b.ARM_LABEL = "" if run is None else "-RUN" + run
    for d in (e3b.RESULT_DIR, e3b.EXPERIMENTS_DIR, e3b.OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    original_arm_config = e3b.arm_config

    def stage_b_arm_config(arm: str, device: str) -> dict:
        config = dict(original_arm_config(arm, device))
        config.update({
            "experiment": e3b.arm_id(arm),
            "stage": "STAGE B -- deterministic baseline re-establishment",
            "not_an_architecture_experiment": True,
            "only_protocol_change_vs_historical_e3b_arm_a":
                "deterministic training controls enabled",
            "deterministic_controls": {
                "torch.use_deterministic_algorithms": True,
                "torch.backends.cudnn.deterministic": True,
                "torch.backends.cudnn.benchmark": False,
                "CUBLAS_WORKSPACE_CONFIG": CUBLAS_CONFIG,
            },
            "stage_a_basis": STAGE_A_DIR,
            "historical_reference_record":
                "EXP-E3B-REFINEMENT-CAPACITY-001-ARM-A-RUN2 (untouched)",
        })
        return config

    e3b.arm_config = stage_b_arm_config


# ----------------------------------------------------------------- preflight
def cmd_preflight(args) -> int:
    out = Path(args.out).resolve()
    repoint(out, None)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Preflight is run WITHOUT determinism controls, exactly as the historical
    # E3b preflight was: it is an identity check, not training, and `thop`
    # mutates the modules it profiles.
    result = e3b.run_preflight(device)

    model = e3b.build_arm("A", device)
    init_sha = weight_sha(model)
    plain_sha = weight_sha(e3b.build_model(e3b.SEED, device))
    result["stage_b"] = {
        "provenance": provenance(),
        "environment": environment(),
        "init_weight_sha_build_arm_A": init_sha,
        "init_weight_sha_build_model": plain_sha,
        "init_weight_sha_expected": EXPECTED_INIT_WEIGHT_SHA,
        "init_weight_sha_matches": init_sha == EXPECTED_INIT_WEIGHT_SHA,
        "hash_method": "sha256 of concatenated model.parameters() fp32 bytes, "
                       "first 16 hex chars (Stage A method)",
        "seed": e3b.SEED,
        "epochs": e3b.EPOCHS,
        "arm_A_spec": e3b.ARMS["A"],
        "parameter_count": model.parameter_count(),
        "cost_volume_shift": model.cost_volume.shift,
        "regression": type(model.regression).__name__,
        "refinement_blocks": len(model.refinement.blocks),
        "refinement_dilations": [
            int(b.conv1.dilation[0]) for b in model.refinement.blocks],
        "train_split": "hailo_calib (scenes 0-159)",
        "validation_split": "hailo_val (scenes 160-199), first 10 validated",
        "crop": [256, 512],
    }
    result["stage_b"]["PASS"] = bool(
        result["checks"]["PASS"]
        and result["stage_b"]["init_weight_sha_matches"]
        and result["stage_b"]["provenance"]["phase_1_diff_vs_frozen"] == ""
        and result["stage_b"]["parameter_count"] == 423586)

    (e3b.RESULT_DIR / "preflight.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["stage_b"], indent=2)[:2400])
    print("checks:", json.dumps(result["checks"], indent=2))

    if not result["stage_b"]["PASS"]:
        print("\nPREFLIGHT FAILED -- STOP. Nothing was trained.")
        return 1
    print("\nPREFLIGHT PASS. init weight sha {} == expected {}".format(
        init_sha, EXPECTED_INIT_WEIGHT_SHA))
    return 0


# --------------------------------------------------------------------- train
def cmd_train(args) -> int:
    out = Path(args.out).resolve()
    repoint(out, args.run)
    controls = enable_determinism()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id = e3b.arm_id("A")

    meta = {"run": args.run, "experiment_id": exp_id, "controls": controls,
            "provenance": provenance(), "environment": environment(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # Confirm the initialisation is still the expected one *with* the controls on.
    init_sha = weight_sha(e3b.build_arm("A", device))
    meta["init_weight_sha"] = init_sha
    meta["init_weight_sha_matches"] = init_sha == EXPECTED_INIT_WEIGHT_SHA
    if not meta["init_weight_sha_matches"]:
        (out / "run_{}_ABORTED.json".format(args.run)).write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        print("STOP: init weight sha {} != expected {}".format(
            init_sha, EXPECTED_INIT_WEIGHT_SHA))
        return 1

    t0 = time.time()
    e3b.run_training("A", device)                 # the historical code path
    meta["wall_clock_s"] = time.time() - t0
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    ckpt = e3b.OUT_DIR / (exp_id + "_checkpoint.pth")
    meta["checkpoint"] = str(ckpt.relative_to(REPO_ROOT))
    meta["checkpoint_sha256"] = file_sha256(ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    tensors = state["model"] if isinstance(state, dict) and "model" in state else state
    meta["final_weight_sha"] = sha(torch.cat(
        [v.reshape(-1).float() for v in tensors.values() if v.is_floating_point()]))
    meta["snapshots"] = {
        p.name: file_sha256(p)
        for p in sorted(e3b.OUT_DIR.glob(exp_id + "_epoch*.pth"))}

    (out / "run_{}_meta.json".format(args.run)).write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print("run {} done in {:.1f}s; final weight sha {}".format(
        args.run, meta["wall_clock_s"], meta["final_weight_sha"]))
    return 0


# ---------------------------------------------------------------- evaluate40
def cmd_evaluate40(args) -> int:
    out = Path(args.out).resolve()
    repoint(out, args.run)
    enable_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id = e3b.arm_id("A")

    model = e3b.build_arm("A", device)
    ckpt = e3b.OUT_DIR / (exp_id + "_checkpoint.pth")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    rows, preds, gts = [], [], []
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
                         "valid_pixels": m.valid_pixels,
                         "pred_sha": sha(pred.astype(np.float32))})
            preds.append(pred[valid])
            gts.append(gt[valid])

    pooled = disparity_metrics(np.concatenate(preds), np.concatenate(gts))
    epes = np.array([r["epe"] for r in rows])
    d1s = np.array([r["d1"] for r in rows])

    result = {
        "experiment_id": exp_id,
        "run": args.run,
        "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
        "checkpoint_sha256": file_sha256(ckpt),
        "protocol": ("all 40 hailo_val scenes, full 368x1232 frames, pooled over "
                     "gt > 0 -- the protocol E1/E2/E3 used"),
        "pooled": {"epe": pooled.epe, "d1": pooled.d1, "rmse": pooled.rmse,
                   "bad1": pooled.bad1, "bad2": pooled.bad2,
                   "valid_pixels": pooled.valid_pixels},
        "per_scene_summary": {
            "scenes": len(rows),
            "epe": {"mean": float(epes.mean()), "median": float(np.median(epes)),
                    "sd": float(epes.std(ddof=1)), "min": float(epes.min()),
                    "max": float(epes.max()),
                    "argmin": rows[int(epes.argmin())]["scene"],
                    "argmax": rows[int(epes.argmax())]["scene"]},
            "d1": {"mean": float(d1s.mean()), "median": float(np.median(d1s)),
                   "sd": float(d1s.std(ddof=1)), "min": float(d1s.min()),
                   "max": float(d1s.max()),
                   "argmin": rows[int(d1s.argmin())]["scene"],
                   "argmax": rows[int(d1s.argmax())]["scene"]},
        },
        "first_10_scene_subset": None,
        "per_scene": rows,
        "environment": environment(),
        "provenance": provenance(),
    }
    sub_p = np.concatenate(preds[:10])
    sub_g = np.concatenate(gts[:10])
    sub = disparity_metrics(sub_p, sub_g)
    result["first_10_scene_subset"] = {
        "epe": sub.epe, "d1": sub.d1, "valid_pixels": sub.valid_pixels,
        "note": "the historical training-time validation subset, for cross-reference only"}

    dest = out / "eval40_run_{}.json".format(args.run)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("pooled 40-scene  EPE {:.6f}  D1 {:.6f}  ({} px)".format(
        pooled.epe, pooled.d1, pooled.valid_pixels))
    print("first-10 subset  EPE {:.6f}  D1 {:.6f}".format(sub.epe, sub.d1))
    print("wrote", dest)
    return 0


# ------------------------------------------------------------------- compare
def cmd_compare(args) -> int:
    out = Path(args.out).resolve()
    res: dict = {"provenance": provenance(), "stage_a_basis": STAGE_A_DIR}

    def rec(run):
        d = out / "experiments" / "{}-ARM-A-RUN{}".format(STAGE_B_PREFIX, run)
        return json.loads((d / "metrics.json").read_text(encoding="utf-8"))["metrics"]

    ma, mb = rec("A"), rec("B")
    meta_a = json.loads((out / "run_A_meta.json").read_text(encoding="utf-8"))
    meta_b = json.loads((out / "run_B_meta.json").read_text(encoding="utf-8"))

    hist_path = (REPO_ROOT / "phase2" / "experiments"
                 / "EXP-E3B-REFINEMENT-CAPACITY-001-ARM-A-RUN2" / "metrics.json")
    hist = json.loads(hist_path.read_text(encoding="utf-8"))["metrics"]

    def series(m):
        h = m["history"]
        return ([r["mean_loss"] for r in h],
                [(r["epoch"], r.get("val_epe"), r.get("val_d1")) for r in h
                 if r.get("val_epe") is not None])

    la, va = series(ma)
    lb, vb = series(mb)
    first_loss_diff = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), None)
    first_val_diff = next((i for i, (x, y) in enumerate(zip(va, vb)) if x != y), None)

    res["reproducibility"] = {
        "final_weight_sha_A": meta_a["final_weight_sha"],
        "final_weight_sha_B": meta_b["final_weight_sha"],
        "final_weight_sha_identical":
            meta_a["final_weight_sha"] == meta_b["final_weight_sha"],
        "checkpoint_sha256_A": meta_a["checkpoint_sha256"],
        "checkpoint_sha256_B": meta_b["checkpoint_sha256"],
        "checkpoint_file_sha256_identical":
            meta_a["checkpoint_sha256"] == meta_b["checkpoint_sha256"],
        "snapshot_sha256_identical": (
            [k.replace("-RUNA", "") for k in meta_a["snapshots"]]
            == [k.replace("-RUNB", "") for k in meta_b["snapshots"]]
            and list(meta_a["snapshots"].values()) == list(meta_b["snapshots"].values())),
        "epoch_loss_series_identical": la == lb,
        "first_epoch_with_differing_loss": first_loss_diff,
        "validation_series_identical": va == vb,
        "first_validation_point_differing": first_val_diff,
        "epochs_completed_A": ma["epochs_completed"],
        "epochs_completed_B": mb["epochs_completed"],
        "late_window_A": ma["late_window"],
        "late_window_B": mb["late_window"],
        "late_window_identical": ma["late_window"] == mb["late_window"],
        "final_checkpoint_metrics_A": {
            k: ma["checkpoint_epoch_200"][k] for k in ("val_epe", "val_d1", "val_loss")},
        "final_checkpoint_metrics_B": {
            k: mb["checkpoint_epoch_200"][k] for k in ("val_epe", "val_d1", "val_loss")},
        "gradient_norms_A": ma["gradient_norms"],
        "gradient_norms_B": mb["gradient_norms"],
        "stereo_verdict_A": ma["stereo_verdict_epoch_200"],
        "stereo_verdict_B": mb["stereo_verdict_epoch_200"],
        "stereo_verdict_identical":
            ma["stereo_verdict_epoch_200"] == mb["stereo_verdict_epoch_200"],
    }
    res["reproducibility"]["ALL_IDENTICAL"] = bool(
        res["reproducibility"]["final_weight_sha_identical"]
        and res["reproducibility"]["epoch_loss_series_identical"]
        and res["reproducibility"]["validation_series_identical"]
        and res["reproducibility"]["late_window_identical"]
        and res["reproducibility"]["stereo_verdict_identical"])

    res["runtime"] = {
        "controlled_wall_clock_s_A": meta_a["wall_clock_s"],
        "controlled_wall_clock_s_B": meta_b["wall_clock_s"],
        "controlled_s_per_epoch_A": meta_a["wall_clock_s"] / e3b.EPOCHS,
        "controlled_s_per_epoch_B": meta_b["wall_clock_s"] / e3b.EPOCHS,
        "historical_uncontrolled_wall_clock_s": hist["wall_clock_s"],
        "historical_uncontrolled_s_per_epoch": hist["wall_clock_s"] / e3b.EPOCHS,
        "stage_a_uncontrolled_s_per_epoch": [36.1, 37.5],
        "stage_a_controlled_s_per_epoch": [42.0, 41.2],
    }
    hist_rate = res["runtime"]["historical_uncontrolled_s_per_epoch"]
    res["runtime"]["overhead_vs_historical_percent"] = 100.0 * (
        res["runtime"]["controlled_s_per_epoch_A"] / hist_rate - 1.0)
    res["runtime"]["overhead_stage_a_percent"] = 100.0 * (
        float(np.mean([42.0, 41.2])) / float(np.mean([36.1, 37.5])) - 1.0)

    res["historical_comparison"] = {
        "note": ("HISTORICAL values come from EXP-E3B-REFINEMENT-CAPACITY-001-ARM-A-RUN2, "
                 "which is untouched. The training protocol differs (no determinism "
                 "controls), so differences are a protocol/reproducibility "
                 "characterisation and NOT an architecture effect."),
        "historical_late_window": hist["late_window"],
        "controlled_late_window": ma["late_window"],
        "delta_late_epe": ma["late_window"]["epe_mean"] - hist["late_window"]["epe_mean"],
        "delta_late_d1": ma["late_window"]["d1_mean"] - hist["late_window"]["d1_mean"],
        "historical_epoch_200": {
            k: hist["checkpoint_epoch_200"][k] for k in ("val_epe", "val_d1")},
        "controlled_epoch_200": {
            k: ma["checkpoint_epoch_200"][k] for k in ("val_epe", "val_d1")},
        "historical_stereo_verdict": hist["stereo_verdict_epoch_200"],
        "historical_wall_clock_s": hist["wall_clock_s"],
    }

    for run in ("A", "B"):
        p = out / "eval40_run_{}.json".format(run)
        if p.exists():
            e = json.loads(p.read_text(encoding="utf-8"))
            res.setdefault("eval40", {})[run] = {
                "pooled": e["pooled"],
                "per_scene_summary": e["per_scene_summary"],
                "first_10_scene_subset": e["first_10_scene_subset"],
                "per_scene_file": str(p.relative_to(REPO_ROOT)),
            }
    if "eval40" in res and set(res["eval40"]) == {"A", "B"}:
        ea, eb = res["eval40"]["A"]["pooled"], res["eval40"]["B"]["pooled"]
        res["eval40"]["identical"] = ea == eb

    (out / "comparison.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({"reproducibility": res["reproducibility"],
                      "runtime": res["runtime"],
                      "historical_comparison": res["historical_comparison"],
                      "eval40": res.get("eval40")}, indent=2)[:6000])
    print("\nwrote", out / "comparison.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "train", "evaluate40", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        if name in ("train", "evaluate40"):
            p.add_argument("--run", required=True, choices=["A", "B"])
    args = ap.parse_args()
    return {"preflight": cmd_preflight, "train": cmd_train,
            "evaluate40": cmd_evaluate40, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
