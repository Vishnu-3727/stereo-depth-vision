"""FACTORIAL CELL -- `cost_volume_shift="none"` + standardised readout, 200 epochs.

The missing cell of the 2x2 {shift on/off} x {standardisation on/off}. Three
cells are already filled at 200 epochs (H1-BASE-v2, H1-WORKING-v2, H2); this
fills the fourth under the Stage B deterministic protocol.

Exactly ONE scientific variable changes from the Stage B control:

    cost_volume_shift: "left" -> "none"

The change is applied **in place on the built model**, after
`build_model(seed=0)` has consumed the RNG, so the initial weights are
bit-identical to the control's (`CostVolume` holds no parameters; `shift` is a
plain attribute read at forward time). This is the same in-place pattern
`exp_o6_refinement_dilation.py` uses for its dilation schedule.

Everything else -- model, six refinement blocks, 12 candidates, standardisation,
soft-argmin, refinement, loss, optimizer, schedule, seed, dataset, crop,
augmentation, batch size, 200 epochs, validation protocol, checkpoint and probe
schedule -- is the historical recipe, reached by calling
`exp_e3b_refinement_capacity.run_training("A", ...)` itself. Only the module's
output paths and record identity are repointed.

    python .../exp_factorial_shift.py preflight   --out DIR
    python .../exp_factorial_shift.py train       --out DIR
    python .../exp_factorial_shift.py evaluate40  --out DIR
    python .../exp_factorial_shift.py probes      --out DIR
    python .../exp_factorial_shift.py compare     --out DIR

Writes only under --out. Phase 1, every Phase 2 scientific script, every
historical record and Stage A / Stage B are read-only here.
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

# Stage B's value, set before the CUDA context exists.
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
    stage_gradients, validation_stage_stats,
)
from phase2.scripts.exp_h1_cost_volume import CroppedKitti      # noqa: E402
from phase2.viz import core                                     # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize   # noqa: E402
from src.evaluation.metrics import disparity_metrics            # noqa: E402
from src.losses.disparity import masked_smooth_l1               # noqa: E402

EXPERIMENT_PREFIX = "EXP-FACTORIAL-SHIFT-NONE-STANDARDISED-001"
TREATMENT_SHIFT = "none"
CONTROL_SHIFT = "left"
EXPECTED_INIT_WEIGHT_SHA = "c4d02385e3da28a5"
STAGE_A_DIR = "phase2/diagnostics/determinism/20260909T022238Z"
STAGE_B_DIR = "phase2/diagnostics/determinism/20260909T041500Z_baseline"
CONTROL_RECORD = (REPO_ROOT / "phase2" / "diagnostics" / "determinism"
                  / "20260909T041500Z_baseline")
CONTROL_ID = "STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA"
PROBE_EPOCHS = (10, 50, 100, 150, 200)
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
        "phase_2_committed": git("ls-files", "phase2") != "",
        "stage_a_basis": STAGE_A_DIR,
        "stage_b_control": STAGE_B_DIR,
    }


def environment() -> dict:
    drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                          "--format=csv,noheader"], capture_output=True,
                         text=True, check=False).stdout.strip().splitlines()
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
        "nvidia_driver": drv[0] if drv else None,
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
    """Stage B's mandatory controls. Errors propagate; nothing is suppressed."""
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


def build_treatment(device: str):
    """The control's model with the shift removed in place. One changed variable.

    `build_model` seeds torch+numpy and builds the H2 architecture; the flip
    happens afterwards and touches no parameter, so the initial weights are
    bit-identical to the control's.
    """
    model = build_model(0, device)                # asserts shift == "left"
    assert model.cost_volume.shift == CONTROL_SHIFT
    model.cost_volume.shift = TREATMENT_SHIFT     # <- the one change
    model.config.cost_volume_shift = TREATMENT_SHIFT
    assert model.cost_volume.shift == TREATMENT_SHIFT
    assert len(model.refinement.blocks) == 6
    assert model.cost_volume.num_disparities == 12
    assert type(model.regression).__name__ == "StandardisedDisparityRegression"
    return model


def repoint(out: Path) -> None:
    out = out.resolve()
    e3b.EXPERIMENT_PREFIX = EXPERIMENT_PREFIX
    e3b.RESULT_DIR = out / "results"
    e3b.EXPERIMENTS_DIR = out / "experiments"
    e3b.OUT_DIR = out / "checkpoints"
    e3b.ARM_LABEL = ""
    for d in (e3b.RESULT_DIR, e3b.EXPERIMENTS_DIR, e3b.OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    original_arm_config = e3b.arm_config

    def factorial_arm_config(arm: str, device: str) -> dict:
        config = dict(original_arm_config(arm, device))
        config.update({
            "experiment": e3b.arm_id(arm),
            "cost_volume_shift": TREATMENT_SHIFT,
            "experiment_kind": "FACTORIAL CELL -- not an architecture search",
            "hypothesis": ("Does cost_volume_shift provide useful stereo value "
                           "after the H2 standardised readout has removed the "
                           "soft-argmin saturation failure?"),
            "changed_variable": "cost_volume_shift: 'left' -> 'none' (in place; "
                                "no parameter is touched, so the initial weights "
                                "are bit-identical to the control's)",
            "control": ("{} -- Stage B deterministic run A, shift='left', "
                        "standardised readout, same seed, same recipe"
                        ).format(CONTROL_ID),
            "control_record": STAGE_B_DIR,
            "refinement_blocks": 6,
            "refinement_dilations": [1, 2, 4, 8, 1, 1],
            "deterministic_controls": {
                "torch.use_deterministic_algorithms": True,
                "torch.backends.cudnn.deterministic": True,
                "torch.backends.cudnn.benchmark": False,
                "CUBLAS_WORKSPACE_CONFIG": CUBLAS_CONFIG,
            },
            "materiality_band": ("NONE. The historical E3b/E3c band (0.2138 px / "
                                 "1.0000 pt) is invalidated and is not reused; no "
                                 "replacement band is invented."),
        })
        return config

    e3b.arm_config = factorial_arm_config
    e3b.build_arm = lambda arm, device: build_treatment(device)


# ----------------------------------------------------------------- preflight
def cmd_preflight(args) -> int:
    out = Path(args.out).resolve()
    repoint(out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    controls = enable_determinism()

    treatment = build_treatment(device)
    control = build_model(0, device)

    init_t, init_c = weight_sha(treatment), weight_sha(control)
    scenes = [core.load_scene(i) for i in FOCUS_SCENES]

    # Initial tensors, on the model's own untrained weights.
    treatment.eval()
    with torch.no_grad():
        s = scenes[0]
        left = torch.from_numpy(normalize(s.left)).to(device)
        right = torch.from_numpy(normalize(s.right)).to(device)
        _, stages = treatment(left, right, return_stages=True)
    cost = stages["aggregated_cost"]
    scaled = (cost - cost.mean(1, keepdim=True)) / (cost.std(1, keepdim=True,
                                                             unbiased=False) + 1e-5)
    probs = torch.softmax(-torch.nn.functional.interpolate(
        scaled, size=(s.left.shape[0], s.left.shape[1]),
        mode="bilinear", align_corners=True), dim=1)
    entropy = float((-(probs * probs.clamp_min(1e-12).log()).sum(1)).mean())
    di = stages["disparity_initial"]

    # Degeneracy check: with shift="none" every candidate slice must be equal.
    vol = stages["cost_volume"]
    slice_max_diff = float(max(
        (vol[:, :, k] - vol[:, :, 0]).abs().max() for k in range(vol.shape[2])))

    # Matching-path gradient at initialisation, one training-shaped crop.
    treatment.train()
    train_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_calib")
    ds = CroppedKitti(train_base, seed=0)
    l0, r0, d0 = ds[0]
    l0, r0, d0 = (l0[None].to(device), r0[None].to(device), d0[None].to(device))
    treatment.zero_grad(set_to_none=True)
    pred = treatment(l0, r0)
    loss0, _ = masked_smooth_l1(pred, d0,
                                max_disparity=float(treatment.config.max_disparity_px))
    loss0.backward()
    grads = stage_gradients(treatment)
    treatment.zero_grad(set_to_none=True)

    result = {
        "experiment": EXPERIMENT_PREFIX,
        "provenance": provenance(),
        "environment": environment(),
        "deterministic_controls": controls,
        "architecture": {
            "refinement_blocks": len(treatment.refinement.blocks),
            "refinement_dilations": [int(b.conv1.dilation[0])
                                     for b in treatment.refinement.blocks],
            "parameter_count": treatment.parameter_count(),
            "parameter_count_control": control.parameter_count(),
            "parameters_equal": treatment.parameter_count() == control.parameter_count(),
            "num_disparities": treatment.cost_volume.num_disparities,
            "cost_volume_method": treatment.cost_volume.method,
            "treatment_shift": treatment.cost_volume.shift,
            "control_shift": control.cost_volume.shift,
            "regression": type(treatment.regression).__name__,
            "standardisation_location": ("after the full-resolution upsample, on "
                                         "the tensor the soft-argmin consumes "
                                         "(upsample_first=True)"),
            "upsample_before_argmin": treatment.config.upsample_before_argmin,
            "macs_256x512": measure_macs(build_treatment(device), device),
        },
        "initialisation": {
            "treatment_weight_sha": init_t,
            "control_weight_sha": init_c,
            "expected": EXPECTED_INIT_WEIGHT_SHA,
            "treatment_matches_expected": init_t == EXPECTED_INIT_WEIGHT_SHA,
            "treatment_equals_control": init_t == init_c,
            "hash_method": ("sha256 of concatenated model.parameters() fp32 bytes, "
                            "first 16 hex chars (Stage A method)"),
        },
        "data": {
            "train_split": "hailo_calib (scenes 0-159)",
            "validation_split": "hailo_val (scenes 160-199), first 10 validated",
            "evaluation": "all 40 hailo_val scenes, pooled over gt > 0",
            "seed": 0,
            "crop": [256, 512],
            "augmentation": "random crop, per-image gain jitter sigma=0.1, no flip",
            "batch_size": 2,
            "epochs": 200,
        },
        "initial_tensors": {
            "scene": scenes[0].name,
            "cost_volume_shape": list(vol.shape),
            "aggregated_cost_shape": list(cost.shape),
            "cost_slice_max_abs_diff_vs_slice0": slice_max_diff,
            "degenerate_volume_confirmed": slice_max_diff == 0.0,
            "standardised_cost_mean": float(scaled.mean()),
            "standardised_cost_std": float(scaled.std()),
            "softmax_entropy_nats": entropy,
            "softmax_max_probability": float(probs.max()),
            "disparity_initial_mean": float(di.mean()),
            "disparity_initial_std": float(di.std()),
            "disparity_initial_min": float(di.min()),
            "disparity_initial_max": float(di.max()),
            "initial_training_loss": float(loss0),
            "matching_path_gradient_norm": grads["matching_path"],
            "matching_path_gradient_present": bool(grads["matching_present"]),
        },
    }
    result["PASS"] = bool(
        result["initialisation"]["treatment_matches_expected"]
        and result["initialisation"]["treatment_equals_control"]
        and result["architecture"]["parameters_equal"]
        and result["architecture"]["refinement_blocks"] == 6
        and result["architecture"]["num_disparities"] == 12
        and result["architecture"]["treatment_shift"] == TREATMENT_SHIFT
        and result["architecture"]["regression"] == "StandardisedDisparityRegression"
        and result["provenance"]["phase_1_diff_vs_frozen"] == ""
        and result["initial_tensors"]["degenerate_volume_confirmed"]
        and result["initial_tensors"]["matching_path_gradient_present"])

    # run_training requires a preflight.json with checks.PASS
    payload = dict(result)
    payload["checks"] = {"PASS": result["PASS"]}
    (e3b.RESULT_DIR / "preflight.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("architecture", "initialisation", "initial_tensors", "PASS")},
                     indent=2))
    if not result["PASS"]:
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
    exp_id = e3b.arm_id("A")

    meta = {"experiment_id": exp_id, "controls": controls,
            "changed_variable": "cost_volume_shift: left -> none",
            "provenance": provenance(), "environment": environment(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    model = build_treatment(device)
    meta["init_weight_sha"] = weight_sha(model)
    meta["init_weight_sha_matches"] = meta["init_weight_sha"] == EXPECTED_INIT_WEIGHT_SHA
    meta["shift_at_build"] = model.cost_volume.shift
    if not meta["init_weight_sha_matches"] or meta["shift_at_build"] != TREATMENT_SHIFT:
        (out / "ABORTED.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("STOP:", meta["init_weight_sha"], meta["shift_at_build"])
        return 1
    del model

    t0 = time.time()
    e3b.run_training("A", device)              # the historical code path
    meta["wall_clock_s"] = time.time() - t0
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    ckpt = e3b.OUT_DIR / (exp_id + "_checkpoint.pth")
    meta["checkpoint"] = str(ckpt.relative_to(REPO_ROOT))
    meta["checkpoint_sha256"] = file_sha256(ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    tensors = state["model"] if "model" in state else state
    meta["final_weight_sha"] = sha(torch.cat(
        [v.reshape(-1).float() for v in tensors.values() if v.is_floating_point()]))
    meta["snapshots"] = {p.name: file_sha256(p)
                         for p in sorted(e3b.OUT_DIR.glob(exp_id + "_epoch*.pth"))}

    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("done in {:.1f}s; final weight sha {}".format(
        meta["wall_clock_s"], meta["final_weight_sha"]))
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
    return {
        "protocol": ("all 40 hailo_val scenes, full 368x1232 frames, pooled over "
                     "gt > 0 -- the existing evaluator, unchanged"),
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
        "per_scene": rows,
    }


def cmd_evaluate40(args) -> int:
    out = Path(args.out).resolve()
    repoint(out)
    enable_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id = e3b.arm_id("A")

    model = build_treatment(device)
    ckpt = e3b.OUT_DIR / (exp_id + "_checkpoint.pth")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    assert model.cost_volume.shift == TREATMENT_SHIFT

    result = _eval40(model, device)
    result.update({"experiment_id": exp_id, "shift": TREATMENT_SHIFT,
                   "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
                   "checkpoint_sha256": file_sha256(ckpt),
                   "environment": environment(), "provenance": provenance()})
    (out / "eval40.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    p = result["pooled"]
    print("treatment 40-scene  EPE {:.7f}  D1 {:.7f}  RMSE {:.7f}".format(
        p["epe"], p["d1"], p["rmse"]))
    return 0


# -------------------------------------------------------------------- probes
def cmd_probes(args) -> int:
    """Stereo probes at epochs 10/50/100/150/200 for treatment AND control.

    Measurement only, on weight snapshots both runs already saved. The probe is
    `stereo_probe`, imported unchanged -- the method EXP-H2-EMERGENCE-001 used.
    """
    out = Path(args.out).resolve()
    repoint(out)
    enable_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id = e3b.arm_id("A")
    scenes = [core.load_scene(i) for i in FOCUS_SCENES]

    def probe_snapshots(builder, ckpt_dir: Path, stem: str, shift: str) -> dict:
        rows = {}
        for epoch in PROBE_EPOCHS:
            path = ckpt_dir / "{}_epoch{}.pth".format(stem, epoch)
            if not path.exists():
                rows[str(epoch)] = {"error": "snapshot missing: " + path.name}
                continue
            model = builder(device)
            assert model.cost_volume.shift == shift
            state = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"] if "model" in state else state)
            model.eval()
            rng = np.random.default_rng(0)      # fresh per checkpoint, as in emergence
            ablation = stereo_probe(model, scenes, device, rng)
            right = min(r[k]["d1_penalty"] for r in ablation
                        for k in ("right_black", "right_noise", "right_equals_left"))
            mmap = min(r[k]["d1_penalty"] for r in ablation
                       for k in ("initial_constant_mean", "initial_shuffled"))
            stats = validation_stage_stats(
                model, Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015",
                                       split="hailo_val"), device)
            rows[str(epoch)] = {
                "epoch": epoch,
                "right_image_min_d1_penalty": right,
                "right_image_mean_d1_penalty": float(np.mean(
                    [r[k]["d1_penalty"] for r in ablation
                     for k in ("right_black", "right_noise", "right_equals_left")])),
                "matching_map_min_d1_penalty": mmap,
                "softmax_entropy": stats["entropy"],
                "max_softmax_weight": stats["max_weight"],
                "final_disparity_std": stats["final_std"],
                "disparity_initial_std": stats["initial_std"],
                "disparity_initial_corr_gt": stats["initial_corr_gt"],
                "final_corr_gt": stats["final_corr_gt"],
                "per_scene": ablation,
            }
            print("  epoch {:>3}  right {:+8.3f}  map {:+8.3f}  entropy {:.4f}  "
                  "init r(GT) {:+.4f}".format(
                      epoch, right, mmap, stats["entropy"], stats["initial_corr_gt"]))
        return rows

    print("treatment (shift=none):")
    treatment = probe_snapshots(build_treatment, e3b.OUT_DIR, exp_id, TREATMENT_SHIFT)
    print("control (shift=left, Stage B run A):")
    control = probe_snapshots(lambda d: build_model(0, d),
                              CONTROL_RECORD / "checkpoints", CONTROL_ID, CONTROL_SHIFT)

    payload = {"probe_epochs": list(PROBE_EPOCHS),
               "probe_implementation": "exp_h2_seed_replication.stereo_probe, imported unchanged",
               "focus_scenes": FOCUS_SCENES,
               "caveat": ("right-image and matching-map probes establish binocular "
                          "dependence, NOT correct disparity search: Phase 1 EXP-007 "
                          "measured +89.7 D1 right-image dependence on the reference "
                          "weights despite a provably degenerate shift."),
               "treatment": treatment, "control": control,
               "environment": environment(), "provenance": provenance()}
    (out / "probes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", out / "probes.json")
    return 0


# ------------------------------------------------------------------- compare
def cmd_compare(args) -> int:
    out = Path(args.out).resolve()
    repoint(out)
    exp_id = e3b.arm_id("A")

    treat_eval = json.loads((out / "eval40.json").read_text(encoding="utf-8"))
    ctrl_eval = json.loads((CONTROL_RECORD / "eval40_run_A.json").read_text(encoding="utf-8"))
    treat_m = json.loads((out / "experiments" / exp_id / "metrics.json")
                         .read_text(encoding="utf-8"))["metrics"]
    ctrl_m = json.loads((CONTROL_RECORD / "experiments" / CONTROL_ID / "metrics.json")
                        .read_text(encoding="utf-8"))["metrics"]
    treat_meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    ctrl_meta = json.loads((CONTROL_RECORD / "run_A_meta.json").read_text(encoding="utf-8"))
    probes = json.loads((out / "probes.json").read_text(encoding="utf-8"))

    tp, cp = treat_eval["pooled"], ctrl_eval["pooled"]
    per = []
    for t, c in zip(treat_eval["per_scene"], ctrl_eval["per_scene"]):
        assert t["scene"] == c["scene"]
        per.append({"scene": t["scene"], "valid_pixels": t["valid_pixels"],
                    "control_epe": c["epe"], "treatment_epe": t["epe"],
                    "delta_epe": t["epe"] - c["epe"],
                    "control_d1": c["d1"], "treatment_d1": t["d1"],
                    "delta_d1": t["d1"] - c["d1"]})
    de = np.array([r["delta_epe"] for r in per])
    dd = np.array([r["delta_d1"] for r in per])

    res = {
        "experiment": EXPERIMENT_PREFIX,
        "control_id": CONTROL_ID, "control_record": STAGE_B_DIR,
        "provenance": provenance(),
        "materiality_band": ("NONE reused, NONE invented. The historical E3b/E3c "
                             "band is invalidated; the effect is classified by "
                             "measured delta, per-scene consistency, stereo "
                             "evidence, and magnitude relative to the measured "
                             "protocol divergence (0.2250805 px / 1.4143253 pt)."),
        "aggregate_40_scene": {
            "control": cp, "treatment": tp,
            "delta_epe": tp["epe"] - cp["epe"], "delta_d1": tp["d1"] - cp["d1"],
            "delta_rmse": tp["rmse"] - cp["rmse"],
            "ratio_epe": tp["epe"] / cp["epe"], "ratio_d1": tp["d1"] / cp["d1"],
        },
        "per_scene": per,
        "per_scene_consistency": {
            "scenes": len(per),
            "treatment_worse_on_epe": int((de > 0).sum()),
            "treatment_worse_on_d1": int((dd > 0).sum()),
            "delta_epe": {"mean": float(de.mean()), "median": float(np.median(de)),
                          "sd": float(de.std(ddof=1)), "min": float(de.min()),
                          "max": float(de.max())},
            "delta_d1": {"mean": float(dd.mean()), "median": float(np.median(dd)),
                         "sd": float(dd.std(ddof=1)), "min": float(dd.min()),
                         "max": float(dd.max())},
        },
        "late_window": {"control": ctrl_m["late_window"],
                        "treatment": treat_m["late_window"],
                        "delta_epe": (treat_m["late_window"]["epe_mean"]
                                      - ctrl_m["late_window"]["epe_mean"]),
                        "delta_d1": (treat_m["late_window"]["d1_mean"]
                                     - ctrl_m["late_window"]["d1_mean"])},
        "training": {
            "control": {"epochs": ctrl_m["epochs_completed"],
                        "aborted": ctrl_m["aborted"],
                        "final_train_loss": ctrl_m["history"][-1]["mean_loss"],
                        "epoch0_val_epe": ctrl_m["checkpoints"]["0"]["val_epe"],
                        "gradient_norms": ctrl_m["gradient_norms"],
                        "matching_gradient": ctrl_m["primary_endpoint_matching_gradient"],
                        "wall_clock_s": ctrl_meta["wall_clock_s"],
                        "final_weight_sha": ctrl_meta["final_weight_sha"]},
            "treatment": {"epochs": treat_m["epochs_completed"],
                          "aborted": treat_m["aborted"],
                          "final_train_loss": treat_m["history"][-1]["mean_loss"],
                          "epoch0_val_epe": treat_m["checkpoints"]["0"]["val_epe"],
                          "gradient_norms": treat_m["gradient_norms"],
                          "matching_gradient": treat_m["primary_endpoint_matching_gradient"],
                          "wall_clock_s": treat_meta["wall_clock_s"],
                          "final_weight_sha": treat_meta["final_weight_sha"]},
        },
        "stereo_verdicts": {"control": ctrl_m["stereo_verdict_epoch_200"],
                            "treatment": treat_m["stereo_verdict_epoch_200"]},
        "probe_series": {
            e: {"epoch": int(e),
                "control_right": probes["control"][e].get("right_image_min_d1_penalty"),
                "treatment_right": probes["treatment"][e].get("right_image_min_d1_penalty"),
                "control_map": probes["control"][e].get("matching_map_min_d1_penalty"),
                "treatment_map": probes["treatment"][e].get("matching_map_min_d1_penalty"),
                "control_init_corr_gt": probes["control"][e].get("disparity_initial_corr_gt"),
                "treatment_init_corr_gt": probes["treatment"][e].get("disparity_initial_corr_gt"),
                "control_entropy": probes["control"][e].get("softmax_entropy"),
                "treatment_entropy": probes["treatment"][e].get("softmax_entropy")}
            for e in map(str, PROBE_EPOCHS)},
        "reference_scales": {
            "stage_b_protocol_divergence_epe": 0.2250805,
            "stage_b_protocol_divergence_d1": 1.4143253,
            "o6_uncontrolled_same_config_epe": 0.3426,
            "o6_uncontrolled_same_config_d1": 1.9701,
            "note": "HISTORICAL magnitudes, quoted for scale only.",
        },
    }
    (out / "comparison.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in
                      ("aggregate_40_scene", "per_scene_consistency", "late_window",
                       "training", "stereo_verdicts", "probe_series")}, indent=2)[:7000])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "train", "evaluate40", "probes", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
    args = ap.parse_args()
    return {"preflight": cmd_preflight, "train": cmd_train,
            "evaluate40": cmd_evaluate40, "probes": cmd_probes,
            "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
