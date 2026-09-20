"""EXP-BLOCKCOUNT-SEEDSCREEN-001 -- cheap 30-epoch seed/trajectory screen.

The deterministic 6/5/4 block series at seed 0 ordered 6 < 5 > 4 on EPE, so the
capacity question cannot be settled from one seed. A full 6-run x 200-epoch
campaign costs ~8 GPU-hours. This is the CHEAP SCREEN that decides whether that
campaign is justified -- it is NOT the block-count experiment.

    block count in {6, 5, 4}  x  seed in {1, 2}  x  30 epochs   = 6 runs

Everything else is frozen and imported unchanged from the Stage B / E3b recipe.
Seed 0 is NOT retrained: its 10/20/30-epoch validation points are read out of the
existing frozen records for context only.

    python .../exp_blockcount_seed_screen.py preflight  --out DIR
    python .../exp_blockcount_seed_screen.py train      --out DIR --arm A --seed 1
    python .../exp_blockcount_seed_screen.py evaluate40 --out DIR --arm A --seed 1
    python .../exp_blockcount_seed_screen.py probes     --out DIR
    python .../exp_blockcount_seed_screen.py compare    --out DIR

Writes only under --out. Phase 1, every Phase 2 scientific script and every
historical record (H1, H2, H3, E1, E2, E3, E3b, E3c, O6, Stage A, Stage B, the
factorial cells) are read-only here.

THE ONE PROTOCOL SUBTLETY, STATED LOUDLY
----------------------------------------
The epoch BUDGET is 30. The learning-rate SCHEDULE is NOT re-fitted to 30: the
cosine schedule keeps its frozen ``T_max = 200``. A 30-epoch run is therefore a
strict PREFIX of the 200-epoch run it screens for, not a separate fully-annealed
recipe. Re-fitting T_max to 30 would change the learning rate at every epoch,
which the design forbids ("same learning-rate schedule", "do not change the
learning rate"), and would make the screen unable to say anything about the
200-epoch trajectory it exists to forecast. This is enforced by
``FrozenCosineAnnealingLR`` and asserted in the preflight.
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
import phase2.scripts.exp_h2_seed_replication as h2             # noqa: E402
from phase2.scripts.exp_h2_seed_replication import (            # noqa: E402
    FOCUS_SCENES, measure_macs, stereo_probe,
)
from phase2.scripts.exp_h2_softargmin_scale import (            # noqa: E402
    validation_stage_stats,
)
from phase2.viz import core                                     # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize   # noqa: E402
from src.evaluation.metrics import disparity_metrics            # noqa: E402

EXPERIMENT_PREFIX = "EXP-BLOCKCOUNT-SEEDSCREEN-001"
ARMS = ("A", "B", "C")            # 6, 5, 4 refinement blocks
SEEDS = (1, 2)
SCREEN_EPOCHS = 30                # the BUDGET
FULL_SCHEDULE_EPOCHS = 200        # the frozen SCHEDULE -- never re-fitted
PROBE_EPOCHS = (10, 20, 30)
SCREEN_CHECKPOINT_EPOCHS = (1, 2, 5, 10, 20, 30)
SCREEN_SNAPSHOT_EPOCHS = (10, 20, 30)
FROZEN_COMMIT_REF = "phase-1-frozen^" + "{commit}"

# Seed-0 records, READ-ONLY. Not retrained, not overwritten, quoted for context.
SEED0 = {
    "A": {"blocks": 6,
          "record": "phase2/diagnostics/determinism/20260909T041500Z_baseline",
          "experiment": "STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA",
          "frozen_40_scene": {"epe": 2.2955181809208267, "d1": 15.6045142562172}},
    "B": {"blocks": 5,
          "record": "phase2/factorial/block_count_deterministic/20260909T131500Z",
          "experiment": "EXP-BLOCKCOUNT-DETERMINISTIC-001-ARM-B",
          "frozen_40_scene": {"epe": 2.6458, "d1": 21.3712}},
    "C": {"blocks": 4,
          "record": "phase2/factorial/block_count_deterministic/20260909T131500Z",
          "experiment": "EXP-BLOCKCOUNT-DETERMINISTIC-001-ARM-C",
          "frozen_40_scene": {"epe": 2.3387, "d1": 16.9797}},
}
# Stage B seed-0 initial weight sha. Fresh seed-1/2 inits MUST differ from it;
# they are never forced to match it.
SEED0_INIT_SHA = "c4d02385e3da28a5"


# --------------------------------------------------------- the frozen schedule
_RealCosineAnnealingLR = torch.optim.lr_scheduler.CosineAnnealingLR


class FrozenCosineAnnealingLR(_RealCosineAnnealingLR):
    """CosineAnnealingLR pinned to the frozen T_max, whatever the caller passes.

    ``e3b.run_training`` builds its scheduler as ``CosineAnnealingLR(opt,
    T_max=EPOCHS)``. Shortening the epoch budget to 30 would otherwise silently
    re-fit the cosine to 30 epochs and change the learning rate at every step.
    The screen must be a prefix of the 200-epoch run, so T_max stays at 200.
    """

    requested_T_max = None

    def __init__(self, optimizer, T_max, *args, **kwargs):
        type(self).requested_T_max = T_max
        super().__init__(optimizer, FULL_SCHEDULE_EPOCHS, *args, **kwargs)


# ------------------------------------------------------------------- utilities
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


def run_id(arm: str, seed: int) -> str:
    return "{}-ARM-{}-SEED{}".format(EXPERIMENT_PREFIX, arm, seed)


# --------------------------------------------------------------------- repoint
def repoint(out: Path, seed) -> None:
    """Point the frozen E3b training code at this screen's identity and budget.

    Only paths, identity, seed and the epoch budget are changed. The model,
    optimizer, loss, dataset, crop, augmentation, batch size, probes and the
    learning-rate schedule are the imported originals.
    """
    out = out.resolve()
    e3b.EXPERIMENT_PREFIX = EXPERIMENT_PREFIX
    e3b.RESULT_DIR = out / "results"
    e3b.EXPERIMENTS_DIR = out / "experiments"
    e3b.OUT_DIR = out / "checkpoints"
    for d in (e3b.RESULT_DIR, e3b.EXPERIMENTS_DIR, e3b.OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Epoch budget -> 30. `h2.EPOCHS` is set too because `_stereo_verdict` reads
    # it to index the final checkpoint; leaving it at 200 would KeyError.
    e3b.EPOCHS = SCREEN_EPOCHS
    h2.EPOCHS = SCREEN_EPOCHS
    e3b.CHECKPOINT_EPOCHS = SCREEN_CHECKPOINT_EPOCHS
    e3b.WEIGHT_SNAPSHOT_EPOCHS = SCREEN_SNAPSHOT_EPOCHS

    # ...but NOT the schedule.
    torch.optim.lr_scheduler.CosineAnnealingLR = FrozenCosineAnnealingLR

    if seed is not None:
        e3b.SEED = seed
        e3b.ARM_LABEL = "-SEED{}".format(seed)

    original = e3b.arm_config

    def screen_arm_config(arm: str, device: str) -> dict:
        config = dict(original(arm, device))
        config.update({
            "experiment": e3b.arm_id(arm),
            "experiment_kind": ("CHEAP 30-EPOCH SEED/TRAJECTORY SCREEN. NOT the "
                                "block-count experiment and NOT a capacity "
                                "measurement. Its only job is to decide whether a "
                                "full multi-seed 200-epoch block-count experiment "
                                "is justified."),
            "seed": e3b.SEED,
            "epochs": SCREEN_EPOCHS,
            "epoch_budget": SCREEN_EPOCHS,
            "scheduler": ("CosineAnnealingLR T_max={} -- FROZEN at the 200-epoch "
                          "value and deliberately NOT re-fitted to the 30-epoch "
                          "budget, so this run is a strict prefix of the "
                          "200-epoch run it screens for".format(FULL_SCHEDULE_EPOCHS)),
            "control": ("the seed-0 records, READ-ONLY and NOT retrained: Stage B "
                        "deterministic run A (6 blocks) and "
                        "EXP-BLOCKCOUNT-DETERMINISTIC-001 arms B/C (5/4 blocks)"),
            "changed_variables": ["refinement block count", "seed"],
            "deterministic_controls": {
                "torch.use_deterministic_algorithms": True,
                "torch.backends.cudnn.deterministic": True,
                "torch.backends.cudnn.benchmark": False,
                "CUBLAS_WORKSPACE_CONFIG": CUBLAS_CONFIG,
            },
            "materiality_band": ("NONE. The historical E3b/E3c band is invalidated. "
                                 "No band is reused and none is invented here, "
                                 "before or after seeing the results."),
            "claim_ceiling": ("A 30-epoch screen CANNOT establish the 200-epoch "
                              "capacity ranking. No statement of the form "
                              "'N blocks are required/equivalent/superior' is "
                              "supportable outside the phrase 'in the 30-epoch "
                              "screen'."),
        })
        return config

    e3b.arm_config = screen_arm_config


# ------------------------------------------------------------------- preflight
def cmd_preflight(args) -> int:
    out = Path(args.out).resolve()
    repoint(out, None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    controls = enable_determinism()

    runs, fresh_ok, identical_ok = {}, True, True
    for seed in SEEDS:
        e3b.SEED = seed
        ref = e3b.build_arm("A", device)
        for arm in ARMS:
            model = e3b.build_arm(arm, device)
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
            identical_ok = identical_ok and ok
            init = weight_sha(model)
            fresh_ok = fresh_ok and init != SEED0_INIT_SHA
            runs[run_id(arm, seed)] = {
                "arm": arm, "seed": seed,
                "blocks": len(model.refinement.blocks),
                "blocks_expected": e3b.ARMS[arm]["blocks"],
                "dilations": [int(b.conv1.dilation[0]) for b in model.refinement.blocks],
                "dilations_expected": list(e3b.ARMS[arm]["dilations"]),
                "parameters": model.parameter_count(),
                "parameters_expected": e3b.ARMS[arm]["parameters"],
                "parameters_match": model.parameter_count() == e3b.ARMS[arm]["parameters"],
                "init_weight_sha": init,
                "init_differs_from_seed0": init != SEED0_INIT_SHA,
                "surviving_weights_bit_identical_within_seed": ok,
                "cost_volume_shift": model.cost_volume.shift,
                "num_disparities": model.cost_volume.num_disparities,
                "regression": type(model.regression).__name__,
                "epoch_budget": SCREEN_EPOCHS,
                "loaded_from_stage_b_checkpoint": False,
                "macs_256x512": measure_macs(e3b.build_arm(arm, device), device),
            }
            del model
        del ref

    # The schedule is pinned, and the optimizer/scheduler are freshly built.
    probe_model = e3b.build_arm("A", device)
    probe_opt = torch.optim.Adam(probe_model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    probe_sched = torch.optim.lr_scheduler.CosineAnnealingLR(probe_opt,
                                                             T_max=e3b.EPOCHS)
    schedule = {
        "class": type(probe_sched).__name__,
        "T_max_requested_by_training_code": FrozenCosineAnnealingLR.requested_T_max,
        "T_max_actually_used": probe_sched.T_max,
        "lr_epoch_0": float(probe_opt.param_groups[0]["lr"]),
        "optimizer_state_empty_at_construction": len(probe_opt.state) == 0,
    }
    del probe_sched, probe_opt, probe_model

    result = {
        "experiment": EXPERIMENT_PREFIX,
        "purpose": ("cheap 30-epoch seed/trajectory screen; decides whether a full "
                    "multi-seed 200-epoch block-count experiment is justified"),
        "provenance": provenance(), "environment": environment(),
        "deterministic_controls": controls,
        "schedule": schedule,
        "seed0_records_readonly": SEED0,
        "runs": runs,
    }
    result["checks"] = {
        "phase_1_diff_empty": result["provenance"]["phase_1_diff_vs_frozen"] == "",
        "deterministic_controls_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
            and torch.backends.cudnn.deterministic
            and not torch.backends.cudnn.benchmark
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_CONFIG),
        "shift_is_left": all(r["cost_volume_shift"] == "left" for r in runs.values()),
        "readout_is_standardised": all(
            r["regression"] == "StandardisedDisparityRegression" for r in runs.values()),
        "candidates_are_12": all(r["num_disparities"] == 12 for r in runs.values()),
        "block_counts_correct": all(
            r["blocks"] == r["blocks_expected"] for r in runs.values()),
        "dilations_correct": all(
            r["dilations"] == r["dilations_expected"] for r in runs.values()),
        "parameter_counts_match_registration": all(
            r["parameters_match"] for r in runs.values()),
        "seeds_are_1_and_2_only": sorted({r["seed"] for r in runs.values()}) == [1, 2],
        "seed_0_not_scheduled": all(r["seed"] != 0 for r in runs.values()),
        "epoch_budget_is_30": all(r["epoch_budget"] == 30 for r in runs.values()),
        "budget_is_not_200": SCREEN_EPOCHS != 200,
        "schedule_T_max_frozen_at_200": schedule["T_max_actually_used"] == 200,
        "initialisation_is_fresh_not_stage_b": bool(fresh_ok),
        "no_stage_b_checkpoint_loaded": all(
            not r["loaded_from_stage_b_checkpoint"] for r in runs.values()),
        "optimizer_state_fresh": bool(schedule["optimizer_state_empty_at_construction"]),
        "surviving_weights_bit_identical_within_seed": bool(identical_ok),
        "run_count_is_6": len(runs) == 6,
    }
    result["checks"]["PASS"] = all(v for v in result["checks"].values()
                                   if isinstance(v, bool))
    (e3b.RESULT_DIR / "preflight.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"schedule": schedule,
                      "runs": {k: {kk: v[kk] for kk in
                                   ("arm", "seed", "blocks", "parameters",
                                    "init_weight_sha", "init_differs_from_seed0")}
                               for k, v in runs.items()},
                      "checks": result["checks"]}, indent=2))
    if not result["checks"]["PASS"]:
        print("\nPREFLIGHT FAILED -- STOP. Nothing was trained.")
        return 1
    print("\nPREFLIGHT PASS.")
    return 0


# ----------------------------------------------------------------------- train
def cmd_train(args) -> int:
    out = Path(args.out).resolve()
    arm, seed = args.arm, int(args.seed)
    if seed == 0:
        print("STOP: seed 0 is frozen and must not be retrained.")
        return 1
    repoint(out, seed)
    controls = enable_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id = e3b.arm_id(arm)
    assert exp_id == run_id(arm, seed), (exp_id, run_id(arm, seed))

    model = e3b.build_arm(arm, device)
    meta = {"arm": arm, "seed": seed, "experiment_id": exp_id, "controls": controls,
            "blocks": len(model.refinement.blocks),
            "parameters": model.parameter_count(),
            "init_weight_sha": weight_sha(model),
            "epoch_budget": e3b.EPOCHS,
            "schedule_T_max": FULL_SCHEDULE_EPOCHS,
            "changed_variables": {"refinement_blocks": e3b.ARMS[arm]["blocks"],
                                  "seed": seed},
            "initialised_from": "fresh, per the frozen recipe; no checkpoint loaded",
            "provenance": provenance(), "environment": environment(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    stop = None
    if meta["parameters"] != e3b.ARMS[arm]["parameters"]:
        stop = "parameter count {}".format(meta["parameters"])
    if meta["init_weight_sha"] == SEED0_INIT_SHA:
        stop = "initial weights match the seed-0 initialisation"
    if e3b.EPOCHS != SCREEN_EPOCHS:
        stop = "epoch budget {}".format(e3b.EPOCHS)
    if stop:
        (out / "ABORTED_{}.json".format(exp_id)).write_text(
            json.dumps(dict(meta, stop=stop), indent=2), encoding="utf-8")
        print("STOP:", stop)
        return 1
    del model

    t0 = time.time()
    e3b.run_training(arm, device)
    meta["wall_clock_s"] = time.time() - t0
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["schedule_T_max_requested_by_training_code"] = FrozenCosineAnnealingLR.requested_T_max
    meta["schedule_note"] = ("training code requested T_max={}; the frozen 200-epoch "
                             "schedule was used instead, on purpose".format(
                                 FrozenCosineAnnealingLR.requested_T_max))

    ckpt = e3b.OUT_DIR / (exp_id + "_checkpoint.pth")
    meta["checkpoint"] = str(ckpt.relative_to(REPO_ROOT))
    meta["checkpoint_sha256"] = file_sha256(ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    tensors = state["model"] if "model" in state else state
    meta["final_weight_sha"] = sha(torch.cat(
        [v.reshape(-1).float() for v in tensors.values() if v.is_floating_point()]))
    meta["snapshots"] = sorted(
        p.name for p in e3b.OUT_DIR.glob(exp_id + "_epoch*.pth"))
    (out / "run_meta_{}.json".format(exp_id)).write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print("{} done in {:.1f}s; final weight sha {}".format(
        exp_id, meta["wall_clock_s"], meta["final_weight_sha"]))
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
            "epe": {"mean": float(epes.mean()), "median": float(np.median(epes)),
                    "sd": float(epes.std(ddof=1)), "min": float(epes.min()),
                    "max": float(epes.max())},
            "d1": {"mean": float(d1s.mean()), "median": float(np.median(d1s)),
                   "sd": float(d1s.std(ddof=1)), "min": float(d1s.min()),
                   "max": float(d1s.max())}},
        "per_scene": rows,
    }


def cmd_evaluate40(args) -> int:
    out = Path(args.out).resolve()
    arm, seed = args.arm, int(args.seed)
    repoint(out, seed)
    enable_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id = e3b.arm_id(arm)
    model = e3b.build_arm(arm, device)
    ckpt = e3b.OUT_DIR / (exp_id + "_checkpoint.pth")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    result = _eval40(model, device)
    result.update({"experiment_id": exp_id, "arm": arm, "seed": seed,
                   "epoch": SCREEN_EPOCHS,
                   "blocks": len(model.refinement.blocks),
                   "checkpoint_sha256": file_sha256(ckpt),
                   "environment": environment(), "provenance": provenance()})
    (out / "eval40_{}.json".format(exp_id)).write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    p = result["pooled"]
    print("{} 40-scene @30ep  EPE {:.7f}  D1 {:.7f}  RMSE {:.7f}".format(
        exp_id, p["epe"], p["d1"], p["rmse"]))
    return 0


# ---------------------------------------------------------------------- probes
def cmd_probes(args) -> int:
    out = Path(args.out).resolve()
    repoint(out, None)
    enable_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scenes = [core.load_scene(i) for i in FOCUS_SCENES]
    val_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val")

    payload = {"probe_epochs": list(PROBE_EPOCHS), "focus_scenes": FOCUS_SCENES,
               "probe_implementation": ("exp_h2_seed_replication.stereo_probe and "
                                        "exp_h2_softargmin_scale."
                                        "validation_stage_stats, imported unchanged "
                                        "-- no new stereo metric was introduced"),
               "caveat": ("right-image and matching-map probes establish binocular "
                          "DEPENDENCE, not correct disparity search: Phase 1 "
                          "EXP-007 measured +89.7 D1 on the reference weights "
                          "despite a provably degenerate shift."),
               "runs": {}}
    for seed in SEEDS:
        e3b.SEED = seed
        e3b.ARM_LABEL = "-SEED{}".format(seed)
        for arm in ARMS:
            exp_id = e3b.arm_id(arm)
            print("{} ({} blocks, seed {}):".format(
                exp_id, e3b.ARMS[arm]["blocks"], seed))
            rows = {}
            for epoch in PROBE_EPOCHS:
                path = e3b.OUT_DIR / "{}_epoch{}.pth".format(exp_id, epoch)
                if not path.exists():
                    rows[str(epoch)] = {"error": "snapshot missing: " + path.name}
                    print("    epoch {:>3}  MISSING".format(epoch))
                    continue
                model = e3b.build_arm(arm, device)
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
                del model
            payload["runs"][exp_id] = {"arm": arm, "seed": seed,
                                       "blocks": e3b.ARMS[arm]["blocks"],
                                       "series": rows}
    payload["environment"] = environment()
    payload["provenance"] = provenance()
    (out / "probes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", out / "probes.json")
    return 0


# --------------------------------------------------------------------- compare
def _val_at(history, done: int):
    """Validation row for the epoch after `done` completed epochs (rows are 0-based)."""
    for row in history:
        if row.get("epoch") == done - 1 and "val_epe" in row:
            return {"epe": row["val_epe"], "d1": row["val_d1"],
                    "train_loss": row.get("mean_loss"),
                    "median_grad_norm": row.get("median_grad_norm"),
                    "median_matching_grad_norm": row.get("median_matching_grad_norm"),
                    "matching_present_fraction_this_epoch":
                        row.get("matching_present_fraction_this_epoch")}
    return None


def _order(entries, key: str):
    ok = [(name, v[key]) for name, v in entries
          if v is not None and v.get(key) is not None]
    return [n for n, _ in sorted(ok, key=lambda kv: kv[1])]


def cmd_compare(args) -> int:
    out = Path(args.out).resolve()
    repoint(out, None)
    probes = json.loads((out / "probes.json").read_text(encoding="utf-8"))
    pre = json.loads((out / "results" / "preflight.json").read_text(encoding="utf-8"))

    res = {"experiment": EXPERIMENT_PREFIX,
           "purpose": ("30-epoch seed/trajectory screen -- decides only whether a "
                       "full multi-seed 200-epoch block-count experiment is "
                       "justified"),
           "provenance": provenance(),
           "epoch_budget": SCREEN_EPOCHS,
           "schedule_T_max": FULL_SCHEDULE_EPOCHS,
           "materiality_band": "NONE reused, NONE invented.",
           "claim_ceiling": ("This 30-epoch screen does not establish the final "
                             "200-epoch capacity ranking."),
           "runs": {}}

    for seed in SEEDS:
        e3b.SEED = seed
        e3b.ARM_LABEL = "-SEED{}".format(seed)
        for arm in ARMS:
            exp_id = e3b.arm_id(arm)
            m = json.loads((out / "experiments" / exp_id / "metrics.json")
                           .read_text(encoding="utf-8"))["metrics"]
            meta = json.loads((out / "run_meta_{}.json".format(exp_id))
                              .read_text(encoding="utf-8"))
            ev_path = out / "eval40_{}.json".format(exp_id)
            ev = (json.loads(ev_path.read_text(encoding="utf-8"))
                  if ev_path.exists() else None)
            res["runs"][exp_id] = {
                "arm": arm, "seed": seed, "blocks": e3b.ARMS[arm]["blocks"],
                "parameters": meta["parameters"],
                "macs_256x512": pre["runs"][exp_id]["macs_256x512"]["macs"],
                "init_weight_sha": meta["init_weight_sha"],
                "final_weight_sha": meta["final_weight_sha"],
                "epochs_completed": m["epochs_completed"],
                "aborted": m["aborted"],
                "at_epoch": {str(e): _val_at(m["history"], e) for e in PROBE_EPOCHS},
                "final_train_loss": m["history"][-1]["mean_loss"],
                "gradient_norms": m["gradient_norms"],
                "matching_gradient": m["primary_endpoint_matching_gradient"],
                "nan_inf": {"any_nan_in_gradients": m["gradient_norms"]["any_nan"],
                            "aborted_on_non_finite": bool(
                                m["aborted"] and "non-finite" in str(m["aborted"]))},
                "gate_epoch_10": m.get("gate_epoch_10"),
                "stereo_status_at_epoch_30": m.get("stereo_verdict_epoch_200"),
                "eval40_pooled": ev["pooled"] if ev else None,
                "wall_clock_s": meta["wall_clock_s"],
                "probes": probes["runs"][exp_id]["series"],
            }

    # Seed-0 context, read out of the frozen records. Never retrained here.
    seed0 = {}
    for arm, spec in SEED0.items():
        mpath = (REPO_ROOT / spec["record"] / "experiments" / spec["experiment"]
                 / "metrics.json")
        hist = json.loads(mpath.read_text(encoding="utf-8"))["metrics"]["history"]
        seed0["ARM-{}-SEED0".format(arm)] = {
            "arm": arm, "seed": 0, "blocks": spec["blocks"],
            "source": spec["record"] + "/experiments/" + spec["experiment"],
            "retrained_here": False,
            "at_epoch": {str(e): _val_at(hist, e) for e in PROBE_EPOCHS},
            "frozen_40_scene_at_epoch_200": spec["frozen_40_scene"],
        }
    res["seed0_context_readonly"] = seed0

    # Architecture ordering at each checkpoint, per seed (best -> worst).
    ordering = {}
    for seed in list(SEEDS) + [0]:
        per_epoch = {}
        for e in PROBE_EPOCHS:
            if seed == 0:
                entries = [("{}b".format(SEED0[a]["blocks"]),
                            seed0["ARM-{}-SEED0".format(a)]["at_epoch"][str(e)])
                           for a in ARMS]
            else:
                entries = [("{}b".format(e3b.ARMS[a]["blocks"]),
                            res["runs"][run_id(a, seed)]["at_epoch"][str(e)])
                           for a in ARMS]
            per_epoch[str(e)] = {"epe_best_to_worst": _order(entries, "epe"),
                                 "d1_best_to_worst": _order(entries, "d1")}
        ordering["seed{}".format(seed)] = per_epoch
    res["architecture_ordering"] = ordering

    # Seed effect: spread across seeds 1 and 2 at each block count and epoch.
    seed_effect = {}
    for arm in ARMS:
        rows = {}
        for e in PROBE_EPOCHS:
            vals = {s: res["runs"][run_id(arm, s)]["at_epoch"][str(e)] for s in SEEDS}
            if all(v is not None for v in vals.values()):
                rows[str(e)] = {
                    "seed1_epe": vals[1]["epe"], "seed2_epe": vals[2]["epe"],
                    "abs_delta_epe": abs(vals[1]["epe"] - vals[2]["epe"]),
                    "seed1_d1": vals[1]["d1"], "seed2_d1": vals[2]["d1"],
                    "abs_delta_d1": abs(vals[1]["d1"] - vals[2]["d1"])}
        seed_effect["{}_blocks".format(e3b.ARMS[arm]["blocks"])] = rows
    res["seed_effect"] = seed_effect

    # Architecture spread within a seed, for comparison with the seed spread.
    arch_spread = {}
    for seed in SEEDS:
        rows = {}
        for e in PROBE_EPOCHS:
            vals = [res["runs"][run_id(a, seed)]["at_epoch"][str(e)] for a in ARMS]
            if all(v is not None for v in vals):
                rows[str(e)] = {
                    "epe_range": max(v["epe"] for v in vals) - min(v["epe"] for v in vals),
                    "d1_range": max(v["d1"] for v in vals) - min(v["d1"] for v in vals)}
        arch_spread["seed{}".format(seed)] = rows
    res["architecture_spread_within_seed"] = arch_spread

    (out / "comparison.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    def fmt(block, seed, at):
        def f(e, k):
            return "{:.4f}".format(at[str(e)][k]) if at.get(str(e)) else "n/a"
        return "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            block, seed, f(10, "epe"), f(20, "epe"), f(30, "epe"),
            f(10, "d1"), f(20, "d1"), f(30, "d1"))

    print("\n| Blocks | Seed | EPE@10 | EPE@20 | EPE@30 | D1@10 | D1@20 | D1@30 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for seed in SEEDS:
        for arm in ARMS:
            r = res["runs"][run_id(arm, seed)]
            print(fmt(r["blocks"], seed, r["at_epoch"]))
    print("\nseed-0 context (read-only, from the frozen records):")
    for r in seed0.values():
        print(fmt(r["blocks"], 0, r["at_epoch"]))
    print("\nordering (best->worst):")
    print(json.dumps(ordering, indent=2))
    print("\nseed effect:")
    print(json.dumps(seed_effect, indent=2))
    print("\narchitecture spread within a seed:")
    print(json.dumps(arch_spread, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "train", "evaluate40", "probes", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        if name in ("train", "evaluate40"):
            p.add_argument("--arm", required=True, choices=list(ARMS))
            p.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()
    return {"preflight": cmd_preflight, "train": cmd_train,
            "evaluate40": cmd_evaluate40, "probes": cmd_probes,
            "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
