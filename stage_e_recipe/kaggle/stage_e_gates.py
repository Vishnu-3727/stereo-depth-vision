#!/usr/bin/env python
"""Stage E Kaggle T4 pre-run gates. NO E0 TRAINING.

Runs the build-and-validate gates inside the environment that will actually
train, and measures the T4 rate so the campaign budget can be recalculated
from a measurement instead of an extrapolation from local hardware.

Gates:
    G1  environment capture
    G2  source integrity (bundle hashes re-verified on Kaggle)
    G3  initialization integrity (14 checks)
    G4  data and evaluation contract
    G5  one batch: forward, backward, step, save, reload
    G6  batch-8 memory probe (decides E2 implementation on T4)
    G7  rate probe (20 timed steps per config + one monitor pass)

What this is NOT: an E0 run. No epoch completes, no run directory is created,
no scored checkpoint is written, no record.json is emitted, and every weight
touched here is discarded. G5 and G7 take optimizer steps deliberately - that
is what proves the path works and measures its speed - but they are throwaway
processes that can never become an E0 result.

Kaggle executes only this file as /kaggle/src/script.py; sibling files pushed
with it are absent at runtime, so the bundle is loaded from its dataset mount.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

OUT = Path("/kaggle/working/stage_e_gates.json")
EXPECTED_INIT_SHA = "3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7"
EXPECTED_PARAMS = 397954
EXPECTED_KEYS = 70
EXPECTED_VALID_PX = 3802797
CFG = dict(downsample_levels=3, num_disparities=24,
           cost_volume_shift="right", regression_normalize=True)
CROP_H, CROP_W = 256, 512
RATE_STEPS = 20          # timed optimizer steps per configuration
MEM_HEADROOM_MIN = 0.20  # pre-registered in the spec, before any measurement

result: dict = {"gates": {}, "training_run": False, "e0_started": False}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bootstrap():
    """Import kaggle_bootstrap.py from the bundle mount (it is not a sibling
    of this script at runtime)."""
    hits = []
    base = Path("/kaggle/input")
    if base.is_dir():
        for dp, _dn, fn in os.walk(base):
            if "BUNDLE_MARKER.json" in fn and "kaggle_bootstrap.py" in fn:
                hits.append(Path(dp))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected exactly 1 bundle mount, found {len(hits)}: {hits}")
    mod_path = hits[0] / "kaggle_bootstrap.py"
    spec = importlib.util.spec_from_file_location("kaggle_bootstrap", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, hits[0]


def g1_environment() -> dict:
    import torch
    rec = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_memory_MiB": (round(torch.cuda.get_device_properties(0).total_memory
                                   / 2**20, 1) if torch.cuda.is_available() else None),
    }
    try:
        import cv2
        import numpy as np
        rec["numpy"], rec["opencv"] = np.__version__, cv2.__version__
    except Exception as e:
        rec["import_error"] = str(e)
    rec["pass"] = bool(rec["cuda_available"])
    return rec


def g2_source_integrity(bundle: Path) -> dict:
    """Re-verify every bundled file against source_integrity.json on Kaggle."""
    man_path = bundle / "source_integrity.json"
    if not man_path.is_file():
        return {"pass": False, "error": "source_integrity.json not in bundle"}
    man = json.loads(man_path.read_text())
    checked, mismatches = 0, []
    for rel, meta in man.get("byte_identical_files", {}).items():
        f = bundle / rel
        if not f.is_file():
            mismatches.append({"file": rel, "error": "missing on Kaggle"})
            continue
        got = sha256(f)
        checked += 1
        if got != meta["sha256"]:
            mismatches.append({"file": rel, "expected": meta["sha256"], "got": got})
    for rel, meta in man.get("declared_deltas", {}).items():
        f = bundle / rel
        got = sha256(f) if f.is_file() else None
        checked += 1
        if got != meta["bundle_sha256"]:
            mismatches.append({"file": rel, "expected": meta["bundle_sha256"],
                               "got": got, "note": "declared delta"})
    return {"pass": not mismatches, "files_checked": checked,
            "mismatches": mismatches,
            "declared_delta_count": len(man.get("declared_deltas", {}))}


def g3_init_integrity(repo: Path) -> dict:
    import torch
    from src.models.stereonet import StereoNet, StereoNetConfig
    ck = repo / "checkpoints" / "armp_stage1_best.pth"
    rec: dict = {"checks": {}}
    got = sha256(ck)
    rec["checks"]["sha256_matches"] = got == EXPECTED_INIT_SHA
    rec["sha256"] = got
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    keys = sorted(blob.keys()) if isinstance(blob, dict) else ["<raw>"]
    forbidden = {"optimizer", "optimizer_state_dict", "scheduler", "lr_scheduler",
                 "scheduler_state_dict", "amp", "scaler", "epoch", "step"}
    leaked = sorted(set(keys) & forbidden)
    rec["blob_keys"] = keys
    rec["checks"]["no_optimizer_or_scheduler_state"] = not leaked
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    rec["checks"]["tensor_key_count"] = len(sd) == EXPECTED_KEYS
    rec["n_keys"] = len(sd)
    model = StereoNet(StereoNetConfig(**CFG))
    n_params = sum(p.numel() for p in model.parameters())
    rec["params"] = n_params
    rec["checks"]["param_count"] = n_params == EXPECTED_PARAMS
    missing, unexpected = model.load_state_dict(sd, strict=True)
    rec["checks"]["strict_missing_empty"] = list(missing) == []
    rec["checks"]["strict_unexpected_empty"] = list(unexpected) == []
    for k, v in CFG.items():
        rec["checks"][f"contract_{k}"] = getattr(model.config, k) == v
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    rec["checks"]["scheduler_epoch_0"] = sch.last_epoch == 0
    rec["checks"]["scheduler_lr_1e-3"] = abs(sch.get_last_lr()[0] - 1e-3) < 1e-12
    rec["pass"] = all(rec["checks"].values())
    return rec


def g4_contract(repo: Path) -> dict:
    import numpy as np
    from src.datasets.kitti2015 import Kitti2015Stereo
    calib = Kitti2015Stereo(repo / "data" / "kitti2015", split="hailo_calib")
    val = Kitti2015Stereo(repo / "data" / "kitti2015", split="hailo_val")
    s = val[0]
    total = 0
    for i in range(len(val)):
        d = val[i].disparity
        total += int((d > 0).sum())
    rec = {
        "hailo_calib_scenes": len(calib), "hailo_val_scenes": len(val),
        "calib_first": calib.names[0], "calib_last": calib.names[-1],
        "val_first": val.names[0], "val_last": val.names[-1],
        "overlap": len(set(calib.names) & set(val.names)),
        "image_dims": list(np.asarray(s.left).shape),
        "disparity_dims": list(np.asarray(s.disparity).shape),
        "disparity_scale": float(val.disparity_scale),
        "valid_pixels_counted": total, "valid_pixels_expected": EXPECTED_VALID_PX,
    }
    rec["pass"] = (rec["hailo_calib_scenes"] == 160 and rec["hailo_val_scenes"] == 40
                   and rec["overlap"] == 0
                   and total == EXPECTED_VALID_PX)
    return rec


def g5_one_batch(repo: Path) -> dict:
    """One real batch end to end. Throwaway: the checkpoint it writes is a
    reload test, not a Stage-E artefact."""
    import numpy as np
    import torch
    from phase1.harness.frozen_eval import strict_load_report
    from src.datasets.kitti2015 import Kitti2015Stereo, normalize
    from src.losses.disparity import masked_smooth_l1
    from src.models.stereonet import StereoNet, StereoNetConfig
    device = torch.device("cuda")
    ds = Kitti2015Stereo(repo / "data" / "kitti2015", split="hailo_calib")
    s = ds[0]
    left = torch.from_numpy(normalize(s.left)).to(device)
    right = torch.from_numpy(normalize(s.right)).to(device)
    # normalize() returns (1,3,H,W); disparity needs channel AND batch axes to
    # match the DataLoader-batched recipe (B,1,H,W). Stage-B's smoke used a
    # single [None] here and failed on the shape check.
    disp = torch.from_numpy(np.ascontiguousarray(
        s.disparity[None, None], dtype=np.float32)).to(device)
    cfg = StereoNetConfig(**CFG)
    model = StereoNet(cfg).to(device)
    model.train()
    blob = torch.load(repo / "checkpoints" / "armp_stage1_best.pth",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"], strict=True)
    out = model(left, right)
    loss, n = masked_smooth_l1(out, disp, max_disparity=float(cfg.max_disparity_px))
    loss.backward()
    grads_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    opt.step()
    tmp = Path("/kaggle/working/_g5_reload_test.pth")
    torch.save({"model": model.state_dict()}, tmp)
    reloaded = torch.load(tmp, map_location="cpu", weights_only=False)
    fresh = StereoNet(StereoNetConfig(**CFG))
    compat = strict_load_report(fresh, reloaded["model"])
    tmp.unlink(missing_ok=True)
    rec = {
        "sample": s.name, "pred_shape": list(out.shape), "disp_shape": list(disp.shape),
        "loss": float(loss), "valid_pixels_in_batch": int(n),
        "grads_present": bool(grads_ok),
        "reload_missing": compat["missing"], "reload_unexpected": compat["unexpected"],
        "params": int(compat["params"]),
        "throwaway": "reload-test checkpoint deleted; not a Stage-E artefact",
    }
    rec["pass"] = bool(grads_ok) and not compat["missing"] and not compat["unexpected"]
    return rec


def _synthetic_batch(batch: int, device):
    import torch
    g = torch.Generator(device="cpu").manual_seed(0)
    left = torch.randn(batch, 3, CROP_H, CROP_W, generator=g).to(device)
    right = torch.randn(batch, 3, CROP_H, CROP_W, generator=g).to(device)
    disp = (torch.rand(batch, 1, CROP_H, CROP_W, generator=g) * 100.0 + 1.0).to(device)
    return left, right, disp


def g6_memory(repo: Path) -> dict:
    """Batch-8 feasibility on T4. No optimizer step; forward+backward only."""
    import torch
    from src.losses.disparity import masked_smooth_l1
    from src.models.stereonet import StereoNet, StereoNetConfig
    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    results = []
    for batch in (2, 8):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = StereoNet(StereoNetConfig(**CFG)).to(device)
        model.train()
        blob = torch.load(repo / "checkpoints" / "armp_stage1_best.pth",
                          map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"], strict=True)
        left, right, disp = _synthetic_batch(batch, device)
        status, err = "FIT", None
        try:
            out = model(left, right)
            loss, _ = masked_smooth_l1(out, disp, max_disparity=184.0)
            loss.backward()
            torch.cuda.synchronize()
        except Exception as e:
            status = "OOM" if "out of memory" in str(e).lower() else "ERROR"
            err = str(e).splitlines()[0]
        results.append({
            "batch": batch, "status": status,
            "peak_allocated_MiB": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "peak_reserved_MiB": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            "error": err})
        del model, left, right, disp
        torch.cuda.empty_cache()
    b8 = next(r for r in results if r["batch"] == 8)
    headroom = (total - b8["peak_reserved_MiB"]) / total if b8["status"] == "FIT" else 0.0
    return {"total_memory_MiB": round(total, 1), "results": results,
            "batch8_fits": b8["status"] == "FIT",
            "headroom_frac": round(headroom, 3),
            "threshold_frac": MEM_HEADROOM_MIN,
            "verdict": ("NATIVE_BATCH_8" if b8["status"] == "FIT"
                        and headroom >= MEM_HEADROOM_MIN else "ACCUMULATION_FALLBACK"),
            "pass": b8["status"] in ("FIT", "OOM")}


def g7_rate(repo: Path, bundle: Path) -> dict:
    """Measure the T4 training rate so the budget is recalculated, not guessed.

    Uses the real recipe module (dataset, augmentation, loss, optimizer) and
    times RATE_STEPS optimizer steps. It does NOT complete an epoch, save a
    checkpoint or produce a record; the weights are discarded.
    """
    import torch
    from torch.utils.data import DataLoader

    scripts = bundle / "scripts"
    ck = repo / "checkpoints" / "armp_stage1_best.pth"
    # finetune_pilot.py parses CLI args at import time; give it a valid argv
    # instead of editing the file.
    saved_argv = sys.argv[:]
    sys.argv = ["finetune_pilot.py", "--arm", "armp", "--init", str(ck)]
    os.environ["P2A_SEED"] = "0"
    os.environ["P2A_OUT_DIR"] = "/kaggle/working/_rate_probe_discard"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "finetune_pilot", scripts / "finetune_pilot.py")
        fp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fp)
    finally:
        sys.argv = saved_argv

    from src.datasets.kitti2015 import Kitti2015Stereo
    from src.losses.disparity import masked_smooth_l1
    from src.models.stereonet import StereoNet, StereoNetConfig

    device = torch.device("cuda")
    train_base = Kitti2015Stereo(repo / "data" / "kitti2015", split="hailo_calib")
    val_base = Kitti2015Stereo(repo / "data" / "kitti2015", split="hailo_val")
    rec: dict = {"steps_timed_per_config": RATE_STEPS, "configs": {}}

    for label, batch in (("e0_batch2", 2), ("e2_batch8", 8)):
        train = fp.ScaledCroppedKitti(train_base, seed=0)
        loader = DataLoader(train, batch_size=batch, shuffle=True, num_workers=0,
                            generator=fp.make_generator(0),
                            worker_init_fn=fp.worker_init_fn)
        model = StereoNet(StereoNetConfig(**CFG)).to(device)
        model.train()
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"], strict=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        it = iter(loader)
        # two warmup steps, untimed
        for _ in range(2):
            left, right, d = next(it)
            left, right, d = left.to(device), right.to(device), d.to(device)
            loss, _ = masked_smooth_l1(model(left, right), d, max_disparity=184.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        t0 = time.time()
        steps = 0
        for _ in range(RATE_STEPS):
            try:
                left, right, d = next(it)
            except StopIteration:
                it = iter(loader)
                left, right, d = next(it)
            left, right, d = left.to(device), right.to(device), d.to(device)
            loss, _ = masked_smooth_l1(model(left, right), d, max_disparity=184.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
        torch.cuda.synchronize()
        step_s = (time.time() - t0) / max(steps, 1)
        steps_per_epoch = -(-len(train_base) // batch)   # ceil
        rec["configs"][label] = {
            "batch": batch, "s_per_step": round(step_s, 4),
            "steps_per_epoch": steps_per_epoch,
            "s_per_epoch_est": round(step_s * steps_per_epoch, 2),
        }
        del model, opt, loader, train
        torch.cuda.empty_cache()

    # One monitor pass: the recipe validates on 10 scenes every 5 epochs.
    model = StereoNet(StereoNetConfig(**CFG)).to(device)
    model.load_state_dict(torch.load(ck, map_location="cpu",
                                     weights_only=False)["model"], strict=True)
    torch.cuda.synchronize()
    t0 = time.time()
    fp.validate(model, val_base, device, limit=10)
    torch.cuda.synchronize()
    rec["monitor_pass_s"] = round(time.time() - t0, 2)
    del model
    torch.cuda.empty_cache()

    # Budget, recalculated from the measurements above.
    def total_h(epochs: int, cfg: str) -> float:
        per_epoch = rec["configs"][cfg]["s_per_epoch_est"]
        monitors = epochs // 5 + 1
        return (per_epoch * epochs + rec["monitor_pass_s"] * monitors) / 3600.0

    e0 = total_h(200, "e0_batch2")
    e2 = total_h(200, "e2_batch8")
    e3 = total_h(400, "e0_batch2")
    rec["budget_recalculated_h"] = {
        "per_seed": {"E0": round(e0, 2), "E1": round(e0, 2),
                     "E2": round(e2, 2), "E3": round(e3, 2)},
        "three_seeds": {"E0": round(e0 * 3, 2), "E1": round(e0 * 3, 2),
                        "E2": round(e2 * 3, 2), "E3": round(e3 * 3, 2)},
        "campaign_total_h": round((e0 * 3) * 2 + e2 * 3 + e3 * 3, 2),
        "basis": f"{RATE_STEPS} timed steps/config + 1 monitor pass, measured "
                 f"on this T4; E1 assumed equal to E0 (EMA adds a weight copy "
                 f"per step, not a forward/backward)",
    }
    rec["note"] = ("Rate probe only. No epoch completed, no checkpoint saved, "
                   "no record written; weights discarded.")
    rec["pass"] = True
    return rec


def main() -> None:
    t_all = time.time()
    boot, bundle = load_bootstrap()
    paths = boot.assemble()
    repo = Path(paths["repo"])
    sys.path.insert(0, str(repo))
    result["mounts"] = {k: str(v) for k, v in paths.items()}
    result["bundle_mount"] = str(bundle)

    for name, fn in (("G1_environment", lambda: g1_environment()),
                     ("G2_source_integrity", lambda: g2_source_integrity(bundle)),
                     ("G3_init_integrity", lambda: g3_init_integrity(repo)),
                     ("G4_contract", lambda: g4_contract(repo)),
                     ("G5_one_batch", lambda: g5_one_batch(repo)),
                     ("G6_memory", lambda: g6_memory(repo)),
                     ("G7_rate", lambda: g7_rate(repo, bundle))):
        t0 = time.time()
        try:
            rec = fn()
        except Exception as e:  # a failing gate must not hide the others
            rec = {"pass": False, "error": f"{type(e).__name__}: {e}"}
        rec["wall_s"] = round(time.time() - t0, 2)
        result["gates"][name] = rec
        print(f"{'PASS' if rec.get('pass') else 'FAIL'}  {name}  "
              f"({rec['wall_s']}s)" + (f"  {rec.get('error','')}" if not rec.get("pass") else ""),
              flush=True)

    result["all_pass"] = all(g.get("pass") for g in result["gates"].values())
    result["wall_s"] = round(time.time() - t_all, 2)
    result["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nALL GATES: {'PASS' if result['all_pass'] else 'FAIL'}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
