"""E4 PRETRAIN: FlyingThings3D_subset corpus, Stage-1 recipe byte-equivalent.

Line-for-line mirror of
stage_b_armp/20260918T062146Z_stage1_pretrain/scripts/train_armp_stage1.py
(FROZEN -- do not edit that file) with ONLY these differences:

  (i)   dataset class is FlyingThings3DSubsetTrain-equivalent
        (ft3d_subset.FlyingThings3DSubset, sign-fixed |d| loader) instead of
        ft3d.FlyingThings3DTrain;
  (ii)  run parameters are argparse arguments, not module constants:
        --manifest (required), --out-dir (required), --epochs (required, NO
        default -- the pre-registration deliberately commits no epoch count
        until the rate probe measures s/epoch), --max-steps (optional,
        default 0 = unlimited; the rate-probe mode, same code path truncated),
        --seed (optional, default 0);
  (iii) the final record JSON carries a `rate` block (s/step, s/epoch, step
        count, triplet count) so the probe run yields the spec's section-6
        gate number with no separate script;
  (iv)  probe mode (--max-steps > 0) writes the record under a probe-specific
        name and saves NO checkpoints (a probe is a measurement, not a
        pretrain; it must not overwrite a real run's artefacts).

Everything else is carried over UNCHANGED from Stage 1, because that
equivalence is the point (the corpus is the only lever):
augmentation math (s ~ logUniform(0.7, 1.7), crop 256x512), loss
(masked_smooth_l1), mask (gt > 0 AND gt < 184), BATCH = 2, LR = 1e-3,
cosine annealing with T_max = EPOCHS, the P2A config, SEED default 0,
the determinism helpers, the per-epoch JSONL log, best/final checkpoint
saving (real runs only), and verify_checkpoint_contract.

NO fine-tuning. NO KITTI training. NO hailo_val evaluation here.
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

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1.harness.determinism import make_generator, seed_all, worker_init_fn  # noqa: E402
from src.datasets.kitti2015 import normalize  # noqa: E402
from src.evaluation.metrics import disparity_metrics  # noqa: E402
from src.losses.disparity import masked_smooth_l1  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

from ft3d_subset import FlyingThings3DSubset  # noqa: E402

ARM = "e4_pretrain"
SEED = 0
CROP_H, CROP_W = 256, 512
BATCH = 2
LR = 1e-3

# the single intervention, carried over unchanged from P2A -- never swept
SCALE_LO, SCALE_HI = 0.7, 1.7

P2A_CONFIG = dict(downsample_levels=3, num_disparities=24,
                  cost_volume_shift="right", regression_normalize=True)
P2A_PARAMS = 397954
MAX_DISP = 184.0  # (24 - 1) * 8, P2A's representable ceiling
VAL_SCENES = 10  # first 10 pretrain-val triplets, mirror of P2A's 10-scene monitor

RECORD_NAME = "e4_pretrain_record.json"
PROBE_RECORD_NAME = "e4_pretrain_rate_probe.json"
LOG_NAME = "pretrain_log.jsonl"
BEST_CKPT_NAME = "e4_pretrain_best.pth"
FINAL_CKPT_NAME = "e4_pretrain_final.pth"


class ScaledCroppedFT3D(Dataset):
    """P2A's ScaledCroppedKitti with the base dataset swapped to FT3D.

    Identical augmentation math: s ~ logUniform(0.7, 1.7), w = round(512/s),
    h = round(256/s), uniform top-left, BILINEAR images / NEAREST disparity
    resize to 256x512, disparity[disparity > 0] *= sx with the REALISED
    sx = 512/w. Op order: crop -> resize -> ImageNet normalise -> gain jitter.
    (FT3D source frame is 540x960; both scale extremes fit, same defence.)
    """

    def __init__(self, base, seed=0, jitter=0.1):
        self.base = base
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)
        self.draws = []          # (s, h, w, sx) provenance, sampled for the record

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        smp = self.base[i]
        H, W = smp.disparity.shape
        s = float(np.exp(self.rng.uniform(np.log(SCALE_LO), np.log(SCALE_HI))))
        w = int(round(CROP_W / s))
        h = int(round(CROP_H / s))
        # defensive only: both extremes of the frozen range already fit
        # (s=0.7 -> 366x731, s=1.7 -> 151x301, source frame 540x960)
        h, w = min(h, H), min(w, W)
        y = int(self.rng.integers(0, H - h + 1))
        x = int(self.rng.integers(0, W - w + 1))
        sl = (slice(y, y + h), slice(x, x + w))

        left_c = smp.left[sl]
        right_c = smp.right[sl]
        disp_c = smp.disparity[sl]
        if (h, w) != (CROP_H, CROP_W):
            left_c = cv2.resize(left_c, (CROP_W, CROP_H), interpolation=cv2.INTER_LINEAR)
            right_c = cv2.resize(right_c, (CROP_W, CROP_H), interpolation=cv2.INTER_LINEAR)
            disp_c = cv2.resize(disp_c, (CROP_W, CROP_H), interpolation=cv2.INTER_NEAREST)
        sx = CROP_W / float(w)                       # REALISED horizontal factor
        disp_c = disp_c.copy()
        disp_c[disp_c > 0] *= sx

        left = normalize(left_c)[0]
        right = normalize(right_c)[0]
        if self.jitter:
            left = left * (1.0 + self.rng.normal(0, self.jitter))
            right = right * (1.0 + self.rng.normal(0, self.jitter))
        if len(self.draws) < 4000:
            self.draws.append((s, h, w, sx))
        return (
            torch.from_numpy(np.ascontiguousarray(left, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(right, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(disp_c[None], dtype=np.float32)),
        )


def validate(model, ds, device, limit=VAL_SCENES):
    """P2A's monitor, repointed: first N pretrain-val triplets, NO scaling.

    Full-resolution forward; valid = gt>0 AND gt<184 (training mask).
    Pretrain-convergence monitoring ONLY.
    """
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for i in range(min(limit, len(ds))):
            s = ds[i]
            out = model(
                torch.from_numpy(normalize(s.left)).to(device),
                torch.from_numpy(normalize(s.right)).to(device),
            )[0, 0].cpu().numpy()
            valid = (s.disparity > 0) & (s.disparity < MAX_DISP)
            preds.append(out[valid].astype(np.float64))
            gts.append(s.disparity[valid].astype(np.float64))
    model.train()
    return disparity_metrics(np.concatenate(preds), np.concatenate(gts))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def integrity_guard(model, config) -> dict:
    """Pre-training guard: EXACT P2A architecture. Raises before any step."""
    keys = sorted(model.state_dict().keys())
    params = sum(p.numel() for p in model.parameters())
    bn = [n for n, m in model.named_modules()
          if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm))]
    cfg_ok = all(getattr(config, k) == v for k, v in P2A_CONFIG.items())
    rep = {"param_count": params, "param_count_expected": P2A_PARAMS,
           "param_count_ok": params == P2A_PARAMS,
           "n_keys": len(keys), "n_keys_expected": 70,
           "n_keys_ok": len(keys) == 70,
           "batchnorm_modules": bn, "no_batchnorm": not bn,
           "config_matches_p2a": bool(cfg_ok),
           "config": {k: getattr(config, k) for k in P2A_CONFIG},
           "max_disparity_px": config.max_disparity_px,
           "feature_stride": config.feature_stride,
           "refinement_dilations": list(config.refinement_dilations)}
    rep["all_ok"] = bool(rep["param_count_ok"] and rep["n_keys_ok"]
                         and rep["no_batchnorm"] and rep["config_matches_p2a"]
                         and config.max_disparity_px == 184
                         and config.feature_stride == 8
                         and tuple(config.refinement_dilations) == (1, 2, 4, 8, 1, 1))
    return rep


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def build_scheduler(optimizer, epochs: int):
    """Cosine annealing to 0 with T_max following the requested epoch count.

    Small helper so the T_max-follows---epochs contract is directly testable
    without running any training (Stage 1 tied T_max to EPOCHS the same way).
    """
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="E4 pretrain (Stage-1 recipe, subset corpus).")
    ap.add_argument("--manifest", required=True,
                    help="path to the triplet manifest from enumerate_subset_triplets.py")
    ap.add_argument("--out-dir", required=True,
                    help="writable dir for checkpoints, the JSONL log and the record JSON")
    # NOTE: deliberately NO default -- the pre-registration commits no epoch
    # count until the rate probe has measured s/epoch.
    ap.add_argument("--epochs", required=True, type=int,
                    help="pretrain epoch count (no default by design)")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="rate-probe mode: stop after N optimizer steps (0 = unlimited)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    OUT_DIR = Path(args.out_dir)
    EPOCHS = int(args.epochs)
    SEED_RUN = int(args.seed)
    MAX_STEPS = int(args.max_steps)
    if EPOCHS <= 0:
        raise SystemExit("--epochs must be positive")
    if MAX_STEPS < 0:
        raise SystemExit("--max-steps must be >= 0")
    is_probe = MAX_STEPS > 0

    (OUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    seed_all(SEED_RUN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = Path(args.manifest)
    train_base = FlyingThings3DSubset(manifest_path, split="train")
    val_base = FlyingThings3DSubset(manifest_path, split="pretrain_val")
    assert not (set(train_base.ids) & set(val_base.ids)), "train/holdout overlap!"

    train = ScaledCroppedFT3D(train_base, seed=SEED_RUN)
    loader = DataLoader(
        train, batch_size=BATCH, shuffle=True, num_workers=0,
        generator=make_generator(SEED_RUN), worker_init_fn=worker_init_fn,
    )

    config = StereoNetConfig(**P2A_CONFIG)
    model = StereoNet(config).to(device)
    model.train()

    guard = integrity_guard(model, config)
    write_atomic(OUT_DIR / "integrity_guard.json", json.dumps(guard, indent=2))
    print("integrity guard:", json.dumps({k: v for k, v in guard.items()
                                          if k.endswith("_ok") or k == "param_count"}), flush=True)
    if not guard["all_ok"]:
        raise SystemExit("ABORT BEFORE TRAINING: integrity guard failed; see "
                         + str(OUT_DIR / "integrity_guard.json"))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = build_scheduler(optimizer, EPOCHS)

    scheduler_desc = "cosine annealing to 0 over {} epochs".format(EPOCHS)
    exp_config = {
        "experiment": "E4-PRETRAIN (FlyingThings3D_subset, Stage-1 recipe)",
        "arm": ("Stage-1 recipe mirrored line-for-line; ONLY the dataset is the "
                "FlyingThings3D_subset corpus (|d| sign fix) instead of TRAIN "
                "A+C; loss mask gt>0 and gt<184 (P2A's ceiling, same formula)."),
        "intervention": {
            "scale_distribution": "logUniform(0.7, 1.7), one draw per sample",
            "window": "w = round(512/s), h = round(256/s), uniform top-left",
            "image_resize": "cv2.INTER_LINEAR -> 256x512",
            "disparity_resize": "cv2.INTER_NEAREST -> 256x512 (sparse GT; 0 = no GT)",
            "disparity_scaling": "d[d>0] *= sx with sx = 512/w (realised, not sampled s)",
            "op_order": "crop -> resize -> ImageNet normalise -> gain jitter (P2A order)",
            "source_frame": "FT3D 540x960 frame",
        },
        "dataset": "flyingthings3d_subset TRAIN (official subset slice, NOT Stage-1 TRAIN A+C)",
        "split": "manifest-driven train / pretrain-val holdout (fixed random 5%, rng seed 0); pretrain monitoring ONLY",
        "split_enforcement": "train split only enters the optimizer loop; holdout is forward-only",
        "resolution": [CROP_H, CROP_W],
        "crop": "random scale-augmented {}x{}".format(CROP_H, CROP_W),
        "disparity_range": config.max_disparity_px,
        "batch_size": BATCH,
        "precision": "fp32",
        "seed": SEED_RUN,
        "epochs": EPOCHS,
        "optimizer": "Adam betas=(0.9, 0.999)",
        "learning_rate": LR,
        "schedule": scheduler_desc,
        "loss": "masked smooth L1, beta=1.0, valid = gt > 0 and gt < 184",
        "augmentation": ("scale-augmented random crop (the intervention), "
                         "independent per-image gain jitter sigma=0.1; no horizontal flip"),
        "initialisation": "PyTorch defaults, random",
        "device": str(device),
        "known_deviation": "no batch normalisation (BN-folded exported artifact), carried into all arms",
        "forbidden": "no fine-tuning, no KITTI training, no hailo_val evaluation in E4 pretrain",
    }

    log_path = OUT_DIR / LOG_NAME
    fh = open(log_path, "w", encoding="utf-8")
    history = []
    t0 = time.time()
    best_val_epe = float("inf")
    best_epoch = -1
    epoch_losses = []
    global_steps = 0
    truncated = False
    first_batch_valid_pixels = None
    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                  capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        git_head = None

    for epoch in range(EPOCHS):
        epoch_loss, epoch_pixels, batches = 0.0, 0, 0
        t_ep = time.time()
        for left, right, disparity in loader:
            left, right, disparity = left.to(device), right.to(device), disparity.to(device)
            out = model(left, right)
            loss, n = masked_smooth_l1(out, disparity, max_disparity=float(config.max_disparity_px))
            if not bool(torch.isfinite(loss).item()):
                fh.close()
                raise SystemExit("ABORT: NaN/inf loss at epoch %d" % epoch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_steps += 1
            if global_steps == 1:
                print("E4 first-batch valid pixels: %d" % n, flush=True)
                first_batch_valid_pixels = n
                if n == 0:
                    fh.close()
                    raise SystemExit("ABORT: zero valid pixels in first batch")
            epoch_loss += float(loss)
            epoch_pixels += n
            batches += 1
            if is_probe and global_steps >= MAX_STEPS:
                truncated = True
                break
        if truncated:
            # Rate-probe truncation: same code path as the real run, just cut
            # short mid-epoch. No validation, no checkpoint for the partial
            # epoch; fall through to the probe record below.
            break
        if epoch_pixels == 0:
            fh.close()
            raise SystemExit("ABORT: zero valid pixels in epoch %d" % epoch)
        scheduler.step()
        mean_loss = epoch_loss / max(batches, 1)
        epoch_losses.append(mean_loss)
        if epoch == 2 and min(epoch_losses[:3]) >= epoch_losses[0]:
            fh.close()
            raise SystemExit("ABORT: loss did not decrease at all over epochs 0-2: %s" % epoch_losses)
        m = validate(model, val_base, device, limit=VAL_SCENES)
        row = {"epoch": epoch, "mean_loss": mean_loss,
               "valid_pixels": epoch_pixels, "lr": float(scheduler.get_last_lr()[0]),
               "epoch_wall_s": time.time() - t_ep,
               "pretrain_val_epe": m.epe, "pretrain_val_d1": m.d1,
               "pretrain_val_pixels": m.valid_pixels,
               "nonfinite_sanitized": train_base.nonfinite_sanitized + val_base.nonfinite_sanitized}
        if m.epe < best_val_epe and not is_probe:
            best_val_epe = m.epe
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "config": exp_config},
                       OUT_DIR / "checkpoints" / BEST_CKPT_NAME)
        elif m.epe < best_val_epe:
            # Probe mode tracks the number but saves nothing: a probe is a
            # measurement, not a pretrain.
            best_val_epe = m.epe
            best_epoch = epoch
        history.append(row)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print("epoch {:>3} loss {:8.4f} lr {:.2e} pretrain-val EPE {:.3f} D1 {:.2f}% wall {:.0f}s".format(
            epoch, row["mean_loss"], row["lr"], row["pretrain_val_epe"],
            row["pretrain_val_d1"], row["epoch_wall_s"]), flush=True)

    wall_s = time.time() - t0
    fh.close()

    n_triplets = len(train_base) + len(val_base)
    epochs_run = len(history) if (is_probe and truncated) else EPOCHS
    rate = {
        "optimizer_steps": global_steps,
        "triplet_count": n_triplets,
        "n_train": len(train_base),
        "n_holdout": len(val_base),
        "wall_clock_s": wall_s,
        "sec_per_step": (wall_s / global_steps) if global_steps > 0 else None,
        "sec_per_epoch": (wall_s / epochs_run) if epochs_run > 0 else None,
        "probe": bool(is_probe),
        "max_steps": MAX_STEPS,
        "first_batch_valid_pixels": first_batch_valid_pixels,
        "truncated": bool(truncated),
    }

    if is_probe:
        # Probe mode: NO checkpoints, probe-specific record name only.
        draws = np.array(train.draws, dtype=np.float64) if train.draws else np.zeros((0, 4))
        record = {"arm": ARM, "experiment": "E4-PRETRAIN-RATE-PROBE",
                  "note": "FlyingThings3D_subset rate probe: same code path, truncated; NOT a pretrain",
                  "config": exp_config, "epochs_run": epochs_run,
                  "wall_clock_s": wall_s, "git_head": git_head,
                  "integrity_guard": guard,
                  "rate": rate,
                  "scale_draw_sample": {
                      "n": int(draws.shape[0]),
                      "s_min": float(draws[:, 0].min()) if draws.size else None,
                      "s_max": float(draws[:, 0].max()) if draws.size else None,
                      "s_mean": float(draws[:, 0].mean()) if draws.size else None,
                      "sx_min": float(draws[:, 3].min()) if draws.size else None,
                      "sx_max": float(draws[:, 3].max()) if draws.size else None,
                      "h_min": int(draws[:, 1].min()) if draws.size else None,
                      "h_max": int(draws[:, 1].max()) if draws.size else None,
                      "w_min": int(draws[:, 2].min()) if draws.size else None,
                      "w_max": int(draws[:, 2].max()) if draws.size else None},
                  "software": {"python": sys.version.split()[0], "torch": torch.__version__,
                               "numpy": np.__version__, "opencv": cv2.__version__,
                               "platform": platform.platform(),
                               "cuda": torch.version.cuda,
                               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
                  "history": history}
        write_atomic(OUT_DIR / PROBE_RECORD_NAME, json.dumps(record, indent=2))
        print("E4 RATE PROBE done: {} steps in {:.1f}s ({:.3f} s/step)".format(
            global_steps, wall_s,
            rate["sec_per_step"] if rate["sec_per_step"] else float("nan")), flush=True)
        return

    final_ckpt = OUT_DIR / "checkpoints" / FINAL_CKPT_NAME
    torch.save({"model": model.state_dict(), "config": exp_config}, final_ckpt)

    contract = verify_checkpoint_contract(final_ckpt, exp_config)
    write_atomic(OUT_DIR / "checkpoint_loading_contract_result.json",
                 json.dumps(contract, indent=2))

    draws = np.array(train.draws, dtype=np.float64) if train.draws else np.zeros((0, 4))
    record = {"arm": ARM, "experiment": "E4-PRETRAIN",
              "note": "FlyingThings3D_subset pretrain, Stage-1 recipe, NOT full SceneFlow",
              "config": exp_config, "epochs_run": EPOCHS,
              "wall_clock_s": wall_s, "git_head": git_head,
              "integrity_guard": guard,
              "rate": rate,
              "best_pretrain_val_epe": best_val_epe, "best_epoch": best_epoch,
              "final_sha256": sha256_file(final_ckpt),
              "best_sha256": sha256_file(OUT_DIR / "checkpoints" / BEST_CKPT_NAME),
              "checkpoint_contract": contract,
              "scale_draw_sample": {
                  "n": int(draws.shape[0]),
                  "s_min": float(draws[:, 0].min()) if draws.size else None,
                  "s_max": float(draws[:, 0].max()) if draws.size else None,
                  "s_mean": float(draws[:, 0].mean()) if draws.size else None,
                  "sx_min": float(draws[:, 3].min()) if draws.size else None,
                  "sx_max": float(draws[:, 3].max()) if draws.size else None,
                  "h_min": int(draws[:, 1].min()) if draws.size else None,
                  "h_max": int(draws[:, 1].max()) if draws.size else None,
                  "w_min": int(draws[:, 2].min()) if draws.size else None,
                  "w_max": int(draws[:, 2].max()) if draws.size else None},
              "software": {"python": sys.version.split()[0], "torch": torch.__version__,
                           "numpy": np.__version__, "opencv": cv2.__version__,
                           "platform": platform.platform(),
                           "cuda": torch.version.cuda,
                           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
              "history": history}
    write_atomic(OUT_DIR / RECORD_NAME, json.dumps(record, indent=2))
    print("E4 PRETRAIN done: wall {:.1f}s best pretrain-val EPE {:.4f} @epoch {}".format(
        wall_s, best_val_epe, best_epoch), flush=True)
    print("checkpoint contract: strict={} loaded={}/{}".format(
        contract["strict_load_result"], contract["loaded_parameter_count"],
        contract["total_parameter_count"]), flush=True)


def verify_checkpoint_contract(ckpt_path: Path, exp_config: dict) -> dict:
    """Load the produced checkpoint into a FRESH P2A model, strict=True."""
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    fresh = StereoNet(StereoNetConfig(**P2A_CONFIG))
    model_keys = set(fresh.state_dict().keys())
    ckpt_keys = set(state.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    shape_mismatches = [
        {"key": k, "checkpoint_shape": list(state[k].shape),
         "model_shape": list(fresh.state_dict()[k].shape)}
        for k in sorted(model_keys & ckpt_keys)
        if tuple(state[k].shape) != tuple(fresh.state_dict()[k].shape)
    ]
    ok = False
    try:
        fresh.load_state_dict(state, strict=True)
        ok = not missing and not unexpected and not shape_mismatches
    except Exception:
        ok = False
    total = sum(p.numel() for p in fresh.parameters())
    result = {
        "checkpoint_path": str(ckpt_path.resolve()),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "architecture_match": bool(
            P2A_CONFIG["downsample_levels"] == 3
            and P2A_CONFIG["num_disparities"] == 24
            and P2A_CONFIG["cost_volume_shift"] == "right"
            and P2A_CONFIG["regression_normalize"] is True),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "loaded_parameter_count": total if ok else 0,
        "total_parameter_count": total,
        "strict_load_result": bool(ok),
    }
    if not (ok and total == P2A_PARAMS and result["loaded_parameter_count"] == P2A_PARAMS):
        raise SystemExit("CHECKPOINT CONTRACT FAILED: " + json.dumps(
            {k: v for k, v in result.items() if k != "checkpoint_path"}))
    return result


if __name__ == "__main__":
    main()
