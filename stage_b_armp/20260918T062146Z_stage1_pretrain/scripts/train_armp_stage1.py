"""ARM P STAGE 1 of 2: PRETRAIN ONLY (FT3D A+C subset, NOT full SceneFlow).

Mirror of phase2/scripts/train_p2a_scale_coverage.py with ONLY these
differences (see pretrain_config.json for the field-by-field diff):
  (i)  dataset is FlyingThings3D TRAIN (A+C staged on C:, 14,460 usable
       triplets; B wholly missing, Monkaa empty) instead of KITTI;
  (ii) loss valid mask gt>0 AND gt<184 -- the same formula P2A uses
       (max_disparity_px = 184), restated with the numeric ceiling;
  (iii) EPOCHS = 20 (not 200); CosineAnnealingLR T_max follows EPOCHS.
Plus monitoring-only consequences of the brief: a fixed random 5% FT3D
holdout (seed 0) is evaluated every epoch for pretrain Log purposes
(P2A evaluated 10 KITTI scenes every 5 epochs); NOTHING from hailo_val
or any KITTI data enters this script.

NO fine-tuning. NO KITTI training. NO hailo_val evaluation here.
New file; train_p2a_scale_coverage.py and every other repo file untouched.

Run (detached): launched via launch_stage1.ps1, stdout tee'd to a log.
"""
from __future__ import annotations

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

from ft3d import FlyingThings3DTrain  # noqa: E402

ARM = "armp_stage1_pretrain"
SEED = 0
OUT_DIR = Path(__file__).resolve().parents[1]
CROP_H, CROP_W = 256, 512
EPOCHS = 20
BATCH = 2
LR = 1e-3
SCHEDULER = "cosine annealing to 0 over 20 epochs"

# the single intervention, carried over unchanged from P2A -- never swept
SCALE_LO, SCALE_HI = 0.7, 1.7

P2A_CONFIG = dict(downsample_levels=3, num_disparities=24,
                  cost_volume_shift="right", regression_normalize=True)
P2A_PARAMS = 397954
MAX_DISP = 184.0  # (24 - 1) * 8, P2A's representable ceiling
VAL_SCENES = 10  # first 10 pretrain-val triplets, mirror of P2A's 10-scene monitor


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


def main() -> None:
    (OUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = OUT_DIR / "dataset_manifest.json"
    train_base = FlyingThings3DTrain(manifest_path, split="train")
    val_base = FlyingThings3DTrain(manifest_path, split="pretrain_val")
    assert len(train_base) == 13737 and len(val_base) == 723, (len(train_base), len(val_base))
    assert not (set(train_base.ids) & set(val_base.ids)), "train/holdout overlap!"

    train = ScaledCroppedFT3D(train_base, seed=SEED)
    loader = DataLoader(
        train, batch_size=BATCH, shuffle=True, num_workers=0,
        generator=make_generator(SEED), worker_init_fn=worker_init_fn,
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    exp_config = {
        "experiment": "ARM-P-STAGE1-PRETRAIN (FT3D A+C subset, NOT full SceneFlow)",
        "arm": ("P2A recipe mirrored line-for-line; ONLY the dataset is FT3D "
                "TRAIN A+C instead of KITTI; loss mask gt>0 and gt<184 "
                "(P2A's ceiling, same formula); EPOCHS=20."),
        "intervention": {
            "scale_distribution": "logUniform(0.7, 1.7), one draw per sample",
            "window": "w = round(512/s), h = round(256/s), uniform top-left",
            "image_resize": "cv2.INTER_LINEAR -> 256x512",
            "disparity_resize": "cv2.INTER_NEAREST -> 256x512 (sparse GT; 0 = no GT)",
            "disparity_scaling": "d[d>0] *= sx with sx = 512/w (realised, not sampled s)",
            "op_order": "crop -> resize -> ImageNet normalise -> gain jitter (P2A order)",
            "source_frame": "FT3D 540x960 frame",
        },
        "dataset": "flyingthings3d TRAIN A+C subset staged on C: (NOT full SceneFlow; B missing, Monkaa empty)",
        "split": "13737 train / 723 pretrain-val holdout (fixed random 5%, rng seed 0); pretrain monitoring ONLY",
        "split_enforcement": "train split only enters the optimizer loop; holdout is forward-only",
        "resolution": [CROP_H, CROP_W],
        "crop": "random scale-augmented {}x{}".format(CROP_H, CROP_W),
        "disparity_range": config.max_disparity_px,
        "batch_size": BATCH,
        "precision": "fp32",
        "seed": SEED,
        "epochs": EPOCHS,
        "optimizer": "Adam betas=(0.9, 0.999)",
        "learning_rate": LR,
        "schedule": SCHEDULER,
        "loss": "masked smooth L1, beta=1.0, valid = gt > 0 and gt < 184",
        "augmentation": ("scale-augmented random crop (the intervention), "
                         "independent per-image gain jitter sigma=0.1; no horizontal flip"),
        "initialisation": "PyTorch defaults, random",
        "device": str(device),
        "known_deviation": "no batch normalisation (BN-folded exported artifact), carried into all arms",
        "forbidden": "no fine-tuning, no KITTI training, no hailo_val evaluation in Stage 1",
    }

    log_path = OUT_DIR / "pretrain_log.jsonl"
    fh = open(log_path, "w", encoding="utf-8")
    history = []
    t0 = time.time()
    best_val_epe = float("inf")
    best_epoch = -1
    epoch_losses = []
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
            epoch_loss += float(loss)
            epoch_pixels += n
            batches += 1
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
        if m.epe < best_val_epe:
            best_val_epe = m.epe
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "config": exp_config},
                       OUT_DIR / "checkpoints" / "armp_stage1_best.pth")
        history.append(row)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print("epoch {:>3} loss {:8.4f} lr {:.2e} pretrain-val EPE {:.3f} D1 {:.2f}% wall {:.0f}s".format(
            epoch, row["mean_loss"], row["lr"], row["pretrain_val_epe"],
            row["pretrain_val_d1"], row["epoch_wall_s"]), flush=True)

    wall_s = time.time() - t0
    fh.close()
    final_ckpt = OUT_DIR / "checkpoints" / "armp_stage1_final.pth"
    torch.save({"model": model.state_dict(), "config": exp_config}, final_ckpt)

    contract = verify_checkpoint_contract(final_ckpt, exp_config)
    write_atomic(OUT_DIR / "checkpoint_loading_contract_result.json",
                 json.dumps(contract, indent=2))

    draws = np.array(train.draws, dtype=np.float64) if train.draws else np.zeros((0, 4))
    record = {"arm": ARM, "experiment": "ARM-P-STAGE1-PRETRAIN",
              "note": "FT3D A+C subset pretrain, NOT full SceneFlow",
              "config": exp_config, "epochs_run": EPOCHS,
              "wall_clock_s": wall_s, "git_head": git_head,
              "integrity_guard": guard,
              "best_pretrain_val_epe": best_val_epe, "best_epoch": best_epoch,
              "final_sha256": sha256_file(final_ckpt),
              "best_sha256": sha256_file(OUT_DIR / "checkpoints" / "armp_stage1_best.pth"),
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
    write_atomic(OUT_DIR / "stage1_record.json", json.dumps(record, indent=2))
    print("ARM-P STAGE1 done: wall {:.1f}s best pretrain-val EPE {:.4f} @epoch {}".format(
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
