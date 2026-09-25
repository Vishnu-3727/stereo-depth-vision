"""EXP-TIER2-PILOT — P2A recipe mirror for the ARM-P Tier-2 pilot (PRE-FLIGHT GATE).

Line-for-line mirror of phase2/scripts/train_p2a_scale_coverage.py with EXACTLY
TWO added flags:

    --init  {random | <path to a .pth>}   weight initialisation source
    --arm   <name used for the output subdir>

Everything else — dataset manifest, split, crop/scale augmentation, loss,
optimizer, LR, scheduler, epochs, batch, seed handling, AMP/device (fp32),
gradient settings, checkpoint-selection rule — is unchanged from P2A.

    ARM-CONTROL : --init random --arm arm_control
    ARM-P       : --init <armp_stage1_best.pth> --arm arm_p

Run (real pilot, NOT part of the pre-flight gate):
    P2A_SEED=0 P2A_OUT_DIR=<pilot>/runs/<arm> python <this file> \
        --init <random|ckpt> --arm <arm>

Env: P2A_SEED (0), P2A_OUT_DIR, P2A_EPOCHS (smoke only) — same as P2A.
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

from phase1.harness.determinism import make_generator, seed_all, worker_init_fn  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.evaluation.metrics import disparity_metrics  # noqa: E402
from src.losses.disparity import masked_smooth_l1  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

# DELTA-1 vs P2A: two CLI flags only. Defaults reproduce P2A exactly
# (--arm default keeps the P2A arm name; --init default keeps random init).
_parser = argparse.ArgumentParser(description="Tier-2 pilot: P2A recipe mirror.")
_parser.add_argument("--init", default="random",
                     help="'random' or path to a .pth checkpoint to load strictly")
_parser.add_argument("--arm", default="p2a_scale_coverage",
                     help="arm name, used for the output subdir")
_ARGS, _ = _parser.parse_known_args()

ARM = _ARGS.arm
INIT = _ARGS.init
SEED = int(os.environ.get("P2A_SEED", "0"))
DEFAULT_DIR = REPO_ROOT / "phase2" / "runs" / (ARM if SEED == 0 else ARM + "_s%d" % SEED)
OUT_DIR = Path(os.environ.get("P2A_OUT_DIR", str(DEFAULT_DIR)))
CROP_H, CROP_W = 256, 512
EPOCHS = int(os.environ.get("P2A_EPOCHS", "40"))
BATCH = 2
LR = 1e-4
SCHEDULER = "cosine annealing to 0 over 40 epochs"

# the single intervention, frozen by preregistration -- never swept
SCALE_LO, SCALE_HI = 0.7, 1.7

ARM_V_CONFIG = dict(downsample_levels=3, num_disparities=24,
                    cost_volume_shift="right", regression_normalize=True)
ARM_V_PARAMS = 397954


class ScaledCroppedKitti(Dataset):
    """ARM-V's CroppedKitti with the preregistered scale augmentation."""

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
        # (s=0.7 -> 366x731, s=1.7 -> 151x301, source frame 368x1232)
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


def validate(model, ds, device, limit=10):
    """ARM-V's monitor, unchanged: first 10 hailo_val scenes, NO scaling."""
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for i in range(min(limit, len(ds))):
            s = ds[i]
            out = model(
                torch.from_numpy(normalize(s.left)).to(device),
                torch.from_numpy(normalize(s.right)).to(device),
            )[0, 0].cpu().numpy()
            valid = s.disparity > 0
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
    """Preregistered pre-training guard. Raises before any optimizer step."""
    ref_blob = torch.load(REPO_ROOT / "phase1" / "runs" / "arm_v" / "arm_v_best.pth",
                          map_location="cpu", weights_only=False)
    ref_keys = sorted(ref_blob["model"].keys())
    keys = sorted(model.state_dict().keys())
    params = sum(p.numel() for p in model.parameters())
    bn = [n for n, m in model.named_modules()
          if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm))]
    cfg_ok = all(getattr(config, k) == v for k, v in ARM_V_CONFIG.items())
    shapes_ok = all(tuple(model.state_dict()[k].shape) == tuple(ref_blob["model"][k].shape)
                    for k in keys) if keys == ref_keys else False
    rep = {"param_count": params, "param_count_expected": ARM_V_PARAMS,
           "param_count_ok": params == ARM_V_PARAMS,
           "state_dict_keys_match_arm_v": keys == ref_keys,
           "state_dict_shapes_match_arm_v": bool(shapes_ok),
           "n_keys": len(keys), "batchnorm_modules": bn, "no_batchnorm": not bn,
           "config_matches_arm_v": bool(cfg_ok),
           "config": {k: getattr(config, k) for k in ARM_V_CONFIG},
           "max_disparity_px": config.max_disparity_px,
           "feature_stride": config.feature_stride,
           "eval_contract": {"scenes": 40, "valid_pixels": 3802797,
                             "gt_source": "disp_occ_0", "gt_scale": 256.0,
                             "crop": "368x1232 top-left", "split": "hailo_val",
                             "unchanged": True}}
    rep["all_ok"] = bool(rep["param_count_ok"] and rep["state_dict_keys_match_arm_v"]
                         and rep["state_dict_shapes_match_arm_v"] and rep["no_batchnorm"]
                         and rep["config_matches_arm_v"]
                         and config.max_disparity_px == 184)
    return rep


def apply_init(model, init: str) -> dict:
    """DELTA-2 vs P2A: weight initialisation source. 'random' keeps P2A's
    PyTorch-default random init (no-op); otherwise strict-load the checkpoint's
    ['model'] state dict into the P2A architecture. Returns a provenance record."""
    if init == "random":
        return {"source": "random", "note": "PyTorch defaults, random (P2A recipe)"}
    ckpt = Path(init)
    if not ckpt.is_file():
        raise SystemExit("ABORT: --init checkpoint not found: " + str(ckpt))
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    # strict: any missing/unexpected key aborts before training
    model.load_state_dict(sd, strict=True)
    return {"source": str(ckpt), "strict": True,
            "sha256": sha256_file(ckpt),
            "n_keys": len(sd),
            "param_count": sum(p.numel() for p in model.parameters())}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_calib")
    val_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val")
    assert len(train_base) == 160 and len(val_base) == 40
    assert train_base.names[0] == "000000_10.png" and train_base.names[-1] == "000159_10.png"
    assert val_base.names[0] == "000160_10.png" and val_base.names[-1] == "000199_10.png"
    assert not (set(train_base.names) & set(val_base.names)), "train/eval overlap!"

    train = ScaledCroppedKitti(train_base, seed=SEED)
    loader = DataLoader(
        train, batch_size=BATCH, shuffle=True, num_workers=0,
        generator=make_generator(SEED), worker_init_fn=worker_init_fn,
    )

    config = StereoNetConfig(**ARM_V_CONFIG)
    model = StereoNet(config).to(device)
    model.train()

    guard = integrity_guard(model, config)
    (OUT_DIR / "integrity_guard.json").write_text(json.dumps(guard, indent=2))
    print("integrity guard:", json.dumps({k: v for k, v in guard.items()
                                          if k.endswith("_ok") or k == "param_count"}), flush=True)
    if not guard["all_ok"]:
        raise SystemExit("ABORT BEFORE TRAINING: integrity guard failed; see "
                         + str(OUT_DIR / "integrity_guard.json"))

    # DELTA-2 applied here: init source only; no recipe change.
    # NOTE: integrity_guard above runs on the randomly initialised model (as in
    # P2A) so architecture identity is established before weights are replaced.
    init_record = apply_init(model, INIT)
    print("init:", json.dumps(init_record), flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    exp_config = {
        "experiment": "EXP-P2A-SCALE-COVERAGE-001",
        "arm": ("P2A: ARM-V recipe + training-time geometric scale augmentation "
                "s ~ logUniform(0.7, 1.7); disparity scaled by the REALISED "
                "horizontal factor sx = 512/round(512/s); bilinear images, "
                "NEAREST disparity; everything else ARM-V"),
        "intervention": {
            "scale_distribution": "logUniform(0.7, 1.7), one draw per sample",
            "window": "w = round(512/s), h = round(256/s), uniform top-left",
            "image_resize": "cv2.INTER_LINEAR -> 256x512",
            "disparity_resize": "cv2.INTER_NEAREST -> 256x512 (sparse GT; 0 = no GT)",
            "disparity_scaling": "d[d>0] *= sx with sx = 512/w (realised, not sampled s)",
            "op_order": "crop -> resize -> ImageNet normalise -> gain jitter (ARM-V order)",
            "source_frame": "the dataset's pad_and_crop 368x1232 frame, the same "
                            "source ARM-V crops from (NOT the raw 375x1242)",
        },
        "dataset": "kitti2015",
        "split": "hailo_calib (scenes 0-159) train, hailo_val (160-199) eval; overlap 0",
        "split_enforcement": "train_base split='hailo_calib' only; no hailo_val sample enters optimizer loop",
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
        "loss": "masked smooth L1, beta=1.0, valid = gt > 0 and gt < max_disparity",
        "augmentation": ("scale-augmented random crop (the intervention), "
                         "independent per-image gain jitter sigma=0.1; no horizontal flip"),
        "initialisation": ("PyTorch defaults, random (P2A recipe)" if INIT == "random"
                           else "strict load of " + str(INIT)),
        "init_record": init_record,
        "device": str(device),
        "known_deviation": "no batch normalisation (BN-folded exported artifact), carried into all arms",
    }

    log_path = OUT_DIR / "training_log.jsonl"
    fh = open(log_path, "w", encoding="utf-8")
    history = []
    t0 = time.time()
    best_val_epe = float("inf")
    best_epoch = -1
    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                  capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        git_head = None

    for epoch in range(EPOCHS):
        epoch_loss, epoch_pixels, batches = 0.0, 0, 0
        for left, right, disparity in loader:
            left, right, disparity = left.to(device), right.to(device), disparity.to(device)
            out = model(left, right)
            loss, n = masked_smooth_l1(out, disparity, max_disparity=float(config.max_disparity_px))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            epoch_pixels += n
            batches += 1
        scheduler.step()
        row = {"epoch": epoch, "mean_loss": epoch_loss / max(batches, 1),
               "valid_pixels": epoch_pixels, "lr": float(scheduler.get_last_lr()[0])}
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            m = validate(model, val_base, device, limit=10)
            row["val_epe"] = m.epe
            row["val_d1"] = m.d1
            if m.epe < best_val_epe:
                best_val_epe = m.epe
                best_epoch = epoch
                torch.save({"model": model.state_dict(), "config": exp_config},
                           OUT_DIR / "p2a_best.pth")
        history.append(row)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print("epoch {:>3} loss {:8.4f} lr {:.2e}{}".format(
            epoch, row["mean_loss"], row["lr"],
            " val EPE {:.3f} D1 {:.2f}%".format(row["val_epe"], row["val_d1"])
            if "val_epe" in row else ""), flush=True)

    wall_s = time.time() - t0
    fh.close()
    final_ckpt = OUT_DIR / "p2a_final.pth"
    torch.save({"model": model.state_dict(), "config": exp_config}, final_ckpt)

    draws = np.array(train.draws, dtype=np.float64) if train.draws else np.zeros((0, 4))
    record = {"arm": ARM, "experiment": "EXP-P2A-SCALE-COVERAGE-001",
              "config": exp_config, "epochs_run": EPOCHS,
              "wall_clock_s": wall_s, "git_head": git_head,
              "integrity_guard": guard,
              "best_val_epe_10scene": best_val_epe, "best_epoch": best_epoch,
              "final_sha256": sha256_file(final_ckpt),
              "best_sha256": sha256_file(OUT_DIR / "p2a_best.pth"),
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
    (OUT_DIR / "p2a_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("P2A seed {} done: wall {:.1f}s best10 EPE {:.4f} @epoch {}".format(
        SEED, wall_s, best_val_epe, best_epoch))


if __name__ == "__main__":
    main()
