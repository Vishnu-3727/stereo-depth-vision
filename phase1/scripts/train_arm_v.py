"""ARM V — 200-epoch KITTI from random init with finer disparity sampling.

Identical to ARM U (phase1/scripts/train_arm_u.py) EXCEPT the matching
geometry: StereoNetConfig(downsample_levels=3, num_disparities=24,
cost_volume_shift="right", regression_normalize=True). Stride halves (1/16 ->
1/8) and candidate count doubles (12 -> 24) TOGETHER so the disparity RANGE
stays about constant (176 -> 184 px) and only the SAMPLING DENSITY changes
(16 px spacing -> 8 px spacing). Changing one without the other would alter
the range instead, which is a different experiment. Seed 0. Everything else
frozen. Trains ONLY on hailo_calib (scenes 0-159); hailo_val (160-199) used
for curve monitoring (first 10) + final frozen scoring (full 40, via
frozen_eval.py — separate step, no early stopping).

Split enforcement in code: train_base built with split="hailo_calib" only;
no hailo_val sample ever enters the optimizer loop; checkpoint selection uses
the train-time 10-scene curve + final-epoch weights, never the 40-scene score.

Single-line shell rule: run as `python phase1\scripts\train_arm_v.py`.
Smoke test: set ARM_V_EPOCHS=2 and ARM_V_OUT_DIR=<scratch> in the environment;
real run: unset both (defaults: 200 epochs, phase1/runs/arm_v).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from phase1.harness.determinism import make_generator, seed_all, worker_init_fn  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.evaluation.metrics import disparity_metrics  # noqa: E402
from src.losses.disparity import masked_smooth_l1  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

ARM = "arm_v"
OUT_DIR = Path(os.environ.get("ARM_V_OUT_DIR", str(REPO_ROOT / "phase1" / "runs" / ARM)))
CROP_H, CROP_W = 256, 512
EPOCHS = int(os.environ.get("ARM_V_EPOCHS", "200"))
BATCH = 2
LR = 1e-3
SEED = 0
SCHEDULER = "cosine annealing to 0 over 200 epochs"


class CroppedKitti(Dataset):
    """Same as exp_train_convergence.CroppedKitti: random crop + gain jitter."""

    def __init__(self, base, seed=0, jitter=0.1):
        self.base = base
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s = self.base[i]
        h, w = s.disparity.shape
        y = int(self.rng.integers(0, h - CROP_H + 1))
        x = int(self.rng.integers(0, w - CROP_W + 1))
        sl = (slice(y, y + CROP_H), slice(x, x + CROP_W))
        left = normalize(s.left[sl])[0]
        right = normalize(s.right[sl])[0]
        if self.jitter:
            left = left * (1.0 + self.rng.normal(0, self.jitter))
            right = right * (1.0 + self.rng.normal(0, self.jitter))
        disparity = s.disparity[sl][None]
        return (
            torch.from_numpy(np.ascontiguousarray(left, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(right, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(disparity, dtype=np.float32)),
        )


def validate(model, ds, device, limit=10):
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)  # python + numpy + torch CPU/CUDA seeds (superset of baseline)
    # NOTE: torch.use_deterministic_algorithms NOT enabled — baseline recipe did
    # not use it and bilinear-backward has no deterministic CUDA impl (would
    # crash training). Eval is forward-only, unaffected. Recorded as deviation: none.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_calib")
    val_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val")
    assert len(train_base) == 160 and len(val_base) == 40
    assert train_base.names[0] == "000000_10.png" and train_base.names[-1] == "000159_10.png"
    assert val_base.names[0] == "000160_10.png" and val_base.names[-1] == "000199_10.png"
    assert not (set(train_base.names) & set(val_base.names)), "train/eval overlap!"

    train = CroppedKitti(train_base, seed=SEED)
    loader = DataLoader(
        train, batch_size=BATCH, shuffle=True, num_workers=0,
        generator=make_generator(SEED), worker_init_fn=worker_init_fn,
    )

    config = StereoNetConfig(downsample_levels=3, num_disparities=24,
                             cost_volume_shift="right", regression_normalize=True)
    model = StereoNet(config).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    exp_config = {
        "arm": "ARM V: random-init + KITTI only, 200 epochs, downsample_levels=3 + num_disparities=24 (stride 1/8, 8px spacing, range 184px) + cost_volume_shift=right + regression_normalize=True",
        "dataset": "kitti2015",
        "split": "hailo_calib (scenes 0-159) train, hailo_val (160-199) eval; overlap 0",
        "split_enforcement": "train_base split='hailo_calib' only; no hailo_val sample enters optimizer loop",
        "resolution": [CROP_H, CROP_W],
        "crop": "random {}x{}".format(CROP_H, CROP_W),
        "disparity_range": config.max_disparity_px,
        "batch_size": BATCH,
        "precision": "fp32",
        "seed": SEED,
        "epochs": EPOCHS,
        "optimizer": "Adam betas=(0.9, 0.999)",
        "learning_rate": LR,
        "schedule": SCHEDULER,
        "loss": "masked smooth L1, beta=1.0, valid = gt > 0 and gt < max_disparity",
        "augmentation": "random crop, independent per-image gain jitter sigma=0.1; no horizontal flip",
        "initialisation": "PyTorch defaults, random",
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
                            OUT_DIR / "arm_v_best.pth")
        history.append(row)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print("epoch {:>3} loss {:8.4f} lr {:.2e}{}".format(
            epoch, row["mean_loss"], row["lr"],
            " val EPE {:.3f} D1 {:.2f}%".format(row["val_epe"], row["val_d1"])
            if "val_epe" in row else ""), flush=True)

    wall_s = time.time() - t0
    fh.close()
    final_ckpt = OUT_DIR / "arm_v_final.pth"
    torch.save({"model": model.state_dict(), "config": exp_config}, final_ckpt)
    record = {"arm": ARM, "config": exp_config, "epochs_run": EPOCHS,
              "wall_clock_s": wall_s, "git_head": git_head,
              "best_val_epe_10scene": best_val_epe, "best_epoch": best_epoch,
              "final_sha256": sha256_file(final_ckpt),
               "best_sha256": sha256_file(OUT_DIR / "arm_v_best.pth"),
              "history": history}
    (OUT_DIR / "arm_v_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("ARM V done: wall {:.1f}s best10 EPE {:.4f} @epoch {}".format(wall_s, best_val_epe, best_epoch))


if __name__ == "__main__":
    main()
