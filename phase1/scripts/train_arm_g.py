"""ARM G — ARM K plus asymmetric photometric augmentation. One variable.

Identical to phase1/scripts/train_arm_k.py (200-epoch cosine recipe, itself
identical to the 20-epoch recipe behind convergence_run.pth
(scripts/exp_train_convergence.py)) EXCEPT an asymmetric photometric
augmentation applied per sample, independently to the left and right image.
One variable: photometric robustness.

Photometric aug (fixed order, in [0,1] space BEFORE ImageNet normalization,
clamped to [0,1] after each step; geometry untouched):
  brightness  multiply by U(0.8, 1.2)
  contrast    (x - mean) * U(0.8, 1.2) + mean   (mean = scalar mean of the crop)
  gamma       x ** U(0.8, 1.2)

The existing gain jitter sigma=0.1 STAYS as it is in ARM K (frozen recipe);
this augmentation is added ON TOP and is NOT a gain-jitter replacement.
No flip, no scale, no rotation, no crop change. The disparity map and the
valid mask are NOT modified in any way.

Seed 0. Architecture frozen. Trains ONLY on hailo_calib (scenes 0-159);
hailo_val (160-199) used for curve monitoring (first 10) + final frozen
scoring (full 40, via frozen_eval.py — separate step, no early stopping).

Split enforcement in code: train_base built with split="hailo_calib" only;
no hailo_val sample ever enters the optimizer loop; checkpoint selection uses
the train-time 10-scene curve + final-epoch weights, never the 40-scene score.

Single-line shell rule: run as `python phase1\\scripts\\train_arm_g.py`.
Run length / augmentation rate may be overridden: `--epochs N` (default 200),
`--photo-prob P` (default 1.0, i.e. always on). `--self-check` runs the
pre-training assertions and exits without training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

ARM = "arm_g"
OUT_DIR = REPO_ROOT / "phase1" / "runs" / ARM
CROP_H, CROP_W = 256, 512
EPOCHS = 200
BATCH = 2
LR = 1e-3
SEED = 0
JITTER = 0.1
PHOTO_PROB = 1.0
SCHEDULER = "cosine annealing to 0 over 200 epochs"


def photo_augment_crop(crop_u8: np.ndarray, rng: np.random.Generator):
    """Apply brightness->contrast->gamma to a uint8 HxWx3 crop.

    Works in [0,1] space, clamps to [0,1] after each step, returns uint8-range
    (0-255) float32 ready for normalize(). Returns (img255, params).
    """
    x01 = crop_u8.astype(np.float32) / 255.0
    b = float(rng.uniform(0.8, 1.2))
    c = float(rng.uniform(0.8, 1.2))
    g = float(rng.uniform(0.8, 1.2))
    x = np.clip(x01 * b, 0.0, 1.0)
    x = np.clip((x - float(x.mean())) * c + float(x.mean()), 0.0, 1.0)
    x = np.clip(np.power(x, g), 0.0, 1.0)
    return (x * 255.0).astype(np.float32), (b, c, g)


class CroppedKitti(Dataset):
    """ARM K dataset + asymmetric photometric aug, gain jitter UNCHANGED.

    Pipeline per sample (seed 0, random 256x512 crop):
      1. random crop (same RNG sequence as ARM K)
      2. photometric aug in [0,1] BEFORE normalization, left/right independent
         draws (skipped entirely — zero RNG draws — when photo_prob <= 0.0,
         which makes photo_prob=0.0 bit-identical to ARM K)
      3. ImageNet normalize
      4. per-image gain jitter sigma=0.1 (same RNG call order as ARM K)
    Disparity is cropped only, never modified.
    """

    def __init__(self, base, seed=0, jitter=JITTER, photo_prob=PHOTO_PROB,
                 return_params=False):
        self.base = base
        self.jitter = jitter
        self.photo_prob = photo_prob
        self.return_params = return_params
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def _maybe_photo(self, crop_u8):
        if self.photo_prob <= 0.0:
            return crop_u8.astype(np.float32), None
        if self.photo_prob < 1.0:
            if float(self.rng.uniform()) >= self.photo_prob:
                return crop_u8.astype(np.float32), None
        return photo_augment_crop(crop_u8, self.rng)

    def __getitem__(self, i):
        s = self.base[i]
        h, w = s.disparity.shape
        y = int(self.rng.integers(0, h - CROP_H + 1))
        x = int(self.rng.integers(0, w - CROP_W + 1))
        sl = (slice(y, y + CROP_H), slice(x, x + CROP_W))
        left255, lp = self._maybe_photo(s.left[sl])
        right255, rp = self._maybe_photo(s.right[sl])
        left = normalize(left255)[0]
        right = normalize(right255)[0]
        if self.jitter:
            left = left * (1.0 + self.rng.normal(0, self.jitter))
            right = right * (1.0 + self.rng.normal(0, self.jitter))
        disparity = s.disparity[sl][None]
        lt = torch.from_numpy(np.ascontiguousarray(left, dtype=np.float32))
        rt = torch.from_numpy(np.ascontiguousarray(right, dtype=np.float32))
        dt = torch.from_numpy(np.ascontiguousarray(disparity, dtype=np.float32))
        if self.return_params:
            return lt, rt, dt, (lp, rp)
        return lt, rt, dt


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


def self_check(n_samples: int = 20) -> None:
    """Pre-training assertions. Raises on any failure; prints a report."""
    sys.path.insert(0, str(REPO_ROOT / "phase1" / "scripts"))
    from train_arm_k import CroppedKitti as CroppedKittiK  # noqa: E402

    train_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_calib")
    assert len(train_base) == 160

    photo = CroppedKitti(train_base, seed=SEED, photo_prob=1.0, return_params=True)
    plain = CroppedKitti(train_base, seed=SEED, photo_prob=0.0)
    armk = CroppedKittiK(train_base, seed=SEED)

    # 1. disparity + valid mask bit-identical before/after augmentation.
    # NOTE: photo_prob=1.0 consumes extra RNG draws (photo params) vs 0.0, so
    # the two datasets' RNG states diverge after one sample. Reset both to the
    #common seed before each sample so both see the SAME crop of base[i].
    for i in range(n_samples):
        photo.rng = np.random.default_rng(SEED)
        plain.rng = np.random.default_rng(SEED)
        lp, rp, dp, _ = photo[i]
        l0, r0, d0 = plain[i]
        assert torch.equal(dp, d0), f"disparity changed by augmentation @sample {i}"
        assert dp.shape == (1, CROP_H, CROP_W), dp.shape
        mp = (dp[0].numpy() > 0) & (dp[0].numpy() < 176)
        m0 = (d0[0].numpy() > 0) & (d0[0].numpy() < 176)
        assert np.array_equal(mp, m0), f"valid mask changed @sample {i}"
    print(f"[1/4] disparity + valid mask bit-identical over {n_samples} samples: PASS")

    # 2. augmented images finite, no NaN/inf, valid normalized range.
    for i in range(n_samples):
        lp, rp, _, _ = photo[i]
        for t, name in ((lp, "left"), (rp, "right")):
            a = t.numpy()
            assert np.isfinite(a).all(), f"{name} non-finite @sample {i}"
            # ImageNet-normalized 0-255 input must lie within
            # [-(max mean)/min std, 255/min std] = approx [-2.2, 4.5].
            assert a.min() >= -3.0 and a.max() <= 5.0, (
                f"{name} out of range @sample {i}: [{a.min()}, {a.max()}]")
    print(f"[2/4] augmented images finite + in normalized range over {n_samples} samples: PASS")

    # 3. left and right receive DIFFERENT parameter draws on some sample.
    seen_diff = False
    for i in range(n_samples):
        _, _, _, (lparams, rparams) = photo[i]
        assert lparams is not None and rparams is not None
        if tuple(lparams) != tuple(rparams):
            seen_diff = True
    assert seen_diff, "left/right photo params never differed — augmentation is symmetric!"
    print(f"[3/4] left/right asymmetric draws differ on some sample over {n_samples}: PASS")

    # 4. photo_prob=0.0 pipeline bit-identical to ARM K, sample for sample.
    # Both consume crop + jitter draws only, in the same order, so sequential
    # calls stay in lockstep once reset to the common seed.
    plain.rng = np.random.default_rng(SEED)
    armk.rng = np.random.default_rng(SEED)
    for i in range(n_samples):
        l0, r0, d0 = plain[i]
        lk, rk, dk = armk[i]
        assert torch.equal(l0, lk), f"left differs from ARM K @sample {i}"
        assert torch.equal(r0, rk), f"right differs from ARM K @sample {i}"
        assert torch.equal(d0, dk), f"disparity differs from ARM K @sample {i}"
    print(f"[4/4] photo_prob=0.0 bit-identical to ARM K over {n_samples} samples: PASS")
    print("SELF-CHECK ALL PASS")


def main(epochs: int = EPOCHS, photo_prob: float = PHOTO_PROB) -> None:
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

    train = CroppedKitti(train_base, seed=SEED, photo_prob=photo_prob)
    loader = DataLoader(
        train, batch_size=BATCH, shuffle=True, num_workers=0,
        generator=make_generator(SEED), worker_init_fn=worker_init_fn,
    )

    config = StereoNetConfig()
    model = StereoNet(config).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    exp_config = {
        "arm": "ARM G (photo-robustness): ARM K + asymmetric photometric aug, 200 epochs",
        "dataset": "kitti2015",
        "split": "hailo_calib (scenes 0-159) train, hailo_val (160-199) eval; overlap 0",
        "split_enforcement": "train_base split='hailo_calib' only; no hailo_val sample enters optimizer loop",
        "resolution": [CROP_H, CROP_W],
        "crop": "random {}x{}".format(CROP_H, CROP_W),
        "disparity_range": config.max_disparity_px,
        "batch_size": BATCH,
        "precision": "fp32",
        "seed": SEED,
        "epochs": epochs,
        "optimizer": "Adam betas=(0.9, 0.999)",
        "learning_rate": LR,
        "schedule": "cosine annealing to 0 over {} epochs".format(epochs),
        "loss": "masked smooth L1, beta=1.0, valid = gt > 0 and gt < max_disparity",
        "augmentation": (
            "random crop, independent per-image gain jitter sigma=0.1 (UNCHANGED from ARM K, "
            "applied on top — this arm is NOT a gain-jitter replacement); PLUS asymmetric "
            "photometric aug per sample, independent draws for left/right, fixed order "
            "brightness U(0.8,1.2) -> contrast U(0.8,1.2) -> gamma U(0.8,1.2) in [0,1] space "
            "BEFORE ImageNet normalization, clamped to [0,1] after each step; "
            "photo_prob={}; no flip/scale/rotation/crop change; disparity + valid mask unmodified".format(photo_prob)
        ),
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

    for epoch in range(epochs):
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
        if epoch % 5 == 0 or epoch == epochs - 1:
            m = validate(model, val_base, device, limit=10)
            row["val_epe"] = m.epe
            row["val_d1"] = m.d1
            if m.epe < best_val_epe:
                best_val_epe = m.epe
                best_epoch = epoch
                torch.save({"model": model.state_dict(), "config": exp_config},
                           OUT_DIR / "arm_g_best.pth")
        history.append(row)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print("epoch {:>3} loss {:8.4f} lr {:.2e}{}".format(
            epoch, row["mean_loss"], row["lr"],
            " val EPE {:.3f} D1 {:.2f}%".format(row["val_epe"], row["val_d1"])
            if "val_epe" in row else ""), flush=True)

    wall_s = time.time() - t0
    fh.close()
    final_ckpt = OUT_DIR / "arm_g_final.pth"
    torch.save({"model": model.state_dict(), "config": exp_config}, final_ckpt)
    record = {"arm": ARM, "config": exp_config, "epochs_run": epochs,
              "wall_clock_s": wall_s, "git_head": git_head,
              "best_val_epe_10scene": best_val_epe, "best_epoch": best_epoch,
              "final_sha256": sha256_file(final_ckpt),
              "best_sha256": sha256_file(OUT_DIR / "arm_g_best.pth"),
              "history": history}
    (OUT_DIR / "arm_g_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("ARM G done: wall {:.1f}s best10 EPE {:.4f} @epoch {}".format(wall_s, best_val_epe, best_epoch))


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="ARM G: ARM K + asymmetric photometric augmentation.")
    _parser.add_argument("--epochs", type=int, default=EPOCHS,
                         help="Run length in epochs (default %(default)s). Cosine T_max matches.")
    _parser.add_argument("--photo-prob", type=float, default=PHOTO_PROB,
                         help="Probability of applying the photometric aug per image (default %(default)s).")
    _parser.add_argument("--self-check", action="store_true",
                         help="Run pre-training assertions and exit without training.")
    _parser.add_argument("--self-check-samples", type=int, default=20,
                         help="Samples per assertion (default %(default)s).")
    _args = _parser.parse_args()
    if _args.self_check:
        self_check(n_samples=_args.self_check_samples)
    else:
        main(epochs=_args.epochs, photo_prob=_args.photo_prob)
