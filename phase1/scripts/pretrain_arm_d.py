"""ARM D pretraining -- Driving-only synthetic pretraining (Option D proxy).

Produces the checkpoint that the KITTI fine-tune stage (finetune_arm_d.py)
consumes. Starting point is the upstream pretrain recipe
(reference/upstream/extracted/StereoNet-master/pretrain-sceneflow/sceneflow-pretrain.py,
READ ONLY): RMSprop lr 1e-3 / weight decay 1e-4, ExponentialLR gamma 0.9,
smooth-L1 masked to disp < 160, batch 16, random 256x512 crop, 20 epochs.

Same model: StereoNet(StereoNetConfig()), architecture unchanged, no BN,
fp32, seed 0.

Batch size: upstream batch 16 does NOT fit 8 GB of VRAM at a 256x512 crop.
Activation memory is dominated by the full-resolution refinement path: one
(B,32,256,512) fp32 activation is B*16.8 MB, ~12 such tensors are live
(~B*200 MB forward, ~2.5x with backward), plus the upsampled cost
(B,12,256,512) at B*6.3 MB; batch 16 lands at ~8 GB+ before CUDA context and
fragmentation, i.e. OOM on the RTX 4060 Laptop (8188 MiB). ARM K proves
batch 2 fits on this exact GPU at this exact crop. Default batch is therefore
4: 2x the proven value with comfortable headroom, the largest justifiable
without a live VRAM probe (no GPU probe permitted while ARM K600 trains).
Batch size is a command-line argument (--batch); the value actually used is
recorded in the run record.

Every deviation from upstream is recorded in DEVIATIONS below and in the run
record. Key ones: batch 16->4 (VRAM, above); num_workers 12->0 (frozen
determinism recipe, as ARM K); seed 1->0; no DataParallel (single GPU);
scheduler stepped AFTER each epoch (upstream steps before, decaying the LR
before epoch 0); checkpoint selection by lowest mean train-epoch loss
(upstream selects nothing during pretraining and reports FlyingThings3D TEST
EPE at the end -- Driving-only has no equivalent held-out set here); no
checkpoint resume (fresh run, as ARM K); model is this repo's StereoNet
reimplementation, not upstream stereonet(); ImageNet normalisation via
src.datasets.kitti2015.normalize (same 0-255 statistics upstream's
preprocess.py uses); loss mask is exactly upstream's (disp < 160, NO gt > 0
floor -- dense synthetic ground truth, unlike the KITTI stage).

Single-line shell rule: run as `python phase1\\scripts\\pretrain_arm_d.py [--epochs N] [--batch B]`.
DO NOT RUN until the Driving archives are on disk and the layout guesses in
src/datasets/driving.py are verified against the real extraction.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from phase1.harness.determinism import make_generator, seed_all, worker_init_fn  # noqa: E402
from src.datasets.driving import DrivingStereo  # noqa: E402
from src.datasets.kitti2015 import normalize  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

ARM = "arm_d_pretrain"
OUT_DIR = REPO_ROOT / "phase1" / "runs" / ARM
CROP_H, CROP_W = 256, 512
EPOCHS = 20
BATCH = 4
LR = 1e-3
WEIGHT_DECAY = 1e-4
GAMMA = 0.9
MAX_DISP_PRETRAIN = 160.0
SEED = 0

DEVIATIONS = [
    "batch 16 -> 4: upstream batch 16 OOMs 8 GB VRAM at 256x512 (see module docstring)",
    "num_workers 12 -> 0: frozen determinism recipe, as ARM K",
    "seed 1 -> 0: frozen seed, as ARM K",
    "no DataParallel: single-GPU run",
    "scheduler stepped after each epoch; upstream steps before (LR decays before epoch 0 there)",
    "checkpoint selection by lowest mean train-epoch loss; upstream reports FlyingThings3D TEST EPE instead",
    "no checkpoint resume: fresh run, as ARM K",
    "model is this repo's StereoNet reimplementation, not upstream stereonet()",
    "no gain jitter / no augmentation: upstream pretrain path applies none (augment=False)",
]


class CroppedDriving(Dataset):
    """Random 256x512 crop + ImageNet normalisation, upstream training path.

    Mirrors upstream SceneFlowLoader.myImageFloder(training=True): same crop
    location applied to left, right and disparity, then normalise. No
    augmentation -- upstream's pretrain transform is augment=False.
    """

    def __init__(self, base: DrivingStereo, seed: int = 0) -> None:
        self.base = base
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        s = self.base[i]
        h, w = s.disparity.shape
        y = int(self.rng.integers(0, h - CROP_H + 1))
        x = int(self.rng.integers(0, w - CROP_W + 1))
        sl = (slice(y, y + CROP_H), slice(x, x + CROP_W))
        left = normalize(s.left[sl])[0]
        right = normalize(s.right[sl])[0]
        # PFM disparity is already in pixels -- do NOT divide by 256 (KITTI-only).
        disparity = s.disparity[sl][None]
        return (
            torch.from_numpy(np.ascontiguousarray(left, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(right, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(disparity, dtype=np.float32)),
        )


def pretrain_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Upstream smooth-L1 masked to disp < 160 (no gt > 0 floor)."""
    mask = target < MAX_DISP_PRETRAIN
    count = int(mask.sum())
    if count == 0:
        return (prediction.sum() * 0.0), 0
    return F.smooth_l1_loss(prediction[mask], target[mask]), count


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(epochs: int = EPOCHS, batch: int = BATCH, datapath: str | Path | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    # NOTE: torch.use_deterministic_algorithms NOT enabled -- same reason as
    # ARM K (bilinear-backward has no deterministic CUDA impl). Recorded as
    # deviation: none (matches ARM K recipe).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(datapath) if datapath is not None else REPO_ROOT / "data" / "sceneflow"

    train_base = DrivingStereo(root)
    loader = DataLoader(
        CroppedDriving(train_base, seed=SEED), batch_size=batch, shuffle=True,
        num_workers=0, generator=make_generator(SEED), worker_init_fn=worker_init_fn,
    )

    config = StereoNetConfig()
    model = StereoNet(config).to(device)
    model.train()
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

    exp_config = {
        "arm": "ARM D pretraining: Driving-only synthetic pretraining",
        "dataset": "sceneflow/driving (frames_cleanpass + disparity PFM)",
        "samples": len(train_base),
        "skipped_triplets": train_base.skipped,
        "frames_root": str(train_base.frames_root),
        "disparity_root": str(train_base.disparity_root),
        "resolution": [CROP_H, CROP_W],
        "crop": "random {}x{}".format(CROP_H, CROP_W),
        "batch_size": batch,
        "precision": "fp32",
        "seed": SEED,
        "epochs": epochs,
        "optimizer": "RMSprop lr=1e-3 weight_decay=1e-4",
        "learning_rate": LR,
        "schedule": "ExponentialLR gamma={}".format(GAMMA),
        "loss": "smooth L1, valid = gt < 160 (upstream mask, no gt>0 floor)",
        "augmentation": "random crop only; no jitter, no flip (upstream augment=False)",
        "initialisation": "PyTorch defaults, random",
        "checkpoint_selection": "lowest mean train-epoch loss",
        "device": str(device),
        "known_deviation": "no batch normalisation (BN-folded exported artifact), carried into all arms",
        "deviations_from_upstream": DEVIATIONS,
    }

    log_path = OUT_DIR / "training_log.jsonl"
    fh = open(log_path, "w", encoding="utf-8")
    history = []
    t0 = time.time()
    best_loss = float("inf")
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
            loss, n = pretrain_loss(out, disparity)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            epoch_pixels += n
            batches += 1
        scheduler.step()
        mean_loss = epoch_loss / max(batches, 1)
        row = {"epoch": epoch, "mean_loss": mean_loss,
               "valid_pixels": epoch_pixels, "lr": float(scheduler.get_last_lr()[0])}
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "config": exp_config},
                       OUT_DIR / "arm_d_pretrain_best.pth")
        history.append(row)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print("epoch {:>3} loss {:8.4f} lr {:.2e}".format(
            epoch, mean_loss, row["lr"]), flush=True)

    wall_s = time.time() - t0
    fh.close()
    final_ckpt = OUT_DIR / "arm_d_pretrain_final.pth"
    torch.save({"model": model.state_dict(), "config": exp_config}, final_ckpt)
    record = {"arm": ARM, "config": exp_config, "epochs_run": epochs,
              "wall_clock_s": wall_s, "git_head": git_head,
              "best_mean_loss": best_loss, "best_epoch": best_epoch,
              "final_sha256": sha256_file(final_ckpt),
              "best_sha256": sha256_file(OUT_DIR / "arm_d_pretrain_best.pth"),
              "history": history}
    (OUT_DIR / "arm_d_pretrain_record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print("ARM D pretrain done: wall {:.1f}s best loss {:.4f} @epoch {}".format(
        wall_s, best_loss, best_epoch))


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="ARM D: Driving-only synthetic pretraining.")
    _parser.add_argument("--epochs", type=int, default=EPOCHS,
                         help="Run length in epochs (default %(default)s).")
    _parser.add_argument("--batch", type=int, default=BATCH,
                         help="Batch size (default %(default)s; upstream 16 OOMs 8 GB VRAM).")
    _parser.add_argument("--datapath", type=str, default=None,
                         help="SceneFlow root (default data/sceneflow).")
    _args = _parser.parse_args()
    main(epochs=_args.epochs, batch=_args.batch, datapath=_args.datapath)
