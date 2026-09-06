"""EXP: does the independent training pipeline actually train?

A short convergence run, not an accuracy result. The goal set out in the Phase 1
charter is to demonstrate that the pipeline trains -- loss decreases, gradients
behave, the implementation can converge -- and to have the full recipe written
down. Full Scene Flow pre-training is deferred to Phase 2 and would occupy the
single available GPU for days.

Trains from random initialisation on Hailo's own quantisation-calibration split
(the first 160 KITTI 2015 training scenes), and validates on the 40-scene
validation split that has been used throughout. The two are disjoint, so the
validation curve is honest even though 160 scenes is far too little data to
reach a competitive result -- and it is not presented as one.

**Deviation from the documented recipe, recorded rather than hidden.** The
upstream training model uses batch normalisation [SOURCE: SR-011]. Our model was
built from the exported ONNX, where batch norm has been folded into the
convolution weights, so it trains without normalisation layers. That is the right
choice for reproducing the deployed artifact and the wrong one for reproducing
training. It is expected to slow convergence, and it means this run demonstrates
that the pipeline works, not that it reproduces the upstream training dynamics.

    python scripts/exp_train_convergence.py [--epochs N] [--batch N]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.conclusions import (  # noqa: E402
    classify_series_trend,
    classify_validation_change,
    describe_training_outcome,
    describe_validation,
)
from src.common.experiment import Experiment  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.evaluation.metrics import disparity_metrics  # noqa: E402
from src.losses.disparity import masked_smooth_l1  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "training"

# Crop must divide by 16 so the 1/16 feature map has integer dimensions.
CROP_H, CROP_W = 256, 512


class CroppedKitti(Dataset):
    """Random crops with a horizontal-flip-free augmentation policy.

    Stereo pairs must not be flipped horizontally: it reverses the sign of
    disparity and swaps which camera is which. Colour jitter and random crops
    are safe; geometric augmentation along the epipolar direction is not.
    """

    def __init__(self, base: Kitti2015Stereo, seed: int = 0, jitter: float = 0.1):
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
            # Independent per-image gain, which is a real stereo nuisance:
            # the two cameras rarely agree exactly on exposure.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_calib")
    val_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val")
    train = CroppedKitti(train_base, seed=args.seed)
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0)

    config = StereoNetConfig()
    model = StereoNet(config).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    exp_config = {
        "dataset": "kitti2015",
        "split": "hailo_calib (scenes 0-159) train, hailo_val (160-199) validate",
        "resolution": [CROP_H, CROP_W],
        "crop": "random {}x{}".format(CROP_H, CROP_W),
        "disparity_range": config.max_disparity_px,
        "batch_size": args.batch,
        "precision": "fp32",
        "seed": args.seed,
        "epochs": args.epochs,
        "optimizer": "Adam betas=(0.9, 0.999)",
        "learning_rate": args.lr,
        "schedule": "cosine annealing to 0 over {} epochs".format(args.epochs),
        "loss": "masked smooth L1, beta=1.0, valid = gt > 0 and gt < max_disparity",
        "augmentation": "random crop, independent per-image gain jitter sigma=0.1; "
                        "no horizontal flip (it inverts disparity)",
        "initialisation": "PyTorch defaults, random",
        "device": str(device),
        "known_deviation": "no batch normalisation, because the model reproduces "
                           "the BN-folded exported artifact rather than the "
                           "training-time upstream model",
    }

    with Experiment(
        "Short convergence run of the independent training pipeline on KITTI "
        "2015",
        config=exp_config,
    ) as exp:
        exp.note(
            "This demonstrates that the training pipeline works. It is not an "
            "accuracy result: 160 scenes from random initialisation cannot "
            "reach a competitive figure, and no claim is made that it does. "
            "Scene Flow pre-training is deferred; the recipe is recorded in "
            "this experiment's config."
        )
        exp.note(exp_config["known_deviation"])

        history = []
        grad_norms = []
        t0 = time.time()
        for epoch in range(args.epochs):
            epoch_loss, epoch_pixels, batches = 0.0, 0, 0
            for left, right, disparity in loader:
                left = left.to(device)
                right = right.to(device)
                disparity = disparity.to(device)

                out = model(left, right)
                loss, n = masked_smooth_l1(
                    out, disparity, max_disparity=float(config.max_disparity_px)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gn = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
                )
                optimizer.step()

                epoch_loss += float(loss)
                epoch_pixels += n
                batches += 1
                grad_norms.append(gn)

            scheduler.step()
            mean_loss = epoch_loss / max(batches, 1)
            row = {
                "epoch": epoch,
                "mean_loss": mean_loss,
                "valid_pixels": epoch_pixels,
                "lr": float(scheduler.get_last_lr()[0]),
                "median_grad_norm": float(np.median(grad_norms[-batches:])),
            }
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                m = validate(model, val_base, device, limit=10)
                row["val_epe"] = m.epe
                row["val_d1"] = m.d1
            history.append(row)
            exp.log(
                "epoch {:>3}  loss {:8.4f}  grad {:9.3f}  lr {:.2e}{}".format(
                    epoch, mean_loss, row["median_grad_norm"], row["lr"],
                    "  val EPE {:.3f}  D1 {:.2f}%".format(row["val_epe"], row["val_d1"])
                    if "val_epe" in row else "",
                )
            )

        exp.metric("history", history)
        exp.metric("wall_clock_s", time.time() - t0)
        losses = [r["mean_loss"] for r in history]
        exp.metric("first_epoch_loss", losses[0])
        exp.metric("final_epoch_loss", losses[-1])
        exp.metric("loss_reduction_factor", losses[0] / max(losses[-1], 1e-9))
        exp.metric("min_loss", min(losses))
        exp.metric(
            "loss_decreased_monotonically_over_halves",
            float(np.mean(losses[: len(losses) // 2]))
            > float(np.mean(losses[len(losses) // 2:])),
        )
        exp.metric(
            "gradient_norms",
            {
                "median": float(np.median(grad_norms)),
                "max": float(np.max(grad_norms)),
                "any_nan": bool(np.any(~np.isfinite(grad_norms))),
            },
        )

        ckpt = OUT_DIR / "convergence_run.pth"
        torch.save({"model": model.state_dict(), "config": exp_config}, ckpt)
        exp.metric("checkpoint", str(ckpt.relative_to(REPO_ROOT)))
        (OUT_DIR / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        # Every statement below is derived from the recorded values. An earlier
        # version of this script asserted "the loss decreases", "no non-finite
        # values" and "validation improves" as literal text, independent of what
        # the run produced -- and EXP-016 duly recorded that validation improved
        # while its validation EPE rose from 18.497 px to 19.216 px. See
        # experiments/EXP-016/CORRECTION.md.
        val_rows = [r for r in history if "val_epe" in r]
        val_initial = val_rows[0]["val_epe"] if val_rows else None
        val_final = val_rows[-1]["val_epe"] if val_rows else None
        val_best_row = min(val_rows, key=lambda r: r["val_epe"]) if val_rows else None

        finite_grads = [g for g in grad_norms if math.isfinite(g)]
        non_finite_grads = len(grad_norms) - len(finite_grads)

        exp.metric(
            "validation_epe_series",
            [{"epoch": r["epoch"], "val_epe": r["val_epe"], "val_d1": r["val_d1"]}
             for r in val_rows],
        )
        if val_rows:
            exp.metric("validation_epe_initial", val_initial)
            exp.metric("validation_epe_final", val_final)
            exp.metric("validation_epe_best", val_best_row["val_epe"])
            exp.metric("validation_epe_best_epoch", val_best_row["epoch"])
            exp.metric(
                "validation_change",
                classify_validation_change(val_initial, val_final),
            )
        exp.metric("loss_change", classify_series_trend(losses)
                   if len(losses) >= 2 else None)
        exp.metric("non_finite_gradient_count", non_finite_grads)

        exp.note(
            "Loss went from {:.4f} to {:.4f} over {} epochs. Gradient norms had "
            "median {:.3f} and maximum {:.3f}, with {} non-finite "
            "values.".format(
                losses[0], losses[-1], args.epochs,
                float(np.median(grad_norms)), float(np.max(grad_norms)),
                non_finite_grads if non_finite_grads else "no",
            )
        )
        if val_rows:
            exp.note(
                describe_validation(
                    val_initial, val_final,
                    best=val_best_row["val_epe"],
                    best_label="epoch {}".format(val_best_row["epoch"]),
                )
                + " For context the reference weights score EPE 1.313 px and "
                "D1 8.15 % on the full split (EXP-005); the gap is a statement "
                "about the data budget, not about the implementation, which "
                "EXP-011 verified against the reference to a relative 1e-7."
            )

        exp.conclude(
            describe_training_outcome(
                losses=losses,
                grad_norms=grad_norms,
                val_initial=val_initial,
                val_final=val_final,
                val_best=val_best_row["val_epe"] if val_rows else None,
                val_best_label=(
                    "epoch {}".format(val_best_row["epoch"]) if val_rows else ""
                ),
                checkpoint_written=ckpt.exists(),
            )
            + " What this run establishes is that the training pipeline runs end "
            "to end and produces a checkpoint; full-scale training remains "
            "deferred, with the recipe recorded in this experiment's "
            "configuration."
        )
        print("\nrecorded as " + exp.id)


if __name__ == "__main__":
    main()
