"""H1 -- what is a working cost volume worth?

Controlled experiment. Two runs, ``EXP-H1-BASE`` (``cost_volume_shift="none"``,
Phase 1's frozen degenerate volume) and ``EXP-H1-WORKING``
(``cost_volume_shift="left"``, the shift the construction was always intended
to perform -- implemented and unit-tested in
``src/models/stereonet/cost_volume.py`` behind a flag Phase 1 never enabled).

Everything except the cost-volume shift is held fixed to Phase 1's own
EXP-016 training recipe, byte-for-byte: same dataset split, same crop, same
augmentation, same optimizer/schedule, same seed, same architecture,
same loss. This isolates the correspondence mechanism as the only changed
variable, for causal attribution (per PHASE_1_FINAL_REPORT.md's own
recommendation and docs/research_questions.md H1).

This script is Phase 2 code. It imports Phase 1's shared library modules
(src/common, src/datasets, src/losses, src/models) read-only and does not
modify scripts/exp_train_convergence.py, which stays frozen with Phase 1.

Known limitation, inherited from EXP-016 and stated rather than hidden: 160
scenes from random initialisation is a convergence-pipeline proof, not a
competitive accuracy result (Phase 1 charter, deferred full-scale training).
Any H1 comparison drawn from this run is bounded by that same data budget for
BOTH arms -- the comparison is still valid as a controlled A/B, but neither
arm should be read as this architecture's ceiling.

    python phase2/scripts/exp_h1_cost_volume.py --shift none    # EXP-H1-BASE
    python phase2/scripts/exp_h1_cost_volume.py --shift left    # EXP-H1-WORKING
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

REPO_ROOT = Path(__file__).resolve().parents[2]
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

PHASE2_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PHASE2_ROOT / "experiments"
OUT_DIR = PHASE2_ROOT / "results" / "training"

# Identical to EXP-016: crop must divide by 16 so the 1/16 feature map has
# integer dimensions.
CROP_H, CROP_W = 256, 512

SHIFT_TO_EXPERIMENT_ID = {"none": "EXP-H1-BASE", "left": "EXP-H1-WORKING"}


class CroppedKitti(Dataset):
    """Identical to EXP-016's CroppedKitti -- random crops, no horizontal flip
    (it would invert disparity sign and swap which camera is which)."""

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
    ap.add_argument("--shift", choices=["none", "left"], required=True,
                     help="cost_volume_shift: 'none'=EXP-H1-BASE (frozen "
                          "degenerate volume), 'left'=EXP-H1-WORKING (working "
                          "correspondence search)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="",
                     help="appended to the experiment id, e.g. '-v2'. Reruns "
                          "at a different budget MUST use one, so the earlier "
                          "records are never overwritten.")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_calib")
    val_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val")
    train = CroppedKitti(train_base, seed=args.seed)
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0)

    config = StereoNetConfig(cost_volume_shift=args.shift)
    model = StereoNet(config).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    exp_config = {
        "hypothesis": "H1: a genuinely shifted stereo correspondence volume, "
                       "everything else held fixed, changes accuracy by a "
                       "measurable amount versus Phase 1's frozen degenerate "
                       "volume (docs/research_questions.md, H1).",
        "changed_variable": "cost_volume_shift",
        "cost_volume_shift": args.shift,
        "held_fixed": "feature extractor, aggregation, refinement, loss, "
                       "optimizer, schedule, seed, data split, crop, "
                       "augmentation -- identical to Phase 1 EXP-016",
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
        "run_variant": (
            "v1 (20 epochs)" if not args.suffix else
            "rerun '{}': identical to the v1 arms except the epoch budget "
            "({} vs 20) and the cosine T_max that tracks it. Both v2 arms "
            "share this budget, so BASE-vs-WORKING remains a controlled A/B; "
            "v1 records are preserved unmodified.".format(args.suffix, args.epochs)
        ),
        "device": str(device),
        "known_deviation": "no batch normalisation, because the model reproduces "
                           "the BN-folded exported artifact rather than the "
                           "training-time upstream model",
        "known_limitation": "160 scenes from random init is a convergence-"
                            "pipeline proof (Phase 1 EXP-016), not a "
                            "competitive accuracy result. Both H1 arms share "
                            "this budget, so the A/B comparison is valid; "
                            "neither arm's absolute number is a ceiling.",
    }

    exp_id = SHIFT_TO_EXPERIMENT_ID[args.shift] + args.suffix
    if (EXPERIMENTS_DIR / exp_id).exists():
        raise SystemExit(
            "{} already exists -- records are never overwritten. Pass a new "
            "--suffix.".format(exp_id)
        )
    with Experiment(
        "H1: cost_volume_shift='{}' -- {}".format(
            args.shift,
            "frozen degenerate volume (baseline)" if args.shift == "none"
            else "working shifted correspondence volume",
        ),
        config=exp_config,
        experiment_id=exp_id,
        experiments_dir=EXPERIMENTS_DIR,
    ) as exp:
        exp.note(exp_config["hypothesis"])
        exp.note(exp_config["known_limitation"])
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

        # Parameter count and MACs are DERIVED here, not re-measured against
        # Hailo's compiled figures (that comparison is only meaningful for
        # shift="none", which reproduces the exact reference architecture).
        exp.metric("parameter_count_unique", model.parameter_count())
        exp.metric("cost_volume_bytes_per_image",
                   model.cost_volume.memory_bytes(1, config.feature_channels, 23, 77))

        ckpt = OUT_DIR / (exp_id + "_checkpoint.pth")
        torch.save({"model": model.state_dict(), "config": exp_config}, ckpt)
        exp.metric("checkpoint", str(ckpt.relative_to(REPO_ROOT)))
        (OUT_DIR / (exp_id + "_history.json")).write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

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
            exp.metric("validation_change",
                       classify_validation_change(val_initial, val_final))
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
            exp.note(describe_validation(
                val_initial, val_final,
                best=val_best_row["val_epe"],
                best_label="epoch {}".format(val_best_row["epoch"]),
            ))

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
            + " cost_volume_shift='{}'. Compare against the paired H1 arm "
              "before drawing any conclusion about the value of a working "
              "cost volume.".format(args.shift)
        )
        print("\nrecorded as " + exp.id)


if __name__ == "__main__":
    main()
