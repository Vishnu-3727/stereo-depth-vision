"""STAGE A -- DIAGNOSTIC ONLY. Not a scientific experiment, not a training run.

Runs ONE epoch of the H2 seed-0 configuration (== E3b arm A: six refinement
blocks, cost_volume_shift="left", standardised soft-argmin) in a fresh process
and records everything needed to tell apart two hypotheses for the observed
same-config reproduction failure:

    Case A -- the two runs do not see the same first training batch
              (RNG / DataLoader / augmentation sequencing)
    Case B -- they see the identical batch but compute different numbers
              (cuDNN / CUDA kernel nondeterminism)

It writes ONLY under phase2/diagnostics/determinism/<run-dir>/. It never writes
into phase2/experiments/, phase2/results/ or any Phase 1 path, it registers no
Experiment record, and it saves no checkpoint. src/ and every Phase 2 scientific
script are imported read-only and are not modified.

    python phase2/diagnostics/determinism/diag_determinism.py --out DIR [--deterministic]

--deterministic enables torch.use_deterministic_algorithms(True),
cudnn.deterministic=True and cudnn.benchmark=False, and records exactly which
operation (if any) refuses to run under them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Deterministic cuBLAS needs this set BEFORE the CUDA context exists.
if "--deterministic" in sys.argv and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
from torch.utils.data import DataLoader, Dataset              # noqa: E402

from src.datasets.kitti2015 import Kitti2015Stereo            # noqa: E402
from src.losses.disparity import masked_smooth_l1             # noqa: E402
from phase2.scripts.exp_h1_cost_volume import CroppedKitti, validate   # noqa: E402
from phase2.scripts.exp_h2_seed_replication import build_model         # noqa: E402

SEED = 0
BATCH_SIZE = 2
LR = 1e-3
EPOCHS_FOR_SCHEDULE = 200      # cosine T_max of the real recipe; one epoch is run
CROP_H, CROP_W = 256, 512
FROZEN_COMMIT_REF = "phase-1-frozen^" + "{commit}"


def sha(obj) -> str:
    """sha256 of the object's bytes, truncated to 16 hex characters."""
    if isinstance(obj, torch.Tensor):
        data = obj.detach().cpu().contiguous().numpy().tobytes()
    elif isinstance(obj, np.ndarray):
        data = np.ascontiguousarray(obj).tobytes()
    elif isinstance(obj, (bytes, bytearray)):
        data = bytes(obj)
    else:
        data = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def rng_snapshot(device: str) -> dict:
    """Hashes of every RNG stream. Hashing method is `sha` above."""
    snap = {
        "python_random": sha(json.dumps(random.getstate(), default=str)),
        "numpy_global_legacy": sha(json.dumps(
            [x.tolist() if hasattr(x, "tolist") else x
             for x in np.random.get_state()], default=str)),
        "torch_cpu": sha(torch.get_rng_state()),
    }
    if device == "cuda" and torch.cuda.is_available():
        snap["torch_cuda"] = sha(torch.cuda.get_rng_state())
    return snap


class RecordingDataset(Dataset):
    """Wraps CroppedKitti and records what each __getitem__ actually produced.

    It never draws a random number of its own. The crop coordinates and jitter
    factors are recovered by replaying the wrapped generator's *saved* state on a
    private copy, so the live stream is untouched.
    """

    def __init__(self, inner: CroppedKitti, base: Kitti2015Stereo):
        self.inner = inner
        self.base = base
        self.log: list[dict] = []

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        state_before = self.inner.rng.bit_generator.state
        left, right, disparity = self.inner[i]
        state_after = self.inner.rng.bit_generator.state

        replay = np.random.default_rng()
        replay.bit_generator.state = state_before
        h, w = self.base[i].disparity.shape
        y = int(replay.integers(0, h - CROP_H + 1))
        x = int(replay.integers(0, w - CROP_W + 1))
        jl = float(replay.normal(0, self.inner.jitter))
        jr = float(replay.normal(0, self.inner.jitter))

        self.log.append({
            "dataset_index": int(i),
            "scene_name": self.base.names[i],
            "crop_y": y, "crop_x": x,
            "source_h": int(h), "source_w": int(w),
            "jitter_left": jl, "jitter_right": jr,
            "rng_state_before": sha(json.dumps(state_before, default=str)),
            "rng_state_after": sha(json.dumps(state_after, default=str)),
            "left_sha": sha(left), "right_sha": sha(right),
            "disparity_sha": sha(disparity),
            "left_mean": float(left.mean()), "left_std": float(left.std()),
            "disparity_valid_px": int((disparity > 0).sum()),
        })
        return left, right, disparity


def environment(deterministic: bool) -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True,
                                  text=True, check=False).stdout.strip()
        except Exception as exc:                              # pragma: no cover
            return "unavailable: {}".format(exc)
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_phase_1_frozen_tag_object": git("rev-parse", "phase-1-frozen"),
        "git_phase_1_frozen_commit": git("rev-parse", FROZEN_COMMIT_REF),
        "git_phase_1_diff_vs_frozen": git("diff", "phase-1-frozen", "--", "src", "scripts"),
        "git_status_short": git("status", "--short"),
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (torch.backends.cudnn.version()
                          if torch.backends.cudnn.is_available() else None),
        "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "gpu_capability": (".".join(map(str, torch.cuda.get_device_capability(0)))
                           if torch.cuda.is_available() else None),
        "nvidia_driver": _driver(),
        "deterministic_requested": bool(deterministic),
        "flags": {
            "cudnn.deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn.benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn.enabled": bool(torch.backends.cudnn.enabled),
            "cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "env_vars": {k: os.environ.get(k) for k in (
            "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED",
            "CUDA_LAUNCH_BLOCKING", "TORCH_USE_CUDA_DSA", "OMP_NUM_THREADS")},
    }


def _driver():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, check=False).stdout.strip()
        return out.splitlines()[0] if out else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="diagnostic output directory")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    determinism_error = None
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            determinism_error = "{}: {}".format(type(exc).__name__, exc)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    record: dict = {"label": args.label,
                    "environment": environment(args.deterministic),
                    "determinism_setup_error": determinism_error,
                    "nondeterministic_op_error": None}

    t0 = time.time()

    # --- construction, in E3b arm A's exact order --------------------------
    train_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_calib")
    val_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val")

    record["rng_before_build"] = rng_snapshot(device)
    model = build_model(SEED, device)          # seeds torch+numpy, then initialises
    record["rng_after_build"] = rng_snapshot(device)
    record["model"] = {
        "parameters_total": int(sum(p.numel() for p in model.parameters())),
        "weights_sha": sha(torch.cat([p.detach().reshape(-1).cpu()
                                      for p in model.parameters()])),
        "refinement_blocks": len(model.refinement.blocks),
        "cost_volume_shift": model.cost_volume.shift,
        "regression": type(model.regression).__name__,
    }

    inner = CroppedKitti(train_base, seed=SEED)
    dataset = RecordingDataset(inner, train_base)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    record["rng_after_loader_built"] = rng_snapshot(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_FOR_SCHEDULE)
    max_disparity = float(model.config.max_disparity_px)
    record["max_disparity_px"] = max_disparity

    per_batch, grad_norms = [], []
    nan_or_inf = False
    try:
        for batch_index, (left, right, disparity) in enumerate(loader):
            left, right = left.to(device), right.to(device)
            disparity = disparity.to(device)

            if batch_index == 0:
                record["rng_at_batch0"] = rng_snapshot(device)
                record["batch0_inputs"] = {
                    "left_shape": list(left.shape), "right_shape": list(right.shape),
                    "disparity_shape": list(disparity.shape),
                    "left_sha": sha(left), "right_sha": sha(right),
                    "disparity_sha": sha(disparity),
                    "left_mean": float(left.mean()), "left_std": float(left.std()),
                    "right_mean": float(right.mean()), "right_std": float(right.std()),
                    "disparity_valid_px": int((disparity > 0).sum()),
                    "disparity_max": float(disparity.max()),
                    "samples": dataset.log[:BATCH_SIZE],
                }
                # Kernel-determinism probe: identical inputs, identical weights,
                # two forward passes inside ONE process. Any difference here is
                # computational, never RNG.
                with torch.no_grad():
                    f1 = model(left, right)
                    f2 = model(left, right)
                record["batch0_same_process_forward_repeat"] = {
                    "max_abs_diff": float((f1 - f2).abs().max()),
                    "bitwise_identical": bool(torch.equal(f1, f2)),
                    "forward_sha_1": sha(f1), "forward_sha_2": sha(f2),
                    "out_mean": float(f1.mean()), "out_std": float(f1.std()),
                }
                out0 = model(left, right)
                loss0, _ = masked_smooth_l1(out0, disparity,
                                            max_disparity=max_disparity)
                record["batch0_forward"] = {
                    "loss": float(loss0),
                    "out_sha": sha(out0), "out_mean": float(out0.mean()),
                    "out_std": float(out0.std()), "out_min": float(out0.min()),
                    "out_max": float(out0.max()),
                }
                # Backward-determinism probe: same graph twice, gradients compared.
                grads = []
                for _ in range(2):
                    model.zero_grad(set_to_none=True)
                    o = model(left, right)
                    l, _n = masked_smooth_l1(o, disparity,
                                             max_disparity=max_disparity)
                    l.backward()
                    grads.append(torch.cat([p.grad.detach().reshape(-1).cpu()
                                            for p in model.parameters()
                                            if p.grad is not None]))
                record["batch0_same_process_backward_repeat"] = {
                    "max_abs_diff": float((grads[0] - grads[1]).abs().max()),
                    "bitwise_identical": bool(torch.equal(grads[0], grads[1])),
                    "grad_sha_1": sha(grads[0]), "grad_sha_2": sha(grads[1]),
                    "grad_norm_1": float(grads[0].norm()),
                    "grad_norm_2": float(grads[1].norm()),
                    "n_elements_differing": int((grads[0] != grads[1]).sum()),
                }
                model.zero_grad(set_to_none=True)

            pred = model(left, right)
            loss, _ = masked_smooth_l1(pred, disparity, max_disparity=max_disparity)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                      max_norm=1e9))
            optimizer.step()

            if not np.isfinite(float(loss)) or not np.isfinite(gn):
                nan_or_inf = True
            grad_norms.append(gn)
            per_batch.append({"batch": batch_index, "loss": float(loss),
                              "grad_norm": gn})
    except RuntimeError as exc:
        # Expected path when --deterministic meets an op with no deterministic
        # implementation. Record the exact operation and stop cleanly.
        record["nondeterministic_op_error"] = "{}: {}".format(type(exc).__name__, exc)
        (out / "record.json").write_text(json.dumps(record, indent=2),
                                         encoding="utf-8")
        print("HALTED:", record["nondeterministic_op_error"])
        print("wrote", out / "record.json")
        return

    scheduler.step()
    record["epoch"] = {
        "batches": len(per_batch),
        "first_batch_loss": per_batch[0]["loss"],
        "last_batch_loss": per_batch[-1]["loss"],
        "mean_loss": float(np.mean([b["loss"] for b in per_batch])),
        "median_grad_norm": float(np.median(grad_norms)),
        "max_grad_norm": float(np.max(grad_norms)),
        "any_nan_or_inf": bool(nan_or_inf),
        "lr_after_step": float(scheduler.get_last_lr()[0]),
        "weights_after_epoch_sha": sha(torch.cat(
            [p.detach().reshape(-1).cpu() for p in model.parameters()])),
        "train_seconds": time.time() - t0,
    }
    record["per_batch"] = per_batch
    record["data_order"] = [
        {k: e[k] for k in ("dataset_index", "scene_name", "crop_y", "crop_x",
                           "jitter_left", "jitter_right", "left_sha")}
        for e in dataset.log]

    v1 = validate(model, val_base, device, limit=10)
    v2 = validate(model, val_base, device, limit=10)
    record["validation"] = {
        "protocol": "first 10 hailo_val scenes, full 368x1232, pooled over gt > 0",
        "run1": {"epe": v1.epe, "d1": v1.d1},
        "run2": {"epe": v2.epe, "d1": v2.d1},
        "repeat_identical": bool(v1.epe == v2.epe and v1.d1 == v2.d1),
        "repeat_delta_epe": float(v1.epe - v2.epe),
        "repeat_delta_d1": float(v1.d1 - v2.d1),
    }
    record["rng_at_end"] = rng_snapshot(device)
    record["total_seconds"] = time.time() - t0

    (out / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("wrote", out / "record.json")
    print("batch0 left sha  ", record["batch0_inputs"]["left_sha"])
    print("batch0 loss      ", record["batch0_forward"]["loss"])
    print("fwd repeat ident ", record["batch0_same_process_forward_repeat"]["bitwise_identical"])
    print("bwd repeat ident ", record["batch0_same_process_backward_repeat"]["bitwise_identical"])
    print("epoch mean loss  ", record["epoch"]["mean_loss"])
    print("val EPE / D1     ", v1.epe, v1.d1)


if __name__ == "__main__":
    main()
