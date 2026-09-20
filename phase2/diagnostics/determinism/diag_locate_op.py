"""STAGE A -- DIAGNOSTIC ONLY. Names the nondeterministic operation(s).

Two independent identifications, neither of which changes the model:

  1. `torch.use_deterministic_algorithms(True, warn_only=True)` around one
     forward+backward. PyTorch emits a warning naming every kernel that has no
     deterministic implementation. The warnings are captured verbatim.

  2. A per-stage repeat probe. The same batch is pushed through the model twice
     with default flags and the gradient of each *stage* is compared, so the
     divergence can be attributed to a part of the network rather than asserted.

    python phase2/diagnostics/determinism/diag_locate_op.py --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
import torch.nn.functional as F                               # noqa: E402

from src.datasets.kitti2015 import Kitti2015Stereo            # noqa: E402
from src.losses.disparity import masked_smooth_l1             # noqa: E402
from phase2.scripts.exp_h1_cost_volume import CroppedKitti     # noqa: E402
from phase2.scripts.exp_h2_seed_replication import build_model  # noqa: E402

SEED = 0
STAGES = ("feature_extractor", "aggregation", "refinement")


def one_batch(device):
    base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_calib")
    ds = CroppedKitti(base, seed=SEED)
    left0, right0, disp0 = ds[111]
    left1, right1, disp1 = ds[54]
    left = torch.stack([left0, left1]).to(device)
    right = torch.stack([right0, right1]).to(device)
    disp = torch.stack([disp0, disp1]).to(device)
    return left, right, disp


def grads_by_stage(model):
    out = {}
    for stage in STAGES:
        mod = getattr(model, stage)
        gs = [p.grad.detach().reshape(-1).cpu() for p in mod.parameters()
              if p.grad is not None]
        out[stage] = torch.cat(gs) if gs else torch.zeros(0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    record = {"device": device, "torch": torch.__version__,
              "cudnn_version": torch.backends.cudnn.version()}

    model = build_model(SEED, device)
    model.train()
    left, right, disp = one_batch(device)
    max_disp = float(model.config.max_disparity_px)

    # --- 1. warn_only pass: PyTorch names the offending kernels ------------
    torch.use_deterministic_algorithms(True, warn_only=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.zero_grad(set_to_none=True)
        pred = model(left, right)
        loss, _ = masked_smooth_l1(pred, disp, max_disparity=max_disp)
        loss.backward()
        torch.cuda.synchronize() if device == "cuda" else None
    record["warn_only_warnings"] = sorted({str(w.message) for w in caught})
    torch.use_deterministic_athlgorithms = None  # noqa: F841  (kept out of the way)
    torch.use_deterministic_algorithms(False)

    # --- 2. per-stage repeat probe, default flags ---------------------------
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    runs = []
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        pred = model(left, right)
        loss, _ = masked_smooth_l1(pred, disp, max_disparity=max_disp)
        loss.backward()
        runs.append({"loss": float(loss), "pred_sum": float(pred.sum()),
                     "grads": grads_by_stage(model)})
    record["forward_loss_repeat"] = {
        "loss_1": runs[0]["loss"], "loss_2": runs[1]["loss"],
        "identical": runs[0]["loss"] == runs[1]["loss"]}
    record["stage_gradient_repeat"] = {}
    for stage in STAGES:
        g1, g2 = runs[0]["grads"][stage], runs[1]["grads"][stage]
        record["stage_gradient_repeat"][stage] = {
            "elements": int(g1.numel()),
            "bitwise_identical": bool(torch.equal(g1, g2)),
            "elements_differing": int((g1 != g2).sum()),
            "max_abs_diff": float((g1 - g2).abs().max()) if g1.numel() else 0.0,
            "relative_max_diff": (float((g1 - g2).abs().max() / (g1.abs().max() + 1e-30))
                                  if g1.numel() else 0.0),
        }

    # --- 3. isolate the suspected kernel: bilinear interpolate backward ----
    # The regression stage upsamples the (B, 12, 23, 77) cost tensor to full
    # resolution with F.interpolate(mode="bilinear"). Its CUDA backward scatters
    # with atomicAdd. Tested here in isolation, on the model's own tensor shape.
    iso = {}
    torch.manual_seed(0)
    x = torch.randn(2, 12, 23, 77, device=device, requires_grad=True)
    gs = []
    for _ in range(2):
        if x.grad is not None:
            x.grad = None
        y = F.interpolate(x, size=(256, 512), mode="bilinear", align_corners=True)
        y.pow(2).sum().backward()
        gs.append(x.grad.detach().clone())
    iso["interpolate_bilinear_backward"] = {
        "shape_in": list(x.shape), "shape_out": [256, 512],
        "bitwise_identical": bool(torch.equal(gs[0], gs[1])),
        "max_abs_diff": float((gs[0] - gs[1]).abs().max()),
        "elements_differing": int((gs[0] != gs[1]).sum()),
    }
    # Same op with determinism controls on.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    gs = []
    for _ in range(2):
        if x.grad is not None:
            x.grad = None
        y = F.interpolate(x, size=(256, 512), mode="bilinear", align_corners=True)
        y.pow(2).sum().backward()
        gs.append(x.grad.detach().clone())
    iso["interpolate_bilinear_backward_deterministic"] = {
        "bitwise_identical": bool(torch.equal(gs[0], gs[1])),
        "max_abs_diff": float((gs[0] - gs[1]).abs().max()),
    }
    torch.use_deterministic_algorithms(False)
    record["isolated_kernel_probe"] = iso

    (out_dir / "locate_op.json").write_text(json.dumps(record, indent=2),
                                            encoding="utf-8")
    print(json.dumps({k: record[k] for k in
                      ("warn_only_warnings", "forward_loss_repeat",
                       "stage_gradient_repeat", "isolated_kernel_probe")},
                     indent=2))
    print("wrote", out_dir / "locate_op.json")


if __name__ == "__main__":
    main()
