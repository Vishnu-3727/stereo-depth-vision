#!/usr/bin/env python
"""E2 feasibility probe: does a native physical batch of 8 fit on this GPU?

This is NOT a training experiment. It performs the minimum forward/backward
needed to establish memory feasibility and then stops:

  * no optimizer is constructed and no optimizer step is taken
  * no scheduler exists
  * no checkpoint is written
  * no weight is modified and nothing is persisted
  * the dataset is not read; a representative synthetic batch is used

It builds the frozen ARM-P architecture, loads the frozen Stage-1
initialization (read-only, strict), runs one forward and one backward at the
frozen training crop (256x512) for each batch size requested, and reports peak
CUDA memory.

    python stage_e_recipe/tests/batch8_memory_probe.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from src.losses.disparity import masked_smooth_l1  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

ARM_V_CONFIG = dict(downsample_levels=3, num_disparities=24,
                    cost_volume_shift="right", regression_normalize=True)
EXPECTED_PARAMS = 397954
INIT_REL = "stage_b_armp/20260918T062146Z_stage1_pretrain/checkpoints/armp_stage1_best.pth"
EXPECTED_INIT_SHA = "3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7"

# Frozen training crop and the loss's representable ceiling.
CROP_H, CROP_W = 256, 512
MAX_DISPARITY = 184.0


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(batch: int, device: torch.device) -> dict:
    """One forward + backward at `batch`. No optimizer step, ever."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = StereoNet(StereoNetConfig(**ARM_V_CONFIG)).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == EXPECTED_PARAMS, f"param count {n_params} != {EXPECTED_PARAMS}"

    init = REPO / INIT_REL
    got = sha256_file(init)
    assert got == EXPECTED_INIT_SHA, f"init hash {got} != {EXPECTED_INIT_SHA}"
    blob = torch.load(init, map_location=device, weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(sd, strict=True)

    # Representative synthetic batch at the frozen crop. Values in the range
    # ImageNet normalisation produces; disparity in the supervised band so the
    # loss has valid pixels and the backward graph is the real one.
    g = torch.Generator(device="cpu").manual_seed(0)
    left = torch.randn(batch, 3, CROP_H, CROP_W, generator=g).to(device)
    right = torch.randn(batch, 3, CROP_H, CROP_W, generator=g).to(device)
    disp = (torch.rand(batch, 1, CROP_H, CROP_W, generator=g) * 100.0 + 1.0).to(device)

    status, err = "FIT", None
    try:
        out = model(left, right)
        loss, n_valid = masked_smooth_l1(out, disp, max_disparity=MAX_DISPARITY)
        loss.backward()          # gradients exist; they are never applied
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        status, err = "OOM", str(e).splitlines()[0]
    except RuntimeError as e:
        status = "OOM" if "out of memory" in str(e).lower() else "ERROR"
        err = str(e).splitlines()[0]

    peak = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    result = {"batch": batch, "status": status,
              "peak_allocated_MiB": round(peak, 1),
              "peak_reserved_MiB": round(reserved, 1),
              "valid_pixels": int(n_valid) if status == "FIT" else None,
              "error": err}

    # Drop everything; no state survives this probe.
    del model, left, right, disp
    if status == "FIT":
        del out, loss
    torch.cuda.empty_cache()
    return result


def main() -> None:
    if not torch.cuda.is_available():
        sys.exit("CUDA required for the memory probe")
    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory / 2**20

    report = {
        "probe": "E2 native batch-8 memory feasibility",
        "not_a_training_run": True,
        "optimizer_step_taken": False,
        "checkpoint_written": False,
        "weights_modified": False,
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_MiB": round(total, 1),
        "torch_version": torch.__version__,
        "crop": [CROP_H, CROP_W],
        "param_count": EXPECTED_PARAMS,
        "init_sha256": EXPECTED_INIT_SHA,
        "results": [],
    }
    for batch in (2, 8):
        r = probe(batch, device)
        report["results"].append(r)
        print(f"batch {r['batch']}: {r['status']:<5} "
              f"peak_alloc {r['peak_allocated_MiB']:>9.1f} MiB  "
              f"peak_reserved {r['peak_reserved_MiB']:>9.1f} MiB"
              + (f"  [{r['error']}]" if r["error"] else ""))

    b2 = next(r for r in report["results"] if r["batch"] == 2)
    b8 = next(r for r in report["results"] if r["batch"] == 8)
    headroom = (total - b8["peak_reserved_MiB"]) / total if b8["status"] == "FIT" else 0.0
    report["batch8_fits"] = b8["status"] == "FIT"
    report["batch8_headroom_frac"] = round(headroom, 3)
    # "Comfortable" is pre-registered here, before the number is known: the
    # batch-8 peak reserve must leave at least 20% of the card free, so a
    # 200-epoch run is not living on the edge of an OOM.
    report["comfortable_threshold_frac"] = 0.20
    report["verdict"] = (
        "NATIVE_BATCH_8" if report["batch8_fits"] and headroom >= 0.20
        else "ACCUMULATION_FALLBACK")
    report["scaling_vs_batch2"] = (
        round(b8["peak_allocated_MiB"] / b2["peak_allocated_MiB"], 2)
        if b2["status"] == "FIT" and b8["status"] == "FIT" else None)

    out = HERE.parent / "e2_batch8" / "batch8_memory_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nbatch8_fits={report['batch8_fits']} "
          f"headroom={report['batch8_headroom_frac']:.1%} "
          f"(threshold 20%) -> {report['verdict']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
