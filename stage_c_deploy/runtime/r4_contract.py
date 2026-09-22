"""R4 40-scene contract score of the recommended fast runtime path.

Scores the recommended deterministic config (cuda, fp32, concurrent-cv2
decode, GPU normalize, frozen net, cudnn.benchmark OFF) over the full
frozen 40-scene hailo_val contract, with an opt-in --checkpoint override.
Geometry stages are not run here: they cannot move the disparity (README
sections 4-5 establish decode/normalize/geometry equivalence), so the
contract score exercises decode + normalize + inference only.

Writes JSON in the exact schema of out/fast_path_40scene_cuda.json.

Usage:
    python stage_c_deploy/runtime/r4_contract.py --out <json>
    python stage_c_deploy/runtime/r4_contract.py --checkpoint <path> --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stage_c_deploy" / "metric_depth"))
sys.path.insert(0, str(REPO / "stage_c_deploy" / "runtime"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import armp_depth as AD  # noqa: E402
from phase1.harness.frozen_eval import (  # noqa: E402
    pooled_metrics,
    refuse_unless_contract,
)
from runtime_path import (  # noqa: E402
    decode_pair,
    load_net,
    normalize_gpu,
    resolve_checkpoint,
)
from src.datasets.kitti2015 import (  # noqa: E402
    Kitti2015Stereo,
    pad_and_crop,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None,
                    help="opt-in checkpoint override: repo-relative or absolute "
                         "path. Absent = the frozen checkpoint.")
    ap.add_argument("--device", default=None,
                    help="cpu | cuda (default: cuda if available else cpu)")
    ap.add_argument("--out", required=True, help="JSON output path")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device not in ("cpu", "cuda"):
        print(f"unknown --device {device!r}", file=sys.stderr)
        sys.exit(2)
    if device == "cuda" and not torch.cuda.is_available():
        print("no CUDA device available for --device cuda", file=sys.stderr)
        sys.exit(2)

    ckpt, ckpt_label = resolve_checkpoint(args.checkpoint)
    if not ckpt.exists():
        print(f"checkpoint missing: {ckpt}", file=sys.stderr)
        sys.exit(2)
    ckpt_sha = AD.sha256_file(ckpt)

    net, dev = load_net(device, ckpt, ckpt_sha,
                      override=args.checkpoint is not None)
    net.eval()
    is_cuda = dev == "cuda"
    if is_cuda:
        torch.backends.cudnn.benchmark = False  # deterministic, as recommended

    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    assert len(ds) == 40, f"expected 40 scenes, got {len(ds)} -> STOP"

    t0 = time.time()
    preds, gts = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            name = ds.names[i]
            root = ds.root
            left_u8, right_u8 = decode_pair(
                root / "training" / "image_2" / name,
                root / "training" / "image_3" / name, concurrent=True)
            left_u8 = pad_and_crop(left_u8)
            right_u8 = pad_and_crop(right_u8)
            sm = ds[i]  # GT + names; left/right already cropped in the dataset
            if is_cuda:
                tl = normalize_gpu(left_u8, dev)
                tr = normalize_gpu(right_u8, dev)
            else:
                from src.datasets.kitti2015 import normalize as _normalize
                tl = torch.from_numpy(_normalize(left_u8)).to(dev)
                tr = torch.from_numpy(_normalize(right_u8)).to(dev)
            out = net(tl, tr)
            disp = out[0, 0].detach().float().cpu().numpy().astype(np.float64)
            valid = sm.disparity > 0
            preds.append(disp[valid])
            gts.append(sm.disparity[valid].astype(np.float64))
            if (i + 1) % 10 == 0:
                print(f"scene {i + 1}/40 done", flush=True)
    if is_cuda:
        torch.cuda.synchronize()
    m = pooled_metrics(preds, gts)
    guard = refuse_unless_contract(len(ds), m["valid_pixels"], 256.0,
                                   "hailo_val", "disp_occ_0")
    elapsed = time.time() - t0

    device_name = torch.cuda.get_device_name(0) if is_cuda else "cpu"
    payload = {
        "device": dev,
        "device_name": device_name,
        "dtype": "fp32",
        "checkpoint": ckpt_label,
        "checkpoint_sha256": ckpt_sha,
        "runtime_path": {
            "decode": "par",
            "norm": "gpu",
            "gpu_geometry": True,
            "cudnn_bench": False,
            "deterministic_default": True,
        },
        "contract": {
            "split": "hailo_val",
            "gt_source": "disp_occ_0",
            "gt_scale": 256.0,
            "resolution": [368, 1232],
            "reference_valid_pixels": 3802797,
        },
        "scenes": 40,
        "scene_names": list(ds.names),
        "metrics": {
            "valid_pixels": m["valid_pixels"],
            "epe": m["epe"],
            "rmse": m["rmse"],
            "d1": m["d1"],
            "bad1": m["bad1"],
            "bad2": m["bad2"],
            "bad3": m["bad3"],
        },
        "guard": guard,
        "note": ("R4 measurement of the recommended fast runtime path; "
                 "does NOT replace any frozen record."),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"checkpoint: {ckpt_label} sha256: {ckpt_sha}")
    print(f"EPE={m['epe']:.10f} D1={m['d1']:.6f} RMSE={m['rmse']:.6f} "
          f"valid={m['valid_pixels']} contract_match={guard['contract_match']} "
          f"({elapsed:.1f} s)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
