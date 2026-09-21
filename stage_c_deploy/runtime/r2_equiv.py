#!/usr/bin/env python
"""R2 equivalence / determinism check for the inference-stage variants.

Dumps the disparity of the first N hailo_val scenes for one runtime
configuration, so two dumps can be compared bit-for-bit:

  * equivalence  -- a variant's dump against the control's dump
  * determinism  -- two dumps of the same configuration, made by two
                    separate processes

Usage:
    python r2_equiv.py dump  <out_dir> [--channels-last] [--scenes 10]
    python r2_equiv.py cmp   <dir_a> <dir_b>

The dumped disparity comes from the same code path the benchmark times:
concurrent decode, GPU normalize, optional channels_last, cudnn.benchmark
off. No frozen artefact is read or written; the checkpoint is only read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import runtime_path as RP  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(HERE.parent / "metric_depth"))
import armp_depth as AD  # noqa: E402

from src.datasets.kitti2015 import Kitti2015Stereo, pad_and_crop  # noqa: E402

REPO = HERE.parents[1]


def dump(out_dir: Path, channels_last: bool, scenes: int) -> None:
    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    torch.backends.cudnn.benchmark = False
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val")
    names = list(ds.names[:scenes])
    net, dev = AD.load_frozen_net("cuda")
    net.eval()
    if channels_last:
        net = RP.apply_channels_last(net)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        left, right = RP.decode_pair(ds.root / "training" / "image_2" / name,
                                     ds.root / "training" / "image_3" / name,
                                     concurrent=True)
        tl = RP.normalize_gpu(pad_and_crop(left), dev)
        tr = RP.normalize_gpu(pad_and_crop(right), dev)
        if channels_last:
            tl = tl.to(memory_format=torch.channels_last)
            tr = tr.to(memory_format=torch.channels_last)
        with torch.no_grad():
            disp = net(tl, tr)[0, 0].detach().float().cpu().numpy()
        np.save(out_dir / f"{Path(name).stem}.npy", disp)
    (out_dir / "meta.json").write_text(json.dumps({
        "channels_last": channels_last, "scenes": scenes, "names": names,
        "checkpoint_sha256": AD.ARMP_SHA, "torch": torch.__version__,
        "device_name": torch.cuda.get_device_name(0), "pid": __import__("os").getpid(),
    }, indent=2))
    print(f"dumped {len(names)} disparities to {out_dir}")


def cmp_dirs(a: Path, b: Path) -> None:
    files = sorted(p.name for p in a.glob("*.npy"))
    assert files, f"no dumps in {a}"
    worst, worst_name, exact = 0.0, None, 0
    for f in files:
        da, db = np.load(a / f), np.load(b / f)
        d = float(np.max(np.abs(da.astype(np.float64) - db.astype(np.float64))))
        exact += int(d == 0.0)
        if d > worst:
            worst, worst_name = d, f
    print(json.dumps({"dir_a": str(a), "dir_b": str(b), "scenes": len(files),
                      "bit_identical_scenes": exact,
                      "max_abs_disparity_diff_px": worst,
                      "worst_scene": worst_name}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("out_dir")
    d.add_argument("--channels-last", action="store_true")
    d.add_argument("--scenes", type=int, default=10)
    c = sub.add_parser("cmp")
    c.add_argument("dir_a")
    c.add_argument("dir_b")
    args = ap.parse_args()
    if args.cmd == "dump":
        dump(Path(args.out_dir), args.channels_last, args.scenes)
    else:
        cmp_dirs(Path(args.dir_a), Path(args.dir_b))


if __name__ == "__main__":
    main()
