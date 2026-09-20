"""CLI: re-score a checkpoint through the frozen-contract evaluator.

Single-line use:
  python phase1\\harness\\rescore.py --target onnx
  python phase1\\harness\\rescore.py --target convergence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase1" / "harness"))

from frozen_eval import score_checkpoint, score_onnx  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["onnx", "convergence"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()
    if args.target == "onnx":
        res = score_onnx(limit=args.limit, provider=args.provider)
    else:
        res = score_checkpoint(limit=args.limit)
    m = res["metrics"]
    print("target      {}".format(args.target))
    print("checkpoint  {}  sha256 {:.16s}...".format(res["checkpoint"], res["sha256"]))
    print("scenes      {}".format(res["scenes"]))
    print("EPE         {:.7f}".format(m["epe"]))
    print("D1          {:.7f}".format(m["d1"]))
    print("RMSE        {:.7f}".format(m["rmse"]))
    print("BAD1        {:.7f}".format(m["bad1"]))
    print("BAD2        {:.7f}".format(m["bad2"]))
    print("BAD3        {:.7f}".format(m["bad3"]))
    print("pixels      {}".format(m["valid_pixels"]))
    print("guard       {}".format(json.dumps(res["guard"])))
    if "compat" in res:
        print("compat      {}".format(json.dumps(res["compat"])))
    out = REPO_ROOT / "phase1" / "harness" / ("rescore_{}.json".format(args.target))
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print("wrote       {}".format(out))


if __name__ == "__main__":
    main()
