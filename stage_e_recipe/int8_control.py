#!/usr/bin/env python
"""Stage E INT8 numerical-survivability gate — control measurement.

For each scored checkpoint: export to ONNX, quantize to INT8 with the frozen
EXP-015 procedure, score both through the frozen contract, and record

    P = EPE_int8 - EPE_fp32

**What this is NOT**: Hailo validation, Hailo compatibility, HEF validation, or
hardware validation. The historical EXP-015 figure (int8 1.655 vs fp32 1.313)
was ONNX Runtime CPU quantization of the *reference* model - not ARM-P, not
Hailo. Nothing here bears on Stage D, which remains blocked.

Definition of the fp32 term, fixed here before any candidate is measured: P is
the **fp32 ONNX -> int8 ONNX** difference, both scored in the same runtime on
the same graph, so P isolates quantization. Using the PyTorch score as the fp32
term would fold in the export-parity gap that C1 measured at 0.001708984375 px,
which is an export property, not a quantization one. The PyTorch score is
recorded alongside for reference.

Runs on CPU. Every candidate must be measured by this same script on the same
machine, or the comparison against P_control is not like-for-like.

    python stage_e_recipe/int8_control.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from phase1.harness.frozen_eval import pooled_metrics, refuse_unless_contract  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

CFG = dict(downsample_levels=3, num_disparities=24,
           cost_volume_shift="right", regression_normalize=True)
H, W = 368, 1232
CALIB_N = 32                     # pre-registered (A1.3), EXP-015's own default
EXPECTED_PARAMS = 397954
EXP = sys.argv[1] if len(sys.argv) > 1 else "e0"
WORK = HERE / f"int8_{EXP}"

# The gate is experiment-agnostic: every candidate (E1, E2, E3, and E4 with
# its broader-pretrain init) is measured by this same script against the
# P_control reference below. Only e0 establishes P_control.
SUPPORTED_EXPS = ("e0", "e1", "e2", "e3", "e4")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_onnx(ckpt: Path, dst: Path) -> dict:
    import onnx
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model = StereoNet(StereoNetConfig(**CFG))
    model.load_state_dict(state, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == EXPECTED_PARAMS, f"param count {n_params}"
    dummy_l = torch.zeros(1, 3, H, W)
    dummy_r = torch.zeros(1, 3, H, W)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, (dummy_l, dummy_r), str(dst),
                          input_names=["left", "right"], output_names=["disparity"],
                          opset_version=13, do_constant_folding=True, dynamo=False)
    g = onnx.load(str(dst))
    onnx.checker.check_model(g)
    return {"path": str(dst.relative_to(REPO)).replace("\\", "/"),
            "sha256": sha256(dst), "bytes": dst.stat().st_size,
            "opset": int(g.opset_import[0].version), "n_nodes": len(g.graph.node),
            "parent_checkpoint_sha256": sha256(ckpt), "params": n_params}


def quantize(src: Path, dst: Path, calib_ds) -> dict:
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]

    class Reader(CalibrationDataReader):
        def __init__(self, ds, count):
            self.items = []
            for i in range(min(count, len(ds))):
                s = ds[i]
                self.items.append({names[0]: normalize(s.left),
                                   names[1]: normalize(s.right)})
            self.it = iter(self.items)

        def get_next(self):
            return next(self.it, None)

    t0 = time.time()
    quantize_static(str(src), str(dst), Reader(calib_ds, CALIB_N),
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=True)
    return {"path": str(dst.relative_to(REPO)).replace("\\", "/"),
            "sha256": sha256(dst), "bytes": dst.stat().st_size,
            "parent_onnx_sha256": sha256(src),
            "procedure": "quantize_static, QDQ, QInt8 act, QInt8 weight, per_channel",
            "calibration": f"first {CALIB_N} scenes of hailo_calib",
            "quantize_s": round(time.time() - t0, 1)}


def score_onnx_model(path: Path, ds) -> dict:
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    ins = [i.name for i in sess.get_inputs()]
    out_name = sess.get_outputs()[0].name
    preds, gts = [], []
    nonfinite = 0
    t0 = time.time()
    for i in range(len(ds)):
        s = ds[i]
        out = sess.run([out_name], {ins[0]: normalize(s.left),
                                    ins[1]: normalize(s.right)})[0]
        pred = out[0, 0].astype(np.float64)
        nonfinite += int((~np.isfinite(pred)).sum())
        valid = s.disparity > 0
        preds.append(pred[valid].astype(np.float64))
        gts.append(s.disparity[valid].astype(np.float64))
    m = pooled_metrics(preds, gts)
    guard = refuse_unless_contract(len(ds), int(sum(p.size for p in preds)),
                                   256.0, "hailo_val", "disp_occ_0")
    return {"metrics": m, "guard": guard, "nonfinite_pixels": nonfinite,
            "score_s": round(time.time() - t0, 1)}


def score_torch(ckpt: Path, ds) -> dict:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = StereoNet(StereoNetConfig(**CFG))
    model.load_state_dict(blob["model"], strict=True)
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            out = model(torch.from_numpy(normalize(s.left)),
                        torch.from_numpy(normalize(s.right)))
            pred = out[0, 0].numpy().astype(np.float64)
            valid = s.disparity > 0
            preds.append(pred[valid].astype(np.float64))
            gts.append(s.disparity[valid].astype(np.float64))
    return {"metrics": pooled_metrics(preds, gts)}


def main() -> None:
    if EXP not in SUPPORTED_EXPS:
        sys.exit(f"unknown experiment {EXP!r}; expected one of {SUPPORTED_EXPS}")
    targets = sorted(
        (int(p.parent.name.replace("seed", "")), p)
        for p in (HERE / "kaggle" / f"{EXP}_output").glob(f"seed*/{EXP}_seed*_best.pth"))
    if not targets:
        sys.exit(f"no {EXP.upper()} best checkpoints found")

    val = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                          disparity_scale=256.0, occluded=True)
    calib = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_calib")
    WORK.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "gate": f"Stage E INT8 numerical-survivability - {EXP.upper()}",
        "experiment": EXP,
        "not_hailo": ("NOT Hailo validation, compatibility, HEF validation or "
                      "hardware validation. Stage D remains blocked."),
        "P_definition": "EPE(int8 ONNX) - EPE(fp32 ONNX), same runtime and graph",
        "calibration": f"first {CALIB_N} scenes of hailo_calib; hailo_val never used",
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__},
        "seeds": {},
    }
    try:
        import onnxruntime as ort
        report["environment"]["onnxruntime"] = ort.__version__
    except Exception as e:
        sys.exit(f"INT8 gate NOT MEASURABLE: {e}")

    penalties = []
    for seed, ckpt in targets:
        print(f"--- seed {seed} ---", flush=True)
        f32 = WORK / f"{EXP}_seed{seed}_best_fp32.onnx"
        i8 = WORK / f"{EXP}_seed{seed}_best_int8.onnx"
        rec: dict = {"checkpoint": str(ckpt.relative_to(REPO)).replace("\\", "/")}
        rec["export"] = export_onnx(ckpt, f32)
        print(f"  exported {rec['export']['n_nodes']} nodes", flush=True)
        rec["quantize"] = quantize(f32, i8, calib)
        print(f"  quantized in {rec['quantize']['quantize_s']}s", flush=True)
        rec["fp32_onnx"] = score_onnx_model(f32, val)
        rec["int8_onnx"] = score_onnx_model(i8, val)
        rec["fp32_torch"] = score_torch(ckpt, val)
        e32 = rec["fp32_onnx"]["metrics"]["epe"]
        e8 = rec["int8_onnx"]["metrics"]["epe"]
        rec["P"] = e8 - e32
        rec["export_parity_delta"] = e32 - rec["fp32_torch"]["metrics"]["epe"]
        penalties.append(rec["P"])
        print(f"  fp32 ONNX {e32:.7f} | int8 {e8:.7f} | P = {rec['P']:.7f} px "
              f"| nonfinite {rec['int8_onnx']['nonfinite_pixels']}", flush=True)
        report["seeds"][str(seed)] = rec

    key = "P_control" if EXP == "e0" else "P_candidate"
    report[f"{key}_per_seed"] = penalties
    report[key] = float(np.mean(penalties))
    report[f"{key}_spread"] = float(max(penalties) - min(penalties))
    if EXP == "e0":
        report["gate_for_candidates"] = (
            f"P_candidate <= P_control + 0.05 = {report[key] + 0.05:.7f} px")
    else:
        P_CONTROL = 5.5865268          # measured; stage_e_recipe/int8_control/
        report["P_control_reference"] = P_CONTROL
        report["gate_threshold"] = P_CONTROL + 0.05
        report["int8_gate"] = "PASS" if report[key] <= P_CONTROL + 0.05 else "FAIL"
    out = WORK / f"int8_{EXP}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n{key} = {report[key]:.7f} px (spread {report[f'{key}_spread']:.7f})")
    if EXP == "e0":
        print(f"candidate gate: {report['gate_for_candidates']}")
    else:
        print(f"gate: {report[key]:.7f} <= {report['gate_threshold']:.7f} -> "
              f"{report['int8_gate']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
