"""C1 parity-failure diagnostic (ZERO TRAINING, diagnosis only).

Reads the two frozen checkpoints + two frozen ONNX artifacts, runs PyTorch
(CPU, eval, no_grad) vs ONNX Runtime (CPUExecutionProvider) on the SAME 5
hailo_val scenes, and writes raw numbers to JSON + top-pixel CSVs.

Creates ONLY files under stage_c_deploy/diagnostics/:
  c1_parity_diagnostic.json, top_pixels_armp.csv, top_pixels_p2a.csv,
  *_instrumented_debug.onnx (debug copies, never deployment artifacts).
The markdown report is written separately by the analyst, not by this script.

Hard rules honored here: no training, no weight/architecture change, no
tolerance change, no modification of any existing repo file, no Hailo work.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx import helper, shape_inference

from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

# ---------------------------------------------------------------- provenance
EXPECTED = {
    "armp_ckpt": ("stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth",
                  "b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454"),
    "p2a_ckpt": ("phase2/runs/p2a_scale_coverage/p2a_best.pth",
                 "0868ffd137a9985306bf5563685fd2362799bdd630d313e1181c980a7fbb6033"),
    "armp_onnx": ("stage_c_deploy/armp_stereonet.onnx",
                  "4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989"),
    "p2a_onnx": ("phase2/deploy/p2a_stereonet.onnx",
                 "0995a6c5722b46250299653e9edaa69ece17c6b86303122345ebf958768bfca9"),
}

CFG = dict(downsample_levels=3, num_disparities=24, cost_volume_shift="right",
           regression_normalize=True)
H, W = 368, 1232
TOL = 1e-3  # frozen gate, restated only; never redefined here
BINS = [1e-4, 5e-4, 1e-3, 1.5e-3, 1.7e-3]
PCTS = [99.0, 99.9, 99.99, 99.999]

# Predeclared materiality criterion for stage localisation (stated BEFORE numbers):
# a stage is MATERIAL iff max|d| > 1e-6 AND (max|d| / max|value|) > 1e-7.
# The first materially divergent stage is the earliest in forward order meeting both.
MAT_ABS = 1e-6
MAT_REL = 1e-7
STAGE_ORDER = ["left_features", "right_features", "cost_volume",
               "aggregated_cost", "disparity_initial",
               "refinement_residual", "disparity_final"]
STAGE_SHAPES = {
    "left_features": [1, 32, 46, 154],
    "right_features": [1, 32, 46, 154],
    "cost_volume": [1, 32, 24, 46, 154],
    "aggregated_cost": [1, 24, 46, 154],
    "disparity_initial": [1, 1, 368, 1232],
    "refinement_residual": [1, 1, 368, 1232],
    "disparity_final": [1, 1, 368, 1232],
}

OUT = REPO / "stage_c_deploy" / "diagnostics"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gradmag(a: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(a.astype(np.float64))
    return np.sqrt(gx * gx + gy * gy)


def border_dist(ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    return np.minimum(np.minimum(ys, xs),
                      np.minimum(H - 1 - ys, W - 1 - xs)).astype(np.float64)


def tie_stats(cost_lowres: torch.Tensor):
    """Reproduce regression.py exactly: interpolate cost to full res
    (bilinear, align_corners=True), optional standardisation across dim=1,
    softmax(-cost); return per-full-res-pixel top1, top2, gap, entropy."""
    up = F.interpolate(cost_lowres, size=(H, W), mode="bilinear", align_corners=True)
    if CFG["regression_normalize"]:
        up = (up - up.mean(dim=1, keepdim=True)) / (up.std(dim=1, keepdim=True) + 1e-6)
    w = torch.softmax(-up, dim=1)  # (1, 24, H, W)
    top2v, _ = torch.topk(w, 2, dim=1)
    t1 = top2v[:, 0:1].numpy().astype(np.float64)
    t2 = top2v[:, 1:2].numpy().astype(np.float64)
    ent = (-(w * torch.log(w + 1e-12)).sum(dim=1, keepdim=True)).numpy().astype(np.float64)
    return t1, t2, (t1 - t2), ent


def summarize(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64).ravel()
    return {"n": int(a.size), "mean": float(a.mean()),
            "median": float(np.median(a)), "max": float(a.max()),
            "p90": float(np.percentile(a, 90)) if a.size else None}


def build_instrumented(onnx_path: Path, out_path: Path) -> dict:
    """Shape-inference-guided debug copy. Returns candidate listing."""
    model = onnx.load(str(onnx_path))
    inferred = shape_inference.infer_shapes(model, strict_mode=False)
    shp: dict[str, list | None] = {}
    for vi in list(inferred.graph.input) + list(inferred.graph.output) + \
            list(inferred.graph.value_info):
        dims = []
        try:
            for d in vi.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.HasField("dim_value") else None)
        except Exception:
            dims = None
        shp[vi.name] = dims
    cands: dict[str, list] = {k: [] for k in STAGE_SHAPES}
    for idx, node in enumerate(inferred.graph.node):
        for oname in node.output:
            s = shp.get(oname)
            if not s or any(v is None for v in s):
                continue
            s = [int(v) for v in s]
            # NOTE: candidate lists are INDEPENDENT per stage (a tensor name
            # may appear under every stage whose shape it matches). A shared
            # dedupe across stages would silently starve later stages.
            for stage, want in STAGE_SHAPES.items():
                if s == want and all(c["name"] != oname for c in cands[stage]):
                    cands[stage].append({"name": oname, "node_idx": idx,
                                         "op": node.op_type})
    existing = {o.name for o in model.graph.output}
    added = []
    for stage, lst in cands.items():
        for c in lst:
            if c["name"] not in existing:
                model.graph.output.append(helper.make_tensor_value_info(
                    c["name"], onnx.TensorProto.FLOAT, STAGE_SHAPES[stage]))
                existing.add(c["name"])
                added.append(c["name"])
    onnx.save(model, str(out_path))
    return {"candidates": cands, "added_outputs": added,
            "debug_copy": out_path.name}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- 0. re-verify hashes; mismatch = STOP ----
    hashes = {}
    for key, (rel, exp) in EXPECTED.items():
        p = REPO / rel
        got = sha256(p)
        hashes[key] = {"path": rel, "sha256": got, "expected": exp,
                       "match": got.lower() == exp.lower()}
        if got.lower() != exp.lower():
            print(f"HASH MISMATCH {rel}: got {got} expected {exp} -> STOP",
                  flush=True)
            sys.exit(2)
    print("hashes OK: both checkpoints + both ONNX re-verified", flush=True)

    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    so_default = ort.SessionOptions()
    so_default.intra_op_num_threads = 1
    so_default.inter_op_num_threads = 1
    so_off = ort.SessionOptions()
    so_off.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so_off.intra_op_num_threads = 1
    so_off.inter_op_num_threads = 1

    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                  capture_output=True, text=True,
                                  timeout=15).stdout.strip()
    except Exception:
        git_head = None

    ds_occ = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                             disparity_scale=256.0, occluded=True)
    ds_noc = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                             disparity_scale=256.0, occluded=False)
    assert [ds_occ[i].name for i in range(5)] == \
        [f"00016{i}_10.png" for i in range(5)], "scene order changed -> STOP"

    models = {
        "armp": (REPO / EXPECTED["armp_ckpt"][0], REPO / EXPECTED["armp_onnx"][0]),
        "p2a": (REPO / EXPECTED["p2a_ckpt"][0], REPO / EXPECTED["p2a_onnx"][0]),
    }
    out: dict = {
        "provenance": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "git_head": git_head, "host": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "onnx": onnx.__version__, "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "ort_intra_threads": 1, "ort_inter_threads": 1,
            "checkpoint_hashes": {k: hashes[k] for k in ("armp_ckpt", "p2a_ckpt")},
            "onnx_hashes": {k: hashes[k] for k in ("armp_onnx", "p2a_onnx")},
            "config_both_models": CFG,
            "scenes": [ds_occ[i].name for i in range(5)],
            "note": "No existing repo artifact was modified; all outputs are "
                    "new files under stage_c_deploy/diagnostics/. "
                    "Instrumented ONNX copies are DEBUG copies, never "
                    "deployment artifacts.",
        },
        "materiality_criterion": {
            "rule": "MATERIAL iff max_abs_diff > 1e-6 AND relative "
                    "(max_abs_diff / max_abs_value) > 1e-7; first material "
                    "stage = earliest in forward order meeting both. "
                    "Stated BEFORE the numbers.",
            "mat_abs": MAT_ABS, "mat_rel": MAT_REL, "order": STAGE_ORDER,
        },
        "models": {},
    }

    for mname, (ckpt_p, onnx_p) in models.items():
        blob = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        net = StereoNet(StereoNetConfig(**CFG))
        net.load_state_dict(blob["model"], strict=True)
        net.eval().cpu()
        sess = ort.InferenceSession(str(onnx_p), sess_options=so_default,
                                    providers=["CPUExecutionProvider"])
        sess_off = ort.InferenceSession(str(onnx_p), sess_options=so_off,
                                        providers=["CPUExecutionProvider"])
        inames = [i.name for i in sess.get_inputs()]
        oname = sess.get_outputs()[0].name
        inames_off = [i.name for i in sess_off.get_inputs()]
        oname_off = sess_off.get_outputs()[0].name

        per_scene = []
        pool_abs, pool_pt, pool_gt = [], [], []
        pool_t1, pool_t2, pool_gap, pool_ent = [], [], [], []
        pool_bdist_1e3, pool_bdist_1e4 = [], []
        dgm_disc_1e3, dgm_all = [], []
        txm_disc_1e3, txm_all = [], []
        gt_disc, gt_all = [], []
        occ_disc_num = occ_disc_den = 0
        occ_all_num = occ_all_den = 0
        top_all: list[tuple] = []  # (absdiff, scene, y, x, ...)
        stage_num: dict[str, list] = {s: [] for s in STAGE_ORDER}  # max|d| per scene (filled later)
        stage_den: dict[str, list] = {s: [] for s in STAGE_ORDER}  # per-scene max|value|
        per_scene_max_default, per_scene_max_off = [], []
        tie_disc: dict[str, list] = {"t1": [], "t2": [], "gap": [], "ent": []}
        tie_ord: dict[str, list] = {"t1": [], "t2": [], "gap": [], "ent": []}

        for i in range(5):
            s = ds_occ[i]
            sn = ds_noc[i]
            assert s.name == sn.name
            L, R = normalize(s.left), normalize(s.right)
            tl, tr = torch.from_numpy(L), torch.from_numpy(R)
            with torch.no_grad():
                pt_out, stages = net(tl, tr, return_stages=True)
            pt = pt_out[0, 0].numpy()
            on = sess.run([oname], {inames[0]: L, inames[1]: R})[0][0, 0]
            on_off = sess_off.run([oname_off],
                                  {inames_off[0]: L, inames_off[1]: R})[0][0, 0]
            pt64 = pt.astype(np.float64)
            on64 = on.astype(np.float64)
            off64 = on_off.astype(np.float64)
            ad = np.abs(pt64 - on64)
            ad_off = np.abs(pt64 - off64)
            per_scene_max_default.append(float(ad.max()))
            per_scene_max_off.append(float(ad_off.max()))

            total = int(ad.size)
            bin_counts = {str(t): int((ad > t).sum()) for t in BINS}
            flat = ad.ravel()
            sc = {"scene": s.name,
                  "total_pixels": total,
                  "bin_counts": bin_counts,
                  "bin_fractions": {k: v / total for k, v in bin_counts.items()},
                  "max": float(ad.max()), "mean": float(ad.mean()),
                  "median": float(np.median(flat)),
                  "percentiles": {str(p): float(np.percentile(flat, p)) for p in PCTS},
                  "max_off_opt": float(ad_off.max()),
                  "mean_off_opt": float(ad_off.mean())}
            per_scene.append(sc)
            pool_abs.append(flat)

            # ---- spatial ----
            m1e3 = ad > 1e-3
            m1e4 = ad > 1e-4
            yy, xx = np.nonzero(m1e3)
            sc["n_above_1e3"] = int(m1e3.sum())
            sc["n_above_1e4"] = int(m1e4.sum())
            if yy.size:
                sc["border_dist_1e3"] = summarize(border_dist(yy, xx))
                pool_bdist_1e3.append(border_dist(yy, xx))
            else:
                sc["border_dist_1e3"] = None
            yy4, xx4 = np.nonzero(m1e4)
            if yy4.size:
                sc["border_dist_1e4"] = summarize(border_dist(yy4, xx4))
                pool_bdist_1e4.append(border_dist(yy4, xx4))
            else:
                sc["border_dist_1e4"] = None

            dgm = gradmag(pt64)
            gray = (0.299 * s.left[:, :, 0].astype(np.float64)
                    + 0.587 * s.left[:, :, 1].astype(np.float64)
                    + 0.114 * s.left[:, :, 2].astype(np.float64))
            txm = gradmag(gray)
            dgm_all.append(dgm.ravel())
            txm_all.append(txm.ravel())
            gt = s.disparity.astype(np.float64)
            gt_all.append(gt[gt > 0])
            occ = (s.disparity > 0) & ~(sn.disparity > 0)
            occ_all_num += int(occ.sum())
            occ_all_den += int((s.disparity > 0).sum())
            if yy.size:
                dgm_disc_1e3.append(dgm[m1e3])
                txm_disc_1e3.append(txm[m1e3])
                g = gt[m1e3]
                gt_disc.append(g)
                sc["gt_valid_frac_1e3"] = float((g > 0).sum() / g.size)
                sc["gt_at_1e3"] = summarize(g[g > 0]) if (g > 0).any() else None
                sc["dispgrad_at_1e3"] = summarize(dgm[m1e3])
                sc["texture_at_1e3"] = summarize(txm[m1e3])
                sc["occ_frac_1e3"] = float(occ[m1e3].sum() / m1e3.sum())
                occ_disc_num += int(occ[m1e3].sum())
                occ_disc_den += int(m1e3.sum())
            else:
                sc["gt_valid_frac_1e3"] = None
                sc["gt_at_1e3"] = None
                sc["dispgrad_at_1e3"] = None
                sc["texture_at_1e3"] = None
                sc["occ_frac_1e3"] = None

            # ---- tie stats (exact full-res reproduction of regression.py) ----
            with torch.no_grad():
                t1, t2, gap, ent = tie_stats(stages["aggregated_cost"].detach())
            t1 = t1[0, 0]
            t2 = t2[0, 0]
            gap = gap[0, 0]
            ent = ent[0, 0]
            pool_pt.append(pt64.ravel())
            pool_gt.append(gt.ravel())
            pool_t1.append(t1.ravel())
            pool_t2.append(t2.ravel())
            pool_gap.append(gap.ravel())
            pool_ent.append(ent.ravel())
            if yy.size:
                for k, arr in (("t1", t1), ("t2", t2), ("gap", gap), ("ent", ent)):
                    tie_disc[k].append(arr[m1e3])
            rng = np.random.default_rng(1000 + i)
            ord_idx = np.flatnonzero(~m1e4.ravel() if m1e4.any() else np.ones_like(flat, bool))
            take = rng.choice(ord_idx, size=min(1000, ord_idx.size), replace=False)
            tie_ord["t1"].append(t1.ravel()[take])
            tie_ord["t2"].append(t2.ravel()[take])
            tie_ord["gap"].append(gap.ravel()[take])
            tie_ord["ent"].append(ent.ravel()[take])

            # ---- top-pixel collection ----
            k = min(20, flat.size)
            idx = np.argpartition(flat, -k)[-k:]
            idx = idx[np.argsort(flat[idx])[::-1]]
            for j in idx:
                y, x = int(j // W), int(j % W)
                a = float(flat[j])
                pv, ov = float(pt64[y, x]), float(on64[y, x])
                top_all.append((a, s.name, y, x, pv, ov,
                                float(t1[y, x]), float(t2[y, x]),
                                float(gap[y, x]), float(ent[y, x]),
                                float(gt[y, x])))

        top_all.sort(key=lambda r: -r[0])
        top20 = top_all[:20]

        P = np.concatenate(pool_abs)
        PT = np.concatenate(pool_pt)
        T1 = np.concatenate(pool_t1)
        T2 = np.concatenate(pool_t2)
        GAP = np.concatenate(pool_gap)
        ENT = np.concatenate(pool_ent)
        total_px = int(P.size)
        mag = {"total_pixels": total_px,
               "bin_counts": {str(t): int((P > t).sum()) for t in BINS},
               "bin_fractions": {str(t): float((P > t).mean()) for t in BINS},
               "max": float(P.max()), "mean": float(P.mean()),
               "median": float(np.median(P)),
               "percentiles": {str(p): float(np.percentile(P, p)) for p in PCTS},
               "per_scene": per_scene,
               "bins_are_diagnostic_only": True}
        spat = {
            "n_above_1e3_pooled": int((P > 1e-3).sum()),
            "n_above_1e4_pooled": int((P > 1e-4).sum()),
            "border_dist_1e3": summarize(np.concatenate(pool_bdist_1e3))
            if pool_bdist_1e3 else None,
            "border_dist_1e4": summarize(np.concatenate(pool_bdist_1e4))
            if pool_bdist_1e4 else None,
            "dispgrad_disc_1e3": summarize(np.concatenate(dgm_disc_1e3))
            if dgm_disc_1e3 else None,
            "dispgrad_all": summarize(np.concatenate(dgm_all)),
            "texture_disc_1e3": summarize(np.concatenate(txm_disc_1e3))
            if txm_disc_1e3 else None,
            "texture_all": summarize(np.concatenate(txm_all)),
            "gt_valid_frac_disc_1e3": float(np.concatenate(gt_disc).ravel().__gt__(0).sum() / max(1, np.concatenate(gt_disc).size)) if gt_disc else None,
            "gt_at_disc_1e3": summarize(np.concatenate(gt_disc)[np.concatenate(gt_disc) > 0]) if gt_disc and (np.concatenate(gt_disc) > 0).any() else None,
            "gt_all_valid": summarize(np.concatenate(gt_all)),
            "occ_frac_disc_1e3": float(occ_disc_num / occ_disc_den) if occ_disc_den else None,
            "occ_frac_all_validgt": float(occ_all_num / occ_all_den) if occ_all_den else None,
        }
        dd = np.concatenate(tie_disc["t1"]) if tie_disc["t1"] else np.array([])
        ties = {
            "mapping": "cost upsampled with F.interpolate(bilinear, align_corners=True) "
                       "to (368,1232), then standardised across dim=1 (mean/std + 1e-6, "
                       "because regression_normalize=True) and softmax(-cost): exact "
                       "reproduction of src/models/stereonet/regression.py soft_argmin "
                       "at full resolution, so no approximate low-res cell mapping is used.",
            "pred_disparity_disc_1e3": summarize(PT[(P > 1e-3)]) if (P > 1e-3).any() else None,
            "pred_disparity_all": summarize(PT),
            "disc": {k: summarize(np.concatenate(tie_disc[k])) for k in tie_disc} if dd.size else None,
            "ordinary_sample": {k: summarize(np.concatenate(tie_ord[k])) for k in tie_ord},
            "n_disc": int(dd.size),
            "n_ordinary_sample": int(np.concatenate(tie_ord["t1"]).size),
        }

        # ---- determinism control (scene 0) ----
        s0 = ds_occ[0]
        L0, R0 = normalize(s0.left), normalize(s0.right)
        with torch.no_grad():
            a1 = net(torch.from_numpy(L0), torch.from_numpy(R0)).numpy()
            a2 = net(torch.from_numpy(L0), torch.from_numpy(R0)).numpy()
        b1 = sess.run([oname], {inames[0]: L0, inames[1]: R0})[0]
        b2 = sess.run([oname], {inames[0]: L0, inames[1]: R0})[0]
        det = {"pytorch_bit_identical": bool(np.array_equal(a1, a2)),
               "pytorch_max_abs_self_diff": float(np.abs(a1.astype(np.float64) - a2.astype(np.float64)).max()),
               "ort_bit_identical": bool(np.array_equal(b1, b2)),
               "ort_max_abs_self_diff": float(np.abs(b1.astype(np.float64) - b2.astype(np.float64)).max())}

        # ---- instrumented debug copy + stage localisation ----
        dbg_name = ("armp" if mname == "armp" else "p2a") + "_instrumented_debug.onnx"
        dbg_path = OUT / dbg_name
        loc_info = build_instrumented(onnx_p, dbg_path)
        isess = ort.InferenceSession(str(dbg_path), sess_options=so_default,
                                     providers=["CPUExecutionProvider"])
        i_in = [i.name for i in isess.get_inputs()]
        i_out = [o.name for o in isess.get_outputs()]
        # per-scene stage diffs pooled
        acc_max: dict[str, list] = {}
        acc_mean: dict[str, list] = {}
        acc_vmax: dict[str, list] = {}
        assign: dict[str, dict] = {}
        for i in range(5):
            s = ds_occ[i]
            L, R = normalize(s.left), normalize(s.right)
            with torch.no_grad():
                _, st = net(torch.from_numpy(L), torch.from_numpy(R),
                            return_stages=True)
            outs = isess.run(i_out, {i_in[0]: L, i_in[1]: R})
            omap = dict(zip(i_out, outs))
            pt_stage = {
                "left_features": st["left_features"].numpy().astype(np.float64),
                "right_features": st["right_features"].numpy().astype(np.float64),
                "cost_volume": st["cost_volume"].numpy().astype(np.float64),
                "aggregated_cost": st["aggregated_cost"].numpy().astype(np.float64),
                "disparity_initial": st["disparity_initial"].numpy().astype(np.float64),
                "refinement_residual": st["refinement_residual"].numpy().astype(np.float64),
                "disparity_final": st["disparity_final"].numpy().astype(np.float64),
            }
            if i == 0:
                # Assignment pass: semantic tensor names (graph position) +
                # value verification against the PyTorch stage. A stage is
                # located only if its semantic tensor matches its own PyTorch
                # stage closely AND (where a sibling decoy exists) mismatches
                # the sibling stage. Otherwise the stage is UNKNOWN.
                def _d(a, b):
                    a = np.asarray(a, dtype=np.float64)
                    b = np.asarray(b, dtype=np.float64)
                    if a.shape != b.shape:
                        return None
                    return float(np.abs(a - b).max())

                def _best(names, pref):
                    scored = []
                    for nm in names:
                        v = _d(omap[nm], pref)
                        if v is not None:
                            scored.append((v, nm))
                    scored.sort()
                    return scored

                names5d = [c["name"] for c in loc_info["candidates"]["cost_volume"]
                           if c["name"] in omap]
                s5 = _best(names5d, pt_stage["cost_volume"])
                # semantic anchor -> exact branch-output tensor names
                anchors = {
                    "left_features": "/feature_extractor/output_conv/Conv_output_0",
                    "right_features": "/feature_extractor/output_conv_1/Conv_output_0",
                    "cost_volume": "/cost_volume/Transpose_23_output_0",
                    "aggregated_cost": "/aggregation/Squeeze_output_0",
                    "refinement_residual": "/refinement/output_conv/Conv_output_0",
                    "disparity_final": "disparity",
                }
                for stage in STAGE_ORDER:
                    if stage == "disparity_initial":
                        # The initial-disparity tensor is the ReduceSum output
                        # (/regression/ReduceSum_output_0), but the Resize in
                        # the regression path defeats shape inference, so the
                        # prescribed shape method cannot locate it.
                        assign[stage] = {
                            "status": "UNKNOWN",
                            "reason": "Resize output shape uninferable "
                                      "((None,None,None,None)); all downstream "
                                      "regression tensors incl. the "
                                      "ReduceSum_output_0 disparity_initial "
                                      "have unknown shapes under "
                                      "onnx.shape_inference.infer_shapes",
                            "suspected_node": "/regression/ReduceSum_output_0"}
                        continue
                    nm = anchors[stage]
                    if nm not in omap:
                        assign[stage] = {"status": "UNKNOWN",
                                         "reason": f"semantic tensor {nm} not "
                                                   "in instrumented outputs"}
                        continue
                    self_d = _d(omap[nm], pt_stage[stage])
                    if self_d is None:
                        assign[stage] = {"status": "UNKNOWN",
                                         "reason": f"{nm} numpy shape "
                                                   f"{np.asarray(omap[nm]).shape} != "
                                                   f"pytorch {pt_stage[stage].shape}"}
                        continue
                    vm = float(np.abs(pt_stage[stage]).max())
                    rel = self_d / vm if vm > 0 else None
                    rec = {"onnx_tensor": nm, "best_max_abs": self_d,
                           "max_abs_value": vm, "relative": rel}
                    # sibling / runner-up checks
                    if stage in ("left_features", "right_features"):
                        sib = "right_features" if stage == "left_features" else "left_features"
                        sib_d = _d(omap[nm], pt_stage[sib])
                        other_nm = anchors[sib]
                        other_d = _d(omap[other_nm], pt_stage[stage]) \
                            if other_nm in omap else None
                        rec["sibling_stage_max_abs"] = sib_d
                        rec["other_branch_max_abs"] = other_d
                        rec["status"] = "located" if (
                            sib_d is not None and sib_d > 100 * max(self_d, 1e-12)
                            and other_d is not None and other_d > 100 * max(self_d, 1e-12)
                            and rel is not None and rel < 1e-5) else "UNKNOWN"
                        if rec["status"] == "UNKNOWN":
                            rec["reason"] = "branch cross-check failed"
                    elif stage == "cost_volume":
                        runner = s5[1][0] if len(s5) > 1 else None
                        rec["runner_up_max_abs"] = runner
                        rec["n_candidates_same_shape"] = len(s5)
                        rec["status"] = "located" if (
                            s5 and s5[0][1] == nm and rel is not None
                            and rel < 1e-5 and (runner is None or runner > 100 * max(self_d, 1e-12))) \
                            else "UNKNOWN"
                        if rec["status"] == "UNKNOWN":
                            rec["reason"] = "not the unique best match"
                    elif stage == "disparity_final":
                        # Structural identity: this IS the graph output, so
                        # its diff is the parity number by definition.
                        rec["status"] = "located"
                    elif stage == "refinement_residual":
                        # Residual scale is O(1)-ish, so a relative gate is
                        # invalid; use the decoy margin instead. The final
                        # diff is ~1e-3, so a true residual diff above 0.05
                        # would already contradict the observed final diff.
                        decoy_d = _d(omap["disparity"], pt_stage[stage]) \
                            if "disparity" in omap else None
                        rec["decoy_final_vs_residual"] = decoy_d
                        rec["status"] = "located" if (
                            self_d < 0.05 and decoy_d is not None
                            and decoy_d > 100 * max(self_d, 1e-12)) else "UNKNOWN"
                        if rec["status"] == "UNKNOWN":
                            rec["reason"] = "residual/decoy margin failed"
                    else:
                        rec["status"] = "located" if (
                            rel is not None and rel < 1e-5) else "UNKNOWN"
                        if rec["status"] == "UNKNOWN":
                            rec["reason"] = "relative divergence too large for identity"
                    assign[stage] = rec
            for stage in STAGE_ORDER:
                a = assign[stage]
                if a.get("status") == "UNKNOWN":
                    continue
                nm = a["onnx_tensor"]
                o = np.asarray(omap[nm], dtype=np.float64)
                p = pt_stage[stage]
                d = np.abs(o - p)
                acc_max.setdefault(stage, []).append(float(d.max()))
                acc_mean.setdefault(stage, []).append(float(d.mean()))
                acc_vmax.setdefault(stage, []).append(float(np.abs(p).max()))
        stages_rep = {}
        for stage in STAGE_ORDER:
            a = assign[stage]
            if a.get("status") == "UNKNOWN" or stage not in acc_max:
                stages_rep[stage] = {**a, "max_abs": None, "mean_abs": None,
                                     "relative": None, "material": False}
                continue
            mx = float(max(acc_max[stage]))
            mn = float(sum(acc_mean[stage]) / len(acc_mean[stage]))
            vm = float(max(acc_vmax[stage]))
            rel = float(mx / vm) if vm > 0 else None
            mat = bool(mx > MAT_ABS and rel is not None and rel > MAT_REL)
            stages_rep[stage] = {**a, "max_abs": mx, "mean_abs": mn,
                                 "max_abs_value": vm, "relative": rel,
                                 "material": mat}
        first_mat = next((s for s in STAGE_ORDER if stages_rep[s].get("material")), None)

        opt = {"per_scene_max_default": per_scene_max_default,
               "per_scene_max_disabled": per_scene_max_off,
               "pooled_max_default": float(max(per_scene_max_default)),
               "pooled_max_disabled": float(max(per_scene_max_off)),
               "changed": bool(per_scene_max_default != per_scene_max_off),
               "providers_requested": ["CPUExecutionProvider"],
               "providers_used": sess.get_providers()}

        mrec = {"magnitude": mag, "spatial": spat, "ties": ties,
                "stages": stages_rep, "first_material_stage": first_mat,
                "localisation_info": loc_info,
                "ort_optimization": opt, "determinism": det,
                "top20": [{"scene": r[1], "y": r[2], "x": r[3],
                           "pytorch_pred": r[4], "onnx_pred": r[5],
                           "abs_diff": r[0],
                           "rel_diff": float(r[0] / (abs(r[4]) + 1e-12)),
                           "gt_disparity": float(r[10]) if r[10] > 0 else None,
                           "top1_prob": r[6], "top2_prob": r[7],
                           "top1_minus_top2": r[8], "entropy": r[9]}
                          for r in top20]}
        out["models"][mname] = mrec

        # ---- CSV ----
        csv_name = "top_pixels_armp.csv" if mname == "armp" else "top_pixels_p2a.csv"
        with open(OUT / csv_name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["scene", "y", "x", "pytorch_pred",
                                              "onnx_pred", "abs_diff", "rel_diff",
                                              "gt_disparity", "top1_prob",
                                              "top2_prob", "top1_minus_top2",
                                              "entropy"])
            w.writeheader()
            w.writerows(mrec["top20"])
        print(f"{mname}: pooled max|d| default={opt['pooled_max_default']:.4e} "
              f"disabled={opt['pooled_max_disabled']:.4e} "
              f"first_material={first_mat}", flush=True)

    with open(OUT / "c1_parity_diagnostic.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", OUT / "c1_parity_diagnostic.json", flush=True)


if __name__ == "__main__":
    main()
