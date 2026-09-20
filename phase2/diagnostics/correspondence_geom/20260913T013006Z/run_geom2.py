"""EXP-CORRESPONDENCE-GEOM-002 -- Pairing-Swap Interaction Diagnostic.

Executes EXACTLY the protocol frozen in PREREGISTRATION.md / preregistration.json
/ design.json, on the CORRECTED domain derived in
phase2/diagnostics/correspondence_geom/domain_audit_20260912T013530Z/.

INFERENCE ONLY.  No optimizer is imported or constructed.  No .backward().
No .step().  No weight is modified anywhere in this file.  No architecture,
feature extractor, cost-volume construction, standardisation, readout or mask is
modified.  No historical record is written to.  No EPE / D1 / RMSE.  Refinement
is never invoked.

WHAT CHANGED FROM GEOM-001 (which halted at HS1 on an invalid domain):

    mask   x in [64,864) -> x in [325,633);  115200 px -> 44352 px
    HS1    one bound on w in [0,54) -> four sub-checks on their correct bands,
           all starting at w = 15 (the measured 15-cell extractor halo)
    HS9    115200 -> 44352
    gate   H* regenerated, never inherited
    arms   the search-free control NEG_shift_none is NOT run (the authorisation
           says "load exactly these three H2 checkpoints")

FIVE PROCESSES, ENFORCED:

    --stage 0        PREREQUISITES.  Read-only.  Verifies the freeze, the 41
                     historical records, Phase 1, and GEOM-001's own manifests.
                     Writes prereq.json + environment.json + checkpoint_hashes.json.
    --stage 1        the RANDOM-WEIGHT GATE.  Loads checkpoints ONLY for their
                     feature extractors, sets model.aggregation = None before any
                     forward pass, and contains NO stage-2 code path.  gate.json.
    --stage 2        the TRAINED arms.  REFUSES TO RUN unless gate.json says
                     G-PASS.  trained.json.
    --stage w4       determinism: recompute one complete trained unit in a fresh
                     process and compare its bits.
    --stage report   assembles results.json, metrics.csv, per_scene.csv,
                     per_tau.csv, hard_stop.json.  Computes NOTHING new.

Any hard stop RAISES, is written to hard_stop.json, and the process exits
non-zero.  The experiment is never repaired and continued.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import struct
import sys
import time
import traceback
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CUBLAS = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS

import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402

from src.models.stereonet.stereonet import StereoNet, StereoNetConfig    # noqa: E402
from src.models.stereonet.aggregation import Aggregation                 # noqa: E402
from src.datasets.kitti2015 import normalize                             # noqa: E402
from phase2.models import scaled_regression                              # noqa: E402
from phase2.viz import core                                              # noqa: E402

# ------------------------------------------------------------ frozen geometry
W, H, STRIDE, D = 1232, 368, 16, 12
CROP_W, CROP_H = 1136, 272
FW, FH = CROP_W // STRIDE, CROP_H // STRIDE          # 71 x 17
TAUS = [1, 2, 3, 4, 5, 6]
CELLS = ["AA", "BB", "AB", "BA"]
POSITIVE, NEGATIVE = ["AA", "BB"], ["AB", "BA"]

# ------------------------------------- the corrected domain (domain_derivation)
HALO = 15                       # measured extractor halo, cells, both sides
EXT_LO, EXT_HI = 15, 55         # boundary-free feature band, inclusive
CV_LO, CV_HI = 15, 44           # cost-volume clean band, inclusive, tau-INDEP
STAT_LO, STAT_HI = 20, 39       # aggregation-supported band, inclusive
MASK_Y0, MASK_Y1 = 64, 208
MASK_X0, MASK_X1 = 325, 633
EXPECTED_MASK = (MASK_Y1 - MASK_Y0) * (MASK_X1 - MASK_X0)               # 44352

CANCEL_TOL = 1e-5               # inherited from GEOM-001
AGG_PARAMS = 111585
SPLIT = "hailo_val"
PAIRS = [(1, 2), (3, 4), (10, 11), (12, 13), (14, 15), (16, 17)]
TRAINED = ["H2_seed0", "H2_seed1", "H2_seed2"]
SEEDS = list(range(32))
GATE_AXIS = "horizontal"
STAGE2_AXES = ("horizontal", "vertical")
W4_UNIT = ("H2_seed0", 0, "horizontal")

CHECKPOINTS = {
    "H2_seed0": {
        "path": "phase2/diagnostics/determinism/20260909T041500Z_baseline/checkpoints/"
                "STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA_checkpoint.pth",
        "sha256": "581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a",
        "shift": "left"},
    "H2_seed1": {
        "path": "phase2/factorial/block_count_full/20260910T005550Z/checkpoints/"
                "EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED1_checkpoint.pth",
        "sha256": "58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf",
        "shift": "left"},
    "H2_seed2": {
        "path": "phase2/factorial/block_count_full/20260910T005550Z/checkpoints/"
                "EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED2_checkpoint.pth",
        "sha256": "245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69",
        "shift": "left"},
}


class HardStop(RuntimeError):
    """Every one of these HALTS the process.  No post-hoc rescue."""


# ------------------------------------------------------------------- helpers
def sha256_bytes(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def bits(x: float) -> str:
    return struct.pack("<d", float(x)).hex()


def read_manifest(p: Path) -> dict:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            digest, name = line.split(None, 1)
            out[name.strip()] = digest
    return out


def verify_freeze(stage: str) -> dict:
    """frozen.sha256 over file bytes, and historical.sha256 (HS8)."""
    frozen = read_manifest(OUT / "frozen.sha256")
    for name, digest in frozen.items():
        actual = sha256_bytes(OUT / name)
        if actual != digest:
            raise HardStop("FREEZE VIOLATION %s: %s != %s" % (name, actual, digest))
    hist = read_manifest(OUT / "historical.sha256")
    for name, digest in hist.items():
        p = REPO_ROOT / name
        if not p.exists():
            raise HardStop("HS8 historical record missing: %s" % name)
        if sha256_bytes(p) != digest:
            raise HardStop("HS8 historical record modified: %s" % name)
    forbidden = {"0": ["gate.json", "trained.json", "results.json", "RESULTS.md"],
                 "1": ["gate.json", "results.json", "RESULTS.md"],
                 "2": ["trained.json", "results.json", "RESULTS.md"],
                 "w4": [],
                 "report": []}[stage]
    for f in forbidden:
        if (OUT / f).exists():
            raise HardStop("%s exists before stage %s" % (f, stage))
    return {"frozen": frozen, "historical_verified": len(hist)}


def setup_determinism() -> str:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    if torch.is_grad_enabled():
        raise HardStop("HS11 grad is enabled")
    if not torch.cuda.is_available():
        raise HardStop("HS6: CUDA required by the frozen determinism block")
    return "cuda"


def check_domain_static(ledger: dict) -> None:
    """HS1-D and HS9 -- pure arithmetic on the frozen constants, no model."""
    if not (EXT_LO == HALO and EXT_HI == FW - 1 - HALO):
        raise HardStop("HS1-D extractor band [%d,%d] != [%d,%d]"
                       % (EXT_LO, EXT_HI, HALO, FW - 1 - HALO))
    if CV_HI != EXT_HI - (D - 1):
        raise HardStop("HS1-D cost-volume upper bound %d != %d - %d"
                       % (CV_HI, EXT_HI, D - 1))
    if CV_HI > FW - D:            # shift_left zero-fill reach at k = D-1
        raise HardStop("HS1-D cost-volume band reaches the zero fill: %d > %d"
                       % (CV_HI, FW - D))
    if not (STAT_LO == CV_LO + 5 and STAT_HI == CV_HI - 5):
        raise HardStop("HS1-D statistic band [%d,%d] is not the 5-cell shrink of [%d,%d]"
                       % (STAT_LO, STAT_HI, CV_LO, CV_HI))
    width = STAT_HI - STAT_LO + 1
    if width < D:
        raise HardStop("HS1-D statistic band is %d columns, narrower than D=%d"
                       % (width, D))
    x0 = -(-1135 * STAT_LO // 70)                       # ceil(1135*w_lo/70)
    x1 = (1135 * STAT_HI) // 70                          # floor, exclusive upper
    if x0 != MASK_X0 or x1 != MASK_X1 - 1:
        raise HardStop("HS1-D pixel band [%d,%d] != frozen [%d,%d]"
                       % (x0, x1, MASK_X0, MASK_X1 - 1))
    if EXPECTED_MASK != 44352:
        raise HardStop("HS9 frozen mask count %d != 44352" % EXPECTED_MASK)
    ledger["HS1-D"] = {
        "status": "PASS",
        "extractor_band": [EXT_LO, EXT_HI],
        "cost_volume_band": [CV_LO, CV_HI],
        "statistic_band": [STAT_LO, STAT_HI],
        "statistic_columns": width,
        "tau_independent": True,
        "pixel_band": [MASK_X0, MASK_X1 - 1],
        "mask_rows": [MASK_Y0, MASK_Y1],
        "mask_pixels": EXPECTED_MASK,
        "fill_free_bound": FW - D,
    }


def crop(img: np.ndarray, y0: int, x0: int) -> np.ndarray:
    if y0 < 0 or x0 < 0 or y0 + CROP_H > H or x0 + CROP_W > W:
        raise HardStop("crop window (%d,%d) leaves the source" % (y0, x0))
    out = img[y0:y0 + CROP_H, x0:x0 + CROP_W]
    if out.shape[:2] != (CROP_H, CROP_W):
        raise HardStop("crop produced %s" % (out.shape[:2],))
    return out


def right_origin(axis: str, t: int) -> tuple[int, int]:
    if axis == "horizontal":
        return 0, t
    if axis == "vertical":
        return t, 0
    raise HardStop("unknown axis %r" % axis)


def hs1a_pixels(src: np.ndarray, left_c: np.ndarray, axis: str, t: int) -> None:
    """HS1-A -- asserted FROM THE ACTUAL ARRAYS, never assumed.  Both arms."""
    y0, x0 = right_origin(axis, t)
    right_c = crop(src, y0, x0)
    if axis == "horizontal":
        ok = np.array_equal(right_c[:, :CROP_W - t], left_c[:, t:])
        msg = "right[:, :-t] != left[:, t:]"
    else:
        ok = np.array_equal(right_c[:CROP_H - t, :], left_c[t:, :])
        msg = "right[:-t, :] != left[t:, :]"
    if not ok:
        raise HardStop("HS1-A pixel construction failed, axis=%s t=%d: %s"
                       % (axis, t, msg))


def hs1b_features(lf: torch.Tensor, rf: torch.Tensor, tau: int, who: str,
                  ledger: dict) -> None:
    """HS1-B -- Rf[w] == Lf[w+tau] EXACTLY on w in [15, 55-tau].  Horizontal only.

    Tolerance is exactly zero.  On the clean band this is a structural identity,
    not an approximation; the domain audit measured bit-equality in 216/216
    records.  GEOM-001 asserted the same identity starting at w = 0, inside the
    15-cell extractor halo, and halted.
    """
    hi = EXT_HI - tau                       # inclusive
    a = rf[..., EXT_LO:hi + 1]
    b = lf[..., EXT_LO + tau:hi + 1 + tau]
    if a.shape != b.shape:
        raise HardStop("HS1-B slice shape %s != %s" % (tuple(a.shape), tuple(b.shape)))
    if a.numel() == 0:
        raise HardStop("HS1-B band is empty at tau=%d" % tau)
    if not torch.equal(a, b):
        worst = float((a - b).abs().max())
        raise HardStop("HS1-B Rf[w] != Lf[w+%d] on w in [%d,%d] (%s): max|d| = %r"
                       % (tau, EXT_LO, hi, who, worst))
    ledger["HS1-B"]["checks"] += 1
    ledger["HS1-B"]["cells_compared"] += int(a.numel())


def hs1c_anchor(vol: torch.Tensor, tau: int, cell: str, ledger: dict) -> None:
    """HS1-C -- |V[:, tau, :, w]| == 0.0 EXACTLY on w in [15,45).  Horizontal only."""
    z = float(vol[0, :, tau, :, CV_LO:CV_HI + 1].abs().max())
    ledger["HS1-C"]["values"].append(z)
    ledger["HS1-C"]["cells_compared"] += int(vol[0, :, tau, :, CV_LO:CV_HI + 1].numel())
    if z != 0.0:
        raise HardStop("HS1-C %s |V[:,tau=%d,:,w in [%d,%d]]| = %r != 0"
                       % (cell, tau, CV_LO, CV_HI, z))


def load_model(key: str, device: str, ledger: dict):
    meta = CHECKPOINTS[key]
    path = REPO_ROOT / meta["path"]
    actual = sha256_bytes(path)
    if actual != meta["sha256"]:
        raise HardStop("HS5 %s sha256 %s != frozen %s" % (key, actual, meta["sha256"]))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = StereoNet(StereoNetConfig(cost_volume_shift=meta["shift"]))
    scaled_regression.apply_to(model)
    model.load_state_dict(ckpt["model"])
    n_readout = sum(p.numel() for p in model.regression.parameters())
    if n_readout != 0:
        raise HardStop("HS5 %s readout has %d parameters" % (key, n_readout))
    if model.cost_volume.shift != meta["shift"]:
        raise HardStop("HS5 %s shift %r != %r" % (key, model.cost_volume.shift,
                                                  meta["shift"]))
    if model.config.num_disparities != D:
        raise HardStop("HS5 %s num_disparities %d != %d"
                       % (key, model.config.num_disparities, D))
    n_agg = sum(p.numel() for p in model.aggregation.parameters())
    if n_agg != AGG_PARAMS:
        raise HardStop("HS5 %s aggregation has %d parameters != %d"
                       % (key, n_agg, AGG_PARAMS))
    ledger["HS5"]["checkpoints"][key] = {
        "sha256": actual, "shift": meta["shift"], "num_disparities": D,
        "readout_parameters": n_readout, "aggregation_parameters": n_agg}
    return model.eval().to(device), actual


def scene_crops(pair_index: int) -> dict:
    ia, ib = PAIRS[pair_index]
    sa, sb = core.load_scene(ia, split=SPLIT), core.load_scene(ib, split=SPLIT)
    return {"A": {"src": sa.left, "left": crop(sa.left, 0, 0), "name": sa.name},
            "B": {"src": sb.left, "left": crop(sb.left, 0, 0), "name": sb.name}}


def build_volumes(model, sc: dict, axis: str, device: str, ledger: dict) -> dict:
    """The 24 cost volumes for one (model, pair, axis).  Aggregation-independent."""
    lf = {who: model.feature_extractor(
              torch.from_numpy(normalize(sc[who]["left"])).to(device))
          for who in ("A", "B")}
    vols = {}
    for tau in TAUS:
        t = tau * STRIDE
        rf = {}
        for who in ("A", "B"):
            hs1a_pixels(sc[who]["src"], sc[who]["left"], axis, t)
            ledger["HS1-A"]["checks"] += 1
            y0, x0 = right_origin(axis, t)
            ledger["right_origins"].add((axis, t, y0, x0))
            rc = crop(sc[who]["src"], y0, x0)
            ledger["crop_shapes"].add(rc.shape[:2])
            rf[who] = model.feature_extractor(
                torch.from_numpy(normalize(rc)).to(device))
            if axis == "horizontal":
                hs1b_features(lf[who], rf[who], tau, who, ledger)
            else:
                ledger["HS1-B"]["inapplicable_vertical"] += 1

        # HS2 -- each frame exactly once as a left and once as a right
        plan = {"AA": ("A", "A"), "BB": ("B", "B"), "AB": ("A", "B"), "BA": ("B", "A")}
        if sorted(p[0] for p in plan.values()) != ["A", "A", "B", "B"] or \
           sorted(p[1] for p in plan.values()) != ["A", "A", "B", "B"]:
            raise HardStop("HS2 pairing plan is not balanced: %r" % plan)
        ledger["HS2"]["checks"] += 1

        taus_used = set()
        for cell, (li, ri) in plan.items():
            v = model.cost_volume(lf[li], rf[ri])
            if tuple(v.shape) != (1, 32, D, FH, FW):
                raise HardStop("unexpected volume shape %s" % (tuple(v.shape),))
            ledger["volume_shapes"].add(tuple(v.shape))
            vols[(cell, tau)] = v
            taus_used.add(t)
        if taus_used != {t}:                                        # HS3
            raise HardStop("HS3 tau differs across cells at t=%d: %r" % (t, taus_used))
        ledger["HS3"]["checks"] += 1

        # HS1-C -- the matched anchor, horizontal only
        if axis == "horizontal" and model.cost_volume.shift == "left":
            for cell in POSITIVE:
                hs1c_anchor(vols[(cell, tau)], tau, cell, ledger)
        elif axis == "vertical":
            ledger["HS1-C"]["inapplicable_vertical"] += 1

        # HS4 -- raw cost-volume 2x2 cancellation
        resid = (vols[("AA", tau)] + vols[("BB", tau)]
                 - vols[("AB", tau)] - vols[("BA", tau)])
        scale = float(vols[("AB", tau)].abs().max())
        rel = float(resid.abs().max()) / max(scale, 1e-12)
        ledger["HS4"]["records"].append({"axis": axis, "tau": tau,
                                         "max_abs_residual": float(resid.abs().max()),
                                         "volume_scale": scale, "relative": rel})
        if rel > CANCEL_TOL:
            raise HardStop("HS4 cancellation %.3e > %.1e at tau=%d axis=%s"
                           % (rel, CANCEL_TOL, tau, axis))
        del resid
    return vols


def cell_readout(agg, regression, volume: torch.Tensor):
    """The FROZEN readout module used verbatim, then the corrected mask.

    `regression.capture` stores the exact tensor the softmax consumes and changes
    no arithmetic.  The mask is a rectangle, so masking is a slice; the
    standardisation and the soft-argmin are pointwise in (y, x), so slicing after
    them is exactly masking.
    """
    regression.capture = True
    try:
        m = regression(agg(volume), (CROP_H, CROP_W))          # (1, 1, 272, 1136)
        z = regression.last["softmax_input"]                   # (1, D, 272, 1136)
    finally:
        regression.capture = False
        regression.last = {}
    if tuple(z.shape) != (1, D, CROP_H, CROP_W):
        raise HardStop("unexpected softmax_input shape %s" % (tuple(z.shape),))
    zs = z[:, :, MASK_Y0:MASK_Y1, MASK_X0:MASK_X1]
    ms = m[:, :, MASK_Y0:MASK_Y1, MASK_X0:MASK_X1]
    if zs.shape[-2] * zs.shape[-1] != EXPECTED_MASK:
        raise HardStop("HS9 mask count %d != %d"
                       % (zs.shape[-2] * zs.shape[-1], EXPECTED_MASK))
    return ms, zs


def unit_measure(agg, regression, vols: dict, ledger: dict) -> dict:
    """One (aggregation, pair, axis) unit: the six tau levels."""
    out = {"tau": {}}
    for tau in TAUS:
        zs, ms = {}, {}
        for cell in CELLS:
            m, z = cell_readout(agg, regression, vols[(cell, tau)])
            zs[cell] = z[0].reshape(D, -1)
            ms[cell] = m[0, 0].reshape(-1)
        inter = 0.5 * (zs["AA"] + zs["BB"]) - 0.5 * (zs["AB"] + zs["BA"])
        arrs = {c: zs[c].detach().float().cpu().numpy().astype(np.float64)
                for c in CELLS}
        ia = inter.detach().float().cpu().numpy().astype(np.float64)
        if not np.isfinite(ia).all():
            raise HardStop("HS10 non-finite interaction at tau=%d" % tau)
        ledger["HS10"]["checks"] += 1
        ibar = np.median(ia, axis=1)
        if not np.isfinite(ibar).all():
            raise HardStop("HS10 non-finite profile at tau=%d" % tau)
        zbar = {c: np.median(arrs[c], axis=1) for c in CELLS}

        k_hat = int(np.argmin(ibar))                 # smallest index on a tie
        i_min = float(ibar[k_hat])
        n_at_min = int(np.sum(ibar == ibar.min()))
        others = np.delete(ibar, k_hat)
        i_second = float(others.min())
        out["tau"][tau] = {
            "Ibar": [float(v) for v in ibar],
            "k_hat": k_hat,
            "I_min": i_min,
            "I_second": i_second,
            "margin": i_second - i_min,
            "tie": bool(n_at_min > 1),
            "n_at_min": n_at_min,
            "argmax": int(np.argmax(ibar)),
            "hit": bool(k_hat == tau),
            "offset": k_hat - tau,
            "amplitude": float(ibar.max() - ibar.min()),
            "zbar": {c: [float(v) for v in zbar[c]] for c in CELLS},
            "m_median": {c: float(np.median(
                ms[c].detach().float().cpu().numpy().astype(np.float64)))
                for c in CELLS},
            "Ibar_bits": [bits(v) for v in ibar],
        }
        if n_at_min > 1:
            ledger["tie_events"].append({"tau": tau, "n_at_min": n_at_min,
                                         "k_hat": k_hat, "I_min": i_min})
    out["h"] = int(sum(out["tau"][t]["hit"] for t in TAUS))
    out["offsets"] = [out["tau"][t]["offset"] for t in TAUS]
    out["k_hats"] = [out["tau"][t]["k_hat"] for t in TAUS]
    out["offset_constant"] = bool(len(set(out["offsets"])) == 1)
    out["tracking_with_offset"] = bool(out["offset_constant"]
                                       and out["offsets"][0] != 0)
    return out


def new_ledger() -> dict:
    return {
        "HS1-A": {"checks": 0},
        "HS1-B": {"checks": 0, "cells_compared": 0, "tolerance": 0.0,
                  "band": "w in [%d, %d - tau]" % (EXT_LO, EXT_HI),
                  "inapplicable_vertical": 0},
        "HS1-C": {"values": [], "cells_compared": 0, "tolerance": 0.0,
                  "band": "w in [%d, %d]" % (CV_LO, CV_HI),
                  "inapplicable_vertical": 0},
        "HS1-D": {},
        "HS2": {"checks": 0},
        "HS3": {"checks": 0},
        "HS4": {"records": [], "tolerance": CANCEL_TOL},
        "HS5": {"checkpoints": {}},
        "HS10": {"checks": 0},
        "tie_events": [],
        "crop_shapes": set(), "volume_shapes": set(), "right_origins": set(),
    }


def ledger_to_json(c: dict) -> dict:
    out = dict(c)
    out["crop_shapes"] = sorted(map(list, c["crop_shapes"]))
    out["volume_shapes"] = sorted(map(list, c["volume_shapes"]))
    out["right_origins"] = sorted(map(list, c["right_origins"]))
    hs1c = dict(c["HS1-C"])
    hs1c["max_observed"] = max(c["HS1-C"]["values"]) if c["HS1-C"]["values"] else None
    hs1c["n"] = len(c["HS1-C"]["values"])
    del hs1c["values"]
    out["HS1-C"] = hs1c
    hs4 = dict(c["HS4"])
    hs4["max_relative"] = (max(r["relative"] for r in c["HS4"]["records"])
                           if c["HS4"]["records"] else None)
    hs4["median_relative"] = (float(np.median([r["relative"]
                                               for r in c["HS4"]["records"]]))
                              if c["HS4"]["records"] else None)
    hs4["n"] = len(c["HS4"]["records"])
    out["HS4"] = hs4
    return out


# ======================================================== STAGE 0: PREREQUISITES
def stage0(log) -> int:
    meta = verify_freeze("0")
    log("freeze verified: %d frozen artifacts, %d historical records (HS8 pre)"
        % (len(meta["frozen"]), meta["historical_verified"]))

    # GEOM-001 must still be byte-identical, including its own manifests
    g1 = REPO_ROOT / "phase2/diagnostics/correspondence_geom/20260912T011452Z"
    g1_state = {}
    for man in ("frozen.sha256", "historical.sha256"):
        entries = read_manifest(g1 / man)
        base = g1 if man == "frozen.sha256" else REPO_ROOT
        bad = [n for n, d in entries.items() if sha256_bytes(base / n) != d]
        if bad:
            raise HardStop("HS8 GEOM-001 %s no longer verifies: %r" % (man, bad))
        g1_state[man] = {"entries": len(entries), "mismatches": 0}
    log("GEOM-001 immutable: frozen %d/%d, historical %d/%d"
        % (g1_state["frozen.sha256"]["entries"], g1_state["frozen.sha256"]["entries"],
           g1_state["historical.sha256"]["entries"],
           g1_state["historical.sha256"]["entries"]))

    ledger = new_ledger()
    check_domain_static(ledger)
    log("HS1-D domain: extractor %s cost-volume %s statistic %s -> pixels %s, "
        "mask %d px" % (ledger["HS1-D"]["extractor_band"],
                        ledger["HS1-D"]["cost_volume_band"],
                        ledger["HS1-D"]["statistic_band"],
                        ledger["HS1-D"]["pixel_band"],
                        ledger["HS1-D"]["mask_pixels"]))

    ck = {}
    for key, m in CHECKPOINTS.items():
        p = REPO_ROOT / m["path"]
        if not p.exists():
            raise HardStop("HS5 checkpoint missing: %s" % m["path"])
        actual = sha256_bytes(p)
        if actual != m["sha256"]:
            raise HardStop("HS5 %s sha256 %s != frozen %s" % (key, actual, m["sha256"]))
        ck[key] = {"path": m["path"], "sha256": actual, "shift": m["shift"],
                   "size_bytes": p.stat().st_size}
        log("checkpoint %s verified %s..." % (key, actual[:12]))
    (OUT / "checkpoint_hashes.json").write_text(
        json.dumps({"checkpoints": ck,
                    "controls_not_run": {"NEG_shift_none":
                                         "excluded by the authorisation; see "
                                         "PREREGISTRATION.md B.4"}},
                   indent=2), encoding="utf-8")

    device = setup_determinism()
    env = {
        "experiment": "EXP-CORRESPONDENCE-GEOM-002",
        "record": "phase2/diagnostics/correspondence_geom/20260913T013006Z",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0))
                      if torch.cuda.is_available() else None,
        "determinism": {
            "use_deterministic_algorithms": True,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "grad_enabled": bool(torch.is_grad_enabled()),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "dtype": "float32",
            "device": device,
        },
        "guarantees": {
            "training_performed": False,
            "optimizer_constructed": False,
            "backward_called": False,
            "refinement_invoked": False,
            "historical_records_written": 0,
        },
    }
    (OUT / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    prereq = {
        "stage": 0,
        "phase_1_frozen": "verified externally against git tag phase-1-frozen; "
                          "the source files under test are in historical.sha256",
        "freeze": {"artifacts": len(meta["frozen"]),
                   "historical_records": meta["historical_verified"],
                   "mismatches": 0},
        "geom001": {"status": "HALTED -- INVALID DOMAIN -- NO-ADMISSIBLE-RESULT",
                    "manifests": g1_state,
                    "gate_result_reused": False},
        "domain_static_check": ledger["HS1-D"],
        "checkpoints": ck,
        "determinism": env["determinism"],
        "no_optimizer_no_backward_no_update": True,
        "verdict": "PREREQUISITES PASS",
    }
    (OUT / "prereq.json").write_text(json.dumps(prereq, indent=2), encoding="utf-8")

    verify_freeze("0")                                   # HS8 post
    log("prerequisites PASS; environment.json, checkpoint_hashes.json, "
        "prereq.json written; no model was instantiated")
    return 0


# ============================================================== STAGE 1: GATE
def stage1(log) -> int:
    meta = verify_freeze("1")
    log("freeze verified: %d frozen artifacts, %d historical records"
        % (len(meta["frozen"]), meta["historical_verified"]))
    if not (OUT / "prereq.json").exists():
        raise HardStop("prereq.json absent -- stage 0 has not run")
    device = setup_determinism()
    t0 = time.time()

    ledger = new_ledger()
    check_domain_static(ledger)
    units, raw = [], []
    n_readout = 0
    for ext in TRAINED:
        model, digest = load_model(ext, device, ledger)
        # ---- HS7 : the trained aggregation is destroyed before it can be used
        trained_agg_hash = hashlib.sha256(
            "".join(hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
                    for _, p in model.aggregation.named_parameters()).encode()
        ).hexdigest()
        model.aggregation = None
        if model.aggregation is not None:
            raise HardStop("HS7 trained aggregation still present")
        log("extractor %s loaded (sha %s...), trained aggregation DISCARDED "
            "(its hash %s... is recorded but never used)"
            % (ext, digest[:12], trained_agg_hash[:12]))

        for pi in range(len(PAIRS)):
            sc = scene_crops(pi)
            vols = build_volumes(model, sc, GATE_AXIS, device, ledger)
            for seed in SEEDS:
                torch.manual_seed(seed)
                agg = Aggregation(in_channels=32, channels=32,
                                  num_layers=4).eval().to(device)
                n_agg = sum(p.numel() for p in agg.parameters())
                if n_agg != AGG_PARAMS:
                    raise HardStop("HS5 random aggregation param count %d != %d"
                                   % (n_agg, AGG_PARAMS))
                u = unit_measure(agg, model.regression, vols, ledger)
                n_readout += len(TAUS) * len(CELLS)
                units.append({"extractor": ext, "pair": pi, "seed": seed,
                              "pair_names": [sc["A"]["name"], sc["B"]["name"]],
                              "h": u["h"], "k_hats": u["k_hats"],
                              "offsets": u["offsets"],
                              "offset_constant": u["offset_constant"],
                              "tracking_with_offset": u["tracking_with_offset"]})
                raw.append({"extractor": ext, "pair": pi, "seed": seed,
                            "pair_names": [sc["A"]["name"], sc["B"]["name"]],
                            "tau": {str(k): v for k, v in u["tau"].items()},
                            "h": u["h"]})
                del agg
            del vols
            torch.cuda.empty_cache()
            log("gate %s pair %d (%s/%s) done  32 seeds  h=%s  (%.1fs)"
                % (ext, pi, sc["A"]["name"], sc["B"]["name"],
                   [x["h"] for x in units[-32:]], time.time() - t0))
        del model

    h_rand = [u["h"] for u in units]
    expected = len(TRAINED) * len(PAIRS) * len(SEEDS)
    if len(units) != expected:
        raise HardStop("gate unit count %d != %d" % (len(units), expected))
    hmax = max(h_rand)
    decision = "G-STOP" if hmax == 6 else "G-PASS"
    hstar = None if decision == "G-STOP" else hmax

    hist = {i: int(sum(1 for x in h_rand if x == i)) for i in range(7)}
    gate = {
        "stage": 1, "axis": GATE_AXIS,
        "n_units": len(units), "n_readout_passes": n_readout,
        "unit_structure": "(extractor, pair, seed) = %d x %d x %d"
                          % (len(TRAINED), len(PAIRS), len(SEEDS)),
        "seeds": SEEDS, "taus": TAUS, "pairs": PAIRS,
        "mask": {"rows": [MASK_Y0, MASK_Y1], "cols": [MASK_X0, MASK_X1],
                 "pixels": EXPECTED_MASK},
        "h_rand_histogram": hist,
        "h_rand_max": hmax, "h_rand_min": min(h_rand),
        "h_rand_mean": float(np.mean(h_rand)),
        "hit_rate": float(sum(h_rand) / (len(h_rand) * len(TAUS))),
        "chance_rate": 1.0 / D,
        "offset_constant_units": int(sum(1 for u in units if u["offset_constant"])),
        "tracking_with_offset_units": int(sum(1 for u in units
                                              if u["tracking_with_offset"])),
        "decision": decision, "H_star": hstar,
        "trained_aggregation_loaded": False,
        "hard_stop_ledger": ledger_to_json(ledger),
        "units": units,
        "wall_clock_s": time.time() - t0,
    }
    (OUT / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    (OUT / "gate_raw.json").write_text(json.dumps(raw), encoding="utf-8")
    verify_freeze("2")                                   # HS8 post (gate.json now exists)
    log("GATE h histogram %s" % hist)
    log("GATE max(h_rand)=%d  ->  %s   H*=%r" % (hmax, decision, hstar))
    log("stage 1 wall_clock_s=%.2f  readouts=%d" % (time.time() - t0, n_readout))
    if decision == "G-STOP":
        log("TERMINAL: STATISTIC-ARCHITECTURALLY-AVAILABLE. "
            "Trained weights are NOT loaded. Stage 2 is not run.")
    return 0


# ========================================================== STAGE 2: TRAINED
def stage2(log) -> int:
    meta = verify_freeze("2")
    log("freeze verified: %d frozen artifacts, %d historical records"
        % (len(meta["frozen"]), meta["historical_verified"]))
    if not (OUT / "gate.json").exists():
        raise HardStop("HS7 gate.json absent -- stage 1 has not run")
    gate = json.loads((OUT / "gate.json").read_text(encoding="utf-8"))
    if gate["decision"] != "G-PASS":
        raise HardStop("HS7 gate decision is %r -- stage 2 must not run"
                       % gate["decision"])
    hstar = gate["H_star"]
    if not isinstance(hstar, int):
        raise HardStop("H* is not an integer: %r" % hstar)
    log("gate: G-PASS, H* = %d (unaltered)" % hstar)

    device = setup_determinism()
    t0 = time.time()
    ledger = new_ledger()
    check_domain_static(ledger)
    results = {"horizontal": [], "vertical": []}
    raw, n_readout = [], 0

    for key in TRAINED:
        model, digest = load_model(key, device, ledger)
        agg_hash = hashlib.sha256(
            "".join(hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
                    for _, p in model.aggregation.named_parameters()).encode()
        ).hexdigest()
        log("%s loaded  ckpt %s...  shift=%s  aggregation %s..."
            % (key, digest[:12], model.cost_volume.shift, agg_hash[:12]))
        for pi in range(len(PAIRS)):
            sc = scene_crops(pi)
            h_by_axis = {}
            for axis in STAGE2_AXES:
                vols = build_volumes(model, sc, axis, device, ledger)
                u = unit_measure(model.aggregation, model.regression, vols, ledger)
                n_readout += len(TAUS) * len(CELLS)
                rec = {"checkpoint": key, "pair": pi, "axis": axis,
                       "pair_names": [sc["A"]["name"], sc["B"]["name"]],
                       "h": u["h"], "k_hats": u["k_hats"], "offsets": u["offsets"],
                       "offset_constant": u["offset_constant"],
                       "tracking_with_offset": u["tracking_with_offset"],
                       "I_min": [u["tau"][t]["I_min"] for t in TAUS],
                       "I_second": [u["tau"][t]["I_second"] for t in TAUS],
                       "margin": [u["tau"][t]["margin"] for t in TAUS],
                       "ties": [u["tau"][t]["tie"] for t in TAUS],
                       "amplitudes": [u["tau"][t]["amplitude"] for t in TAUS]}
                results[axis].append(rec)
                raw.append({"checkpoint": key, "pair": pi, "axis": axis,
                            "pair_names": [sc["A"]["name"], sc["B"]["name"]],
                            "h": u["h"], "aggregation_sha256": agg_hash,
                            "tau": {str(k): v for k, v in u["tau"].items()}})
                h_by_axis[axis] = u["h"]
                del vols
                torch.cuda.empty_cache()
            log("%s pair %d (%s/%s) done  h_horiz=%d h_vert=%d  (%.1fs)"
                % (key, pi, sc["A"]["name"], sc["B"]["name"],
                   h_by_axis["horizontal"], h_by_axis["vertical"],
                   time.time() - t0))
        del model

    w4 = [r for r in raw if r["checkpoint"] == W4_UNIT[0]
          and r["pair"] == W4_UNIT[1] and r["axis"] == W4_UNIT[2]][0]
    (OUT / "w4_inline.json").write_text(
        json.dumps({str(t): w4["tau"][str(t)]["Ibar_bits"] for t in TAUS}, indent=2),
        encoding="utf-8")

    out = {"stage": 2, "H_star": hstar, "n_readout_passes": n_readout,
           "checkpoints": TRAINED,
           "controls_not_run": ["NEG_shift_none"],
           "arms": results, "hard_stop_ledger": ledger_to_json(ledger),
           "wall_clock_s": time.time() - t0}
    (OUT / "trained.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT / "trained_raw.json").write_text(json.dumps(raw), encoding="utf-8")
    verify_freeze("w4")                                  # HS8 post
    log("stage 2 wall_clock_s=%.2f  readouts=%d" % (time.time() - t0, n_readout))
    return 0


# ================================================================= STAGE W4
def stage_w4(log) -> int:
    verify_freeze("w4")
    device = setup_determinism()
    key, pi, axis = W4_UNIT
    ledger = new_ledger()
    check_domain_static(ledger)
    model, _ = load_model(key, device, ledger)
    sc = scene_crops(pi)
    vols = build_volumes(model, sc, axis, device, ledger)
    u = unit_measure(model.aggregation, model.regression, vols, ledger)
    out = {str(t): u["tau"][t]["Ibar_bits"] for t in TAUS}
    (OUT / "w4_restart.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    ref = json.loads((OUT / "w4_inline.json").read_text(encoding="utf-8"))
    identical = all(ref[str(t)] == out[str(t)] for t in TAUS)
    log("W4 unit %s pair %d %s : bit-identical after restart = %s"
        % (key, pi, axis, identical))
    if not identical:
        raise HardStop("HS6 W4 repeat is NOT bit-identical")
    return 0


# ================================================================ STAGE REPORT
def stage_report(log) -> int:
    verify_freeze("report")
    gate = json.loads((OUT / "gate.json").read_text(encoding="utf-8"))
    hstar = gate["H_star"]
    decision = gate["decision"]

    trained_p = OUT / "trained.json"
    trained = json.loads(trained_p.read_text(encoding="utf-8")) if trained_p.exists() else None
    graw_p, traw_p = OUT / "gate_raw.json", OUT / "trained_raw.json"
    graw = json.loads(graw_p.read_text(encoding="utf-8"))
    traw = json.loads(traw_p.read_text(encoding="utf-8")) if traw_p.exists() else []

    # ---- per_tau.csv : every unit, every tau, every argmin and margin
    with open(OUT / "per_tau.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["arm", "aggregation", "seed", "axis", "pair", "scene_A",
                     "scene_B", "tau", "k_hat", "hit", "offset", "I_min",
                     "I_second", "margin", "tie", "n_at_min", "amplitude"])
        for r in graw:
            for t in TAUS:
                d = r["tau"][str(t)]
                wr.writerow(["gate", r["extractor"], r["seed"], GATE_AXIS, r["pair"],
                             r["pair_names"][0], r["pair_names"][1], t, d["k_hat"],
                             int(d["hit"]), d["offset"], "%.17g" % d["I_min"],
                             "%.17g" % d["I_second"], "%.17g" % d["margin"],
                             int(d["tie"]), d["n_at_min"], "%.17g" % d["amplitude"]])
        for r in traw:
            for t in TAUS:
                d = r["tau"][str(t)]
                wr.writerow(["trained", r["checkpoint"], "", r["axis"], r["pair"],
                             r["pair_names"][0], r["pair_names"][1], t, d["k_hat"],
                             int(d["hit"]), d["offset"], "%.17g" % d["I_min"],
                             "%.17g" % d["I_second"], "%.17g" % d["margin"],
                             int(d["tie"]), d["n_at_min"], "%.17g" % d["amplitude"]])

    # ---- per_scene.csv : one row per unit
    with open(OUT / "per_scene.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["arm", "aggregation", "seed", "axis", "pair", "scene_A",
                     "scene_B", "h", "k_hats", "offsets", "offset_constant",
                     "tracking_with_offset"])
        for u in gate["units"]:
            wr.writerow(["gate", u["extractor"], u["seed"], GATE_AXIS, u["pair"],
                         u["pair_names"][0], u["pair_names"][1], u["h"],
                         " ".join(map(str, u["k_hats"])),
                         " ".join(map(str, u["offsets"])),
                         int(u["offset_constant"]), int(u["tracking_with_offset"])])
        if trained:
            for axis in STAGE2_AXES:
                for r in trained["arms"][axis]:
                    wr.writerow(["trained", r["checkpoint"], "", axis, r["pair"],
                                 r["pair_names"][0], r["pair_names"][1], r["h"],
                                 " ".join(map(str, r["k_hats"])),
                                 " ".join(map(str, r["offsets"])),
                                 int(r["offset_constant"]),
                                 int(r["tracking_with_offset"])])

    # ---- the comparison
    comparison = {"H_star": hstar, "gate_decision": decision}
    verdict = None
    if decision == "G-STOP":
        verdict = "STATISTIC-ARCHITECTURALLY-AVAILABLE"
        comparison["note"] = ("max(h_rand) == 6: a random aggregation reproduces "
                              "the statistic perfectly. Trained weights were never "
                              "loaded. This does NOT establish correspondence.")
    elif trained is None:
        verdict = "INCOMPLETE -- stage 2 has not run"
    else:
        horiz = trained["arms"]["horizontal"]
        h_units = [r["h"] for r in horiz]
        per_ck = {}
        for key in TRAINED:
            hs = [r["h"] for r in horiz if r["checkpoint"] == key]
            per_ck[key] = {
                "h_per_pair": hs,
                "min": min(hs), "max": max(hs),
                "mean": float(np.mean(hs)),
                "units_exceeding_H_star": int(sum(1 for x in hs if x > hstar)),
                "all_units_exceed_H_star": bool(min(hs) > hstar),
                "any_unit_exceeds_H_star": bool(max(hs) > hstar),
            }
        n_exceed = int(sum(1 for x in h_units if x > hstar))
        comparison.update({
            "trained_units": len(h_units),
            "h_trained_min": min(h_units), "h_trained_max": max(h_units),
            "h_trained_mean": float(np.mean(h_units)),
            "h_trained_histogram": {i: int(sum(1 for x in h_units if x == i))
                                    for i in range(7)},
            "units_exceeding_H_star": n_exceed,
            "min_h_trained_gt_H_star": bool(min(h_units) > hstar),
            "per_checkpoint": per_ck,
        })
        ck_all = [k for k in TRAINED if per_ck[k]["all_units_exceed_H_star"]]
        ck_none = [k for k in TRAINED if not per_ck[k]["any_unit_exceeds_H_star"]]
        if min(h_units) > hstar:
            verdict = "EXCEEDS-RANDOM-GATE"
        elif n_exceed == 0:
            verdict = "NOT-DEMONSTRATED"
        elif ck_all and ck_none:
            verdict = "CASE C -- HETEROGENEOUS"
        else:
            verdict = "NOT-DEMONSTRATED (partial exceedance, strongest bar not met)"

        vert = trained["arms"]["vertical"]
        v_units = [r["h"] for r in vert]
        comparison["vertical_control"] = {
            "status": "CONTAMINATED NULL -- diagnostic only, never compared to H*, "
                      "never overrides the horizontal result",
            "h_per_unit": v_units,
            "h_max": max(v_units), "h_min": min(v_units),
            "h_mean": float(np.mean(v_units)),
            "units_at_or_above_H_star": int(sum(1 for x in v_units if x >= hstar))
            if isinstance(hstar, int) else None,
            "interpretation": "a negative here is UNINFORMATIVE; a positive MAY "
                              "downgrade the interpretation",
        }
        comparison["tracking_with_offset"] = {
            "horizontal_units": [
                {"checkpoint": r["checkpoint"], "pair": r["pair"],
                 "offset": r["offsets"][0]}
                for r in horiz if r["tracking_with_offset"]],
            "role": "DIAGNOSTIC ONLY -- h is not redefined to count offset matches",
        }

    # ---- metrics.csv : the headline numbers, one per row
    rows = [("H_star", hstar), ("gate_decision", decision),
            ("gate_units", gate["n_units"]),
            ("gate_h_max", gate["h_rand_max"]), ("gate_h_min", gate["h_rand_min"]),
            ("gate_h_mean", gate["h_rand_mean"]),
            ("gate_hit_rate", gate["hit_rate"]), ("chance_rate", gate["chance_rate"]),
            ("mask_pixels", EXPECTED_MASK), ("mask_rows", "%d-%d" % (MASK_Y0, MASK_Y1)),
            ("mask_cols", "%d-%d" % (MASK_X0, MASK_X1)),
            ("verdict", verdict)]
    if trained:
        rows += [("trained_units", comparison["trained_units"]),
                 ("h_trained_min", comparison["h_trained_min"]),
                 ("h_trained_max", comparison["h_trained_max"]),
                 ("h_trained_mean", comparison["h_trained_mean"]),
                 ("units_exceeding_H_star", comparison["units_exceeding_H_star"]),
                 ("min_h_trained_gt_H_star", comparison["min_h_trained_gt_H_star"])]
        for key in TRAINED:
            rows.append(("h_%s_per_pair" % key,
                         " ".join(map(str, comparison["per_checkpoint"][key]["h_per_pair"]))))
        rows.append(("vertical_h_max", comparison["vertical_control"]["h_max"]))
    with open(OUT / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["metric", "value"])
        for k, v in rows:
            wr.writerow([k, v])

    ties = []
    for r in graw:
        for t in TAUS:
            if r["tau"][str(t)]["tie"]:
                ties.append({"arm": "gate", "aggregation": r["extractor"],
                             "seed": r["seed"], "pair": r["pair"], "tau": t,
                             "n_at_min": r["tau"][str(t)]["n_at_min"],
                             "k_hat": r["tau"][str(t)]["k_hat"]})
    for r in traw:
        for t in TAUS:
            if r["tau"][str(t)]["tie"]:
                ties.append({"arm": "trained", "aggregation": r["checkpoint"],
                             "axis": r["axis"], "pair": r["pair"], "tau": t,
                             "n_at_min": r["tau"][str(t)]["n_at_min"],
                             "k_hat": r["tau"][str(t)]["k_hat"]})

    results = {
        "experiment_id": "EXP-CORRESPONDENCE-GEOM-002",
        "record": "phase2/diagnostics/correspondence_geom/20260913T013006Z",
        "status": "COMPLETE" if trained or decision == "G-STOP" else "INCOMPLETE",
        "verdict": verdict,
        "comparison": comparison,
        "gate": {k: gate[k] for k in
                 ("axis", "n_units", "n_readout_passes", "unit_structure",
                  "h_rand_histogram", "h_rand_max", "h_rand_min", "h_rand_mean",
                  "hit_rate", "chance_rate", "decision", "H_star",
                  "offset_constant_units", "tracking_with_offset_units",
                  "trained_aggregation_loaded", "wall_clock_s")},
        "mask": {"rows": [MASK_Y0, MASK_Y1], "cols": [MASK_X0, MASK_X1],
                 "pixels": EXPECTED_MASK, "retained_rows": MASK_Y1 - MASK_Y0,
                 "retained_cols": MASK_X1 - MASK_X0},
        "domain": {"extractor": [EXT_LO, EXT_HI], "cost_volume": [CV_LO, CV_HI],
                   "statistic": [STAT_LO, STAT_HI],
                   "pixel": [MASK_X0, MASK_X1 - 1]},
        "taus": TAUS, "seeds": SEEDS, "pairs": PAIRS,
        "checkpoints": {k: v["sha256"] for k, v in CHECKPOINTS.items()},
        "controls_not_run": {
            "NEG_shift_none": "excluded by the authorisation (load exactly these "
                              "three H2 checkpoints). This record therefore carries "
                              "ONE null -- the random-aggregation gate -- and no "
                              "architecture-level null."},
        "tie_events": ties,
        "n_tie_events": len(ties),
        "claim_ceiling": {
            "permitted": ("Under the specified synthetic horizontal pairing "
                          "intervention, the trained nonlinear aggregation-plus-"
                          "readout response tracks the known geometric pairing "
                          "displacement more strongly than the preregistered "
                          "random-aggregation population.")
            if verdict == "EXCEEDS-RANDOM-GATE" else
            "TRUE CORRESPONDENCE -- NOT DEMONSTRATED",
            "prohibited": ["full stereo correspondence", "general disparity search",
                           "correct disparity on natural scenes",
                           "metric depth recovery", "causal uniqueness",
                           "architectural necessity", "generalization",
                           "absence of correspondence"],
        },
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log("verdict: %s   H*=%r" % (verdict, hstar))
    log("results.json, metrics.csv, per_scene.csv, per_tau.csv written")
    return 0


def write_hard_stop(stage: str, exc: BaseException | None) -> None:
    """hard_stop.json -- required output whether or not one fired."""
    p = OUT / "hard_stop.json"
    prior = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"stages": {}}
    if exc is None:
        prior["stages"][stage] = {"fired": False, "hard_stop": None}
    else:
        prior["stages"][stage] = {
            "fired": True,
            "class": type(exc).__name__,
            "hard_stop": str(exc),
            "traceback": traceback.format_exc().splitlines()[-12:],
            "action": "the process raised and exited non-zero; nothing was "
                      "repaired and continued",
        }
    prior["any_fired"] = any(v["fired"] for v in prior["stages"].values())
    prior["policy"] = ("any hard stop RAISES and the process exits; no post-hoc "
                       "restart is permitted without a new preregistration")
    p.write_text(json.dumps(prior, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["0", "1", "2", "w4", "report"])
    args = ap.parse_args()
    lines = []

    def log(s: str) -> None:
        lines.append(s)
        print(s, flush=True)

    exc = None
    try:
        rc = {"0": stage0, "1": stage1, "2": stage2,
              "w4": stage_w4, "report": stage_report}[args.stage](log)
    except BaseException as e:                                   # noqa: BLE001
        exc = e
        log("HARD STOP: %s" % e)
        raise
    finally:
        with open(OUT / ("run_stage%s.log" % args.stage), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        write_hard_stop(args.stage, exc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
