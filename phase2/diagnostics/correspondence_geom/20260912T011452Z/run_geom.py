"""EXP-CORRESPONDENCE-GEOM-001 -- Pairing-Swap Interaction Diagnostic.

Executes EXACTLY the protocol frozen in PREREGISTRATION.md / spec.json.

INFERENCE ONLY.  No optimizer is imported or constructed.  No .backward().
No .step().  No weight is modified anywhere in this file.  No architecture,
feature extractor, cost-volume construction, standardisation, readout or mask
is modified.  No historical record is written to.  No EPE / D1 / RMSE.

TWO PROCESSES, ENFORCED:

    --stage 1   the RANDOM-WEIGHT GATE.  Loads checkpoints ONLY for their
                feature extractors, sets model.aggregation = None immediately,
                and contains NO stage-2 code path.  Writes gate.json.
    --stage 2   the TRAINED arms.  REFUSES TO RUN unless gate.json says
                G-PASS.  Writes trained.json.
    --stage w4  determinism: recompute one complete trained unit in a fresh
                process and dump its bits for byte comparison.

Any hard stop RAISES and the process exits non-zero.  The experiment is never
repaired and continued.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
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

# ---------------------------------------------------------------- frozen spec
W, H, STRIDE, D = 1232, 368, 16, 12
CROP_W, CROP_H = 1136, 272
FW, FH = CROP_W // STRIDE, CROP_H // STRIDE          # 71 x 17
TAUS = [1, 2, 3, 4, 5, 6]
CELLS = ["AA", "BB", "AB", "BA"]
POSITIVE, NEGATIVE = ["AA", "BB"], ["AB", "BA"]
MASK_Y0, MASK_Y1, MASK_X0, MASK_X1 = 64, 208, 64, 864
EXPECTED_MASK = (MASK_Y1 - MASK_Y0) * (MASK_X1 - MASK_X0)               # 115200
SAFE_W = 54                                                             # feature cols free of shift_left fill
CANCEL_TOL = 1e-5                                                       # float32 round-off bound, 10x margin
SPLIT = "hailo_val"
PAIRS = [(1, 2), (3, 4), (10, 11), (12, 13), (14, 15), (16, 17)]
TRAINED = ["H2_seed0", "H2_seed1", "H2_seed2"]
SEARCH_FREE = "NEG_shift_none"
SEEDS = list(range(32))
W4_UNIT = ("H2_seed0", 0, "horizontal")


class HardStop(RuntimeError):
    """spec.json hard_stops.  Every one of these HALTS.  No post-hoc rescue."""


# ------------------------------------------------------------------- helpers
def sha256_bytes(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def bits(x: float) -> str:
    return struct.pack("<d", float(x)).hex()


def verify_freeze(stage: str) -> dict:
    """frozen.sha256 over file bytes, and historical.sha256 (HS8)."""
    frozen = {}
    for line in (OUT / "frozen.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            frozen[name.strip()] = digest
    for name, digest in frozen.items():
        actual = sha256_bytes(OUT / name)
        if actual != digest:
            raise HardStop("FREEZE VIOLATION %s: %s != %s" % (name, actual, digest))
    hist = {}
    for line in (OUT / "historical.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            hist[name.strip()] = digest
    for name, digest in hist.items():
        actual = sha256_bytes(REPO_ROOT / name)
        if actual != digest:
            raise HardStop("HS8 historical record modified: %s" % name)
    forbidden = {"1": ["gate.json", "results.json", "RESULTS.md"],
                 "2": ["trained.json", "results.json", "RESULTS.md"],
                 "w4": []}[stage]
    for f in forbidden:
        if (OUT / f).exists():
            raise HardStop("%s exists before stage %s" % (f, stage))
    return {"frozen": frozen, "historical_verified": len(hist)}


def setup_determinism() -> str:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise HardStop("HS6: CUDA required by the frozen determinism block")
    return "cuda"


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


def verify_geometry_from_arrays(src: np.ndarray, left_c: np.ndarray,
                                axis: str, t: int) -> None:
    """HS1 -- asserted FROM THE ACTUAL ARRAYS, never assumed."""
    y0, x0 = right_origin(axis, t)
    right_c = crop(src, y0, x0)
    if axis == "horizontal":
        ok = np.array_equal(right_c[:, :CROP_W - t], left_c[:, t:])
        msg = "right[:, :-t] != left[:, t:]"
    else:
        ok = np.array_equal(right_c[:CROP_H - t, :], left_c[t:, :])
        msg = "right[:-t, :] != left[t:, :]"
    if not ok:
        raise HardStop("HS1 geometry check failed, axis=%s t=%d: %s" % (axis, t, msg))


def load_model(spec: dict, key: str, device: str):
    meta = spec["checkpoints"][key]
    path = REPO_ROOT / meta["path"]
    actual = sha256_bytes(path)
    if actual != meta["sha256"]:
        raise HardStop("HS5 %s sha256 %s != frozen %s" % (key, actual, meta["sha256"]))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = StereoNet(StereoNetConfig(cost_volume_shift=meta["shift"]))
    scaled_regression.apply_to(model)
    model.load_state_dict(ckpt["model"])
    if sum(p.numel() for p in model.regression.parameters()) != 0:
        raise HardStop("HS5 %s readout has parameters" % key)
    if model.cost_volume.shift != meta["shift"] or model.config.num_disparities != D:
        raise HardStop("HS5 %s config mismatch" % key)
    return model.eval().to(device), actual


def scene_crops(pair_index: int, device: str):
    """Left crops and the 4x6 right crops for one pair, both axes."""
    ia, ib = PAIRS[pair_index]
    sa, sb = core.load_scene(ia, split=SPLIT), core.load_scene(ib, split=SPLIT)
    la, lb = crop(sa.left, 0, 0), crop(sb.left, 0, 0)
    return {"A": {"src": sa.left, "left": la, "name": sa.name},
            "B": {"src": sb.left, "left": lb, "name": sb.name}}


def build_volumes(model, sc: dict, axis: str, device: str, checks: dict) -> dict:
    """24 cost volumes for one (model, pair, axis).  Aggregation-independent."""
    lf = {}
    for who in ("A", "B"):
        lf[who] = model.feature_extractor(
            torch.from_numpy(normalize(sc[who]["left"])).to(device))
    vols = {}
    for tau in TAUS:
        t = tau * STRIDE
        rf = {}
        for who in ("A", "B"):
            verify_geometry_from_arrays(sc[who]["src"], sc[who]["left"], axis, t)
            y0, x0 = right_origin(axis, t)
            checks["right_origins"].add((axis, t, y0, x0))
            rc = crop(sc[who]["src"], y0, x0)
            checks["crop_shapes"].add(rc.shape[:2])
            rf[who] = model.feature_extractor(
                torch.from_numpy(normalize(rc)).to(device))
        # HS2 -- the exact AA/BB/AB/BA construction; each frame once as a left
        # and once as a right.
        plan = {"AA": ("A", "A"), "BB": ("B", "B"), "AB": ("A", "B"), "BA": ("B", "A")}
        lefts = sorted(p[0] for p in plan.values())
        rights = sorted(p[1] for p in plan.values())
        if lefts != ["A", "A", "B", "B"] or rights != ["A", "A", "B", "B"]:
            raise HardStop("HS2 pairing plan is not balanced: %r" % plan)
        taus_used = set()
        for cell, (li, ri) in plan.items():
            v = model.cost_volume(lf[li], rf[ri])
            if tuple(v.shape) != (1, 32, D, FH, FW):
                raise HardStop("unexpected volume shape %s" % (tuple(v.shape),))
            checks["volume_shapes"].add(tuple(v.shape))
            vols[(cell, tau)] = v
            taus_used.add(t)
        if taus_used != {t}:                                      # HS3
            raise HardStop("HS3 tau differs across cells at t=%d: %r" % (t, taus_used))

        # HS1 -- matched volume exactly zero at k = tau (horizontal, shift=left)
        if axis == "horizontal" and model.cost_volume.shift == "left":
            for cell in POSITIVE:
                z = float(vols[(cell, tau)][0, :, tau, :, :SAFE_W].abs().max())
                checks["zero_at_tau"].append(z)
                if z != 0.0:
                    raise HardStop("HS1 %s |V[:,tau=%d,:,w<%d]| = %r != 0"
                                   % (cell, tau, SAFE_W, z))

        # HS4 -- cost-volume 2x2 cancellation
        resid = (vols[("AA", tau)] + vols[("BB", tau)]
                 - vols[("AB", tau)] - vols[("BA", tau)])
        scale = float(vols[("AB", tau)].abs().max())
        rel = float(resid.abs().max()) / max(scale, 1e-12)
        checks["cancellation"].append({"axis": axis, "tau": tau,
                                       "max_abs_residual": float(resid.abs().max()),
                                       "volume_scale": scale, "relative": rel})
        if rel > CANCEL_TOL:
            raise HardStop("HS4 cancellation %.3e > %.1e at tau=%d axis=%s"
                           % (rel, CANCEL_TOL, tau, axis))
    return vols


def cell_readout(agg, regression, volume: torch.Tensor):
    """The FROZEN readout module, used verbatim, then the frozen mask.

    `regression` is the checkpoint's own StandardisedDisparityRegression.  Its
    instrumentation flag `capture` stores the exact tensor the softmax consumes
    (`softmax_input`) and changes no arithmetic.  The mask is a rectangle, so
    masking is a slice; the standardisation and the soft-argmin are pointwise
    in (y, x), so slicing after them is exactly masking.
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
    zs = z[:, :, MASK_Y0:MASK_Y1, MASK_X0:MASK_X1]             # (1, D, 144, 800)
    ms = m[:, :, MASK_Y0:MASK_Y1, MASK_X0:MASK_X1]
    if zs.shape[-2] * zs.shape[-1] != EXPECTED_MASK:
        raise HardStop("HS9 mask count %d != %d"
                       % (zs.shape[-2] * zs.shape[-1], EXPECTED_MASK))
    return ms, zs


def unit_measure(agg, regression, vols: dict) -> dict:
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
        zbar = {c: np.median(arrs[c], axis=1) for c in CELLS}
        ibar = np.median(ia, axis=1)
        amin, amax = int(np.argmin(ibar)), int(np.argmax(ibar))
        out["tau"][tau] = {
            "Ibar": [float(v) for v in ibar],
            "argmin": amin, "argmax": amax,
            "hit": bool(amin == tau), "anchor": amin - tau,
            "amplitude": float(ibar.max() - ibar.min()),
            "zbar": {c: [float(v) for v in zbar[c]] for c in CELLS},
            "m": {c: float(np.median(
                ms[c].detach().float().cpu().numpy().astype(np.float64)))
                for c in CELLS},
            "Ibar_bits": [bits(v) for v in ibar],
        }
    out["h"] = int(sum(out["tau"][t]["hit"] for t in TAUS))
    out["anchors"] = [out["tau"][t]["anchor"] for t in TAUS]
    out["anchor_constant"] = bool(len(set(out["anchors"])) == 1)
    return out


def new_checks() -> dict:
    return {"crop_shapes": set(), "volume_shapes": set(), "right_origins": set(),
            "zero_at_tau": [], "cancellation": []}


def checks_to_json(c: dict) -> dict:
    return {"crop_shapes": sorted(map(list, c["crop_shapes"])),
            "volume_shapes": sorted(map(list, c["volume_shapes"])),
            "n_right_origins": len(c["right_origins"]),
            "right_origins": sorted(map(list, c["right_origins"])),
            "zero_at_tau_max": (max(c["zero_at_tau"]) if c["zero_at_tau"] else None),
            "zero_at_tau_n": len(c["zero_at_tau"]),
            "cancellation_max_relative": (max(x["relative"] for x in c["cancellation"])
                                          if c["cancellation"] else None),
            "cancellation": c["cancellation"]}


# ============================================================== STAGE 1: GATE
def stage1(log) -> int:
    meta = verify_freeze("1")
    log("freeze verified: %d frozen artifacts, %d historical records"
        % (len(meta["frozen"]), meta["historical_verified"]))
    spec = json.loads((OUT / "spec.json").read_text(encoding="utf-8"))
    device = setup_determinism()
    t0 = time.time()

    checks = new_checks()
    units, raw = [], []
    n_readout = 0
    for ext in TRAINED:
        model, digest = load_model(spec, ext, device)
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
            sc = scene_crops(pi, device)
            vols = build_volumes(model, sc, "horizontal", device, checks)
            for seed in SEEDS:
                torch.manual_seed(seed)
                agg = Aggregation(in_channels=32, channels=32, num_layers=4).eval().to(device)
                if sum(p.numel() for p in agg.parameters()) != 111585:
                    raise HardStop("HS5 random aggregation param count %d"
                                   % sum(p.numel() for p in agg.parameters()))
                u = unit_measure(agg, model.regression, vols)
                n_readout += len(TAUS) * len(CELLS)
                units.append({"extractor": ext, "pair": pi, "seed": seed,
                              "h": u["h"], "anchors": u["anchors"],
                              "anchor_constant": u["anchor_constant"]})
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
    if len(units) != 576:
        raise HardStop("gate unit count %d != 576" % len(units))
    hmax = max(h_rand)
    decision = "G-STOP" if hmax == 6 else "G-PASS"
    hstar = None if decision == "G-STOP" else hmax

    hist = {i: int(sum(1 for x in h_rand if x == i)) for i in range(7)}
    gate = {
        "stage": 1, "n_units": len(units), "n_readout_passes": n_readout,
        "unit_structure": "(extractor, pair, seed) = 3 x 6 x 32",
        "axis": "horizontal",
        "h_rand_histogram": hist,
        "h_rand_max": hmax, "h_rand_min": min(h_rand),
        "h_rand_mean": float(np.mean(h_rand)),
        "hit_rate": float(sum(h_rand) / (len(h_rand) * len(TAUS))),
        "chance_rate": 1.0 / D,
        "anchor_constant_units": int(sum(1 for u in units if u["anchor_constant"])),
        "decision": decision, "H_star": hstar,
        "trained_aggregation_loaded": False,
        "checks": checks_to_json(checks),
        "units": units,
        "wall_clock_s": time.time() - t0,
    }
    (OUT / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    (OUT / "gate_raw.json").write_text(json.dumps(raw), encoding="utf-8")
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

    spec = json.loads((OUT / "spec.json").read_text(encoding="utf-8"))
    device = setup_determinism()
    t0 = time.time()
    checks = new_checks()
    results = {"horizontal": [], "vertical": [], "search_free": []}
    raw, n_readout = [], 0

    for key in TRAINED + [SEARCH_FREE]:
        model, digest = load_model(spec, key, device)
        agg_hash = hashlib.sha256(
            "".join(hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
                    for _, p in model.aggregation.named_parameters()).encode()
        ).hexdigest()
        log("%s loaded  ckpt %s...  shift=%s  aggregation %s..."
            % (key, digest[:12], model.cost_volume.shift, agg_hash[:12]))
        for pi in range(len(PAIRS)):
            sc = scene_crops(pi, device)
            h_by_axis = {}
            for axis in ("horizontal", "vertical"):
                vols = build_volumes(model, sc, axis, device, checks)
                u = unit_measure(model.aggregation, model.regression, vols)
                n_readout += len(TAUS) * len(CELLS)
                arm = "search_free" if key == SEARCH_FREE else axis
                rec = {"checkpoint": key, "pair": pi, "axis": axis,
                       "pair_names": [sc["A"]["name"], sc["B"]["name"]],
                       "h": u["h"], "anchors": u["anchors"],
                       "anchor_constant": u["anchor_constant"],
                       "argmins": [u["tau"][t]["argmin"] for t in TAUS],
                       "argmaxs": [u["tau"][t]["argmax"] for t in TAUS],
                       "amplitudes": [u["tau"][t]["amplitude"] for t in TAUS]}
                results[arm].append(rec)
                raw.append({"checkpoint": key, "pair": pi, "axis": axis,
                            "pair_names": [sc["A"]["name"], sc["B"]["name"]],
                            "h": u["h"],
                            "tau": {str(k): v for k, v in u["tau"].items()}})
                h_by_axis[axis] = u["h"]
                del vols
                torch.cuda.empty_cache()
            log("%s pair %d (%s/%s) done  h_horiz=%d h_vert=%d  (%.1fs)"
                % (key, pi, sc["A"]["name"], sc["B"]["name"],
                   h_by_axis["horizontal"], h_by_axis["vertical"],
                   time.time() - t0))
        del model

    # W4 bits for the designated unit
    w4 = [r for r in raw if r["checkpoint"] == W4_UNIT[0]
          and r["pair"] == W4_UNIT[1] and r["axis"] == W4_UNIT[2]][0]
    w4_bits = {str(t): w4["tau"][str(t)]["Ibar_bits"] for t in TAUS}
    (OUT / "w4_inline.json").write_text(json.dumps(w4_bits, indent=2), encoding="utf-8")

    out = {"stage": 2, "H_star": hstar, "n_readout_passes": n_readout,
           "arms": results, "checks": checks_to_json(checks),
           "wall_clock_s": time.time() - t0}
    (OUT / "trained.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT / "trained_raw.json").write_text(json.dumps(raw), encoding="utf-8")
    log("stage 2 wall_clock_s=%.2f  readouts=%d" % (time.time() - t0, n_readout))
    return 0


# ================================================================= STAGE W4
def stage_w4(log) -> int:
    verify_freeze("w4")
    spec = json.loads((OUT / "spec.json").read_text(encoding="utf-8"))
    device = setup_determinism()
    key, pi, axis = W4_UNIT
    model, _ = load_model(spec, key, device)
    sc = scene_crops(pi, device)
    checks = new_checks()
    vols = build_volumes(model, sc, axis, device, checks)
    u = unit_measure(model.aggregation, model.regression, vols)
    out = {str(t): u["tau"][t]["Ibar_bits"] for t in TAUS}
    (OUT / "w4_restart.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    ref = json.loads((OUT / "w4_inline.json").read_text(encoding="utf-8"))
    identical = all(ref[str(t)] == out[str(t)] for t in TAUS)
    log("W4 unit %s pair %d %s : bit-identical after restart = %s"
        % (key, pi, axis, identical))
    if not identical:
        raise HardStop("HS6 W4 repeat is NOT bit-identical")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["1", "2", "w4"])
    args = ap.parse_args()
    lines = []

    def log(s: str) -> None:
        lines.append(s)
        print(s, flush=True)

    try:
        rc = {"1": stage1, "2": stage2, "w4": stage_w4}[args.stage](log)
    finally:
        with open(OUT / ("run_stage%s.log" % args.stage), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
