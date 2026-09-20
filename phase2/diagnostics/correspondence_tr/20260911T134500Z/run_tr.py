"""EXP-CORRESPONDENCE-TR-001 -- Trained-vs-Random Geometric Separation.

Executes EXACTLY the protocol frozen in PREREGISTRATION.md / spec.json, as
qualified by PRE_EXECUTION_ADDENDUM.md (the slope fit window is T_FIT, six
levels, IDENTICAL to the 384 frozen random values; the seven-level literal
reading is recorded as alpha_h7 and compared to nothing).

INFERENCE ONLY. No optimizer is imported or constructed. No .backward().
No .step(). No weight is modified anywhere in this file.

The random reference is READ ONLY and is never regenerated, re-scored or
re-run, whatever the trained results look like.

No p-value is computed anywhere. Units are not independent replicates.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CUBLAS = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

from src.models.stereonet.stereonet import StereoNet, StereoNetConfig   # noqa: E402
from src.datasets.kitti2015 import normalize                            # noqa: E402
from phase2.models import scaled_regression                             # noqa: E402
from phase2.viz import core                                             # noqa: E402

OUT = Path(__file__).resolve().parent
W, H, STRIDE = 1232, 368, 16
TMAX = 96
CROP_W, CROP_H = W - TMAX, H - TMAX            # 1136 x 272
BORDER = 64
EXPECTED_MASK = 145152
T_LEVELS = [0, 16, 32, 48, 64, 80, 96]
T_FIT = [16, 32, 48, 64, 80, 96]               # PRIMARY fit window (reference-identical)
AXES = ["horizontal", "vertical"]
RAIL_LOW, RAIL_HIGH, FLAT_RANGE = 1.0, 10.0, 1.0     # inherited unchanged
SPLIT = "hailo_val"
TRAINED = ["H2_seed0", "H2_seed1", "H2_seed2"]
SEARCH_FREE = "NEG_shift_none"
CLASSES = ["CLIPPED", "FLAT", "ANTI_CORRELATED", "NON_MONOTONE",
           "NON_SPECIFIC", "QUALIFYING"]


class HardStop(RuntimeError):
    """spec.json hard_stops. Every one of these HALTS. No post-hoc rescue."""


def sha256_bytes(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def arr_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def param_sha256(module) -> str:
    return hashlib.sha256("".join(tensor_sha256(p) for _, p in
                                  module.named_parameters()).encode()).hexdigest()


def bits(x: float) -> str:
    """Exact IEEE-754 identity. Used for the W4 bit-identical comparison."""
    return struct.pack("<d", x).hex()


def crop(img: np.ndarray, y0: int, x0: int) -> np.ndarray:
    if y0 < 0 or x0 < 0 or y0 + CROP_H > H or x0 + CROP_W > W:
        raise HardStop("crop window (%d,%d) leaves the source -> would need fill" % (y0, x0))
    out = img[y0:y0 + CROP_H, x0:x0 + CROP_W]
    if out.shape[:2] != (CROP_H, CROP_W):
        raise HardStop("crop produced %s" % (out.shape[:2],))
    return out


def right_origin(axis: str, t: int) -> tuple[int, int]:
    """Frozen: a_R = a_L + t. Axis is an argument, not a branch."""
    if axis == "horizontal":
        return 0, t
    if axis == "vertical":
        return t, 0
    raise HardStop("unknown axis %r" % axis)


def verify_sign_from_data(src: np.ndarray, left_c: np.ndarray, t: int) -> None:
    """W2 / sign verification -- asserted from the ACTUAL ARRAYS, not assumed.

    right_c[y, x] must equal left_c[y, x + t]: a scene feature sitting at left
    column x+t sits at right column x, i.e. d = a_L - a_R = +t px, k_true = t/16.
    """
    if t == 0:
        return
    right_c = crop(src, 0, t)
    if not np.array_equal(right_c[:, :CROP_W - t], left_c[:, t:]):
        raise HardStop("horizontal sign check failed at t=%d: right[:, :-t] != left[:, t:]" % t)


def slope(m: dict, levels: list[int]) -> tuple[float, float]:
    """FREE-INTERCEPT OLS of m(t) on u = t/16. Through-origin is PROHIBITED."""
    x = np.array([t / STRIDE for t in levels], dtype=np.float64)
    y = np.array([m[t] for t in levels], dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    a = float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))
    return a, float(ym - a * xm)


def classify(mh: dict, mv: dict) -> tuple[str, dict]:
    """Frozen ordered taxonomy, inherited verbatim. First match wins.
    CLIPPED is evaluated SECOND, before every ordinary failure class."""
    vals_h = [mh[t] for t in T_LEVELS]
    vals_v = [mv[t] for t in T_LEVELS]
    if not (all(np.isfinite(vals_h)) and all(np.isfinite(vals_v))):
        raise HardStop("non-finite response -> implementation fault")
    lo, hi = min(vals_h), max(vals_h)
    a_h, b_h = slope(mh, T_FIT)
    a_v, b_v = slope(mv, T_FIT)
    a_h7, _ = slope(mh, T_LEVELS)      # reported only; compared to nothing
    a_v7, _ = slope(mv, T_LEVELS)
    diffs = [vals_h[i + 1] - vals_h[i] for i in range(len(vals_h) - 1)]
    Q1 = bool(a_h > 0)
    Q2 = bool(all(d > 0 for d in diffs))
    Q3 = bool(abs(a_h) > abs(a_v))
    facts = {"alpha_h": a_h, "beta_h": b_h, "alpha_v": a_v, "beta_v": b_v,
             "alpha_h7": a_h7, "alpha_v7": a_v7,
             "min_h": lo, "max_h": hi, "range_h": hi - lo,
             "first_diffs": diffs,
             "n_pos_diffs": int(sum(1 for d in diffs if d > 0)),
             "n_neg_diffs": int(sum(1 for d in diffs if d < 0)),
             "min_diff": float(min(diffs)), "max_diff": float(max(diffs)),
             "Q1": Q1, "Q2": Q2, "Q3": Q3}
    if lo <= RAIL_LOW or hi >= RAIL_HIGH:
        return "CLIPPED", facts
    if (hi - lo) < FLAT_RANGE:
        return "FLAT", facts
    if a_h < 0:
        return "ANTI_CORRELATED", facts
    if not Q2:
        return "NON_MONOTONE", facts
    if not Q3:
        return "NON_SPECIFIC", facts
    return "QUALIFYING", facts


# --------------------------------------------------------------------------
#  model / measurement primitives
# --------------------------------------------------------------------------

def load_model(spec: dict, key: str, shift: str, device: str):
    path = REPO_ROOT / spec["path"]
    actual = sha256_bytes(path)
    if actual != spec["sha256"]:
        raise HardStop("%s sha256 %s != frozen %s" % (key, actual, spec["sha256"]))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = StereoNet(StereoNetConfig(cost_volume_shift=shift))
    scaled_regression.apply_to(model)
    model.load_state_dict(ckpt["model"])
    if sum(p.numel() for p in model.regression.parameters()) != 0:
        raise HardStop("%s readout has parameters" % key)
    if model.cost_volume.shift != shift or model.config.num_disparities != 12:
        raise HardStop("%s config mismatch" % key)
    return model.eval().to(device), actual


def build_volumes(model, src: np.ndarray, left_c: np.ndarray, device: str,
                  sanity: dict) -> tuple[dict, int]:
    """13 distinct conditions. Aggregation-independent -> cached."""
    vols, n = {}, 0
    L = torch.from_numpy(normalize(left_c)).to(device)
    with torch.no_grad():
        lf = model.feature_extractor(L)
        for axis in AXES:
            for t in T_LEVELS:
                if t == 0 and axis == "vertical":
                    vols[(axis, 0)] = vols[("horizontal", 0)]
                    continue
                y0, x0 = right_origin(axis, t)
                if (axis == "horizontal" and (y0, x0) != (0, t)) or \
                   (axis == "vertical" and (y0, x0) != (t, 0)):
                    raise HardStop("W2: right origin %r wrong for %s t=%d" % ((y0, x0), axis, t))
                sanity["right_origins"].add((axis, t, y0, x0))
                right_c = crop(src, y0, x0)
                sanity["crop_shapes"].add(right_c.shape[:2])
                rf = model.feature_extractor(torch.from_numpy(normalize(right_c)).to(device))
                v = model.cost_volume(lf, rf)
                if tuple(v.shape) != (1, 32, 12, 17, 71):
                    raise HardStop("unexpected volume shape %s" % (tuple(v.shape),))
                sanity["volume_shapes"].add(tuple(v.shape))
                vols[(axis, t)] = v
                n += 1
    return vols, n


def readout(model, vols: dict, mask: np.ndarray) -> tuple[dict, dict, int]:
    """14 readout passes: 7 horizontal + 7 vertical. The t=0 vertical pass
    re-runs the shared volume -- replicating the ARCH-RATE procedure exactly so
    the measurement path is identical between the two populations."""
    mh, mv, n = {}, {}, 0
    with torch.no_grad():
        for axis in AXES:
            for t in T_LEVELS:
                d = model.regression(model.aggregation(vols[(axis, t)]), (CROP_H, CROP_W))
                a = d[0, 0].detach().float().cpu().numpy().astype(np.float64)
                n += 1
                (mh if axis == "horizontal" else mv)[t] = float(np.median(a[mask]))
    return mh, mv, n


def make_mask() -> np.ndarray:
    m = np.zeros((CROP_H, CROP_W), dtype=bool)
    m[BORDER:CROP_H - BORDER, BORDER:CROP_W - BORDER] = True
    if int(m.sum()) != EXPECTED_MASK:
        raise HardStop("mask count %d != frozen %d" % (int(m.sum()), EXPECTED_MASK))
    return m


def setup_determinism() -> str:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
#  W4 -- one complete unit, recomputed in a FRESH PROCESS
# --------------------------------------------------------------------------

def w4_unit(ext_key: str, agg_key: str, scene_index: int) -> dict:
    spec = json.loads((OUT / "spec.json").read_text(encoding="utf-8"))
    device = setup_determinism()
    mask = make_mask()
    sanity = {"crop_shapes": set(), "volume_shapes": set(), "right_origins": set()}
    em, _ = load_model(spec["checkpoints"][ext_key], ext_key, "left", device)
    sc = core.load_scene(scene_index, split=SPLIT)
    left_c = crop(sc.left, 0, 0)
    vols, _ = build_volumes(em, sc.left, left_c, device, sanity)
    if agg_key != ext_key:
        am, _ = load_model(spec["checkpoints"][agg_key], agg_key, "left", device)
        em.aggregation = am.aggregation
    mh, mv, _ = readout(em, vols, mask)
    return {"extractor": ext_key, "aggregation": agg_key, "scene_index": scene_index,
            "m_h_bits": [bits(mh[t]) for t in T_LEVELS],
            "m_v_bits": [bits(mv[t]) for t in T_LEVELS]}


# --------------------------------------------------------------------------
#  comparison against the frozen random population (read only)
# --------------------------------------------------------------------------

def compare(value: float, pop: list[float]) -> dict:
    a = np.sort(np.asarray(pop, dtype=np.float64))
    n = a.size
    below = int(np.sum(a < value))
    equal = int(np.sum(a == value))
    lo, hi = float(a[0]), float(a[-1])
    nb = [float(x) for x in a[a <= value]]
    na = [float(x) for x in a[a >= value]]
    return {
        "percentile_rank": 100.0 * (below + 0.5 * equal) / n,
        "empirical_cdf": (below + equal) / n,
        "n_random_below": below, "n_random_equal": equal,
        "n_random_above": n - below - equal,
        "nearest_random_below": (nb[-1] if nb else None),
        "nearest_random_above": (na[0] if na else None),
        "random_min": lo, "random_max": hi,
        "outside_observed_random_range": bool(value < lo or value > hi),
    }


def summary(v: list[float]) -> dict:
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(a.size), "min": float(a.min()), "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
            "max": float(a.max()), "mean": float(a.mean()),
            "std": (float(a.std(ddof=1)) if a.size > 1 else None)}


def rank_biserial(trained: list[float], random_pop: list[float]) -> dict:
    """Mann-Whitney U -> rank-biserial correlation, AS AN EFFECT SIZE ONLY.
    NO p-value is computed. Units are not independent replicates."""
    x = np.asarray(trained, dtype=np.float64)
    y = np.asarray(random_pop, dtype=np.float64)
    gt = float(np.sum(x[:, None] > y[None, :]))
    lt = float(np.sum(x[:, None] < y[None, :]))
    n = float(x.size * y.size)
    return {"U_trained_greater": gt, "U_trained_less": lt, "ties": n - gt - lt,
            "rank_biserial": (gt - lt) / n,
            "note": "EFFECT SIZE ONLY. No p-value. Units are not independent."}


def overlap_fraction(trained: list[float], lo: float, hi: float) -> float:
    return float(np.mean([(lo <= v <= hi) for v in trained]))


# --------------------------------------------------------------------------

def main() -> int:
    t0 = time.time()
    log = []

    def say(s: str) -> None:
        log.append(s)
        print(s, flush=True)

    # ================= A. PRE-EXECUTION FREEZE VERIFICATION =================
    for f in ["PREREGISTRATION.md", "DESIGN_AUDIT.md", "EXECUTION_PROMPT.md",
              "PRE_EXECUTION_ADDENDUM.md", "spec.json", "random_reference.json",
              "frozen.sha256", "documents.sha256"]:
        if not (OUT / f).exists():
            raise HardStop("missing freeze artifact %s" % f)
    for f in ["results.json", "results.csv", "RESULTS.md", "AUDIT.md"]:
        if (OUT / f).exists():
            raise HardStop("%s exists before execution" % f)

    pre = (OUT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    frozen = {}
    for fname in ("frozen.sha256", "documents.sha256"):
        for line in (OUT / fname).read_text(encoding="utf-8").splitlines():
            if line.strip():
                h, n = line.split()
                frozen[n.lstrip("*")] = h
    for n, h in frozen.items():
        a = sha256_bytes(OUT / n)
        if a != h:
            raise HardStop("%s file-bytes sha256 %s != frozen %s" % (n, a, h))
    for n in ("spec.json", "random_reference.json"):
        if frozen[n] not in pre:
            raise HardStop("%s digest not recorded in PREREGISTRATION.md" % n)
    say("freeze verified: %d artifacts, all file-bytes digests OK" % len(frozen))

    spec = json.loads((OUT / "spec.json").read_text(encoding="utf-8"))
    ref = json.loads((OUT / "random_reference.json").read_text(encoding="utf-8"))

    # ---- the random reference, re-verified at its source ------------------
    src_results = REPO_ROOT / ref["source_record"] / "results.json"
    a = sha256_bytes(src_results)
    if a != ref["source_results_sha256"]:
        raise HardStop("random reference results.json sha256 %s != %s"
                       % (a, ref["source_results_sha256"]))
    RR = json.loads(src_results.read_text(encoding="utf-8"))
    for k, want in (("construction_spec_sha256", ref["construction_spec_sha256"]),
                    ("randomisation_spec_sha256", ref["randomisation_spec_sha256"]),
                    ("classification_spec_sha256", ref["classification_spec_sha256"])):
        if RR[k] != want:
            raise HardStop("random reference %s mismatch" % k)
    if len(RR["units"]) != 384 or ref["n"] != 384:
        raise HardStop("random reference is not 384 units")
    if sorted(ref["alpha_h_values"]) != sorted(u["alpha_h"] for u in RR["units"]):
        raise HardStop("frozen alpha_h values do not match the source record")
    if RR["scenes"] != spec["scenes"] or ref["scenes"] != spec["scenes"]:
        raise HardStop("scene set differs from the random reference")
    if RR["mask_rule"] != spec["mask"]["definition_crop_coords"]:
        raise HardStop("mask rule differs from the random reference")
    if RR["t_levels_px"] != T_LEVELS or RR["t_fit_levels_px"] != T_FIT:
        raise HardStop("translation levels differ from the random reference")
    say("random reference verified: 384 units, %s, spec hashes OK"
        % ref["source_results_sha256"][:12])

    # ---- the fit window, settled against the reference's own raw curves ----
    for u in RR["units"][:32]:
        mh = {t: u["m_h"][str(t)] for t in T_LEVELS}
        if slope(mh, T_FIT)[0] != u["alpha_h"]:
            raise HardStop("T_FIT slope does not reproduce the frozen alpha_h")
    say("PRIMARY statistic reproduces the frozen random alpha_h exactly "
        "(T_FIT, six levels; see PRE_EXECUTION_ADDENDUM.md)")

    if spec["construction"]["t_levels_px"] != T_LEVELS:
        raise HardStop("t levels differ from spec.json")
    if spec["mask"]["retained_pixels_per_scene"] != EXPECTED_MASK:
        raise HardStop("mask size differs from spec.json")
    if spec["classification"]["order"] != ["NONFINITE"] + CLASSES:
        raise HardStop("classification order differs from spec.json")

    device = setup_determinism()
    mask = make_mask()
    SCENES, SNAMES = spec["scenes"], spec["scene_names"]
    say("device=%s torch=%s numpy=%s" % (device, torch.__version__, np.__version__))
    say("scenes %r %r" % (SCENES, SNAMES))

    sanity = {"crop_shapes": set(), "volume_shapes": set(), "right_origins": set(),
              "left_crop_hashes_per_scene": {}, "sign_checks": 0,
              "trained_agg_hashes": {}, "agg_identity_asserted": 0}
    n_vol = n_fwd = 0
    units = []

    # ================= TRAINED ARM: full 3 x 3 x 4 cross ====================
    ckpt_hashes = {}
    aggregations = {}
    for k in TRAINED:                       # load the three trained aggregations once
        m, h = load_model(spec["checkpoints"][k], k, "left", device)
        ckpt_hashes[k] = h
        aggregations[k] = m.aggregation
        sanity["trained_agg_hashes"][k] = param_sha256(m.aggregation)
    if len(set(sanity["trained_agg_hashes"].values())) != len(TRAINED):
        raise HardStop("two trained aggregations are identical")
    say("loaded %d trained checkpoints, aggregation hashes distinct" % len(TRAINED))

    for ekey in TRAINED:
        em, _ = load_model(spec["checkpoints"][ekey], ekey, "left", device)
        for si, sname in zip(SCENES, SNAMES):
            sc = core.load_scene(si, split=SPLIT)
            if sc.left.shape[:2] != (H, W):
                raise HardStop("scene %s is %s" % (sc.name, sc.left.shape[:2]))
            if sc.name != sname:
                raise HardStop("scene index %d is %s, spec says %s" % (si, sc.name, sname))
            left_c = crop(sc.left, 0, 0)
            sanity["crop_shapes"].add(left_c.shape[:2])
            sanity["left_crop_hashes_per_scene"].setdefault(sc.name, set()).add(arr_sha256(left_c))
            for t in T_LEVELS:               # sign verified from the arrays themselves
                verify_sign_from_data(sc.left, left_c, t)
                sanity["sign_checks"] += 1
            vols, nv = build_volumes(em, sc.left, left_c, device, sanity)
            n_vol += nv
            for akey in TRAINED:
                em.aggregation = aggregations[akey]
                if param_sha256(em.aggregation) != sanity["trained_agg_hashes"][akey]:
                    raise HardStop("aggregation identity assertion failed")
                sanity["agg_identity_asserted"] += 1
                mh, mv, nf = readout(em, vols, mask)
                n_fwd += nf
                cls, f = classify(mh, mv)
                units.append({
                    "arm": "TRAINED", "matched": (akey == ekey),
                    "extractor": ekey, "aggregation": akey,
                    "scene": sc.name, "scene_index": si,
                    "k_true": {str(t): t / STRIDE for t in T_LEVELS},
                    "m_h": {str(t): mh[t] for t in T_LEVELS},
                    "m_v": {str(t): mv[t] for t in T_LEVELS},
                    "classification": cls, **f,
                })
            say("done TRAINED  ext=%-9s scene=%-13s  3 aggregations" % (ekey, sc.name))
        del em

    # ================= SEARCH-FREE CONTROL (separate, never pooled) ========
    sm, sh = load_model(spec["checkpoints"][SEARCH_FREE], SEARCH_FREE, "none", device)
    ckpt_hashes[SEARCH_FREE] = sh
    if sm.cost_volume.shift != "none":
        raise HardStop("search-free control must run its own shift='none' construction")
    for si, sname in zip(SCENES, SNAMES):
        sc = core.load_scene(si, split=SPLIT)
        left_c = crop(sc.left, 0, 0)
        sanity["left_crop_hashes_per_scene"].setdefault(sc.name, set()).add(arr_sha256(left_c))
        vols, nv = build_volumes(sm, sc.left, left_c, device, sanity)
        n_vol += nv
        mh, mv, nf = readout(sm, vols, mask)
        n_fwd += nf
        cls, f = classify(mh, mv)
        units.append({
            "arm": "SEARCH_FREE", "matched": True,
            "extractor": SEARCH_FREE, "aggregation": SEARCH_FREE,
            "scene": sc.name, "scene_index": si,
            "k_true": {str(t): t / STRIDE for t in T_LEVELS},
            "m_h": {str(t): mh[t] for t in T_LEVELS},
            "m_v": {str(t): mv[t] for t in T_LEVELS},
            "classification": cls, **f,
        })
        say("done SEARCH-FREE scene=%-13s" % sc.name)
    del sm

    # ================= W1 ==================================================
    w1 = {k: len(v) for k, v in sanity["left_crop_hashes_per_scene"].items()}
    if set(w1.values()) != {1}:
        raise HardStop("W1: left crop not byte-identical across conditions: %r" % w1)
    say("W1 OK: left crop byte-identical across every condition, all %d scenes" % len(w1))

    # ================= W4: fresh process, bit-identical ====================
    tgt = next(u for u in units if u["arm"] == "TRAINED" and u["matched"])
    say("W4: recomputing %s/%s/%s in a fresh process"
        % (tgt["extractor"], tgt["aggregation"], tgt["scene"]))
    cp = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--w4",
                         tgt["extractor"], tgt["aggregation"], str(tgt["scene_index"])],
                        capture_output=True, text=True, cwd=str(REPO_ROOT))
    if cp.returncode != 0:
        raise HardStop("W4 subprocess failed:\n%s" % cp.stderr[-2000:])
    rep = json.loads(cp.stdout.strip().splitlines()[-1])
    want_h = [bits(tgt["m_h"][str(t)]) for t in T_LEVELS]
    want_v = [bits(tgt["m_v"][str(t)]) for t in T_LEVELS]
    if rep["m_h_bits"] != want_h or rep["m_v_bits"] != want_v:
        raise HardStop("W4 deterministic repeat is NOT bit-identical")
    say("W4 OK: all 14 values bit-identical after process restart")

    # ================= COMPARISON (random reference read only) =============
    POP = {"alpha_h": ref["alpha_h_values"], "alpha_v": ref["alpha_v_values"],
           "range_h": ref["range_h_values"], "n_pos_diffs": ref["positive_diff_counts"]}
    for u in units:
        u["vs_random"] = {q: compare(float(u[q]), POP[q]) for q in POP}

    def group(sel) -> dict:
        g = [u for u in units if sel(u)]
        if not g:
            return {}
        out = {"n": len(g),
               "tally": {c: sum(1 for u in g if u["classification"] == c) for c in CLASSES}}
        for q in POP:
            vals = [float(u[q]) for u in g]
            lo, hi = min(POP[q]), max(POP[q])
            n_out = sum(1 for u in g if u["vs_random"][q]["outside_observed_random_range"])
            out[q] = {
                "trained": summary(vals),
                "random": summary(POP[q]),
                "n_outside_observed_random_range": int(n_out),
                "outside_observed_random_range_rate": n_out / len(g),
                "overlap_fraction_inside_random_range": overlap_fraction(vals, lo, hi),
                "percentile_ranks": [u["vs_random"][q]["percentile_rank"] for u in g],
                "effect_size": rank_biserial(vals, POP[q]),
            }
        return out

    matched = [u for u in units if u["arm"] == "TRAINED" and u["matched"]]
    cross = [u for u in units if u["arm"] == "TRAINED"]
    free = [u for u in units if u["arm"] == "SEARCH_FREE"]
    if len(matched) != 12 or len(cross) != 36 or len(free) != 4:
        raise HardStop("unit counts %d/%d/%d != 12/36/4" % (len(matched), len(cross), len(free)))

    analysis = {
        "PRIMARY_MATCHED": group(lambda u: u["arm"] == "TRAINED" and u["matched"]),
        "SECONDARY_CROSS": group(lambda u: u["arm"] == "TRAINED"),
        "SEARCH_FREE_CONTROL": group(lambda u: u["arm"] == "SEARCH_FREE"),
        "MATCHED_by_checkpoint": {k: group(lambda u, k=k: u["arm"] == "TRAINED"
                                           and u["matched"] and u["extractor"] == k)
                                  for k in TRAINED},
        "MATCHED_by_scene": {s: group(lambda u, s=s: u["arm"] == "TRAINED"
                                      and u["matched"] and u["scene"] == s)
                             for s in SNAMES},
        "CROSS_by_aggregation": {k: group(lambda u, k=k: u["arm"] == "TRAINED"
                                          and u["aggregation"] == k) for k in TRAINED},
        "pooling_prohibited": "MATCHED and CROSS are never pooled. SEARCH_FREE is never "
                              "mixed into the random-weight null.",
        "p_value": None,
        "p_value_note": "No p-value is computed anywhere. Units are not independent replicates.",
        "invalid_inference_explicitly_rejected":
            "'0/384 random qualified therefore any trained qualification proves learning' "
            "is INVALID and is not used.",
    }

    # alpha_h7: the literal seven-level reading, recorded, compared to nothing
    analysis["alpha_h7_reported_only"] = {
        "MATCHED": summary([u["alpha_h7"] for u in matched]),
        "CROSS": summary([u["alpha_h7"] for u in cross]),
        "SEARCH_FREE": summary([u["alpha_h7"] for u in free]),
        "note": "seven-level fit; NO comparable random population exists; "
                "used in no classification and no decision. See PRE_EXECUTION_ADDENDUM.md.",
    }

    sanity["crop_shapes"] = sorted(str(x) for x in sanity["crop_shapes"])
    sanity["volume_shapes"] = sorted(str(x) for x in sanity["volume_shapes"])
    sanity["right_origins"] = sorted(str(x) for x in sanity["right_origins"])
    sanity["left_crop_distinct_per_scene"] = {k: len(v) for k, v in
                                              sanity.pop("left_crop_hashes_per_scene").items()}
    sanity["cost_volumes_built"] = n_vol
    sanity["readout_passes"] = n_fwd
    sanity["cost_volumes_specified"] = spec["compute"]["cost_volumes"]
    sanity["readout_passes_specified"] = (spec["compute"]["trained_readout_passes"]
                                          + spec["compute"]["search_free_readout_passes"])
    sanity["training_occurred"] = False
    sanity["weights_modified"] = False
    sanity["random_reference_regenerated"] = False

    results = {
        "experiment": "EXP-CORRESPONDENCE-TR-001",
        "title": "Trained-vs-Random Geometric Separation",
        "kind": "inference-only descriptive comparison against a hash-verified "
                "pre-existing random population",
        "spec_sha256": frozen["spec.json"],
        "random_reference_sha256": frozen["random_reference.json"],
        "prereg_sha256": frozen["PREREGISTRATION.md"],
        "addendum_sha256": frozen["PRE_EXECUTION_ADDENDUM.md"],
        "random_reference_source_sha256": ref["source_results_sha256"],
        "checkpoint_sha256": ckpt_hashes,
        "trained_aggregation_sha256": sanity["trained_agg_hashes"],
        "statistic": {
            "primary": "alpha_h = FREE-INTERCEPT OLS slope of m_h(t) on t/16 over "
                       "T_FIT = {16,32,48,64,80,96}; IDENTICAL to the 384 frozen values",
            "reported_only": "alpha_h7 = same fit over all seven levels; compared to nothing",
            "governing_document": "PRE_EXECUTION_ADDENDUM.md",
        },
        "t_levels_px": T_LEVELS, "t_fit_levels_px": T_FIT, "axes": AXES,
        "scenes": SCENES, "scene_names": SNAMES,
        "classification_order": ["NONFINITE"] + CLASSES,
        "thresholds": {"RAIL_LOW": RAIL_LOW, "RAIL_HIGH": RAIL_HIGH,
                       "FLAT_RANGE": FLAT_RANGE,
                       "provenance": "candidate grid spacing; inherited unchanged from "
                                     "ARCH-RATE-001; NOT from any trained observation"},
        "wiring_checks": {
            "W1_left_byte_identical": w1,
            "W2_right_origins_asserted": sanity["right_origins"],
            "W2_sign_verified_from_arrays": sanity["sign_checks"],
            "W3_k_true_recorded": "per unit; the model output is NOT required to equal it",
            "W4_bit_identical_after_restart": True,
            "W4_unit": {"extractor": tgt["extractor"], "aggregation": tgt["aggregation"],
                        "scene": tgt["scene"]},
            "no_numeric_t0_expectation": True,
        },
        "units": units,
        "analysis": analysis,
        "sanity": sanity,
        "determinism": {"cudnn.deterministic": True, "cudnn.benchmark": False,
                        "use_deterministic_algorithms": True,
                        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                        "no_grad": True, "eval": True, "dtype": "fp32"},
        "wall_clock_s": time.time() - t0,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---- results.csv: one row per unit -----------------------------------
    cols = ["arm", "matched", "extractor", "aggregation", "scene", "scene_index",
            "classification", "alpha_h", "beta_h", "alpha_v", "beta_v", "alpha_h7",
            "min_h", "max_h", "range_h", "n_pos_diffs", "n_neg_diffs",
            "min_diff", "max_diff", "Q1", "Q2", "Q3"]
    rows = [",".join(cols + ["alpha_h_pct_rank", "alpha_h_outside_random_range"]
                     + ["m_h_%d" % t for t in T_LEVELS] + ["m_v_%d" % t for t in T_LEVELS])]
    for u in units:
        rows.append(",".join(
            [str(u[c]) for c in cols]
            + ["%.10f" % u["vs_random"]["alpha_h"]["percentile_rank"],
               str(u["vs_random"]["alpha_h"]["outside_observed_random_range"])]
            + ["%.10f" % u["m_h"][str(t)] for t in T_LEVELS]
            + ["%.10f" % u["m_v"][str(t)] for t in T_LEVELS]))
    (OUT / "results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="")

    m = analysis["PRIMARY_MATCHED"]
    say("volumes=%d (spec %d)  readout passes=%d (spec %d)"
        % (n_vol, sanity["cost_volumes_specified"], n_fwd, sanity["readout_passes_specified"]))
    say("MATCHED tally: " + json.dumps(m["tally"]))
    say("MATCHED alpha_h: " + json.dumps(m["alpha_h"]["trained"]))
    say("MATCHED outside observed random alpha_h range: %d/%d = %.4f"
        % (m["alpha_h"]["n_outside_observed_random_range"], m["n"],
           m["alpha_h"]["outside_observed_random_range_rate"]))
    say("CROSS   outside rate: %.4f" % analysis["SECONDARY_CROSS"]["alpha_h"]
        ["outside_observed_random_range_rate"])
    say("SEARCH-FREE tally: " + json.dumps(analysis["SEARCH_FREE_CONTROL"]["tally"]))
    say("wall_clock_s=%.2f" % results["wall_clock_s"])
    (OUT / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8", newline="")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--w4":
        print(json.dumps(w4_unit(sys.argv[2], sys.argv[3], int(sys.argv[4]))))
        raise SystemExit(0)
    raise SystemExit(main())
