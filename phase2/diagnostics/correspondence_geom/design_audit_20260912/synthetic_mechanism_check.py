"""Synthetic mechanism check for the EXP-CORRESPONDENCE-GEOM-001 design audit.

ANALYTICAL / SYNTHETIC ONLY.

  * No trained checkpoint is loaded. No trained model is instantiated.
  * No real image and no dataset is read.
  * No training, no optimizer, no .backward(), no .step().
  * The only learned-architecture module used is `Aggregation`, constructed with
    FRAMEWORK-DEFAULT RANDOM WEIGHTS, exactly as
    `correspondence_arch_rate/20260911T130507Z/randomisation_spec.json` does.
  * Nothing here is an experiment result. It is a mechanism demonstration whose
    only job is to decide whether a proposed intervention is discriminative.

THE PROPOSED INTERVENTION
-------------------------
A 2x2 pairing design at each candidate offset tau.  A and B are two sources.

    cell AA : left = A,  right = A shifted by tau   -> corresponds, d = tau
    cell BB : left = B,  right = B shifted by tau   -> corresponds, d = tau
    cell AB : left = A,  right = B shifted by tau   -> does NOT correspond
    cell BA : left = B,  right = A shifted by tau   -> does NOT correspond

    INTERACTION   I_tau(k) = 1/2[z_AA + z_BB](k) - 1/2[z_AB + z_BA](k)

Every main effect (left identity, right identity, tau, architecture, padding)
enters all four cells additively and cancels.  What survives is the LEFT x RIGHT
interaction -- whether the two images actually correspond.

WHAT IS TESTED HERE
-------------------
S0  STRUCTURAL. V_AA + V_BB - V_AB - V_BA == 0 for the COST VOLUME, because the
    cost volume is linear in both feature maps.  Consequence: the interaction is
    generated ENTIRELY by the aggregation's nonlinearity.  A linear aggregation
    must return an identically-zero cost interaction for ANY weights.

S1  CLAIM: the translation-slope class is dead.  Under random weights, compare
    the free-intercept slope alpha of the soft-argmin response for MATCHED and
    CROSSED pairs.  Both cost volumes are joint (k, w) translates of a
    tau-independent object, so both translate; the slope therefore cannot carry
    the correspondence distinction on its own.

S2  CLAIM: the matched arm ALONE is not enough.  How often does
    argmin_k z_AA(k) == tau under random weights?  If it is already reliable,
    the interaction is unnecessary; if it is not, the interaction is required.

S3  CLAIM: the interaction ANCHOR is NOT architecturally forced.  How often does
    argmin_k I_tau(k) == tau under random NONLINEAR weights, and under a LINEAR
    aggregation (LeakyReLU slope set to 1.0, which makes the whole module
    affine)?  The linear arm measures the artifact channel with the nonlinearity
    switched off; the nonlinear arm measures it switched on.

SYNTHETIC STIMULUS
------------------
Feature maps are simulated directly, one level below the pixel stimulus, which
is exact for this construction: the extractor is fully convolutional and every
shift is a whole multiple of the stride 16, so a t-pixel crop shift is exactly a
tau = t/16 cell shift of the feature map.  Two independent smoothed Gaussian
fields stand in for the features of two different scenes.  Smoothing is used
because real feature maps are spatially correlated.  The check is about
MECHANISM (can the signature arise at all), never about magnitudes, and no
number produced here is used as a threshold anywhere.

Run:  python synthetic_mechanism_check.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.stereonet.aggregation import Aggregation          # noqa: E402
from src.models.stereonet.cost_volume import build_cost_volume    # noqa: E402
from src.models.stereonet.regression import soft_argmin           # noqa: E402
from phase2.models.scaled_regression import (                     # noqa: E402
    standardise_across_disparity,
)

OUT = Path(__file__).resolve().parent

# Geometry: identical in shape to the frozen construction (71 x 17 feature
# cells = a 1136 x 272 crop at stride 16), so the D-axis padding boundary and
# the right-edge zero fill of shift_left occupy the same fraction as in a real
# run.
FH, FW = 17, 71
STRIDE = 16
D = 12
CHANNELS = 32
TAUS = [0, 1, 2, 3, 4, 5, 6]
TAU_TEST = [1, 2, 3, 4, 5, 6]         # tau = 0 excluded: matched cells are degenerate
SEEDS = list(range(8))                # random aggregation seeds
FIELD_SEEDS = [1000, 2000]            # the two synthetic "scenes"
SMOOTH = 1.5                          # gaussian sigma in feature cells

# Mask, DERIVED (not chosen):
#   * shift_left(x, k) zero-fills feature columns [FW-k, FW); max k = D-1 = 11
#     -> columns [60, 71) can be fill.
#   * the aggregation is 5 x Conv3d(3x3x3) -> spatial radius 5 cells.
#   * bilinear upsample mixes one further neighbour.
#   -> a feature column is safe iff w + 5 + 1 < 60, i.e. w < 54  ->  x < 864 px.
#   * top / bottom / left border 64 px, inherited unchanged from the frozen mask.
BORDER = 64
SAFE_X_PX = 54 * STRIDE               # 864


def smoothed_field(seed: int, h: int, w: int, channels: int, sigma: float) -> torch.Tensor:
    """A spatially correlated random stand-in for one scene's feature map."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, channels, h, w, generator=g)
    radius = max(1, int(3 * sigma))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum()
    x = Fn.conv2d(x, kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1),
                  padding=(0, radius), groups=channels)
    x = Fn.conv2d(x, kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1),
                  padding=(radius, 0), groups=channels)
    return x / x.std()


def make_mask(h_px: int, w_px: int) -> torch.Tensor:
    m = torch.zeros(h_px, w_px, dtype=torch.bool)
    m[BORDER:h_px - BORDER, BORDER:SAFE_X_PX] = True
    return m


def free_intercept_slope(taus: list[int], values: list[float]) -> float:
    x = np.asarray(taus, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    return float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))


def linearised(agg: Aggregation) -> Aggregation:
    """Same weights, LeakyReLU slope 1.0 -> the module becomes affine.

    Switches the artifact channel off: with an affine aggregation the cost
    interaction is identically zero for any weights (S0), so whatever hit rate
    this arm shows is the floor produced by the readout alone.
    """
    for m in agg.modules():
        if isinstance(m, nn.LeakyReLU):
            m.negative_slope = 1.0
    return agg


def zprofile(agg, volume: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Deployed readout up to the softmax input: aggregate -> upsample ->
    standardise across the candidate axis.  Returns (D, H, W)."""
    cost = agg(volume)
    up = Fn.interpolate(cost, size=size, mode="bilinear", align_corners=True)
    return standardise_across_disparity(up, dim=1)[0]


def main() -> int:
    t0 = time.time()
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)

    h_px, w_px = FH * STRIDE, FW * STRIDE
    mask = make_mask(h_px, w_px)
    print("feature grid %dx%d -> %dx%d px   mask %d px   taus %s"
          % (FH, FW, h_px, w_px, int(mask.sum()), TAU_TEST), flush=True)

    base = [smoothed_field(s, FH, FW + max(TAUS), CHANNELS, SMOOTH) for s in FIELD_SEEDS]
    CELLS = [("AA", 0, 0), ("BB", 1, 1), ("AB", 0, 1), ("BA", 1, 0)]

    # ---- build the 4 x 7 cost volumes once (aggregation-independent) --------
    volumes: dict[tuple[str, int], torch.Tensor] = {}
    zero_at_tau = []
    for name, li, rj in CELLS:
        left = base[li][..., 0:FW]
        for tau in TAUS:
            right = base[rj][..., tau:tau + FW]
            v = build_cost_volume(left, right, D, method="subtract", shift="left")
            assert tuple(v.shape) == (1, CHANNELS, D, FH, FW), v.shape
            volumes[(name, tau)] = v
            if li == rj:
                zero_at_tau.append(float(v[0, :, tau, :, :54].abs().max()))

    # ---- S0 : the cost-volume 2x2 interaction is zero ----------------------
    s0 = []
    for tau in TAUS:
        resid = (volumes[("AA", tau)] + volumes[("BB", tau)]
                 - volumes[("AB", tau)] - volumes[("BA", tau)])
        scale = max(float(volumes[("AB", tau)].abs().max()), 1e-12)
        s0.append({"tau": tau, "max_abs_residual": float(resid.abs().max()),
                   "volume_scale": scale,
                   "relative": float(resid.abs().max()) / scale})
    print("S0  matched volume exactly 0 at k=tau : max|V| = %.3e" % max(zero_at_tau))
    print("S0  cost-volume 2x2 interaction       : max relative residual = %.3e"
          % max(r["relative"] for r in s0), flush=True)

    # ---- run both aggregation arms ----------------------------------------
    slopes, matched_alone, interaction = [], [], []
    for arm in ("nonlinear", "linear"):
        for seed in SEEDS:
            torch.manual_seed(seed)
            agg = Aggregation(in_channels=CHANNELS, channels=CHANNELS, num_layers=4).eval()
            if arm == "linear":
                agg = linearised(agg)

            z = {}
            for name, _li, _rj in CELLS:
                for tau in TAUS:
                    z[(name, tau)] = zprofile(agg, volumes[(name, tau)], (h_px, w_px))

            # S1 -- soft-argmin response slope, matched vs crossed
            for name, li, rj in CELLS:
                m = [float(soft_argmin(z[(name, tau)][None], dim=1)[0, 0][mask].median())
                     for tau in TAUS]
                slopes.append({"arm": arm, "seed": seed, "cell": name,
                               "matched": li == rj,
                               "alpha": free_intercept_slope(
                                   TAU_TEST, [m[TAUS.index(t)] for t in TAU_TEST]),
                               "m": m})

            # S2 -- matched arm alone: argmin_k median z_AA(k)
            for name in ("AA", "BB"):
                for tau in TAU_TEST:
                    prof = np.array([float(z[(name, tau)][k][mask].median())
                                     for k in range(D)])
                    matched_alone.append({"arm": arm, "seed": seed, "cell": name,
                                          "tau": tau, "argmin": int(prof.argmin()),
                                          "hit": bool(int(prof.argmin()) == tau)})

            # S3 -- the interaction, per pixel then median over the mask
            for tau in TAU_TEST:
                inter = 0.5 * (z[("AA", tau)] + z[("BB", tau)]) \
                      - 0.5 * (z[("AB", tau)] + z[("BA", tau)])
                prof = np.array([float(inter[k][mask].median()) for k in range(D)])
                interaction.append({
                    "arm": arm, "seed": seed, "tau": tau,
                    "I": [float(x) for x in prof],
                    "argmin": int(prof.argmin()), "argmax": int(prof.argmax()),
                    "hit": bool(int(prof.argmin()) == tau),
                    "anchor": int(prof.argmin()) - tau,
                    "amplitude": float(prof.max() - prof.min()),
                })
            print("  %-9s seed %2d done  (%.1fs)" % (arm, seed, time.time() - t0),
                  flush=True)

    # ---- summarise ---------------------------------------------------------
    def rate(rows, pred=lambda r: r["hit"]):
        return (sum(1 for r in rows if pred(r)), len(rows))

    def per_unit_h(rows):
        """h = number of the six tau levels hit, per (arm, seed) unit."""
        out = {}
        for r in rows:
            out.setdefault((r["arm"], r["seed"]), 0)
            out[(r["arm"], r["seed"])] += int(r["hit"])
        return out

    nl = [r for r in interaction if r["arm"] == "nonlinear"]
    ln = [r for r in interaction if r["arm"] == "linear"]
    ma_nl = [r for r in matched_alone if r["arm"] == "nonlinear"]
    ma_ln = [r for r in matched_alone if r["arm"] == "linear"]
    a_m = [s["alpha"] for s in slopes if s["arm"] == "nonlinear" and s["matched"]]
    a_c = [s["alpha"] for s in slopes if s["arm"] == "nonlinear" and not s["matched"]]
    h_nl = per_unit_h(nl)
    h_ln = per_unit_h(ln)

    summary = {
        "kind": "synthetic mechanism check -- NOT an experiment result",
        "trained_model_used": False, "real_data_used": False,
        "training_performed": False,
        "geometry": {"feature_grid": [FH, FW], "pixels": [h_px, w_px], "D": D,
                     "channels": CHANNELS, "taus": TAUS, "tau_test": TAU_TEST,
                     "border_px": BORDER, "safe_x_px": SAFE_X_PX,
                     "mask_px": int(mask.sum()), "smooth_sigma": SMOOTH},
        "n_random_seeds": len(SEEDS),
        "S0_cost_volume_interaction": {
            "matched_volume_max_abs_at_k_equals_tau": max(zero_at_tau),
            "per_tau": s0,
            "max_relative_residual": max(r["relative"] for r in s0),
        },
        "S1_slope_matched_vs_crossed_random_nonlinear": {
            "matched": {"n": len(a_m), "min": min(a_m),
                        "median": float(np.median(a_m)), "max": max(a_m)},
            "crossed": {"n": len(a_c), "min": min(a_c),
                        "median": float(np.median(a_c)), "max": max(a_c)},
        },
        "S2_matched_arm_alone_argmin_hits": {
            "nonlinear": {"hits": rate(ma_nl)[0], "n": rate(ma_nl)[1]},
            "linear": {"hits": rate(ma_ln)[0], "n": rate(ma_ln)[1]},
        },
        "S3_interaction_argmin_hits": {
            "nonlinear": {"hits": rate(nl)[0], "n": rate(nl)[1],
                          "per_unit_h": sorted(h_nl.values()),
                          "max_h": max(h_nl.values()),
                          "anchors": sorted({r["anchor"] for r in nl}),
                          "median_amplitude": float(np.median(
                              [r["amplitude"] for r in nl]))},
            "linear": {"hits": rate(ln)[0], "n": rate(ln)[1],
                       "per_unit_h": sorted(h_ln.values()),
                       "max_h": max(h_ln.values()),
                       "anchors": sorted({r["anchor"] for r in ln}),
                       "median_amplitude": float(np.median(
                           [r["amplitude"] for r in ln]))},
        },
        "records": {"slopes": slopes, "matched_alone": matched_alone,
                    "interaction": interaction},
        "wall_clock_s": time.time() - t0,
    }
    (OUT / "synthetic_mechanism_check.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("S1  soft-argmin slope alpha, RANDOM NONLINEAR weights")
    print("    matched n=%2d  min %+.4f  median %+.4f  max %+.4f"
          % (len(a_m), min(a_m), float(np.median(a_m)), max(a_m)))
    print("    crossed n=%2d  min %+.4f  median %+.4f  max %+.4f"
          % (len(a_c), min(a_c), float(np.median(a_c)), max(a_c)))
    print()
    print("S2  matched arm ALONE:  argmin_k z(k) == tau")
    print("    nonlinear  %d / %d        linear  %d / %d"
          % (rate(ma_nl) + rate(ma_ln)))
    print()
    print("S3  INTERACTION:  argmin_k I_tau(k) == tau")
    print("    nonlinear  %d / %d   per-unit h %s   anchors %s"
          % (rate(nl) + (sorted(h_nl.values()), sorted({r['anchor'] for r in nl}))))
    print("    linear     %d / %d   per-unit h %s   anchors %s"
          % (rate(ln) + (sorted(h_ln.values()), sorted({r['anchor'] for r in ln}))))
    print()
    print("wrote synthetic_mechanism_check.json   (%.1fs)" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
