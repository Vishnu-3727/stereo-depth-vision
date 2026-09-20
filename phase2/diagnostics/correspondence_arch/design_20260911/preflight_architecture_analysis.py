"""EXP-CORRESPONDENCE-ARCH-001 -- Section H / G pre-flight analysis.

SYNTHETIC ONLY. This script loads NO checkpoint, NO scene and NO repository
model class. It instantiates bare `torch.nn.Conv3d` primitives on synthetic
tensors in order to establish, BEFORE any real output exists, two things the
protocol requires to be settled in advance:

  H. whether Agg(shift_m(V)) induces an approximately shifted response for
     ARBITRARY (untrained) weights;
  G. whether circular rank total variation is an appropriate primary statistic
     for the trained-vs-random comparison.

It produces no experimental output and no `disparity_initial`. Nothing here may
be cited as evidence about the trained model.

Re-run with `python preflight_architecture_analysis.py` to reproduce the numbers
quoted in ARCHITECTURE_CONTROL_DESIGN.md sections 5 and 6.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

D = 12
EPS = 1e-5          # matches phase2/models/scaled_regression.py


# ----------------------------------------------------------------- H: the operator
def make_stack(seed: int, ch: int = 4) -> nn.Sequential:
    """Mirrors the repo aggregation's LAYER SPEC (4 x [Conv3d 3x3x3 pad 1 +
    LeakyReLU 0.01] then Conv3d(->1, 3x3x3, pad 1)) with arbitrary weights.
    Deliberately NOT the repo class: this is an operator property check, not a
    model instantiation."""
    torch.manual_seed(seed)
    layers: list[nn.Module] = []
    for _ in range(4):
        layers += [nn.Conv3d(ch, ch, 3, stride=1, padding=1), nn.LeakyReLU(0.01)]
    layers += [nn.Conv3d(ch, 1, 3, stride=1, padding=1)]
    return nn.Sequential(*layers).eval()


def circshift(V: torch.Tensor, m: int) -> torch.Tensor:
    """The actual intervention: index_select with pi_m(k) = (k - m) mod 12."""
    idx = torch.tensor([(k - m) % D for k in range(D)])
    return V.index_select(2, idx)


def zerofill_shift(V: torch.Tensor, m: int) -> torch.Tensor:
    """Translation with zero fill, for contrast with the circular intervention."""
    out = torch.zeros_like(V)
    for k in range(D):
        src = k - m
        if 0 <= src < D:
            out[:, :, k] = V[:, :, src]
    return out


def h_equivariance() -> None:
    torch.manual_seed(123)
    V = torch.randn(1, 4, D, 3, 3)
    A = make_stack(7)                      # arbitrary weights, never trained
    with torch.no_grad():
        base = A(V)[0, 0]
        print("H-1  exact positions where Agg(shift_m V)(k) == Agg(V)(k-m)")
        for m in (1, 2, 3):
            for name, fn in (("circular", circshift), ("zero-fill", zerofill_shift)):
                sh = A(fn(V, m))[0, 0]
                exact = [k for k in range(D)
                         if 0 <= k - m < D and torch.allclose(sh[k], base[k - m], atol=1e-6)]
                print("   m=%d %-9s -> %s" % (m, name, exact if exact else "none"))
        print()
        print("H-2  per-position relative error of the circular-shift approximation")
        scale = float(base.abs().max())
        for m in (1, 2):
            errs = [float((A(circshift(V, m))[0, 0][k] - base[(k - m) % D]).abs().max()) / scale
                    for k in range(D)]
            print("   m=%d  " % m + "  ".join("k%02d:%.3f" % (k, e) for k, e in enumerate(errs)))


# ------------------------------------------------- G: is circular rank TV appropriate?
def standardise(c: np.ndarray) -> np.ndarray:
    return (c - c.mean()) / (c.std() + EPS)


def soft_argmin(c: np.ndarray) -> float:
    z = standardise(c)
    w = np.exp(-z - (-z).max())
    w /= w.sum()
    return float((w * np.arange(D)).sum())


def orbit(c: np.ndarray) -> np.ndarray:
    return np.array([soft_argmin(np.array([c[(k - m) % D] for k in range(D)]))
                     for m in range(D)])


def midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    r = np.empty(D)
    r[order] = np.arange(1, D + 1)
    for v in np.unique(x):
        idx = np.where(x == v)[0]
        if idx.size > 1:
            r[idx] = r[idx].mean()
    return r


def circ_tv(s: np.ndarray) -> float:
    r = midrank(s)
    return float(sum(abs(r[(m + 1) % D] - r[m]) for m in range(D)))


def zscore(s: np.ndarray) -> np.ndarray:
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else np.zeros_like(s)


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Primary statistic candidate: Pearson r of the z-scored 12-position orbits."""
    return float(np.mean(zscore(a) * zscore(b)))


def g_audit() -> None:
    rng = np.random.default_rng(0)
    profiles = {
        "sharp peak @3": np.array([5, 4, 2, 0, 2, 4, 5, 6, 6, 6, 6, 6], float),
        "broad peak @3": np.array([4, 3, 2.4, 2, 2.4, 3, 3.6, 4, 4.4, 4.6, 4.8, 5], float),
        "sharp peak @8": np.array([6, 6, 6, 6, 5, 4, 2, 0.5, 0, 1, 3, 5], float),
        "random-ish A": rng.normal(size=D),
        "random-ish B": rng.normal(size=D),
    }
    print("G-1  orbit shape and circular rank TV under ideal circular equivariance")
    for name, c in profiles.items():
        o = orbit(c)
        print("   %-15s TV=%4.1f  range=%.2f  orbit=%s"
              % (name, circ_tv(o), o.max() - o.min(), " ".join("%5.2f" % v for v in o)))
    print("   TV floor for a monotone-around-the-circle rank sequence = 22")
    print()
    print("G-2  headroom of the replacement statistic (orbit-shape agreement)")
    peaked = []
    for pk in (2, 3, 4):
        c = np.full(D, 6.0)
        c[pk] = 0.0
        c[max(0, pk - 1)] = 2.0
        c[min(D - 1, pk + 1)] = 2.0
        peaked.append(orbit(c))
    randoms = [orbit(rng.normal(size=D)) for _ in range(6)]

    def within(X):
        return np.array([agreement(X[i], X[j])
                         for i in range(len(X)) for j in range(i + 1, len(X))])

    def across(X, Y):
        return np.array([agreement(a, b) for a in X for b in Y])

    for nm, v in (("peaked-peaked", within(peaked)),
                  ("random-random", within(randoms)),
                  ("peaked-random", across(peaked, randoms))):
        print("   %-15s n=%3d mean=%+.3f min=%+.3f max=%+.3f"
              % (nm, len(v), v.mean(), v.min(), v.max()))
    print()
    print("G-3  phase sensitivity (same mechanism, different peak position)")
    for i in range(len(peaked)):
        for j in range(i + 1, len(peaked)):
            print("   peak%d vs peak%d : r=%+.3f" % (i + 2, j + 2, agreement(peaked[i], peaked[j])))


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    h_equivariance()
    print()
    g_audit()
