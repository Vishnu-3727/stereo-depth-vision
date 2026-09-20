"""M2 offset provenance, M3 mean aligned profile, M4 gate-rule arithmetic."""
import json, sys, collections, math
import numpy as np
from pathlib import Path

D = Path(sys.argv[1])
g = json.loads((D / "gate.json").read_text(encoding="utf-8"))
raw = json.loads((D / "gate_raw.json").read_text(encoding="utf-8"))
U = g["units"]
TAUS = [1, 2, 3, 4, 5, 6]; LAGS = list(range(-1, 6))

print("=== M2a: is the constant offset c a property of the WEIGHTS or the CONTENT? ===")
byseed = collections.defaultdict(list)
for u in U:
    if u["offset_constant"]:
        byseed[u["seed"]].append(u["offsets"][0])
n_const_seeds = 0; same = 0
for s in sorted(byseed):
    vals = byseed[s]
    n_const_seeds += 1
    if len(set(vals)) == 1:
        same += 1
print("seeds with >=1 offset-constant unit: %d" % n_const_seeds)
print("  of those, seeds whose c is identical across ALL their constant units: %d" % same)
print("  seed : n_constant_of_18 : c values")
for s in sorted(byseed, key=lambda s: -len(byseed[s]))[:12]:
    print("   %2d  :  %2d  : %s" % (s, len(byseed[s]), sorted(collections.Counter(byseed[s]).items())))

print()
print("=== M2b: within a seed, does c change with extractor / scene pair? ===")
cell = {(u["seed"], u["extractor"], u["pair"]):
        (u["offsets"][0] if u["offset_constant"] else None) for u in U}
flip = 0; tot = 0
for s in sorted(byseed):
    vals = [cell[(s, e, p)] for e in ("H2_seed0", "H2_seed1", "H2_seed2")
            for p in range(6) if cell[(s, e, p)] is not None]
    tot += 1
    if len(set(vals)) > 1:
        flip += 1
print("seeds whose c varies across (extractor, pair): %d of %d" % (flip, tot))

print()
print("=== M3: mean aligned profile Phi(d), per-unit RMS-normalised ===")
acc = []
for r in raw:
    P = np.array([[r["tau"][str(t)]["Ibar"][t + d] for d in LAGS] for t in TAUS])
    rms = np.sqrt((P ** 2).mean())
    if rms:
        acc.append(P.mean(axis=0) / rms)
A = np.array(acc)
print("  d    : " + "  ".join("%6d" % d for d in LAGS))
print("  mean : " + "  ".join("%6.3f" % v for v in A.mean(axis=0)))
print("  sd   : " + "  ".join("%6.3f" % v for v in A.std(axis=0)))
print("  argmin of the mean aligned profile: d = %d" % LAGS[int(A.mean(axis=0).argmin())])
am = np.array([LAGS[int(row.argmin())] for row in A])
print("  per-unit argmin of aligned profile:", sorted(collections.Counter(am.tolist()).items()))

print()
print("=== M4: the gate rule's own arithmetic (no data needed for the derivation) ===")
p6 = 31 / 576
print("observed P(unit h=6 | random) = 31/576 = %.4f" % p6)
lo = p6 - 1.96 * math.sqrt(p6 * (1 - p6) / 576); hi = p6 + 1.96 * math.sqrt(p6 * (1 - p6) / 576)
print("  95%% CI  [%.4f, %.4f]" % (lo, hi))
print("P(max over 576 random units < 6) = (1-p)^576 = %.3e" % ((1 - p6) ** 576))
print("  -> G-STOP was certain to within %.0e" % ((1 - p6) ** 576))
print("smallest p for which a 576-unit max-gate passes with prob >= 0.5: p <= %.5f"
      % (1 - 0.5 ** (1 / 576)))
print("  i.e. the null must produce a perfect unit less than once in %d draws"
      % round(1 / (1 - 0.5 ** (1 / 576))))
print("and even on G-PASS the brief requires min(h_trained) > H*, so with H*=5")
print("  ALL 18 trained units must score 6; with H*=6 the design is unfalsifiable-positive.")

print()
print("=== M4b: a RATE comparison on the same data, for contrast ===")
print("null rate of c==0 per unit            : 31/576 = %.4f" % p6)
print("null rate of offset-constant per unit : 153/576 = %.4f" % (153 / 576))
print("null rate of c==0 GIVEN constant      : 31/153 = %.4f" % (31 / 153))
for k in (18, 15, 12):
    print("  P(>= %2d of 18 trained units at c==0 | null rate %.4f) = %.3e"
          % (k, p6, sum(math.comb(18, j) * p6 ** j * (1 - p6) ** (18 - j)
                        for j in range(k, 19))))
