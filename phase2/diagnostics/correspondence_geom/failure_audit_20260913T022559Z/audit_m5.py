"""M5: is the DISPARITY OUTPUT itself architecturally forced to track tau?

m_median[cell][tau] is the median soft-argmin output (candidate units) over the
mask, already recorded for every random-weight unit.  If random weights already
produce slope ~ 1 of m_AA against tau, then the disparity output is as
architecturally available as the argmin was.
Random weights only.
"""
import json, sys, collections
import numpy as np
from pathlib import Path

D = Path(sys.argv[1])
raw = json.loads((D / "gate_raw.json").read_text(encoding="utf-8"))
TAUS = np.array([1., 2., 3., 4., 5., 6.])
out = collections.defaultdict(list)
for r in raw:
    for cell in ("AA", "BB", "AB", "BA"):
        m = np.array([r["tau"][str(int(t))]["m_median"][cell] for t in TAUS])
        A = np.vstack([TAUS, np.ones_like(TAUS)]).T
        slope, inter = np.linalg.lstsq(A, m, rcond=None)[0]
        ss = ((m - m.mean()) ** 2).sum()
        r2 = 1 - ((m - (slope * TAUS + inter)) ** 2).sum() / ss if ss > 0 else np.nan
        out[cell].append((slope, inter, r2, float(np.abs(m - TAUS).mean())))
for cell in ("AA", "BB", "AB", "BA"):
    a = np.array(out[cell])
    print("%s  slope: mean %+.3f  median %+.3f  [p05 %+.3f, p95 %+.3f] | "
          "R2 median %.3f | mean|m-tau| median %.3f"
          % (cell, a[:, 0].mean(), np.median(a[:, 0]),
             np.percentile(a[:, 0], 5), np.percentile(a[:, 0], 95),
             np.nanmedian(a[:, 2]), np.median(a[:, 3])))
a = np.array(out["AA"])
print()
print("AA slope within +-0.2 of 1.0 : %d / %d units"
      % (int((np.abs(a[:, 0] - 1) <= 0.2).sum()), len(a)))
print("AA slope within +-0.5 of 1.0 : %d / %d units"
      % (int((np.abs(a[:, 0] - 1) <= 0.5).sum()), len(a)))
print("AA median |m - tau| <= 0.5   : %d / %d units"
      % (int((a[:, 3] <= 0.5).sum()), len(a)))
print("AA median |m - tau| <= 1.0   : %d / %d units"
      % (int((a[:, 3] <= 1.0).sum()), len(a)))
