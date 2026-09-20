import json
from pathlib import Path
p = Path("phase2/diagnostics/correspondence_index/20260910T164346Z/results.json")
d = json.loads(p.read_text())
print("rows:", len(d["rows"]))
vd = d.get("volume_diffs", [])
neg = [x for x in vd if x["checkpoint"].startswith("NEG")]
pos = [x for x in vd if x["checkpoint"].startswith("POS")]
print("neg vol diffs, n=", len(neg), "max=", max(x["max_abs_diff_vs_original"] for x in neg))
print("pos vol diffs sample:")
for x in pos[:4]:
    print(" ", x)
# checkpoint-level table: mean of per-scene medians + deltas
import collections
for ckpt in ["NEG_shift_none", "POS_6b_seed0", "POS_6b_seed1", "POS_6b_seed2"]:
    print("==", ckpt)
    for perm in ["minus1", "identity", "plus1", "plus2", "random"]:
        meds = [r["median"] for r in d["rows"] if r["checkpoint"] == ckpt and r["permutation"] == perm]
        dm = [r["delta_median_vs_identity"] for r in d["rows"] if r["checkpoint"] == ckpt and r["permutation"] == perm]
        print(f"  {perm:8s} medians={['%.3f' % v for v in meds]} mean_of_med=%.3f mean_dmed=%+.3f" % (sum(meds)/len(meds), sum(dm)/len(dm)))
