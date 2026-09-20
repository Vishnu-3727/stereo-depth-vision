"""Figures for the ARM-V high-disparity diagnostic. Read-only, no training."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
REF = json.loads((HERE / "apparatus_and_reference.json").read_text())
ST = json.loads((HERE / "scale_transfer.json").read_text())

centers, ref_epe, ref_tr = [], [], []
for b in REF["bins"]:
    centers.append((b["lo"] + b["hi"]) / 2)
    ref_epe.append(b["ref_epe"])
    ref_tr.append(b["train_frac"])
centers = np.array(centers, dtype=float)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

# (a) per-bin EPE
for s, c in zip((0, 1, 2), ("C0", "C1", "C2")):
    bins = R["seeds"][str(s)]["hailo_val"]["bins"]
    y = [b.get("epe") for b in bins]
    ax[0].plot(centers, y, "o-", color=c, label="ARM-V seed %d" % s)
sc = ST["seeds"]["0"]["per_scale"]["0.5"]["bins"]
ax[0].plot(centers, [b["epe"] for b in sc], "s--", color="C4",
           label="ARM-V seed 0, input scale 0.5")
ax[0].plot(centers, ref_epe, "k^-", label="frozen reference ONNX")
ax[0].set_yscale("log"); ax[0].set_xlabel("ground-truth disparity (px)")
ax[0].set_ylabel("EPE (px, log)"); ax[0].set_title("(a) error by GT disparity")
ax[0].axvline(64, color="grey", ls=":"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

# (b) mean prediction vs mean GT
ax[1].plot([0, 160], [0, 160], "k:", lw=1, label="ideal")
for s, c in zip((0, 1, 2), ("C0", "C1", "C2")):
    bins = R["seeds"][str(s)]["hailo_val"]["bins"]
    x = [b.get("mean_gt") for b in bins]
    y = [b.get("mean_pred") for b in bins]
    ax[1].plot(x, y, "o-", color=c, label="ARM-V seed %d" % s)
ax[1].set_xlabel("mean GT in bin (px)"); ax[1].set_ylabel("mean prediction (px)")
ax[1].set_title("(b) prediction saturates above ~64 px")
ax[1].axvline(64, color="grey", ls=":"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)

# (c) training pixel density vs ARM-V error
ax2 = ax[2]
ax2.plot(centers, ref_tr, "ks-", label="training pixel fraction")
ax2.set_yscale("log"); ax2.set_xlabel("ground-truth disparity (px)")
ax2.set_ylabel("fraction of training pixels (log)")
ax3 = ax2.twinx()
bins = R["seeds"]["0"]["hailo_val"]["bins"]
ax3.plot(centers, [b.get("epe") for b in bins], "C0o-", label="ARM-V seed 0 EPE")
ax3.set_yscale("log"); ax3.set_ylabel("EPE (px, log)")
ax2.set_title("(c) training coverage vs error")
ax2.axvline(64, color="grey", ls=":"); ax2.grid(alpha=.3)

fig.tight_layout()
fig.savefig(HERE / "high_disparity_diagnostic.png", dpi=140)
print("wrote", HERE / "high_disparity_diagnostic.png")
