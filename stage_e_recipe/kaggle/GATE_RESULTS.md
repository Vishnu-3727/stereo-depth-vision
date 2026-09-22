# Stage E pre-run gates — Kaggle T4, all PASS

Kernel `vishnu3727/stage-e-pre-run-gates`, version 1, wall 42.99 s.
Raw result: `gate_output/stage_e_gates.json`. **No E0 training was started.**

Note on "Successful": that is the *kernel's* exit status. The gate script
catches each gate's exceptions so one failure cannot hide the others, which
means a kernel can exit successfully with gates failing. The verdict below is
`all_pass` from the result file, not the Kaggle badge.

## Environment (G1)

| | |
|---|---|
| GPU | **Tesla T4 x2** (training uses `cuda:0`; no data-parallel) |
| Memory | 14,911.7 MiB |
| torch / CUDA / cuDNN | 2.10.0+cu128 / 12.8 / 91002 |
| Python | 3.12.13 |

## Gate results

| gate | result |
|---|---|
| G1 environment | PASS — CUDA available, T4 |
| G2 source integrity | PASS — **25 files re-hashed on Kaggle, zero mismatches**, 2 declared deltas |
| G3 initialization integrity | PASS — 12/12; 70 keys, 397,954 params, blob keys exactly `['config', 'model']` |
| G4 data / evaluation contract | PASS — calib 160, val 40, overlap 0, img `[368,1232,3]`, disp `[368,1232]`, scale 256.0, **valid px 3,802,797 == expected** |
| G5 one batch | PASS — pred `[1,1,368,1232]` matches disp `[1,1,368,1232]`, loss 5.1339, grads present, strict reload clean |
| G6 batch-8 memory | PASS — see below |
| G7 rate probe | PASS — see below |

G5 confirms Stage B's phase-6 shape bug is fixed: prediction and disparity now
agree at `(1,1,368,1232)`. Stage B's `[None]` gave `(1,368,1232)` and failed.

## G6 — batch 8 on T4: NATIVE_BATCH_8 confirmed

| batch | status | peak allocated | peak reserved |
|---|---|---|---|
| 2 | FIT | 1,186.7 MiB | 1,278.0 MiB |
| 8 | **FIT** | 4,741.7 MiB | 5,766.0 MiB |

Headroom **61.3%** against the 20% threshold fixed before any measurement
(local RTX 4060 gave 41.3%). **E2 runs native batch 8 on T4**; the
pixel-weighted accumulation fallback is not used and its correctness gate stays
off the critical path. This satisfies amendment A2.4 — the decision now rests
on a measurement in the training environment, not on local VRAM.

## G7 — measured rate and the recalculated budget

20 timed optimizer steps per configuration after 2 untimed warmups, plus one
10-scene monitor pass. No epoch completed, no checkpoint saved, no record
written, weights discarded.

| config | batch | s/step | steps/epoch | s/epoch |
|---|---|---|---|---|
| E0 / E1 | 2 | 0.2276 | 80 | 18.21 |
| E2 | 8 | 0.7484 | 20 | 14.97 |

Monitor pass: 1.28 s, fired every 5 epochs.

### Recalculated campaign budget

| experiment | per seed | 3 seeds |
|---|---|---|
| E0 | 1.03 h | 3.08 h |
| E1 | 1.03 h | 3.08 h |
| E2 | 0.85 h | 2.54 h |
| E3 | 2.05 h | 6.16 h |
| **campaign total** | | **14.85 h** |

**This corrects the earlier estimate of 24–39 h, which was wrong.** That figure
assumed the T4 would be meaningfully slower than the local RTX 4060; measured,
they are near-identical for this model (1.03 h against the local 62 min for the
same 200 epochs). The consequences:

- The campaign fits inside a **single ~30 h quota week**, rather than spanning
  several. The "quota-bound, not compute-bound" warning is withdrawn.
- Every run fits inside a 12-hour session with wide margin; E3's 2.05 h/seed is
  the longest single run.
- Batch 8 is **18% faster per epoch** than batch 2 (14.97 s against 18.21 s),
  so E2 is the cheapest candidate rather than the most expensive.

The estimates assume E1's EMA adds a weight copy per optimizer step rather than
another forward/backward, so E1 is costed equal to E0. That is an assumption,
not a measurement; E1's first seed will confirm or correct it.

---

## Addendum — the rate probe underestimated by ~24%

Measured after E0's three seeds ran:

| | predicted by G7 | actual |
|---|---|---|
| per seed | 1.03 h | **1.27–1.33 h** (4,647.3 / 4,556.6 / 4,772.2 s) |
| E0, 3 seeds | 3.08 h | **3.88 h** |
| campaign | 14.85 h | **~18 h** |

G7 timed 20 steady-state optimizer steps and extrapolated. It did not capture
kernel startup and bundle assembly, per-epoch data-loading variance, the 41
monitor passes across 200 epochs, or the final scoring pass over 40 scenes at
full resolution for two checkpoints.

The conclusions the probe was run to reach are unaffected: the campaign still
fits inside a single ~30 h quota week, and every run clears the 12-hour session
cap by a wide margin. But the figure was stated as measured, so the correction
is recorded here rather than quietly replaced. A rate probe of this shape
should be read as a **lower bound** on wall time, not an estimate of it.

## Addendum — E4 rate probe: two attempts, no rate number yet

The E4 rate probe runs under a different Kaggle account from the earlier
Stage-E work, which used `vishnu3727`. The E4 account is `vishnuvardhanksece`
and the probe kernel slug is `vishnuvardhanksece/stage-e-e4-probe`. It has so
far made two attempts and produced no rate number.

The first attempt, kernel version 1, aborted in `find_mounts()`. It could not
resolve either FlyingThings3D_subset mount, reporting images None and disparity
None, while the `/kaggle/input` listing showed only `['/kaggle/input/datasets']`.
The cause was a changed mount layout rather than a bad attachment. Kaggle's
runtime now mounts attached datasets nested, at
`/kaggle/input/datasets/<owner>/<slug>`, not at `/kaggle/input/<slug>`, and the
probe's finder only looked one level deep. The kernel's server-side metadata was
correct and listed all three dataset sources, so nothing was wrong with the
attachment itself.

The fix recorded here for the next reader is that `find_mounts()` in
`kernel_e4_probe/stage-e-e4-probe.py` now walks `/kaggle/input` to any depth,
pruning at any directory named `FlyingThings3D_subset`, and on failure prints
the listing to depth 2. `find_bundle()` already walked recursively and needed
no change.

The second attempt, kernel version 2, resolved both mounts, with images at
`/kaggle/input/datasets/arjun12367/sceneflow-flyingthings-images` and disparity
at `/kaggle/input/datasets/arjun12367/sceneflow-flyingthings-disparity`. It
then aborted at the device gate with `E4 PROBE ABORT: no GPU (rate numbers need
the T4)`. The environment record it printed was Python 3.12.13, torch
2.10.0+cpu, CUDA null, cuDNN null, GPU null, device count 0, numpy 2.0.2, and
OpenCV 4.13.0, which is a CPU-only torch build on a machine with zero visible
devices. This happened although the kernel metadata requests `enable_gpu: true`
and `machine_shape: NvidiaTeslaT4`. Kaggle accepted the push and ran the
session on CPU.

The consequence is plain. The probe has produced no rate number yet, so gate 3
of the E4 pre-run gates is still open and no epoch count is committed. Why the
GPU was not granted is not measured from inside the kernel, so it remains an
open question whether the cause is an exhausted weekly GPU quota or an
unverified account, and that has to be checked on the Kaggle website rather
than from the API.
