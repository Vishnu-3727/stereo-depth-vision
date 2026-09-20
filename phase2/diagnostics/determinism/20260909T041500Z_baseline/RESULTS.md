# Stage B — RESULTS

Deterministic baseline re-establishment, run `20260909T041500Z_baseline`. Two
200-epoch runs of the H2 / E3b arm-A configuration under the determinism controls
Stage A demonstrated, plus a 40-scene evaluation of each. **Not an architecture
experiment.** No historical record was altered.

Tags: `MEASURED` (produced here) · `HISTORICAL` (previously recorded, cited
unchanged) · `DERIVED` · `INFERRED` · `UNKNOWN`.

---

## 0. Provenance

`MEASURED`:

| | |
|---|---|
| `HEAD` | `58e8a19908ddbd35451652c61aef478f56b51ebd` — *docs: add the Phase 1 overview* |
| `phase-1-frozen^{commit}` | **`b4207e518166f5089731b4d037a4a79234f70986`** |
| `git rev-parse phase-1-frozen` | `0fc4f296b409f79bc12cd8e294a1bdb34ed1e262` — the **annotated tag object**, not a different commit. Reported, not repaired |
| `git diff phase-1-frozen -- src scripts` | **empty**, before and after both runs |
| `git status --short` | `M .gitignore`, `?? phase2/` |
| Phase 2 tracked in git | **no** — still untracked, as recorded in Stage A §0 |
| Stage A basis | `phase2/diagnostics/determinism/20260909T022238Z` |
| superseded first attempt | `phase2/diagnostics/determinism/20260909T024500Z_baseline` (preserved, `NOTE.md`, **not authoritative**) |

Full detail, including the preflight and both runs' environments, in
`ENVIRONMENT.txt`.

---

## 1. Preflight — passed before either run started

`MEASURED` (`results/preflight.json`): initial weight sha
**`c4d02385e3da28a5`**, equal to the established E3b arm-A initialisation;
423,586 parameters; 16.1995 GMAC @256×512; six refinement blocks, dilations
`[1,2,4,8,1,1]`; `cost_volume_shift=left`; `StandardisedDisparityRegression`;
seed 0; 200 epochs; `hailo_calib` 0–159 train, `hailo_val` 160–199 validate
(first 10); crop 256×512; surviving weights bit-identical across arms; H2
reference checkpoint sha256 `e3d48021d7f6a3d4…` unchanged; Phase 1 diff empty.
**PASS.**

`MEASURED` — both runs re-checked the initial hash *with the controls enabled*
and both matched. No stop condition fired.

---

## VERDICT

## `CONTROLLED-REPRODUCIBLE`

---

## REPRODUCIBILITY

`MEASURED`:

| | Run A | Run B | |
|---|---|---|---|
| **final weight sha** (concatenated parameters) | **`3ad382c50d413ad2`** | **`3ad382c50d413ad2`** | **IDENTICAL** |
| **all 72 weight tensors compared directly** | — | — | **BITWISE EQUAL — max abs diff `0.0`** |
| all 6 intermediate snapshots (ep 10/20/50/100/150/200) | — | — | **BITWISE EQUAL** |
| **200-epoch train-loss series** | — | — | **IDENTICAL — first differing epoch: none** |
| **validation series (42 points)** | — | — | **IDENTICAL — first differing point: none** |
| late window EPE mean ± sd | 3.8670192 ± 0.1113151 | 3.8670192 ± 0.1113151 | IDENTICAL |
| late window D1 mean ± sd | 23.4954125 ± 0.9342410 | 23.4954125 ± 0.9342410 | IDENTICAL |
| all ten late-window points | — | — | IDENTICAL to 10 decimals |
| epoch-200 val EPE / D1 / loss | 3.8346742 / 23.0470937 / 3.3030083 | same | IDENTICAL |
| gradient median / max | 21.9161959 / 308.0343933 | same | IDENTICAL |
| NaN·Inf / batches > 1e4 | none / 0 | none / 0 | IDENTICAL |
| matching-path gradient | 100 % of 16,000 | 100 % of 16,000 | IDENTICAL |
| epoch-10 gate / epoch-100 collapse check | PASS / PASS | PASS / PASS | IDENTICAL |
| stereo verdict @200, all five probes | +80.8127994 / +66.1876924 / 1.7754723 nats / 18.0000028 px / FUNCTIONAL | same | IDENTICAL |
| **40-scene pooled EPE / D1** | 2.2955182 / 15.6045143 | 2.2955182 / 15.6045143 | IDENTICAL |
| all 40 per-scene rows | — | — | IDENTICAL |

**Trajectory equality:** `MEASURED` — the full 200-epoch loss series and the full
42-point validation series agree exactly; the comparison reports no first
differing epoch and no first differing validation point.

**Metric equality:** `MEASURED` — every metric above, including the late-window
standard deviations and all four stereo probes, is identical.

### The two quantities that differ, and why — measured, not assumed

`MEASURED` — the checkpoint **file** sha256 differs
(`581d62e6…` vs `adcfb6d6…`). Loading both files and diffing their contents:
all 72 tensors bitwise equal (max abs diff `0.0`), and the sole differing element
of the saved payload is the embedded record name
`config.experiment: "…-ARM-A-RUNA"` vs `"…-ARM-A-RUNB"`. `DERIVED` — each
checkpoint stores its own record id beside identical weights, so the file hashes
must differ while the weights do not. The same applies to the six snapshots.

`MEASURED` — wall clock differs (4,948.997 s vs 4,927.781 s, −0.43 %). Wall clock
is not a deterministic quantity.

### Third, independent reproduction

`MEASURED` — the superseded first attempt
(`20260909T024500Z_baseline`, a different wrapper build, killed after its 200
epochs by an output-path defect) recorded **the same** late window
(3.8670191778987144 ± 0.1113151, 23.4954124793027 ± 0.9342410), the same
epoch-200 metrics, the same gradient statistics and the same four stereo-probe
values. `DERIVED` — three 200-epoch runs across two wrapper builds agree exactly.

`MEASURED` — Stage A's one-epoch controlled runs, produced by a *different
script*, recorded epoch-0 mean loss `10.532406491041183`, val EPE
`15.32539188316842`, D1 `87.44677261821234`; Stage B's first-epoch log line
reproduces all three. `DERIVED` — the deterministic protocol reproduces across
code paths, not merely across processes.

### Scope of the claim — stated exactly

```
controlled same-config variance = 0 under the tested deterministic protocol
```

`UNKNOWN` — this is **not** a statement that true experimental variance is zero.
One A/B pair tests reproducibility; it does not estimate a distribution. The
result is scoped to this execution path: this machine, this driver (616.64),
CUDA 12.8, cuDNN 90701, torch 2.7.0+cu128, these four controls, seed 0, this
configuration. Nothing here bounds variation across seeds, machines, driver
versions or library versions.

---

## FULL-40-SCENE BASELINE

`MEASURED` — all 40 `hailo_val` scenes, full 368×1232 frames, pooled over
`gt > 0` (3,802,797 pixels) — the protocol E1/E2/E3 used. Identical for Run A and
Run B. Per-scene rows: **`eval40_run_A.json`** and **`eval40_run_B.json`**
(`per_scene` array: index, scene, EPE, D1, RMSE, bad1, bad2, valid pixels,
prediction hash).

### Primary aggregate

| metric | value |
|---|---:|
| **EPE** | **2.2955182 px** |
| **D1** | **15.6045143 %** |
| RMSE | 6.0437075 px |
| bad1 / bad2 | 48.8446793 % / 25.1886966 % |
| valid pixels | 3,802,797 |

### Per-scene distribution (40 scenes)

| statistic | EPE (px) | D1 (%) |
|---|---:|---:|
| mean | 2.2591602 | 15.4255496 |
| median | 1.6920491 | 13.9697973 |
| standard deviation | 1.7051294 | 7.8169763 |
| minimum | 1.0931443 (`000171_10.png`) | 4.9776906 (`000185_10.png`) |
| maximum | 10.6710401 (`000161_10.png`) | 39.3893676 (`000161_10.png`) |

`MEASURED` — one scene dominates the spread: `000161_10.png` is the worst on both
metrics (10.67 px, 39.39 %) against a median of 1.69 px / 13.97 %. `UNKNOWN` —
why; no per-scene diagnosis was performed and none is claimed.

### Per-scene table

| # | scene | EPE | D1 | valid px |
|---:|---|---:|---:|---:|
| 0 | 000160_10.png | 1.1599 | 8.4623 | 100,174 |
| 1 | 000161_10.png | **10.6710** | **39.3894** | 114,668 |
| 2 | 000162_10.png | 2.1094 | 18.0050 | 58,878 |
| 3 | 000163_10.png | 2.2240 | 21.3468 | 100,610 |
| 4 | 000164_10.png | 5.0929 | 26.1691 | 104,123 |
| 5 | 000165_10.png | 2.6617 | 32.3245 | 101,598 |
| 6 | 000166_10.png | 1.6868 | 14.5787 | 109,818 |
| 7 | 000167_10.png | 1.9050 | 16.5400 | 98,138 |
| 8 | 000168_10.png | 5.2002 | 26.6739 | 88,776 |
| 9 | 000169_10.png | 4.5741 | 23.6780 | 53,894 |
| 10 | 000170_10.png | 1.6279 | 15.1587 | 88,820 |
| 11 | 000171_10.png | **1.0931** | 5.6767 | 84,556 |
| 12 | 000172_10.png | 1.5682 | 10.5529 | 86,649 |
| 13 | 000173_10.png | 3.0891 | 21.2422 | 116,377 |
| 14 | 000174_10.png | 2.1330 | 20.1199 | 98,882 |
| 15 | 000175_10.png | 2.4455 | 20.1341 | 65,342 |
| 16 | 000176_10.png | 2.5154 | 23.5217 | 93,637 |
| 17 | 000177_10.png | 3.2421 | 14.4195 | 96,550 |
| 18 | 000178_10.png | 2.1029 | 16.4147 | 97,571 |
| 19 | 000179_10.png | 2.1505 | 24.2079 | 78,243 |
| 20 | 000180_10.png | 1.1617 | 6.7234 | 72,597 |
| 21 | 000181_10.png | 1.4045 | 9.5326 | 73,327 |
| 22 | 000182_10.png | 1.4197 | 10.9158 | 73,343 |
| 23 | 000183_10.png | 1.1516 | 5.9315 | 97,632 |
| 24 | 000184_10.png | 1.3324 | 8.1546 | 89,496 |
| 25 | 000185_10.png | 1.1717 | **4.9777** | 101,975 |
| 26 | 000186_10.png | 1.6087 | 11.6164 | 79,095 |
| 27 | 000187_10.png | 1.3206 | 9.6030 | 77,101 |
| 28 | 000188_10.png | 1.2007 | 9.0410 | 113,240 |
| 29 | 000189_10.png | 1.6973 | 16.9839 | 128,157 |
| 30 | 000190_10.png | 1.2473 | 9.7701 | 118,771 |
| 31 | 000191_10.png | 1.2915 | 8.6610 | 81,041 |
| 32 | 000192_10.png | 1.3548 | 10.9703 | 112,239 |
| 33 | 000193_10.png | 3.2353 | 24.2126 | 121,065 |
| 34 | 000194_10.png | 1.8389 | 11.7274 | 104,209 |
| 35 | 000195_10.png | 1.3927 | 8.5950 | 117,243 |
| 36 | 000196_10.png | 1.3911 | 10.7030 | 122,704 |
| 37 | 000197_10.png | 2.1596 | 13.5201 | 125,132 |
| 38 | 000198_10.png | 2.2702 | 17.5977 | 108,997 |
| 39 | 000199_10.png | 1.4633 | 9.1691 | 48,129 |

### Why the 40-scene number must be the primary one

`MEASURED` — the same checkpoint scores **3.8346742 px / 23.0470937 %** on the
first-10-scene subset used for training-time validation and **2.2955182 px /
15.6045143 %** on all 40. `DERIVED` — the historical training-time subset is
**1.67× harder on EPE** and 1.48× on D1 than the full set. The two numbers are
not interchangeable, and the 10-scene value reproduces the epoch-200
training-time validation exactly, confirming both were computed on the same
subset.

---

## HISTORICAL COMPARISON

**Read this section as a protocol characterisation. It is NOT an architecture
comparison.** The training protocol changed; the architecture, seed, recipe,
dataset and budget did not.

`HISTORICAL` — `EXP-E3B-REFINEMENT-CAPACITY-001-ARM-A-RUN2`, **untouched**, no
determinism controls:

| | `HISTORICAL` (uncontrolled) | `MEASURED` (controlled) | `DERIVED` Δ |
|---|---:|---:|---:|
| late-window EPE mean | 4.0920997 | 3.8670192 | **−0.2250805 px** |
| late-window EPE sd | 0.1000050 | 0.1113151 | +0.0113101 |
| late-window D1 mean | 24.9097378 | 23.4954125 | **−1.4143253 pt** |
| late-window D1 sd | 1.0820962 | 0.9342410 | −0.1478552 |
| epoch-200 val EPE | 4.0752637 | 3.8346742 | −0.2405895 |
| epoch-200 val D1 | 24.2950025 | 23.0470937 | −1.2479088 |
| right-image dependence | +81.9993212 | +80.8127994 | −1.1865218 |
| matching-map dependence | +68.0680763 | +66.1876924 | −1.8803839 |
| softmax entropy | 1.7797850 | 1.7754723 | −0.0043128 |
| final disparity std | 17.6555153 | 18.0000028 | +0.3444875 |
| stereo verdict | STEREO_FUNCTIONAL | STEREO_FUNCTIONAL | same |
| wall clock | 4,361.486 s | 4,948.997 s | +587.511 s |

`DERIVED` — **enabling determinism controls, changing nothing else, moved the
late window by 0.225 px and 1.414 D1 points.** That is a *protocol* effect. It
is the same size as the effects E3b and E3c were adjudicating: E3b's frozen band
was 0.2138 px / 1.0000 pt, and O6's `HISTORICAL` same-config reproduction error
was 0.343 px / 1.970 pt.

`INFERRED`, and stated as inference — the controlled run is one draw from a
different (degenerate) execution path, not a corrected version of the historical
one. A single controlled run and a single uncontrolled run cannot separate "the
controls shift the optimum" from "the historical run was one sample of a wide
uncontrolled distribution". **`UNKNOWN` — which of those it is.** Deciding it
needs repeats of the *uncontrolled* configuration, which Stage B did not run.

`DERIVED` — **no architectural conclusion follows from this table, and none is
drawn.** The six-block architecture is the only architecture measured here.

---

## DETERMINISTIC OVERHEAD

`MEASURED`:

| | wall clock (200 epochs) | s/epoch |
|---|---:|---:|
| `HISTORICAL` E3b arm A, uncontrolled | 4,361.486 s = 72.69 min | 21.807 |
| Stage B Run A, controlled | **4,948.997 s = 82.48 min** | **24.745** |
| Stage B Run B, controlled | **4,927.781 s = 82.13 min** | **24.639** |

`DERIVED` — overhead **+13.47 %** (Run A) and **+12.98 %** (Run B) against the
historical uncontrolled run.

`MEASURED` — Stage A's independent one-epoch measurement gave 36.1 / 37.5 s
uncontrolled against 42.0 / 41.2 s controlled, `DERIVED` **+13.04 %** — agreeing
with the 200-epoch figure to within half a point.

`DERIVED` — cost of full reproducibility on this harness is **≈ +13 %**, i.e.
about **10 minutes per 200-epoch run**. This is an engineering metric; it was not
optimised and should not be.

---

## IMPACT ON E3b / E3c

> **Historical E3b/E3c materiality conclusions remain INDETERMINATE. This
> experiment does not retroactively alter them.**

`MEASURED` — no E3b, E3c, O6 or H2 record, metric, verdict, preregistration or
report was read-write during Stage B. All were imported or cited read-only.

`DERIVED` — Stage B establishes that the protocol under which E3b and E3c ran is
replaceable by one with zero same-config variance, and that switching protocols
itself moves the late window by 0.225 px / 1.414 points. Neither fact revises an
E3b or E3c number; both mean any *future* comparison should be made under the
controlled protocol, where no materiality band is needed at all.

`UNKNOWN` — what E3b's and E3c's arms would measure under the controlled
protocol. Nothing in Stage B answers that, and Stage B deliberately trained no
five-block, four-block or three-block arm.

---

## NEXT EXPERIMENT

**One recommendation.**

> **Train the missing factorial cell: `cost_volume_shift="none"` with the
> standardised readout, 200 epochs, seed 0, under the Stage B deterministic
> protocol, against this Stage B Run A baseline as its control.** ≈83 minutes,
> one arm, one changed variable.

Why this one, from the evidence:

- `MEASURED` — Stage B provides, for the first time in this project, a
  contemporary control with **zero** same-config variance. Any single arm trained
  under the same protocol is now directly comparable to it with no band and no
  noise model.
- `HISTORICAL` — the 2×2 of {shift on/off} × {standardisation on/off} has three
  cells filled at 200 epochs and this one empty. The `PHASE_2_H2_EVIDENCE_REVIEW`
  ranks it its own **#1 unknown**; H3 decided only that it did not merit the
  compute, on an epoch-10 criterion the project itself later invalidated.
- `DERIVED` — its expected effect is the H2-vs-H1 gap, of order 10 px EPE and
  50 D1 points: two orders of magnitude above the protocol effects measured here,
  so it is interpretable regardless of any residual uncertainty about bands.
- `DERIVED` — it decides whether Stage E should ever open, which is the project's
  oldest standing question.

Explicitly **not** recommended next, and **not run here**: E3b or E3c reruns, any
block-count or dilation arm, E2b, Scene Flow, or any architecture search.

**Not executed in this task.**
