# EXP-CORRESPONDENCE-GEOM-A — PRE-REGISTRATION (Option A)

**Does `disparity_initial` shift by `−Δ/16` candidates under horizontal
right-image translation, on scenes never previously used — with a signed,
axis-asymmetric response that exceeds the search-free baseline?**

Record: `phase2/diagnostics/correspondence_geom/20260914T072255Z/`.
Staged 2026-09-14. Stage 1b of EXP-CORRESPONDENCE-GEOM-003 (Option A).
**BUILD AND FREEZE ONLY. NOT EXECUTED. Freeze COMPLETE (see §0.2).
No checkpoint was opened in this record.**

Inference only. No training, no fine-tuning, no weight change, no architecture
change, no new checkpoint, no preprocessing change. Phase 1 untouched. No
historical record modified. No `RESULTS.md`. No git commit.

**Artefacts produced in Stage 1b, with file-byte sha256:**

```
translation_spec.json  sha256 7d0f0e10910fcd72f76f130665a59bbe238d146d110c993d843ef3737fc4267c
mask_spec.json         sha256 61e81f72e6fe553f52abfaad97ef1cb56cf6a30f8194f235ce9d4dde1ac03a2e
scene_selection.json   sha256 edbe049119a1f0a9e0d6aeb08aee8785a711ffeb48a6b511560b6a2066980789
```

**Freeze status.** The freeze COMPLETED: `freeze_geomA.py` ran to completion
and wrote `mask_spec.json` plus `frozen.sha256` over all three specs. The
three digests above are file-byte digests of the as-frozen files and are the
digests `run_geomA.py` re-verifies at HS-FREEZE before any model is
instantiated. No specification content was relaxed to reach the freeze: no
threshold relaxation, no border reduction, no band change, no decision-rule
change, no claim-ceiling change. The only change relative to the halted
predecessor `20260914T070551Z` is the removal of the invented fixed scene
count N = 4 (see §0.2).

---

## 0.1. KNOWN LIMITATIONS, ACCEPTED BEFORE EXECUTION

This section stands before the design, is part of the preregistration, and its
wording is not softened:

- This design was assessed in `SUCCESSOR_OPTIONS.md` as "defensible" but
  underpowered: weaker than the experiment it replaces, and was NOT the
  recommended successor. (The recommended successor was Option B, the
  synthetic fronto-parallel self-pair sweep.)
- The image-translation slope class and the vertical-shift null were both
  marked RETIRED / INVALID in
  `direction_decision_20260913T052115Z/DECISION.md` section 1. Quoted verbatim:
  - * **Image-translation slope test** — REJECTED; the degenerate mechanism imitates sign and monotonicity.
  - * **Vertical shift as a null** — INVALID on two independent grounds (no axis asymmetry for `shift="none"`; and `FH = 17 < 30` cells makes a halo-free vertical band empty, unfixable at any crop size below 496 px).
  The vertical arm is therefore reported as a CONTAMINATED comparison, never
  as a valid null.
- Only 2 signed offsets exist, so C2 is a single inequality: probability 1/2
  under any null.
- The 3 trained checkpoints are reused and irreplaceable; the designer has
  seen the GEOM-001 response curve. Achievable status is at most CONSISTENCY
  EVIDENCE, never independent confirmation.
- This run is executed at the project owner's explicit direction, recorded as
  such.
- On PASS the claim ceiling is frozen verbatim and does not exceed:
  "consistency evidence for a signed, axis-asymmetric horizontal translation
  response on held-out scenes; it establishes nothing at Level D."
- On FAIL the verdict string is GEOMETRIC-CORRESPONDENCE-NOT-DEMONSTRATED,
  and it must not be read as evidence that correspondence is absent.

### 0.1.1. STAGE-1b SCENE COUNT: 2 SCENES, 6 POSITIVE UNITS (power loss recorded, nothing softened)

- The frozen scene count in this record is 2 (scenes 5 and 9 — the only
  held-out `hailo_val` scenes whose band retention at border 96 clears 0.50).
  Nothing above is softened by this subsection; every limitation stated in
  §0.1 stands as written.
- Relative to GEOM-001 this halves the unit count: 3 trained checkpoints × 2
  scenes = **6 positive units instead of 12**. The global PASS bar is C1∧C2∧C3∧C4
  at every positive unit (6/6).
- The power loss from the reduced unit count is **on top of** the
  already-stated 1/2 monotonicity probability (§0.1: only 2 signed offsets
  exist, so C2 is a single inequality with probability 1/2 under any null).
- The preceding record `20260914T070551Z` halted at HS-SELECTION (its brief
  demanded an invented fixed N = 4; only 2 scenes qualified) and is preserved
  unmodified. This record removes only that invented N; border 96, threshold
  0.50, band [2,9], decision rule C1..C4, and claim ceiling are unchanged.

## 0.2. STAGE-1b FINDING: SCENE SELECTION (criterion corrected, freeze completed)

`select_scenes.py` was run as required (GT only; no checkpoint opened). The
frozen criterion — exclude GEOM-001 scenes {27, 0, 31, 6}; keep ALL remaining
held-out scenes with band retention ≥ 0.50, in ascending scene index; no N;
halt if fewer than 2 qualify — admits **exactly 2 scenes**:

| scene index | name | band retention (band px / valid-interior px) |
|---:|---|---:|
| 5 | 000165_10.png | 0.7124 (41068/57645) |
| 9 | 000169_10.png | 0.7129 (24692/34635) |

Next-best held-out scenes: 12 (0.4403), 8 (0.4215), 13 (0.4042) — none within
reach of the threshold under any admissible reading of the retention
definition (the implemented reading, denominator = valid-GT pixels inside the
geometric mask interior, is the most favourable one; broader denominators
only lower every retention). The full per-candidate table is frozen in
`scene_selection.json` (`criterion_met: true`, `chosen_scenes: [5, 9]`).

History, stated plainly: the predecessor record `20260914T070551Z` carried a
brief that additionally demanded a fixed N = 4. That N was invented by the
manager — the authoritative design (`successor_audit_20260911/SUCCESSOR_OPTIONS.md`,
OPTION A) fixes only the criterion, never a scene count. At the mandated
border 96 = M(32) + B(64) (§4) only 2 held-out scenes clear 0.50 (the audit's
"17 of 40 scenes qualify" figure in `AUDIT.md` §3 was MEASURED at border 32,
which removes fewer of the bottom rows where the large disparities live), so
the predecessor correctly halted at HS-SELECTION and is preserved unmodified.
This record removes only the invented N; border 96, threshold 0.50, band
[2,9], decision rule C1..C4, and claim ceiling are UNCHANGED.

Consequences, all recorded in advance:

- `freeze_geomA.py` ran to completion: `mask_spec.json` (scenes [5, 9],
  pooled retained pixels 65760: 41068 + 24692) and `frozen.sha256` over all
  three specs exist with the digests listed in the header.
- `run_geomA.py` is written but NOT executed in this stage regardless.
  `N_SCENES` and the forward-pass budget are read from the frozen
  `scene_selection.json` at runtime (HS-COUNT: 4 models × 2 scenes × 9
  conditions = 72 forward passes).
- Status of this record: **FROZEN-NOT-EXECUTED**. No repair-and-continue is
  authorised by this document; any further change belongs in a NEW record
  with its own preregistration, never as an edit to this one.
- The GT pipeline itself is validated: before any new measurement was
  trusted, this stage bit-exactly reproduced GEOM-001's four frozen mask
  counts (34245 / 46199 / 33250 / 50747) from GT alone.

---

## 1. SOURCE-DERIVED GEOMETRY — proof, not inference

As preregistered in GEOM-001 §1, re-derived here from the same named files.
**No previous experimental output was used to establish any of it.**

### 1.1 The cost volume

`src/models/stereonet/cost_volume.py`:

```python
def shift_left(x, k):  return F.pad(x, (0, k))[..., k:]      # shifted[..., u] = x[..., u+k]
levels[k] = shifter(left, k) - right
volume = torch.stack(levels, dim=1).permute(0, 2, 1, 3, 4)   # (B, C, D, H, W)
```

⇒ **`V[:, c, k, y, u] = Lf[c, y, u+k] − Rf[c, y, u]`**

### 1.2 Feature stride

`src/models/stereonet/feature_extractor.py`: `4 × Conv2d(5, stride=2, padding=2)`,
then `6 × ResBlock(3×3, stride 1)`, then `Conv2d(3, stride=1, padding=1)`.
**Fully convolutional — no pooling, no normalisation, no global operator.**
`scale = 2**downsample_levels = 2**4 = 16`.

⇒ an image translation of `16n` px produces **exactly** an `n`-sample feature
translation in the interior. Every `Δ` is a multiple of 16.

### 1.3 Candidate ↔ displacement ↔ disparity

```
feature column u  ↔  image column 16u
x_L = 16(u+k),  x_R = 16u    ⇒    x_L − x_R = 16k
⇒ candidate k is a disparity of 16k full-resolution px;  k = d/16
```

`src/models/stereonet/regression.py`: `soft_argmin` over the index grid `0…11`
⇒ `disparity_initial` is in **candidate units**.

### 1.4 Sign convention — independently corroborated

`src/datasets/kitti2015.py`: GT in full-resolution px.
`phase2/viz/core.correspondence(x_left, d) = x_left − d`, photometrically
confirmed. GT is **left-referenced**; the volume is **right-referenced**. The
origin cancels because the statistic is a *difference* between conditions over
a fixed pixel set.

### 1.5 Horizontal translation — the anchor, derived two independent ways

**Route 1 — through the volume.** With `Rf'[c,y,u] = Rf[c,y,u−δ]`, `δ = Δx/16`:
`k = k₀ − δ`. **Route 2 — through the GT convention.**
`d' = d − Δx` ⇒ `d'/16 = d/16 − Δx/16`. Both give

```
Δ(disparity_initial) = − Δx / 16   candidates
ANCHOR:  S_h = −1/16 = −0.0625 candidates per image pixel
```

**The anchor is reported DESCRIPTIVELY ONLY, never as a pass criterion.**

### 1.6 Vertical translation — no anchor exists; contaminated comparison

As in GEOM-001 §1.6 the candidate axis indexes horizontal displacement only,
so no candidate shift is predicted under any hypothesis — and per
`DECISION.md` §1 (quoted in §0.1) the vertical shift is INVALID as a null on
two independent grounds. The vertical arm is a matched-magnitude
CONTAMINATED comparison: it is measured, reported, and used in C3 exactly as
preregistered, but it is never a valid null.

---

## 2. WHY EACH MECHANISM PREDICTS WHAT IT PREDICTS

```
Odd(Δ)  = [ m(+Δ) − m(−Δ) ] / 2        — where a SIGNED geometric response lives
Even(Δ) = [ m(+Δ) + m(−Δ) ] / 2        — where a MAGNITUDE artifact lives (descriptive only)
```

| Mechanism | Predicts | Because |
|---|---|---|
| **true geometric correspondence** | `Odd_h(Δ) = −Δ/16`, `S_h = −0.0625`; `S_v ≈ 0` | the readout locates the candidate at which `Lf[u+k] ≈ Rf[u]` (§1.5) |
| **generic binocular dependence** | `Odd ≈ 0` both axes; `S_h ≈ S_v` | `Rf[c,y,u]` enters the volume with **no `k` dependence** (§1.1), and `standardise_across_disparity` removes any `k`-constant term exactly |
| **even architectural artifact** | `Odd ≈ 0`, non-monotone | the aggregation is `5 × Conv3d(3×3×3, padding=1)` over `D = 12`; D-axis zero padding produces structure from perturbation *size*, not direction |
| **search-free / degenerate response** | measured, not assumed | `shift="none"` volume is `k`-constant by construction; whatever `S_h` it shows is odd-leakage artifact |
| **image-statistical response** | `S_h ≈ S_v` up to anisotropy | same `|Δ|`, same crop, same code path; differs only in axis |

---

## 3. INTERVENTION — crop-based translation, frozen in `translation_spec.json`

**No fill. No padding. No interpolation.** All crops are integer-aligned. The
axis is a function ARGUMENT (`right_origin(axis, d)`), not a branch in the
forward path.

```
left  crop : cols [M, W−M),      rows [M, H−M)        FIXED for every condition
right crop : horizontal  cols [M−Δ, W−M−Δ), rows [M, H−M)
             vertical    cols [M, W−M),      rows [M−Δ, H−M−Δ)

W = 1232, H = 368, M = 32 = max|Δ|
crop size = 1168 × 304 = (73 × 16) × (19 × 16)     identical for EVERY condition
```

```
d_crop = (x_L − M) − (x_R − M + Δ) = d_orig − Δ     [exact, integer]
```

### 3.1 Translation values — frozen

```
Δ ∈ { −32, −16, 0, +16, +32 }   px, both axes
9 distinct conditions (Δ = 0 shared between axes)
```

`±48` is rejected on geometry: the symmetric representability band would be
`[3,8]`, which reaches ≥ 50 % retention on 0 of 40 scenes (`AUDIT.md` §3,
MEASURED). Only 2 signed offsets exist, so C2 is a single inequality with
probability 1/2 under any null (§0.1).

---

## 4. MASK — single fixed band mask, frozen in `mask_spec.json`

```
retain (y,x) iff  GT[y,x] > 0  AND  GT[y,x]/16 ∈ [2,9]
                  AND  96 ≤ y < 272  AND  96 ≤ x < 1136     (ORIGINAL coords)
96 = M(32) + B(64)
mask_crop[y, x] = mask_original[y + 32, x + 32]
```

Derivation of the border: M = 32 is this experiment's crop margin
(`max|Δ|`). B = 64 is the inner border GEOM-001 used (its `mask_spec.json`
`edge_exclusion_decomposition`: M = 48, B = 64, EDGE = 112). The brief
requires the border to be at least M plus the border GEOM-001 used;
32 + 64 = 96 is that minimum, adopted exactly. (Full derivation also frozen
in `mask_spec.json` `edge_derivation`.)

- **Single, Δ-independent, axis-independent, model-independent.**
- The band `[2,9]` is the representability band derived in `AUDIT.md` §1 for
  `|Δ| = 32` (2 candidates): `Δ/16 ≤ d ≤ 11 − Δ/16`.

---

## 5. SCENES — GT-only selection, fixed before any model is built

Split `hailo_val`. Exclude the 4 GEOM-001 scenes {27, 0, 31, 6}. From the
remaining held-out scenes keep ALL whose band retention (fraction of
valid-GT pixels inside the geometric mask interior with GT/16 in [2,9]) is
≥ 0.50, in ascending scene index. No N. Halt if fewer than 2 qualify
(HS-SELECTION).

`select_scenes.py` computes this from GT alone and writes
`scene_selection.json` (criterion, retention per candidate scene, chosen
list). Measured outcome: scenes 5 and 9 qualify — see §0.2. **The scene list
[5, 9] is frozen; the mask over those scenes is frozen in `mask_spec.json`.**

---

## 6. PRIMARY STATISTIC — frozen

```
m_axis(Δ) = np.median( disparity_initial[mask] )            candidate units

Odd_axis(Δ)  = [ m_axis(+Δ) − m_axis(−Δ) ] / 2              Δ ∈ {16, 32}
Even_axis(Δ) = [ m_axis(+Δ) + m_axis(−Δ) ] / 2              descriptive only

S_axis = Σ_Δ Odd_axis(Δ)·Δ  /  Σ_Δ Δ²                       OLS through the origin
F_axis = S_axis / (−1/16)                                   descriptive only
```

`np.median` — the same aggregate used by INDEX-001/002/ARCH-001/GEOM-001.
**`m(0)` cancels from `Odd` by construction.** No plain OLS over the raw
response is computed or reported as primary. `F_h` is reported descriptively
and is **not** a pass criterion.

---

## 7. CONDITIONS AND CONTROLS

| role | model | shift | seed | sha256 |
|---|---|---|---|---|
| **test** | `POS_6b_seed0` | `left` | 0 | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` |
| **test** | `POS_6b_seed1` | `left` | 1 | `58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf` |
| **test** | `POS_6b_seed2` | `left` | 2 | `245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69` |
| **artifact baseline** | `NEG_shift_none` | `none` | — | `d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1` |

The SAME four checkpoints as GEOM-001, sha256 copied verbatim including all
values. All frozen, read-only, with the standardised parameter-free readout
(verified from source at load). The 3 trained checkpoints are reused and
irreplaceable (§0.1).

---

## 8. DECISION RULE — threshold-free, unit-level

Per positive unit `u = (checkpoint, scene)`:

```
C1  sign          :  S_h(u) < 0                                correct sign
C2  monotone      :  m_h(−32) > m_h(−16) > m_h(0) > m_h(+16) > m_h(+32)
                                                              strictly; probability 1/2 under the null
C3  axis-specific :  |S_h(u)| > |S_v(u)|
C4  artifact-specific : |S_h(u)| > |S_h(NEG, same scene)|      exceeds the search-free baseline
```

Global:

```
PASS  iff  C1 ∧ C2 ∧ C3 ∧ C4 at EVERY positive unit (6/6)
otherwise FAIL
```

No majority rule, no partial counts, no threshold relaxation, no post-hoc
exception. No p-values: the 6 units share 3 weight sets and 2 scenes and are
not independent replicates; this is a deterministic causal diagnostic, not a
population inference.

---

## 9. FAILURE MODES — distinguished, not collapsed

| # | Outcome | Signature |
|---|---|---|
| 1 | **consistency-positive** | C1∧C2∧C3∧C4 at 6/6 (claim ceiling §0.1, §12) |
| 2 | **generic binocular dependence** | `S_h ≈ S_v`, both small; C3 fails |
| 3 | **even architectural artifact** | `Odd` small and non-monotone; C2 fails |
| 4 | **search-free artifact with odd leakage** | positives do not exceed NEG on the same scene; C4 fails |
| 5 | **ambiguous / mixed** | C1–C4 hold at some units, not all |
| 6 | **protocol failure** | any §10/§11 hard stop (including HS-SELECTION, HS-FREEZE) |

---

## 10. CONTROLS AND HARD STOPS

Each raises and halts; no repair-and-continue.

| check | criterion |
|---|---|
| HS-FREEZE | `translation_spec.json`, `mask_spec.json`, `scene_selection.json` sha256 OVER FILE BYTES, and the four checkpoint sha256s, re-verified BEFORE the first model is instantiated; digests recorded in this document |
| HS-REPRESENTABILITY | from the specs IN THIS RECORD ONLY (never a figure imported from another record): for each chosen scene and each Delta, the predicted response (GT/16 + Delta/16, horizontal arm) stays inside the 0..11 candidate axis for ≥ 50 % of masked pixels |
| HS-CROP | every crop window inside the image; one crop shape only (1168 × 304); all Delta multiples of 16; `max\|Delta\|` == crop margin |
| HS-NOTRAIN | no optimizer imported or constructed; no `.backward()`, no `.step()`; all model parameters re-hashed after every scene and asserted unchanged |
| HS-DETERMINISM | `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `no_grad`, `eval`, fp32 |
| HS-COUNT | exactly 72 forward passes accounted for (4 models × 2 frozen scenes × 9 conditions); `N_SCENES` and the budget are read from the frozen `scene_selection.json`, never hard-coded |
| HS-SELECTION (freeze) | `scene_selection.json` carries `criterion_met: true` with ALL qualifying held-out scenes in ascending order (at least 2), else the freeze writes nothing |
| mask invariance | retained counts equal the frozen `mask_spec.json` table exactly, every scene |
| checkpoint hashes | the four §7 sha256, verified at load |
| readout parameter-free | `sum(p.numel() for p in model.regression.parameters()) == 0`, `num_disparities == 12` |
| identical code path | axis is a function argument; asserted |

HARD STOP on any failure above, on any protocol modification after execution
begins, or if any historical record would need modification.

---

## 11. COMPUTE

```
4 models × 2 scenes × 9 conditions = 72 forward passes to disparity_initial
```

Each pass: two feature towers + cost volume + aggregation + standardised
readout on an `1168 × 304` crop. Refinement never invoked. **Estimate well
under 2 minutes.** No training, no new checkpoint, no new architecture.

---

## 12. CLAIM CEILING

**On PASS**, the maximum claim is frozen verbatim:

> "consistency evidence for a signed, axis-asymmetric horizontal translation
> response on held-out scenes; it establishes nothing at Level D."

Status at most CONSISTENCY EVIDENCE, never independent confirmation (§0.1).

**On FAIL**: `GEOMETRIC-CORRESPONDENCE-NOT-DEMONSTRATED`, classified by §9.
It must **not** be concluded that correspondence is absent.

---

## 13. PREREGISTRATION ORDER — as executed

1. scene criterion fixed (ALL qualifying held-out scenes ≥ 0.50, ascending; no N; halt if < 2); `select_scenes.py` written ✔
2. `select_scenes.py` run (GT only; required and allowed) → `scene_selection.json` ✔ (`criterion_met: true`, `chosen_scenes: [5, 9]`, §0.2)
3. translation values frozen → `translation_spec.json` ✔ (byte-identical to `freeze_geomA.py` canonical output)
4. mask freeze → `freeze_geomA.py` ran to completion → `mask_spec.json` + `frozen.sha256` over all three specs ✔
5. geometry derivation frozen → §1 ✔
6. statistic frozen → §6 ✔
7. decision rule frozen → §8 ✔ (PASS iff C1∧C2∧C3∧C4 at 6/6)
8. controls frozen → §7, §10 ✔ (`N_SCENES` and forward budget from frozen `scene_selection.json`; HS-COUNT 72)
9. `PREREGISTRATION.md` written → this document ✔ (records all three frozen spec digests)
10. `frozen.sha256` → produced by the completed freeze ✔
11. **execute** → NOT DONE (`run_geomA.py` written, not run; Stage 1b is BUILD AND FREEZE ONLY)
12. `RESULTS.md` → NOT CREATED

**No result may be inspected between 9 and 11. No protocol modification after
execution begins.**

---

## 14. OUTPUTS

Present now (design + frozen artifacts): `PREREGISTRATION.md`,
`translation_spec.json`, `mask_spec.json`, `scene_selection.json`,
`frozen.sha256`, `select_scenes.py`, `freeze_geomA.py`, `run_geomA.py`
(written, NOT run), `RELATED_RUNS.md`.

Withheld: nothing — the freeze is complete.

To be created **at execution only** (execution is NOT part of this stage): `RESULTS.md`, `results.json`, `ENVIRONMENT.txt`,
`run.log`.

**`RESULTS.md` does not exist and must not be created before execution. No
checkpoint was loaded in this stage.**
