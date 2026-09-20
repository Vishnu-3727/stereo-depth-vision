# EXP-CORRESPONDENCE-GEOM-001 — PRE-REGISTRATION

**Does the trained network's response to a controlled horizontal right-image
translation behave differently from its response to a matched vertical
translation, in the direction predicted by the exact source-derived feature
geometry?**

Record: `phase2/diagnostics/correspondence_geom/20260911T111809Z/`.
Frozen 2026-09-11T11:18Z. **NOT EXECUTED.**

Inference only. No training, no fine-tuning, no weight change, no architecture
change, no new checkpoint, no preprocessing change. Phase 1 untouched. No
historical record modified.

**Frozen artefacts, written before this document and hashed into it (B14 steps
1–2):**

```
translation_spec.json  sha256 4f5047404b41ec906a98e62f9d31159f9ac8f11fc49455f0fcf64ef8e2397c97
mask_spec.json         sha256 5a2927a783b1569f67a7cca460ffb041bc281e26fc736d417a99f8dbcf417330
```

**Integrity note, recorded rather than silently corrected.** The two digests
above were first computed over the JSON *content string*. `Path.write_text` then
applied Windows CRLF translation, so the *file bytes* initially hashed
differently and an external `sha256sum` would not have reproduced these values.
Before this preregistration was finalised — and with **nothing executed and no
result in existence** — both spec files were rewritten with `newline=""` so that
LF is preserved and file bytes equal content bytes. **The recorded digests above
are unchanged and now verify both ways**; `frozen.sha256` carries the same two
values. No specification content was altered: only the line terminators. The
harness must verify these digests over **file bytes** at start-up.

---

## 1. SOURCE-DERIVED GEOMETRY (B1) — proof, not inference

Every link read from the file named. **No previous experimental output was used
to establish any of it.**

### 1.1 The cost volume

`src/models/stereonet/cost_volume.py`:

```python
def shift_left(x, k):  return F.pad(x, (0, k))[..., k:]      # shifted[..., u] = x[..., u+k]
levels[k] = shifter(left, k) - right
volume = torch.stack(levels, dim=1).permute(0, 2, 1, 3, 4)   # (B, C, D, H, W)
```

⇒ **`V[:, c, k, y, u] = Lf[c, y, u+k] − Rf[c, y, u]`**

Note for §2: `Rf[c,y,u]` carries **no `k` dependence**.

### 1.2 Feature stride

`src/models/stereonet/feature_extractor.py`: `4 × Conv2d(5, stride=2, padding=2)`,
then `6 × ResBlock(3×3, stride 1)`, then `Conv2d(3, stride=1, padding=1)`.
**Fully convolutional — no pooling, no normalisation, no global operator.**
`scale = 2**downsample_levels = 2**4 = 16`.

⇒ an image translation of `16n` px produces **exactly** an `n`-sample feature
translation in the interior. This is why every `Δ` is a multiple of 16.

### 1.3 Candidate ↔ displacement ↔ disparity

```
feature column u  ↔  image column 16u
x_L = 16(u+k),  x_R = 16u    ⇒    x_L − x_R = 16k
⇒ candidate k is a disparity of 16k full-resolution px;  k = d/16
```

`src/models/stereonet/regression.py`: `soft_argmin` over the index grid `0…11`
⇒ `disparity_initial` is in **candidate units**.

### 1.4 Sign convention — independently corroborated

`src/datasets/kitti2015.py`: `read_disparity_png(scale=256.0)` ⇒ GT in
full-resolution px. `phase2/viz/core.correspondence(x_left, d) = x_left − d`,
i.e. `x_R = x_L − d`, docstring: *"Confirmed photometrically against KITTI ground
truth in the test suite; the opposite sign is roughly five times worse on real
data."*

⇒ GT is **left-referenced**; the volume is **right-referenced** (indexed at
`x_R`). Magnitudes agree (`k = d/16`); only the indexing origin differs. **The
origin cancels in this experiment** because the statistic is a *difference*
between conditions over a fixed pixel set.

### 1.5 Horizontal translation — the anchor, derived two independent ways

Implemented as a crop (§3), algebraically identical to translating the right
image and discarding the vacated region.

**Route 1 — through the volume.** With `Rf'[c,y,u] = Rf[c,y,u−δ]`, `δ = Δx/16`:

```
V'[:,c,k,y,u] = Lf[c,y,u+k] − Rf[c,y,u−δ]
a match needs Lf[u+k] ≈ Rf[u−δ];  the original match at right-column u−δ was
Lf[(u−δ)+k₀] ≈ Rf[u−δ]   ⇒   u+k = u−δ+k₀   ⇒   k = k₀ − δ
```

**Route 2 — through the GT convention.** A point at `x_R` moves to `x_R+Δx`;
`x_L` is unchanged; `d' = x_L − (x_R+Δx) = d − Δx` ⇒ `d'/16 = d/16 − Δx/16`.

Both give

```
Δ(disparity_initial) = − Δx / 16   candidates
ANCHOR:  S_h = −1/16 = −0.0625 candidates per image pixel
```

**The anchor is accepted because §1.1–§1.5 prove it, not because it was proposed
before** (B4 requirement).

### 1.6 Vertical translation — why no anchor exists

`Rf'[c,y,u] = Rf[c,y−η,u]`, `η = Δy/16`, so
`V'[:,c,k,y,u] = Lf[c,y,u+k] − Rf[c,y−η,u]`. The volume compares **within the
same row `y`**; the candidate axis indexes horizontal displacement only. No `k`
restores a match. **No candidate shift is predicted, under any hypothesis**,
while the perturbation magnitude is comparable to the horizontal case. This is
what makes it a matched null rather than a second test.

---

## 2. WHY EACH MECHANISM PREDICTS WHAT IT PREDICTS (B2)

Define, for the spatially aggregated response `m(Δ)`:

```
Odd(Δ)  = [ m(+Δ) − m(−Δ) ] / 2        — where a SIGNED geometric response lives
Even(Δ) = [ m(+Δ) + m(−Δ) ] / 2        — where a MAGNITUDE artifact lives
```

| Mechanism | Predicts | Because |
|---|---|---|
| **true geometric correspondence** | `Odd_h(Δ) = −Δ/16`, `S_h = −0.0625`; `S_v ≈ 0` | the readout locates the candidate at which `Lf[u+k] ≈ Rf[u]`; translating the right image moves that candidate by exactly `−Δx/16` (§1.5). Vertical translation moves content off-row, where the candidate axis cannot reach (§1.6). |
| **generic binocular dependence** | `Odd ≈ 0` both axes; `Even` grows with `\|Δ\|`; `S_h ≈ S_v` | `Rf[c,y,u]` enters the volume with **no `k` dependence** (§1.1), and `standardise_across_disparity` subtracts the per-pixel mean across `k`, **exactly removing** any `k`-constant term. A perturbation of the right image therefore carries no candidate-selective signal; it can change the response magnitude but has no mechanism to produce a *signed* candidate displacement proportional to `Δ`. |
| **even architectural artifact** | `Odd ≈ 0`, `Even` dominant, non-monotone | the aggregation is `5 × Conv3d(3×3×3, padding=1)` over `D = 12`; the D-axis zero padding produces structure from perturbation *size*, not direction. Stage A measured exactly this on a bit-exactly constant volume. |
| **search-free / degenerate response** | measured, not assumed | `shift="none"`: `reference_shift` pads right then slices `[0:W]`, returning its input, so `V(k)` is `k`-constant and **no search exists by construction** (EXP-010, re-confirmed by Stage A). Whatever `S_h` it shows is pure artifact odd-leakage. |
| **image-statistical response** | `S_h ≈ S_v` up to image anisotropy | horizontal and vertical translations of the same `\|Δ\|` over the same crop differ only in axis; any residual asymmetry is image anisotropy, which the `shift="none"` control also experiences and therefore bounds. |

**Correspondence is the only mechanism with a reason to produce a signed
response whose sign and slope are fixed by geometry.**

---

## 3. INTERVENTION — crop-based translation, frozen in `translation_spec.json`

**No fill. No padding. No interpolation.** All crops are integer-aligned.

```
left  crop : cols [M, W−M),      rows [M, H−M)        FIXED for every condition
right crop : horizontal  cols [M−Δ, W−M−Δ), rows [M, H−M)
             vertical    cols [M, W−M),      rows [M−Δ, H−M−Δ)

W = 1232, H = 368, M = 48 = max|Δ|
crop size = 1136 × 272 = (71 × 16) × (17 × 16)     identical for EVERY condition
```

**Disparity algebra, exact and verified by integer arithmetic:**

```
d_crop = (x_L − M) − (x_R − M + Δ) = d_orig − Δ
```

**Why not the zero-fill of Stage A (a deliberate, declared change).** A
zero-filled shift puts the fill discontinuity on the *left* edge for `+Δ` and the
*right* edge for `−Δ`. Those are different perturbations, so a pure fill artifact
can contribute to the **odd** part and contaminate the primary statistic.
Cropping discards the vacated region instead of filling it, so every condition
sees only real image content and every condition has identical output size. This
is a strict improvement and is recorded as a departure from Stage A.

**Residual limitation, declared.** The feature receptive field is ≈477 px
(`4 × 5×5 stride-2` → 61 px, plus 13 × `3×3` at stride 16 → 416 px). No
affordable border can exclude it — the image is only 368 px tall. Differing real
content therefore still enters the receptive field near the crop edge. It is not
removed; it is **measured**, by the `shift="none"` control (§6).

### 3.1 Translation values — frozen, and `±64` rejected

```
Δ ∈ { −48, −32, −16, 0, +16, +32, +48 }   px, both axes
13 distinct conditions (Δ = 0 shared between axes)
```

Chosen by the B5 criteria, from geometry only:

| criterion | how `±16/±32/±48` satisfies it |
|---|---|
| measurable signal if correspondence exists | `48/16 = 3` candidates — large against the model's own ≈0.5-candidate noise floor |
| avoids boundary contamination | `max\|Δ\| = M`, so the right crop window never leaves the image |
| within valid image support | right crop bounds over the sweep: cols `[0, 1232)`, rows `[0, 368)` — verified |
| permits odd/even decomposition | 3 signed pairs; monotonicity over 3 points is a real constraint |
| no clipping artifacts | see below |
| cheap | 208 forward passes |

**`±64` is rejected on geometry.** Using the identity medians already published
in ARCH-001 (3.84 … 5.83 candidates), the predicted response range is:

```
|Δ| = 16 →  2.84 … 6.83   safe
|Δ| = 32 →  1.84 … 7.83   safe
|Δ| = 48 →  0.84 … 8.83   safe
|Δ| = 64 → −0.16 … 9.83   RAILS at the candidate floor 0
```

At `±64` the soft-argmin would clip, breaking the linearity the odd-part slope
assumes. **Rejected because the geometry says so, not because of expected
results.** This overrides the `±64` proposed in `next_direction_audit_20260911/ADDENDUM.md`.

---

## 4. MASK — single fixed mask, frozen in `mask_spec.json`

```
retain (y,x) iff  GT[y,x] > 0  AND  112 ≤ y < 256  AND  112 ≤ x < 1120     (ORIGINAL coords)
112 = M(48) + B(64)
mask_crop[y, x] = mask_original[y + 48, x + 48]
```

**Frozen retained counts** — these must reproduce exactly at execution:

| scene | retained | GT/16 mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| 000187_10 | **34 245** | 1.132 | 1.045 | 0.305 | 2.597 |
| 000160_10 | **46 199** | 1.204 | 1.165 | 0.294 | 3.167 |
| 000191_10 | **33 250** | 1.161 | 1.105 | 0.297 | 3.392 |
| 000166_10 | **50 747** | 1.496 | 1.552 | 0.295 | 2.844 |
| **pooled** | **164 441** | | | | |

- **Single, Δ-independent, axis-independent, model-independent.** The crop size
  is identical for every condition, so one pixel set is well defined for all of
  them. A Δ-dependent mask would change the sample between conditions and bias
  the paired difference — it is therefore rejected.
- **Why a border at all**, given there is no fill: to reduce the residual
  influence of differing real content entering the receptive field near the crop
  edge (§3). It cannot eliminate it; the `shift="none"` control bounds it.
- **The `GT/16 ∈ [2,8]` band is NOT applied to the primary.** The provenance
  audit measured that band to be effectively `[2,4]` (99.4 % of its pixels). The
  statistic is a difference of medians over a fixed pixel set, so the band
  narrows the sample without changing the structure. Kept as a **descriptive
  secondary** for continuity with INDEX-001/002.

---

## 5. PRIMARY STATISTIC (B3, B4) — frozen

```
m_axis(Δ) = np.median( disparity_initial[mask] )            candidate units

Odd_axis(Δ)  = [ m_axis(+Δ) − m_axis(−Δ) ] / 2              Δ ∈ {16, 32, 48}
Even_axis(Δ) = [ m_axis(+Δ) + m_axis(−Δ) ] / 2              descriptive only

S_axis = Σ_Δ Odd_axis(Δ)·Δ  /  Σ_Δ Δ²                       OLS through the origin
F_axis = S_axis / (−1/16)                                   fraction of the anchor
```

`np.median` — the same aggregate used by INDEX-001/002/ARCH-001.

**`m(0)` cancels from `Odd` by construction**, so the primary statistic needs no
baseline subtraction and is immune to any per-model offset.

**No plain OLS over the raw response is computed or reported as primary** (B3).
Stage A's own record states: *"The near-zero OLS slope is an artefact of fitting
a line through a V."* That estimator is excluded here.

**Anchor check.** Under correspondence `m(Δ) = m(0) − Δ/16`, so
`Odd(Δ) = −Δ/16` and `S_h = Σ(−Δ/16)Δ / ΣΔ² = −1/16`. ✓

**Declared provenance limitation of the decomposition.** The odd/even split
follows from the hypothesis structure — a signed prediction *is* odd, a
magnitude artifact *is* even — and Stage A had already preregistered a *signed*
slope as primary. It was nonetheless selected with knowledge that Stage A's
published negative control is even-dominated. **The negative side therefore
carries no predictive content. The positive side carries full predictive
content: no positive-checkpoint translation number exists in any record.**

**Is `|S_h| > |S_v|` a valid axis-specificity test?** It is **necessary but not
sufficient**, because image anisotropy could make horizontal perturbations larger
than vertical for non-geometric reasons. It is therefore used only in
combination with the sign condition and with the `shift="none"` control, which
experiences the identical anisotropy and bounds it (§7 C3 + C4).

---

## 6. CONDITIONS AND CONTROLS (B7, B8, B12)

| role | model | shift | seed | sha256 |
|---|---|---|---|---|
| **test** | `POS_6b_seed0` | `left` | 0 | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` |
| **test** | `POS_6b_seed1` | `left` | 1 | `58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf` |
| **test** | `POS_6b_seed2` | `left` | 2 | `245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69` |
| **artifact baseline** | `NEG_shift_none` | `none` | — | `d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1` |

All trained, all frozen, all already used in this campaign, all read-only, all
with the standardised parameter-free readout (`StandardisedDisparityRegression`,
0 parameters — verified from source).

- **Positive condition:** right-image **horizontal** translation — preserves
  correspondence and displaces it by a known amount.
- **Matched negative condition:** right-image **vertical** translation, same
  `|Δ|`, same crop mechanism, same code path (the axis is a function argument,
  not a branch), same crop size, same number of pixels displaced, same absence of
  interpolation, same preprocessing, same mask. Appearance statistics, texture,
  contrast, sharpness and binocular co-presence are preserved; only the epipolar
  relation is broken. **Not a structure-destroying shuffle. No 90° rotation.**
- **Search-free artifact control:** `shift="none"`, whose volume is `k`-constant
  by construction. **Its `S_h` and `S_v` are NOT assumed to be zero** — they are
  measured, and they are the purpose of the control (B8). Stage A's published
  data already indicate its horizontal odd component is small (≈1–9 % of anchor
  at `|Δ| = 16, 32`) and even-dominated, but that was a different sweep with
  zero-fill; it will be re-measured here under the frozen crop protocol.

---

## 7. DECISION RULE (B9) — threshold-free, unit-level

Per positive unit `u = (checkpoint, scene)`:

```
C1  sign          :  S_h(u) < 0                                source-derived direction (§1.5)
C2  monotone      :  Odd_h(16) > Odd_h(32) > Odd_h(48)         strictly, as −Δ/16 requires
C3  axis-specific :  |S_h(u)| > |S_v(u)|
C4  artifact-specific : S_h(u) < S_h(NEG, same scene)          strictly more negative
```

Global:

```
PASS  iff  C1 ∧ C2 ∧ C3 ∧ C4 at 12/12 positive units
otherwise FAIL, classified by §9
```

- **`≥9/12` is rejected.** Nine is not derived from anything, and a partial count
  reintroduces the aggregation-level freedom that INDEX-002 §10.3 and ARCH-001 §7
  deliberately closed. **12/12 or nothing**, with per-unit results fully reported
  so a partial outcome is visible and can motivate a *fresh* preregistration.
- **No numeric threshold anywhere.** `C4` compares against the artifact baseline
  **measured in-run at the same scene**, not against any published number. In
  particular the cut `β_H < −0.0321` proposed in
  `next_direction_audit_20260911/design.json` is **rejected**: it is the midpoint
  between the anchor and a *published measured* value (`−0.001713`), which would
  let a previous result set a threshold — forbidden by this project's standing
  rule.
- **No p-values.** Exchangeability is unjustified: trained and search-free models
  are not exchangeable, conditions are not exchangeable, and the 12 units share
  3 weight sets and 4 scenes. The comparison is paired and descriptive.

---

## 8. UNIT OF ANALYSIS (B11)

```
12 positive units = 3 weight sets (seeds 0,1,2) × 4 scenes
 4 control units  = 1 search-free model         × 4 scenes
```

**Three weight sets, not four checkpoints.** The units are **not independent
statistical replicates**: units sharing a checkpoint share the entire weight set;
units sharing a scene share the images and the GT mask; within a unit every
condition shares the left crop. This is a **deterministic causal diagnostic, not
a population inference**, and nothing in §7 treats the units as a random sample.

Scenes: `FOCUS_SCENES = [27, 0, 31, 6]`, split `hailo_val`.

---

## 9. FAILURE MODES (B17) — distinguished, not collapsed

| # | Outcome | Signature |
|---|---|---|
| 1 | **correspondence-positive** | C1∧C2∧C3∧C4 at 12/12 |
| 2 | **generic binocular dependence** | `S_h ≈ S_v`, both small; `Even` grows with `\|Δ\|`; C3 fails |
| 3 | **even architectural artifact** | `Odd` small and non-monotone; C2 fails; `Even` dominates |
| 4 | **search-free artifact with odd leakage** | `NEG` shows appreciable `S_h`; positives do not exceed it; C4 fails |
| 5 | **ambiguous / mixed** | C1–C4 hold at some units, not all |
| 6 | **protocol failure** | any §10 hard stop |

---

## 10. CONTROLS AND HARD STOPS

| check | criterion |
|---|---|
| identical code path for both axes | axis is a function argument; asserted |
| crop size invariant | every condition produces exactly `1136 × 272`; asserted |
| `Δ` multiples of 16 | asserted against `translation_spec.json` |
| right crop window in bounds | asserted for every `Δ` |
| mask invariance | retained counts equal the §4 table exactly, in every condition, every axis, every model |
| checkpoint hashes | the four §6 sha256, verified at load |
| readout parameter-free | `sum(p.numel() for p in model.regression.parameters()) == 0` |
| `shift="none"` degeneracy | `max|slice_k − slice_0| == 0` re-verified per scene |
| determinism | `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `no_grad`, `eval`, fp32 |
| no training | no optimizer imported or constructed; no `.backward()`, no `.step()`; checkpoints read-only |
| frozen specs unmodified | both sha256 re-verified against this document at start |

**HARD STOP** on any failure above, on any protocol modification after execution
begins, on runtime exceeding 10 minutes, or if any historical record would need
modification.

---

## 11. COMPUTE (B13)

```
4 models × 4 scenes × 13 conditions = 208 forward passes to disparity_initial
```

Each pass: two feature towers + cost volume + aggregation + standardised readout
on a `1136 × 272` crop (smaller than the native `1232 × 368`). Refinement never
invoked. No caching across conditions is possible because the right crop moves.
**Estimate well under 2 minutes.** No training, no new checkpoint, no new
architecture.

---

## 12. CLAIM CEILING (B10, B16)

**On PASS**, the maximum claim is exactly:

> "The trained model exhibits a horizontal, sign-consistent odd response to
> controlled right-image translation that is stronger and more axis-specific than
> the matched vertical and search-free artifact controls."

Status: **`GEOMETRIC-CORRESPONDENCE-EVIDENCE`**.

It does **not** prove correct disparity magnitude, correct final disparity,
genuine full disparity search, stereo accuracy, or generalisation.

**Magnitude alone proves nothing** (B10). Even if `S_h ≈ −1/16`, the evidence is
**sign + odd structure + horizontal axis specificity** relative to the vertical
and search-free controls. `F_h` is reported descriptively and is **not** a pass
criterion, because no tolerance on it can be justified a priori.

**On FAIL**: `GEOMETRIC-CORRESPONDENCE-NOT-DEMONSTRATED`, classified by §9. It
must **not** be concluded that correspondence is absent.

---

## 13. PREREGISTRATION ORDER (B14) — as executed

1. translation values frozen → `translation_spec.json` ✔
2. mask frozen → `mask_spec.json` ✔
3. geometry derivation frozen → §1 ✔
4. statistic frozen → §5 ✔
5. decision rule frozen → §7 ✔
6. controls frozen → §6, §10 ✔
7. `PREREGISTRATION.md` written → this document ✔
8. **execute** → NOT DONE
9. `RESULTS.md` → NOT CREATED

**No result may be inspected between 7 and 8. No protocol modification after
execution begins.**

---

## 14. OUTPUTS

Present now (design artifacts only): `PREREGISTRATION.md`,
`translation_spec.json`, `mask_spec.json`, `frozen.sha256`,
`freeze_geom_specs.py`, `RELATED_RUNS.md`, `PROVENANCE_ANOMALY.md`.

To be created **at execution only**: `RESULTS.md`, `results.json`,
`ENVIRONMENT.txt`, `run.log`, and the harness with its sha256.

**`RESULTS.md` does not exist and must not be created before execution.**
