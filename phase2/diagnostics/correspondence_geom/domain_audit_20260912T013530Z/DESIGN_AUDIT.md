# DESIGN AUDIT — corrected validity domain for the geometric correspondence diagnostic

Record `phase2/diagnostics/correspondence_geom/domain_audit_20260912T013530Z/`.
2026-09-12 UTC.

**DESIGN AND VALIDATION ONLY. THE EXPERIMENT WAS NOT RUN.**

No aggregation forward pass on real data (trained or random). No standardisation,
softmax, soft-argmin or readout on real data. No interaction statistic, no
`argmin`, no `h`, no `h_rand`, no `h_trained`. No gate, no `G-PASS`, no vertical
arm, no search-free arm. No `gate.json`, no `trained.json`, no `results.json`.
No training, no optimizer, no `.backward()`, no `.step()`. No historical record
modified — **41 prior records re-verified byte-identical**.

**Two things were executed, both mandated by the brief:**

| what | brief | what it touches |
|---|---|---|
| `validate_domain.py` | §4 — *"extract features … verify `Rf[w]=Lf[w+τ]` … verify `V[τ,w]=0` … record exact counts"* | the **feature extractor and the cost volume only**, both parameter-frozen wiring stages. It sets `model.aggregation = None` and `model.refinement = None` at load so neither can be invoked, and it stops before aggregation as §4 directs |
| `synthetic_mask_check.py` | §10 — *"update the synthetic mechanism check"* | **synthetic fields and random aggregation weights only**. No checkpoint, no dataset. It is **not** the experiment's random gate: it computes no `h_rand` over the preregistered units and makes no gate decision |

**Reading of §12 vs §4, stated explicitly.** §12 forbids running the experiment
arms; §4 requires running the extractor and the cost volume. These are
reconcilable in exactly one way, and that is the way taken: wiring stages
executed, experimental stages not. If the intent was that *no* model code run at
all, §4 could not have been carried out and this audit would have had to report
`REDESIGN REQUIRED` for want of measurement.

Tags: **SOURCE** · **MEASURED** · **DERIVED** · **FROZEN** · **UNKNOWN**.

---

## 1. ROOT-CAUSE CONFIRMED

**Confirmed, and the confirmation is stronger than the halt record's.**

GEOM-001's HS1 asserted `V(τ, w) = 0` over `w < 54` starting at `w = 0`. The
identity `V(τ, ·) = 0` is correct **only where the extractor is exactly
translation-equivariant**, which excludes a **15-feature-cell (240 px)** halo at
every crop edge whose source column differs between the two crops. `w ∈ [0, 14]`
is inside that halo.

**The same omission was in the mask**, and there it was worse: the GEOM-001 mask
`x ∈ [64, 864)` = feature columns `[4, 54)` includes contaminated columns
`4…14`. In that band the POSITIVE cells do not correspond, so the **treatment**
was corrupted, not a nuisance term.

**New, and decisive — the contamination was not hypothetical.** Injecting two
*different* arbitrary halos into otherwise identical fields
(`synthetic_mask_check.json`, S6):

| mask | statistic bit-identical under two arbitrary halos | max \|difference\| | `argmin` changed by halo |
|---|---|---|---|
| **GEOM-001** `x ∈ [64, 864)` | **0 / 24** | `1.157e-01` | **8 / 24** |
| **corrected** `x ∈ [325, 633)` | **24 / 24** | **`0.000e+00`** | **0 / 24** |

> Under the GEOM-001 mask, **arbitrary boundary garbage flipped the primary
> statistic's `argmin` in a third of cells.** Had HS1 not fired, the experiment
> would have produced a number driven in part by values that carry no image
> correspondence at all.

**Classification:** false premise in a frozen bound. Not a harness bug, not a
property of the trained model. The design record derived the 477 px receptive
field correctly (`source_trace.md` §3) and declared the contamination (§10), then
argued it away as a main effect cancelled by the 2×2. **That argument is true of
the cancellation and false of the anchor** — see §8 below. This is the same error
class as GEOM-002's C3, one level further out.

---

## 2. EXACT VALID FEATURE DOMAIN

Full derivation in `domain_derivation.md` §2–§3.

```
boundary-free cells of any crop      :  w ∈ [15, 55]      41 cols   (adopted, symmetric)
                                        w ∈ [15, 56]      42 cols   (measured, tight)
translation identity Rf[w]=Lf[w+τ]   :  w ∈ [15, 55−τ]              (adopted)
                                        w ∈ [15, 56−τ]              (measured)
```

**MEASURED, 216 records** (3 extractors × 6 pairs × 2 scenes × 6 τ):

| | |
|---|---|
| adopted band exact in every record | **TRUE** — 216/216 |
| observed valid-run **lo** | `[15]` — every record, without exception |
| observed valid-run **hi** | `{50, 51, 52, 53, 54, 55}` `= 56 − τ` |
| equality fails at `w = 14` | **TRUE** — 216/216 |

The left halo is **15** cells; the right halo is **14**. The one-cell asymmetry
comes from stride/padding parity and is **not relied upon**: the adopted
symmetric bound is a strict subset of the measured one.

---

## 3. EXACT VALID COST-VOLUME DOMAIN

`V[k,w] = Lf[w+k] − Rf[w]`; `argmin_k` ranges over **all twelve** candidates, so
every slice must be admissible:

```
(a) fill-free ∀k ≤ 11 :  w ≤ FW − 1 − 11 = 59
(b) Lf[w+k] clean ∀k  :  w ≥ 15  ∧  w + 11 ≤ 55  ⟹  w ≤ 44
(c) Rf[w] clean       :  w ∈ [15, 55]
────────────────────────────────────────────────────────────
       VALID COST-VOLUME DOMAIN :  w ∈ [15, 44]   30 columns
```

**The binding constraint is `k = 11`, not `τ`.** `w ≤ 44` is stricter than
`w ≤ 55 − τ` for every `τ ≤ 6`, so **the domain is τ-independent** — which the
across-`τ` comparison requires.

**MEASURED:** `V[τ, :, :, 15:45] == 0.0` **exactly** in all **216** checks,
`16 320` cells each = **3 525 120 cells, every one exactly zero**, across three
extractors, twelve scenes and six `τ`.

---

## 4. EXACT VALID AGGREGATION / STATISTIC DOMAIN

`5 × Conv3d(3×3×3, padding=1)` ⇒ spatial radius **5** cells. Output at `w` reads
`[w−5, w+5]`:

```
[w−5, w+5] ⊆ [15, 44]   ⟹   w ∈ [20, 39]      20 feature columns, τ-independent
```

Bilinear upsample, `align_corners=True`, `70x/1135`:

```
x ≥ ⌈1135·20/70⌉ = 325 ;   70x < 1135·39 ⟹ x ≤ 632

       PIXEL BAND :  x ∈ [325, 632]     308 columns
```

**The aggregation's own zero padding needs no exclusion**, and this is not a
loophole: padding is part of the **linear** operator, applied identically to all
four cells, and therefore lives in the channel the interaction cancels exactly.
**Confirmed** (S4): an affine aggregation gives an aggregated-cost interaction of
`≤ 2.3e-06` relative — round-off — against `0.71 … 0.86` nonlinear (S5). Five
orders of magnitude.

Invalid **input** is a different matter, and is exactly what the 5-cell shrink
excludes.

---

## 5. MASK DERIVATION

```
CORRECTED MASK :   y ∈ [64, 208)   ∧   x ∈ [325, 633)
                   144 × 308 = 44 352 px per scene, per τ, per cell
```

Identical in all four cells, at every `τ`, on both arms, for every model —
`τ`-independent, GT-independent, response-independent.

| bound | value | provenance |
|---|---|---|
| `x ≥ 325` | ← `w ≥ 20` ← `w ≥ 15 + 5` ← extractor halo + aggregation radius | **DERIVED**, §2–§4 |
| `x < 633` | ← `w ≤ 39` ← `w ≤ 55 − 11 − 5` ← halo + `k`-reach + aggregation radius | **DERIVED**, §2–§4 |
| `y ∈ [64, 208)` | **inherited unchanged** from `ARCH-RATE/classification_spec.json` | **FROZEN** — the derivation independently shows *all 17 rows* are admissible on the horizontal arm, so this border is strictly conservative. It is inherited, not derived, and that is stated rather than hidden |

**The brief's §5 warning is confirmed by derivation.** `[15, 55−τ)` would have
been wrong twice over: *too wide* (it ignores the `k ≤ 11` reach and the
aggregation's 5-cell radius) and *τ-dependent* (a support that shrinks with `τ`
confounds the across-`τ` comparison the statistic is built on).

The corrected band is a **strict subset** of the GEOM-001 band: of its 800 pixel
columns, **492 (61.5 %) were inadmissible.**

---

## 6. BOUNDARY-CONTAMINATION ANALYSIS

Three channels, each traced and closed:

| channel | reach | closed by |
|---|---|---|
| extractor boundary halo | 15 cells from each crop edge | `w ≥ 15`, `w ≤ 55` |
| `shift_left` zero fill | last `k` columns, `k ≤ 11` | subsumed by `w ≤ 44` (fill needs only `w ≤ 59`) |
| candidate reach `Lf[w+k]` | `+11` cells | `w ≤ 55 − 11 = 44` |
| aggregation spatial radius | 5 cells each side | `w ∈ [20, 39]` |
| bilinear upsample | 1 cell each side | folded into the pixel-band inequality |

**Empirical closure (S6) — the operational form of HS1-C.** Two different
arbitrary halos, same fields, same weights:

```
corrected mask   :  24/24 bit-identical,  max |diff| = 0.000e+00
GEOM-001 mask    :   0/24 bit-identical,  max |diff| = 1.157e-01
                     argmin flipped in 8/24 cells
```

This is stronger than a derivation: it shows the statistic on the corrected mask
is **provably invariant to whatever the halo contains.**

### 6.1 The vertical control has no admissible region — and cannot be given one

**DERIVED.** On the vertical arm the crops share columns and differ in rows, so
the roles swap and the rows must be halo-free:

```
clean rows = [15, FH−1−15] = [15, 1] = ∅        (FH = 17)
```

Empty. Nor can a taller crop rescue it: the source frame is 368 px = 23 feature
rows, and `[15, 7] = ∅` too. A non-empty band needs `FH ≥ 31` cells `= 496` px,
**more than the KITTI frame height.** The asymmetry is structural — the crop is
71 cells wide but only 17 tall, against a 15-cell halo.

**Handling (see §12):** the vertical arm is **retained and reclassified as a
contaminated null**, usable only as a downgrade trigger. It is not removed.

---

## 7. HARD STOPS

The old HS1 is **withdrawn** — it was over-broad. Replaced by a hierarchy, every
condition structural or exact, none a post-hoc tolerance.

| id | condition | status of its premise |
|---|---|---|
| **HS1-A** feature translation validity | for every positive pair, scene, `τ`, extractor: `Rf[w] == Lf[w+τ]` **bit-exactly** for all `w ∈ [15, 44]` | **VALIDATED** 216/216 |
| **HS1-B** matched cost-volume anchor | `V[τ, c, y, w] == 0.0` **EXACTLY** for all `w ∈ [15, 44]`, all rows, all channels. **Preregistered tolerance = EXACTLY ZERO** | **VALIDATED** — 3 525 120 cells, `max |V| = 0.0`. Exact equality is *achievable*, so the strictest possible tolerance is the correct one, and it is fixed before the experiment rather than chosen after seeing errors |
| **HS1-C** boundary contamination | integer assertions, checked in code before any measurement: `CV_LO ≥ HALO`; `CV_HI ≤ FW−1−HALO−(D−1)`; `ST_LO ≥ CV_LO+AGG_R`; `ST_HI ≤ CV_HI−AGG_R`; `PX_LO·(FW−1) ≥ ST_LO·(CROP_W−1)`; `(PX_HI+1)·(FW−1) > ST_HI·(CROP_W−1) > PX_HI·(FW−1)` | **VALIDATED** empirically by S6 (24/24 bit-identical) |
| **HS1-D** scene coverage | **structural, defined before execution**: halt if the admissible column band (i) is empty at any `τ`, (ii) **differs across `τ`**, or (iii) is narrower than the aggregation's spatial **diameter** (`2·AGG_R+1 = 11` cells). No magnitude threshold | band `= 20` columns at every `τ`; `20 ≥ 11` ✓ |
| **HS2** | exact `AA/BB/AB/BA` construction; each frame exactly once as a left and once as a right | unchanged |
| **HS3** | `τ` identical across the four cells of a level | unchanged |
| **HS4** | 2×2 cancellation `max|resid| / max|V| ≤ 1e-5` (float32 round-off bound, 10× margin) | **now MEASURED on real features: `1.81e-07` max, `1.19e-07` median** |
| **HS5** | checkpoint sha256; `num_disparities == 12`; `cost_volume_shift` as specified; readout parameter count `== 0` | unchanged |
| **HS6** | W4 process-restart repeat bit-identical | unchanged |
| **HS7** | no trained aggregation loaded before `G-PASS`; two-process enforcement | unchanged — it held in GEOM-001 |
| **HS8** | no historical record modified | unchanged — held in GEOM-001; **41 records re-verified here** |
| **HS9** | mask retained-pixel count `== 44 352` | **new value** |
| **HS10** | any non-finite response | unchanged |
| **HS11** | no optimizer, no `.backward()`, no `.step()`, refinement never invoked | unchanged |

**Policy unchanged: any hard stop RAISES and the process exits. Nothing is
repaired and continued.**

---

## 8. INTERACTION VALIDITY

**The interaction remains valid. The re-derivation changes nothing about the
cancellation and everything about where the anchor lives.**

### 8.1 Cancellation — unconditional, and now measured on real features

```
V_AA + V_BB − V_AB − V_BA ≡ 0     pointwise, ∀ k, w, τ
```

It is **content-independent** (so different spatial content in `AA/BB` vs
`AB/BA` is irrelevant), and it holds **inside the halo and inside the zero-fill
region too** — in the fill region every term reduces to `−Rf` and cancels.

| | max relative residual |
|---|---|
| **real features** (this record, 108 checks, 3 extractors × 6 pairs × 6 τ) | **`1.8089e-07`** (median `1.1943e-07`) |
| synthetic, entire tensor incl. injected halo and fill (S3) | `1.391e-07` |

Both are float32 round-off. **This is the first measurement of the identity on
real extractor outputs.**

### 8.2 The anchor — this is what needed the mask

`H1`'s prediction is *"the minimum sits at `k = τ`"*, and it exists **only where
`V(τ) = 0`**. Outside the clean domain there is no anchor, so there is nothing
for the interaction to reveal and nothing for the cancellation to protect.

> **The GEOM-001 design conflated the two.** "Main effects cancel" is *true* and
> was used to justify ignoring the halo. But the halo does not introduce a main
> effect — it **destroys the treatment** in the positive cells. Cancellation
> protects against `H2`/`H3`; it cannot manufacture an anchor that is not there.

### 8.3 Does the aggregation's spatial receptive field break cancellation?

**No.** Under an affine aggregation the identically-zero cost interaction maps to
an identically-zero aggregated interaction, receptive field and padding included
(S4: `≤ 2.3e-06` relative). Under the nonlinear aggregation nothing cancels
exactly — **that residual is the signal by design** (S5: `0.71 … 0.86`). The
receptive field is handled by the 5-cell shrink, which guarantees every
aggregated value used is supported entirely by admissible `V`.

### 8.4 Is cancellation only approximate?

Only to float32 round-off (`≈ 1.8e-07` relative), which HS4 bounds at `1e-5` from
machine precision with a 10× margin. **Nothing further needs controlling.**

---

## 9. RANDOM-GATE VALIDITY

**The gate remains necessary, remains valid, and must run first — unchanged in
form. Its numbers must be regenerated.**

| question | answer |
|---|---|
| does the corrected mask change what the gate asks? | **No.** *"Can the architecture alone produce argmin-at-`τ` tracking?"* is unchanged |
| does it change the gate's validity? | **It improves it.** Every aggregated location the gate scores is now supported entirely by admissible `V`, so a random-weight hit can no longer be manufactured by halo garbage — which S6 shows was possible under the old mask (`argmin` flipped in 8/24 cells from arbitrary boundary values) |
| can the old gate numbers be reused? | **No.** The design-audit synthetic figures (`5/48`, `h ≤ 3`) were computed on the old mask and on synthetic fields. They are void. `H*` must be generated afresh in Stage 1 |
| is a new statistic needed? | **No — and one must not be invented.** The statistic is unchanged: `argmin_k Ī_τ(k) == τ`, `h = Σ_τ hit_τ` |
| gate rule | unchanged: `G-STOP` if `max(h_rand) == 6` ⇒ terminal `STATISTIC-ARCHITECTURALLY-AVAILABLE`, trained weights never loaded; else `H* := max(h_rand)`, unaltered |

---

## 10. SYNTHETIC VALIDATION

`synthetic_mask_check.py` — synthetic fields, random aggregation weights, **no
checkpoint, no dataset, no gate decision**. All six items of §10:

| id | test | result |
|---|---|---|
| **S1** | matched anchor with a halo injected | zero band `= [15, 55−τ]` at every `τ`, **exactly as predicted**; anchor exact on `[15, 44]` ✓ |
| **S2** | cross-pair construction | `AB`/`BA` built from the identical crops, each field once as a left and once as a right ✓ |
| **S3** | cost-volume cancellation over the **entire** tensor (halo and fill included) | max relative `1.391e-07` — round-off ✓ |
| **S4** | affine aggregation | aggregated-cost interaction `≤ 2.308e-06` relative — round-off ✓ |
| **S5** | nonlinear aggregation | `0.711 … 0.859` relative — the signal channel is the nonlinearity ✓ |
| **S6** | **boundary-contamination exclusion** | corrected mask **24/24 bit-identical**, `max |diff| = 0.000e+00`; GEOM-001 mask **0/24**, `argmin` flipped **8/24** ✓ |

**No new discriminator was introduced.** S6 is an *invariance* test of the mask,
not a statistic about the model.

---

## 11. COVERAGE BY τ AND SCENE

**The admissible domain is τ-independent** — the `k = 11` reach binds, not `τ`.
The table is therefore identical for `τ = 1 … 6`:

| quantity | value | of 71 cols |
|---|---|---|
| total feature columns | 71 | 100 % |
| extractor-valid (`[15,55]`) | 41 | 57.7 % |
| translation-valid (`[15,55−τ]`) | `41 − τ` = 40 … 35 | 56.3 … 49.3 % |
| **cost-volume-valid (`[15,44]`)** | **30** | **42.3 %** |
| **aggregation-valid / final (`[20,39]`)** | **20** | **28.2 %** |
| final pixel columns (`[325,633)`) | 308 of 1136 | 27.1 % |
| final rows (`[64,208)`) | 144 of 272 | 52.9 % |
| **admissible pixels per scene per τ** | **44 352** of 308 992 | **14.4 %** |

**Per scene:** identical for all 12 preregistered scenes — the mask is purely
geometric and depends on no image content. Validation covered all 12 scenes × 6 τ
× 3 extractors and every one passed.

**Independent support, stated honestly.** The 44 352 pixels are a bilinear
upsampling of **20 feature columns × 11 feature rows ≈ 220 feature cells**. The
pixel count overstates independence by roughly `16²`. The real support for each
of the twelve per-candidate medians is ~220 cells.

**Against GEOM-001:** the old mask nominally held 115 200 px, of which **only
44 352 (38.5 %) were admissible** — in feature-cell terms, ~550 cells of which
~220 were valid.

**HS1-D verdict:** band non-empty ✓, τ-independent ✓, `20 ≥ 11` (aggregation
diameter) ✓. **Support is sufficient by the preregistered structural criterion.**

---

## 12. CHANGES FROM GEOM-001

| # | change | classification | justification |
|---|---|---|---|
| 1 | mask `x ∈ [64,864)` → `x ∈ [325,633)`; retained pixels `115 200 → 44 352` | **VALIDITY CORRECTION** | 61.5 % of the old columns were inadmissible; S6 shows the statistic there is not halo-invariant |
| 2 | HS1 → HS1-A / HS1-B / HS1-C / HS1-D | **VALIDITY CORRECTION** | the old bound was over-broad and encoded a false premise |
| 3 | HS1-B tolerance fixed at **exactly zero** | **WIRING CORRECTION** | exact equality is achievable (3 525 120 cells, `max = 0.0`), so the strictest tolerance is correct and is fixed before the experiment |
| 4 | HS9 pixel count `115 200 → 44 352` | **WIRING CORRECTION** | follows mechanically from 1 |
| 5 | vertical arm: "control with a predicted null" → **"contaminated null, downgrade-trigger only"** | **VALIDITY CORRECTION** | no halo-clean region exists and none can exist (§6.1). Strictly conservative: a false positive there only *downgrades* to CASE C; a negative there is declared uninformative. **The arm is retained, not removed** |
| 6 | random-gate `H*` regenerated rather than inherited | **VALIDITY CORRECTION** | the old synthetic figures were computed on an invalid mask |

**Unchanged, and deliberately so:** the `AA/BB` positive pair · the `AB/BA`
crossed negative · matched image marginals · identical translation magnitude ·
identical crops · the same feature extractor · the same trained aggregation
weights · the parameter-free z-score/readout · the interaction statistic
`I_τ(k)` and `argmin_k == τ` · the random-aggregation gate before trained
Stage 2 · seeds `0…31` · the six `τ` · the six scene pairs · the three
checkpoints · the search-free control · the two-process HS7 enforcement.

```
SCIENTIFIC-DESIGN CHANGES PROPOSED :  NONE.
```

Every change above is a wiring or validity correction. Item 5 is the one that
comes closest to a design change, and it is deliberately framed so that it can
only make the verdict **more** conservative, never more permissive. **Removing**
the vertical arm *would* be a scientific-design change and is **not** proposed.

---

## 13. CLAIM CEILING

**Maximum admissible claim as of this audit — unchanged from the brief's §15,
plus the four facts this audit measured:**

- the pixel construction is exact — `right[:, :−t] == left[:, t:]`, every scene,
  every `τ`;
- the feature translation law is bit-exact on the measured interior, `216/216`
  records, three extractors;
- the extractor halo is **15 feature cells** on the left and **14** on the right;
  the adopted bound is the conservative 15 on both sides;
- **the GEOM-001 mask was invalid**, and `61.5 %` of its columns were
  inadmissible;
- **GEOM-001 produced no admissible geometric result**;
- *(new)* the 2×2 cancellation identity holds on **real** extractor features to
  float32 round-off (`1.81e-07`);
- *(new)* `V(τ) = 0` **exactly** on `w ∈ [15,44]` — `3 525 120` cells, all zero;
- *(new)* the corrected mask makes the statistic **provably invariant** to
  arbitrary halo content (`24/24` bit-identical, `max |diff| = 0.0`), while the
  GEOM-001 mask does not (`0/24`; `argmin` flipped `8/24`).

**NOT claimed, and not claimable from this audit:** correspondence · geometric
matching · correct disparity search · candidate-coordinate correspondence ·
trained geometric sensitivity · any statement whatever about the trained
aggregation's behaviour.

**Status if the corrected experiment eventually runs and passes: CONSISTENCY
EVIDENCE AT BEST** — three checkpoints are irreducibly reused and the designer
has read eight prior response curves. Disclosed, not mitigated.

---

## 14. EXECUTABILITY VERDICT

```
EXECUTABLE
```

for the **horizontal arm**, on the corrected mask, with the HS1-A…D hierarchy.

Every gate in §11 clears on structural, preregistered criteria: the admissible
band is non-empty at every `τ` (20 columns), is **τ-independent**, exceeds the
aggregation's spatial diameter (`20 ≥ 11`), is identical across all four cells
and all three checkpoints, and was validated exactly — `216/216` translation
records, `3 525 120` anchor cells all exactly zero, `24/24` halo-invariance.

**With one qualification that must travel with the verdict:**

```
THE VERTICAL CONTROL IS NOT EXECUTABLE AS A CLEAN NULL.
```

No halo-free region exists for it, and none can be constructed from KITTI frames
(§6.1). It is retained as a **contaminated null / downgrade trigger only**; a
negative result from it is **uninformative** and must be reported as such rather
than as a passed control.

**Honest accounting of what this costs the design.** The vertical arm was the
second geometry-breaking negative, the one that preserved content *exactly*. Its
demotion leaves the cross-pairing negative (`AB`/`BA`) carrying the design alone
— which is the negative the whole intervention was built on, and which is
unaffected. The loss is a redundancy, not a load-bearing control.

---

## 15. NEXT-ACTION AUTHORIZATION

**Not authorized here, and not initiated.** This audit is design and validation
only; the corrected experiment is **not** preregistered, frozen or launched in
this record.

What a successor authorisation would need to carry, all already derived:

| element | value |
|---|---|
| mask | `y ∈ [64,208) ∧ x ∈ [325,633)`, `44 352` px |
| hard stops | HS1-A, HS1-B (tolerance **exactly zero**), HS1-C, HS1-D, HS2–HS11 |
| statistic | **unchanged** — `argmin_k Ī_τ(k) == τ`, `h = Σ_τ hit_τ` |
| gate | **unchanged** — Stage 1 random aggregation, seeds `0…31`, separate process, `G-STOP` at `max h_rand == 6`; `H*` **regenerated, never inherited** |
| arms | horizontal (primary) · vertical (**contaminated null, downgrade trigger only**) · search-free `NEG_shift_none` (separate, mask `[20,39]` is valid for it too, being a subset of its own `[20,50]`) |
| scenes, `τ`, checkpoints, seeds | **unchanged** |
| compute | unchanged in structure; ~15 000 readouts, 5–15 min |

**Open questions carried forward, unresolved and not guessed:**

1. whether ~220 feature cells of support yield a stable `argmin` — **UNKNOWN
   until Stage 1 runs**; it is precisely what the gate measures;
2. why the right halo is 14 cells and the left 15 — **UNKNOWN**, not
   load-bearing, the conservative bound is adopted;
3. whether a learned non-correspondence shortcut could place its minimum at
   `k = τ` — **UNKNOWN**, not addressable by any ablation of this architecture;
4. whether the `y ∈ [64,208)` row border should be widened to the full
   derived-valid `[0,272)` — **deferred**; it would roughly double the support
   and the derivation permits it, but changing it after seeing that support is
   thin would be support-shopping and is therefore **not** proposed here.

```
STOPPING AT THE AUDIT.  THE EXPERIMENT IS NOT EXECUTED.
```

---

## 16. FILES IN THIS RECORD

| file | role |
|---|---|
| `DESIGN_AUDIT.md` | this document — the §14 report |
| `domain_derivation.md` | the full coordinate chain and the quantity table |
| `design.json` | machine-readable corrected domain, mask and hard stops |
| `RELATED_RUNS.md` | provenance and immutability verification |
| `validate_domain.py` / `.json` / `.log` | §4 validation — **extractor + cost volume only** |
| `synthetic_mask_check.py` / `.json` / `.log` | §10 synthetic check — **no trained model, no real data, not the gate** |

**No `gate.json`. No `trained.json`. No `results.json`. No preregistration. No
successor launched.**
