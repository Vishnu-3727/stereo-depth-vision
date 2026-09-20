# DOMAIN DERIVATION — the complete coordinate chain

Record `phase2/diagnostics/correspondence_geom/domain_audit_20260912T013530Z/`.
2026-09-12 UTC. Companion to `DESIGN_AUDIT.md`.

**Design only.** Every boundary below is justified from source or from the
measurement recorded in `validate_domain.json`. Nothing is assumed.

Tags: **SOURCE** (read from the implementation) · **MEASURED** (this record or
the GEOM-001 halt record) · **DERIVED** (algebra on the two) · **FROZEN**
(inherited preregistration) · **UNKNOWN**.

---

## 1. THE QUANTITY TABLE (required by §3 of the brief)

| quantity | exact value | source | status |
|---|---|---|---|
| source frame | `1232 × 368` px | `kitti2015.pad_and_crop` | **SOURCE** |
| image crop geometry | left `rows [0,272) cols [0,1136)`; right-H `cols [16τ, 1136+16τ)`; right-V `rows [16τ, 272+16τ)` | frozen construction, `ARCH-RATE/construction_spec.json` | **FROZEN** |
| extractor stride | `16` (`2**4`, four `Conv2d(k=5, s=2, p=2)`) | `feature_extractor.py` | **SOURCE** |
| feature grid | `FW = 1136/16 = 71`, `FH = 272/16 = 17` | derived from the crop | **SOURCE/DERIVED** |
| extractor receptive field | `477` px ⇒ one-sided radius `(477−1)/2 = 238` px `= 14.875` cells | back-propagated through `4×(r←2r+3)` after `27` | **DERIVED** |
| **extractor halo (adopted)** | **`15` feature cells (240 px), both sides** | GEOM-001 halt measurement; re-measured here | **MEASURED** |
| extractor halo (tight, right side) | `14` cells — the right boundary is one cell wider than the symmetric bound | `validate_domain.json`, 216/216 records | **MEASURED** |
| candidate count `D` | `12` | `StereoNetConfig` | **SOURCE** |
| shift convention | `shift_left(x,k)[w] = x[w+k]`, zero-fill for `w ≥ FW−k` | `cost_volume.py` | **SOURCE** |
| aggregation spatial kernel | `5 × Conv3d(3×3×3, padding=1)` ⇒ radius **5** cells in `(k, y, w)` | `aggregation.py` | **SOURCE** |
| readout | upsample `(17,71)→(272,1136)` bilinear `align_corners=True`, then per-pixel `z`, then `softmax(−z)` | `regression.py`, `scaled_regression.py` | **SOURCE** |
| `τ` values | `{1,2,3,4,5,6}` (`t = 16τ`) | preregistration | **FROZEN** |
| **valid feature domain** | `w ∈ [15, 55]` (either crop, standalone) | §2 | **DERIVED + MEASURED** |
| **valid translation domain** | `w ∈ [15, 55−τ]` adopted; `[15, 56−τ]` measured | §3 | **DERIVED + MEASURED** |
| **valid cost-volume domain** | `w ∈ [15, 44]` — **τ-independent** | §4 | **DERIVED + MEASURED** |
| aggregation spatial halo | `5` cells each side | `aggregation.py` | **SOURCE** |
| **valid statistic domain (feature)** | `w ∈ [20, 39]` — 20 columns, **τ-independent** | §5 | **DERIVED** |
| **valid statistic domain (pixel)** | `x ∈ [325, 632]` — 308 columns | §6 | **DERIVED + MEASURED** |
| rows | all `17` feature rows admissible (horizontal arm); `y ∈ [64,208)` retained as inherited conservatism | §7 | **DERIVED** |

---

## 2. STAGE 1 — pixel → feature, and the halo

**SOURCE.** The extractor is fully convolutional: four `Conv2d(k=5, s=2, p=2)`
with no activations, six `ResBlock` (`k=3, p=1`), one `Conv2d(k=3, s=1, p=1)`.

**DERIVED.** For `Conv2d(k=5, s=2, p=2)`, `out[i] = Σ_j w[j]·x[2i+j−2]`, so
translating the input by 2 px translates the output by 1 cell. Over four levels,
**`t` px ⇒ `τ = t/16` cells, exactly, in the interior.**

**DERIVED.** Receptive field, back-propagated: final `k=3` ⇒ 3; six ResBlocks ×
two `k=3` convs ⇒ `3+24 = 27`; then four `k=5, s=2` convs, each `r ← 2r+3`:
`27 → 57 → 117 → 237 → 477`. One-sided radius `238` px `= 14.875` cells.

**MEASURED (this record, 216 records = 3 extractors × 6 pairs × 2 scenes × 6 τ).**
A feature cell is boundary-free iff it is at least **15 cells** from the left
edge and at least **14 cells** from the right edge:

```
observed valid-run lo values : [15]                  (every single record)
observed valid-run hi values : [50, 51, 52, 53, 54, 55]   = 56 − τ
```

The left halo is exactly 15 cells; the right halo is exactly 14. The asymmetry
is one cell and comes from stride/padding parity at the two boundaries. **It is
not relied upon.**

```
ADOPTED (symmetric, conservative):   clean(crop) = w ∈ [15, 55]
MEASURED (tight):                    clean(crop) = w ∈ [15, 56]
```

**The adopted band is a strict subset of the measured band.** Validation
confirms it holds exactly in 216/216 records.

---

## 3. STAGE 2 — the translation identity

The positive-pair identity `Rf[w] = Lf[w+τ]` requires **both** operands to be
boundary-free **in their own crop's frame**:

```
Rf[w]    clean  ⟺  w   ∈ [15, 55]
Lf[w+τ]  clean  ⟺  w+τ ∈ [15, 55]  ⟺  w ∈ [15−τ, 55−τ]
──────────────────────────────────────────────────────────
                    w ∈ [15, 55−τ]          (since 15−τ < 15)
```

**MEASURED.** Exact bit-equality holds on `[15, 55−τ]` in **216/216** records
(`derived_band_exact_everywhere: true`), and fails at `w = 14` in **216/216**
(`fails_immediately_left_of_band: true`). The tight band extends one cell
further right (`[15, 56−τ]`); the adopted band is the conservative one.

**This band is τ-dependent — and it is NOT the mask.** Using it as the mask
would make the support shrink with `τ`, confounding the across-`τ` comparison
that the statistic depends on. §4 shows the binding constraint is stronger and
`τ`-independent.

---

## 4. STAGE 3 — the cost volume

**SOURCE.** `V[k, w] = Lf[w+k] − Rf[w]` for `w < FW−k`, and `= −Rf[w]` (zero
fill) for `w ≥ FW−k`.

The statistic evaluates `Ī_τ(k)` over **the whole candidate axis** `k ∈ [0, 11]`
— `argmin_k` ranges over all twelve. **Every candidate slice must therefore be
admissible at every location used.** Three constraints:

```
(a) fill-free, all k ≤ 11 :  w < FW − k  ∀k ≤ 11   ⟹  w ≤ 71 − 12 = 59
(b) Lf[w+k] clean, all k  :  w+k ∈ [15,55] ∀k∈[0,11] ⟹ w ≥ 15  and  w ≤ 44
(c) Rf[w] clean           :  w ∈ [15, 55]
──────────────────────────────────────────────────────────────────────────
        VALID COST-VOLUME DOMAIN :   w ∈ [15, 44]     30 columns
```

**Constraint (b) at `k = 11` binds, not `τ`.** `w ≤ 55 − 11 = 44` is stricter
than `w ≤ 55 − τ` for every `τ ≤ 6`, so the domain is **independent of `τ`** —
exactly the property the statistic needs.

**MEASURED.** `V[τ, :, :, 15:45] == 0.0` **exactly**, in all **216** checks,
`16 320` cells each (`32 channels × 17 rows × 30 columns`) — **3 525 120 cells,
every one exactly zero**, across three extractors, twelve scenes and six `τ`.

> **This is why `[15, 55−τ)` must not be used as the mask.** It is simultaneously
> *too wide* (it ignores the `k ≤ 11` reach and the aggregation halo) and
> *τ-dependent* (it confounds the across-`τ` comparison). The brief's warning in
> §5 is confirmed by derivation.

---

## 5. STAGE 4 — the aggregation

**SOURCE.** `4 × [Conv3d(32→32, 3×3×3, p=1) + LeakyReLU] + Conv3d(32→1, 3×3×3, p=1)`
— five convolutions, so spatial radius **5 cells** along `w` (and along `y`, and
along `k`).

An aggregated value at column `w` reads input columns `[w−5, w+5]`. For it to be
free of inadmissible input:

```
[w−5, w+5] ⊆ [15, 44]   ⟹   VALID STATISTIC DOMAIN :  w ∈ [20, 39]
                                                       20 columns
```

### 5.1 What about the aggregation's own zero padding?

**DERIVED — it needs no exclusion, and this is not a loophole.** The padding at
`k = 0, 11`, at `y = 0, 16` and at the `w` ends is part of the **linear**
operator, applied identically to all four cells of the 2×2. Because the
cost-volume interaction is identically zero (§8), an **affine** aggregation
maps it to an identically-zero aggregated interaction *including all padding
effects*. Padding therefore lives entirely in the channel the interaction
cancels.

**Confirmed synthetically** (`synthetic_mask_check.json`, S4): with the
LeakyReLU slope set to 1.0 the aggregated-cost interaction is `≤ 2.3e-06`
relative — round-off — while the nonlinear arm gives `0.71 … 0.86` (S5). Five
orders of magnitude.

**Invalid *input* is a different matter entirely**, and is what §5's bound
excludes: where `V` is halo-corrupted the positive cells do not satisfy
`V(τ) = 0`, so the treatment itself is destroyed, not a nuisance term.

---

## 6. STAGE 5 — upsample → pixel coordinates

**SOURCE.** `F.interpolate(..., size=(272,1136), mode="bilinear", align_corners=True)`.

With `align_corners=True`, output column `x` samples at
`u = x·(FW−1)/(CROP_W−1) = 70x/1135`, and bilinear reads cells `⌊u⌋` and
`⌊u⌋+1`. For `x` to depend only on `[w_lo, w_hi]`:

```
⌊u⌋   ≥ w_lo  ⟺  70x ≥ 1135·w_lo
⌊u⌋+1 ≤ w_hi  ⟺  70x <  1135·w_hi
```

With `[w_lo, w_hi] = [20, 39]`:

```
x ≥ ⌈1135·20/70⌉ = ⌈324.2857⌉ = 325
x <  1135·39/70  =  632.357    ⟹  x ≤ 632

        PIXEL BAND :  x ∈ [325, 632]        308 columns
```

Checks: `x = 324 → u = 19.98`, `⌊u⌋ = 19 < 20` ✗. `x = 325 → u = 20.04` ✓.
`x = 632 → u = 38.98`, cells `38, 39` ✓. `x = 633 → u = 39.04`, needs cell `40` ✗.

**MEASURED — the band is verified empirically, not merely derived.** Injecting
two *different* arbitrary halos into the same fields leaves the statistic on
`x ∈ [325, 632]` **bit-identical, 24/24, max |difference| exactly `0.000e+00`**
(`synthetic_mask_check.json`, S6).

---

## 7. ROWS

**DERIVED, horizontal arm.** The left and right crops occupy the **same rows**
`[0, 272)`. Their row halos are therefore *identical* and cancel in
`Lf[w+τ] − Rf[w]`. **All 17 feature rows are admissible**, and the measurement
confirms it: the `V[τ] == 0` check covered **all 17 rows** and returned exactly
zero everywhere.

**ADOPTED.** `y ∈ [64, 208)` is nevertheless retained, **unchanged from the
frozen classification spec**. The derivation permits all rows; the inherited
border only removes support and can never admit an invalid location. It is kept
for continuity, and the fact that it is *inherited rather than derived* is
stated rather than hidden.

### 7.1 The vertical control — no admissible region exists

**DERIVED.** For the vertical arm the crops occupy the same *columns* but
different *rows*, so the roles swap: columns cancel, rows must be halo-free.

```
clean rows = [15, FH−1−15] = [15, 1] = ∅          (FH = 17)
```

**Empty.** And it cannot be rescued by a taller crop: the source frame is 368 px
`= 23` feature rows, and `[15, 23−1−15] = [15, 7] = ∅` as well. A non-empty band
needs `FH ≥ 31` cells `= 496` px, which **exceeds the KITTI frame height**.

```
THE VERTICAL ARM HAS NO HALO-CLEAN REGION, AND CANNOT BE GIVEN ONE.
```

The asymmetry is structural: the crop is **71 cells wide but only 17 tall**,
against a 15-cell halo. `71 − 30 = 41 > 0`; `17 − 30 < 0`.

---

## 8. THE CANCELLATION IDENTITY — unaffected by any of this

```
V_AA + V_BB − V_AB − V_BA
  = (Af[w+k] − Af[w+τ]) + (Bf[w+k] − Bf[w+τ])
  − (Af[w+k] − Bf[w+τ]) − (Bf[w+k] − Af[w+τ])  =  0
```

Pointwise, content-independent, and **valid in the halo and in the zero-fill
region too** — in the fill region every term reduces to `−Rf`, and
`(−Rf_A − Rf_B) − (−Rf_B − Rf_A) = 0`.

**MEASURED on real features for the first time** (`validate_domain.json`):
max relative residual `1.8089e-07`, median `1.1943e-07` over 108 checks — pure
float32 round-off. **MEASURED synthetically over the entire tensor including the
injected halo and the fill:** `1.391e-07` (S3).

**The consequence that matters:** the *cancellation* never needed the mask. The
*anchor* `V(τ) = 0` — which is H1's entire prediction — did. That is precisely
the distinction the GEOM-001 design collapsed.

---

## 9. THE COMPLETE CHAIN, ONE BLOCK

```
 image crop            1136 × 272 px          (left fixed; right at +16τ)
   │  extractor, stride 16, RF 477 px, halo 15 cells
   ▼
 feature grid          71 × 17 cells
   │  boundary-free:                       w ∈ [15, 55]        41 cols
   │  translation identity Rf[w]=Lf[w+τ]:  w ∈ [15, 55−τ]      41−τ cols   (τ-dep)
   ▼
 cost volume           V[k,w] = Lf[w+k] − Rf[w],  k ∈ [0,11]
   │  all k clean + fill-free:             w ∈ [15, 44]        30 cols     (τ-INDEP)
   │  V[τ,w] = 0 exactly here             ← MEASURED, 3 525 120 cells, all 0.0
   ▼
 aggregation           5 × Conv3d(3×3×3), spatial radius 5
   │  fully supported:                     w ∈ [20, 39]        20 cols     (τ-INDEP)
   ▼
 upsample + readout    bilinear align_corners, 71 → 1136
   │  depends only on [20,39]:             x ∈ [325, 632]      308 px cols
   ▼
 MASK                  y ∈ [64, 208)  ∧  x ∈ [325, 633)
                       144 × 308 = 44 352 px  per scene, per τ, per cell
                       identical in all four cells, both arms, every τ, every model
```

---

## 10. UNKNOWNS, STATED

| item | status |
|---|---|
| why the right halo is 14 cells and the left 15 | **UNKNOWN** — stride/padding parity is the likely cause. **Not load-bearing**: the adopted bound is the conservative 15 on both sides |
| whether 220 feature cells of support give a stable `argmin` in practice | **UNKNOWN** — it cannot be known before the gate runs. Declared, not guessed |
| whether the trained aggregation maps small `\|V\|` to low cost | **UNKNOWN** — this is the hypothesis under test, not a premise |
| whether a learned non-correspondence shortcut could place its minimum at `k = τ` | **UNKNOWN** — not addressable by any ablation of this architecture on synthetic input |
