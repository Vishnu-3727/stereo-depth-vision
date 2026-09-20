# SOURCE TRACE — pixel ↔ feature ↔ candidate ↔ disparity

Record `phase2/diagnostics/correspondence_geom/design_audit_20260912/`.
2026-09-12. Companion to `DESIGN_AUDIT.md`.

**Derived from source only.** Every line below is read out of the frozen
implementation, not inferred from any observed output. Where a quantity cannot
be established from source it is marked **UNKNOWN** and no prediction is built
on it.

Tags: **MEASURED** (read from source/frozen record) · **DERIVED** (algebra) ·
**UNKNOWN**.

---

## 0. FILES READ

| stage | file |
|---|---|
| image preprocessing | `src/datasets/kitti2015.py` (`normalize`, `pad_and_crop`) |
| scene loading | `phase2/viz/core.py` (`load_scene`) |
| feature extraction | `src/models/stereonet/feature_extractor.py`, `blocks.py` |
| cost volume + shift | `src/models/stereonet/cost_volume.py` |
| aggregation | `src/models/stereonet/aggregation.py` |
| standardisation | `phase2/models/scaled_regression.py` |
| soft-argmin / readout | `src/models/stereonet/regression.py` |
| assembly | `src/models/stereonet/stereonet.py` |
| crop / mask convention | `phase2/diagnostics/correspondence_tr/20260911T134500Z/run_tr.py` |

---

## 1. IMAGE PREPROCESSING (MEASURED)

`kitti2015.pad_and_crop` anchors **top-left**: a 375×1242 KITTI frame becomes
`img[:368, :1232]`. Never centred, never resized. Smaller frames are
zero-padded bottom/right, content stays top-left.

`kitti2015.normalize(image)`:

```
x = (float32(image) - NORM_MEAN) / NORM_STD          # per-channel, 0..255 RGB
return x.transpose(2, 0, 1)[None]                    # NCHW, contiguous
```

**DERIVED:** normalisation is **pointwise** — it commutes with any translation
or crop. No geometric consequence.

**UNKNOWN and irrelevant here:** nothing in the pipeline resizes. There is no
interpolation anywhere on the input path.

---

## 2. CROPPING (MEASURED, from the frozen construction)

Source frame `1232 × 368`. Crop `CROP_W × CROP_H = 1136 × 272`
(`= 71 × 16` by `17 × 16`, stride-aligned).

```
left  crop  = src[ 0 : 272 ,  0 : 1136 ]                       FIXED, every condition
right crop  = src[ 0 : 272 ,  t : 1136+t ]     horizontal
right crop  = src[ t : 272+t ,  0 : 1136 ]     vertical
```

Integer slicing only. **No fill, no padding, no interpolation.**
`run_tr.py::crop` raises if the window leaves the source, so `t ≤ 96` with these
crop dimensions.

---

## 3. FEATURE EXTRACTION AND THE PIXEL → FEATURE MAP (DERIVED)

`FeatureExtractor` (MEASURED):

```
4 × Conv2d(·→32, k=5, stride=2, padding=2)      # NO activation between them
6 × ResBlock(32)                                 # Conv2d(k=3,pad=1) ×2, LeakyReLU 0.2
1 × Conv2d(32→32, k=3, stride=1, padding=1)
scale = 2**4 = 16
```

Fully convolutional; every layer is stride 1 except the four stride-2 convs.

**Translation law.** For a stride-2, kernel-5, padding-2 convolution,
`out[i] = Σ_j w[j]·x[2i + j − 2]`. If `x′[n] = x[n + 2]` then `out′[i] =
out[i + 1]`. So **2 input px = 1 output cell** per level, and over four levels:

```
input translated by  t  px   ⇒   output translated by  τ = t / 16  cells
                                  EXACT when 16 | t, in the interior
```

Stride-1 padded layers are exactly translation-equivariant in the interior.

**Receptive field (DERIVED).** Backwards from one output cell: final `k=3` → 3;
six ResBlocks × two `k=3` convs → `3 + 24 = 27`; then four `k=5,s=2` convs,
each `r ← 2r + 3` → `57 → 117 → 237 → 477`.

```
RF ≈ 477 input pixels
```

A 64-px mask border therefore does **not** remove receptive-field contamination
from the crop edge. GEOM-002 declared this in advance; it is declared again here
(`DESIGN_AUDIT.md` §10.3). It is a **main effect of crop geometry** and is
cancelled by the 2×2 design to the extent that the design's cancellation is
exact (§6).

---

## 4. COST VOLUME AND CANDIDATE INDEXING (MEASURED)

`cost_volume.py`:

```python
def shift_left(x, k):                      # shift == "left"
    if k == 0: return x
    return F.pad(x, (0, k))[..., k:]       # out[w] = x[w+k] ; zero for w >= W-k

def reference_shift(x, k):                 # shift == "none"  -- a NO-OP
    padded = F.pad(x, (0, k)); return padded[..., :width]      # == x

levels[k] = shifter(left, k) - right       # method == "subtract"
volume    = stack(levels, 1).permute(0, 2, 1, 3, 4)     # (B, C, D, H, W)
```

**Candidate index `k` is the disparity axis**, `D = 12`, at 1/16 resolution.

### 4.1 The exact indexing equation (DERIVED)

Let `Lf`, `Rf` be the feature maps, `w` a feature column. For `shift="left"`:

```
V[k][w]  =  Lf[w + k] − Rf[w]                    for w < W_f − k
V[k][w]  =  0        − Rf[w]                     for w ≥ W_f − k     (zero fill)
```

For `shift="none"`, `V[k][w] = Lf[w] − Rf[w]` **for every `k`** — the volume is
candidate-constant (EXP-010).

### 4.2 The synthetic self-pair substitution (DERIVED)

With `right crop = src[:, t : 1136+t]` and `16 | t`, §3 gives `Rf[w] = Lf[w + τ]`,
so for `shift="left"`:

```
V[k][w] = Lf[w + k] − Lf[w + τ] = F[k − τ][w + τ],     F[κ][w] := Lf[w+κ] − Lf[w]

⇒  V(k, w) = F(k − τ, w + τ)        F is τ-INDEPENDENT
⇒  V = 0 EXACTLY at k = τ           (interior; verified numerically, §8)
```

### 4.3 Sign convention (DERIVED, and checked against the frozen W2 assertion)

`right[y, x] = src[y, x + t] = left[y, x + t]`. A scene feature that sits at
**left column `x + t`** sits at **right column `x`**. Stereo disparity is
`d = x_left − x_right = (x + t) − x = +t` px. Therefore:

```
imposed disparity d = +t px            (positive, the physical convention)
matching candidate  k_true = τ = t/16  (the candidate at which V vanishes)
1 candidate = 16 full-resolution pixels
```

This agrees with the independently-verified assertion frozen in
`correspondence_tr/.../run_tr.py::verify_sign_from_data`
(`right[:, :-t] == left[:, t:]`, checked from the arrays, 84 checks, PASS).

### 4.4 Right-edge zero fill (DERIVED — this bounds the usable mask)

`shift_left(x, k)` zero-fills feature columns `[W_f − k, W_f)`. With
`W_f = 71` and `k ≤ 11`, columns `[60, 71)` can be fill. The aggregation adds a
spatial radius of 5 cells (§5) and the bilinear upsample one more, so a feature
column is uncontaminated iff `w + 6 < 60`, i.e. **`w < 54`**, i.e.
**`x < 864 px`**.

---

## 5. AGGREGATION (MEASURED)

```
4 × [ Conv3d(32→32, k=3×3×3, padding=1) + LeakyReLU(negative_slope=0.01) ]
1 × Conv3d(32→1,  k=3×3×3, padding=1) , squeeze(1)       ->  (B, D, H, W)
```

5 convolutions ⇒ radius **5** along each of `(k, y, w)`.
`D = 12` and radius 5 ⇒ only `k ∈ {5, 6}` are padding-free along the candidate
axis. **This is the artifact channel** named in `predesign_audit_20260911`.

**DERIVED:** the module is translation-**equivariant** in `(k, y, w)` up to
padding, for *any* weights — this is the property that forces a unit ramp.

**DERIVED and decisive for this design:** the module is **linear in its input
except for the LeakyReLU**. With the slope set to 1 it is exactly affine. See §6.

---

## 6. THE 2×2 CANCELLATION (DERIVED — the load-bearing algebra)

Four cells at a fixed `τ`, from two sources `A`, `B`:

```
V_AA[k][w] = Af[w+k] − Af[w+τ]        V_AB[k][w] = Af[w+k] − Bf[w+τ]
V_BB[k][w] = Bf[w+k] − Bf[w+τ]        V_BA[k][w] = Bf[w+k] − Af[w+τ]
```

```
V_AA + V_BB − V_AB − V_BA
  = (Af[w+k] − Af[w+τ]) + (Bf[w+k] − Bf[w+τ])
  − (Af[w+k] − Bf[w+τ]) − (Bf[w+k] − Af[w+τ])
  = 0                                   IDENTICALLY, for every k, w, τ
```

**The cost-volume 2×2 interaction is exactly zero** because the cost volume is
**linear in both feature maps**. Consequences, all DERIVED:

1. Any **affine** aggregation returns an identically-zero aggregated-cost
   interaction, for **any** weights — including all padding-boundary structure,
   which is part of the linear operator and therefore cancels exactly.
2. Whatever interaction survives is produced **entirely by the LeakyReLU
   nonlinearity** (and, downstream, by the per-condition standardisation).
3. Every additive main effect — left identity, right identity, `τ`, crop
   geometry, receptive-field edge contamination, architecture, padding —
   cancels.

**MEASURED (`synthetic_mechanism_check.py`, S0):** max relative residual
`1.566e-07` over all `τ`, i.e. float32 round-off. Exact-zero confirmed
numerically.

---

## 7. STANDARDISATION AND READOUT (MEASURED)

`stereonet.forward` → `regression`, with `scaled_regression.apply_to(model)`
installed (the H2 configuration used by every correspondence experiment):

```
cost  (B, 12, 17, 71)
  -> F.interpolate(bilinear, align_corners=True, size=(272, 1136))
  -> z = (cost − mean_k cost) / (std_k cost + 1e-5)      per pixel, across k,
                                                          population sd
  -> p = softmax(−z, dim=k)
  -> disparity_initial = Σ_k p(k)·k                       soft-arg*min*
```

`DisparityRegression` and `StandardisedDisparityRegression` hold **zero
parameters** (asserted at load in `run_tr.py`).

**DERIVED, and it fixes the direction of the primary prediction:** the negation
inside the softmax makes **low `z` ⇒ high weight**. A candidate that the network
treats as the match must be a **MINIMUM** of `z(k)`.

**DERIVED:** standardisation is, at a fixed pixel, an affine map of `c(k)` with
`k`-independent coefficients. It therefore **cannot move `argmin_k` within a
condition**. It is *not* linear *across* conditions, so it contributes a second
(small) nonlinear channel to the interaction — measured in §8.

**Refinement is never invoked** by any correspondence experiment: the readout
path stops at `disparity_initial` / the `z` tensor.

---

## 8. WHAT THE SYNTHETIC CHECK CONFIRMED (MEASURED, no trained model)

`synthetic_mechanism_check.py`, 8 framework-default random aggregations,
synthetic smoothed feature fields, geometry identical in shape to the frozen
construction.

| id | property | result |
|---|---|---|
| S0 | matched `V` exactly 0 at `k = τ` on the safe region | `max|V| = 0.0` |
| S0 | cost-volume 2×2 interaction | max relative residual `1.566e-07` (float32 round-off) |
| S3 | interaction amplitude, **nonlinear** random weights | median `1.124` |
| S3 | interaction amplitude, **affine** aggregation (slope 1.0) | median `0.0789` — ~14× smaller |

The residual affine-arm amplitude is the per-condition standardisation, exactly
as §7 predicts. The mechanism is confirmed: **the interaction is a
nonlinearity-generated quantity, and the linear artifact channel is cancelled.**

---

## 9. GROUND-TRUTH DISPARITY (MEASURED) — and why it is not used

`Kitti2015Stereo` loads `disp_occ_0` and divides by `disparity_scale = 256.0`;
`Scene.gt_disparity` carries it. **This design uses no ground truth at all**:
the synthetic self-pair *imposes* `d = t` px uniformly, so GT would be both
redundant and wrong (it describes the real right image, which is never used).

This removes the GT-band machinery, the scene-selection question and the
mask-occupancy confound that recurred from INDEX-001 through GEOM-001.

---

## 10. VALID MASK (DERIVED — every bound has a source)

| bound | value | source |
|---|---|---|
| top / bottom / left border | 64 px | inherited unchanged from `correspondence_arch_rate/.../classification_spec.json` |
| **right border** | `x < 864 px` | **DERIVED in §4.4**: `shift_left` fill (11 cells) + aggregation radius (5) + upsample slop (1) ⇒ `w < 54` |
| retained | `64 ≤ y < 208` ∧ `64 ≤ x < 864` = `144 × 800 = 115 200 px` | |

**GT-independent, `τ`-independent, axis-independent, model-independent.**
It is identical in every cell of the 2×2 and at every `τ`, so it cannot
introduce a pairing-dependent effect.

---

## 11. THE COMPLETE CHAIN, ONE EQUATION

```
 imposed pixel displacement   t   (integer, 16 | t, 0 < t ≤ 96)
      │  extractor stride 16, fully convolutional, exact in the interior
      ▼
 feature displacement         τ = t / 16   cells
      │  V[k][w] = Lf[w+k] − Rf[w] ,  Rf[w] = Lf[w+τ]  for a matched self-pair
      ▼
 candidate index              k ;   V = 0 exactly at k = τ
      │  aggregation (equivariant in k up to padding) → standardise → softmax(−z)
      ▼
 readout                      a match must be a MINIMUM of z(k)
      │  1 candidate = 16 full-resolution pixels   (exact, by construction)
      ▼
 disparity                    d = 16·k px      — but see §12
```

---

## 12. IS `disparity_initial` A PHYSICAL DISPARITY? (the §VIII question)

**No — not as a scalar, and the design does not treat it as one.**

- **MEASURED (source):** `soft_argmin` emits **candidate units `0…11`**, and the
  docstring states the scale is supplied by the refinement stage, which is never
  run here.
- **MEASURED (TR-001 §3, cited only as a statement that a conversion is
  unavailable — no magnitude of it is used anywhere in this design):** the
  matched response is affine in `τ` with a non-zero intercept and saturates near
  the top of the grid. It is **not** the identity map on `τ`.
- **DERIVED:** the soft-argmin is a softmax-weighted mean over a finite support,
  so it is compressed toward the support's centre and clipped at the rails. The
  scalar therefore carries an unknown monotone distortion.

**Resolution adopted by this design.** The primary statistic is stated in the
model's **native candidate units** and is the **integer `argmin_k` of a cost
profile**, not the soft-argmin scalar. `argmin_k` is invariant to the per-pixel
affine standardisation (§7) and is not subject to softmax compression at all.
The conversion `1 candidate = 16 px` is exact by construction and is recorded,
but **no step of the decision rule depends on it.**

---

## 13. WHAT COULD NOT BE ESTABLISHED FROM SOURCE

| item | status |
|---|---|
| the value of `disparity_initial` for any input | **UNKNOWN** — deliberately. No numeric expectation is placed anywhere in this design (the GEOM-002 `5.5` failure). |
| whether the trained aggregation maps small `|V|` to low cost | **UNKNOWN** — this is the hypothesis under test, not a premise. |
| the per-layer decomposition of the `V ≡ 0` output | **UNKNOWN** — would require running the aggregation on a zero tensor with trained weights; not done. |
| whether a trained non-correspondence shortcut could mimic the signature | **UNKNOWN** — not addressable by any ablation of this architecture (`predesign_audit_20260911` §5.3). Reflected in the claim ceiling. |
