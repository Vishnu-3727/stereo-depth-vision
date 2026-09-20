# PREREGISTRATION — EXP-CORRESPONDENCE-GEOM-001

## Pairing-Swap Interaction Diagnostic

Record `phase2/diagnostics/correspondence_geom/20260912T011452Z/`.
Frozen 2026-09-12 UTC, **before any model was instantiated**.

Design authority: `phase2/diagnostics/correspondence_geom/design_audit_20260912/`
(`DESIGN_AUDIT.md` sha256 `e99bb53069fee829dfd3c1539198b49f0f5dcdd30922de7cca69271c082152c6`,
`design.json`, `source_trace.md`). Every element below is inherited from that
record without alteration.

**INFERENCE ONLY. NO TRAINING.** No optimizer is imported or constructed. No
`.backward()`. No `.step()`. No weight is modified anywhere. No architecture,
feature extractor, cost-volume construction, standardisation, readout or mask is
modified. No historical record is modified. No EPE / D1 / RMSE is computed.

---

## 0. CONTAMINATION DISCLOSURE

The designer has read the response curves of all eight prior correspondence
experiments. The statistic below is nevertheless fixed by an **algebraic identity
in the source** (`V_AA + V_BB − V_AB − V_BA ≡ 0`, because `build_cost_volume` is
linear in both feature maps) together with the **sign convention of the readout**
(`softmax(−z)` ⇒ a match is a **minimum**). Neither input is an observed curve.

**Status achievable, declared in advance: CONSISTENCY EVIDENCE AT BEST.**
Three checkpoints are irreducibly reused. This is not a confirmatory experiment.

---

## 1. PRIMARY QUESTION

Does the trained aggregation response contain a **nonlinear pairing
interaction** whose minimum occurs at the imposed true candidate coordinate
`k = τ`?

Causal contrast: **correct pairing vs. incorrect cross-pairing**, holding fixed
the two source images, the imposed displacement, the image marginals, the crop
geometry, the candidate grid, the model, the architecture, the mask and the
preprocessing.

Mechanisms to be distinguished:

| | prediction for `I_τ(k)` |
|---|---|
| **H1** genuine geometric pairing information | a **minimum at `k = τ`**, at every `τ` |
| **H2** generic binocular / right-image dependence | ≈ 0 — a right-image main effect cancels identically |
| **H3** linear candidate-axis architectural artifact | **exactly 0** for any affine aggregation; no reliable tracking |

---

## 2. SOURCE-DERIVED GEOMETRY (used as the only mapping; not inferred from any observed output)

```
extractor stride = 16 px, fully convolutional, exact in the interior
τ = t / 16

synthetic self-pair:   right[y,x] = left[y, x+t]
                  ⇒   d = x_L − x_R = +t px          ⇒   k_true = τ

cost volume:      V[k,w] = Lf[w+k] − Rf[w]
matched pair:     Rf[w]  = Lf[w+τ]
                  ⇒  V(k,w) = F(k−τ, w+τ),  F τ-independent
                  ⇒  V(τ,w) = 0  EXACTLY

readout:          aggregate → bilinear upsample (align_corners=True)
                  → z = (c − mean_k c)/(sd_k c + 1e-5)
                  → softmax(−z)  ⇒  A MATCH IS A MINIMUM OF z(k)
```

Full trace: `design_audit_20260912/source_trace.md` (sha256
`bcdd4e3384ecb07630a68016e3b174b2012510b322b4e3a9cc99f182077de63e`), copied into
this record as `source_trace.md` without modification.

**Candidate units are never converted to physical disparity** except in the
wiring checks that verify the construction (`1 candidate = 16 px`).

---

## 3. CONSTRUCTION (inherited verbatim)

Source frame `1232 × 368`. Crop `1136 × 272` (`71 × 17` feature cells).

```
left  crop        rows [0, 272)          cols [0, 1136)         FIXED every condition
right crop  HORIZ rows [0, 272)          cols [16τ, 1136+16τ)
right crop  VERT  rows [16τ, 272+16τ)    cols [0, 1136)
```

Integer slicing only. **No fill, no padding, no interpolation.**

`t ∈ {16, 32, 48, 64, 80, 96}` ⇔ `τ ∈ {1, 2, 3, 4, 5, 6}`.
**`τ = 0` is excluded entirely** — at `τ = 0` the matched cells are
byte-identical, `V ≡ 0`, a structurally distinct regime. It is not a level, not
a baseline and not a normaliser.

### 3.1 The four cells

| cell | left | right | corresponds |
|---|---|---|---|
| `AA` | `A` | `A_τ` | **yes**, exactly, uniformly |
| `BB` | `B` | `B_τ` | **yes**, exactly, uniformly |
| `AB` | `A` | `B_τ` | **no** |
| `BA` | `B` | `A_τ` | **no** |

Each frame appears **exactly once as a left image and once as a right image**.
Marginals matched by construction. `τ` identical in all four cells. **Only the
pairing changes.**

---

## 4. STATISTIC (frozen)

For each unit and each `τ`:

```
z_cell(k, y, x)  = the standardised cost the softmax consumes,  cell ∈ {AA,BB,AB,BA}

I_τ(k, y, x)     = ½[z_AA + z_BB](k,y,x) − ½[z_AB + z_BA](k,y,x)      PER PIXEL

Ī_τ(k)           = median over the fixed mask of I_τ(k, ·, ·)          k = 0…11

c_τ              = argmin_k Ī_τ(k)

hit_τ            = [ c_τ == τ ]

h                = Σ_{τ=1..6} hit_τ      ∈ {0,…,6}
```

The statistic is an **interaction**, not a matched-minus-crossed difference of
outputs. It is an **integer index** in native candidate units.

### 4.1 Non-injectivity (the anti-fooling requirement)

| alternative shape | `h` |
|---|---|
| `Ī` constant in `k` | 0 |
| `Ī` a monotone ramp | ≤ 1 (argmin pinned at an endpoint) |
| `Ī` a step of any height | ≤ 1 |
| `Ī` a dip at a fixed `k₀ ≠ τ` | ≤ 1 |
| `Ī` a dip tracking `τ` with constant offset `c ≠ 0` | 0 — captured as TRACKING-WITH-OFFSET |
| `Ī` pure noise | `~Binomial(6, ~1/12)` |
| a **maximum** at `k = τ` | 0 — `argmax` recorded separately |

---

## 5. MASK (frozen, GT-independent, `τ`-independent, axis-independent, model-independent)

```
retain (y, x)  iff  64 ≤ y < 208   AND   64 ≤ x < 864
retained = 144 × 800 = 115 200 px
```

- top/bottom/left 64 px: inherited unchanged from
  `correspondence_arch_rate/.../classification_spec.json`
  (sha256 `273f7562b62dd4c29a2d617e3b2b4b24d98532fcc7d99531b82b55b377d0cd28`).
- right bound `x < 864`: **DERIVED** — `shift_left(x,k)` zero-fills feature
  columns `[71−k, 71)`, max `k = 11` ⇒ `[60, 71)`; aggregation spatial radius 5
  cells; upsample slop 1 cell; safe iff `w + 6 < 60` ⇒ `w < 54` ⇒ `x < 864`.

**Declared limitation:** the extractor receptive field is ≈477 px, so a 64-px
border does not remove crop-edge contamination. It is a geometric **main effect**,
identical in all four cells, cancelled by the interaction.

---

## 6. SCENES (frozen; selection rule admits no choice)

`hailo_val`, 40 scenes. Previously used by correspondence experiments:
`{0, 5, 6, 7, 8, 9, 27, 31}`. Rule: **the twelve lowest unused indices in
ascending order, paired consecutively.**

```
P0 (1,2)    000161_10 / 000162_10        P3 (12,13)  000172_10 / 000173_10
P1 (3,4)    000163_10 / 000164_10        P4 (14,15)  000174_10 / 000175_10
P2 (10,11)  000170_10 / 000171_10        P5 (16,17)  000176_10 / 000177_10
```

All twelve never measured in any correspondence experiment. No GT-based and no
response-based selection anywhere. **Pairs may not be added or removed after
execution begins.**

---

## 7. CHECKPOINTS (inherited verbatim from `correspondence_tr/.../spec.json`)

| key | path | sha256 |
|---|---|---|
| `H2_seed0` | `phase2/diagnostics/determinism/20260909T041500Z_baseline/checkpoints/STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA_checkpoint.pth` | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` |
| `H2_seed1` | `phase2/factorial/block_count_full/20260910T005550Z/checkpoints/EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED1_checkpoint.pth` | `58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf` |
| `H2_seed2` | `phase2/factorial/block_count_full/20260910T005550Z/checkpoints/EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED2_checkpoint.pth` | `245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69` |
| `NEG_shift_none` | `phase2/factorial/shift_none_standardized/20260909T071500Z/checkpoints/EXP-FACTORIAL-SHIFT-NONE-STANDARDISED-001-ARM-A_checkpoint.pth` | `d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1` |

Verified over file bytes at load. `cost_volume_shift="left"` for the three
trained checkpoints; `"none"` for the search-free control.

---

## 8. STAGE 1 — RANDOM-WEIGHT GATE

**Runs in its own process. Completes before any trained aggregation exists.**

Randomisation inherited verbatim from
`correspondence_arch_rate/.../randomisation_spec.json`
(sha256 `e52b5b03c5d4406e5ed9212f6681ee53bb448927eebddd367a7698ce9ce203ad`):

- **aggregation only** (111 585 params), `Aggregation(32, 32, 4)`;
- framework-default init `kaiming_uniform_(a=√5)`, bias `U(±1/√fan_in)`;
- `torch.manual_seed(seed)` immediately before construction;
- **seeds `0…31`**, unchanged;
- feature extractor **trained and frozen** (randomising it would destroy the
  `V = 0` structure at `κ = 0`);
- readout parameter-free (asserted); refinement never invoked;
- cost volumes **bit-identical** to those the trained arm will use.

```
unit  = (extractor, pair, seed)  = 3 × 6 × 32 = 576
axis  = horizontal only
h_rand(unit) = Σ_{τ=1..6} [ argmin_k Ī_τ(k) == τ ]
```

### 8.1 GATE DECISION — applied exactly

```
G-STOP :  max(h_rand) == 6
          FINAL VERDICT = STATISTIC-ARCHITECTURALLY-AVAILABLE
          STOP. Trained weights are NOT loaded. Terminal. Not reopened
          because a trained result might be interesting.

G-PASS :  max(h_rand) < 6
          record  H* := max(h_rand)          H* IS NOT ALTERED
          proceed to Stage 2
```

---

## 9. STAGE 2 — TRAINED CONDITIONS (only on G-PASS, only in a separate process)

Three checkpoints × six pairs = **18 units**, horizontal axis, `τ ∈ {1…6}`,
four cells each.

```
c_τ  = argmin_k Ī_τ(k)          h_tr = Σ_{τ=1..6} [ c_τ == τ ]
```

### 9.1 PRIMARY DECISION

| case | condition | meaning |
|---|---|---|
| **A** | `min(h_trained) > H*` **AND** all three checkpoints separate **AND** all six pairs separate | **PASS** |
| **B** | trained responses overlap the random reference | **NOT-DEMONSTRATED** — *not* evidence that correspondence is absent |
| **C** | any partial / heterogeneous separation | **CASE C** — never promoted to a pass |

### 9.2 TRACKING-WITH-OFFSET

If `h` is low **but** the anchor `c_τ − τ` is identical across all six `τ`,
classify as **TRACKING-WITH-OFFSET**. Not silently reclassified as success.

---

## 10. CONTROLS

### 10.1 Vertical control (frozen intervention, not reinvented)

Same two images, same displacement magnitude, right crop
`rows [16τ, 272+16τ)`. Same statistic, same criterion. Three checkpoints × six
pairs. **DERIVED prediction:** the cost volume compares within a row, so no `κ`
exists making `V` vanish; there is no mechanism tying `argmin_k` to `τ`.

**Frozen downgrade:** if the vertical interaction systematically tracks under the
same criterion, the verdict is downgraded to **CASE C**.

*Declared caveat:* roads, lane markings and horizon lines are vertically
self-similar, so a partial spurious match at some `τ` is possible. The vertical
arm is a magnitude-matched null and is never assigned a predicted location.

### 10.2 Search-free control

`NEG_shift_none` under its own `shift="none"` construction (candidate-constant
cost volume), six pairs, both axes. **Reported separately. Never mixed into the
random null. Never used to redefine `H*`.**

*Declared confound (inherited):* it differs from the trained arm in two ways at
once — a different checkpoint **and** a different cost-volume construction — so
it cannot isolate the construction as the cause.

### 10.3 Not run, with reason

A right-image-corruption control is **not needed**: corrupting the right image
perturbs a **main effect**, and every main effect cancels identically in the
interaction. Such a control would test a channel the statistic has already
closed algebraically.

---

## 11. HARD STOPS — every one raises and halts the process

| id | condition |
|---|---|
| **HS1** | source-derived geometry mapping wrong: for every `τ` and both axes, the right crop must satisfy `right[:, :−t] == left[:, t:]` (horizontal) / `right[:−t, :] == left[t:, :]` (vertical) **from the arrays**; and for every horizontal POSITIVE cell `max|V[:, τ, :, w < 54]| == 0.0` **exactly** |
| **HS2** | the pairing does not satisfy the exact `AA/BB/AB/BA` construction, or a frame does not appear exactly once as a left and once as a right |
| **HS3** | the displacement `τ` differs across the four cells of a level |
| **HS4** | cost-volume cancellation `max|V_AA+V_BB−V_AB−V_BA| / max|V_AB| > 1e-5`. **Bound derived from float32 round-off** (`eps = 1.19e-7`, accumulated over 32 channels and 4 terms ⇒ `~1e-6`), with a stated 10× margin. Not from any observation |
| **HS5** | wrong checkpoint (sha256 mismatch) or wrong configuration (`num_disparities ≠ 12`, wrong `cost_volume_shift`, readout parameter count `≠ 0`) |
| **HS6** | non-deterministic execution — the W4 process-restart repeat is not **bit-identical** (IEEE-754 byte comparison). Determinism is **verified by output identity**, never claimed from a framework flag |
| **HS7** | any trained aggregation is loaded before `G-PASS`. Enforced structurally: Stage 1 runs in its own process, sets `model.aggregation = None` immediately after the checkpoint load, and contains no Stage-2 code path |
| **HS8** | any historical record is modified — all prior record sha256 values re-verified before and after execution |
| — | mask retained-pixel count `≠ 115 200`; any non-finite response; any optimizer/`.backward()`/`.step()`; refinement invoked |
| — | **no numeric expectation is placed on any disparity value anywhere** |

**If a hard stop fires: STOP IMMEDIATELY. The experiment is not repaired and
continued.** This is the GEOM-002 breach and it is forbidden.

---

## 12. DETERMINISM

```
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
CUBLAS_WORKSPACE_CONFIG            = :4096:8
torch.set_grad_enabled(False)
device = cuda, dtype = float32
```

**W4:** one complete unit is recomputed in a **fresh process**; the full response
vector must be bit-identical.

---

## 13. NO POST-HOC THRESHOLDS

No threshold is taken from `α_h ≈ 1`, TR-001 amplitudes, ARCH-RATE behaviour,
INDEX-002 ordering, or any previous separation margin. The primary criterion is
the frozen `argmin` tracking rule and the random-weight gate. **No p-value is
computed** — units are not independent replicates (3 shared checkpoints, 12
shared scenes).

The only observed value entering the decision rule is `H*`, generated by this
experiment in Stage 1 before any trained aggregation exists, used solely as an
**ordering reference** for complete separation.

---

## 14. RAW DATA TO BE PUBLISHED

For every unit and every `τ`: `zbar_AA(k)`, `zbar_BB(k)`, `zbar_AB(k)`,
`zbar_BA(k)` (12 values each), `Ī_τ(k)` (12 values), `argmin`, `argmax`,
`hit`, the amplitude, the anchor, and the soft-argmin scalars
`m_AA, m_BB, m_AB, m_BA`. Aggregate `h` values alone are **not** sufficient.

---

## 15. CLAIM CEILING

**If CASE A passes — the maximum permitted claim:**

> Under the specified synthetic translation construction, the nonlinear pairing
> interaction of the trained aggregation response consistently localizes at the
> imposed candidate coordinate `k = τ` and separates from the observed random
> aggregation population.

This is evidence that the response is sensitive to **which** images are paired,
rather than merely to the applied translation.

**If G-STOP:** claim only *"the diagnostic signature is architecturally available
under the tested random aggregation population."* **Do not claim correspondence.**

**NOT proven under any outcome:** full geometric correspondence · correct matching
everywhere · correct disparity · genuine disparity search · final stereo
correctness · real-scene correspondence · generalisation.

**Required phrasing:** *"outside the observed random population"* — the reference
is 576 sampled parameterisations, not the space of random weights.

**OOD caveat, declared in advance:** a constant-disparity plane is not natural
stereo, so a negative result remains ambiguous.

---

## 16. COMPUTE

Inference only. Gate 13 824 readouts; trained horizontal 432; trained vertical
432; search-free 288. ≈ 15 000 readouts. Estimate 5–15 minutes on one GPU.

---

**Frozen. Execution begins only after these bytes are hashed.**
