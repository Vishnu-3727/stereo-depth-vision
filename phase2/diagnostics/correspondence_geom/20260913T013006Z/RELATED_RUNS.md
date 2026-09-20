# RELATED RUNS — EXP-CORRESPONDENCE-GEOM-002

Record `phase2/diagnostics/correspondence_geom/20260913T013006Z/`.

Provenance for every record this experiment inherits from, depends on, or
supersedes. All 63 files listed in `historical.sha256` are verified byte-identical
before **and** after every stage (HS8). **Nothing in this list was modified.**

Tags: **MEASURED** · **DERIVED** · **FROZEN** · **VOID** (result exists but is
inadmissible) · **UNKNOWN**.

---

## 1. THE DIRECT PREDECESSOR

### `20260912T011452Z` — EXP-CORRESPONDENCE-GEOM-001 · **HALTED AT HS1**

```
STATUS  : HALTED AT HS1 -- INVALID DOMAIN
VERDICT : NO-ADMISSIBLE-RESULT
```

The same intervention, on a mask that included the extractor's receptive-field
halo. It halted on the **first cost volume constructed**, four forward passes
into a planned fifteen thousand:

```
HardStop: HS1 AA |V[:,tau=1,:,w<54]| = 7233118.0 != 0
```

**What GEOM-002 takes from it, unchanged:** the intervention (AA/BB vs AB/BA),
the six `tau`, the six scene pairs, the three checkpoints, the 32 gate seeds, the
statistic and its median reduction, the unit `(aggregation, scene pair)`, the
`h = sum of hits` primary, the two-process HS7 enforcement, `CANCEL_TOL = 1e-5`,
and the row band `y in [64,208)`.

**What GEOM-002 changes, and why:** the mask, from `x in [64,864)` (115 200 px)
to `x in [325,633)` (44 352 px). GEOM-001's bound started at feature column
`w = 0`, inside the measured 15-cell halo, where `V(tau) = 0` is false — so the
**positive cells did not actually correspond** and the treatment itself was
corrupted, not a nuisance main effect.

**What GEOM-002 does NOT reuse:** any gate number. GEOM-001 measured **0 of 576**
gate units before halting, so there is nothing to reuse; and every earlier gate
figure in the campaign was computed on the invalid mask or on synthetic fields.
`H*` is regenerated from scratch in this record.

**Its one procedural success, preserved here:** the hard stop fired and was
*enforced* — the process raised and exited, with no trained aggregation ever
applied. The same policy is in this harness.

---

## 2. THE DOMAIN AUTHORITY

### `domain_audit_20260912T013530Z` — the corrected coordinate chain

**This record designs nothing; this audit designed the domain.** Every band in
`design.json` is copied from it. The chain, all of it either **SOURCE**,
**DERIVED** or **MEASURED**:

```
 image crop            1136 x 272 px
   | extractor, stride 16, RF 477 px, halo 15 cells      <- MEASURED 216/216
 feature grid          71 x 17 cells
   | boundary-free:                       w in [15, 55]
   | translation identity Rf[w]=Lf[w+tau]: w in [15, 55-tau]   (tau-dependent)
 cost volume           V[k,w] = Lf[w+k] - Rf[w]
   | all k clean + fill-free:             w in [15, 44]        (tau-INDEPENDENT)
   | V[tau,w] = 0 exactly here   <- MEASURED, 3 525 120 cells, every one 0.0
 aggregation           5 x Conv3d(3x3x3), spatial radius 5
   | fully supported:                     w in [20, 39]
 upsample + readout    bilinear align_corners, 71 -> 1136
   | depends only on [20,39]:             x in [325, 632]
 MASK                  y in [64,208) AND x in [325,633)  =  44 352 px
```

Key measurements this record relies on and does not re-derive:

| quantity | value | status |
|---|---|---|
| extractor halo, left / right | 15 / 14 cells (15 adopted both sides) | **MEASURED** |
| `Rf[w] == Lf[w+tau]` on `[15, 55-tau]` | exact in **216/216** records | **MEASURED** |
| `V[tau, :, :, 15:45] == 0.0` | **3 525 120** cells, all exactly zero | **MEASURED** |
| 2x2 cancellation on real features | max relative residual `1.8089e-07` | **MEASURED** |
| mask invariance to injected halos | **24/24 bit-identical**, max diff `0.0` | **MEASURED** |
| the GEOM-001 mask under the same test | argmin flipped in **8 of 24** | **MEASURED** |
| vertical clean row band | `[15, 1]` = **EMPTY**, unrescuable | **DERIVED** |

---

## 3. THE DESIGN AUDIT GEOM-001 WAS BUILT FROM

### `design_audit_20260912` — **superseded on the domain question only**

Its synthetic mechanism check remains valid and is still cited: the affine-arm
control (LeakyReLU slope 1.0) gives an aggregated interaction of `<= 2.3e-06`
relative against `0.71 … 0.86` for the nonlinear arm — five orders of magnitude,
which is why the aggregation's own padding needs no exclusion.

Its **domain reasoning is superseded.** It derived the 477 px receptive field
and then argued the contamination was a main effect the 2x2 would cancel. That
argument is wrong for the positive cells, and the domain audit replaced it.

---

## 4. EARLIER CORRESPONDENCE RECORDS (context; no number reused)

| record | what it was | disposition here |
|---|---|---|
| `correspondence/20260910T142622Z` | the original shift probe | superseded; no figure reused |
| `correspondence_index/20260911T015124Z` | candidate-index probe | superseded; no figure reused |
| `correspondence_arch/20260911T025312Z` | architecture arm | superseded; no figure reused |
| `correspondence_geom/20260911T111809Z` | first geometric attempt | superseded; **VOID** |
| `correspondence_geom/20260911T115917Z` | second geometric attempt | superseded; **VOID** |
| `correspondence_arch_rate/20260911T130507Z` | rate arm | superseded; specs only |
| `correspondence_tr/20260911T134500Z` | translation arm | superseded; no figure reused |

Plus the four audits that dismantled them —
`postmortem_20260911/`, `predesign_audit_20260911/ALGEBRA_AUDIT.md`,
`successor_audit_20260911/`, `gate_audit_20260911/GATE_AUDIT.md` — all carried
in `historical.sha256` and all unmodified.

> **The campaign's standing position before this record: 0 admissible Level-D
> results.** This record does not change that position by existing; only its
> outcome can, and only up to the ceiling in `PREREGISTRATION.md` A.16.

---

## 5. THE CHECKPOINTS

| key | record of origin | sha256 | shift |
|---|---|---|---|
| `H2_seed0` | `determinism/20260909T041500Z_baseline` (STAGEB ARM-A RUNA) | `581d62e6…` | left |
| `H2_seed1` | `factorial/block_count_full/20260910T005550Z` (ARM-A SEED1) | `58117f26…` | left |
| `H2_seed2` | `factorial/block_count_full/20260910T005550Z` (ARM-A SEED2) | `245a3f5b…` | left |

All three verified at load (HS5) and again in `checkpoint_hashes.json`.
**No checkpoint was trained, fine-tuned, or selected in this record.**

**Not independent.** Three seeds of one recipe, and the designer has read eight
prior response curves from them. The ceiling is consistency evidence at best.

### Not run

`NEG_shift_none`
(`factorial/shift_none_standardized/20260909T071500Z`) — the search-free control,
a `shift="none"` model whose twelve disparity slices are provably identical
(EXP-010). The authorisation says *"Load exactly these frozen H2 checkpoints"*,
so it is excluded. **Cost:** this record carries one null — the
random-aggregation gate — and no architecture-level null. Declared in
`PREREGISTRATION.md` B.4 and in `results.json`.

---

## 6. PHASE 1

Frozen at git tag `phase-1-frozen` (`0fc4f296`). `git diff` against that tag over
`src/ data/ scripts/ tests/ reference/` is empty. The nine Phase-1 source files
this experiment executes are individually hashed in `historical.sha256`:

```
src/models/stereonet/{cost_volume,aggregation,feature_extractor,
                      regression,stereonet,blocks,refinement}.py
phase2/models/scaled_regression.py
src/datasets/kitti2015.py
phase2/viz/core.py
```

**No Phase-1 file is written to by any stage of this experiment.**
