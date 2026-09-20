# RESULTS — EXP-CORRESPONDENCE-GEOM-001

## Pairing-Swap Interaction Diagnostic

```
STATUS :  HALTED AT HS1
VERDICT:  NO-ADMISSIBLE-RESULT
```

The hard stop fired on the **first cost volume constructed**, before any
aggregation forward pass, before any readout, before the gate, and before any
trained aggregation was ever applied. **The experiment was halted and was not
repaired and continued.**

Record `phase2/diagnostics/correspondence_geom/20260912T011452Z/`.
Frozen 2026-09-12 UTC; executed the same day.

Tags: **MEASURED** (from this run) · **DERIVED** (algebra/source) · **INFERRED**
· **UNKNOWN**.

---

## 1. FREEZE INTEGRITY

| artifact | sha256 |
|---|---|
| `PREREGISTRATION.md` | `d335f00eec35e1757b1c15115ee46d0bfb2c2bcefcd87bafff63e0afd3dbc204` |
| `spec.json` | `2b7b0a24144d506b72e10a461b50f2704eda880cc0f57eb2029bf13526d2a3a4` |
| `source_trace.md` | `bcdd4e3384ecb07630a68016e3b174b2012510b322b4e3a9cc99f182077de63e` |
| `run_geom.py` | `6a9b5e4e68e498dfee0b382208682e799c956315eb915b47fda624ff19ab3981` |
| `freeze_geom.py` | `1b0c21de4feb29f186c6d03dd0a30dd4ddd33f2d3a47f80f5173bef1de311cd7` |

All five verified over **file bytes** before the first model was instantiated.
**36 historical records** verified before execution and **re-verified after the
halt** — `0` mismatches both times (**HS8 PASSED**). No historical record, and
no Phase-1 file, was modified.

`gate.json`, `trained.json`, `results.json` and `RESULTS.md` were all confirmed
absent before execution began.

---

## 2. SOURCE GEOMETRY VERIFICATION

**MEASURED — the pixel-level construction is correct.** At every tested `τ`, on
both axes, asserted from the actual arrays:

```
right[:, :−t] == left[:, t:]        exactly true      (horizontal)
```

**MEASURED — the feature-level mapping is correct, but only on a bounded band.**
`Rf[w] == Lf[w+τ]` holds **bit-exactly** for feature columns

```
w ∈ [15, 55 − τ]        every row, every τ ∈ {1,…,6}
```

and **fails outside it**:

| band | cause | `max |Lf[w+τ] − Rf[w]|` |
|---|---|---|
| `w ∈ [0, 14]` | the **right** crop's own **left-edge padding halo** | `5.3e6 … 9.0e6` at `w = 0` |
| `w ∈ [56−τ, 70]` | the **left** crop's **right-edge halo**, reached by the `+τ` index | `438 … 1.9e4` |

**DERIVED, and it matches exactly.** `source_trace.md` §3 computed the
extractor's receptive field as **477 px**. The one-sided radius is
`(477−1)/2 = 238 px = 14.875` feature cells ⇒ **15 cells**. The measured
contaminated band is **exactly 15 cells (240 px)**.

**MEASURED — rows are unaffected on this axis.** The horizontal arm's left and
right crops use the *same* rows `[0, 272)`, so their vertical padding halos are
identical and cancel in the difference. No row-direction contamination exists
for the horizontal arm.

---

## 3. COST-VOLUME CANCELLATION (HS4)

**NOT REACHED.** The run halted before the first cancellation check executed.

The algebraic identity `V_AA + V_BB − V_AB − V_BA ≡ 0` remains **DERIVED** from
`cost_volume.py` and was confirmed numerically in the design audit's synthetic
check (max relative residual `1.566e-07`, float32 round-off) — but it was
**not** measured on real features in this run, and is not claimed to have been.

---

## 4. THE HARD STOP

```
HardStop: HS1 AA |V[:,tau=1,:,w<54]| = 7233118.0 != 0
```

| | |
|---|---|
| where | extractor `H2_seed0`, pair 0 (`000161_10` / `000162_10`), axis horizontal, `τ = 1`, cell `AA` |
| which volume | **the first cost volume constructed in the experiment** |
| condition | `max|V[:, τ, :, w < 54]| == 0.0` exactly |
| observed | `7 233 118.0` |
| array half of HS1 | **PASSED** — the pixel geometry is exactly right |
| action | the process raised and exited; nothing was repaired, nothing continued |

### 4.1 Execution ledger (MEASURED)

| quantity | value |
|---|---|
| checkpoints loaded | 1 (`H2_seed0`) |
| feature-extractor forward passes | 3 |
| cost volumes constructed | 4 |
| **aggregation forward passes** | **0** |
| **readout passes** | **0** |
| interaction profiles computed | **0** |
| `argmin` / `h` computed | **0** |
| random-weight gate units measured | **0 of 576** |
| trained aggregation ever applied | **NO** |
| training / optimizer / `.backward()` / `.step()` | **NONE** |
| EPE / D1 / RMSE | **none computed** |
| p-values | **none computed** |

**HS7 held structurally.** `H2_seed0`'s aggregation was set to `None`
immediately after the checkpoint load — before any forward pass — and its
parameter hash (`d8d12446c584…`) was recorded but never used. Stage 2 lives in
a **separate process** that refuses to start without a `G-PASS` `gate.json`;
**no `gate.json` exists**, so Stage 2 cannot run.

---

## 5. RANDOM GATE

```
h_rand distribution :  NOT MEASURED  (0 of 576 units)
H*                  :  NONE
G-STOP / G-PASS     :  NEITHER — the gate never executed
```

---

## 6. TRAINED ARM · 7. VERTICAL CONTROL · 8. SEARCH-FREE CONTROL

**None was run.** Stage 2 was never entered: there is no `G-PASS`.
No trained `h` value, no per-checkpoint value, no per-pair value, no vertical
result and no `NEG_shift_none` result exists.

---

## 9. DETERMINISM

The frozen determinism block was set and active (`ENVIRONMENT.txt`):
`use_deterministic_algorithms(True)`, `cudnn.deterministic = True`,
`cudnn.benchmark = False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `no_grad`, fp32,
CUDA.

**W4 was NOT run** — the process-restart repeat requires a completed Stage 2
reference, which does not exist. **Determinism is therefore NOT claimed for
this run.** (A pre-freeze mechanics check on synthetic tensors did show
bit-identical in-process repetition, but that is not the frozen W4 test and is
not offered as one.)

---

## 10. HARD-STOP RESULTS

| id | outcome |
|---|---|
| **HS1** | **FIRED** — halted (§4) |
| HS2 | not reached |
| HS3 | not reached |
| HS4 | not reached |
| **HS5** | **PASSED** — sha256 match, `num_disparities == 12`, `shift == left`, readout parameter count `0` |
| HS6 | not reached (W4 requires a completed Stage 2) |
| **HS7** | **PASSED** — trained aggregation discarded before use; Stage 2 unreachable |
| **HS8** | **PASSED** — 36 historical records byte-identical after the halt |
| HS9, HS10 | not reached |
| **HS11** | **PASSED** — no optimizer, no `.backward()`, no `.step()`, refinement never invoked |

---

## 11. ROOT CAUSE

**This is a false premise in a frozen bound. It is not a harness bug, and it is
not a property of the trained model.**

The derivation `V(τ, w) = 0` is **correct** — but only where the feature
extractor is *exactly* translation-equivariant. That region excludes a
**15-feature-cell (240 px) receptive-field halo at every crop edge whose source
column differs between the two crops.** HS1 was written over `w < 54` *starting
at `w = 0`*, which includes the contaminated left band `w ∈ [0, 14]`.

### 11.1 The same omission is in the frozen mask — and it is worse there

The frozen mask retains `x ∈ [64, 864)` px = feature columns `[4, 54)`, which
**includes contaminated columns 4…14**. In that band the POSITIVE cells do not
actually correspond, so what is corrupted is **the treatment itself**, not a
nuisance main effect.

### 11.2 Why the design audit missed it

`source_trace.md` §3 **derived** the 477 px receptive field, and §10
**declared** the contamination — and then argued it was a geometric **main
effect** cancelled by the 2×2 interaction. *That argument is wrong for the
positive cells.* Where equivariance fails, `V` is not zero at `k = τ`, so the
hypothesis has no region in which to hold; there is nothing for the interaction
to cancel. The mask's **right** bound was correctly derived from the
`shift_left` fill plus the aggregation radius. The extractor halo was simply
never propagated into the **left** bound.

### 11.3 Error class

The same class as GEOM-002's C3, one level further out: **the statistic was
verified under the hypothesis, and the region in which the hypothesis actually
holds was never verified.** The design audit's own §8.2 non-injectivity proof
answered *"what else maps to h = 6?"* and never asked *"where is `V = 0`
actually true?"* — a question its own source trace had already supplied the
numbers to answer.

---

## 12. WHAT THIS RUN DOES AND DOES NOT ESTABLISH

**Establishes (MEASURED, wiring only):**

- the synthetic self-pair construction is **pixel-exact** at every `τ`;
- the feature-level translation law `Rf[w] = Lf[w+τ]` is **bit-exact** on
  `w ∈ [15, 55−τ]`, for every row and every `τ`;
- the contaminated halo is **exactly 15 feature cells**, matching the 477 px
  receptive field derived from source;
- `618 … 702` of `1207` feature cells are exactly zero at `k = τ`, in a
  full-height column band — so an exact-correspondence region **does exist**,
  it was simply not the region the protocol froze.

**Does not establish anything about correspondence.**

```
Level D — geometric correspondence :  NOT-DEMONSTRATED     (UNCHANGED)

  evidence FOR correspondence      :  NONE produced
  evidence AGAINST correspondence  :  NONE produced
  gate outcome                     :  NONE — no random unit measured
```

**No claim is made from this run beyond the wiring facts in §2 and §12.**
The claim ceiling in `PREREGISTRATION.md` §15 is not engaged at all, because no
case was reached.

---

## 13. PROCEDURAL ASSESSMENT

| | |
|---|---|
| hard stop fired | **yes** |
| hard stop **enforced** | **yes** — the process raised and exited |
| repaired and continued | **no** |
| trained result inspected before the gate | **no** — impossible; the trained aggregation was destroyed at load |

**This is the one thing that went right.** GEOM-002's declared hard stop fired
and its harness recorded the violation and kept going, which is what destroyed
that experiment. This harness halted on the first violating tensor, four
forward passes into a planned fifteen thousand, with no trained aggregation ever
applied and no statistic ever computed.

**INFERRED:** the cost of the defect is therefore ~30 seconds of compute and one
false bound, not a contaminated result that would have had to be withdrawn
later. The failure is cheap precisely because the stop was placed on a
structural property and was allowed to kill the run.

---

## 14. IS A FURTHER CORRESPONDENCE EXPERIMENT JUSTIFIED?

**Not decided here, and deliberately not designed here.**

What this run *does* change, factually:

- the pairing-swap intervention itself is **untested** — it was never reached;
- the cancellation identity, the gate, and the statistic are all **unmeasured**;
- an exact-correspondence region **provably exists** and has now been *measured*
  rather than assumed: the full-height column band `w ∈ [15, 55−τ]`, i.e.
  `x ∈ [240, 896−16τ)` px, `618…702` of `1207` feature cells;
- the frozen mask `x ∈ [64, 864)` **overlaps but does not lie inside** that
  band.

**No successor is designed, preregistered or launched in this record.** Doing so
would be repairing the experiment and continuing, which the protocol forbids.
Whether a successor is warranted — and with what bound, what mask, and what
audit of *where the hypothesis holds* — is a separate authorisation.

---

## 15. FILES IN THIS RECORD

| file | role |
|---|---|
| `PREREGISTRATION.md` | the frozen protocol |
| `spec.json` | machine-readable frozen specification |
| `source_trace.md` | the source-derived geometry (copied unmodified from the design record) |
| `run_geom.py` | the harness (frozen before execution) |
| `freeze_geom.py` | the freeze tool |
| `frozen.sha256`, `historical.sha256` | integrity manifests |
| `run_stage1.log`, `run.log` | execution logs up to the halt |
| `ENVIRONMENT.txt` | execution environment |
| `results.json` | machine-readable outcome |
| `RESULTS.md` | this document |
| `RELATED_RUNS.md` | provenance |
| `diagnose_hs1.py`, `diagnose_hs1.log` | **post-halt wiring diagnostic** — cost-volume zero-set only; no aggregation, no interaction, no statistic |

**No `gate.json`. No `trained.json`. No successor.**
