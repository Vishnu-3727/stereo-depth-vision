# PREREGISTRATION — EXP-CORRESPONDENCE-GEOM-002

## Pairing-Swap Interaction Diagnostic, on the corrected domain

Record `phase2/diagnostics/correspondence_geom/20260913T013006Z/`.
Frozen 2026-09-13 UTC, before any model was instantiated in this record.

This document reproduces the authorised protocol **verbatim** in Part A, then
declares in Part B every point on which the protocol is silent and how it was
resolved. **Every resolution in Part B is an inheritance from an already-frozen
record, not a new choice.** No resolution was made after seeing any result from
this record; the freeze manifest (`frozen.sha256`) fixes this file's bytes
before execution.

Tags used throughout: **SOURCE** (read from the implementation) · **MEASURED**
(recorded in a named record) · **DERIVED** (algebra on the two) · **FROZEN**
(inherited preregistration) · **UNKNOWN**.

---

# A. THE AUTHORISED PROTOCOL (verbatim)

### Objective

Determine whether the trained H2 aggregation produces a pairing-dependent
response that tracks a known horizontal geometric displacement, beyond what is
obtained from randomized aggregation weights under the same complete statistic.

This experiment tests geometric pairing sensitivity.

It does **not** by itself establish full real-scene correspondence, general
disparity search, correct metric disparity, or causal uniqueness of the learned
mechanism.

---

## A.1 Immutable prerequisites

Before execution verify:

* Phase 1 remains frozen.
* All historical records remain byte-identical.
* GEOM-001 remains immutable and is recorded as `HALTED — INVALID DOMAIN`.
* No old GEOM gate result is reused.
* No architecture, candidate count, dilation, refinement, temperature, or
  checkpoint is changed.
* Deterministic execution controls are active.
* No optimizer, backward pass, or parameter update is permitted.

If any prerequisite fails: `HARD STOP`.

---

## A.2 Valid horizontal domain

Use exactly:

```text
Feature translation:
w in [15,55]

Cost-volume:
w in [15,44]

Aggregation/statistic:
w in [20,39]

Pixel:
x in [325,632]

Final mask:
y in [64,208)
x in [325,633)

Pixel count:
44,352
```

Do not widen this domain.

Do not change it after observing results.

---

## A.3 Geometry

Use the frozen horizontal translation construction.

For each displacement tau:

```text
left  = frozen left crop
right = horizontally translated corresponding crop
```

The validated construction must satisfy:

```text
R_f[w] == L_f[w + tau]
```

throughout the validated translation domain.

The matched cost-volume anchor must satisfy:

```text
V[tau,w] == 0
```

exactly on the valid cost-volume domain.

If either condition fails: `HS1 HARD STOP`.

---

## A.4 Pairing cells

For every scene pair and tau construct exactly four cells:

```text
AA = (A,A)
BB = (B,B)
AB = (A,B)
BA = (B,A)
```

Each source image must occur exactly once as a left image and once as a right
image. The same tau must be used for all four cells. No independent random
translations are permitted.

If pairing or tau equality fails: `HS2/HS3 HARD STOP`.

---

## A.5 Primary interaction statistic

Run the complete frozen aggregation and readout.

For each candidate k calculate:

```text
I_tau(k) = 0.5*[z_AA(k) + z_BB(k)] - 0.5*[z_AB(k) + z_BA(k)]
```

The z-score/readout must remain exactly the frozen H2 readout. No additional
normalization, smoothing, filtering, temperature, or post-processing is
permitted.

The primary estimate is `k_hat_tau = argmin_k I_tau(k)`.

Exact ties use the smallest candidate index.

Record: `tau`, `k_hat`, `I_min`, `I_second`, `I_second - I_min`.

The margin is descriptive only and must not alter the verdict.

---

## A.6 Primary statistic

For each complete experiment: `h = sum over tau of 1[k_hat_tau == tau]`.

There are six tau values, therefore `0 <= h <= 6`.

The six tau values are frozen before execution. No alternate tolerance or
shifted-success criterion may be introduced after seeing results.

---

## A.7 Stage 1 — Random aggregation gate

This stage MUST run before any trained aggregation checkpoint is loaded.

For each seed `0..31` randomize **only** the aggregation parameters.

Keep frozen: feature extractor; cost-volume construction; candidate count;
readout; geometry; pairing; mask; refinement disabled.

Use the same deterministic execution protocol. Compute `h` for every random
seed. Record the complete result for every seed.

Define `H* = max(h_rand)`.

If `H* == 6` then `G-STOP`, `verdict = STATISTIC-ARCHITECTURALLY-AVAILABLE`.
Do not load trained aggregation checkpoints. Do not execute Stage 2. This
result does not establish correspondence.

---

## A.8 Stage 2 — Trained aggregation

Execute Stage 2 only if `H* < 6`.

Load exactly these frozen H2 checkpoints: `H2 seed0`, `H2 seed1`, `H2 seed2`.

No retraining. No fine-tuning. No checkpoint selection.

For every checkpoint use: the same six scene pairs; the same six tau values; the
same AA/BB/AB/BA construction; the same corrected mask; the same frozen readout.

Compute `h_trained` for each checkpoint.

---

## A.9 Main comparison

The primary trained-versus-random comparison is `h_trained > H*`.

The strongest admissible outcome requires `min(h_trained) > H*` across all three
trained checkpoints.

A result that does not exceed `H*` is not evidence of correspondence. It is
`NOT-DEMONSTRATED`. Do not interpret this as evidence that correspondence is
absent.

---

## A.10 Partial outcome

If some trained checkpoints exceed `H*` and others do not: `CASE C —
HETEROGENEOUS`. Do not pool checkpoints to manufacture a stronger result.
Report every checkpoint separately.

---

## A.11 Tracking-with-offset diagnostic

If `k_hat_tau != tau` but `k_hat_tau - tau` is identical across all six tau
values for a checkpoint and scene pair, record `TRACKING-WITH-OFFSET`.

Diagnostic only. Do not redefine the primary `h` statistic to count offset
matches.

---

## A.12 Vertical control

Retain the vertical perturbation exactly as designed.

It is NOT a clean matched null because no halo-clean vertical domain exists.

Therefore: a negative vertical result is uninformative; a positive vertical
response may downgrade the interpretation. The vertical arm must never override
the horizontal primary result.

---

## A.13 Algebraic validation

Before trained/random interpretation verify `AA + BB - AB - BA` at the raw
cost-volume level is numerically zero within the previously validated
floating-point tolerance.

The previous real-feature bound `max relative residual ~ 1.81e-7` is the
historical validation reference.

Do not claim that this cancellation must remain exact after the nonlinear
z-score. The z-score is part of the measured statistic and is calibrated
empirically by the random gate.

---

## A.14 Hard stops

Immediately stop if any of the following occurs:

| id | condition |
|---|---|
| **HS1** | Feature translation fails on the validated domain. |
| **HS2** | AA/BB/AB/BA construction is incorrect. |
| **HS3** | tau differs across the four cells. |
| **HS4** | Raw cost-volume cancellation fails beyond the frozen tolerance. |
| **HS5** | Checkpoint identity, candidate count, or readout parameter count differs from the frozen specification. |
| **HS6** | Process-restart determinism fails. |
| **HS7** | A trained aggregation checkpoint is loaded before random G-PASS. |
| **HS8** | Any historical record is modified. |
| **HS9** | Mask pixel count differs from 44,352. |
| **HS10** | Any response contains NaN/Inf. |
| **HS11** | Optimizer, backward, parameter update, or refinement is invoked. |

No post-hoc restart is permitted after a hard stop without creating a new
preregistration.

---

## A.15 Required outputs

`design.json`, `preregistration.json`, `gate.json`, `trained.json`,
`results.json`, `metrics.csv`, `per_scene.csv`, `per_tau.csv`,
`hard_stop.json`, `environment.json`, `checkpoint_hashes.json`.

Also record: all random seeds; all tau values; all scene-pair identifiers; all
argmins; all candidate-response vectors; all tie events; all minimum/second-
minimum margins; all mask dimensions; all checkpoint SHA256 values.

---

## A.16 Claim ceiling

If trained response exceeds the random gate, the strongest admissible claim is:

> Under the specified synthetic horizontal pairing intervention, the trained
> nonlinear aggregation-plus-readout response tracks the known geometric pairing
> displacement more strongly than the preregistered random-aggregation
> population.

Do NOT claim: full stereo correspondence; general disparity search; correct
disparity on natural scenes; metric depth recovery; causal uniqueness;
architectural necessity; generalization.

If the trained response does not exceed the random gate:
`TRUE CORRESPONDENCE — NOT DEMONSTRATED`. Do not claim absence of
correspondence.

---

## A.17 Immutable experimental rules

Do not: widen the mask; add additional tau values; remove difficult scene pairs;
change the argmin rule; introduce an offset tolerance; tune temperature; alter
z-score; change aggregation; add refinement; train new checkpoints; reuse
historical random-gate results; use the vertical arm as the primary statistic;
select results after execution.

Any such change requires a new experiment and new preregistration.

---
---

# B. RESOLUTIONS OF POINTS THE PROTOCOL LEAVES OPEN

The protocol in Part A fixes the domain, the geometry, the cells, the
statistic's algebraic form, the gate rule, the hard stops and the claim ceiling.
It does not state seven mechanical details that the harness cannot run without.
Each is resolved below **by inheritance from a named frozen record**, and each
is fixed in `preregistration.json` and hashed into `frozen.sha256` before
execution.

The declaration exists because this campaign has twice been destroyed by a
detail that was settled silently. Silence is now an error condition, not a
default.

---

## B.1 — R1 · the pixel reduction of the statistic

**The gap.** A.5 writes `I_tau(k)` as one number per candidate. The readout
produces `z` per pixel, per candidate: `z[k, y, x]`. A reduction over the mask
is therefore unavoidable, and A.5 does not name it.

**Resolution — INHERITED, unchanged.**

```text
Ibar_tau(k) = median over the 44,352 masked pixels of I_tau(k, y, x)
```

**Authority.** `phase2/diagnostics/correspondence_geom/20260912T011452Z/spec.json`
section `statistic`:

```json
"per_pixel_interaction": "I_tau(k,y,x) = 0.5*[z_AA + z_BB] - 0.5*[z_AB + z_BA]",
"profile": "Ibar_tau(k) = median over mask of I_tau(k,.,.)",
"c_tau": "argmin_k Ibar_tau(k)"
```

and `domain_audit_20260912T013530Z/design.json` section `unchanged`, which lists
*"interaction statistic I_tau(k) and argmin_k == tau"* among the items carried
forward untouched.

**Why this is not "post-processing" in the sense A.5 forbids.** A.5 forbids
additional *normalization, smoothing, filtering, temperature*. Those all change
the per-candidate response before the argmin. The median is the reduction that
turns a per-pixel field into the per-candidate profile the protocol already
presupposes; without it `I_tau(k)` has no value. It is applied identically to
all four cells and to all twelve candidates, and it is the reduction that was
frozen before GEOM-001.

**What is NOT done.** No mean, no trimmed mean, no per-candidate re-centering,
no cross-tau normalisation, no smoothing along k. Both the median profile and
the raw per-candidate cell medians are recorded.

---

## B.2 — R2 · the unit, and what "each complete experiment" means

**The gap.** A.6 defines `h` over the six tau, so `h` is a property of one
(aggregation, scene pair). A.7 says "compute `h` for every random seed" and
A.8 says "compute `h_trained` for each checkpoint" — but the design runs six
scene pairs, so one seed and one checkpoint each yield several `h` values.

**Resolution — INHERITED, unchanged.**

```text
unit                 := (aggregation instance, scene pair),  h in [0, 6]
gate population      := 3 extractors x 6 pairs x 32 seeds  =  576 units
trained population   := 3 checkpoints x 6 pairs            =   18 units
H*                   := max(h) over all 576 gate units
primary comparison   := min(h) over all 18 trained units  >  H*
```

**Authority.** GEOM-001's harness defines exactly this unit
(`run_geom.py::stage1`, `units.append({"extractor", "pair", "seed", "h"})`,
`if len(units) != 576`), and `H* = max(h_rand)` over that population.
`domain_audit_20260912T013530Z/design.json` section `random_gate` carries the
rule forward verbatim: *"G-STOP if max(h_rand) == 6 ... else H* := max(h_rand),
unaltered"*, with the same 32 seeds, the same six pairs and the same three
checkpoints listed under section `unchanged`.

**Why the maximum, and why it is the conservative direction.** `H*` is the
ceiling of the null population, not its mean. Requiring the *minimum* trained
unit to exceed the *maximum* of 576 random units is the strongest bar the design
admits, and it is the bar A.9 names ("the strongest admissible outcome requires
`min(h_trained) > H*`"). Per-checkpoint reporting under A.10 is by the
six-value vector of that checkpoint's per-pair `h`, never by pooling.

**Ambiguity that remains, declared not hidden.** A.9's phrase
"`min(h_trained)` across all three trained checkpoints" is satisfied by this
reading (the minimum over all trained units is the minimum over all checkpoints
and pairs). A weaker reading — minimum over three per-checkpoint summaries —
would require a per-checkpoint pooling rule that A.10 forbids. The stronger
reading is adopted. Both the 18 unit values and the three six-vectors are
reported, so a reader can apply either.

---

## B.3 — R3 · which axis the gate runs on

**The gap.** A.7 lists what Stage 1 holds frozen ("geometry") without naming
the axis. A.12 keeps the vertical perturbation but forbids it from carrying the
primary result.

**Resolution — INHERITED, unchanged.**

```text
Stage 1 (gate)     : HORIZONTAL only.       H* is a horizontal quantity.
Stage 2 (trained)  : HORIZONTAL (primary) + VERTICAL (diagnostic).
```

The vertical arm is never compared to `H*`, never enters `h_trained`, and never
changes the verdict. It can only trigger the A.12 downgrade.

**Authority.** GEOM-001 `run_geom.py::stage1` builds volumes for
`"horizontal"` only; `stage2` loops `("horizontal", "vertical")`.
`domain_audit_.../design.json` section `vertical_control` reclassifies the
vertical arm as *"a CONTAMINATED NULL / downgrade-trigger only"*.

**Why a vertical gate would be meaningless anyway.** The vertical arm has no
halo-clean region and cannot be given one: the crop is 71 feature cells wide but
only 17 tall against a 15-cell halo, and a non-empty vertical band needs
`FH >= 31` cells `= 496` px, exceeding the 368 px KITTI frame
(`domain_derivation.md` section 7.1, **DERIVED**). A null calibrated on a
contaminated arm would not bound the clean arm.

---

## B.4 — R4 · the search-free control is NOT run

**The gap.** The GEOM-001 design carried a fourth checkpoint, `NEG_shift_none`
(a `shift="none"` model, whose cost volume is k-constant), as a search-free
control. A.8 does not mention it.

**Resolution — EXCLUDED.**

A.8 says *"Load exactly these frozen H2 checkpoints: H2 seed0, H2 seed1,
H2 seed2"* and *"No checkpoint selection."* Loading a fourth checkpoint would
violate "exactly these". The control is therefore **not executed in this
record**, and no statement about it is made here.

**What this costs, stated plainly.** `NEG_shift_none` was the arm that would
have shown what the statistic does when the architecture provably cannot search
(EXP-010: every disparity slice identical). Its absence means this record
carries **one** null — the random-aggregation gate — and no architecture-level
null. That is a real reduction in discriminating power and it is a consequence
of the authorisation, not a judgement made here. It is recorded in
`results.json` under `controls_not_run`.

---

## B.5 — R5 · how HS1 is actually tested

**The gap.** A.3 states two conditions and A.14 names one hard stop. The two
conditions live on different domains (A.2 gives four different bands) and one of
them has no vertical analogue.

**Resolution — the single HS1 is tested as four named sub-checks**, all of which
halt the process, none of which is new:

| id | condition | domain | arms |
|---|---|---|---|
| **HS1-A** | `right[:, :CROP_W-t] == left[:, t:]` element-wise, from the actual arrays | whole crop | horizontal |
|  | `right[:CROP_H-t, :] == left[t:, :]` | whole crop | vertical |
| **HS1-B** | `R_f[w] == L_f[w+tau]` **exactly** (bit-equality, tolerance 0) | `w in [15, 55-tau]` | horizontal |
| **HS1-C** | `max abs V[:, tau, :, w] == 0.0` **exactly** for cells AA and BB | `w in [15, 45)` | horizontal |
| **HS1-D** | statistic domain non-empty, tau-independent, and at least `D` wide | static | both |

**Vertical:** HS1-B and HS1-C are **INAPPLICABLE** and are recorded as such, not
as passes. There is no halo-clean vertical band (B.3), so neither condition can
hold and neither is asserted. HS1-A is asserted for the vertical arm. This is
exactly the status A.12 assigns the vertical arm.

**Authority.** The four bands are `domain_derivation.md` sections 2-6
(**DERIVED** and **MEASURED**: 216/216 records for HS1-B's band; 3 525 120 cells
all exactly zero for HS1-C's band). GEOM-001 halted because its HS1 was written
on `w in [0, 54)`, which includes the 15-cell left halo; the corrected band
starts at `w = 15`.

**Tolerance is exactly zero, deliberately.** HS1-B and HS1-C are *structural*
identities on the clean band, not approximations — the measurement records
bit-exact equality. A tolerance here would be the same error GEOM-001 made in
the other direction.

---

## B.6 — R6 · the cancellation tolerance (HS4)

**Resolution — INHERITED, unchanged.** Relative tolerance `1e-5` on
`max|V_AA + V_BB - V_AB - V_BA| / max|V_AB|`.

**Authority.** GEOM-001 `run_geom.py`, `CANCEL_TOL = 1e-5`. Historical
validation references, both **MEASURED**: real features `1.8089e-07`
(`domain_audit_.../validate_domain.json`, 108 checks); synthetic over the entire
tensor including halo and fill `1.391e-07`. The frozen bound sits about 55x
above the observed real-feature residual.

Per A.13 this is asserted at the **raw cost-volume level only**. No claim is
made, and no check is written, that the cancellation survives the nonlinear
z-score — that is the quantity the gate calibrates.

---

## B.7 — R7 · ties, margins, rows

* **Ties.** `numpy.argmin` returns the smallest index on an exact tie, which is
  the rule A.5 specifies. Every tie is detected separately (more than one `k`
  attaining `Ibar.min()`) and recorded in `tie_events`; no tie is silently
  resolved.
* **Margin.** `I_min = min_k Ibar(k)`; `I_second = min over k != k_hat of
  Ibar(k)`; `margin = I_second - I_min`. Recorded for every unit and every tau.
  **Descriptive only** — it appears in no verdict rule anywhere in this record.
* **Rows.** `y in [64, 208)` is retained **unchanged from the frozen
  classification spec**, although `domain_derivation.md` section 7 shows all 17
  feature rows are admissible for the horizontal arm. Widening it after the
  GEOM-001 halt would be support-shopping; the inherited border only removes
  support and can never admit an invalid location. `domain_audit_.../design.json`
  section `unresolved_questions` records the same decision and the same reason.

---

# C. WHAT THIS RECORD CANNOT DECIDE, STATED BEFORE EXECUTION

| question | status |
|---|---|
| whether ~220 feature cells of support yield a stable argmin | **UNKNOWN** — this is what Stage 1 measures; it cannot be known before it runs |
| whether a learned non-correspondence shortcut could place its minimum at `k = tau` | **UNKNOWN** — not addressable by any ablation of this architecture on synthetic input |
| whether any of this transfers to natural stereo pairs | **UNKNOWN** — out of scope by A.16 |
| whether three checkpoints trained from one recipe are independent | **NO** — irreducibly reused; the designer has read eight prior response curves. Ceiling: consistency evidence at best |
| what the architecture does when it provably cannot search | **NOT MEASURED HERE** — the search-free control is excluded by A.8 (B.4) |

---

# D. FREEZE

The following are hashed into `frozen.sha256` before the first model is
instantiated, and re-verified at the start of every stage:

```text
PREREGISTRATION.md      this document
preregistration.json    machine-readable form of Part A + Part B
design.json             the domain derivation and the frozen specification
run_geom2.py            the harness
freeze_geom2.py         the freeze tool
```

`historical.sha256` covers every prior record and source file this experiment
depends on — including the whole of GEOM-001 and the domain audit — and is
verified before **and** after every stage (HS8).

Any hard stop RAISES and the process exits non-zero. Nothing is repaired and
continued.
