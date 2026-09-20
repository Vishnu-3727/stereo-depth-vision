# Pre-freeze audit of `EXP-CORRESPONDENCE-ZONE-001`

**Id:** `correspondence_pipeline/prefreeze_audit_20260913T075420Z`
**Trigger:** step 1 of the authorised sequence — freeze the preregistration from
the measured hard-stop values.
**Outcome:** **HALTED before the freeze.** The preregistration was not written.
**Nothing was executed on trained weights.** No checkpoint was opened, no
training, no optimiser, no historical record modified. Stage 1 has not started.

**Claim ceiling unchanged: Level D — genuine geometric correspondence
NOT-DEMONSTRATED.**

The reported hard stops (`HS-SLACK`, `HS-GT`, `HS-FRAME1/2`, `HS-Z`, `HS-SYM`)
are not disputed and are not re-run here. The defect is in the **primary
statistic**, and it is visible only at the `disparity_initial` stage, which none
of the hard stops measured — every one of them stopped at the feature or
cost-volume level.

---

## 1. What was measured

Two receptive-field **support** questions at the `disparity_initial` stage.
Support properties are weight-independent, so randomly initialised weights are
admissible — the same justification `HS-Z` and `HS-BAND` used. Three seeds
(11, 12, 13), `cost_volume_shift="left"`, scene `000162_10.png`,
`Delta` in {1, 3, 6}, CPU, single thread, `use_deterministic_algorithms(True)`.

Scripts and raw output in this directory: `zi_support.py` / `zi_support.json`,
`zi_boundary.py` / `zi_boundary.json`.

| Check | Question | Result |
|---|---|---|
| **ZI-1** | Does `disparity_initial` in the **complement** interior change with Delta under the zone intervention? | **No. `max abs diff = 0.0` exactly**, all 3 seeds, all Delta, over `x` in `[912, 1136)`. |
| **ZI-2** | Does `disparity_initial` in the **zone** interior differ between the **zone** arm and the **global** arm at the same Delta? | **No. `max abs diff = 0.0` exactly**, all 3 seeds, all Delta, over `x` in `[0, 224)`. |

Both are bit-identity, not approximate agreement.

---

## 2. The finding — the primary statistic collapses onto a rejected statistic

`DESIGN.md` section 5.3 fixes the primary statistic as

    S = beta_in - beta_out ,   beta_R = d(disparity_initial)/d(Delta) over region R

and argues (5.3, 6) that `S` is a within-pass differential in which every global
nuisance cancels, and that an adversary using a global shift estimate `g_hat`
must satisfy `beta_in = beta_out`, hence `S = 0`.

The two measurements above settle both terms directly:

* **ZI-1 implies `beta_out` is exactly 0, for every model.** The complement
  interior's prediction is bit-identical across Delta, so its slope is not
  small — it is zero by construction. There is no nuisance left in `beta_out`
  for `beta_in` to cancel against. The differential structure is **vacuous, not
  protective**.
* **ZI-2 implies `beta_in(zone arm)` equals `beta_in(global arm)` exactly**, on
  the same pixels. A zone-interior pixel's entire dependency window lies inside
  the zone, so its input is bit-identical to the input it would receive under a
  *uniform* shift of the whole right crop.

Together:

    S  ==  beta_in(zone arm)  ==  beta_global restricted to the zone interior

So the design's primary statistic is **numerically identical to the slope of
`disparity_initial` against a global right-image translation, measured over a
region.**

That is the statistic `DESIGN.md` section 4 rejects: *"Global right-image
translation, slope of d-hat vs shift — correspondence-discriminating? **NO** …
REJECT as evidence. Retain only as an internal positive control."* It is also in
the section 12 prohibited list. The zone construction does not change what the
treatment arm measures; it only changes which pixels it is measured over.

This is not an execution risk that a control could catch. The statistic cannot
carry the interpretation section 11 assigns to it, so freezing it would
preregister a result that is uninterpretable whichever way it comes out.

---

## 3. Why the section 6 identification argument does not hold

Section 6 turns on `A4`'s shift estimate being *"a single global scalar"*, so
that the zone intervention forces the adversary to become spatially varying, and
a spatially varying offset estimator is inside `H_CORR`.

StereoNet as deployed is **fully convolutional with no pooling anywhere**. Its
dependency window at `disparity_initial` is bounded:

| Stage | Reach |
|---|---|
| feature extractor | 477 px total, radius **238 px** (4x `Conv 5x5 s2` gives 61; 6 `ResBlock` = 12x `Conv 3x3` at stride 16 gives +384; `output_conv 3x3` gives +32) |
| aggregation | 5x `Conv3d 3x3x3` = **5 cells = 80 px** |
| bilinear upsample (`align_corners=True`) | at most **1 cell = 16 px** |
| **total** | **about 573 px window** |

An adversary whose `g_hat` is a statistic of the *whole image* is therefore **not
realisable in this network at all**. `A4`'s `g_hat` was already necessarily a
function of a 573 px window before any intervention was applied. Imposing two
offsets in one image at the half-image scale does not force an upgrade the
architecture had already forced.

The measured consequence is exactly what ZI-2 shows: the network cannot
distinguish the zone arm from the global arm at a zone-interior pixel, because
it has no pathway by which the complement could reach that pixel.

**Consequence for `HS-SYM`.** `HS_RESULT.md` already recorded, correctly, that
`HS-SYM`'s fixed-`w` scan does not by itself eliminate the combined
candidate-plus-spatial reparametrisation, and rested the design instead on two
stronger facts: that the intervention leaves the complement bit-identical, and
that `S` is a within-pass differential. The first fact is confirmed here
(ZI-1) — but it is the *reason* the second fact is empty. The two supports were
not independent; they were the same fact, and it does not support the design.

---

## 4. What survives, and what it would cost

The **swap control** is the one arm in the design whose input **cannot** be
produced by translating the right crop of a real pair: its zone content comes
from a different scene. It is therefore not reducible to the rejected global
statistic, and it is the only arm that destroys left-right correspondence while
preserving the translation of right-image content by `16*Delta`.

That points at a different primary statistic, on identical pixel sets, identical
Delta set, identical (fixed) left image:

    S_star = beta_in(treatment) - beta_in(swap)

* `H_CORR` predicts `beta_in(treatment) = +1`, `beta_in(swap)` about 0, so
  `S_star` about +1.
* A mechanism that reads Delta from right-image content alone produces the same
  slope in both arms, so `S_star` about 0. This is the adversary the *zone* was
  mistakenly credited with killing; the **swap** is what kills it.
* A constant difference in content statistics between donor and own content
  cancels, because `S_star` differences two **slopes**, not two levels. What
  does not cancel is a Delta-dependent content effect whose *magnitude* differs
  between donor and own content. That residual must be declared, not argued
  away.

Under `S_star` the zone's remaining function is modest and should be stated as
such: it keeps the complement veridical so the model is not evaluated far
off-distribution, and it matches zone-level statistics between arms. **It is no
longer a locality argument, and no claim of locality can be made from it.**

Adopting `S_star` replaces the design's primary statistic **and** its
identification argument. That is a design amendment, not a preregistration
detail, so it is not taken here.

**Sign, carried forward.** Under the amendment already on record
(`x_R = x_L + 16*Delta`, `d' = d + 16*Delta`), `H_CORR` predicts
`beta_in = +1` and `HS-POSCTRL` predicts `beta_global` about `+1`. `DESIGN.md`
sections 5.3 and 7 still say `-1`, from the superseded offset direction. Any
future preregistration must state `+1`.

---

## 5. Secondary finding — the 318 px exclusion is marginally insufficient

Measured per column, the bit-exact clean bounds at the `disparity_initial` stage
are `x <= 243-245` (zone side) and `x >= 907-908` (complement side), varying by
one or two pixels with seed. `DESIGN.md`'s interiors, `[0, 250)` and
`[886, 1136)`, therefore include contaminated columns.

**The leak is 4.8e-07 to 9.5e-07**, i.e. 1-2 float32 ULPs at a prediction
magnitude of about 4 candidates. It is a bookkeeping error, **not a confound**,
and it is reported so it is not rediscovered later as if it were one. Bit-exact
interiors, with one feature cell of margin, are `x` in `[0, 224)` and
`x` in `[912, 1136)`; those are the ranges used for ZI-1 and ZI-2 above.

---

## 6. State

* Preregistration: **not written**. Step 1 halted.
* Stage 1 (`NEG_shift_none`, swap control): **not started**. No trained
  checkpoint has been opened in this branch.
* Gate threshold: **not derived** — deriving it from null spreads would have
  frozen a threshold for a statistic that cannot be interpreted.
* `shift="left"` treatment results: **not inspected**, and no arm of the
  experiment has been run on them.
* Ceiling: **Level D**.

Decision required from the operator before any execution: either amend
`EXP-CORRESPONDENCE-ZONE-001` to the swap-referenced statistic `S_star` with a
rewritten identification argument, or close the pipeline branch as not
identifiable with the available interventions.
