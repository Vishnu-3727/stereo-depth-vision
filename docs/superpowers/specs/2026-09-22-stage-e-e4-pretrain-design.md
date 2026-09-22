# E4 — broader pretraining: design and pre-registration

Status: **DESIGN ONLY. NO TRAINING AUTHORIZED.** Written 2026-09-22, before any
E4 run, before any pretrain, and before any number from a candidate is seen.

Authority for everything it does not restate:
`docs/superpowers/specs/2026-09-21-stage-e-recipe-design.md` and
`stage_e_recipe/PREREGISTRATION.md`. Data facts:
`stage_e_recipe/E4_DATA_AVAILABILITY.md`.

---

## 1 What E4 is

E4 is the last lever of the Stage-E ladder and the only one that changes what
the model is trained on rather than how. Every earlier experiment (E0 control,
E1 EMA, E2 batch 8, E3 400 epochs) started from the **same** Stage-1 init:
ARM-P pretrained on the FlyingThings3D **TRAIN A+C** subset, 14,460 triplets,
20 epochs (`stage_b_armp/20260918T062146Z_stage1_pretrain/`).

E4 replaces that init with one pretrained on a **larger corpus**, then runs the
**byte-identical E0 finetune recipe** on top of it.

The reason to try it is measured, not hoped for. Stage B's tier-2 experiment
compared a random-init arm against the Stage-1-pretrained arm under one recipe:

| run | random init | FT3D-pretrained init | delta |
|---|---|---|---|
| seed A | 1.5587716 | 1.2057590 | -0.3530126 |
| seed B | 1.4440242 | 1.1912168 | -0.2528074 |
| seed C | 1.4932978 | 1.1996447 | -0.2936531 |
| env control | 1.4178492 | 1.1872901 | -0.2305591 |

Pretraining is worth **0.23 to 0.35 px**. The entire recipe campaign — four
experiments, twelve training runs — bought **0.0250 px**. That is the case for
spending a lever here.

## 2 What E4 is not

- It is **not** evidence that more pretraining helps more. The table above
  measures *pretraining versus none*. Extrapolating it to *more versus some* is
  an assumption, and a 397,954-parameter model may already be saturated by
  14,460 triplets. E4 exists to measure that, and a null result is a real
  result.
- It is **not** a Hailo or INT8 result. The INT8 gate is a limb of the
  acceptance rule, not a survivability claim; it has certified nothing three
  times already (`stage_e_recipe/INT8_CONTROL_REPORT.md` §4).
- It is **not** combinable with E3. E3's 400-epoch finetune won, and folding it
  in here would be two levers at once. E4 uses 200 epochs. A "400 epochs on the
  better init" run is a separate experiment with its own pre-registration.

## 3 The corpus, and the confounds it carries

E4 will train on the public Kaggle mirror of the official
**FlyingThings3D_subset**:

| dataset | role | measured |
|---|---|---|
| `arjun12367/sceneflow-flyingthings-images` | left/right frames | 21,818 train + 4,248 val per eye, exact symmetry |
| `arjun12367/sceneflow-flyingthings-disparity` | disparity PFMs | complete to the last train index; 540x960, single-channel, 100 % finite |

Verified, not assumed: the two uploads correspond. Shifting the right image by
`-|d|` cuts the mean absolute intensity difference 3.4x to 4.8x below the
unshifted control on three sampled indices, while the opposite sign is worse
than the control. Full numbers in `stage_e_recipe/E4_DATA_AVAILABILITY.md` §4.

**Three confounds are unavoidable and are declared here, before the run:**

1. **Corpus identity changes with corpus size.** FlyingThings3D_subset
   (21,818 train frames) is not a superset of TRAIN A+C (14,460 triplets) — it
   is a different, officially filtered slice. A gain therefore cannot be
   attributed to volume alone. The honest lever name is "a broader, differently
   filtered pretrain corpus", not "more data".
2. **The sign convention changes.** These PFMs store disparity **negative**;
   the A/B/C PFMs Stage 1 used store it positive. `ft3d.py` applies no sign
   handling and the training mask is `gt > 0 AND gt < 184`, so the corpus must
   be read as `|d|`. Applied without the fix it yields **zero valid pixels** —
   a run that looks successful and teaches nothing.
3. **Provenance is third-party.** These are a re-upload by a Kaggle user, not
   the official download, and our own archives are the A/B/C layout, so there is
   no checksum path to verify them against. The sampling in §4 of the data file
   is the only integrity evidence and it is a sample, not a census.

## 4 Design

One pretrain produces one init; three finetune seeds are scored. This mirrors
E0 exactly, where three seeds shared one Stage-1 init.

| stage | what | fixed by |
|---|---|---|
| pretrain | ARM-P on FlyingThings3D_subset, `\|d\|` sign fix, epochs set by §6 | this spec |
| init | the pretrain's best-by-pretrain-val checkpoint, sha256 recorded | this spec |
| finetune | **byte-identical to E0**: batch 2, 200 epochs, `T_max` 200, KITTI `hailo_calib` only | `PREREGISTRATION.md` |
| scoring | frozen 40-scene contract via `frozen_eval` | unchanged |
| seeds | 0, 1, 2 | unchanged |

Architecture stays frozen at 397,954 parameters. The contract, the split
boundary and the 10-scene monitor are untouched.

## 5 Acceptance — no new control is needed

E4 changes **only the init**. The finetune recipe, the seeds, the contract and
the environment are E0's, so **E0 remains the valid control and its
instantiated constants apply unchanged**:

```
M0_best  = 1.2037841 px      S0_best  = 0.0174488 px
min(control best) = 1.1962889 px
INT8 gate: P_candidate <= 5.6365268 px
```

| verdict | condition |
|---|---|
| VOID | fewer than 3 completed seeds, or the INT8 gate not measured |
| REJECT | `Δ_best <= 0`, or the INT8 gate fails |
| INCONCLUSIVE | `0 < Δ_best < 0.0174488` |
| ACCEPT | `Δ_best >= 0.0174488` |
| ACCEPT (non-overlapping) | additionally `max(E4 best seeds) < 1.1962889` |

**The E3-only `Δ_final` requirement does not apply to E4**, and this is fixed
now rather than after seeing results. That rule existed because 400 epochs give
the 10-scene monitor 81 selection opportunities against the control's 41, making
"best" a softer claim. E4 runs 200 epochs, the same 41 selections as the
control, so the asymmetry is absent. `verdict.py` already keys that clause on
`exp == "e3"` and needs no change.

`Δ_final` will still be **recorded** for E4, because E3 showed final beating
best on every seed. Recording it is not scoring against it.

## 6 Budget, and the rate gate that must run first

Kaggle caps a session at ~12 h, and the account has a weekly GPU quota. The
pretrain is the expensive part and its cost is **not yet measured on a T4**.

Projection, explicitly labelled as such: E0 measured 23.2 s/epoch for 160 KITTI
pairs at 368x1232 on a T4, i.e. ~0.145 s per pair-epoch. FT3D_subset frames are
540x960, 1.14x the pixels, and there are 21,818 of them, which projects to
**~3,600 s/epoch**, so ~10 epochs fills ~10 h and 20 epochs cannot fit one
session.

**No epoch count is committed by this spec.** Before the pretrain is authorized,
a rate probe must run on Kaggle — the G7 pattern from
`stage_e_recipe/kaggle/GATE_RESULTS.md` — and measure the true s/epoch. The
epoch count is then chosen as the largest that fits inside a single session with
margin, and written into this file as an amendment **before** the pretrain runs.
If no useful epoch count fits one session, the options are resume-chaining
across sessions or a corpus subsample, and either one is an amendment here, not
an improvisation at run time.

## 7 Pre-run gates

All must pass, and all are read-only or cheap, before the pretrain is authorized:

1. **Sign fix present and effective**: first training batch reports a non-zero
   valid-pixel count. The existing `zero_valid_pixels` STOP covers the failure;
   the gate is to confirm the positive case, not merely the absence of a crash.
2. **Manifest built and counted**: the triplet manifest resolves image and
   disparity paths for every entry it lists, and its count is recorded.
3. **Rate probe** (§6) complete, with the epoch count amended in.
4. **Dataset versions pinned**: both Kaggle dataset refs recorded with the
   version actually attached, so a later re-upload by their owner cannot
   silently change the corpus.
5. **Account correct**: the resolved Kaggle owner printed and checked against
   the intended account before any push.
6. **KITTI subset and bundle present under the active account**, since the
   finetune stage needs them.

## 8 Stop conditions

Beyond the existing `run_arm.py` guards, E4 stops and reports rather than
adapting if: the pretrain's own validation EPE diverges or goes non-finite; the
manifest count differs from the announced 21,818; the attached dataset version
differs from the pinned one; or a finetune seed fails its contract guard.

Adapting the design mid-run to rescue a result is what the pre-registration
exists to prevent.

## 9 What a null result means

If E4 lands INCONCLUSIVE or REJECT, the conclusion is that **a broader
pretrain corpus does not help this model at this capacity** — which closes the
most promising remaining lever and makes the case that the 397,954-parameter
budget, not the data, is the binding constraint. That is a publishable finding
for this project's purposes and must be recorded as prominently as an
acceptance would be.
