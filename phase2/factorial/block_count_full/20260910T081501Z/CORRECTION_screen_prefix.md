# CORRECTION — `SCREEN_PREFIX` reference constants, 2026-09-10

**Status: disclosed, not hidden. Applied before `verify_prefix` ran, while
seed 1 was still training. No threshold, gate, decision boundary or verdict is
affected.**

## What was wrong

`exp_blockcount_batch3.py` freezes the 30-epoch seed-screen prefix for 4 blocks
at seeds 1 and 2, so the new 200-epoch runs can be checked against it by exact
float equality (`PREREGISTRATION.md` §7).

The validation values first written into `SCREEN_PREFIX` were transcribed from a
listing printed at **10 decimal places**, and their trailing digits beyond that
were wrong. Example, seed 1, epoch 2:

```
frozen (wrong)         val_epe = 18.945652640925597
authoritative record   val_epe = 18.94565264092234
```

The error is ~3e-12 absolute — invisible at the printed precision, fatal to an
exact-equality check.

## How it was found

While seed 1 was training, its `checkpoint_epoch_2` metric was read from the
live log:

```
val_epe = 18.94565264092234   val_d1 = 87.00300963707065
```

That value matches the **authoritative screen record** exactly and disagrees
with the frozen constant. The discrepancy was therefore in the constant, not in
the run: the deterministic protocol was holding.

## What was done

The constants were replaced with `repr()` of the authoritative source records,

```
phase2/factorial/block_count_seed_screen/20260909T145659Z/results/
    history_arm_C-SEED1.json
    history_arm_C-SEED2.json
```

read read-only. The screen records themselves were **not modified**.

| seed | epoch | corrected val_epe | corrected val_d1 |
| ---: | ----: | ----------------: | ---------------: |
| 1 | 1 | 16.031864940965352 | 89.06086644453445 |
| 1 | 2 | 18.94565264092234 | 87.00300963707065 |
| 1 | 10 | 18.36530086611511 | 88.2830455679038 |
| 1 | 20 | 12.078822974549746 | 78.73558710487097 |
| 2 | 1 | 21.45397410346398 | 89.77658199353804 |
| 2 | 2 | 17.84664041672191 | 89.12050045289612 |
| 2 | 10 | 16.342920881565185 | 87.35554870271855 |
| 2 | 20 | 15.506311573290349 | 85.64163506780548 |

## Why this is not a post-hoc threshold change

- `SCREEN_PREFIX` is **not** a threshold, gate or decision rule. It feeds exactly
  two things: the preflight initialisation-hash check, and `verify_prefix`.
- The **initialisation SHA strings were always correct** (`2cac2916ed802495`,
  `e622226a5a22fc9b`) and passed preflight before training started. They are
  unchanged.
- The correction makes the check **stricter and correct**, not looser: it
  replaces values that could never be matched by any run with the true values.
- The pre-registered decision boundary — the frozen 6-block EPE range
  `[2.2955181809208267, 2.3982217171269924]` — is a separate constant, was read
  back from the frozen records and asserted at preflight, and is **untouched**.
- Nothing about the training configuration, seeds, recipe, epoch budget, gates
  or verdict rules changed. No result had been produced when the correction was
  made.

## Residual limitation

Because the correction was made while seed 1 was mid-flight, the epoch 1 / 2 /
10 / 20 expectations for **seed 1** were adjusted after that run's epoch-2 value
had been seen in the log. The adjustment set them equal to the independent
screen record, not to the live run — but the honest statement is that for seed 1
this specific comparison is *confirmatory* rather than blind. For **seed 2** the
constants were corrected before that run started, so its prefix check is blind.
Both are reported in `prefix_verification.json`.
