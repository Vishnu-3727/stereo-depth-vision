# EXP-E3-REFINEMENT-ABLATION-001 — pre-registration, frozen before the run

**Status: pre-registration.** Written and frozen before the experiment was
executed and before any ablated accuracy number existed.

**Claim tags:** `MEASURED`, `DERIVED`, `INFERRED`, `UNKNOWN`.

---

## 1. Question

`MEASURED` (Phase 1, EXP-001/013/018) — the refinement stage is **90.6 % of the
model's MACs** and **73.3 % of its GPU time**, and 90.8 % of activation traffic.

`MEASURED` (`EXP-E1-ERROR-PARTITION-003-HELDOUT-ORACLE`) — a perfect matching
input removes only 29.3 / 41.5 / 32.9 % of H2's pooled EPE, so **58–71 % of the
error is imposed downstream**, by these refinement weights.

Both facts point at the same stage. The question this run asks is the cheap half
of it:

> With the matching path working, how much of the six-block refinement stack does
> H2 actually need, and what does removing the rest cost?

`UNKNOWN`, and outside this run — what a refinement stack **retrained** at a
smaller size would do. This is an inference-time ablation of trained weights and
is therefore a **lower bound** on how small a retrained stage could be.

## 2. Design

**No training.** The three trained H2 seeds (`core.MODELS` keys `H2`, `SEED1`,
`SEED2`), all 40 `hailo_val` scenes, full 368×1232 frames, pooled over `gt > 0` —
the same recorded pooled protocol and the same `_forward_parts` / `_finish`
harness E1 used.

The refinement stage is `input_conv → 6 × ResBlock(dilations 1,2,4,8,1,1) →
output_conv`, and each block is `out = act(conv2(act(conv1(x))) + x)`. Because
the blocks are residual and shape-preserving, a subset of them is a valid
network with the remaining weights untouched: removing block *i* replaces
`act(f_i(x) + x)` with `x`.

### Variants (14 per model)

| group | variants |
|---|---|
| baseline | the full 6-block stack |
| single drops | drop block *i*, for *i* = 0…5 (dilations 1, 2, 4, 8, 1, 1) |
| prefixes | keep the first *k* blocks, for *k* = 0…5 |
| ladder | drop blocks 1, 2, 3 together (the dilation ladder 2/4/8), keeping 1, 1, 1 |

"keep 5" and "drop block 5" are the same network by construction; both are
computed and their agreement is a gate.

### Measured per variant

- pooled EPE and D1 over the 40 scenes (`gt > 0`);
- MACs at 368×1232 (the deployment resolution behind Phase 1's 56.04 GMAC) and at
  256×512 (comparable with the replication's 16.1995 GMAC), via `thop`, on a
  freshly constructed model;
- for the baseline and every prefix variant, the **right-image corruption probe**
  (`stereo_probe`, imported unchanged, four focus scenes 27/0/31/6), because
  constraint **C2** of the evidence review requires ≥ +20 D1 points of
  right-image dependence and an ablation that quietly returns the model to
  monocular behaviour must be caught, not rewarded.

## 3. Gates that must pass before any verdict is read

1. **Harness fidelity** — the baseline variant reproduces
   `EXP-E1-ERROR-PARTITION-001`'s recorded as-trained pooled figures (H2 2.400 px
   / 16.72 %, SEED1 2.319 / 17.25, SEED2 2.556 / 19.76) to within 0.01 px and
   0.05 points.
2. **Internal consistency** — "keep 5" equals "drop block 5" to within 1e-9 px.
3. **Finiteness** — no NaN or Inf in any variant's output.
4. **Checkpoint integrity** — every checkpoint SHA-256 unchanged after the run.

If gate 1 or 2 fails the run is void and reports no verdict.

## 4. Acceptance bands — frozen

Relative to the same model's full stack, on **all three seeds**:

| band | condition |
|---|---|
| **NEGLIGIBLE** | ΔD1 ≤ +1.0 point **and** ΔEPE ≤ +0.10 px |
| **TOLERABLE** | ΔD1 ≤ +3.0 points **and** ΔEPE ≤ +0.30 px |
| **COSTLY** | anything else |

Where the numbers come from, stated so they cannot be read as tuned: +1.0 D1
point is below the 3.04-point spread the three seeds already show among
themselves as trained, and ~1/56 of the H2-vs-H1 effect; +3.0 points is that
seed spread. Both were fixed before any ablated number existed.

A variant is **VIABLE** only if it is NEGLIGIBLE or TOLERABLE **and** (where the
stereo probe is run) still meets C2's ≥ +20 D1 points of right-image dependence.

## 5. Headline endpoints — defined before the results exist

1. The largest **MAC saving** achieved by a NEGLIGIBLE variant, and by a
   TOLERABLE one.
2. Whether any single block is free to remove, and which.
3. Whether the dilation ladder (2/4/8) earns its place.

## 6. Verdict rule

- **REFINEMENT IS OVERSIZED** — some VIABLE variant removes ≥ 25 % of whole-model
  MACs.
- **REFINEMENT IS RIGHT-SIZED** — no VIABLE variant removes ≥ 10 % of whole-model
  MACs.
- **PARTIALLY OVERSIZED** — the best VIABLE saving falls between 10 % and 25 %.

Thresholds fixed here, before the run. Whole-model MACs, not refinement-only
MACs, because the deployment number is the whole model.

## 7. What may not change

No weight, checkpoint, scene set, protocol, band or threshold in this document
may change once the run starts. The record is
`EXP-E3-REFINEMENT-ABLATION-001`; a rerun takes a new ID.

## 8. Limitations, stated in advance

- Inference-time ablation of weights trained as a 6-block stack: a **lower
  bound** on how small a retrained stage could be, never an upper bound.
- Three seeds, 40 scenes, one budget. No significance test, none claimed.
- MACs are not latency — Phase 1 measured that MACs mispredict latency on this
  stack by 0.67×–142×. Any latency claim needs a measurement, and this run makes
  none.
- The stereo probe runs on four focus scenes chosen in earlier work, not sampled
  at random.
