# EXP-BLOCKCOUNT-DETERMINISTIC-001 — RESULTS

Protocol frozen in advance: `PREREGISTRATION.md`. Two treatment arms, one
execution each, no tuning, no rerun. Six-block control frozen, not retrained.
Nothing historical was modified.

Tags: `MEASURED` · `HISTORICAL` (previously recorded, cited unchanged) ·
`DERIVED` · `INFERRED` · `UNKNOWN`.

---

## QUESTION

How much of the six-block refinement capacity is necessary, now that the stereo
mechanism is known to be working (readout necessary for learning; shift necessary
for useful correspondence)?

## CONTROL — frozen

`STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA`, six blocks, `shift=left`,
standardised readout, seed 0, 200 epochs, deterministic protocol.
**Not retrained** — Stage B measured a fresh-process rerun of this configuration
to be bit-identical, so retraining could only reproduce it. Read-only here.

## TREATMENTS

Arms B (5 blocks, dilations 1,2,4,8,1) and C (4 blocks, dilations 1,2,4,8),
trailing dilation-1 blocks removed and the ladder kept intact — `HISTORICAL`
E3b's own frozen choice, reused verbatim.

`MEASURED` (preflight) — every surviving weight tensor is **bit-identical** to the
control's; parameter counts match the E3b registration exactly; `shift=left`,
12 candidates and the standardised readout are unchanged in both arms. All ten
preflight checks passed.

## TRAINING

`MEASURED` — deterministic protocol
(`use_deterministic_algorithms(True)`, `cudnn.deterministic=True`,
`cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`); no determinism error
raised or suppressed in either arm.

| | control (6) | arm B (5) | arm C (4) |
|---|---:|---:|---:|
| epochs / aborted | 200 / `null` | **200 / `null`** | **200 / `null`** |
| parameters | 423,586 | 405,090 | 386,594 |
| MACs @256×512 (measured) | 16.1995 G | 13.7836 G | 11.3677 G |
| `DERIVED` MAC saving | — | **14.91 %** | **29.83 %** |
| final train loss | 0.9037397 | 1.1846590 | 0.9392961 |
| gradient median / max | 21.9162 / 308.034 | 19.0011 / 317.137 | 20.6612 / 319.282 |
| NaN·Inf / batches > 1e4 | none / 0 | none / 0 | none / 0 |
| matching-path gradient | 100 % of 16,000 | 100 % of 16,000 | 100 % of 16,000 |
| wall clock | 4,948.997 s | **4,764.574 s** | **4,626.606 s** |
| final weight sha | `3ad382c50d413ad2` | `7b1b05716d8451ac` | `79b7daa568636434` |

Both arms trained cleanly. Neither is a failed or unstable run.

---

## FULL-40-SCENE RESULTS

`MEASURED` — all 40 `hailo_val` scenes, full 368×1232 frames, pooled over
`gt > 0` (3,802,797 px), existing evaluator unchanged.

| Metric | 6 blocks (control) | 5 blocks (B) | Δ vs control | 4 blocks (C) | Δ vs control |
|---|---:|---:|---:|---:|---:|
| **EPE** | **2.2955182** | **2.6457595** | **+0.3502413** | **2.3387143** | **+0.0431962** |
| **D1** | **15.6045143** | **21.3711907** | **+5.7666765** | **16.9797389** | **+1.3752246** |
| RMSE | 6.0437075 | 6.0530906 | +0.0093831 | 5.8646747 | **−0.1790328** |
| bad1 | 48.8446793 | 56.4567606 | +7.6120813 | 49.4621985 | +0.6175192 |
| bad2 | 25.1886966 | 32.5231402 | +7.3344436 | 26.7842328 | +1.5955362 |

Late window (10 points, 10-scene training-time protocol, cross-reference only):

| | control | arm B | Δ | arm C | Δ |
|---|---:|---:|---:|---:|---:|
| EPE | 3.8670192 ± 0.1113151 | 4.2930984 ± 0.1063132 | +0.4260793 | 3.8730704 ± 0.0583333 | **+0.0060513** |
| D1 | 23.4954125 ± 0.9342410 | 29.7537814 ± 0.8655503 | +6.2583689 | 25.4206454 ± 0.6297648 | +1.9252329 |

### The headline: the series is **not monotone in block count**

`MEASURED` — 40-scene D1 by block count: **15.6045 (6) → 21.3712 (5) → 16.9797 (4)**.
EPE: **2.2955 (6) → 2.6458 (5) → 2.3387 (4)**.

`DERIVED` — **five blocks is worse than four blocks**, on both metrics, by
+0.307 px and +4.39 D1 points. If refinement capacity were the dominant variable,
four blocks would be worse than five. It is not, and the gap is large.

---

## PER-SCENE RESULTS

`MEASURED`:

| | arm B (5 blocks) | arm C (4 blocks) |
|---|---|---|
| scenes worse on EPE | **35 / 40** | **22 / 40** |
| scenes worse on D1 | **38 / 40** | **29 / 40** |
| ΔEPE mean / median / sd | +0.3465 / +0.3311 / 0.3912 | +0.0444 / +0.0193 / 0.2985 |
| ΔEPE min … max | −0.9728 … +1.4423 | −0.9642 … +0.7549 |
| ΔD1 mean / median / sd | +5.5190 / +4.1848 / 5.1100 | +1.3584 / +0.8600 / 2.6597 |
| ΔD1 min … max | −0.5590 … +27.4950 | −4.6386 … +7.7782 |

`DERIVED` — arm B's per-scene direction is **systematically adverse** (38 of 40 on
D1). Arm C's is **weak**: EPE is a near coin-flip (22/40) and D1 is mildly adverse
(29/40) with five scenes improving by more than 1 D1 point.

<details>
<summary>Full 40-scene table (EPE and D1 for all three configurations)</summary>

| scene | 6blk EPE | 5blk EPE | 4blk EPE | ΔB | ΔC | 6blk D1 | 5blk D1 | 4blk D1 | ΔB | ΔC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 000160_10 | 1.1599 | 1.4980 | 1.2893 | +0.3381 | +0.1294 | 8.4623 | 11.3063 | 8.6849 | +2.8441 | +0.2226 |
| 000161_10 | 10.6710 | 10.6366 | 10.3801 | −0.0345 | −0.2909 | 39.3894 | 42.1478 | 41.3385 | +2.7584 | +1.9491 |
| 000162_10 | 2.1094 | 2.2054 | 1.9948 | +0.0960 | −0.1147 | 18.0050 | 18.9409 | 16.4221 | +0.9358 | −1.5829 |
| 000163_10 | 2.2240 | 2.9152 | 2.8587 | +0.6912 | +0.6347 | 21.3468 | 32.3477 | 28.0549 | +11.0009 | +6.7081 |
| 000164_10 | 5.0929 | 5.9442 | 5.2354 | +0.8514 | +0.1425 | 26.1691 | 32.8016 | 30.6810 | +6.6325 | +4.5120 |
| 000165_10 | 2.6617 | 3.1821 | 2.6623 | +0.5204 | +0.0006 | 32.3245 | 33.7566 | 30.7486 | +1.4321 | −1.5758 |
| 000166_10 | 1.6868 | 2.1568 | 1.6638 | +0.4700 | −0.0230 | 14.5787 | 22.1412 | 13.4076 | +7.5625 | −1.1710 |
| 000167_10 | 1.9050 | 2.5236 | 1.8965 | +0.6186 | −0.0085 | 16.5400 | 23.1990 | 14.1362 | +6.6590 | −2.4038 |
| 000168_10 | 5.2002 | 4.9641 | 4.2360 | −0.2361 | −0.9642 | 26.6739 | 35.6650 | 32.9706 | +8.9912 | +6.2967 |
| 000169_10 | 4.5741 | 5.5948 | 5.2193 | +1.0208 | +0.6453 | 23.6780 | 37.5756 | 31.4562 | +13.8977 | +7.7782 |
| 000170_10 | 1.6279 | 1.9266 | 1.5363 | +0.2987 | −0.0916 | 15.1587 | 20.7262 | 10.5202 | +5.5674 | −4.6386 |
| 000171_10 | 1.0931 | 1.8451 | 1.3109 | +0.7519 | +0.2178 | 5.6767 | 16.7439 | 7.5299 | +11.0672 | +1.8532 |
| 000172_10 | 1.5682 | 2.1158 | 1.7706 | +0.5476 | +0.2024 | 10.5529 | 16.1144 | 14.2044 | +5.5615 | +3.6515 |
| 000173_10 | 3.0891 | 3.0139 | 3.0370 | −0.0753 | −0.0521 | 21.2422 | 25.1691 | 22.2742 | +3.9269 | +1.0320 |
| 000174_10 | 2.1330 | 2.4663 | 2.2180 | +0.3333 | +0.0850 | 20.1199 | 23.6909 | 22.2629 | +3.5709 | +2.1430 |
| 000175_10 | 2.4455 | 2.8360 | 2.4719 | +0.3905 | +0.0263 | 20.1341 | 25.9787 | 21.2911 | +5.8446 | +1.1570 |
| 000176_10 | 2.5154 | 3.1357 | 3.2704 | +0.6203 | +0.7549 | 23.5217 | 26.0036 | 28.0680 | +2.4819 | +4.5463 |
| 000177_10 | 3.2421 | 2.2693 | 2.6851 | −0.9728 | −0.5570 | 14.4195 | 16.1326 | 17.0792 | +1.7131 | +2.6598 |
| 000178_10 | 2.1029 | 2.3085 | 2.0538 | +0.2057 | −0.0490 | 16.4147 | 18.8458 | 17.2787 | +2.4311 | +0.8640 |
| 000179_10 | 2.1505 | 2.6899 | 2.1934 | +0.5393 | +0.0428 | 24.2079 | 28.4486 | 24.3626 | +4.2406 | +0.1546 |
| 000180_10 | 1.1617 | 1.3517 | 1.0455 | +0.1901 | −0.1161 | 6.7234 | 8.7276 | 5.7716 | +2.0042 | −0.9518 |
| 000181_10 | 1.4045 | 1.4892 | 1.3310 | +0.0847 | −0.0735 | 9.5326 | 11.7924 | 8.9721 | +2.2597 | −0.5605 |
| 000182_10 | 1.4197 | 1.9411 | 1.6174 | +0.5214 | +0.1977 | 10.9158 | 15.8352 | 12.2193 | +4.9194 | +1.3035 |
| 000183_10 | 1.1516 | 1.2820 | 1.1276 | +0.1305 | −0.0240 | 5.9315 | 8.2954 | 6.7160 | +2.3640 | +0.7846 |
| 000184_10 | 1.3324 | 1.7804 | 1.3447 | +0.4479 | +0.0123 | 8.1546 | 14.2532 | 8.6361 | +6.0986 | +0.4816 |
| 000185_10 | 1.1717 | 1.1712 | 1.0925 | −0.0005 | −0.0792 | 4.9777 | 7.8696 | 5.8338 | +2.8919 | +0.8561 |
| 000186_10 | 1.6087 | 2.1009 | 2.0273 | +0.4922 | +0.4186 | 11.6164 | 12.8592 | 13.0931 | +1.2428 | +1.4767 |
| 000187_10 | 1.3206 | 1.4143 | 1.2848 | +0.0937 | −0.0358 | 9.6030 | 9.0440 | 8.4759 | −0.5590 | −1.1271 |
| 000188_10 | 1.2007 | 1.2874 | 1.2482 | +0.0867 | +0.0475 | 9.0410 | 9.2856 | 8.8405 | +0.2446 | −0.2005 |
| 000189_10 | 1.6973 | 2.0262 | 1.7477 | +0.3289 | +0.0504 | 16.9839 | 21.6063 | 16.4002 | +4.6225 | −0.5837 |
| 000190_10 | 1.2473 | 1.4759 | 1.2419 | +0.2285 | −0.0055 | 9.7701 | 13.8990 | 9.8290 | +4.1290 | +0.0589 |
| 000191_10 | 1.2915 | 1.5327 | 1.3444 | +0.2412 | +0.0529 | 8.6610 | 11.9162 | 9.8185 | +3.2551 | +1.1574 |
| 000192_10 | 1.3548 | 1.4836 | 1.3871 | +0.1289 | +0.0323 | 10.9703 | 10.8198 | 11.3695 | −0.1506 | +0.3991 |
| 000193_10 | 3.2353 | 3.3744 | 3.3505 | +0.1391 | +0.1153 | 24.2126 | 31.8193 | 25.0386 | +7.6067 | +0.8260 |
| 000194_10 | 1.8389 | 2.3070 | 1.7851 | +0.4681 | −0.0538 | 11.7274 | 22.0432 | 15.5975 | +10.3158 | +3.8701 |
| 000195_10 | 1.3927 | 2.3490 | 1.8053 | +0.9563 | +0.4126 | 8.5950 | 23.0001 | 14.8572 | +14.4051 | +6.2622 |
| 000196_10 | 1.3911 | 1.8549 | 1.4901 | +0.4638 | +0.0990 | 10.7030 | 20.3506 | 11.9572 | +9.6476 | +1.2542 |
| 000197_10 | 2.1596 | 3.6019 | 2.5283 | +1.4423 | +0.3687 | 13.5201 | 41.0151 | 18.6731 | +27.4950 | +5.1530 |
| 000198_10 | 2.2702 | 2.5733 | 1.9704 | +0.3032 | −0.2998 | 17.5977 | 23.1346 | 16.5803 | +5.5368 | −1.0175 |
| 000199_10 | 1.4633 | 1.6031 | 1.3869 | +0.1398 | −0.0764 | 9.1691 | 12.4810 | 9.9067 | +3.3119 | +0.7376 |

</details>

---

## STEREO FUNCTIONALITY

`MEASURED` — `stereo_probe` imported unchanged, four focus scenes, epochs
10/50/100/150/200, on snapshots each run had already saved. The control was
probed from its frozen snapshots, **not retrained**. Worst case over variants.

| epoch | right-image ΔD1 6 / 5 / 4 | matching-map ΔD1 6 / 5 / 4 | `disparity_initial` r(GT) 6 / 5 / 4 |
|---:|---|---|---|
| 10 | +31.476 / **−3.079** / +18.976 | +33.439 / **−2.279** / +27.254 | +0.7548 / +0.4137 / +0.7403 |
| 50 | +58.072 / +29.838 / +64.792 | +47.845 / +28.722 / +53.152 | +0.8596 / +0.7828 / +0.8574 |
| 100 | +71.175 / +62.171 / +75.198 | +57.584 / +50.818 / +63.376 | +0.8865 / +0.8713 / +0.8888 |
| 150 | +80.043 / +63.695 / +76.991 | +65.027 / +53.711 / +65.982 | +0.9158 / +0.8925 / +0.9125 |
| 200 | +80.813 / +70.411 / **+79.251** | +66.188 / +58.676 / **+67.248** | +0.9168 / +0.9014 / **+0.9195** |

`MEASURED` — frozen epoch-200 verdict: **all three are `STEREO_FUNCTIONAL`**.
Arm B +70.411 / +58.676, entropy 1.7518, disparity std 17.783. Arm C +79.251 /
+67.248, entropy 1.7812, disparity std 18.305 — matching-map dependence and
r(GT) marginally **above** the six-block control's.

`MEASURED` — arm B is the only arm whose stereo dependence is still negative at
epoch 10 (−3.079); it is also the arm that lagged most in early training
(17.508 px at epoch 20 against the control's 7.332 and arm C's 6.940).

**Probe caveat, as pre-registered.** `HISTORICAL` — Phase 1 EXP-007 measured
+89.7 D1 right-image dependence on the *reference* weights despite a provably
degenerate shift, so passing these probes does not prove disparity search. Here
they are used only to confirm no arm became monocular. None did.

---

## INTERPRETATION

**`MEASURED`**

1. 40-scene: 6 blocks 2.2955 px / 15.6045 %; 5 blocks 2.6458 / 21.3712
   (+0.3502 / +5.7667); 4 blocks 2.3387 / 16.9797 (+0.0432 / +1.3752).
2. **Non-monotone**: 5 blocks is worse than 4 blocks by +0.307 px and +4.39 pt.
3. Arm B worse on 35/40 (EPE) and 38/40 (D1); arm C on 22/40 and 29/40.
4. All three `STEREO_FUNCTIONAL`; arm C's matching-map dependence (+67.248) and
   r(GT) (+0.9195) slightly exceed the control's.
5. Arm C's RMSE is **better** than the control's (−0.1790).
6. Measured MAC savings 14.91 % (5 blocks) and 29.83 % (4 blocks); wall clock
   fell 3.7 % and 6.5 %.
7. Both arms trained cleanly: 200/200 epochs, no abort, no NaN, zero gradient
   spikes, matching gradient on 100 % of 16,000 batches.
8. Surviving weights were bit-identical to the control's at initialisation.

**`DERIVED`**

9. **Block count is not the dominant explanatory variable at seed 0.** A
   capacity effect must be monotone in capacity; this series is not, and the
   non-monotonicity (4.39 D1 points between 5 and 4 blocks) is larger than the
   whole 6→4 difference (1.375). The variation is dominated by something that is
   not block count.
10. `HISTORICAL` — this reproduces E3's finding from the other direction: E3
    measured that *which* block is redundant differs by run (block 4 cost seed 0
    +0.19 px and seeds 1 and 2 +24.16 and +16.37). E3 called it per-run
    redundancy; the same signature appears here as per-arm trajectory
    idiosyncrasy under a protocol with **zero execution noise**.
11. **The deterministic protocol worked exactly as designed, and that is what
    exposes the real limit.** Execution noise is now provably zero, so every
    number above is exact and reproducible. What remains is **seed noise**, which
    Stage B did not and could not remove — and it is now visibly the binding
    constraint on this question.
12. Arm B's Δ is large against every reproducibility scale
    (`HISTORICAL` O6 1.9701 D1; Stage B protocol divergence 1.4143 D1;
    controlled same-config variance 0.0) and its per-scene direction is
    consistent. Arm C's ΔD1 (+1.3752) is **below** both non-zero scales, and its
    ΔEPE (+0.0432) is an order of magnitude below them.

**`INFERRED`**

13. Arm B's deficit looks like a slow-start trajectory that never fully closed:
    it was 10.2 px behind at epoch 20, closed to 0.09 px by epoch 80, then
    drifted back out. Not established — one run cannot separate that from a real
    5-block penalty.

**`UNKNOWN`**

14. **Seed-to-seed variation of every difference above.** One seed per arm. Not
    measured, no significance claimed, none claimable.
15. Whether four blocks would still track the control at another seed, and
    whether five blocks would still be the worst of the three. Given §9, this is
    the decisive missing measurement.
16. Three blocks and below under this protocol.
17. Latency. `HISTORICAL` Phase 1 measured MAC-to-latency misprediction of
    0.67×–142×; the MAC savings above are arithmetic, and the measured training
    wall-clock savings (3.7 %, 6.5 %) are already far below the MAC savings
    (14.9 %, 29.8 %).
18. Transfer to a pretrained model. 160 scenes from random initialisation.

---

## VERDICT

| arm | verdict |
|---|---|
| **B — 5 blocks** | **`CAPACITY-REQUIRED-AT-SEED-0`** |
| **C — 4 blocks** | **`INCONCLUSIVE`** |
| **series overall** | **`INCONCLUSIVE`** |

Arm B degrades on both metrics with a systematically adverse per-scene direction
(38/40 on D1) and a D1 magnitude 2.9–4.1× every non-zero reproducibility scale.
Its accuracy arm is unambiguous; note honestly that its **EPE** magnitude
(+0.3502) is only 1.0–1.6× those scales, so the D1 metric carries the verdict.

Arm C's magnitudes sit **inside** the reproducibility scales (ΔD1 +1.3752 below
both 1.4143 and 1.9701; ΔEPE +0.0432 far below), with a near-coin-flip per-scene
EPE direction. Under the frozen rule that is `INCONCLUSIVE`, not
`CAPACITY-REDUCIBLE-AT-SEED-0`.

**The series verdict is `INCONCLUSIVE`, and the reason is more informative than
either arm.** A capacity ordering must be monotone; this one is not. Five blocks
being clearly worse than four blocks means the measured differences are not
attributable to block count at one seed. **This experiment does not establish
that refinement capacity can be reduced, and it does not establish that it
cannot.**

---

## IMPACT ON PROJECT

**What this establishes**

- `DERIVED` — E3b's question is now measured with **zero execution noise** and on
  the authoritative 40-scene protocol. That was the stated goal of Stages A and B
  and it was achieved: the numbers are exact, reproducible and band-free.
- `DERIVED` — and the answer is that **the question cannot be settled at one
  seed**. The non-monotone ordering identifies the remaining obstacle
  unambiguously: seed/trajectory variation, not execution nondeterminism and not
  a bad materiality band.
- `DERIVED` — the historical E3b verdict `CAPACITY-REDUCIBLE` is **not**
  reproduced under the repaired protocol. `HISTORICAL` E3b measured 5 blocks as
  −0.2333 px / +0.8759 pt and 4 blocks as −0.0893 / +1.1610 on the 10-scene
  protocol; this experiment measures +0.4261 / +6.2584 and +0.0061 / +1.9252 on
  the same 10-scene protocol. The 4-block figures are broadly similar; the
  5-block figures are not.
- `MEASURED` — four blocks remains the more interesting candidate: 29.83 % fewer
  MACs, better RMSE, stereo probes at or above the control's, ΔD1 inside the
  reproducibility scales.

**What this does NOT establish**

- **Not** that 4 blocks matches 6. `INCONCLUSIVE` means inconclusive; the
  pre-registered claim ceiling (§7) forbids reading it as equivalence.
- **Not** that 5 blocks is architecturally worse than 4 — that ordering is
  precisely what one run cannot support.
- Nothing about three blocks or below, about latency, or about a pretrained
  model.
- `HISTORICAL` E3b, E3c and O6 verdicts remain **INDETERMINATE** and were not
  modified. This experiment is additive and is not a correction of their numbers:
  the protocol differs.
- **Stage E stays CLOSED.** Nothing here touches the cost volume.

---

## NEXT EXPERIMENT

> **Run seeds 1 and 2 for the 6-, 5- and 4-block configurations under the Stage B
> deterministic protocol — six runs, ≈7.9 GPU-hours — and report each
> architecture's 40-scene EPE/D1 as a mean over three seeds with its observed
> spread.**

Why this one, and why nothing else: this experiment removed execution noise and
proved that seed noise is what remains. The non-monotone ordering is a direct
measurement that one seed cannot answer the capacity question, and three seeds
per architecture is the smallest design that can separate a block-count effect
from a trajectory effect. Every other candidate — three blocks, Stage E, E2b,
Scene Flow — would inherit the same unresolved confound.

Explicitly **not** recommended: adopting any reduced-block configuration on this
evidence; extending to three blocks before the seed question is settled; opening
Stage E.

**Not executed in this task.**
