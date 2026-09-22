# Stage E leaderboard

One row per completed experiment. Three Kaggle T4 seeds each, Stage-1
pretrained ARM-P init, frozen architecture at 397,954 parameters, scored only
through `phase1/harness/frozen_eval.py` on the 40-scene contract.

| exp | intervention | seeds | mean best EPE | spread | mean final EPE | verdict |
|---|---|---|---|---|---|---|
| **E0** | none (control) | 0,1,2 | **1.2037841** | 0.0174488 | **1.2125409** | control — no verdict |
| E1 | EMA 0.999 | 0,1,2 | 1.2036174 | 0.0314905 | 1.2044355 | **INCONCLUSIVE** |
| E2 | native batch 8 | 0,1,2 | 1.2390444 | 0.0037600 | 1.2480458 | **REJECT** |
| **E3** | 400 epochs | 0,1,2 | **1.1787445** | 0.0147367 | **1.1675765** | **ACCEPT (non-overlapping)** \* |
| E4 | broader pretraining | — | — | — | — | **BLOCKED** (FT3D absent) |

Acceptance thresholds instantiated from E0: `PREREGISTRATION.md`.

\* E3's contract scoring ran on the development box (RTX 4060 Laptop, torch
2.7.0+cu128, fp32), not on Kaggle — see the E3 status entries below for why and
for the device calibration that makes it comparable. Training was on Kaggle T4
like every other row.

## Reference figures — not the acceptance baseline

| figure | value | what it is |
|---|---|---|
| ARM-P local 3-seed mean | 1.1988735 px | historical, different hardware and torch version |
| ARM-P seed 1 local | 1.1912168 px | a single seed, the best of three — never a baseline |
| E0 Kaggle 3-seed mean | **1.2037841 px** | **the Stage-E control** |

The Kaggle control sits 0.0049106 px above the historical local mean, inside
the ~0.022 px environment shift Stage B measured. Expected, pre-registered as
expected, and not a regression.

## Status

- E0 training: **COMPLETE**, 3/3 seeds, `contract_match` true on all six
  checkpoints.
- E0 INT8 control: **MEASURED** — `P_control` = **5.5865268 px**, candidate
  gate `P_candidate <= 5.6365268`.
- **ARM-P does not survive INT8**: int8 D1 80–86 % against 6.4 % fp32, a
  penalty 16.3x the reference model's under the identical procedure. Not a
  Hailo statement; Stage D stays BLOCKED. See `INT8_CONTROL_REPORT.md`.
- **E1 complete — verdict INCONCLUSIVE** (`e1_verdict.json`, computed by
  `verdict.py`). `Δ_best` = +0.0001667: positive, but ~100x smaller than the
  `S0_best` bar of 0.0174488, so indistinguishable from the control's own seed
  noise. Per-seed best: 1.2164890 / 1.2093647 / 1.1849985.
- E1's INT8 limb passed (P 5.0562891 ≤ 5.6365268) and that decides nothing
  about survivability — E1 is destroyed by int8 just as the control is. This is
  the vacuity recorded in `INT8_CONTROL_REPORT.md` §4, now demonstrated.
- E1 seed spread is 0.0314905, **1.8x the control's 0.0174488**. EMA did not
  stabilise seed-to-seed variation here; it widened it. E1 produced both the
  best single run of the campaign so far (seed 2, 1.1849985) and the worst
  (seed 0, 1.2164890).
- **E2 complete — verdict REJECT** (`e2_verdict.json`). `Δ_best` = −0.0352603,
  about **2x the control's entire seed spread in the wrong direction**. Every
  E2 seed is worse than every control seed and every E1 seed. This is not
  noise. Per-seed best: 1.2414324 / 1.2380283 / 1.2376725.
- E2's INT8 limb also passed (P 5.5653839 ≤ 5.6365268) while the model is
  rejected outright — a second demonstration that the gate certifies nothing.
- **E2's spread is 0.0037600: 4.6x tighter than the control and 8.4x tighter
  than E1.** Batch 8 made runs highly consistent with each other and
  consistently worse — the classic large-batch signature at fixed LR, with 20
  optimizer steps per epoch instead of 80. The pre-registration deliberately
  held LR at 1e-3 so batch size was the single lever; this is the measured cost
  of that lever as specified, not a defect. An LR-scaled variant would be a
  separate experiment with its own pre-registration.
- E2 was also the fastest (~70 min/seed against E0's ~78), the 18 % per-epoch
  gain the T4 rate probe predicted. Cheapest and worst.
- **E3 complete — verdict ACCEPT (non-overlapping)** (`e3_verdict.json`,
  computed by `verdict.py` with the pre-registered constants untouched).
  `Δ_best` = +0.0250396 against the `S0_best` bar of 0.0174488, and
  `Δ_final` = +0.0449644 against the E3-only `S0_final` bar of 0.0090235, so
  both limbs clear. The worst E3 seed (1.1860654) beats the best control seed
  (1.1962889): the two seed sets do not overlap. Per-seed best: 1.1860654 /
  1.1788394 / 1.1713288. Per-seed final: 1.1628883 / 1.1728822 / 1.1669590.
  This is the campaign's first acceptance.
- **E3 is also the campaign's first run to beat the historical local reference.**
  Its mean best 1.1787445 sits below the ARM-P local 3-seed mean 1.1988735 and
  below the single best historical seed 1.1912168, and its mean final
  1.1675765 is lower still. The acceptance itself is against the E0 control, as
  pre-registered; those two figures remain reference-only.
- **Every E3 seed scores better on the final checkpoint than on the best
  checkpoint** (mean final 1.1675765 vs mean best 1.1787445). The 10-scene
  training monitor's selection does not transfer to the 40-scene contract.
  A1.2 anticipated exactly this — 400 epochs gives the monitor 81 selection
  opportunities against the control's 41 — which is why the pre-registration
  required `Δ_final` as well, and why clearing best alone would have been
  recorded as BEST PASS / FINAL FAIL rather than an acceptance. It clears both.
- **The acceptance cost 2x the compute.** E3 wall time was 9250 / 9163 / 9267 s
  per seed against E0's 4647 / 4557 / 4772 s — 1.98x for a 2.1 % mean-best
  improvement. Doubling epochs is the most expensive lever in the campaign and
  the only one that worked.
- E3's spread is 0.0147367, slightly tighter than the control's 0.0174488 —
  unlike E1, the gain did not come with extra seed-to-seed variance.
- E3's INT8 limb passed (P 4.3230999 ≤ 5.6365268) and, for the third time,
  that certifies nothing: per-seed int8 EPE is 5.2489414 / 5.3849188 /
  5.8715627 against fp32 ONNX 1.1859853 / 1.1788213 / 1.1713167. E3 is
  destroyed by int8 just as the control is, only slightly less so. The Stage-D
  blocker is untouched; see `INT8_CONTROL_REPORT.md` §4.
- **E3's Kaggle runs all ended in status ERROR, and that was a false alarm, not
  a training failure.** `run_arm.py`'s two post-hoc guards hardcoded the
  literal 200, so all three seeds trained their full 400 epochs and were then
  failed by a read-only check with `epochs_incomplete {"rows": 400}`,
  returncode 3. `kaggle/e3_patch.py` fixes both guards to compare against
  `args.epochs`; `kaggle/recover_e3.py` re-runs the identical guard block
  offline and records 8/8 guards passing with 400 rows on every seed
  (`e3_recovery.json`). Seed 2 was deliberately run against E0's unpatched
  bundle so all three seeds had byte-identical inputs, which is why it ERRORed
  too. No seed was retrained.
- **E3's contract scoring had to happen off-Kaggle, and the device gap was
  measured rather than assumed.** `run_experiment.py` exits as soon as
  `STOP.json` exists, so the false STOP aborted the scoring stage for all three
  seeds — no `hailo_val` numbers came back from the T4. The checkpoints did,
  so `kaggle/complete_e3.py` scores them locally through the same
  `frozen_eval` / `eval_tier2.score` path. To bound the comparability risk,
  E0's own checkpoints were re-scored locally and diffed against their recorded
  T4 values: max |local − T4| = **0.000129 px** across all six, i.e. 135x
  smaller than `S0_best`, with `contract_match` still true and every
  `weight_sha16` matching (`e0_device_recheck.json`). `eval_tier2.py` is
  byte-identical in `bundle/`, `bundle_e3/` and the code that ran on Kaggle.
  The original STOPPED Kaggle records are preserved at
  `e3_output/seed<N>/kaggle_stopped_seed<N>.json`.
- The recipe campaign is finished: E0 control, E1 INCONCLUSIVE, E2 REJECT,
  E3 ACCEPT. E4 stays BLOCKED (FlyingThings3D absent from disk). No new
  candidate is authorized.
