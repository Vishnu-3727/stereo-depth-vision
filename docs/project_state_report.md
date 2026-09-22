# Project state report — 2026-09-22

Scope: where the stereo depth project stands after the Stage-E recipe campaign
closed. Every number below is measured and cited to the file that holds it.
Claims that are not measured are labelled as such.

Repository: `C:\Users\vishn\stereo_depth_vision`, branch `master`, pushed to
`origin/main` at `3000331`. Local and remote are in sync as of this report.

---

## 1 Headline

The model is now materially better than any previous version of itself, and it
is still not deployable to the Hailo-8 accelerator. Those two facts are
independent and both are load-bearing.

- **Best model to date:** E3 seed 0 **final** checkpoint, EPE **1.1628883 px**
  on the frozen 40-scene contract
  (`stage_e_recipe/kaggle/e3_output/seed0/e3_seed0_final.pth`).
- **Previous all-time best:** 1.1912168 px (ARM-P seed 1, the figure quoted
  throughout Stage B and Stage C). The new checkpoint beats it by 0.0283285 px,
  or 2.38 %.
- **Architecture unchanged** at 397,954 parameters. The gain came from the
  training recipe only — specifically from training twice as long.
- **INT8 still destroys the model** (EPE ~5.2–5.9 against ~1.17 fp32), so the
  accelerator story has not improved at all.

---

## 2 Stage-E recipe campaign — CLOSED

One lever per experiment, three seeds each, acceptance formula fixed in
`stage_e_recipe/PREREGISTRATION.md` before any candidate ran, verdicts computed
by `stage_e_recipe/verdict.py` and never by hand.

| exp | lever | mean best EPE | verdict |
|---|---|---|---|
| E0 | none (control) | 1.2037841 | control |
| E1 | EMA 0.999 | 1.2036174 | INCONCLUSIVE |
| E2 | native batch 8 | 1.2390444 | REJECT |
| **E3** | **400 epochs** | **1.1787445** | **ACCEPT (non-overlapping)** |
| E4 | broader pretraining | — | BLOCKED |

E3's verdict in full (`stage_e_recipe/e3_verdict.json`):

```
mean best  1.1787445  (control 1.2037841)  delta +0.0250396  bar 0.0174488
mean final 1.1675765  (control 1.2125409)  delta +0.0449644  bar 0.0090235
spread     0.0147367
INT8       P 4.3230999 <= 5.6365268  PASS
VERDICT: ACCEPT (non-overlapping)
```

Both pre-registered limbs clear, and the seed sets do not overlap: the worst E3
seed (1.1860654) beats the best control seed (1.1962889).

### Every contract EPE in the campaign, ranked

| rank | EPE | checkpoint |
|---|---|---|
| 1 | 1.1628883 | E3 seed0 final |
| 2 | 1.1669590 | E3 seed2 final |
| 3 | 1.1713288 | E3 seed2 best |
| 4 | 1.1728822 | E3 seed1 final |
| 5 | 1.1788394 | E3 seed1 best |
| 6 | 1.1849985 | E1 seed2 best |
| 7 | 1.1860654 | E3 seed0 best |
| 8 | 1.1904644 | E1 seed2 final |
| 9 | 1.1962889 | E0 seed1 best |
| 10 | 1.2013258 | E0 seed2 best |

E3 holds the top five places. The one prior result it does not beat outright is
E1 seed 2's best checkpoint, 0.0011 px below E3 seed 0's best — so E3 is lower
than every prior *mean*, every control seed, and the all-time record, but not
lower than every single prior number.

### Two findings that matter more than the verdict

1. **The monitor-selected "best" checkpoint is the weaker artefact.** On all
   three E3 seeds the *final* checkpoint scores better on the 40-scene contract
   than the *best* checkpoint. The 10-scene training monitor's selection does
   not transfer. Amendment A1.2 anticipated this — 400 epochs gives the monitor
   81 selection opportunities against the control's 41 — which is why
   `delta_final` was pre-registered as a second requirement. It clears by more
   than `delta_best` does. If you want the strongest model from this campaign,
   take a final checkpoint, not a best one.
2. **The acceptance cost 1.98x the compute** — 9250 / 9163 / 9267 s per seed
   against E0's 4647 / 4557 / 4772 s, for 2.1 % mean best. Doubling epochs is
   the most expensive lever tried and the only one that worked. EMA was free
   and did nothing; batch 8 was cheaper and actively worse.

### How E3 was recovered and scored (read before reusing these numbers)

- All three E3 Kaggle runs ended in status **ERROR**, and that was a false
  alarm. `run_arm.py` had two post-hoc guards hardcoding the literal 200, so
  every seed trained its full 400 epochs and was then failed by a read-only
  check with `epochs_incomplete {"rows": 400}`.
  `stage_e_recipe/kaggle/e3_patch.py` fixes both guards to compare against
  `args.epochs`; `stage_e_recipe/kaggle/recover_e3.py` re-runs the identical
  guard block offline and records 8/8 guards passing with 400 rows on every
  seed (`stage_e_recipe/e3_recovery.json`). No seed was retrained.
- Seed 2 was deliberately run against E0's *unpatched* bundle so that all three
  seeds had byte-identical inputs. It therefore ERRORed for the same false
  reason, by design.
- **Contract scoring never ran on Kaggle for any E3 seed.** `run_experiment.py`
  exits as soon as `STOP.json` exists, so the false STOP aborted it before the
  `hailo_val` stage. The trained checkpoints were in the pulled artefacts, so
  `stage_e_recipe/kaggle/complete_e3.py` scores them on the development box
  through the same `frozen_eval` / `eval_tier2.score` path.
- **The device gap was measured, not assumed.** E0's own checkpoints were
  re-scored locally and diffed against their recorded T4 values: the largest
  local-minus-T4 difference is **0.000129 px** across all six, 135x smaller
  than the control's own seed spread, with `contract_match` still true and
  every `weight_sha16` matching
  (`stage_e_recipe/e0_device_recheck.json`). `eval_tier2.py` is byte-identical
  (sha256 `326e9058...`) in `bundle/`, `bundle_e3/` and the code that ran on
  Kaggle.
- Each seed's original STOPPED Kaggle record is preserved beside the canonical
  one as `e3_output/seed<N>/kaggle_stopped_seed<N>.json`.
- The training-log best10 metric sits about 0.4 px above contract EPE on both
  E0 and E3 (`stage_e_recipe/e3_logmetric_compare.json`). Never compare one
  against the other.

---

## 3 Stage status

| stage | state | note |
|---|---|---|
| Phase 1 (forensic reproduction) | **FROZEN** at tag `phase-1-frozen` | reverse-engineered and reproduced the Hailo reference; changed nothing |
| Phase 2 (architecture search) | closed campaigns, `phase2/` untracked | 3x3 block-count campaign CLOSED; correspondence campaign has 0 admissible Level-D results |
| Stage A (diagnostics) | complete | `stage_a_diagnostics/DIAGNOSTIC_REPORT.md` |
| Stage B (ARM-P training) | complete | `stage_b_armp/ARMP_CLOSURE_RECORD.md`; produced the 397,954-parameter ARM-P and the Stage-1 pretrain checkpoint |
| Stage C (deployment validation) | complete, C1 FAIL preserved | export parity fails at 1e-3; no export-level mitigation identified |
| Stage C runtime (new work) | complete | 4.51 to 9.46 FPS on CUDA, contract EPE unchanged |
| Stage D (Hailo-8) | **BLOCKED** | no Hailo toolchain reachable from this machine |
| Stage E (recipe) | **CLOSED** | E3 ACCEPT; E4 blocked |

---

## 4 Host runtime — where the speed stands

Measured on RTX 4060 Laptop, torch 2.7.0+cu128, fp32, 368x1232, 10 scenes from
the head of `hailo_val`, medians. Full detail in
`stage_c_deploy/runtime/README.md`.

| config | total ms | FPS |
|---|---|---|
| cpu fp32 baseline | 912.5 | 1.10 |
| cuda fp32 baseline | 221.7 | 4.51 |
| cuda fp32, preprocess optimised | 140.8 | 7.10 |
| **cuda fp32, + gpu geometry (deterministic) — recommended** | **105.7** | **9.46** |
| cuda fp32, + cudnn.benchmark | 89.7 | 11.14 — REJECTED |

The recommended path scores EPE 1.191216765057325 on the full contract, equal
to the frozen CPU figure, and is bit-identical across separate processes.

What was tried and rejected, with the reason worth remembering:

- **fp16 is impossible for this model.** Activations climb 2.6 to 23.7 to 1351
  to 43264 and hit the fp16 ceiling of 65504 at `residual.0.conv2`; the final
  disparity is 100 % NaN.
- **bf16** runs finite but moves disparity 2.317255 px. Rejected.
- **cudnn.benchmark** is 16 ms faster but shifts disparity up to 2.39 px, about
  twice the model's own EPE, and is nondeterministic across processes.
- **channels_last** is deterministic but not equivalent: 0.5596 px off the
  control, 0/10 scenes bit-identical.
- **torch.compile** is not runnable on this host (`TritonMissing`).
- **Decode prefetch was ACCEPTED** (10.3 to 13.0 FPS over a sequence,
  bit-identical) but is **not wired into the demo**, because it only pays off
  over a sequence and the demo is single-shot.

---

## 5 Blockers, stated plainly

1. **Hailo-8 deployment is blocked, and not by the cost volume.** The Dataflow
   Compiler is x86-64 Linux and this is a Windows box, so no toolchain is
   reachable; the target device is only *assumed* from a retail listing. Stage
   D did establish that the 3D cost volume is **not** the blocker — the vendor
   StereoNet has the same five `Conv3d` ops, the same rank-5 tensors and a
   published HEF. The real risk is ARM-P's shift construction: 689 nodes
   against the vendor's 168, with 12 op types the vendor never uses.
2. **INT8 destroys the model.** Every Stage-E experiment passed the INT8 gate
   and every one is still ruined by quantization — E3's int8 EPE is 5.2489414 /
   5.3849188 / 5.8715627 against fp32 ONNX 1.1859853 / 1.1788213 / 1.1713167.
   The gate certifies nothing; that vacuity is recorded in
   `stage_e_recipe/INT8_CONTROL_REPORT.md` section 4 and has now been
   demonstrated three times.
3. **Export parity fails.** Stage C's C1 gate fails at the 1e-3 tolerance and
   no export-level mitigation was found within the tested representation space.
   The residual term of the refinement stage carries the failure, not the
   initial regression.
4. **E4 (broader pretraining) is blocked by missing data.** FlyingThings3D is
   gone from disk — `data/sceneflow/flyingthings3d` holds 5 KB of leftovers and
   `monkaa` is empty. Only `data/sceneflow/driving` (39 GB) and
   `data/kitti2015` (3.2 GB) are present. The Stage-1 pretrain checkpoint is on
   disk and hash-verified, so fine-tune-only work runs today.

---

## 6 What is NOT established

- That the new checkpoint helps on anything other than the 40-scene KITTI 2015
  contract. It has not been evaluated on another split, another dataset, or
  live camera input.
- That 400 epochs is the optimum. 400 was the pre-registered lever value; 300
  and 600 are untested, and the per-seed best epochs (260 / 290 / 270) all fell
  short of 400, which is a hint and not a measurement.
- That E3's gain survives quantization, pruning, or any accelerator path. The
  int8 numbers say it does not survive int8.
- Anything about Hailo-8 latency, memory or power for this model. No figure of
  that kind exists anywhere in this repository, measured or estimated.

---

## 7 Options from here

Listed as options, not a recommendation, and none is authorized.

1. **Bank the win and re-validate downstream.** Re-run the Stage C runtime path
   and the demo against E3 seed 0's final checkpoint, so the fast path and the
   best model are the same artefact. Cheap, and right now the two disagree.
2. **Epoch-count sweep.** 400 worked; 300 and 600 would say whether the gain is
   monotone or whether 400 is near a plateau. Same harness, one lever, needs
   its own pre-registration.
3. **Attack quantization directly.** The int8 collapse is the only blocker that
   bears on every accelerator, not just Hailo. Quantization-aware training or a
   range-limiting architectural change would be a new stage, not a recipe
   lever.
4. **Unblock E4** by re-fetching FlyingThings3D from `D:\sceneflow_archives`
   (not mounted) and pretraining broader.
5. **Unblock Stage D** by running the Dataflow Compiler on the Ubuntu 24
   machine, or on the Pi 5 with the AI HAT+ when it is reachable.

---

## 8 Provenance

| artefact | path |
|---|---|
| Stage-E leaderboard | `stage_e_recipe/LEADERBOARD.md` |
| acceptance formula | `stage_e_recipe/PREREGISTRATION.md` |
| E3 verdict | `stage_e_recipe/e3_verdict.json` |
| E3 guard recovery | `stage_e_recipe/e3_recovery.json` |
| device calibration | `stage_e_recipe/e0_device_recheck.json` |
| log-metric comparison | `stage_e_recipe/e3_logmetric_compare.json` |
| E3 INT8 gate | `stage_e_recipe/int8_e3/int8_e3.json` |
| E3 per-seed evidence | `stage_e_recipe/kaggle/e3_output/seed<N>/` |
| host runtime | `stage_c_deploy/runtime/README.md` |
| INT8 vacuity | `stage_e_recipe/INT8_CONTROL_REPORT.md` |
| Hailo blocker | `stage_d_hailo8/D0_TOOLCHAIN_AUDIT.md`, `stage_d_hailo8/D1_EXPORT.md` |

Commits carrying this state: `8962c97` (E3 tooling), `3000331` (E3 evidence,
verdict and leaderboard).
