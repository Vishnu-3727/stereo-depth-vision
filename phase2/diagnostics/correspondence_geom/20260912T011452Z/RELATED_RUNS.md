# RELATED RUNS — `20260912T011452Z`

EXP-CORRESPONDENCE-GEOM-001, **HALTED AT HS1**.

Every record below was **READ ONLY**. All 36 were hashed at freeze time into
`historical.sha256`, verified before execution and **re-verified after the
halt** — `0` mismatches. Nothing was modified, re-scored, regenerated, re-run or
re-interpreted.

---

## 1. DESIGN AUTHORITY

| file | sha256 |
|---|---|
| `design_audit_20260912/DESIGN_AUDIT.md` | `e99bb53069fee829dfd3c1539198b49f0f5dcdd30922de7cca69271c082152c6` |
| `design_audit_20260912/design.json` | *(hashed in `historical.sha256`)* |
| `design_audit_20260912/source_trace.md` | `bcdd4e3384ecb07630a68016e3b174b2012510b322b4e3a9cc99f182077de63e` |
| `design_audit_20260912/RELATED_RUNS.md` | *(hashed in `historical.sha256`)* |
| `design_audit_20260912/synthetic_mechanism_check.py` | `7a4c69e9ade002f6c952d23f233fc71b5069e6d4c5277d0b73c9397df119cace` |
| `design_audit_20260912/synthetic_mechanism_check.json` | `06fa8ef8ef0c5f485fa6282a0341a1e9cae8285d81ab6e852b52f71e92a982c7` |

The design record is **unmodified**. The defect this run exposed is recorded
here and in `results.json`, **not** by editing the design.

### 1.1 What the design record got right, and what it got wrong

| | |
|---|---|
| **right** | the cancellation identity; the statistic and its non-injectivity proof; the two-process gate; hard stops on structural properties; the receptive field, **correctly derived as 477 px in `source_trace.md` §3** |
| **wrong** | the mask's **left** bound and HS1's `w < 54` window. Both were derived from the `shift_left` fill and the aggregation radius only. The 477 px extractor halo was **declared** (`source_trace.md` §10) and then argued away as a cancelling main effect — which is false for the POSITIVE cells, where it destroys the `V = 0` premise itself |

---

## 2. PRIOR CORRESPONDENCE RECORDS (unmodified)

| record | sha256 | relevance to this run |
|---|---|---|
| `correspondence/20260910T142622Z/RESULTS.md` (SHIFT-001) | `c031799c9921fc41659c2e6040594ab03560aefb13f89c5798e3849777a00154` | precedent for a halt on a falsified analytic anchor |
| `correspondence_index/20260911T015124Z/RESULTS.md` | `0cb5e11ce2ef5091526f2d4a25522ba8a8c632c9181e53d6b9ce68d6e483f0c3` | Level-C closure |
| `correspondence_arch/20260911T025312Z/RESULTS.md` | `db6d9f176b21170442f46b99b97802cc9699cf8c2b0a59067b3e1583d58a81ca` | the containment hazard |
| `correspondence_geom/20260911T111809Z/RESULTS.md` (GEOM-001) | `16e9dddbe6c955bfa8585fd64305645ba57c3d926c277b27563dceeda0e7ea5b` | representability defect |
| `correspondence_geom/20260911T115917Z/RESULTS.md` (GEOM-002) | `9b6f133d3bdbbf377bff488266f4d923ed2174d96f1b8ff05b8b8258300bb106` | **no admissible result**; the hard-stop-not-enforced precedent |
| `correspondence_geom/.../construction_spec.json` | `a9e8f250bfef49607951fae0b304d5713091553ad2da885b911f4de711bdea10` | origin of the self-pair construction |
| `correspondence_arch_rate/20260911T130507Z/results.json` | `1f0463ec08b950d3c1b69804407299f66b4b6cee5c04af51419a061dba7458f1` | random-weight procedure (population **not** reused) |
| `…/construction_spec.json` | `7a4e2341fc50e3695648cc286cdb4219390a0378a6669994ec0e4555b75cf9b0` | crop geometry, `t` levels — **inherited verbatim** |
| `…/randomisation_spec.json` | `e52b5b03c5d4406e5ed9212f6681ee53bb448927eebddd367a7698ce9ce203ad` | aggregation-only randomisation, seeds 0–31 — **inherited verbatim** |
| `…/classification_spec.json` | `273f7562b62dd4c29a2d617e3b2b4b24d98532fcc7d99531b82b55b377d0cd28` | the 64-px border — **inherited, and this is where the defect entered** |
| `correspondence_tr/20260911T134500Z/RESULTS.md` (TR-001) | `49574f2521312c326d310f41bdee9ef697c3664b74403603649ae9a0bb11c22c` | the C′ result; no magnitude reused |
| `correspondence_tr/20260911T134500Z/spec.json` | `5e6a0bca39408018454fb69e3287f26f8524fc89475ee564a5da5c54a5fa0280` | checkpoint paths + sha256 — **inherited verbatim** |
| `correspondence_geom/postmortem_20260911/POSTMORTEM.md` | `e6f8a93e15e36932afcffe8002223150fb654d9494cb069ab8d7f0e85495906b` | the hard-stop lesson this run honoured |
| `…/STATISTIC_AUDIT.md` | `f85bc599271b15ca9f83a0a35cbd74682dd076b252cddd7b2029b6694098daf7` | the C3 non-injectivity lesson |
| `…/SUCCESSOR_ASSESSMENT.md` | `aa3bb252055ead6652f5e1301337641c0fe38f93c86bf95c2a0cf5582d6d5519` | the seven preconditions |
| `predesign_audit_20260911/ALGEBRA_AUDIT.md` | `004ca38cddf1cf149fb0fec64f223f2ab889cfce0a1f9fe4ab66bfe714c9678f` | the forced ramp, the gate structure |
| `successor_audit_20260911/{AUDIT,RECOMMENDATION,SUCCESSOR_OPTIONS}.md`, `gate_audit_20260911/GATE_AUDIT.md` | *(in `historical.sha256`)* | prior option surveys |

## 2.1 Frozen source under test (unmodified)

`cost_volume.py` · `aggregation.py` · `feature_extractor.py` · `regression.py` ·
`stereonet.py` · `blocks.py` · `refinement.py` · `scaled_regression.py` ·
`kitti2015.py` — all nine hashed in `historical.sha256` and byte-identical after
the halt. **No architecture, feature extractor, cost-volume construction,
standardisation, readout or mask implementation was modified.**

---

## 3. CHECKPOINTS

| key | sha256 | touched this run |
|---|---|---|
| `H2_seed0` | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` | **loaded** — feature extractor used for 3 forward passes; **aggregation discarded before any use** |
| `H2_seed1` | `58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf` | never loaded |
| `H2_seed2` | `245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69` | never loaded |
| `NEG_shift_none` | `d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1` | never loaded |

No checkpoint file was written to. No weight was modified.

---

## 4. SCENE LEDGER

| indices | used by |
|---|---|
| `27, 0, 31, 6` | SHIFT-001, GEOM-001 |
| `5, 7, 8, 9` | GEOM-002, ARCH-RATE-001, TR-001 |
| `1, 2` | **read by this run** (`000161_10`, `000162_10`) — pair 0 only, and only as far as four cost volumes and the post-halt wiring diagnostic. **No statistic was computed on them.** |
| `3, 4, 10, 11, 12, 13, 14, 15, 16, 17` | preregistered but **never loaded** — still unmeasured |

**INFERRED:** ten of the twelve preregistered scenes remain entirely unmeasured,
and the two that were read yielded only cost-volume zero-set facts.

---

## 5. ANTI-POST-HOC LEDGER

No threshold was taken from `α_h ≈ 1`, TR-001 amplitudes, ARCH-RATE behaviour,
INDEX-002 ordering, or any previous separation margin. No threshold was changed
after execution — **no threshold was ever evaluated**, because the run halted
before the first statistic. `H*` does not exist. No p-value, no EPE, no D1, no
RMSE was computed.

The post-halt diagnostic (`diagnose_hs1.py`) measures **only the zero-set of the
cost volume**. It runs no aggregation, computes no interaction, no `argmin` and
no `h`. Its numbers are wiring facts and are used for nothing else.

---

## 6. STATUS

```
HALTED AT HS1.   NO ADMISSIBLE RESULT.   LEVEL D UNCHANGED: NOT-DEMONSTRATED.
Hard stop fired AND enforced.  Nothing repaired.  Nothing continued.
No successor designed, preregistered or launched.
```
