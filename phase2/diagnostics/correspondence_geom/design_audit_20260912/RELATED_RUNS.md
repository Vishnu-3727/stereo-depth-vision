# RELATED RUNS — provenance for `design_audit_20260912`

Record `phase2/diagnostics/correspondence_geom/design_audit_20260912/`.
2026-09-12.

**Every record below was READ ONLY.** None was modified, re-scored, regenerated,
re-run or re-interpreted. The sha256 values were computed **after** this design
record was written and are recorded so that any later reader can verify the
historical record is untouched.

**Nothing in this design was executed.** No trained checkpoint was loaded.

---

## 1. RECORDS RELIED ON

| # | record | sha256 | what this design takes from it |
|---|---|---|---|
| 1 | `correspondence/20260910T142622Z/RESULTS.md` (**SHIFT-001**) | `c031799c9921fc41659c2e6040594ab03560aefb13f89c5798e3849777a00154` | the **empirical** reason the translation class fails: a provably search-free model responded to right-image translation by up to 0.70 candidates. Used as a *statement that a design class is inadequate*, never as a magnitude |
| 2 | `correspondence_index/20260911T015124Z/RESULTS.md` (**INDEX-003**) | `0cb5e11ce2ef5091526f2d4a25522ba8a8c632c9181e53d6b9ce68d6e483f0c3` | the Level-C closure: the right image enters as a `k`-constant term |
| 3 | `correspondence_arch/20260911T025312Z/RESULTS.md` (**ARCH-001**) | `db6d9f176b21170442f46b99b97802cc9699cf8c2b0a59067b3e1583d58a81ca` | the **containment hazard**: a reference interval spanning 97.4 % of the statistic's range. Motivates §9.3 (separation, never containment) |
| 4 | `correspondence_geom/20260911T111809Z/RESULTS.md` (**GEOM-001**) | `16e9dddbe6c955bfa8585fd64305645ba57c3d926c277b27563dceeda0e7ea5b` | the representability defect that the synthetic self-pair dissolves |
| 5 | `correspondence_geom/20260911T115917Z/RESULTS.md` (**GEOM-002**) | `9b6f133d3bdbbf377bff488266f4d923ed2174d96f1b8ff05b8b8258300bb106` | **no admissible result.** Cited only for the fact of its defects, never for a response value |
| 6 | `correspondence_geom/20260911T115917Z/construction_spec.json` | `a9e8f250bfef49607951fae0b304d5713091553ad2da885b911f4de711bdea10` | the origin of the synthetic self-pair construction |
| 7 | `correspondence_arch_rate/20260911T130507Z/results.json` (**ARCH-RATE-001**) | `1f0463ec08b950d3c1b69804407299f66b4b6cee5c04af51419a061dba7458f1` | that 378/384 sampled random aggregations were `FLAT` — cited as the *existence* of a random-weight reference procedure, **not** as a threshold. **This design builds its own gate population on a different statistic; the 384 `α_h` values are NOT reused** |
| 8 | `correspondence_arch_rate/20260911T130507Z/construction_spec.json` | `7a4e2341fc50e3695648cc286cdb4219390a0378a6669994ec0e4555b75cf9b0` | **inherited verbatim**: crop geometry, `t` levels, disparity algebra |
| 9 | `correspondence_arch_rate/20260911T130507Z/randomisation_spec.json` | `e52b5b03c5d4406e5ed9212f6681ee53bb448927eebddd367a7698ce9ce203ad` | **inherited verbatim**: aggregation-only randomisation, framework-default init, seeds 0–31, frozen extractor |
| 10 | `correspondence_arch_rate/20260911T130507Z/classification_spec.json` | `273f7562b62dd4c29a2d617e3b2b4b24d98532fcc7d99531b82b55b377d0cd28` | the 64-px mask border, inherited unchanged (the **right** border is re-derived — §10.3) |
| 11 | `correspondence_tr/20260911T134500Z/RESULTS.md` (**TR-001**) | `49574f2521312c326d310f41bdee9ef697c3664b74403603649ae9a0bb11c22c` | the **C′** result and, decisively, its own §7: the trained slope `0.95…1.22` is *exactly* the architecturally forced prediction. **No TR-001 magnitude is used as a threshold** (§2 below) |
| 12 | `correspondence_tr/20260911T134500Z/spec.json` | `5e6a0bca39408018454fb69e3287f26f8524fc89475ee564a5da5c54a5fa0280` | **inherited verbatim**: checkpoint paths and sha256; determinism settings; the no-p-value discipline |
| 13 | `correspondence_geom/postmortem_20260911/POSTMORTEM.md` | `e6f8a93e15e36932afcffe8002223150fb654d9494cb069ab8d7f0e85495906b` | Task B — why `V ≡ 0` does **not** imply `5.5`; why a hard stop must encode a *derivable property*, not a predicted value |
| 14 | `correspondence_geom/postmortem_20260911/STATISTIC_AUDIT.md` | `f85bc599271b15ca9f83a0a35cbd74682dd076b252cddd7b2029b6694098daf7` | C3's non-injectivity — the origin of the §8.2 requirement |
| 15 | `correspondence_geom/postmortem_20260911/SUCCESSOR_ASSESSMENT.md` | `aa3bb252055ead6652f5e1301337641c0fe38f93c86bf95c2a0cf5582d6d5519` | `SUCCESSOR-JUSTIFIED` and its seven preconditions |
| 16 | `correspondence_geom/predesign_audit_20260911/ALGEBRA_AUDIT.md` | `004ca38cddf1cf149fb0fec64f223f2ab889cfce0a1f9fe4ab66bfe714c9678f` | §1.6 (the forced unit ramp), §3.4 (the random arm as a **gate**), §4 (`t = 0` handling), and the ten mathematical requirements — **all ten are satisfied, §3 below** |
| 17 | `correspondence_geom/successor_audit_20260911/SUCCESSOR_OPTIONS.md` | `bd4aa70774d714b1618e2e1abca3cc8aeeece3264d8638b3faccb2451ec78ac0` | the four options previously surveyed; this design's intervention is **not** among them |

### 1.1 Source files traced (read-only)

| file | sha256 |
|---|---|
| `src/models/stereonet/cost_volume.py` | `e83f7fa462955eef0694aba5f0b4a27205d7c975de36497a46547d7355db1685` |
| `src/models/stereonet/aggregation.py` | `a734782a80256cc6732036fd154706833adb23c64cb03b9c2287b3b7ad7c6702` |
| `src/models/stereonet/feature_extractor.py` | `0c49f78ed074907afc37dc09f6eb017be0f268df17ab279c91f45976bd2d3d05` |
| `src/models/stereonet/regression.py` | `53326856ce6bfe0bfe859ec8e0a9780df713e23a89bdd6ef831ca314defb68dd` |
| `src/models/stereonet/stereonet.py` | `f5d847831ff23f4ee204cca1e793babd388ba7c4307c0ab4c8923d4dee368605` |
| `src/models/stereonet/blocks.py` | `fcdc9ab0c0f7d337144bd2432cd43d0dce6abd419b4058630d0b5b4ace0eac8a` |
| `phase2/models/scaled_regression.py` | `400996700e5663d06de659d249f48e98ec633dbd723e6a7ce138f74310703d6c` |
| `src/datasets/kitti2015.py` | `9f3369d85c07b35680b928767d7b521fc9110d14a594adf7165fcf36a1bec25e` |

### 1.2 This record's own artifacts

| file | sha256 |
|---|---|
| `synthetic_mechanism_check.py` | `7a4c69e9ade002f6c952d23f233fc71b5069e6d4c5277d0b73c9397df119cace` |
| `synthetic_mechanism_check.json` | `06fa8ef8ef0c5f485fa6282a0341a1e9cae8285d81ab6e852b52f71e92a982c7` |
| `source_trace.md` | `bcdd4e3384ecb07630a68016e3b174b2012510b322b4e3a9cc99f182077de63e` |
| `DESIGN_AUDIT.md` | `e99bb53069fee829dfd3c1539198b49f0f5dcdd30922de7cca69271c082152c6` |

*(`DESIGN_AUDIT.md` is hashed as written; `design.json` and this file are hashed
at freeze time by the successor harness, since they reference each other.)*

---

## 2. ANTI-POST-HOC LEDGER — what was taken, and what was refused

The rule (§XII of the brief): prior experiments may motivate hypotheses; their
**observed magnitudes may not define thresholds**.

### 2.1 Taken — structural facts, definitions and algebra

| taken from | what | why it is not a threshold |
|---|---|---|
| ARCH-RATE construction spec | crop `1136 × 272`, `t` levels, `d = t` algebra | a geometric definition |
| ARCH-RATE randomisation spec | aggregation-only randomisation, seeds 0–31, frozen extractor | a procedure |
| ARCH-RATE classification spec | the 64-px border | a geometric definition, inherited unchanged |
| TR-001 spec | checkpoint paths + sha256, determinism settings, the no-p-value rule | identifiers and discipline |
| ALGEBRA_AUDIT | the forced-ramp derivation; the gate structure; the `t = 0` prohibition; requirement 5 (non-injectivity) | algebra and method |
| POSTMORTEM | the `V ≡ 0 ⇏ 5.5` refutation; the hard-stop lesson | a refutation and a rule |
| SHIFT-001 / GEOM-001 / GEOM-002 | *that* their designs were inadequate | a statement about design, not data |

### 2.2 Refused — every magnitude

| refused | where it would have crept in | replaced by |
|---|---|---|
| `α_h ≈ 1` as a success criterion | a tolerance around the anchor | **not used at all** — the slope is not this design's statistic, and ALGEBRA_AUDIT §2.3 item 6 proves it is defeated by the artifact |
| TR-001's 3.9× separation factor | a "require 2×" rule | **complete separation** against an empirical population, an ordering comparison with no numeric cut |
| the 384 random `α_h` values | reuse as this design's null | **not reused.** A different statistic requires its own gate population, generated afresh under the same frozen randomisation procedure |
| `RAIL_LOW = 1.0`, `RAIL_HIGH = 10.0`, `FLAT_RANGE = 1.0` | an inherited classification taxonomy | **not inherited.** Those thresholds classify a *soft-argmin response curve*; this design's statistic is an integer `argmin` index, to which they do not apply |
| the four-arm ordering from TR-001 §5 | an expected ranking | **no expected ranking is registered** |
| `disparity_initial = 7.603` (the `V ≡ 0` constant) | an anchor or a baseline | **no numeric disparity expectation exists anywhere in this design** |
| GEOM-002's `m_h(t)`, `m_v(t)`, `Slope`, C1–C4, the 3/12 count | any reuse | **none.** GEOM-002 has no admissible result; its data may be cited only to show that a design choice was inadequate |
| `synthetic_mechanism_check.py`'s own numbers (`5/48`, `h ≤ 3`, `1.124`) | a bar for the trained arm | **not used as a threshold.** Eight seeds on *synthetic* fields calibrate nothing. They answer one yes/no question — *can the artifact produce the signature?* — and the real gate supplies `H*` |

### 2.3 The one observed value that does enter the decision rule, and why it is legitimate

`H*` = the maximum `h` over the 576 **gate** units. It is:

- **generated by this experiment**, in Stage 1, before any trained aggregation is
  loaded — not imported from a previous record;
- used **only as an ordering reference** for complete separation
  (`min h_trained > H*`), never as a numeric cut chosen by anyone;
- **not adjustable**: the rule is fixed before Stage 1 runs, and Stage 1's
  outcome is not seen by anyone with the freedom to change it.

This is the same threshold-free device TR-001 used ("outside the observed random
population") and is the form ALGEBRA_AUDIT §3.2 requires.

---

## 3. THE TEN `ALGEBRA_AUDIT` REQUIREMENTS — compliance

| # | requirement | how this design satisfies it |
|---|---|---|
| 1 | **no through-origin fit** | no fit of any kind is in the decision rule; the statistic is an integer `argmin`. The free-intercept slope is reported as a secondary quantity only, compared to nothing |
| 2 | **`t = 0` may not be a baseline** | `τ = 0` is **excluded entirely** — not baselined, not normalised by, not fitted through. `τ ∈ {1,…,6}` |
| 3 | **threshold-free gating conditions only** | every gating condition is an equality of integers (`argmin == τ`) or an ordering (`min h_trained > H*`). No magnitude gates anything |
| 4 | **shape conditions applied symmetrically to every arm** | `h` is computed identically for the trained arm, the 576 gate units, the search-free arm and the vertical arm. This is the GEOM-002 C2 asymmetry, fixed |
| 5 | **non-injectivity proven absent** | `DESIGN_AUDIT.md` §8.2 exhibits seven alternative shapes and shows each maps to `h ≤ 1`, and that a constant-offset tracker maps to `h = 0` and is captured as its own outcome class |
| 6 | **the random-weight arm is mandatory and must gate** | Stage 1 runs first, aggregation-only, extractor frozen, readout untouched, bit-identical cost volumes, 32 seeds, full six-level curves; `G-STOP` halts the process and Stage-2 code is unreachable in the same run (HS7) |
| 7 | **separation, not containment** | §9.3; `G-PASS` + `min h_trained > H*` |
| 8 | **hard stops encode derivable properties and are enforced** | HS1–HS8, each with its derivation; HS3 and HS4 are exact algebraic consequences confirmed numerically without any trained model; every stop must `raise` and the harness must be shown to have exited |
| 9 | **rail clearance and clipping detectability** | the statistic is an `argmin` over the **full** `k = 0…11` grid, so saturation at a rail *is* an observable value of `argmin` and is recorded per cell. The predicted locations `τ = 1…6` sit clear of both rails |
| 10 | **contamination disclosure** | `DESIGN_AUDIT.md` §0; status declared in advance as **consistency evidence at best** |

### 3.1 The seven `SUCCESSOR_ASSESSMENT` preconditions

| precondition | status |
|---|---|
| a shape-sensitive primary condition, fixed by the hypothesis | **met** — extremum *location*, fixed by `V = 0` at `k = τ` and `softmax(−z)` |
| the `t = 0` degeneracy resolved or excluded | **met** — excluded |
| the random-weight aggregation arm included | **met** — and it gates |
| hard stops written against verified premises | **met** — HS1–HS8, derivations given |
| hard stops actually enforced in the harness | **required by HS7**, to be asserted in code |
| contamination disclosed, status stated in advance | **met** — §0 |
| ideally specified by a party that has not read the curves | **NOT met** — disclosed as an irreducible limitation |

---

## 4. WHAT THIS DESIGN ADDS THAT NO PRIOR RECORD CONTAINS

1. **The cancellation identity** `V_AA + V_BB − V_AB − V_BA ≡ 0`, from the
   linearity of `build_cost_volume` in both feature maps. Not present in any
   prior record.
2. **The proof that a cross-paired volume translates exactly as a matched one
   does** (`V_AB = G(k−τ, w+τ)`, `G` τ-independent), which closes the
   translation-slope class against *correspondence* and not merely against
   *learning* — a strictly stronger statement than `ALGEBRA_AUDIT` §1.6.
3. **The first intervention in this campaign that varies correspondence while
   holding the candidate-axis translation structure fixed.** All four options in
   `SUCCESSOR_OPTIONS.md` and all of GEOM-001/002 vary the translation.
4. **A synthetic mechanism test that switches the artifact channel off** (an
   affine aggregation), measuring the artifact with and without the
   nonlinearity, with no trained model.

---

## 5. SCENE LEDGER

**MEASURED:** `hailo_val` holds 40 scenes.

| indices | used by |
|---|---|
| `27, 0, 31, 6` | SHIFT-001, GEOM-001 |
| `5, 7, 8, 9` | GEOM-002, ARCH-RATE-001, TR-001 |
| **`1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17`** | **proposed here — all twelve never measured in any correspondence experiment** |
| `18…26, 28, 29, 30, 32…39` | still unused after this design (20 scenes) |

Selection rule, fixed in advance and admitting no choice: **the twelve lowest
unused indices in ascending order, paired consecutively.** No GT-based and no
response-based selection anywhere.

---

## 6. STATUS

```
DESIGN AND AUDIT ONLY.
NOTHING EXECUTED.  NO TRAINED MODEL INSTANTIATED.  NO RESULTS.md.
NO HISTORICAL RECORD MODIFIED.  THE EXPERIMENT IS NOT LAUNCHED.
```
