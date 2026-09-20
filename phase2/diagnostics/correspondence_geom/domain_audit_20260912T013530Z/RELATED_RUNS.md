# RELATED RUNS — `domain_audit_20260912T013530Z`

Corrected validity domain for the geometric correspondence diagnostic.
**Design and validation only. The experiment was not run.**

---

## 1. IMMUTABILITY VERIFICATION

**41 historical records re-verified byte-identical — 0 mismatches.** Verified by
re-running the GEOM-001 halt record's own two manifests (`frozen.sha256`, 5
protocol artifacts; `historical.sha256`, 36 prior records and source files)
against the filesystem after this audit completed.

Nothing was edited. The GEOM-001 record, its preregistration, its source trace
and the `design_audit_20260912` design record are all untouched.

### 1.1 The GEOM-001 halt record — full manifest as it now stands

`phase2/diagnostics/correspondence_geom/20260912T011452Z/`

| file | sha256 |
|---|---|
| `PREREGISTRATION.md` | `d335f00eec35e1757b1c15115ee46d0bfb2c2bcefcd87bafff63e0afd3dbc204` |
| `spec.json` | `2b7b0a24144d506b72e10a461b50f2704eda880cc0f57eb2029bf13526d2a3a4` |
| `source_trace.md` | `bcdd4e3384ecb07630a68016e3b174b2012510b322b4e3a9cc99f182077de63e` |
| `run_geom.py` | `6a9b5e4e68e498dfee0b382208682e799c956315eb915b47fda624ff19ab3981` |
| `freeze_geom.py` | `1b0c21de4feb29f186c6d03dd0a30dd4ddd33f2d3a47f80f5173bef1de311cd7` |
| `frozen.sha256` | `63c2e0676087a91ff01dbb383040f770be236ec703a1e18bbb69e012ee885959` |
| `historical.sha256` | `02ad2f8aa1ca0756c9ea6af4b8198d74581c4bccf3a5aaa1757dfe7dd586cfe1` |
| `RESULTS.md` | `fe0919ba3d210e4a21135b32bf9a7774aebcb687fc3108c799d37c129d205609` |
| `results.json` | `bebed4c7563c9d5876ee6d5742128b0f90da831c7a0efe606445082304dfec93` |
| `RELATED_RUNS.md` | `12a6e788915f9df8c4ff6a6b1e7c5b90bfc22e2c34947b3fcb6b84cd02935bff` |
| `ENVIRONMENT.txt` | `54eeadb3e3997ac6abe3f699908b32774a20cf5bcda29a89943f6f9f6fda280a` |
| `run.log` | `f09f27047addd0cb36e683745d1a619f791767302e7ba1090db8feb3caa22b79` |
| `run_stage1.log` | `020a507da02bcd91e57c07853e588e368573d46a049e1384e7193a15b9bf410b` |
| `diagnose_hs1.py` | `a6ea8f5ff3cb95930f56f88edc5320069e4270feaac854d0d412dae2bfe28d31` |
| `diagnose_hs1.log` | `9b16610f27b83c3e8bef54618166d5fd3632ea3aa1a866d4a6f3854af9663762` |

**GEOM-001 produced no admissible experimental result.** It is not treated as a
failed experiment; it is treated as a halt with no determination.

### 1.2 This record

| file | sha256 |
|---|---|
| `DESIGN_AUDIT.md` | `96212b973fd157aea18876f98f02707135c8508300ea510e093c7df19394c3ea` |
| `domain_derivation.md` | `12b2111aa00c7f882fae3afb1b8c65c5d555a0f8531590c258ff0db024599856` |
| `design.json` | `977d9935b795e3c2bc9fb17414706d174f195e384df5f7c616079f97368d4512` |
| `validate_domain.py` | `00a80ed59ac29b28acaf46af548c6cade379f51fa540ab65c53078e6227e4aea` |
| `validate_domain.json` | `0d0f4e827d55313d5f65dceb74cbb235fb6a9de0254e648f835db57e65b2969e` |
| `validate_domain.log` | `7ac1ef5142b2aff9c81cb09b83ea23ced8a9fd1fd7722e077911fa60d823d9ae` |
| `synthetic_mask_check.py` | `f0f42116b33ad926378b189cdc0602a88be164acbc26ef616083ed7bc3112769` |
| `synthetic_mask_check.json` | `76f467643296a86e88880597c83182bd55b97d18c34d92aad004decf869df642` |
| `synthetic_mask_check.log` | `8230e405dd503afa8db9736cb2a996ae59fd5fc5de6bfec6650a4a68878366c7` |

*(`RELATED_RUNS.md` is this file and is hashed by a successor.)*

---

## 2. EXACT SOURCE FILES INSPECTED

All read-only, all hashed in the GEOM-001 `historical.sha256` and byte-identical.

| file | what was taken from it |
|---|---|
| `src/models/stereonet/feature_extractor.py` | stride 16; four `Conv2d(k=5,s=2,p=2)` with no activations; six `ResBlock`; final `Conv2d(k=3,p=1)`; receptive field `477` px |
| `src/models/stereonet/blocks.py` | `ResBlock` = two `Conv2d(k=3, p=dilation)`; LeakyReLU slopes |
| `src/models/stereonet/cost_volume.py` | `shift_left(x,k)[w] = x[w+k]`, zero fill for `w ≥ FW−k`; `V = shifted − right`; **linearity in both feature maps** |
| `src/models/stereonet/aggregation.py` | `5 × Conv3d(3×3×3, padding=1)` ⇒ spatial radius **5** cells |
| `src/models/stereonet/regression.py` | bilinear upsample `align_corners=True`; `softmax(−z)` ⇒ **a match is a MINIMUM**; readout has zero parameters |
| `phase2/models/scaled_regression.py` | per-pixel `z` across `k`, `eps = 1e-5`, population sd; pointwise in `(y,x)` |
| `src/models/stereonet/stereonet.py` | stage order; refinement is a separate stage and is never invoked |
| `src/datasets/kitti2015.py` | `pad_and_crop` anchors top-left; `normalize` is pointwise |
| `phase2/viz/core.py` | `load_scene`, `hailo_val` split |

---

## 3. PRIOR RECORDS RELIED ON

| record | relevance |
|---|---|
| `20260912T011452Z/` (**GEOM-001, HALTED AT HS1**) | the halt; the first measurement of the halo (one scene, one extractor). This audit re-measured it across 3 extractors × 12 scenes × 6 τ |
| `design_audit_20260912/` | the design whose mask and HS1 bound are corrected here. **Unmodified.** Its `source_trace.md` §3 derived the 477 px receptive field correctly — the defect was failing to propagate it into the mask's left bound |
| `correspondence_arch_rate/20260911T130507Z/construction_spec.json` | crop geometry, `t` levels — **inherited unchanged** |
| `…/randomisation_spec.json` | aggregation-only randomisation, seeds `0…31` — **inherited unchanged** |
| `…/classification_spec.json` | the 64-px row border — **inherited unchanged**, and now flagged as inherited rather than derived |
| `correspondence_tr/20260911T134500Z/spec.json` | checkpoint paths + sha256 — **inherited unchanged** |
| `correspondence_geom/postmortem_20260911/POSTMORTEM.md` | the rule that a hard stop must encode a derivable property — honoured; HS1 fired and halted |
| `…/STATISTIC_AUDIT.md` | the C3 non-injectivity lesson, whose generalisation this audit adds: *verify not only the statistic but the region in which the hypothesis holds* |
| `predesign_audit_20260911/ALGEBRA_AUDIT.md` | the gate-before-trained structure, retained unchanged |

---

## 4. CHECKPOINTS TOUCHED

| key | sha256 | use in this audit |
|---|---|---|
| `H2_seed0` | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` | **feature extractor only** — aggregation and refinement set to `None` at load |
| `H2_seed1` | `58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf` | **feature extractor only** — same |
| `H2_seed2` | `245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69` | **feature extractor only** — same |
| `NEG_shift_none` | `d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1` | **never loaded** |

No aggregation forward pass on real data. No readout on real data. No statistic.
No weight modified. No checkpoint written to.

---

## 5. SCENES READ

All twelve preregistered scenes were read, for wiring validation only:

```
pair 0  000161_10 / 000162_10      pair 3  000172_10 / 000173_10
pair 1  000163_10 / 000164_10      pair 4  000174_10 / 000175_10
pair 2  000170_10 / 000171_10      pair 5  000176_10 / 000177_10
```

**What was computed on them:** feature maps, the translation-identity zero-set,
the `V[τ] = 0` anchor, and the 2×2 cost-volume cancellation residual.
**What was not:** any aggregation output, any `z`, any interaction, any `argmin`,
any `h`. The scenes remain unmeasured with respect to the experiment's statistic.

---

## 6. ANTI-POST-HOC LEDGER

| quantity | provenance | post-hoc? |
|---|---|---|
| halo `= 15` cells | derived (`477 px` RF) **and** measured (216/216) | **no** — derived before measurement, measurement confirmed it |
| adopted vs tight halo | adopted the **conservative** symmetric bound though the measured right halo is one cell wider | **no** — the conservative choice cannot flatter a result |
| cost-volume domain `[15,44]` | derived from the `k ≤ 11` reach | **no** |
| statistic domain `[20,39]` | derived from the aggregation's 5-cell radius | **no** |
| pixel band `[325,632]` | derived from `align_corners` integer arithmetic; **verified** by S6 invariance | **no** |
| HS1-B tolerance `= 0.0` | the **strictest possible value**, shown achievable over 3 525 120 cells | **no** — a tolerance of exactly zero cannot be loosened to rescue anything |
| HS1-D criterion | structural (non-empty, τ-independent, `≥ 11` cells) — no magnitude | **no** |
| HS4 bound `1e-5` | float32 round-off with a 10× margin, stated before the measurement of `1.81e-07` | **no** |
| row border `[64,208)` | **inherited**, not derived; derivation permits all rows | **flagged** — kept conservative; widening it *after* seeing that support is thin would be support-shopping and is explicitly **not** proposed |
| old gate figures (`5/48`, `h ≤ 3`) | **discarded as void** — computed on the invalid mask | **n/a** |
| `H*` | must be regenerated in Stage 1, never inherited | **no** |

**No new statistic and no new discriminator was introduced.** S6 is an
*invariance* test of the mask, not a statistic about the model.

---

## 7. STATUS

```
DESIGN AND VALIDATION ONLY.
EXECUTABLE (horizontal arm)  ·  vertical control NOT executable as a clean null.
41 historical records re-verified, 0 mismatches.
No gate.json, no trained.json, no results.json, no preregistration.
No successor designed, authorized or launched.
```
