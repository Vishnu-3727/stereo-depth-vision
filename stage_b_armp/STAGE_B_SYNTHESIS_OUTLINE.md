> ---
> ## STATUS NOTICE — added 2026-09-20 by the documentation closure pass
>
> **This outline: COMPLETE** — as the planning artifact it declares itself to be. Nothing below
> is amended.
>
> **The downstream Stage-B synthesis manuscript: NOT REQUIRED FOR CURRENT PROJECT.**
> Basis: Stage B is closed, and `stage_b_armp/ARMP_CLOSURE_RECORD.md` is its authoritative
> record — it already carries the results, the limitations, the confounds and the
> identifiability verdicts in full. A manuscript would add presentation, not evidence. No
> research is required, authorized, or permitted to produce it.
>
> This document therefore stands as a retained planning/evidence map, **not** as unfinished
> work. The three first-class findings stated in RULE 2 below (coverage-vs-pretraining NOT
> IDENTIFIABLE; True GWC on frozen P2A NOT IDENTIFIABLE; GWC itself OPEN BUT UNSUPPORTED, NOT
> REFUTED) remain current and are carried into `RESULTS_INDEX.md` §6.
> ---

# Stage B Synthesis Outline — Planning Document (Zero-Compute Evidence Map)

> Status: OUTLINE AND EVIDENCE MAP ONLY. This document is a planning artefact.
> It does NOT draft the manuscript, does NOT argue findings in polished prose,
> and does NOT launch, propose, or prepare any experiment. Zero GPU, zero
> training, zero code changes, zero model changes were used for it.
> Phases 0–2 CLOSED. Stage A COMPLETE. No Phase 3 invented or proposed.
>
> RULE 2 — three first-class findings, kept visible, distinct, never softened:
> - Coverage vs pretraining: NOT IDENTIFIABLE
> - True GWC on frozen P2A: NOT IDENTIFIABLE
> - GWC itself: OPEN BUT UNSUPPORTED, NOT REFUTED
> The third is a different claim from the second. "Not identifiable on this
> platform" is not "does not work".

Scope note for every section below: each section lists structure, scope notes,
evidence pointers, and what the eventual paper/report section will need to
contain. Bullet-level content is correct; finished narrative is not.

---

## 1. Stage B scope and status

- Scope note: will state what Stage B is and is not; the paper will need a
  closed-state paragraph up front, not a methods discussion.
- Stage B = ARM-P replication + environment-confound control + two
  identifiability analyses (Candidate C, True GWC) + post-identifiability
  boundary + this synthesis outline. Nothing else.
- Status to record (paper will need each as a cited fact, not prose):
  - Phases 0–2 CLOSED; Stage A COMPLETE (architecture gate:
    "NO ARCHITECTURE JUSTIFIED"); no Phase 3 exists.
  - ARM-P replication COMPLETE: three predeclared seeds + one paired
    environment-confound control, all under the frozen evaluation contract.
  - Candidate C: NOT IDENTIFIABLE (finding, not failure).
  - True GWC on frozen P2A: NOT IDENTIFIABLE (finding, not failure).
  - GWC itself: OPEN BUT UNSUPPORTED, NOT REFUTED (distinct status, see
    Sections 8, 11, 12).
- Evidence pointers: `stage_b_armp/ARMP_CLOSURE_RECORD.md` Secs 1–3, 11–15;
  `stage_b_armp/POST_IDENTIFIABILITY_BOUNDARY.md` Secs 1–2, 12–13.

## 2. ARM-P intervention question

- Scope note: will state the single frozen question verbatim-adjacent; the
  paper will need the exact independent variable and why the control is what
  it is. No mechanism language belongs here.
- Frozen question (paper will quote from Closure Record Sec 1): does
  initializing the P2A architecture from a SceneFlow-pretrained checkpoint of
  itself, then running the exact P2A KITTI fine-tuning recipe, produce a lower
  frozen-contract KITTI error than the same recipe from random initialization?
- Intervention definition (paper will need all of):
  - ARM-P = INITIALIZATION INTERVENTION: strict load of
    `armp_stage1_best.pth`, sha256
    `3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7`,
    no optimizer state inherited, no architecture/loss/augmentation change.
  - CONTROL = RANDOM INITIALIZATION under the original P2A recipe. Never call
    it "P2A checkpoint initialization".
  - Reason to record (paper will need one sentence): a P2A-weights control
    would have answered "does 400 epochs beat 200", a different question.
- Evidence pointers: `stage_b_armp/ARMP_CLOSURE_RECORD.md` Secs 1–3;
  `stage_b_armp/20260918T142128Z_tier2_pilot/README.md` (Design; Why the
  control is RANDOM init); `stage_b_armp/ENV_CONTROL_PREREGISTRATION.md`
  Sec 5 rule 6.

## 3. ARM-P experimental design

- Scope note: will list frozen variables and per-run mechanics; the paper will
  need a design table, not results. Report both selections always; never only
  the flattering one.
- Frozen within each run (paper will need as a checklist):
  architecture, loss, augmentation, optimizer, scheduler, dataset, seed
  handling, device protocol, checkpoint selection, frozen evaluation contract.
- Mechanics to cite (paper will need each with file pointer):
  - Strict `model.load_state_dict(sd, strict=True)` of `["model"]` weights
    before optimizer/scheduler construction; fresh Adam +
    CosineAnnealingLR per arm in a fresh process.
  - Checkpoints store only `{"model", "config"}`.
  - 200 epochs per arm; checkpoint selection by minimum 10-scene in-training
    validation EPE (argmin; frozen score excluded).
  - Trainer is a line-for-line mirror of the P2A recipe with ONLY `--init` /
    `--arm` added (per `recipe_diff.json`: 6 non-equal blocks, all classified
    as the two flags or direct consequences).
  - Frozen contract on every scored checkpoint: 40 scenes, 3,802,797 valid
    pixels, gt_scale 256.0, gt_source disp_occ_0, contract_match true.
  - Model constants at scoring: 397,954 params, 70 keys, 24 disparity
    candidates, 3 downsample levels, shift right, regression_normalize true.
  - Evaluation deliberately held LOCAL for all four runs (constant measurement
    layer); seed 2 and the control TRAINED on Kaggle Tesla T4 /
    torch 2.10.0+cu128 / py3.12.13 vs seeds 0–1 local torch 2.7.0+cu128 —
    a between-seed environment difference that is NOT the intended independent
    variable, bounded (not eliminated) by the control.
- Evidence pointers: `stage_b_armp/ARMP_CLOSURE_RECORD.md` Secs 2–3, 8–9;
  `stage_b_armp/20260918T142128Z_tier2_pilot/TIER2_FROZEN_AUDIT.md` Secs A2–A5, A7.

## 4. Three-seed replication evidence

- Scope note: will present four runs as SEPARATE rows; the paper will need a
  table with best + final per run, full precision beside rounded, no pooling,
  no averaging, no significance, no confidence intervals.
- Numbers to reproduce exactly (paper will copy, not recompute):
  - seed 0 local: C best 1.5587716 / A best 1.2057590 / d -0.3530126;
    C final 1.4809434 / A final 1.2087613 / d -0.2721821.
  - seed 1 local: C best 1.4440242 / A best 1.1912168 / d -0.2528074;
    C final 1.4517077 / A final 1.2017376 / d -0.2499701.
  - seed 2 Kaggle: C best 1.4932978 / A best 1.1996447 / d -0.2936531;
    C final 1.4850973 / A final 1.1963930 / d -0.2887042.
  - Cross-seed spreads (plain observation, not a variance estimate):
    CONTROL best ~0.1147 px, ARM-P best ~0.0145 px.
  - 0.0444 px is a HISTORICAL run-to-run spread quoted as CONTEXT ONLY,
    never a significance threshold.
- Per-run audit pointers the paper will need:
  - Seed 0: `stage_b_armp/20260918T142128Z_tier2_pilot/` + `TIER2_FROZEN_AUDIT.md`.
  - Seed 1: `stage_b_armp/20260919T012646Z_tier2_seed1/` + `CORRECTIONS.json`.
  - Seed 2: `stage_b_armp/20260919T092454Z_tier2_seed2/` (local frozen eval).
- Descriptive licence only: ARM-P lower on best and final in all six
  seed-by-selection cells. No causal-mechanism sentence belongs here.

## 5. Seed-1 environment-confound control

- Scope note: will state the control's binding preregistration and outcome;
  the paper will need the within-delta shift numbers and the branch that
  applied. This is NOT a new seed and never replaces seed 1.
- Preregistration (paper will cite, not restate):
  `stage_b_armp/ENV_CONTROL_PREREGISTRATION.md` followed exactly, not modified.
- Numbers to reproduce exactly:
  - Seed 1 Kaggle ENVIRONMENT CONTROL: C best 1.4178492233455906 /
    A best 1.1872901156020892 / d -0.2305591077435014;
    C final 1.4283060535886267 / A final 1.2010106373630214 /
    d -0.2272954162256053.
  - environment_shift_best +0.022248292256498603;
    environment_shift_final +0.02267468377439466.
- Interpretation slot (paper will need verbatim-adjacent, bounded):
  the preregistered "small shift relative to the quoted context spreads"
  branch applied; the environment-confound concern on seed 2 is substantially
  reduced; the three seeds stand as they are — which is not proof the
  environment has no effect, only that this control did not reveal a material one.
- Evidence pointers: `stage_b_armp/20260919T130843Z_seed1_kaggle_envcontrol/`
  (`ENV_CONTROL_REPORT.md` Secs 1–2, 5, 7, 10–11, 13);
  `stage_b_armp/ENV_CONTROL_PREREGISTRATION.md` Secs 1–4, 6.

## 6. ARM-P closure / bounded interpretation

- Scope note: will state what the closed experiment does and does not
  establish; the paper will need two adjacent lists (establishes / does not
  establish) in bounded descriptive language. No universal, significance,
  causal-mechanism, generalization, deployment, or architecture sentence.
- Establishes slot: repeated directional observation under the stated recipe
  and contract (three same-evaluator seeds + control, both selections,
  contract held on all 16 scored checkpoints, config/strict-load/completion/
  argmin verified per run). That is the extent.
- Does-not-establish slot (paper will need each as an explicit boundary):
  universal superiority; statistical significance; causal proof that SceneFlow
  is the mechanism; causal proof that pretraining alone explains the gain;
  generalization to untested datasets; any Hailo deployment benefit;
  architectural superiority; any mechanism from disparity bins (the >=128 bin
  holds only 1,080 pixels / 0.028% and carries no substantive conclusion).
- Evidence pointers: `stage_b_armp/ARMP_CLOSURE_RECORD.md` Secs 11–13.

## 7. Candidate C — coverage vs pretraining (NOT IDENTIFIABLE)

- Scope note: will map motivation → separability test → verdict; the paper
  will need the same-knob identity and the six-route failure enumerated, not
  a new experiment.
- Motivation slot: Stage A gate rationale recorded the largest single
  identified effect as ~2x supervision under-coverage at GT>=96, classified
  as a DATA fix not an architecture change
  (`stage_a_diagnostics/mechanism_verdict.json`; `high_disparity.json`
  supervision_coverage regime_mass ge96: effective_train 0.004444737832728484
  vs eval 0.008980758110411888). Staged FT3D [64,128) pixels ~1.23 billion vs
  ~559,057 raw KITTI training pixels (~2,200x)
  (`stage_b_armp/20260918T053135Z/README.md`).
- Separability slot (paper will need each route with verdict first):
  - Route 1 (widen KITTI scale range): FAILS — moves augmentation by construction.
  - Route 2 (bias crops to high-disp): FAILS — moves dataset/domain composition.
  - Route 3 (disparity-mask via `max_disparity`): FAILS — subtractive only,
    moves coverage the wrong way.
  - Route 4 (mix FT3D into fine-tuning): FAILS — moves dataset/domain + curriculum.
  - Route 5 (alter pretraining corpus): FAILS — moves initialization itself
    (coverage absorbed into the weight point at pretraining time).
  - Route 6 (loss reweighting): FAILS — moves loss, not exposure matching.
  - Load-bearing identity: coverage is tied to the scale sampler
    (`ScaledCroppedKitti`: `disp_c[disp_c > 0] *= sx`) and, at pretraining
    time, is absorbed into initialization itself.
- Verdict slot (first-class, unsoftened): Coverage vs pretraining:
  NOT IDENTIFIABLE. No matched-coverage experiment specified; no strained
  proxy substituted.
- Evidence pointers: `stage_b_armp/CANDIDATE_C_SEPARABILITY.md` Secs 2–4, 9;
  `stage_b_armp/ARMP_CLOSURE_RECORD.md` Secs 13a, 14;
  `stage_a_diagnostics/mechanism_verdict.json`, `DIAGNOSTIC_REPORT.md`.

## 8. True GWC (NOT IDENTIFIABLE; GWC OPEN BUT UNSUPPORTED, NOT REFUTED)

- Scope note: will separate four things the paper must never merge:
  (a) D3 limitation, (b) True-GWC definition, (c) ARM-W distinction,
  (d) identifiability verdict + open status. "Not identifiable on this
  platform" is not "does not work".
- (a) D3 limitation slot: the group-averaged representation rescaled the
  full-channel dot by 1/G and produced identical G4/G8/G16 ranks BY
  CONSTRUCTION — a known gap, NOT a refutation.
- (b) Definition slot (paper will need exact mathematics + native-axis point):
  - P2A's native matching op is a SIGNED FEATURE DIFFERENCE
    (`src/models/stereonet/cost_volume.py:110-111`, method "subtract",
    volume (B,32,24,H,W)), so "G=1 -> G>1" is not the native axis; the native
    axis is operation-type replacement (signed difference → grouped inner
    product), which bundles operation swap + aggregation-width change + sign
    handling at minimum.
  - Aggregation first Conv3d weights = in_channels*32*27: 27,648 at 32,
    6,912 at G=8, delta -20,736, total 397,954 → 377,250.
    G=C=32 degenerates to an elementwise product and does not test grouping.
  - A difference is a COST, a correlation is a SIMILARITY, so the
    softmax(-cost) sign convention is a named component (negation-or-flip
    forced, second graph edit).
- (c) ARM-W distinction slot: ARM-W is MECHANISTICALLY NON-EQUIVALENT:
  `phase1/scripts/train_arm_w.py:1-14` and
  `src/models/stereonet/cost_volume.py:160-171` show
  `nn.Conv2d(32, 8, 1, groups=8, bias=False)` applied to the 32-channel
  DIFFERENCE map — learned grouped channel compression, no left/right feature
  product anywhere. ARM-W 1.7776 vs incumbent ARM-V 1.7727 = 0.0049 px, inside
  the 0.0444 px context spread; "refuted" was a preregistered decision-rule
  outcome, not a significance claim. Transfers zero evidential weight to True GWC.
- (d) Verdict slot (two statuses, kept distinct):
  - True GWC on frozen P2A: NOT IDENTIFIABLE (operation + width bundle;
    sign/init bundle; parameter-matched rescue is a second component;
    degenerate rescue is hypothesis-vacuous).
  - GWC itself: OPEN BUT UNSUPPORTED, NOT REFUTED (untested, not disproved;
    no ranking, no scoring, no winner among D1–D5 or any design).
- Evidence pointers: `stage_b_armp/GWC_IDENTIFIABILITY_DEFINITION.md` Secs 1–8, 11, 13;
  `phase1/docs/MATCHING_REPRESENTATION_AUDIT.md`;
  `phase1/docs/FINAL_PHASE1_BOTTLENECK_AUDIT.md`;
  `stage_a_diagnostics/mechanism_verdict.json`.

## 9. Post-identifiability boundary

- Scope note: will map what relaxing each frozen constraint would cost; the
  paper will need the boundary table, not a proposal. No design ranked, no
  winner, no follow-up authorized.
- Constraint slots (paper will need each with blocker + cost):
  architecture frozen; matching op fixed; aggregation width fixed; parameter
  count fixed; init shape fixed; readout sign fixed; recipe frozen (blocks C,
  does NOT block GWC — GWC fails in the graph itself); dataset frozen;
  evaluation contract frozen (holds unconditionally for comparability).
- Relaxation-cost slots: matching-op and width relaxations NECESSARY for any
  GWC-family design (with stated travelling companions); parameter/sign/init/
  normalization choices NECESSARY-to-decide and each confounded; recipe,
  dataset, and evaluation-contract relaxations SCIENTIFICALLY DANGEROUS
  alongside the graph (destroy the only anchor / reopen C confounds / break
  comparability); architecture gate reopenable only as a declared new
  multi-component question, never as quiet continuation.
- Boundary decision slot: Decision A — freeze current evidence and move toward
  paper/report synthesis (argued on unresolved-question, identifiability,
  intervention-definition, compute, deployment-relevance, provenance-burden,
  and interpretability grounds). Option B declined without any promise
  assessment of GWC.
- Evidence pointers: `stage_b_armp/POST_IDENTIFIABILITY_BOUNDARY.md` Secs 3–8, 12;
  `stage_b_armp/ARMP_CLOSURE_RECORD.md` Sec 15 (one-primary-intervention rule).

## 10. What is established

- Scope note: will list only closed, cited observations; the paper will need
  each as a bounded descriptive sentence with pointer. No extension beyond
  the stated recipe/contract.
- Inventory for the paper (each will need its citation):
  - ARM-P directional observation across three seeds + environment control,
    both selections, contract held on all 16 checkpoints (Secs 4–6).
  - Configuration/strict-load/completion/argmin verification per run.
  - Stage A gate: NO ARCHITECTURE JUSTIFIED; ~31% of the Stage-A gap below
    64 px with healthy diagnostics; D7 refutes long-range context on the
    model's own numbers.
  - Supervision-coverage DATA finding (~2x under at GT>=96) as the largest
    single identified effect.
  - Two NOT IDENTIFIABLE boundary results with independent blocking grounds
    (Secs 7–8), plus the OPEN-BUT-UNSUPPORTED status of GWC.
  - Provenance/audit facts as recorded (Secs 4–6 pointers + Sec 13).

## 11. What remains unresolved

- Scope note: will list open questions without ranking or scoring; the paper
  will need them as questions, not claims. Verbatim set lives in Sec 14.
- Headline slots (paper will expand from Sec 14 quotes):
  - Internal decomposition of the ARM-P delta (coverage vs other pretraining
    content) unmeasured.
  - Grouped feature correlation with G preserved (U2) untested either way.
  - Similarity-vs-difference at full width (D3-family) untested as an
    operation question distinct from grouping.
  - Below-64 attribution diffuse (U3 never run; existing dumps only).
  - Readout–refinement co-adaptation (U1) not identifiable under frozen protocols.
  - Any multi-component GWC bundle's accuracy + memory/MAC + Hailo-toolchain
    mapping: both halves unknown.
- Evidence pointers: `stage_b_armp/POST_IDENTIFIABILITY_BOUNDARY.md` Sec 11;
  `stage_b_armp/ARMP_CLOSURE_RECORD.md` Secs 13–15.

## 12. What is explicitly not claimed

- Scope note: will list prohibitions the paper must observe; each is a
  boundary, not modesty prose.
- Slots (paper will need each stated plainly):
  - No universal superiority; three seeds of one recipe on one split.
  - No statistical significance; no confidence intervals; 0.0444 px context only.
  - No causal proof that SceneFlow / pretraining alone is the mechanism
    (initialization bundle unmeasured internally).
  - No generalization to untested datasets; ARM-P→KITTI D3 domain result was
    confounded; symmetry control showed general degradation but did NOT show
    pretraining improves fine-tuning.
  - No Hailo deployment benefit; toolchain unverified, no on-device numbers.
  - No architectural superiority; architecture never changed.
  - No mechanism from disparity bins; >=128 bin (1,080 px, 0.028%) carries nothing.
  - No GWC verdict in either direction (OPEN BUT UNSUPPORTED, NOT REFUTED);
    no ranking/scoring/winner among GWC designs; ARM-W transfers nothing.
  - No coverage attribution of the ARM-P delta (NOT IDENTIFIABLE).
- Evidence pointers: `stage_b_armp/ARMP_CLOSURE_RECORD.md` Secs 12–13;
  `stage_b_armp/GWC_IDENTIFIABILITY_DEFINITION.md` Secs 8, 13;
  `stage_b_armp/POST_IDENTIFIABILITY_BOUNDARY.md` Secs 2, 6–8.

## 13. Evidence inventory with exact file-path citations

- Scope note: every entry verified on disk for this outline; the paper will
  need each path cited beside the claim it supports. No unverified path listed.
- Boundary/analysis records (what each evidences):
  - `stage_b_armp/ARMP_CLOSURE_RECORD.md` — authoritative closed record:
    frozen question, exact intervention/control, all four runs, contract,
    integrity/provenance, establishes/does-not-establish, confounds, U1–U5.
  - `stage_b_armp/CANDIDATE_C_SEPARABILITY.md` — Candidate C motivation,
    six-route separability audit, NOT IDENTIFIABLE verdict + decision tree.
  - `stage_b_armp/GWC_IDENTIFIABILITY_DEFINITION.md` — True-GWC definition,
    baseline op, parameter/capacity + init audits, ARM-W non-equivalence,
    NOT IDENTIFIABLE verdict.
  - `stage_b_armp/POST_IDENTIFIABILITY_BOUNDARY.md` — frozen-constraint
    boundary, D1–D5 neutral designs with bundle-level interpretation,
    deployment separation, Decision A, verbatim open questions.
  - `stage_b_armp/ENV_CONTROL_PREREGISTRATION.md` — pre-fixed delta/shift
    definitions, reference deltas, reporting rules, interpretation branches.
- Run directories (what each evidences):
  - `stage_b_armp/20260918T053135Z/` — pre-flight audit (FT3D A+C subset
    14,460 triplets / B missing / monkaa empty; staged [64,128) ~1.23B vs
    ~559,057 KITTI pixels; provenance NOT ESTABLISHED verdicts).
  - `stage_b_armp/20260918T062146Z_stage1_pretrain/` — Stage-1 checkpoint
    source (`armp_stage1_best.pth`, sha256) + 20-epoch A+C recipe.
  - `stage_b_armp/20260918T105050Z_d3_transfer/` — confounded ARM-P→KITTI D3
    diagnostic (evidence against the mechanism in D3-observable form, not a
    validation/falsification of ARM-P).
  - `stage_b_armp/20260918T114100Z_domain_symmetry/` — symmetry control
    (domain degradation general; did NOT show pretraining improves fine-tuning).
  - `stage_b_armp/20260918T142128Z_tier2_pilot/` — seed-0 run + frozen audit
    (includes `TIER2_FROZEN_AUDIT.md`: tier2_eval, contract, recipe_diff,
    provenance).
  - `stage_b_armp/20260919T012646Z_tier2_seed1/` — seed-1 run + replication
    report (includes `CORRECTIONS.json`: stale seed-literal audit, seed
    consumption proof).
  - `stage_b_armp/20260919T092454Z_tier2_seed2/` — seed-2 Kaggle-trained,
    locally-evaluated run + replication report.
  - `stage_b_armp/20260919T130843Z_seed1_kaggle_envcontrol/` — environment
    control (preregistration followed; local frozen eval; shift numbers).
  - `stage_b_armp/20260919T035455Z_kaggle_seed2/` — Kaggle bundle +
    portability record (entrypoint lesson, dataset-B fix, nesting fix, re-gate).
- Supporting diagnostics (what each evidences):
  - `stage_a_diagnostics/` — Stage-A mechanism record
    (`mechanism_verdict.json`: NO ARCHITECTURE JUSTIFIED gate, coverage DATA
    fix, matching-representation PLAUSIBLE, readout/aggregation NOT
    IDENTIFIABLE limits, D7, True-GWC UNTESTED; `DIAGNOSTIC_REPORT.md`;
    `high_disparity.json` coverage masses).
  - `phase1/docs/` — Phase-1 audits (`FINAL_PHASE1_BOTTLENECK_AUDIT.md`:
    ARM-W 1.7776 vs ARM-V 1.7727 decision-rule refutation;
    `MATCHING_REPRESENTATION_AUDIT.md`: precedent shapes, baseline difference
    op, sign convention, ARM-W compression mechanism).
- Code pointers cited but not quoted (paper will cite line-anchored):
  `src/models/stereonet/cost_volume.py:110-111` + `:160-171`;
  `src/models/stereonet/aggregation.py:26-40`;
  `src/models/stereonet/stereonet.py:108-119`;
  `src/losses/disparity.py:21-51`; `phase1/scripts/train_arm_w.py:1-14`.

## 14. Open questions carried forward verbatim

- Scope note: quoted from the source, not paraphrased; the paper will need
  them as listed questions with no winner picked. Source:
  `stage_b_armp/POST_IDENTIFIABILITY_BOUNDARY.md` Section 11
  ("REMAINING RESEARCH QUESTIONS … stated without ranking").
- Verbatim quotes:

> 1. "Does FT3D pretraining help KITTI fine-tuning because of high-disparity
>    coverage content, or because of other pretraining content (weight scale,
>    feature statistics, trajectory position)? Unseparable on frozen P2A
>    (Candidate C); the ARM-P delta's internal decomposition is unmeasured."

> 2. "Does grouped feature correlation with G preserved into aggregation help
>    or hurt under any recipe? UNTESTED (D3's rescaling was a constructional
>    gap; ARM-W was non-equivalent). Every constructible test is
>    multi-component."

> 3. "Is similarity-vs-difference at full channel width (D3) neutral, helpful,
>    or harmful? This is the only width-preserving operation question, and it
>    is not the grouping question."

> 4. "How much of the Stage-A sub-64px gap (~31% where every diagnostic looks
>    healthy) is attributable to which mechanism? Still diffuse; below-64
>    attribution from existing dumps only was the U3 candidate, never run."

> 5. "Are readout and refinement co-adapted such that neither can be judged
>    without joint retraining (the U1 candidate)? Still NOT IDENTIFIABLE
>    under frozen protocols."

> 6. "Does any multi-component GWC bundle preserve accuracy while reducing
>    memory/MACs enough to matter for the Hailo target — and does the Hailo
>    toolchain actually map products, per-group reductions, and negation
>    efficiently? Both halves unknown; the second half has zero measurements."

## 15. Recommended paper/report structure

- Scope note: outline of containers only; each entry states what the eventual
  section will need, not its argument. The paper will be assembled from
  Secs 1–14 above; no new measurement belongs in any section.
- Recommended containers (paper will fill each from the mapped evidence):
  1. Closed-state + scope (from Sec 1; will need phase/stage statuses + no-Phase-3).
  2. Intervention question + design (from Secs 2–3; will need frozen-question
     box, ARM-P/CONTROL definitions, frozen-variable checklist, both-selections rule).
  3. Replication evidence (from Secs 4–5; will need separate-rows table with
     full-precision numbers, preregistration citation, shift + branch).
  4. Bounded interpretation (from Sec 6; will need establishes / does-not-establish lists).
  5. Coverage-vs-pretraining analysis (from Sec 7; will need motivation,
     same-knob identity, six-route table, NOT IDENTIFIABLE verdict).
  6. True-GWC analysis (from Sec 8; will need D3 gap, definition + native axis,
     parameter arithmetic, ARM-W non-equivalence, NOT IDENTIFIABLE verdict,
     OPEN-BUT-UNSUPPORTED status kept distinct).
  7. Boundary + decision (from Sec 9; will need constraint table with costs,
     D1–D5 bundle-level interpretation rule, Decision A reasoning).
  8. Established / unresolved / not-claimed (from Secs 10–12; will need three
     adjacent lists, no merging).
  9. Evidence inventory + reproducibility appendix (from Sec 13; will need
     every path with per-entry "what it evidences", sha256s, contract values,
     audit notes from Sec 16).
  10. Open questions + stopping point (from Secs 14, 16; will need verbatim
     quotes + research boundary).
- Explicit exclusions for the paper: no pooled/averaged seeds; no significance
  or CIs; no GWC ranking/scoring/winner; no Phase 3; no softening of the three
  Rule-2 statuses; no mechanism from bins; no deployment claim.

## 16. Current stopping point / research boundary

- Scope note: will state where work stops and what would reopen it; the paper
  will need this as the final boundary paragraph. No next experiment is
  authorized here.
- Stopping point: evidence frozen at three seeds + environment control (all
  scored, all audited), two NOT IDENTIFIABLE verdicts with independent grounds,
  GWC OPEN BUT UNSUPPORTED / NOT REFUTED, Decision A (freeze + synthesize).
  The single zero-compute action authorized by the boundary document is this
  outline itself — now discharged.
- Audit notes the paper will need to preserve (provenance limits + lessons,
  recorded without re-derivation):
  - Five pre-existing tracked working-tree modifications (`.gitignore` + 4
    `src/models/stereonet` files, provenance unverified); `phase1/`,
    `phase2/`, `stage_b_armp/` UNTRACKED so "unmodified" rests on absence of
    tracked diffs plus git_head agreement; `d3_matching.py` MD5
    `ab29a921c121f5e404014989532172a6`.
  - First Kaggle portability gate PASSED without exercising the
    `finetune_pilot.py` entrypoint and so missed a runtime data dependency
    (`phase1/runs/arm_v/arm_v_best.pth`) and the REPO_ROOT nesting depth,
    both later fixed wrapper/bundle-side and re-gated with a 2-epoch
    entrypoint run.
  - `run_arm.py` stale "seed": 0 literal at seed 1 (cosmetic, symmetric,
    never read by training/init/data-ordering/eval).
  - `phase1/harness/frozen_eval.py` stale EXPECTED_STATE_KEYS=72 /
    EXPECTED_PARAMS=423586 producing informational keys_ok/params_ok false on
    clean checkpoints.
  - One Write-tool failure and retry during authoring of the boundary document
    (the successful artefact is what stands).
- Reopen rule: any future multi-component GWC or data program is a NEW
  research line with its own preregistration (hypothesis, frozen variables,
  success/rejection criteria), recorded separately without overwriting any
  prior run directory, checkpoint, or record — never a continuation of the
  frozen program, never justified by cost alone.

---

*End of outline. Zero GPU used. No training run. No code or model changed.
One new file created; no existing file modified.*
