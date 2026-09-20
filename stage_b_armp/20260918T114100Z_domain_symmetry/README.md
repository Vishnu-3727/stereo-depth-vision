# Domain-symmetry control (frozen inference only, no training)

## Why this shape — substitution justification

The literal request, "run frozen P2A -> FT3D through the D3 machinery", is
infeasible under the evaluation contract: `stage_a_diagnostics/scripts/
d3_matching.py` hardcodes `Kitti2015Stereo(..., split='hailo_val',
disparity_scale=256.0, occluded=True)` (line 187), asserts `len(ds) == 40`
(line 189), and asserts shape `(368, 1232)` (line 195). FT3D is 540x960,
14,460 triplets, and has no occlusion masks. Forcing it through would need an
anisotropic resize that rescales disparity and changes valid-pixel semantics —
materially different preprocessing, hence uninterpretable. So the EPE-based
symmetry control below is run instead. `d3_matching.py` was not used and is
byte-identical before and after (md5 `ab29a921c121f5e404014989532172a6`).

## The 2x2 table

|                       | KITTI hailo_val EPE | FT3D pretrain-val EPE |
|-----------------------|---------------------|-----------------------|
| P2A (seed 0)          | 1.4149796 (KNOWN)   | **9.0692242** (CELL 1, measured; D1 39.4329%) |
| ARM-P stage1 best     | **4.8172650** (CELL 2, measured, ZERO-SHOT; D1 35.9763%) | 3.0486 (KNOWN) |

Known-cell sources: P2A KITTI from
`phase2/runs/p2a_scale_coverage/p2a_eval_best.json` seed 0 (1.4149795737);
ARM-P FT3D from `stage_b_armp/20260918T062146Z_stage1_pretrain/
pretrain_log.jsonl` epoch 5 best-by-pretrain-val (3.0486497).

CELL 1 method: the EXACT `validate()` path imported from
`stage_b_armp/20260918T062146Z_stage1_pretrain/scripts/train_armp_stage1.py`
(not copied, not altered), same FT3D pretrain-val holdout slice (rng seed 0,
first 10 holdout ids — asserted identical to the manifest ordering, which is
the slice that produced the 3.0486 figure; valid pixels 5184000 match the
pretrain log), same limit (`VAL_SCENES` = 10), same mask (gt>0 AND gt<184).
Only the weights change: P2A seed 0
(`phase2/runs/p2a_scale_coverage/p2a_best.pth`).

CELL 2 method: `stage_b_armp/20260918T062146Z_stage1_pretrain/checkpoints/
armp_stage1_best.pth` on KITTI `hailo_val` under the frozen evaluation
contract, via a line-for-line mirror of the scoring in
`phase2/scripts/eval_p2a.py` (`eval_p2a.py` NOT modified; dataset split, GT
scale, valid mask, pooled metrics and contract guard imported unmodified from
`phase1.harness.frozen_eval`). Contract guard: match = true (40 scenes,
3802797 valid pixels). This is a ZERO-SHOT score of a model that has never
seen KITTI. It is NOT an ARM-P result and must never be compared to
1.4409826 as if it were a fine-tuned outcome.

## Transfer penalty (each direction, difference and ratio)

- P2A: in-domain KITTI 1.4149796 -> out-of-domain FT3D 9.0692242:
  difference **+7.6542446**, ratio **x6.409**.
- ARM-P: in-domain FT3D 3.0486 -> out-of-domain KITTI 4.8172650 (zero-shot):
  difference **+1.7686650**, ratio **x1.580**.

P2A ALSO degrades substantially outside its training domain — in fact its
penalty is the larger of the two, in both absolute (+7.65 vs +1.77) and
relative (x6.41 vs x1.58) terms. The two penalties are of comparable
character in the only sense that matters here (both are substantial
out-of-domain degradations), but they are NOT paired measurements from the
same domain: the domains, image statistics and GT densities differ. This is
a directional, asymmetric control and is described that way.

## ARM-P zero-shot KITTI detail (CELL 2)

- Pooled: EPE 4.8172650, D1 35.9763%, 3802797 px, contract match true.
- GT>=96 stratum: px 34152, EPE 19.9176, signed_err -18.9295, mean_gt
  109.7183, mean_pred 90.7888, slope +1.0883, intercept -28.6142, R2 0.1723,
  Pearson r 0.4151.
- Per-bin EPE (D1 bin edges from `eval_p2a.py`, MIN_BIN 1000):
  0-16: 5.2549 (D1 35.01%); 16-32: 4.1878 (32.44%); 32-48: 4.6066
  (35.05%); 48-64: 4.8168 (39.46%); 64-80: 6.2112 (47.22%); 80-96: 7.1330
  (46.82%); 96-112: 22.8843 (76.80%); 112-128: 14.5238 (64.55%);
  128-144: 20.4654 (75.35%); 144-160: INSUFFICIENT (9 px).
  Full rows in `raw/cell2_armp_kitti.json`.

## Classification

**DOMAIN CONFOUND PLAUSIBLE** — P2A also transfers poorly out of domain
(KITTI 1.4150 -> FT3D 9.0692, +7.6542, x6.409), so the earlier ARM-P D3
negative cannot discriminate between a weak frozen representation and a
pure domain shift.

Not claimed: ARM-P is neither validated nor falsified. Nothing is claimed
about EPE after fine-tuning (not tested). The earlier ARM-P D3 result keeps
its status: OBSERVED DEGRADATION / MECHANISM NOT SUPPORTED / CAUSAL
CONFIDENCE LOW / DOMAIN-CONFOUNDED / REASSESS. This is not Phase 3 and no
historical verdict is rewritten.

## Integrity

- `d3_matching.py` md5 before AND after: `ab29a921c121f5e404014989532172a6`
  (unchanged; not used, proof of non-interference).
- Both models asserted P2A architecture before scoring:
  downsample_levels=3, num_disparities=24, shift='right',
  regression_normalize=True, 397954 params, strict load (70 keys, no
  missing/unexpected). Abort on mismatch.
- sha256: P2A `0868ffd1...fbb6033` (full value in `symmetry_control.json`);
  ARM-P `3ae6fb3b...046176d9F1A29BE7`.
- FT3D slice used for CELL 1 is IDENTICAL to the one that produced the
  3.0486 figure (same first-10 scene list, same manifest ordering, same
  5184000 valid pixels). Had it differed, the run would have aborted.
- No training, no fine-tuning, no sweeps. No existing repo file modified
  (see `git status`). All new files under this directory. No invented
  numbers; everything not measurable would read
  'NOT MEASURABLE WITH CURRENT ARTIFACTS' (nothing fell in that bucket).

## Runtime

- Wall clock 11.1 s (frozen inference only), device cuda, GPU NVIDIA
  GeForce RTX 4060 Laptop GPU, torch 2.7.0+cu128, numpy 2.5.1,
  python 3.12.9.

## Files

- `README.md` (this file)
- `symmetry_control.json` (2x2 table, penalties, integrity fields)
- `raw/cell1_p2a_ft3d.json`, `raw/cell2_armp_kitti.json`
- `scripts/run_symmetry.py` (the frozen-only runner)
