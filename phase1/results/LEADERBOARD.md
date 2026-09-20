# Phase 1 training-regime leaderboard (frozen contract ONLY)

Reference = 1.3134471 EPE

| Model | Main Change | Pretraining | Seed | EPE | D1 | dEPE vs Ref | Status |
|---|---|---|---|---|---|---|---|
| Reference ONNX | — | upstream (export) | — | 1.3134471 | 8.1543664% | 0.0000000 | frozen reference (harness re-score, DETERMINISM_HARNESS §4) |
| ARM K20 (convergence_run.pth) | 20-epoch baseline | none | 0 | 15.3958267 | 88.3922807% | +14.0823796 | frozen baseline (harness re-score, NOT retrained) |
| ARM K (200-epoch KITTI) | budget 20→200 epochs | none | 0 | 5.5271927 | 45.6085613% | +4.2137456 | scored (best-val snap, epoch 180); final-epoch snap 5.6693759 |
| ARM K-LR (200-epoch, step LR) | scheduler cosine→step | none | 0 | 12.7830 | 86.1812% | +11.4695 | scored (best-val snap, epoch 0; final-epoch snap 12.8253) |
| ARM K600 (600-epoch KITTI) | budget 200→600 epochs | none | 0 | 8.5583436 | 70.1146814% | +7.2448965 | scored (best-val snap, epoch 350; final-epoch snap 8.8405404) |
| ARM D (Driving-pretrain → 200-epoch KITTI) | init random→Driving-pretrained | Driving only (4400 pairs) | 0 | 10.5748918 | 77.0048204% | +9.2614447 | scored (best-val snap, epoch 155; final-epoch snap 10.6663718) |
| ARM G (200-epoch + asymmetric photo aug) | photo brightness/contrast/gamma U(0.8,1.2), indep. L/R, always on | none | 0 | 10.4747403 | 78.6592605% | +9.1612932 | scored (best-val snap, epoch 140; final-epoch snap 11.2448508); REFUTED vs ARM K (+4.9475476, bar 1.0) |
| ARM S (200-epoch, cost-volume shift→right) | cost_volume_shift none→right (LEFT-frame disparity search) | none | 0 | 10.8866094 | 77.0809486% | +9.5731623 | scored (best-val snap, epoch 140; final-epoch snap 10.8518055); REFUTED vs ARM K (+5.3594167, bar 1.0) |
| ARM T (200-epoch, scale-invariant readout) | regression_normalize False→True (cost standardised over disparity axis pre-softmax) | none | 0 | 8.2702542 | 68.2488705% | +6.9568071 | scored (best-val snap, epoch 165; final-epoch snap 8.4647908); REFUTED vs ARM K (+2.7430615, bar 1.0) |
| ARM U (200-epoch, disparity search + scale-invariant readout) | cost_volume_shift none→right AND regression_normalize False→True (each measured alone in ARM S / ARM T) | none | 0 | 2.2866369 | 15.3067597% | +0.9731898 | scored (best-val snap, epoch 199; final-epoch snap identical weights, same score); ACCEPTED vs ARM K (−3.2405558, bar 1.0); stereo path LIVE (shift→none substitution moves output ~57.5 px); CONFIRMED by multi-seed replication (seeds 1, 2 below) |
| ARM U-S1 (ARM U recipe, seed 1) | seed 0→1 only (train_arm_u_seed.py diff is 2-line env plumbing) | none | 1 | 2.4207853 | 18.4118689% | +1.1073382 | scored (best-val snap, epoch 175; final-epoch snap 2.4651119); ACCEPTED vs ARM K (−3.1064074, bar 1.0); stereo path LIVE (shift→none substitution 2.4208→21.3743 EPE) |
| ARM U-S2 (ARM U recipe, seed 2) | seed 0→2 only (train_arm_u_seed.py diff is 2-line env plumbing) | none | 2 | 2.4121953 | 17.3953803% | +1.0987482 | scored (best-val snap, epoch 145; final-epoch snap 2.3018475); ACCEPTED vs ARM K (−3.1149974, bar 1.0); stereo path LIVE (shift→none substitution 2.4122→26.3412 EPE) |
| ARM V (200-epoch, finer disparity sampling) | downsample_levels 4→3 AND num_disparities 12→24 (stride 1/16→1/8, spacing 16→8 px, range 176→184 px; shift=right + regression_normalize=True kept) | none | 0 | 1.8903392 | 10.7425666% | +0.5768921 | scored (best-val snap, epoch 180; final-epoch snap 1.9043133); ACCEPTED vs ARM U mean (−0.4828666, bar 0.40 = 3x ARM U 3-seed spread); stereo path LIVE (shift→none substitution 1.8903→39.5550 EPE); 397,954 params (NOT parameter-free) |
| ARM V-S1 (ARM V recipe, seed 1) | seed 0→1 only (train_arm_v_seed.py diff is 2-line env plumbing) | none | 1 | 1.5493088 | 9.3507752% | +0.2358617 | scored (best-val snap, epoch 199; final-epoch snap identical weights, same score 1.5493088); stereo path LIVE (shift→none substitution 1.5493→30.8966 EPE); NO verdict here (manager decides) |
| ARM V-S2 (ARM V recipe, seed 2) | seed 0→2 only (train_arm_v_seed.py diff is 2-line env plumbing) | none | 2 | 1.8785827 | 10.7135353% | +0.5651356 | scored (best-val snap, epoch 180; final-epoch snap 1.8894581); stereo path LIVE (shift→none substitution 1.8786→34.9067 EPE); NO verdict here (manager decides) |

Frozen contract: KITTI 2015 / hailo_val 40 scenes / disp_occ_0 / 1/256 /
368x1232 top-left crop / gt>0 / pooled (3,802,797 px). Scored ONLY through
`phase1/harness/frozen_eval.py`. No historical H1/H2/Phase-2 figure appears here.

Rejected before execution: ARM F (horizontal flip) — KITTI left-view-only GT
makes mirroring inexact by construction (flipped pair cannot match stored
left-view disparities); no code written, no training run. Full note in
`phase1/runs/arm_g/RESULTS.md`.

## ARM V multi-seed decision (manager, 2026-09-15)

ARM V 3-seed frozen-contract EPE: 1.8903392 (s0), 1.5493088 (s1), 1.8785827 (s2).
Mean 1.7727436. Spread (max-min) 0.3410304.

Decision rule as pre-registered in `phase1/runs/arm_v/results.json`: compare the
ARM V mean against the ARM U 3-seed MEAN 2.3732058, accept bar 0.40 px.

  improvement = 2.3732058 - 1.7727436 = 0.6004622 px  >  0.40 bar

ARM V is CONFIRMED and replaces ARM U as the Phase-1 incumbent. The ranges do not
overlap: ARM V's worst seed (1.8903392) beats ARM U's best seed (2.2866369).
Stereo path LIVE in all three seeds (shift->none substitution: 39.5550 / 30.8966 /
34.9067 EPE). 397,954 params, 25,632 fewer than ARM U.

Gap to reference from the ARM V mean: 1.7727436 - 1.3134471 = 0.4592965 px.

Recorded caveat for future bars: ARM V's seed spread (0.3410304) is 2.5x ARM U's
(0.1341484), and is itself comparable to the 0.40 bar it just cleared. A next bar
set at 3x the ARM V spread would be 1.02 px, which EXCEEDS the 0.459 px that
remains to the reference. Future acceptance bars must therefore be re-derived from
the then-current spread and the then-remaining gap, not carried over from here.
