# PHASE 0 — REFERENCE BASELINE (Objective B)

The reference ONNX is the primary accuracy reference. The recorded result
(EPE ~1.3134 px, D1 ~8.1544%, 3,802,797 valid pixels, 40 scenes) was checked,
not assumed: Phase 0 re-ran the reference evaluation read-only (in-memory
inference, no file written) and reproduced the recorded numbers EXACTLY —
every digit below under "VERIFIED LIVE" is measured in Phase 0, not quoted.

## MODEL

Hailo StereoNet export, held at `reference/onnx/stereonet.onnx`
(23,690,586 bytes; sha256 `b1a01d85…` per `reference/manifest.json:22-27`).
Runtime used here: onnxruntime 1.27.0, CPUExecutionProvider (provider choice
is not part of the accuracy contract; EXP-005 used the same provider).

## WEIGHTS

Vendor pretrained weights (SceneFlow pretraining -> KITTI fine-tuning
lineage per the upstream sources held in `reference/upstream/extracted/`,
e.g. `StereoNet-master/pretrain-sceneflow/sceneflow-pretrain.py` and
`StereoNet-master/finetune-kitti15/finetune-kitti15.py`; training-time details
such as batch norm are folded at export and NOT re-verified here).
Provenance chain below the export (exact upstream commit, exact fine-tune
seed/epochs for THIS artifact): NOT VERIFIED — the export is verified by its
behavior, not by its training log, which the repository does not hold.

## DATASET

KITTI 2015 training set, local copy `data/kitti2015`
(200 `_10` frames present; 200 `disp_occ_0` files present; verified by
directory count in Phase 0).

## SCENES

40 scenes: `hailo_val` = training scenes 160–199, frame `_10`
(`src/datasets/kitti2015.py:125-126`; `experiments/EXP-005/config.json:2-3`).

## VALID PIXELS

3,802,797 (`gt > 0`, `disp_occ_0`, 1/256, pooled over 40 scenes).
VERIFIED LIVE: 3802797.

## EPE (frozen contract: 1/256, pooled, pixels)

1.3134470770188373 px. VERIFIED LIVE — exact match to the recorded
`experiments/EXP-005/metrics.json:62-80` (`kitti_official_d1_all`) value
1.3134470770188373.

## D1 (frozen contract: 1/256, pooled, percent)

8.154366378221082%. VERIFIED LIVE — exact match to the recorded
`experiments/EXP-005/metrics.json:62-80` value 8.154366378221082.

## RMSE

2.5829573145561677 px. VERIFIED LIVE (recorded same location).

## BAD1 / BAD2 / BAD3

38.89637022433751% / 16.079375259841637% / 8.639640769675584%.
VERIFIED LIVE (recorded same location).

## GT SCALE

1/256 (official). GT SOURCE: `disp_occ_0`.

## EVALUATION PROTOCOL

`kitti_official_d1_all` as implemented in `scripts/exp_reproduce_hailo.py:62-70`
(1/256, occluded=True, pooled) + `src/evaluation/metrics.py:52-84`
(`disparity_metrics`, pooled). Predictions scored raw (`exp_reproduce_hailo.py:149-151`;
Phase 0 live run: same). Scene list, crop, mask, units per
`BASELINE_CONTRACT.md`. Full protocol label:
KITTI2015 / hailo_val 40 scenes / 3,802,797 px / 1/256 / D1-all pooled /
EPE+D1 / reference/onnx/stereonet.onnx.

## HAILO-EXACT RESULT (separate protocol, never substituted)

HAILO-EXACT RESULT: 8.2237% D1 (8.223686464356268%, per-image mean, 1/255,
`disp_occ_0`, same 40 scenes, same checkpoint). VERIFIED LIVE — exact match
to `experiments/EXP-005/metrics.json:24-42` (`hailo_exact.headline_percent`).
Never call 8.2237 an EPE. Companion isolation figure, also VERIFIED LIVE:
same per-image protocol at 1/256 scores 8.168465183586012%, exact match to
`hailo_scale_kitti_gt` (`metrics.json:43-61`).

## METRIC-LABEL DISCREPANCY (still present in the vendor configuration)

`reference/hailo_model_zoo/stereonet.yaml:43-44` declares
`eval_metric: EPE` with `full_precision_result: 8.223`. The quantity 8.223 is
a D1 outlier rate in percent, not an end-point error in pixels: the true EPE
on the same predictions is 1.3134 px (verified above), and the vendor
evaluator computes a per-image outlier rate (reimplemented at
`src/evaluation/metrics.py:87-110`, `hailo_image_outlier_rate`).
CONTRADICTION — REQUIRES RESOLUTION is resolved in favor of the evaluator:
quote 8.223 only as D1-%, and never compare it with any EPE figure.
Also recorded: `results/reproduction/variants.json` is byte-identical in
content to the EXP-005 variants block (checked in Phase 0), so there is a
single consistent reference record, not two competing ones.

## WHAT WAS NOT RE-RUN

`reference/stereonet.hef` behavior (no device; bytes only) and the int8 ONNX
artifact remain NOT VERIFIED in Phase 0, as in all prior phases.
