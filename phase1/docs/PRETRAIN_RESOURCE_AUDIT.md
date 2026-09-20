# PHASE 1, STEP 0, TASK 2 — PRETRAINING RESOURCE AUDIT

Status: COMPLETE. All numbers below are VERIFIED LIVE on this machine 2026-09-14
by read-only probes. Nothing was extracted, downloaded, or trained.
Binding numbers reused (not re-verified here): reference ONNX EPE
1.3134470770188373 / D1 8.154366378221082 and convergence_run.pth EPE
15.395826671257934 / D1 88.39228073441733, both on 3,802,797 px
(per `phase1/docs/DETERMINISM_HARNESS.md` §4 via `phase1/harness/`).

Probes (all read-only, seconds-scale; weights never written):
- `phase1/harness/zip_audit.py` — `zipfile` name listing only.
- `phase1/harness/zip_detail.py` — second-level breakdown + marker counts.
- `phase1/harness/data_inventory.py` — directory counts + split lists via
  `src/datasets/kitti2015.py` (names only, no training).

## 1. The 1.68 GB zip — what it actually is

File: `data/kitti2015/data_scene_flow.zip`, VERIFIED LIVE size 1,681,488,619 bytes,
3,415 entries. NOT extracted (prohibition); entries listed read-only.

Second-level breakdown (VERIFIED LIVE via `zip_detail.py`):

- `training/`: 1 dir entry + image_2: 401 / image_3: 401 / disp_noc_0: 201 /
  disp_noc_1: 201 / disp_occ_0: 201 / disp_occ_1: 201 / flow_noc: 201 /
  flow_occ: 201 / obj_map: 201 / viz_flow_occ: 201 / viz_flow_occ_dilate_1: 201
- `testing/`: 1 dir entry + image_2: 401 / image_3: 401
- (`401 = 400 PNGs + 1 dir entry; 201 = 200 PNGs + 1 dir entry`)

Top level: `training`: 2,612 / `testing`: 803. Extensions: 3,400 × `.png`,
15 × dir.

SceneFlow-synthetic markers — all ZERO (VERIFIED LIVE):
`frames_cleanpass`: 0, `frames_disparity`: 0, `.pfm`: 0, `monkaa`: 0,
`flyingthings`: 0, `driving`: 0.

KITTI scene-flow markers — present (VERIFIED LIVE):
`image_2`: 802, `image_3`: 802, `disp_occ_0`: 201, `disp_noc_0`: 201,
`disp_occ_1`: 201, `flow_occ` (incl. `flow_noc`/`viz_flow_occ*`): 603,
`obj_map`: 201, `viz`: 402.

Second archive: `data/kitti2015/data_scene_flow_calib.zip`, VERIFIED LIVE
size 1,631,055 bytes, 1,208 entries: `training/`: 604 / `testing/`: 604,
8 × dir + 1,200 × `.txt`, marker `calib`: 1,206. Calibration text files only.

DECISION (VERIFIED LIVE, evidence above): `data_scene_flow.zip` is KITTI's OWN
stereo/flow benchmark archive (stereo pairs + disparity + optical flow +
object maps + visualisations + KITTI test pairs), NOT SceneFlow
(FlyingThings3D / Driving / Monkaa). The filename `data_scene_flow.zip` is the
name KITTI uses for its own archive; SceneFlow is a different, synthetic
dataset (PFM disparities, `frames_cleanpass`/`frames_disparity` layout per
`reference/upstream/extracted/StereoNet-master/dataloader/listflowfile.py:16-117`).

CONSEQUENCE: SceneFlow is NOT AVAILABLE locally. Saying it plainly as required:
no FlyingThings3D / Driving / Monkaa pixels exist on this disk. Phase 0 §13.1
("SceneFlow pretraining already on disk") is SUPERSEDED by this finding —
that assumption was wrong. No previously estimated SceneFlow size or training
duration is trusted here (all marked NOT VERIFIED in §5).

## 2. What other image/GT data exists locally, and in what quantity

Local root `data/kitti2015/` contains exactly 4 entries (VERIFIED LIVE):
`data_scene_flow.zip`, `data_scene_flow_calib.zip`, `training/`, `testing/`.

`training/` subdirs (14, VERIFIED LIVE): `calib_cam_to_cam`,
`calib_imu_to_velo`, `calib_velo_to_cam`, `disp_noc_0`, `disp_noc_1`,
`disp_occ_0`, `disp_occ_1`, `flow_noc`, `flow_occ`, `image_2`, `image_3`,
`obj_map`, `viz_flow_occ`, `viz_flow_occ_dilate_1` — i.e. the extracted KITTI
scene-flow archive content (flow/obj/viz present because the zip was unpacked
for the benchmark, not because SceneFlow exists).

`testing/` subdirs (5, VERIFIED LIVE): `calib_cam_to_cam`,
`calib_imu_to_velo`, `calib_velo_to_cam`, `image_2`, `image_3` — no GT dirs.

Counts (VERIFIED LIVE via `data_inventory.py`):

- `training/image_2/*_10.png`: 200; `training/image_2/*.png`: 400
  (200 × `_10` + 200 × `_11`; frame 11 has no disparity GT).
- `training/image_3/*_10.png`: 200; `training/image_3/*.png`: 400 (same).
- `training/disp_occ_0/*.png`: 200; `training/disp_noc_0/*.png`: 200.
- `training/calib_cam_to_cam/*.txt`: 200.
- `testing/image_2/*.png`: 400; `testing/image_3/*.png`: 400. No testing GT
  (KITTI withholds test GT — no `disp_*` under `testing/`).
- `**/*.pfm` anywhere in repo: 0 (VERIFIED LIVE glob — no SceneFlow PFM).

Splits actually drawn (VERIFIED LIVE via `Kitti2015Stereo` name lists):

- `hailo_calib`: 160 scenes, `000000_10.png` … `000159_10.png`
  (`src/datasets/kitti2015.py:34-35,127-128`).
- `hailo_val` (the 40-scene evaluation set): 40 scenes,
  `000160_10.png` … `000199_10.png`
  (`src/datasets/kitti2015.py:34-35,125-126`). Full list recorded in
  `data_inventory.py` output (000160 … 000199, every integer, all `_10`).
- `all`: 200 scenes. Overlap `hailo_calib ∩ hailo_val`: 0 — disjoint VERIFIED.
- Training slice used by `scripts/exp_train_convergence.py:128-129`:
  train on `hailo_calib`, validate on `hailo_val` (10-scene train-time subset
  in that script's `validate(..., limit=10)` at line 98/214; the frozen
  40-scene contract scoring is separate via `phase1/harness/frozen_eval.py`).

## 3. Does this repo's training implementation support a SceneFlow pretraining stage?

YES — but only in the held upstream sources, NOT in this repo's own training
script. Paths + line numbers (all read, no execution):

Upstream (held at `reference/upstream/extracted/StereoNet-master/`):

- SceneFlow pretraining entry point:
  `pretrain-sceneflow/sceneflow-pretrain.py` — datapath arg (`:22-23`),
  epochs default 20 (`:24-25`), batch 16 (`:46`), train/test DataLoaders
  (`:47-53`), `stereonet(batch_size, cost_volume_method="subtract")`
  (`:57-58`), RMSprop lr 1e-3 wd 0.0001 (`:66`; Adam commented `:65`),
  ExponentialLR γ=0.9 (`:78-80`), checkpoint
  `checkpoints/checkpoint_sceneflow.tar` (`:70-77,155-160`), smooth-L1 loss
  masked `disp < maxdisp` (`:96-108`), EPE test (`:110-132`), seed default 1
  (`:32-33,40-42`), maxdisp default 160 (`:20-21`).
- SceneFlow file layout + splits: `dataloader/listflowfile.py:16-117`
  (`frames_cleanpass` / `disparity` top dirs; monkaa `:32-48`,
  flying TRAIN/TEST `:50-87`, driving `:89-108`; returns
  train lists + FlyingThings TEST split as test).
- SceneFlow loader semantics: `dataloader/SceneFlowLoader.py:30-79`
  (PFM GT via `readpfm` `:25-27,48-49`; train random 256×512 crop `:51-61`;
  test bottom-right crop 960×544 `:69-71`; `preprocess.get_transform(augment=False)`
  `:63-65,72-74`).
- KITTI fine-tune entry point: `finetune-kitti15/finetune-kitti15.py` —
  datapath default `/datasets/data_scene_flow/training/` (`:27-28`), epochs
  default 2000 (`:29-30`), batch 16 (`:54`), loaders (`:55-61`), subtract
  volume (`:63-65`), Adam lr 1e-3 betas (0.9, 0.999) (`:72`),
  ExponentialLR constructed (`:74`) but stepped LR is via
  `adjust_learning_rate` (`:136-143` called `:166`: lr 0.001 epochs ≤200 else
  0.0001), loads `checkpoints/checkpoint_sceneflow.tar` when present
  (`:76-81`), smooth-L1 masked `disp > 0` (`:96-109`), 3-px val error
  (`:112-134`), best-`max_acc` save to `checkpoint_finetune_kitti15.tar`
  (`:186-199`), seed default 1 (`:38-44`).
- KITTI split + semantics: `dataloader/KITTIloader2015.py:17-39`
  (`image_2`/`image_3`/`disp_occ_0`, `_10` filter `:24`, train `image[:160]`
  / val `image[160:]` `:26-27` — the same 160/40 boundary as
  `src/datasets/kitti2015.py:34-35`); `dataloader/KITTILoader.py:30-96`
  (train random 256×512 crop `:59-70`, GT `/256` `:69,87`,
  `preprocess.get_transform(augment=False)` `:72-73,89-91`,
  val bottom-right 1232×368 crop `:80-87`).
- Normalisation both stages: `dataloader/preprocess.py:5-6,76-84`
  (ImageNet mean [0.485, 0.456, 0.406] / std [0.229, 0.224, 0.225],
  `augment=False` → `scale_crop` = ToTensor + Normalize; the loaders call it
  with `augment=False`, so no colour jitter in the executed path despite the
  `inception_color_preproccess` definition at `:61-73`).
- Upstream author's own results (claims, NOT re-verified — see §5):
  `README.md:19-48` (SceneFlow epoch-22 loss 3.956 / test EPE 3.496; KITTI
  300-epoch variants 80.893 / 83.527 / 90.054; 2000-epoch 93.680 after
  4.98 h finetune).

This repo's own training script does NOT support SceneFlow:

- `scripts/exp_train_convergence.py:128-131` builds only
  `Kitti2015Stereo(..., split="hailo_calib")` + `CroppedKitti` (`:60-95`)
  + `DataLoader(batch, shuffle=True, num_workers=0)` (`:131`); model
  `StereoNet(StereoNetConfig())` (`:133-135`), Adam + CosineAnnealing
  (`:136-139`); no import of `SceneFlowLoader`/`listflowfile`, no `.pfm`
  reader, no `checkpoint_sceneflow.tar` load, no two-stage logic. A
  SceneFlow stage would be NEW code (allowed in a later dispatch as a regime
  intervention with frozen architecture; NOT built here per prohibitions).

## 4. Upstream training recipe as far as verifiable from the repo

Pretrain (SceneFlow) — from `sceneflow-pretrain.py` + loaders above:
data = SceneFlow (FlyingThings3D + Monkaa + Driving via `listflowfile`);
epochs default 20; batch 16; shuffle True / workers 12 / drop_last False;
crop train random 256×512, test bottom-right 960×544; norm ImageNet
(0.485/0.456/0.406, 0.229/0.224/0.225) via ToTensor+Normalize, no jitter in
executed path; model `stereonet(batch_size=16, subtract)`; optimizer RMSprop
lr 1e-3 wd 0.0001 (Adam line commented out); scheduler ExponentialLR γ=0.9
stepped per epoch; loss smooth-L1 on `disp < 160`; seed default 1
(`torch.manual_seed` + `cuda.manual_seed` only — no numpy/python seeding);
precision NOT VERIFIED (era: PyTorch 1.0 / CUDA 10 per `README.md:50-52`;
 dtype path is FloatTensor, fp32-implied but not pinned); BN handling
 NOT VERIFIED from these files (model def at `models/stereonet.py:8` read
 only for constructor signature — full BN audit not done here); checkpoint
 every epoch to `checkpoint_sceneflow.tar` (overwrite, keeps optimizer+epoch).

Fine-tune (KITTI 2015) — from `finetune-kitti15.py` + loaders above:
init = `checkpoint_sceneflow.tar` when present else random (NOT VERIFIED which
case produced any particular export); data = KITTI 2015 training `_10`,
`disp_occ_0`, split first-160 train / last-40 val (same boundary as ours);
epochs default 2000; batch 16; train shuffle True workers 12 drop_last True,
val shuffle False workers 4; crop train random 256×512, val bottom-right
1232×368; GT scale /256; norm same ImageNet, no jitter in executed path;
model subtract; optimizer Adam lr 1e-3 betas (0.9, 0.999); scheduler:
ExponentialLR constructed but the stepped schedule is `adjust_learning_rate`
(0.001 ≤ epoch 200 else 0.0001); loss smooth-L1 on `disp > 0` (no maxdisp
mask, unlike pretrain and unlike our `gt < max_disparity` in
`src/losses/disparity.py:43-45`); val metric 3-px error rate
(`test()` `:112-134`); seed default 1 (same seeding limits); precision
 NOT VERIFIED; BN NOT VERIFIED; checkpoint policy best-`max_acc` only to
 `checkpoint_finetune_kitti15.tar` (plus resume from that path `:151-158`).

Our KITTI-only recipe (the control arm baseline) — from
`scripts/exp_train_convergence.py:56-162` + `experiments/EXP-016/config.json`
(all 22 fields): dataset kitti2015; split hailo_calib 0–159 train /
hailo_val validate; train crop random 256×512; GT `disp_occ_0`;
batch 2; precision fp32; seed 0 (`torch.manual_seed` + `np.random.seed`,
`DataLoader` shuffle True workers 0); epochs 20; optimizer Adam betas
(0.9, 0.999); LR 1e-3; schedule cosine annealing to 0 over 20 epochs; loss
masked smooth-L1 beta=1.0 valid `gt > 0 and gt < max_disparity(=176)`;
augmentation random crop + independent per-image gain jitter σ=0.1, no
h-flip; initialisation PyTorch-default random; device cuda; KNOWN DEVIATION
recorded there: no batch normalisation (model reproduces the BN-folded
exported artifact, `exp_train_convergence.py:15-21,158-162`).

## 5. NOT VERIFIED / NOT AVAILABLE (honesty markers)

- SceneFlow pixels on disk: NOT AVAILABLE (proven §1 — zero markers, zero PFM).
- SceneFlow dataset size (≈35k often quoted incl. in Phase 0 §9): NOT VERIFIED —
  previously estimated, explicitly untrusted per this dispatch; no count is
  claimed here. Requires the real archive + `listflowfile` count.
- SceneFlow download size / acquisition time: NOT VERIFIED — no download
  performed (prohibition); the only in-repo pointer is the Freiburg URL in
  `StereoNet-master/README.md:56`. No byte/time estimate is given.
- Which upstream optimizer/schedule/epoch count produced the Hailo export
  (`reference/onnx/stereonet.onnx`): NOT VERIFIED — the held scripts show
  multiple tested variants (`README.md:23-48`: RMSprop vs Adam, exp-schedule
  vs step-LR, 300 vs 2000 epochs) with no training log tying one to the
  export; `REFERENCE_BASELINE.md:23-25` already records this.
- Upstream BN placement/handling, exact augmentation executed historically,
  exact commit, exact fine-tune seed/epochs for THIS artifact: NOT VERIFIED
  (same reason; export verified by behaviour per Phase 0 §6, not by log).
- Upstream training durations on THIS machine: NOT VERIFIED — the only
  durations in-repo are the author's 4.98 h (different hardware, `README.md:41`)
  and Phase 0's prior ~22 h-scale estimate (explicitly untrusted per this
  dispatch). Our only measured timing is the KITTI-only probe (§6 of the run
  plan; SceneFlow pretraining time is NOT VERIFIED and not extrapolated).
- Upstream test-crop 960×544 vs our 368×1232 eval, and val bottom-right vs our
  top-left crop: noted as DIFFERENT, causal share NOT VERIFIED.
- Per-checkpoint loadability beyond the two harness checks: NOT VERIFIED
  (out of scope; nothing hinges on it).
- CONTRADICTION — REQUIRES RESOLUTION: none new. The resolved vendor
  metric mislabel (`stereonet.yaml:43-44` EPE vs D1) stays resolved per
  `REFERENCE_BASELINE.md:86-98`. Phase 0 §13.1's "SceneFlow already on disk"
  is corrected (not merely unresolved) by §1 above.

## 6. Commands executed and outcomes

1. `Get-Location` — OK (repo root confirmed).
2. `python phase1\harness\zip_audit.py` — OK after a parents-index fix
   (`parents[1]`→`parents[2]`): both zips listed read-only; key output
   `data_scene_flow.zip` 1,681,488,619 B / 3,415 entries, top-level
   training/testing, `.png` 3,400; `data_scene_flow_calib.zip` 1,631,055 B /
   1,208 entries, `.txt` 1,200.
3. `python phase1\harness\data_inventory.py` — OK after same fix: 200/400
   image counts, 200/200 disp counts, 200 calib, testing 400/400 no GT,
   splits 160/40/200, overlap 0, full 40-scene list.
4. `python phase1\harness\zip_detail.py` — OK: per-dir breakdown + marker
   counts (SceneFlow markers all 0; KITTI markers as in §1).
5. `python -c "import pathlib; ..."` (single-line namespace probe) — OK:
   confirmed `data/kitti2015/` holds exactly the 2 zips + training/ + testing/.
6. `python phase1\harness\timing_probe.py --steps 5 --batch 2` — OK (see run
   plan §4): 5 timed steps, mean 0.1016 s/step, weights discarded by design.
   Full outputs preserved in §4 of `REGIME_RUN_PLAN.md`.
7. Grep/glob probes (`SceneFlow|pretrain`, `\.pfm$`, `**/*.pfm`) — OK:
   confirmed upstream stage locations + zero PFM on disk.

No archive extracted, nothing downloaded, no training/fine-tuning/sweep run.
No existing file outside `phase1/` modified.
