# RUN A — deterministic H2 / E3b arm-A baseline, 200 epochs

Record: `experiments/STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA/`,
`run_A_meta.json`, `eval40_run_A.json`.

```
python phase2/diagnostics/determinism/stageb_deterministic_baseline.py \
    train --run A --out <this directory>
```

## Preflight — all checks passed before a single step

`MEASURED` (`results/preflight.json`):

| check | result |
|---|---|
| **initial weight sha** | **`c4d02385e3da28a5`** — matches the established E3b arm-A initialisation exactly |
| same hash from `build_model` alone | `c4d02385e3da28a5` (truncation to six blocks changes nothing) |
| hash method | sha256 of concatenated `model.parameters()` fp32 bytes, first 16 hex chars (Stage A's method) |
| parameter count | 423,586 |
| MACs @256×512 | 16,199,516,160 (`thop`) |
| refinement blocks / dilations | 6 / `[1, 2, 4, 8, 1, 1]` |
| `cost_volume_shift` | `left` |
| regression | `StandardisedDisparityRegression` |
| seed / epochs | 0 / 200 |
| train / validation split | `hailo_calib` scenes 0–159 / `hailo_val` 160–199, first 10 validated |
| crop | 256×512 |
| surviving weights bit-identical across arms | true |
| `git diff phase-1-frozen -- src scripts` | empty |
| H2 reference checkpoint sha256 | `e3d48021d7f6a3d4…` — unchanged |
| **preflight PASS** | **true** |

## Deterministic controls, as recorded in the run's own metadata

```json
{"torch.use_deterministic_algorithms": true,
 "torch.backends.cudnn.deterministic": true,
 "torch.backends.cudnn.benchmark": false,
 "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
```

`MEASURED` — no operation refused to run; no determinism error was raised or
suppressed; the record's `status` is `completed` and `error` is `null`.

## Training

`MEASURED`:

| | value |
|---|---|
| epochs completed | **200 / 200** |
| aborted | `null` |
| wall clock | **4,948.997 s = 82.48 min** (24.745 s/epoch) |
| epoch 0 (untrained) | val EPE 31.9576479, D1 95.1349394, loss 31.0579514 |
| epoch 200 | val EPE **3.8346742**, D1 **23.0470937**, val loss 3.3030083 |
| final train loss | 0.9037397 |
| gradient norms | median 21.9161959, max 308.0343933 |
| NaN / Inf | **none** |
| batches with gradient norm > 1e4 | **0** |
| matching-path gradient | **100 % of 16,000 batches** (median norm 7.3321800, max 306.6581794) |

### Gates

`MEASURED` — epoch-10 gate **PASS**: val EPE 31.9576 → 9.0015 (improvement
22.9562), matching gradient 100 %. Right-image ΔD1 at epoch 10 was +31.4757 and
matching-map ΔD1 +33.4390 — **recorded, not enforced**, per
`EXP-H2-EMERGENCE-001`.

`MEASURED` — epoch-100 collapse check **PASS**: EPE 9.0015 → 4.7963, no
regression, matching gradient 100 %, disparity std 17.1827, entropy 1.7561.

### Late window — the ten recorded validation points

`MEASURED` (epochs labelled 0-indexed by the harness, i.e. 155–200 completed):

| epoch | EPE (px) | D1 (%) |
|---:|---:|---:|
| 154 | 4.0916932497 | 25.1677005019 |
| 159 | 4.0510232251 | 25.3887224032 |
| 164 | 3.7946955019 | 23.5356627487 |
| 169 | 3.7465212189 | 22.7692314305 |
| 174 | 3.7456174701 | 22.5713109919 |
| 179 | 3.8958069064 | 23.4370248754 |
| 184 | 3.8243570282 | 22.9612421925 |
| 189 | 3.8322391157 | 22.8708778663 |
| 194 | 3.8535638289 | 23.2052581078 |
| 199 | 3.8346742341 | 23.0470936748 |
| **mean ± sd** | **3.8670192 ± 0.1113151** | **23.4954125 ± 0.9342410** |

## Stereo functionality at epoch 200

`MEASURED` — **STEREO_FUNCTIONAL**, every constraint cleared by 3–4×:

| probe | value | threshold |
|---|---:|---:|
| right-image dependence (worst case) | **+80.8127994** | ≥ +20.0 |
| matching-map dependence (worst case) | **+66.1876924** | ≥ +20.0 |
| matching-path gradient | 100 % | ≥ 95 % |
| softmax entropy | 1.7754723 nats | > 0.5 |
| final disparity std | 18.0000028 px | > 1.0 |
| NaN / Inf | none | none |

## Artefacts

| | |
|---|---|
| final weight sha (concatenated parameters) | **`3ad382c50d413ad2`** |
| checkpoint file sha256 | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` |
| weight snapshots | 6 (epochs 10, 20, 50, 100, 150, 200) |

## 40-scene evaluation

See `RESULTS.md` §"FULL-40-SCENE BASELINE"; per-scene rows in
`eval40_run_A.json`.
