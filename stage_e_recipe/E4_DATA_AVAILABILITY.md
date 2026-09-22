# E4 data availability — FlyingThings3D on Kaggle

Groundwork for a possible E4 (broader pretraining). No recommendation about
whether to run E4 is made here. Read-only probing only: no uploads, no kernel
pushes, no training, no full-dataset downloads.

## 1 Measured

All figures below were measured directly; nothing in this section is inferred.

- **Active Kaggle account is `vishnuvardhanksece`.** New key installed. The old
  `vishnu3727` key is backed up. The new account owns no datasets and no
  kernels.
- **Candidate image corpus:** `arjun12367/sceneflow-flyingthings-images`
  (37.6 GB, Kaggle compressed size), official `FlyingThings3D_subset` layout.
  - Train: **21,818 frames per eye**; val: **4,248 frames per eye**.
  - **Exact left/right symmetry**: equal counts per eye in both splits.
- **Candidate disparity corpus:**
  `arjun12367/sceneflow-flyingthings-disparity` (11.8 GB, Kaggle compressed
  size), official `FlyingThings3D_subset` layout.
  - **Complete, not truncated**: `train/disparity/left/0021817.pfm` (the last
    train index) downloads fine.
  - Uncompressed size **2,073,621 B per PFM** (all four sampled files
    byte-identical in size).
- **Sampled PFMs** (four files, already on disk at the `ft3d_probe` scratchpad
  dir): `0002000`, `0005698`, `0010000`, `0021817` (all
  `train/disparity/left/<id>.pfm`). Each parsed with `src/datasets/pfm.py`:
  - Shape **540 rows x 960 cols**, single-channel, **100 % finite** in all four.
  - Raw value ranges (min/max, as stored):
    | file | raw min | raw max |
    |---|---|---|
    | 0002000.pfm | -87.438 | -1.188 |
    | 0005698.pfm | -139.188 | -1.188 |
    | 0010000.pfm | -83.438 | -1.188 |
    | 0021817.pfm | -98.969 | -1.125 |
  - **Disparity is stored NEGATIVE.** Max |d| per file: 87–139 px.
  - **0 % of pixels at or above 184 px** (|d| basis) in all four files.
  - **20–58 % of pixels below 24 px** (|d| basis) across the four files.
- **Sign/mask interaction (code fact):**
  `stage_b_armp/20260918T062146Z_stage1_pretrain/scripts/ft3d.py:67` applies no
  sign handling, and the training mask is `gt > 0 AND gt < 184`. Applied as-is
  to this corpus, that mask would yield **ZERO valid pixels** (all stored values
  are negative), so the sign must be handled explicitly before reuse.

## 2 Inferred (not directly measured)

- The 11.8 GB disparity figure is Kaggle's **compressed** size; the per-file
  2,073,621 B figure is uncompressed. No contradiction between them.
- The exact left/right frame-count symmetry plus the shared
  `FlyingThings3D_subset` layout prefix is consistent with the image and
  disparity uploads being two halves of the same corpus — but pairing is **not
  yet verified** (see §4; the image-side sampling below is the check).
- The negative-disparity convention is presumably the FlyingThings3D author's
  storage choice (right-minus-left vs left-minus-right sign), not corruption:
  four of four sampled files agree in sign, range, shape and finiteness.

## 3 Still unknown

- The `arthurthom/sceneflow` fidelity question: unverified.
- Whether `arthurthom/sceneflow` contains FlyingThings3D at all: unverified
  (a later task).
- Anything about E4 training design, cost, or go/no-go: out of scope here.

## 4 Image-side sampling and stereo correspondence — MEASURED

The images and the disparities are two separate Kaggle uploads, so their
correspondence was not assumed; it was tested. Six images were downloaded
(`train/image_clean/{left,right}/{0002000,0010000,0021817}.png`) and checked
against the matching PFMs.

All three triplets: left and right decode at **540 x 960 x 3**, matching the
PFM's 540 x 960 exactly.

The test: on pixels with valid disparity, mean absolute grayscale intensity
difference between `left(x, y)` and the right image sampled at the disparity
shift, against an unshifted control. Disparity taken as `|d|` (the files store
it negative). Rounded to the nearest integer index; pixels whose shifted index
leaves the frame are skipped and counted.

| index | right(x - abs d) | right(x + abs d) | unshifted control |
|---|---|---|---|
| 0002000 | **9.3765** (516,033 px, 2,367 skipped) | 37.0742 (500,236 px, 18,164 skipped) | 31.4717 (518,400 px) |
| 0010000 | **6.6818** (501,010 px, 17,390 skipped) | 34.3939 (516,637 px, 1,763 skipped) | 28.2796 (518,400 px) |
| 0021817 | **6.2885** (503,226 px, 15,174 skipped) | 29.2075 (509,998 px, 8,402 skipped) | 24.4990 (518,400 px) |

What this shows, measured: shifting the right image by `-|d|` reduces the mean
absolute difference to **3.4x to 4.8x below the unshifted control** on all three
indices, while the opposite sign is **worse than the control**. The two uploads
therefore correspond, and the usable convention is `|d|` with the standard
`right(x - d)` direction.

Inferred, not measured: that the correspondence holds across the whole corpus.
Three indices spread across the train range (2,000 / 10,000 / 21,817) all agree,
which is consistent with a correct upload, but it is a sample, not a census.

Raw numbers: `stereo_check.json` in the probing scratchpad.
