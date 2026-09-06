# Stereo fundamentals

Everything here is prerequisite to reading the rest of the knowledge base. It
starts in plain language and then gets formal. The numbers at the end are
measured on the real KITTI 2015 calibration, not quoted from anywhere.

---

## 1. The plain version

Hold a finger in front of your face and blink one eye, then the other. The
finger jumps. Move it further away and the jump gets smaller. That jump is
**disparity**, and its size is the entire basis of stereo depth: near things
shift a lot between the two views, far things barely shift at all.

A stereo camera is two cameras a fixed distance apart. To measure depth you
need to answer, for every pixel in the left image, *where did this same piece of
the world end up in the right image?* The horizontal distance between those two
positions is the disparity. Divide a constant by it and you have distance in
metres.

That is the whole idea. Everything difficult about stereo comes from one step:
finding, reliably, which pixel in the right image corresponds to which pixel in
the left. Blank walls have nothing to match. Repeating patterns match in many
places equally well. Things visible to one camera and hidden from the other
cannot be matched at all. A neural stereo network is a machine for making that
one step work; the rest is fixed geometry.

---

## 2. The setup

**Baseline `B`** — the distance between the two camera centres, in metres. A
wider baseline gives larger disparities and therefore better depth resolution at
range, at the cost of more occlusion (more of the scene is visible to only one
camera) and a larger minimum range.

**Focal length `f`** — expressed in *pixels*, not millimetres. It is the
physical focal length divided by the pixel pitch, and it is what appears in the
rectified projection matrix. Using pixels keeps the geometry independent of
sensor size.

**Epipolar geometry** — for any point in the left image, its match in the right
image is constrained to a single line, the epipolar line. This reduces the
search from two dimensions to one. It is the reason stereo is tractable at all.

**Rectification** — a pair of homographies applied to both images so that all
epipolar lines become horizontal and aligned to the same image row. After
rectification, the match for pixel `(u, v)` in the left image lies at
`(u - d, v)` in the right image, for some `d >= 0`. Same row, shifted left.
Every learned stereo network in this project assumes rectified input; feeding it
unrectified images breaks the assumption silently, producing a plausible-looking
and completely wrong disparity map.

**Disparity `d`** — that horizontal shift, in pixels.

**Disparity range** — the network only searches `d` in `[0, D_max)`. Anything
closer than `fB / D_max` is outside the search and cannot be measured. Choosing
`D_max` is choosing a minimum range, and it directly sets the cost of the
matching stage. StereoNet's choice is analysed in `cost_volume_analysis.md`.

**Occlusion** — surfaces visible in one image and not the other, typically at
depth discontinuities where a foreground object hides the background from one
camera. There is no correct disparity for an occluded pixel; any value the
network outputs there is an interpolation, not a measurement. KITTI ships
separate `disp_occ_0` and `disp_noc_0` ground truth precisely so results can be
reported with and without these pixels.

**Invalid pixels** — pixels with no ground truth. In KITTI the ground truth
comes from accumulated LiDAR and is sparse: roughly a third of pixels have a
value, the sky and reflective surfaces have none. Ground truth of exactly `0`
means *no data*, not *zero disparity*. Treating it as a measurement is a
straightforward way to produce a wrong benchmark number.

**Subpixel disparity** — disparity does not have to be an integer. A matcher
that only produces whole-pixel disparities quantises depth into shells, and at
range those shells are enormous (see section 5). Sub-pixel precision is not a
refinement; at distance it is the difference between a depth measurement and a
depth bucket. The original StereoNet paper's central claim is precisely that its
learned sub-pixel precision is what lets it get away with a very low resolution
cost volume [SR-010].

---

## 3. Disparity to depth

For a rectified pair:

```
Z = f * B / d
```

`Z` is depth along the optical axis in metres, `f` in pixels, `B` in metres,
`d` in pixels. `f * B` is a single constant of the rig — call it `fB`, in
pixel-metres — so depth is just `fB / d`.

Three consequences follow immediately, and they shape everything downstream:

1. **Depth is inversely proportional to disparity.** The relationship is a
   hyperbola, not a line. Equal steps in disparity are not equal steps in depth.
2. **Disparity zero is not depth infinity, it is depth undefined.** A zero or
   negative disparity carries no information. Code that clamps it to a small
   positive number produces a large finite depth that looks like data and is
   not. `src/geometry/stereo.py` returns invalid instead, and the unit tests pin
   that behaviour.
3. **All the depth resolution is at small disparity, where there is least of
   it.** Most of the world beyond a few tens of metres is compressed into the
   first few pixels of the disparity range.

---

## 4. How disparity error becomes depth error

Differentiate `Z = fB/d` with respect to `d`:

```
dZ/dd = -fB / d²
```

Substituting `d = fB/Z` to express it in terms of depth:

```
dZ/dd = -Z² / (fB)
```

so for a disparity error `Δd`, the first-order depth error is

```
ΔZ ≈ (Z² / fB) · Δd
```

**Depth error grows with the square of depth for a fixed disparity error.** A
matcher whose accuracy is uniform in pixels has depth accuracy that degrades
quadratically with distance. This is geometry; no architecture avoids it. What
an architecture *can* change is `Δd` — and because the penalty is quadratic,
sub-pixel improvements at range are worth far more than the same improvement up
close.

The linear form above is an approximation. Because `Z(d)` is convex, the true
error is asymmetric: underestimating disparity pushes a point much further away
than overestimating it brings it closer, and past a certain error the disparity
reaches zero and the depth becomes undefined rather than large. Both forms are
implemented; `depth_error_exact` reports the asymmetric truth and `inf` for the
undefined case.

---

## 5. The numbers, on real KITTI calibration

**[MEASUREMENT: EXP-003]**, reproducible with
`python scripts/depth_error_curve.py`.

Parsed from all 200 KITTI 2015 training scenes [SR-020]. The rig is close to
constant across scenes: focal length 707.0–721.5 px, baseline 0.5301–0.5373 m.
Median values:

```
f  = 721.5377 px
B  = 0.5327 m
fB = 384.3815 px·m
```

### What the disparity actually is, at each depth

| Depth | True disparity |
|---:|---:|
| 5 m | 76.88 px |
| 10 m | 38.44 px |
| 20 m | 19.22 px |
| 50 m | 7.69 px |
| 80 m | 4.80 px |

Half of KITTI's usable depth range lives in under 8 pixels of disparity.

### What one pixel of disparity error costs

| Depth | Depth error | As % of depth | If disparity under-estimated | If over-estimated |
|---:|---:|---:|---:|---:|
| 5 m | 0.065 m | 1.3 % | 5.1 m | 4.9 m |
| 10 m | 0.260 m | 2.6 % | 10.3 m | 9.7 m |
| 20 m | 1.041 m | 5.2 % | 21.1 m | 19.0 m |
| 50 m | 6.504 m | 13.0 % | 57.5 m | 44.2 m |
| 80 m | 16.650 m | 20.8 % | 101.0 m | 66.2 m |

The 80 m figure is **256×** the 5 m figure for the same one-pixel error. That
factor is `(80/5)²`, exactly as the derivation predicts.

### What sub-pixel precision buys

At a quarter-pixel error the 80 m figure drops from 16.65 m to 4.16 m — a
quarter of the error for a quarter of the disparity error, linearly. At the
3-pixel D1 threshold it rises to 49.95 m, which is 62 % of the true depth. **A
prediction that passes KITTI's D1 correctness test at 80 m can still be wrong
about the distance by more than half.**

### Where depth stops being measurable

Beyond `fB / 1 px = 384 m` the entire remaining scene occupies less than one
pixel of disparity. Long before that, at 8.223 px of error — the float EPE Hailo
publishes for this model [SR-002] — a disparity underestimate drives disparity
to zero at **46.7 m**, past which depth is undefined rather than merely
inaccurate.

That last figure is an illustration of what an average pixel error is worth in
metres, not a prediction of the model's depth error: EPE is a mean over the
image and is not distributed uniformly. Establishing the actual distribution is
W7's job, and it is the reason this project reports depth error binned by range
rather than a single disparity number.

---

## 6. What this implies for the rest of Phase 1

- **A disparity metric with no depth range attached is close to meaningless
  here.** The same EPE means centimetres near the camera and tens of metres far
  from it. Every accuracy result in this project is reported binned by range.
- **Sub-pixel behaviour is the property to measure at range**, not average EPE.
- **Invalid and occluded pixels must be handled explicitly at every stage.** The
  geometry module refuses to convert them, and the evaluation must not quietly
  include them.
- **The disparity range chosen by the architecture sets a hard minimum range**
  and is a first-class architectural constraint, not a hyperparameter.

---

*Evidence: derivations are self-contained and checked against a numerical
derivative in `tests/test_geometry.py`. All numeric values are
[MEASUREMENT: EXP-003] on [SR-020]. The 8.223 EPE figure is [SOURCE: SR-002].*
