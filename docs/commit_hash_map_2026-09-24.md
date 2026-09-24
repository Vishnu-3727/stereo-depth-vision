# Commit hash map — pre-push trailer removal (2026-09-24)

The 25 commits in `origin/main..master` had never been pushed. Before the
first push, the `Co-Authored-By: Claude` trailers were stripped from the
commit messages with:

```
git -c core.quotepath=off filter-branch -f \
  --msg-filter "sed -e '/^Co-Authored-By: Claude/d'" \
  -- origin/main..master
```

Only message bytes changed: every old/new pair has an identical tree
(`git rev-parse <old>^{tree} == git rev-parse <new>^{tree}` for all 25),
and `git diff backup/pre-trailer-rewrite master` is empty. The first 6
commits carried no trailer, so their hashes are unchanged.

Old hashes remain reachable locally via branch `backup/pre-trailer-rewrite`
(this branch is local-only and is not pushed). Any citation of a pre-rewrite
hash below resolves by looking up this table, then reading the old commit
off that branch.

## Old -> new (full hashes, oldest first)

| old | new | subject |
|-----|-----|---------|
| 5b188243c738fbee01d3c339230731ff7f7c8dfe | 5b188243c738fbee01d3c339230731ff7f7c8dfe | exp(runtime): R4 - E3 seed-0 final through the recommended path, measurement only |
| ea7274c6f21e4ef31be52963181f068d22b179ff | ea7274c6f21e4ef31be52963181f068d22b179ff | docs(stage-e): E4 data availability - the public FT3D subset on Kaggle is usable |
| 3e29146c6f521afb64ae7e783f2a8e3d6beea6f2 | 3e29146c6f521afb64ae7e783f2a8e3d6beea6f2 | docs(stage-e): E4 pretrain design and pre-registration; make the Kaggle owner configurable |
| 442d8dc66953802a95305e6b458326b7acacd4b6 | 442d8dc66953802a95305e6b458326b7acacd4b6 | feat(stage-e): E4 data plumbing - FlyingThings3D_subset manifest builder and sign-fixed loader |
| 4d9109fa092c93096e106d617ca8664d1c230bef | 4d9109fa092c93096e106d617ca8664d1c230bef | feat(stage-e): E4 pretrain trainer - Stage 1's recipe, new corpus, probe mode |
| d88a2247008f7cd20826d28499839cbbacf97b46 | d88a2247008f7cd20826d28499839cbbacf97b46 | feat(stage-e): E4 Kaggle bundle, rate-probe kernel and push helper |
| f6f3565656c9a4c41de0bf53886b9f6b10e6c443 | b3eec7c686f00b6a1650f2691568a694d622e9da | fix(stage-e): E4 probe resolves the nested Kaggle dataset mounts |
| cb4c7f7ef4512218642e8e57bda1d1473228f76d | 337dd93da657b1f973ee6080863d8e8d4519fca0 | docs(stage-e): fix a grammar slip in the E4 probe addendum |
| 1af076d34f7fe439303e6ce53fbdbb3b5fac74d8 | 4729c30111cc711d675777ed5cc87f09c7640213 | feat(stage-e): E4 trainer reports the first batch's valid-pixel count |
| a94d7c80aa00eb44b4814b598012d1abe2b3e528 | 6398ba9bd71fec7b734905e0cc226a936bf8dce7 | feat(stage-e): E4 pretrain survives a killed session |
| 9bf3de08efe491e01f4603596a4dd812a858bb83 | 6006183ff7d372dd39ef5d1ef6e05924105561df | docs(stage-e): amend E4 section 6 with the measured rate and commit 8 epochs |
| 428722057e3437de3e4e71f71951a738d81bce89 | abb04f61f537fdc0feac05407fa336ccec0d479a | feat(stage-e): E4 pretrain kernel, wall-clock budget, gate 4 closed |
| b17be8f6dc4a85b2c1b9df2c8ce85cb22fc2b28e | c3bcb1bc3e863fb80f08b9dadaf65982fdf2a940 | docs(stage-e): E4 gate 5 pass, gate 6 closed by private KITTI re-upload |
| f503e40e83c68ec3c71e1ffb4797a60dd208a53b | 7109b90124bc505cf6b1f0b97f0d74a915166a4e | feat(stage-e): E4 finetune tooling, init packaging and verdict support |
| d2500322d364827bec67cd1bdd3d82a471312097 | c53ebe718077489219474f93da321f3d4e53191b | docs(stage-e): pre-register INT8 per-layer sensitivity |
| 544f7b0c02270e49fa4c9dc389ebab648f253252 | 503d35172d8547a586933317dde61839d2f1bf6d | feat(demo): show the E3 checkpoint in the mentor demo |
| e437f51a6aff79ed5ae5b1612d839deb6a6af6e8 | 14b7f91d174a5a1dc1582223870cf996a3460bb6 | docs(stage-e): amend INT8 sensitivity spec - QDQ coverage and op-type arm |
| 28523d7027121ba2d47c7cdfe507302d8ba097f5 | 6c930a5929955169215e4f33ca0105d35d21b66f | feat(stage-e): INT8 per-layer and op-type sensitivity sweep and report |
| fbef867d84cb1a67768c7f0fc226b41da703a2af | e6cf3f0e0b6c57baec664026aa8b587fce02c900 | docs(stage-e): INT8 report - no-op configs and regression-head location |
| 33a44ebd9927d3e73bb3ae06608300520fd76921 | e6dce956e38f3dc56a4014ecaff78e7803205abc | fix(stage-e): retarget E4 finetune init-sha guard and recover seeds 0-1 |
| 8bdb7f499c6f80a7c6ec3da1a0fac7cac3623fba | f2c58122b8325bc4039b09023adf3bc3384662ca | feat(stage-e): INT8 gate for E4 seeds 0-1 |
| 687ae01a966cccd5ce7ef0bf398057944e80a5e4 | 00a5e50eea36536bdbb922d7d77442b27d09f03b | docs(stage-e): E4 interim entries, seed 2 pending |
| e726220374daa0aab47037465e2beb7959f6a56e | 28cb107222645c53a8129d967e33c005d135463f | feat(stage-e): one-shot E4 finish script, seed-parametric recovery |
| 84644bb379abbc1516a920b46d60e0bddd086d7d | eef919bd7bb20ff2071ab586ebaade7e91788905 | feat(stage-e): E4 verdict with seed 2 |
| f8a298a3f1d008ba443db68ac6bf2f0e7580f8ce | 6a7eaf2f28587750e248cf70b22517e6cc5bd809 | chore(stage-e): E4 Kaggle evidence records (probe, pretrain, stopped finetune seeds) |

Short citations remapped alongside: `d250032` -> `c53ebe7`,
`e437f51` -> `14b7f91`, `1af076d` -> `4729c30`, `a94d7c8` -> `6398ba9`.

## Records left with pre-rewrite hashes

Run-output records stay exactly as produced; they are NOT edited. Resolve
their embedded pre-rewrite hashes through the table above:

- `stage_e_recipe/int8_sensitivity/armc.json` — `spec` cites amendment
  `e437f51`; `git_head` is `e437f51a6aff79ed5ae5b1612d839deb6a6af6e8`.
- `stage_e_recipe/int8_sensitivity/int8_sensitivity.json` — `git_head` is
  `d2500322d364827bec67cd1bdd3d82a471312097`.
- `stage_e_recipe/int8_sensitivity/qdq_coverage.json` — `git_head` is
  `544f7b0c02270e49fa4c9dc389ebab648f253252`.
