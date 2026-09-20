# SceneFlow acquisition measurement + ARM P status

All numbers below MEASURED 2026-09-14 by range requests and a timed partial
download against
https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/ .
Nothing below is estimated beyond the stated rate extrapolations. Do not reuse
these numbers as anything other than what they are: a throughput probe plus
upstream file sizes.

## Measured throughput

- Sustained throughput (live download, 2026-09-14): 5.9 MB/s
  (2.83 GB in 8 minutes = 2.83e9 B / 480 s = 5,895,833 B/s).
  This supersedes the 45-second probe below; all estimates in this file
  are derived from the 5.9 MB/s figure unless marked otherwise.
- SUPERSEDED measurement (kept for provenance, do NOT use for planning):
  383,821 B/s (17,272,000 bytes in 45 s, FlyingThings3D cleanpass range
  probe). The live download runs ~15.4x faster than this probe suggested;
  the probe likely measured throttled/range-request throughput, not
  sustained bulk-transfer throughput.

## Upstream file sizes (bytes, as listed)

- driving__frames_cleanpass.tar: 6,681,896,960 B
- driving__disparity.tar.bz2: 9,561,161,211 B
- monkaa__frames_cleanpass.tar: 9,672,867,840 B
- monkaa__disparity.tar.bz2: 29,940,902,883 B
- flyingthings3d__frames_cleanpass.tar: 39,473,469,440 B
- flyingthings3d__disparity.tar.bz2: 93,213,362,434 B

## Corpus totals (from the sizes above)

- Full upstream corpus (listflowfile.py = FlyingThings3D + Monkaa + Driving) = 188.5 GB
  => approx 9-10 h at the live 5.9 MB/s rate
  (188,543,660,768 B / 5.9e6 B/s = 31,979 s = 8.9 h).
- Driving alone = 16.2 GB, approx 45 min
  (16,243,058,171 B / 5.9e6 B/s = 2,755 s = 45.9 min).
- FlyingThings3D alone = 132.7 GB, approx 6-7 h
  (132,686,831,874 B / 5.9e6 B/s = 22,505 s = 6.3 h).
- (Superseded, from the 383,821 B/s probe: full corpus ~140 h, Driving ~12 h.)

## Availability notes (measured 2026-09-14)

- FlyingThings3D_subset images/disparity are TORRENT-ONLY on the Freiburg page;
  no direct HTTP link exists for them.
- Local disk free at probe time: 298.1 GB on C:.
- GPU: RTX 4060 Laptop, 8188 MiB.

## ARM P status

- ARM P (full SceneFlow pretraining) status: BLOCKED -- for a different
  reason than before. The full corpus is now acquired and byte-size-verified
  on D:\sceneflow_archives (see "Acquisition complete" below), so the
  download blocker is gone; the new blocker is staging: ARM P needs the
  FlyingThings3D TRAIN/left subset extracted onto C: first, because the
  external disk is NOT a training data path (see "External-disk
  performance"). NOT run, NOT failed.
  (Previous note "BLOCKED -- until the corpus is on disk and verified" is
  superseded: the corpus IS on disk and verified, on D:.)
  (Previous note "acquisition infeasible within the sprint" is superseded by
  the 5.9 MB/s measurement: the full 188.5 GB corpus now projects to ~9-10 h,
  which fits the sprint.)

## ARM D status (proxy, not equivalent)

- ARM D (Driving-only synthetic pretraining -> KITTI fine-tuning) is a
  PROXY/regime experiment. It must never be reported as equivalent to full
  SceneFlow pretraining, and any ARM D result is evidence only about
  driving-domain synthetic pretraining.

## Download in progress

- SUPERSEDED (acquisition complete, 2026-09-14): the download described below
  finished. The full corpus now lives on D:\sceneflow_archives -- see
  "Acquisition complete" below. Previous in-progress note kept for
  provenance, do NOT use for planning:
- (Old note:) The download in progress: data/sceneflow/driving/ via fetch_driving.sh (resumable).
- (Old note:) Queued after Driving: FlyingThings3D then Monkaa via data/sceneflow/fetch_rest.sh.

## Acquisition complete (2026-09-14)

- The full upstream corpus is now acquired and verified by byte size against
  the sizes listed under "Upstream file sizes" above:
  - flyingthings3d__frames_cleanpass.tar: 39,473,469,440 B
  - flyingthings3d__disparity.tar.bz2: 93,213,362,434 B
  - monkaa__frames_cleanpass.tar: 9,672,867,840 B
  - monkaa__disparity.tar.bz2: 29,940,902,883 B
  - driving__frames_cleanpass.tar: 6,681,896,960 B
  - driving__disparity.tar.bz2: 9,561,161,211 B
- Location: D:\sceneflow_archives (external USB disk). data/sceneflow/ on C:
  holds only the Driving archives + extracted/.

## External-disk performance (measured 2026-09-14)

- Sequential write: 30 MB/s; sequential read: 37 MB/s; random small-file:
  28 MB/s (14 files/s, 71 ms/file).
- The disk disconnected once mid-copy under load.
- Verdict: usable for cold archive storage but NOT as a training data path --
  training must read from C:.
