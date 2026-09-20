Read-only measurement probes used for the Stage-B ARM P pre-flight audit (20260918T053135Z).
All probes are READ-ONLY: pathlib counts, PFM histogram sampling, model construction
for param count. No optimizer, no backward pass, no checkpoint save, no training.

Probes (run from repo root C:\Users\vishn\stereo_depth_vision):
1. stage_b_probe1.py - per-subset png/pfm counts (FT3D TRAIN frames vs disparity)
2. stage_b_probe2.py - triplet matching (L+R+disp keys), orphans, missing sequences
3. stage_b_probe3.py - TEST presence, monkaa/ entries, driving/extracted counts, D: listing
4. stage_b_probe4.py - D: archive sizes, C: script/log inventory
5. stage_b_probe5.py - 400-map FT3D disparity histogram via src/datasets/pfm.py (seed 0)
6. stage_b_probe6.py - P2A construction: param count / stride / keys / BN check
7. stage_b_ratios.py - three-way per-bin ratios from raw histogram + high_disparity.json
8. stage_b_ts.py   - UTC timestamp helper

The originals live in C:\Users\vishn\AppData\Local\Temp\opencode\ (scratch, outside
the repo); this file records what they did. Re-running any probe reproduces raw/.
