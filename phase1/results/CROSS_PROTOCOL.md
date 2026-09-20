# Cross-protocol measurement (Task A)

OUR models scored under TWO crop anchors on the same 40 scenes
(hailo_val = last 40 of the sorted image_2 listing, GT disp_occ_0 / 256.0).
The teammate checkpoints are NOT published (no releases, no .pt in his
repo), so the reverse comparison — his model under our contract — cannot
be run. This table therefore compares OUR models across two protocols,
NOT the two models head to head. Nothing here changes the frozen contract
(`phase1/harness/frozen_eval.py` untouched; top-left 368x1232, gt>0,
pooled, 3,802,797 px).

| Model | Protocol | EPE | D1 | RMSE | Valid px |
|---|---|---|---|---|---|
| arm_u | frozen-top-left official (gt>0) | 2.2866369 | 15.3067597% | 6.1837683 | 3802797 |
| arm_k | frozen-top-left official (gt>0) | 5.5271927 | 45.6085613% | 10.7479083 | 3802797 |
| onnx | frozen-top-left official (gt>0) | 1.3134471 | 8.1543664% | 2.5829573 | 3802797 |
| arm_u | teammate bottom-right official (disp>0) | 2.3205291 | 15.4687603% | 6.1554555 | 3861014 |
| arm_k | teammate bottom-right official (disp>0) | 5.5660400 | 47.2175444% | 10.6536597 | 3861014 |
| onnx | teammate bottom-right official (disp>0) | 1.2995328 | 8.1186445% | 2.5947845 | 3861014 |
| arm_u | teammate bottom-right masked (disp>0, disp<192, col>=192) | 2.0396484 | 13.9669399% | 5.2399002 | 3353290 |
| arm_k | teammate bottom-right masked (disp>0, disp<192, col>=192) | 4.7965367 | 44.4143215% | 8.9804029 | 3353290 |
| onnx | teammate bottom-right masked (disp>0, disp<192, col>=192) | 1.2124993 | 7.2664756% | 2.4235630 | 3353290 |

Valid pixel counts by crop (official mask): top-left 3802797 px, bottom-right 3861014 px.
Masked protocol (bottom-right, disp<192 and col>=192): 3353290 px.

## Crop-attribution finding (measured, not guessed)

ARM U scores 2.2866369 EPE under our frozen top-left contract and 2.3205291 EPE under the teammate bottom-right crop (official mask), a measured crop-anchor shift of +0.0338922 px.
The ARM U (2.2866) vs teammate-reported (1.659) gap is 0.6276 px; the crop anchor accounts for +0.0339 px of it, i.e. 5.4% of the gap.
The remainder is architecture, training recipe, and his masked protocol — NOT resolved by this measurement, because his weights are unpublished and his model cannot be scored here. His 1.659 number is not claimed to be wrong; it is simply not comparable to any number in the table above without his checkpoints.

