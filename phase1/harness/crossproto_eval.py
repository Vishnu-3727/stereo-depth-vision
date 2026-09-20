"""Cross-protocol measurement: OUR models under TWO crop anchors.

Measurement only. No training, no source changes. Does NOT modify
``phase1/harness/frozen_eval.py`` — it imports dataset helpers, metrics, and
the reference-checkpoint contract from it.

Background: a teammate repo (hailostereovision) reports 1.659 EPE / 9.92% D1
on KITTI with a different architecture. His evaluation protocol, read from his
``src/data.py`` and ``src/eval_kitti.py``, differs from our frozen contract in
the CROP ANCHOR:

- Same 40 scenes: the LAST 40 of the sorted image_2 listing (= our hailo_val).
- Same GT: disp_occ_0, 16-bit PNG divided by 256.0.
- DIFFERENT CROP: multiple-of-16 anchored BOTTOM-RIGHT::

      nh, nw = (h // 16) * 16, (w // 16) * 16
      img = img[h - nh:, w - nw:]

  On 375x1242 KITTI frames that is rows 7..375, cols 10..1242.
  OUR frozen contract uses the TOP-LEFT 368x1232 crop (rows 0..368,
  cols 0..1232). Same size, shifted 7 rows down and 10 columns right.
  KITTI LiDAR GT is denser toward the bottom of the frame, so these are
  different pixel populations.
- TWO protocols, both pooled over pixels (epe*n accumulated, divided by
  total n — pooled, not per-image)::

      official mask = disp > 0
      masked   mask = (disp > 0) AND (disp < 192) AND (column >= 192)

  where ``column`` is the column index in the CROPPED frame (the mask is
  applied after the crop in his code). D1 = (err > 3.0) AND
  (err > 0.05 * abs(disp)), as a percentage — same as ours.

This script scores, under BOTH his protocols AND our frozen top-left
contract, each of:

- ``phase1/runs/arm_u/arm_u_best.pth`` (shift=right, normalize=True)
- ``phase1/runs/arm_k/arm_k_best.pth`` (shift=none,  normalize=False)
- the reference ONNX (``reference/onnx/stereonet.onnx``)

The ONNX session setup (SessionOptions, ORT_ENABLE_ALL, CPUExecutionProvider)
is line-for-line the setup in ``frozen_eval.score_onnx``; the frozen-branch
ONNX row is asserted to reproduce the contract figure 1.3134471, which proves
the session path matches. Per model per protocol: EPE, D1, RMSE, valid pixel
count. Writes ``phase1/results/CROSS_PROTOCOL.md``.

His checkpoints are NOT published (no releases, no .pt in his repo), so the
reverse comparison (his model under our contract) cannot be run. This table
compares OUR models across two protocols, NOT the two models head to head.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase1.harness import frozen_eval as fe  # noqa: E402
from src.datasets.kitti2015 import normalize, read_disparity_png, read_image  # noqa: E402
from src.evaluation.metrics import disparity_metrics  # noqa: E402

DATA_ROOT = REPO_ROOT / "data" / "kitti2015" / "training"

ARM_U_CKPT = "phase1/runs/arm_u/arm_u_best.pth"
ARM_K_CKPT = "phase1/runs/arm_k/arm_k_best.pth"

FROZEN_EPE_ARM_U = 2.2866369
FROZEN_PX = 3802797


def hailo_val_names() -> list[str]:
    left_dir = DATA_ROOT / "image_2"
    names = sorted(p.name for p in left_dir.iterdir() if p.name.endswith("_10.png"))
    return names[160:]  # last 40, same as Kitti2015Stereo(split="hailo_val")


def crop_topleft(img: np.ndarray) -> np.ndarray:
    """Frozen contract: 368x1232 anchored top-left (Hailo pad_and_crop)."""
    return img[:368, :1232]


def crop_bottomright(img: np.ndarray) -> np.ndarray:
    """Teammate protocol: multiple of 16 anchored bottom-right."""
    h, w = img.shape[:2]
    nh, nw = (h // 16) * 16, (w // 16) * 16
    return img[h - nh:, w - nw:]


def official_mask(disp: np.ndarray) -> np.ndarray:
    return disp > 0


def masked_mask(disp: np.ndarray) -> np.ndarray:
    h, w = disp.shape
    cols = np.broadcast_to(np.arange(w)[None, :], (h, w))
    return (disp > 0) & (disp < 192) & (cols >= 192)


def pooled(pred_list: list[np.ndarray], gt_list: list[np.ndarray]) -> dict:
    return disparity_metrics(
        np.concatenate(pred_list), np.concatenate(gt_list)
    ).as_dict()


def score_torch(ckpt_rel: str, shift: str, reg_norm: bool,
                crop_fn, mask_fn) -> dict:
    import torch

    from src.models.stereonet import StereoNet, StereoNetConfig

    ckpt_path = REPO_ROOT / ckpt_rel
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    config = StereoNetConfig(cost_volume_shift=shift,
                             regression_normalize=reg_norm)
    model = StereoNet(config)
    model.load_state_dict(state, strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(dev)
    preds, gts = [], []
    with torch.no_grad():
        for name in hailo_val_names():
            left = crop_fn(read_image(DATA_ROOT / "image_2" / name))
            right = crop_fn(read_image(DATA_ROOT / "image_3" / name))
            disp = crop_fn(
                read_disparity_png(DATA_ROOT / "disp_occ_0" / name, scale=256.0)
            ).astype(np.float64)
            out = model(torch.from_numpy(normalize(left)).to(dev),
                        torch.from_numpy(normalize(right)).to(dev))
            pred = out[0, 0].cpu().numpy().astype(np.float64)
            m = mask_fn(disp)
            preds.append(pred[m])
            gts.append(disp[m])
    return pooled(preds, gts)


def open_reference_session():
    """ONNX session setup reused from frozen_eval.score_onnx (same options,
    same provider list, same checkpoint path from fe.CONTRACT)."""
    import onnxruntime as ort

    model_path = REPO_ROOT / fe.CONTRACT["reference"]["checkpoint"]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    return session


def score_onnx(crop_fn, mask_fn) -> dict:
    session = open_reference_session()
    in_names = [i.name for i in session.get_inputs()]
    out_name = session.get_outputs()[0].name
    preds, gts = [], []
    for name in hailo_val_names():
        left = crop_fn(read_image(DATA_ROOT / "image_2" / name))
        right = crop_fn(read_image(DATA_ROOT / "image_3" / name))
        disp = crop_fn(
            read_disparity_png(DATA_ROOT / "disp_occ_0" / name, scale=256.0)
        ).astype(np.float64)
        out = session.run([out_name], {in_names[0]: normalize(left),
                                       in_names[1]: normalize(right)})[0]
        pred = out[0, 0].astype(np.float64)
        m = mask_fn(disp)
        preds.append(pred[m])
        gts.append(disp[m])
    return pooled(preds, gts)


def main() -> None:
    names = hailo_val_names()
    assert len(names) == 40, len(names)
    assert names[0] == "000160_10.png" and names[-1] == "000199_10.png"

    # Sanity: both crops are 368x1232 on KITTI ~375x1242 frames (exact input
    # size varies by scene, e.g. 374x1238; the multiple-of-16 rule adapts).
    probe = read_image(DATA_ROOT / "image_2" / names[0])
    h0, w0 = probe.shape[:2]
    assert crop_topleft(probe).shape[:2] == (368, 1232), probe.shape
    assert crop_bottomright(probe).shape[:2] == (368, 1232), (
        probe.shape, crop_bottomright(probe).shape)
    tl = crop_topleft(probe)
    br = crop_bottomright(probe)
    assert np.array_equal(tl, probe[:368, :1232])
    nh0, nw0 = (h0 // 16) * 16, (w0 // 16) * 16
    assert np.array_equal(br, probe[h0 - nh0:, w0 - nw0:])
    if (h0, w0) == (375, 1242):  # the canonical frame: rows 7..375, cols 10..1242
        assert np.array_equal(br, probe[7:, 10:])

    protocols = [
        ("frozen-top-left official (gt>0)", crop_topleft, official_mask),
        ("teammate bottom-right official (disp>0)",
         crop_bottomright, official_mask),
        ("teammate bottom-right masked (disp>0, disp<192, col>=192)",
         crop_bottomright, masked_mask),
    ]
    rows: dict[str, dict[str, dict]] = {}
    for pname, crop_fn, mask_fn in protocols:
        print("scoring ARM U [{}] ...".format(pname), flush=True)
        u = score_torch(ARM_U_CKPT, "right", True, crop_fn, mask_fn)
        print("scoring ARM K [{}] ...".format(pname), flush=True)
        k = score_torch(ARM_K_CKPT, "none", False, crop_fn, mask_fn)
        print("scoring ONNX [{}] ...".format(pname), flush=True)
        o = score_onnx(crop_fn, mask_fn)
        rows[pname] = {"arm_u": u, "arm_k": k, "onnx": o}
        print("  ARM U EPE {:.7f} D1 {:.7f} RMSE {:.7f} px={}".format(
            u["epe"], u["d1"], u["rmse"], u["valid_pixels"]), flush=True)
        print("  ARM K EPE {:.7f} D1 {:.7f} RMSE {:.7f} px={}".format(
            k["epe"], k["d1"], k["rmse"], k["valid_pixels"]), flush=True)
        print("  ONNX  EPE {:.7f} D1 {:.7f} RMSE {:.7f} px={}".format(
            o["epe"], o["d1"], o["rmse"], o["valid_pixels"]), flush=True)

    frozen = rows["frozen-top-left official (gt>0)"]
    # Contract self-checks: frozen branch must reproduce frozen_eval exactly.
    assert frozen["arm_u"]["valid_pixels"] == FROZEN_PX, frozen["arm_u"]
    assert abs(frozen["arm_u"]["epe"] - FROZEN_EPE_ARM_U) < 1e-4, frozen["arm_u"]
    assert frozen["arm_k"]["valid_pixels"] == FROZEN_PX
    assert abs(frozen["arm_k"]["epe"] - 5.5271927) < 1e-4, frozen["arm_k"]
    assert frozen["onnx"]["valid_pixels"] == FROZEN_PX
    assert abs(frozen["onnx"]["epe"] - fe.CONTRACT["reference"]["epe"]) < 1e-4, \
        frozen["onnx"]

    br_off = rows["teammate bottom-right official (disp>0)"]
    br_msk = rows["teammate bottom-right masked (disp>0, disp<192, col>=192)"]

    crop_effect_u = br_off["arm_u"]["epe"] - frozen["arm_u"]["epe"]

    md = []
    md.append("# Cross-protocol measurement (Task A)")
    md.append("")
    md.append("OUR models scored under TWO crop anchors on the same 40 scenes")
    md.append("(hailo_val = last 40 of the sorted image_2 listing, GT disp_occ_0 / 256.0).")
    md.append("The teammate checkpoints are NOT published (no releases, no .pt in his")
    md.append("repo), so the reverse comparison — his model under our contract — cannot")
    md.append("be run. This table therefore compares OUR models across two protocols,")
    md.append("NOT the two models head to head. Nothing here changes the frozen contract")
    md.append("(`phase1/harness/frozen_eval.py` untouched; top-left 368x1232, gt>0,")
    md.append("pooled, 3,802,797 px).")
    md.append("")
    md.append("| Model | Protocol | EPE | D1 | RMSE | Valid px |")
    md.append("|---|---|---|---|---|---|")
    for pname, crop_fn, mask_fn in protocols:
        for model in ("arm_u", "arm_k", "onnx"):
            m = rows[pname][model]
            md.append("| {} | {} | {:.7f} | {:.7f}% | {:.7f} | {} |".format(
                model, pname, m["epe"], m["d1"], m["rmse"], m["valid_pixels"]))
    md.append("")
    md.append("Valid pixel counts by crop (official mask): top-left {} px, "
              "bottom-right {} px.".format(
                  frozen["arm_u"]["valid_pixels"],
                  br_off["arm_u"]["valid_pixels"]))
    md.append("Masked protocol (bottom-right, disp<192 and col>=192): {} px.".format(
        br_msk["arm_u"]["valid_pixels"]))
    md.append("")
    md.append("## Crop-attribution finding (measured, not guessed)")
    md.append("")
    md.append("ARM U scores {:.7f} EPE under our frozen top-left contract and "
              "{:.7f} EPE under the teammate bottom-right crop (official mask), "
              "a measured crop-anchor shift of {:+.7f} px.".format(
                  frozen["arm_u"]["epe"], br_off["arm_u"]["epe"], crop_effect_u))
    md.append("The ARM U ({:.4f}) vs teammate-reported (1.659) gap is {:.4f} px; "
              "the crop anchor accounts for {:+.4f} px of it, i.e. "
              "{:.1f}% of the gap.".format(
                  frozen["arm_u"]["epe"],
                  frozen["arm_u"]["epe"] - 1.659, crop_effect_u,
                  100.0 * crop_effect_u / (frozen["arm_u"]["epe"] - 1.659)))
    md.append("The remainder is architecture, training recipe, and his masked "
              "protocol — NOT resolved by this measurement, because his weights "
              "are unpublished and his model cannot be scored here. His 1.659 "
              "number is not claimed to be wrong; it is simply not comparable to "
              "any number in the table above without his checkpoints.")
    md.append("")
    out_path = REPO_ROOT / "phase1" / "results" / "CROSS_PROTOCOL.md"
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote {}".format(out_path))


if __name__ == "__main__":
    main()
