"""STAGE A -- DIAGNOSTIC ONLY. Compares two `diag_determinism.py` records.

Answers, from the records alone: are the two runs identical at batch 0, and if
they are, at which batch does the first numerical difference appear?

    python phase2/diagnostics/determinism/compare_runs.py RUN_DIR_1 RUN_DIR_2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(d):
    return json.loads((Path(d) / "record.json").read_text(encoding="utf-8"))


def main() -> None:
    a, b = load(sys.argv[1]), load(sys.argv[2])
    la, lb = Path(sys.argv[1]).name, Path(sys.argv[2]).name
    out = {"run_1": la, "run_2": lb}

    def same(path, va, vb):
        out[path] = {"run_1": va, "run_2": vb, "identical": va == vb}
        return va == vb

    print("=" * 74)
    print("COMPARE  {}  vs  {}".format(la, lb))
    print("=" * 74)

    print("\n-- environment ------------------------------------------------")
    for k in ("git_head", "python", "torch", "cuda_version", "cudnn_version",
              "gpu_name", "nvidia_driver", "deterministic_requested"):
        ok = same("env." + k, a["environment"][k], b["environment"][k])
        print("  {:26s} {:5s}  {}".format(k, "SAME" if ok else "DIFF",
                                          a["environment"][k]))
    for k in sorted(a["environment"]["flags"]):
        ok = same("flag." + k, a["environment"]["flags"][k],
                  b["environment"]["flags"][k])
        print("  flag {:21s} {:5s}  {}".format(k, "SAME" if ok else "DIFF",
                                               a["environment"]["flags"][k]))

    print("\n-- initialisation ---------------------------------------------")
    for k in ("weights_sha", "parameters_total", "refinement_blocks",
              "cost_volume_shift", "regression"):
        ok = same("model." + k, a["model"][k], b["model"][k])
        print("  {:26s} {:5s}  {}".format(k, "SAME" if ok else "DIFF", a["model"][k]))
    for stage in ("rng_before_build", "rng_after_build", "rng_after_loader_built",
                  "rng_at_batch0"):
        for k in sorted(a.get(stage, {})):
            ok = same(stage + "." + k, a[stage][k], b[stage][k])
            print("  {:26s} {:5s}  {}".format(stage + "/" + k,
                                              "SAME" if ok else "DIFF", a[stage][k]))

    print("\n-- batch 0 data -----------------------------------------------")
    for k in ("left_sha", "right_sha", "disparity_sha", "left_shape",
              "disparity_valid_px", "disparity_max"):
        ok = same("batch0." + k, a["batch0_inputs"][k], b["batch0_inputs"][k])
        print("  {:26s} {:5s}  {}".format(k, "SAME" if ok else "DIFF",
                                          a["batch0_inputs"][k]))
    for i, (sa, sb) in enumerate(zip(a["batch0_inputs"]["samples"],
                                     b["batch0_inputs"]["samples"])):
        for k in ("dataset_index", "scene_name", "crop_y", "crop_x",
                  "jitter_left", "jitter_right"):
            ok = same("batch0.sample{}.{}".format(i, k), sa[k], sb[k])
            print("  sample{} {:19s} {:5s}  {}".format(i, k, "SAME" if ok else "DIFF",
                                                       sa[k]))

    print("\n-- full epoch data order (80 batches) --------------------------")
    oa = [(e["dataset_index"], e["crop_y"], e["crop_x"], e["left_sha"])
          for e in a["data_order"]]
    ob = [(e["dataset_index"], e["crop_y"], e["crop_x"], e["left_sha"])
          for e in b["data_order"]]
    out["data_order_identical"] = oa == ob
    out["data_order_items"] = len(oa)
    first_data_diff = next((i for i, (x, y) in enumerate(zip(oa, ob)) if x != y), None)
    out["first_data_order_difference_at_item"] = first_data_diff
    print("  items                      {} vs {}".format(len(oa), len(ob)))
    print("  identical                  {}".format(oa == ob))
    print("  first difference at item   {}".format(first_data_diff))

    print("\n-- batch 0 computation ----------------------------------------")
    same("batch0.loss", a["batch0_forward"]["loss"], b["batch0_forward"]["loss"])
    print("  loss                       {!r}".format(a["batch0_forward"]["loss"]))
    print("                             {!r}".format(b["batch0_forward"]["loss"]))
    print("  loss identical             {}".format(
        a["batch0_forward"]["loss"] == b["batch0_forward"]["loss"]))
    same("batch0.out_sha", a["batch0_forward"]["out_sha"], b["batch0_forward"]["out_sha"])
    print("  output sha                 {} / {}   {}".format(
        a["batch0_forward"]["out_sha"], b["batch0_forward"]["out_sha"],
        "SAME" if a["batch0_forward"]["out_sha"] == b["batch0_forward"]["out_sha"]
        else "DIFF"))

    print("\n-- in-process repeat probes (per run) --------------------------")
    for lbl, r in ((la, a), (lb, b)):
        f = r["batch0_same_process_forward_repeat"]
        g = r["batch0_same_process_backward_repeat"]
        print("  {}".format(lbl))
        print("    forward  x2 bitwise identical  {}   max|diff| {:.3e}".format(
            f["bitwise_identical"], f["max_abs_diff"]))
        print("    backward x2 bitwise identical  {}   max|diff| {:.3e}  "
              "elements differing {}".format(g["bitwise_identical"],
                                             g["max_abs_diff"],
                                             g.get("n_elements_differing")))
    out["forward_repeat_identical"] = {
        la: a["batch0_same_process_forward_repeat"]["bitwise_identical"],
        lb: b["batch0_same_process_forward_repeat"]["bitwise_identical"]}
    out["backward_repeat_identical"] = {
        la: a["batch0_same_process_backward_repeat"]["bitwise_identical"],
        lb: b["batch0_same_process_backward_repeat"]["bitwise_identical"]}

    print("\n-- first divergence in the training series ---------------------")
    pa, pb = a["per_batch"], b["per_batch"]
    first_loss = next((i for i, (x, y) in enumerate(zip(pa, pb))
                       if x["loss"] != y["loss"]), None)
    first_grad = next((i for i, (x, y) in enumerate(zip(pa, pb))
                       if x["grad_norm"] != y["grad_norm"]), None)
    out["first_batch_with_differing_loss"] = first_loss
    out["first_batch_with_differing_grad_norm"] = first_grad
    print("  first batch, loss differs        {}".format(first_loss))
    print("  first batch, grad_norm differs   {}".format(first_grad))
    for i in range(min(6, len(pa))):
        print("    batch {:>2}  loss {:.10f} / {:.10f}   grad {:.6f} / {:.6f}".format(
            i, pa[i]["loss"], pb[i]["loss"], pa[i]["grad_norm"], pb[i]["grad_norm"]))

    print("\n-- epoch outcome ----------------------------------------------")
    for k in ("first_batch_loss", "last_batch_loss", "mean_loss",
              "median_grad_norm", "max_grad_norm", "weights_after_epoch_sha",
              "any_nan_or_inf"):
        va, vb = a["epoch"][k], b["epoch"][k]
        same("epoch." + k, va, vb)
        extra = ""
        if isinstance(va, float) and isinstance(vb, float):
            extra = "   delta {:+.6e}".format(vb - va)
        print("  {:24s} {!r:>22} / {!r:<22}{}".format(k, va, vb, extra))

    print("\n-- validation after one epoch ----------------------------------")
    for k in ("epe", "d1"):
        va, vb = a["validation"]["run1"][k], b["validation"]["run1"][k]
        out["validation." + k] = {"run_1": va, "run_2": vb, "delta": vb - va}
        print("  {:4s}  {:.10f} / {:.10f}   delta {:+.6f}".format(k, va, vb, vb - va))
    for lbl, r in ((la, a), (lb, b)):
        print("  {} in-process validate x2 identical: {}  (dEPE {:+.3e}, dD1 {:+.3e})"
              .format(lbl, r["validation"]["repeat_identical"],
                      r["validation"]["repeat_delta_epe"],
                      r["validation"]["repeat_delta_d1"]))

    dest = Path(sys.argv[1]).parent / "comparison_{}_vs_{}.json".format(la, lb)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote {}".format(dest))


if __name__ == "__main__":
    main()
