"""Score a Phase 1 regime checkpoint through the frozen harness. Single-line run."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from phase1.harness.frozen_eval import score_checkpoint  # noqa: E402

ckpt_rel = sys.argv[1]
out_path = sys.argv[2]
rec = score_checkpoint(ckpt_rel)
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
Path(out_path).write_text(json.dumps(rec, indent=2), encoding="utf-8")
m = rec["metrics"]
print("EPE {:.7f} D1 {:.7f} RMSE {:.7f} px={} guard={}".format(
    m["epe"], m["d1"], m["rmse"], m["valid_pixels"], rec["guard"]["contract_match"]))
