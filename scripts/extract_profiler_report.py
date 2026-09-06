"""Extract the machine-readable payload from Hailo's compiled profiler report.

`stereonet_profiler_results_compiled_runtime_data.html` [SR-006] is a 40.9 MB
single-page application. Buried in it is a semicolon-separated payload holding a
model-level summary row and a per-layer table with Hailo's own on-device figures:
MACs, latency, FPS, utilisation, memory cuts, quantisation bit-widths and the
defusion assignments.

This is the only public source of per-layer behaviour on actual Hailo silicon
available without a device, so it is worth extracting carefully rather than
reading by eye.

Writes CSVs and a JSON summary into results/profiler/.

    python scripts/extract_profiler_report.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    REPO_ROOT / "reference" / "stereonet_profiler_results_compiled_runtime_data.html"
)
OUT_DIR = REPO_ROOT / "results" / "profiler"

# The payload is a run of semicolon-separated records. Two tables are present:
# a model-level one whose header ends at "l4_cut_size", and a per-layer one whose
# header starts at "layer_name,layer_type". A second, unrelated occurrence of
# "l4_cut_size" appears later in the bundle as a JavaScript identifier; the
# search is anchored to the layer header so it cannot be reached.
MODEL_HEADER_TAIL = "l4_cut_size;"
LAYER_HEADER_HEAD = "layer_name,layer_type,"


def find_payload(text: str) -> tuple[str, str]:
    """Return (model_table, layer_table) as raw semicolon-separated strings."""
    li = text.index(LAYER_HEADER_HEAD)
    mi = text.rindex(MODEL_HEADER_TAIL, 0, li) + len(MODEL_HEADER_TAIL)

    # Walk backwards from the model header tail to the start of its header. The
    # header is a comma-separated identifier list, so stop at the first
    # character that cannot appear in one.
    start = mi - len(MODEL_HEADER_TAIL)
    while start > 0 and re.match(r"[A-Za-z0-9_,]", text[start - 1]):
        start -= 1
    model_block = text[start:li].rstrip(";")

    # The layer table runs to the end of the payload: stop at the first
    # character that is clearly JavaScript again rather than data.
    end = li
    limit = min(len(text), li + 40_000_000)
    while end < limit and text[end] not in "\n\r":
        # a quote or backslash marks the end of the embedded string literal
        if text[end] in '"\\' or text[end] == "`":
            break
        end += 1
    layer_block = text[li:end]
    return model_block, layer_block


def parse_block(block: str) -> list[dict]:
    records = [r for r in block.split(";") if r.strip()]
    header = next(csv.reader([records[0]]))
    rows = []
    for rec in records[1:]:
        values = next(csv.reader([rec]))
        if len(values) != len(header):
            # keep it, flagged, rather than dropping data we do not understand
            rows.append({"_malformed": True, "_fields": len(values), "_raw": rec[:200]})
            continue
        rows.append(dict(zip(header, values)))
    return header, rows


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = REPORT.read_text(encoding="utf-8", errors="replace")
    model_block, layer_block = find_payload(text)

    model_header, model_rows = parse_block(model_block)
    layer_header, layer_rows = parse_block(layer_block)

    good = [r for r in layer_rows if not r.get("_malformed")]
    bad = [r for r in layer_rows if r.get("_malformed")]

    for name, header, rows in (
        ("model_summary.csv", model_header, model_rows),
        ("layers.csv", layer_header, good),
    ):
        with (OUT_DIR / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            for r in rows:
                if not r.get("_malformed"):
                    w.writerow(r)

    model = model_rows[0] if model_rows else {}
    summary = {
        "source": "reference/" + REPORT.name,
        "model_fields": len(model_header),
        "layer_fields": len(layer_header),
        "layer_rows": len(good),
        "malformed_rows": len(bad),
        "model": model,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("model-level fields: {}, layer rows: {}, malformed: {}".format(
        len(model_header), len(good), len(bad)))
    print("\nModel-level summary (non-empty fields):")
    for k, v in model.items():
        if v not in ("", None):
            print("  {:<38} {}".format(k, v))
    print("\nwritten to " + str(OUT_DIR))


if __name__ == "__main__":
    main()
