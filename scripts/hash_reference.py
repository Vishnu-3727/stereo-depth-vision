"""Hash every file in reference/ and regenerate reference/MANIFEST.md.

reference/ holds downloaded vendor and upstream artifacts. It is read-only once
acquired; the manifest is what makes a result traceable to the exact bytes it
was produced from. Re-running this script after a re-download will show whether
an artifact changed upstream.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = REPO_ROOT / "reference"

# Where each artifact came from. Anything not listed here is recorded as
# "derived" -- extracted from one of the archives above rather than downloaded.
ORIGINS = {
    "stereonet.zip": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DisparityEstimation/stereonet/pretrained/2023-05-31/stereonet.zip",
    "stereonet.hef": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/stereonet.hef",
    "stereonet_profiler_results_compiled_runtime_data.html": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/stereonet_profiler_results_compiled_runtime_data.html",
    "hailo_model_zoo/stereonet.yaml": "https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/hailo_model_zoo/cfg/networks/stereonet.yaml",
    "hailo_model_zoo/stereonet.alls": "https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/hailo_model_zoo/cfg/alls/generic/stereonet.alls",
    "hailo_model_zoo/HAILO8_stereo_depth_estimation.rst": "https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/docs/public_models/HAILO8/HAILO8_stereo_depth_estimation.rst",
    "upstream/nivosco_StereoNet.zip": "https://codeload.github.com/nivosco/StereoNet/zip/refs/heads/master",
    "upstream/hailo_model_zoo_master.zip": "https://codeload.github.com/hailo-ai/hailo_model_zoo/zip/refs/heads/master",
    "upstream/hailo_apps_main.zip": "https://codeload.github.com/hailo-ai/hailo-apps/zip/refs/heads/main",
}

# Extracted trees are large and fully reproducible from their archive, so the
# manifest lists the archive rather than every file inside it.
SKIP_DIRS = {"upstream/extracted"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(REFERENCE.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(REFERENCE).as_posix()
        if rel in {"MANIFEST.md", ".gitkeep"}:
            continue
        if any(rel.startswith(d + "/") for d in SKIP_DIRS):
            continue
        rows.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "origin": ORIGINS.get(rel, "derived (extracted from an archive above)"),
            }
        )

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Reference artifact manifest",
        "",
        "Public artifacts downloaded for Phase 1 forensics. **This directory is",
        "read-only after acquisition** -- vendor and upstream files are never edited.",
        "The archives are not committed to git (see `.gitignore`); this manifest is,",
        "so any result can be traced back to the exact bytes it came from.",
        "",
        "Regenerate with `python scripts/hash_reference.py`.",
        "",
        "Generated " + stamp + ".",
        "",
        "| Artifact | Bytes | SHA-256 | Origin |",
        "|---|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            "| `{path}` | {bytes:,} | `{sha}` | {origin} |".format(
                path=r["path"], bytes=r["bytes"], sha=r["sha256"][:32] + "...",
                origin=r["origin"],
            )
        )
    lines += [
        "",
        "Full hashes are in `reference/manifest.json`.",
        "",
        "`upstream/extracted/` holds the unpacked archives and is intentionally not",
        "hashed file-by-file -- it is reproducible from the archives listed above.",
        "",
    ]
    (REFERENCE / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
    (REFERENCE / "manifest.json").write_text(
        json.dumps({"generated_utc": stamp, "artifacts": rows}, indent=2),
        encoding="utf-8",
    )
    for r in rows:
        print("{:>12,}  {}  {}".format(r["bytes"], r["sha256"][:16], r["path"]))
    print("\n" + str(len(rows)) + " artifacts hashed -> reference/MANIFEST.md")


if __name__ == "__main__":
    sys.exit(main())
