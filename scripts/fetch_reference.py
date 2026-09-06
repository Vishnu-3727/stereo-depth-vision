"""Restore the reference artifacts into a fresh clone, and verify their hashes.

`reference/` holds vendor and upstream artifacts that are downloaded rather than
committed: they are large, and they belong to their publishers. What *is*
committed is `reference/manifest.json`, which records every artifact's URL, size
and SHA-256. This script replays it.

Without these artifacts a clone still passes its tests, but eight checks in
`scripts/verify_claims.py` cannot run and say so. Running this restores the full
set.

    python scripts/fetch_reference.py            # download what is missing
    python scripts/fetch_reference.py --verify   # check what is present, no download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = REPO_ROOT / "reference"
MANIFEST = REFERENCE / "manifest.json"

# Archives that later steps unpack, and where they unpack to. Derived artifacts
# are not listed in the manifest because they are reproducible from these.
EXTRACTIONS = [
    ("stereonet.zip", REFERENCE / "onnx"),
    ("upstream/nivosco_StereoNet.zip", REFERENCE / "upstream" / "extracted"),
    ("upstream/hailo_model_zoo_master.zip", REFERENCE / "upstream" / "extracted"),
    ("upstream/hailo_apps_main.zip", REFERENCE / "upstream" / "extracted"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check present artifacts only; download nothing")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print("no manifest at " + str(MANIFEST))
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    artifacts = [a for a in manifest["artifacts"]
                 if a["origin"].startswith("http")]

    ok = downloaded = missing = mismatched = 0
    for a in artifacts:
        dest = REFERENCE / a["path"]
        if not dest.exists():
            if args.verify:
                print("  missing   {}".format(a["path"]))
                missing += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            print("  fetching  {}  ({:,} bytes)".format(a["path"], a["bytes"]))
            try:
                with urllib.request.urlopen(a["origin"], timeout=120) as r, \
                        dest.open("wb") as out:
                    shutil.copyfileobj(r, out)
            except Exception as exc:
                print("    FAILED: " + repr(exc)[:160])
                missing += 1
                continue
            downloaded += 1

        digest = sha256(dest)
        if digest == a["sha256"]:
            print("  verified  {}".format(a["path"]))
            ok += 1
        else:
            print("  MISMATCH  {}\n    expected {}\n    got      {}".format(
                a["path"], a["sha256"], digest))
            mismatched += 1

    if not args.verify and mismatched == 0 and missing == 0:
        for archive, target in EXTRACTIONS:
            src = REFERENCE / archive
            if not src.exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(src) as zf:
                zf.extractall(target)
            print("  extracted {} -> {}".format(
                archive, target.relative_to(REPO_ROOT)))

    print("\n{} verified, {} downloaded, {} missing, {} mismatched".format(
        ok, downloaded, missing, mismatched))
    if mismatched:
        print("A mismatch means the upstream artifact changed. Do not overwrite the")
        print("recorded hash: investigate, because every result was produced from")
        print("the bytes the manifest describes.")
    return 1 if (mismatched or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
