"""Snapshot the machine and software stack into experiments/_env_baseline.json.

Run once per machine, and again whenever the environment changes. Every
experiment already records its own environment; this file exists so the
documents can point at a single canonical description of the primary machine.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.experiment import REPO_ROOT, git_state, hardware, software_versions


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "primary"
    out = REPO_ROOT / "experiments" / ("_env_baseline_" + label + ".json")
    payload = {
        "label": label,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "hardware": hardware(),
        "software": software_versions(),
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    print("\nwritten to " + str(out))


if __name__ == "__main__":
    main()
