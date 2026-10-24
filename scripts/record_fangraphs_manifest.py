"""Record metadata for manually exported FanGraphs files."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/acquisition_manifest.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    existing = {item["file"] for item in manifest.get("downloads", [])}
    for path in sorted(args.data_dir.glob("fangraphs_batting_*.csv")):
        if path.as_posix() in existing or str(path) in existing:
            continue
        frame = pd.read_csv(path, nrows=0)
        modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        manifest.setdefault("downloads", []).append({
            "file": str(path),
            "provider": "FanGraphs",
            "source_url": "https://www.fangraphs.com/leaders/major-league",
            "retrieval_method": "manual authenticated member export",
            "retrieved_at_utc": modified,
            "schema": list(frame.columns),
            "row_count": int(sum(1 for _ in path.open("rb")) - 1),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "limitations": "Manual export; 2026 retained as locked holdout and excluded from audit.",
        })
    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
