"""Download permitted Baseball Savant CSV exports and Chadwick Register shards.

FanGraphs files must be exported through the provider's available interface
and placed in data/raw/fangraphs_batting_{year}.csv. Do not circumvent access
controls or ignore provider terms. No modeling occurs in this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

PILOT_YEARS = [2015, 2016, 2018, 2019, 2021, 2022, 2024, 2025]
# Selections are season-level metrics available through the custom leaderboard.
SAVANT_COLUMNS = (
    "pa,k_percent,bb_percent,woba,xwoba,sweet_spot_percent,"
    "barrel_batted_rate,hard_hit_percent,exit_velocity_avg"
)


def checksum(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def fetch_csv(session: requests.Session, url: str, params: dict | None = None) -> tuple[bytes, str]:
    """Fail closed on HTTP denial, HTML fallbacks, or missing CSV columns."""
    response = session.get(url, params=params, timeout=45)
    response.raise_for_status()
    content = response.content
    if not content or b"<html" in content[:1000].lower() or b"<!doctype" in content[:1000].lower():
        raise ValueError(f"Unexpected non-CSV response from {response.url}")
    decoded = content.decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(decoded), strict=False)
    parsed = [row for _, row in zip(range(2), reader)]
    if len(parsed) < 2 or len(parsed[0]) < 3:
        raise ValueError(f"Missing CSV header/data from {response.url}")
    return content, response.url


def save_file(folder: Path, filename: str, contents: bytes, url: str, manifest: list[dict]):
    dest = folder / filename
    dest.write_bytes(contents)
    manifest.append({
        "file": str(dest), "source_url": url,
        "sha256": checksum(contents), "bytes": len(contents),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"OK {filename}: {len(contents):,} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--years", type=int, nargs="+", default=PILOT_YEARS)
    parser.add_argument("--sources", nargs="+", choices=["savant", "chadwick"], default=["savant", "chadwick"])
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.data_dir.parent / "acquisition_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"downloads": [], "failures": []}
    log = manifest.setdefault("downloads", [])
    session = requests.Session()
    session.headers.update({"User-Agent": "BeyondTheBoxScore-ResearchFeasibility/0.1 (academic use)"})
    failures = []

    def attempt(filename: str, url: str, params: dict | None = None):
        try:
            contents, resolved_url = fetch_csv(session, url, params)
            save_file(args.data_dir, filename, contents, resolved_url, log)
        except (requests.RequestException, ValueError) as exc:
            failures.append({"file": filename, "url": url, "error": str(exc)})
            print(f"BLOCKED {filename}: {exc}")
        time.sleep(1.5)

    if "savant" in args.sources:
        for year in sorted(set(args.years)):
            attempt(
                f"savant_custom_{year}.csv",
                "https://baseballsavant.mlb.com/leaderboard/custom",
                {"year": year, "type": "batter", "min": "1", "chart": "false",
                 "selections": SAVANT_COLUMNS, "csv": "true"},
            )
    if "chadwick" in args.sources:
        for suffix in "0123456789abcdef":
            attempt(
                f"chadwick_people-{suffix}.csv",
                f"https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{suffix}.csv",
            )
    manifest["failures"] = failures
    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} acquisition failures; see data/acquisition_manifest.json")

if __name__ == "__main__":
    main()
