"""Produce a source-level Phase 1 audit when FanGraphs exports are unavailable.

This does not infer FanGraphs-dependent joins, wRC+, or next-season outcomes.
It reports only facts observed in the acquired Savant and Chadwick files and
records the missing FanGraphs dependency as an explicit blocker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

YEARS = [2015, 2016, 2018, 2019, 2021, 2022, 2024, 2025]
FEATURES = [
    "xwoba", "woba", "hard_hit_percent", "barrel_batted_rate",
    "sweet_spot_percent", "exit_velocity_avg",
]


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_savant(raw_dir: Path, year: int) -> dict:
    path = raw_dir / f"savant_custom_{year}.csv"
    if not path.exists():
        return {"status": "missing", "file": path.name}
    frame = pd.read_csv(path, low_memory=False)
    required = {"player_id", "year", "pa", *FEATURES}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        return {"status": "invalid", "file": path.name, "missing_columns": missing_columns}
    numeric_pa = pd.to_numeric(frame["pa"], errors="coerce")
    return {
        "status": "observed",
        "file": path.name,
        "sha256": file_sha(path),
        "rows": int(len(frame)),
        "player_season_duplicates": int(frame.duplicated(["player_id", "year"]).sum()),
        "invalid_player_ids": int(pd.to_numeric(frame["player_id"], errors="coerce").isna().sum()),
        "season_values_other_than_expected": int(frame["year"].ne(year).sum()),
        "pa_missing": int(numeric_pa.isna().sum()),
        "pa_below_150": int((numeric_pa < 150).sum()),
        "pa_at_least_150": int((numeric_pa >= 150).sum()),
        "pa_min": float(numeric_pa.min()),
        "pa_max": float(numeric_pa.max()),
        "feature_missingness": {
            feature: {
                "count": int(frame[feature].isna().sum()),
                "fraction": round(float(frame[feature].isna().mean()), 6),
            }
            for feature in FEATURES
        },
    }


def audit_chadwick(raw_dir: Path) -> dict:
    frames = []
    files = []
    for suffix in "0123456789abcdef":
        path = raw_dir / f"chadwick_people-{suffix}.csv"
        if not path.exists():
            continue
        files.append(path)
        frames.append(pd.read_csv(path, usecols=["key_fangraphs", "key_mlbam"], low_memory=False))
    if len(files) != 16:
        return {"status": "incomplete", "files_found": len(files), "expected_files": 16}
    frame = pd.concat(frames, ignore_index=True)
    fg = pd.to_numeric(frame["key_fangraphs"], errors="coerce")
    mlbam = pd.to_numeric(frame["key_mlbam"], errors="coerce")
    pairs = pd.DataFrame({"fg": fg, "mlbam": mlbam}).dropna().drop_duplicates()
    fg_conflicts = pairs.groupby("fg")["mlbam"].nunique()
    mlbam_conflicts = pairs.groupby("mlbam")["fg"].nunique()
    return {
        "status": "observed",
        "files": [path.name for path in files],
        "sha256": {path.name: file_sha(path) for path in files},
        "raw_rows": int(len(frame)),
        "missing_fangraphs_id": int(fg.isna().sum()),
        "missing_mlbam_id": int(mlbam.isna().sum()),
        "complete_pair_rows": int(pd.DataFrame({"fg": fg, "mlbam": mlbam}).dropna().shape[0]),
        "distinct_complete_pairs": int(len(pairs)),
        "conflicted_fangraphs_ids": int((fg_conflicts > 1).sum()),
        "conflicted_mlbam_ids": int((mlbam_conflicts > 1).sum()),
    }


def markdown(report: dict) -> str:
    lines = [
        "# Phase 1 quantitative audit",
        f"Generated UTC: {report['generated_at_utc']}",
        "",
        "## Scope and evidence boundary",
        "This is a partial empirical audit of the acquired Baseball Savant and Chadwick files.",
        "FanGraphs season exports were not available, so no FanGraphs-to-MLBAM joins, wRC+",
        "statistics, next-season outcomes, or retention rates are reported or inferred.",
        "The 2026 holdout was not acquired or inspected.",
        "",
        "## Acquisition status",
        f"- Savant pilot seasons acquired: {', '.join(map(str, report['years']))}",
        f"- Chadwick shards acquired: {report['chadwick'].get('files_found', 16)}/16",
        "- FanGraphs pilot exports: 0/8; required to complete the cross-source audit",
        "",
        "## Baseball Savant source audit",
        "| Season | Rows | PA <150 | PA >=150 | Duplicate keys | Missing xwOBA | Missing exit velocity |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for year in report["years"]:
        rec = report["savant"][str(year)]
        miss = rec["feature_missingness"]
        lines.append(
            f"| {year} | {rec['rows']} | {rec['pa_below_150']} | {rec['pa_at_least_150']} | "
            f"{rec['player_season_duplicates']} | {miss['xwoba']['count']} | {miss['exit_velocity_avg']['count']} |"
        )
    lines += [
        "",
        "Missing-feature rates are calculated over the acquired Savant rows; they are not",
        "conditional on a successful FanGraphs/Chadwick join.",
        "",
        "## Chadwick identifier crosswalk",
        f"- Raw register rows: {report['chadwick'].get('raw_rows', 'unavailable'):,}",
        f"- Complete ID-pair rows: {report['chadwick'].get('complete_pair_rows', 'unavailable'):,}",
        f"- Distinct complete pairs: {report['chadwick'].get('distinct_complete_pairs', 'unavailable'):,}",
        f"- Conflicting FanGraphs IDs: {report['chadwick'].get('conflicted_fangraphs_ids', 'unavailable')}",
        f"- Conflicting MLBAM IDs: {report['chadwick'].get('conflicted_mlbam_ids', 'unavailable')}",
        "",
        "## Blockers and recommendation",
        "1. Obtain permitted, all-player FanGraphs exports for the eight pilot seasons with IDfg, PA, and wRC+.",
        "2. Re-run the strict combined audit to measure join coverage, missingness after joining, duplicate keys, and 150-PA transitions.",
        "3. Do not begin predictive modeling or claim that Statcast adds predictive value until those checks pass.",
        "",
        "Current recommendation: conditional GO for completing local ingestion; NO-GO for EDA/modeling based on this partial audit.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--json", type=Path, default=Path("data/audit/phase1_quantitative.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/phase1_quantitative_audit.md"))
    args = parser.parse_args()
    savant = {str(year): audit_savant(args.data_dir, year) for year in YEARS}
    chadwick = audit_chadwick(args.data_dir)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "years": YEARS,
        "holdout_year_excluded": 2026,
        "savant": savant,
        "chadwick": chadwick,
        "fangraphs": {"status": "missing", "files_found": 0, "files_expected": len(YEARS)},
        "transitions": {"status": "blocked_until_fangraphs_acquired"},
        "limitations": [
            "No FanGraphs exports were available, so cross-source joins and outcomes are not estimable.",
            "Savant missingness is source-level and may reflect metric availability rather than join failure.",
            "No conclusions about model feasibility or Statcast incremental value are justified yet.",
        ],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
