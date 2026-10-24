"""Audit baseball season-level CSVs; NEVER calculate models or inspect 2026 outcomes.

Outputs an evidence log plus a machine-readable audit; fails on duplicate player/season
keys instead of silently arbitrating them. Run only after authentic CSV retrieval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PILOT_YEARS = [2015, 2016, 2018, 2019, 2021, 2022, 2024, 2025]
# Primary transitions cover all acquired development seasons while excluding
# transitions involving the pandemic-shortened 2020 season. The 2025->2026
# transition is intentionally absent because 2026 is the locked holdout.
TRANSITIONS = [
    (2015, 2016), (2016, 2017), (2017, 2018), (2018, 2019),
    (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025),
]
SAVANT_FEATURES = ["xwoba", "woba", "hard_hit_percent", "barrel_batted_rate", "sweet_spot_percent", "exit_velocity_avg"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def id_to_int(series: pd.Series) -> pd.Series:
    """Pandas nullable integers avoid float-based ID corruption."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def read_fangraphs(path: Path, year: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # FanGraphs' current member export labels the stable FanGraphs identifier
    # `PlayerId` and omits Season because the export is already season-specific.
    # Older/exported formats may use `IDfg` and/or include `Season`.
    id_column = "IDfg" if "IDfg" in df.columns else "PlayerId"
    needed = [id_column, "PA", "wRC+"]
    if any(c not in df.columns for c in needed):
        raise ValueError(f"{path.name}: missing FanGraphs fields {set(needed) - set(df.columns)}")
    if "Season" in df and df["Season"].astype(str).ne(str(year)).any():
        raise ValueError(f"{path.name}: season values differ from {year}")
    out = pd.DataFrame({
        "fg_id": id_to_int(df[id_column]),
        "season": year,
        "pa_fg": pd.to_numeric(df["PA"], errors="coerce"),
        "wrc_plus": pd.to_numeric(df["wRC+"], errors="coerce"),
    })
    if out["fg_id"].isna().any():
        raise ValueError(f"{path.name}: missing/unparseable FanGraphs IDs")
    if out.duplicated(["fg_id", "season"]).any():
        raise ValueError(f"{path.name}: duplicate player-season ID; export player totals, not team splits")
    if out["pa_fg"].isna().any() or (out["pa_fg"] < 0).any():
        raise ValueError(f"{path.name}: invalid PA")
    return out


def read_savant(path: Path, year: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Custom CSV typically supplies MLBAM "player_id", year and selected metric names.
    needed = ["player_id", "year", "pa", *SAVANT_FEATURES]
    if any(c not in df for c in needed):
        raise ValueError(f"{path.name}: missing required Savant columns {set(needed)-set(df.columns)}; inspect header")
    df["mlbam_id"] = id_to_int(df["player_id"])
    df["season"] = id_to_int(df["year"])
    if df["mlbam_id"].isna().any() or df["season"].isna().any() or df["season"].ne(year).any():
        raise ValueError(f"{path.name}: invalid player_id or season")
    if df.duplicated(["mlbam_id", "season"]).any():
        raise ValueError(f"{path.name}: duplicate MLBAM player-season keys")
    out = df[["mlbam_id", "season"]].copy()
    out["pa_savant"] = pd.to_numeric(df["pa"], errors="coerce")
    for metric in SAVANT_FEATURES:
        if metric in df:
            out[metric] = pd.to_numeric(df[metric].astype(str).str.rstrip("%"), errors="coerce")

    return out


def read_chadwick(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    paths = [raw_dir / f"chadwick_people-{part}.csv" for part in "0123456789abcdef"]
    absent = [p.name for p in paths if not p.exists()]
    if absent:
        raise FileNotFoundError(f"Missing Chadwick shards: {absent}")
    parts = []
    for path in paths:
        shard = pd.read_csv(path, usecols=["key_fangraphs", "key_mlbam"], low_memory=False)
        parts.append(shard)
    df = pd.concat(parts, ignore_index=True)
    df["fg_id"] = id_to_int(df["key_fangraphs"])
    df["mlbam_id"] = id_to_int(df["key_mlbam"])
    paired = df.dropna(subset=["fg_id", "mlbam_id"])[["fg_id", "mlbam_id"]].drop_duplicates()
    conflicting_fg = paired.groupby("fg_id")["mlbam_id"].nunique()
    conflicting_mlbam = paired.groupby("mlbam_id")["fg_id"].nunique()
    conflicts_fg = set(conflicting_fg[conflicting_fg > 1].index)
    conflicts_mlbam = set(conflicting_mlbam[conflicting_mlbam > 1].index)
    clean = paired[~paired["fg_id"].isin(conflicts_fg) & ~paired["mlbam_id"].isin(conflicts_mlbam)]
    return clean, {"raw_rows": len(df), "distinct_complete_pairs": len(paired),
                   "conflicted_fangraphs_ids": len(conflicts_fg),
                   "conflicted_mlbam_ids": len(conflicts_mlbam),
                   "unambiguous_pairs": len(clean)}


def compute(raw_dir: Path, years: list[int], transitions: list[tuple[int, int]]) -> dict:
    reg, regstats = read_chadwick(raw_dir)
    report = {"as_of_utc": datetime.now(timezone.utc).isoformat(),
              "source_checksums": {}, "chadwick": regstats,
              "seasons": {}, "transitions": {}, "warnings": []}
    fangraphs: dict[int, pd.DataFrame] = {}
    for year in years:
        fgpath, sapath = (raw_dir / f"fangraphs_batting_{year}.csv",
                          raw_dir / f"savant_custom_{year}.csv")
        if not fgpath.exists() or not sapath.exists():
            raise FileNotFoundError(f"Missing {year} FG and/or Savant CSV; must be complete low-PA player population")
        fg, sa = read_fangraphs(fgpath, year), read_savant(sapath, year)
        fangraphs[year] = fg
        # Ensure full FG cohort still includes <150 PA batters before selecting eligibles.
        below_150 = int((fg["pa_fg"] < 150).sum())
        if below_150 == 0:
            raise ValueError(f"{fgpath.name}: no hitters below 150 PA — possible truncated/qualified-only export")
        eligible = fg.loc[fg["pa_fg"] >= 150].copy()
        joined = eligible.merge(reg, on="fg_id", how="left", validate="many_to_one", indicator="id_join")
        if joined.loc[joined["mlbam_id"].notna(), "mlbam_id"].duplicated().any():
            raise ValueError(f"{year}: multiple eligible FanGraphs players resolve to one MLBAM ID")
        # Multiple *unmapped* FG IDs legitimately share null MLBAM, so 1:1
        # validation would falsely reject the cohort. The Savant side is unique.
        merged = joined.merge(sa, on=["mlbam_id", "season"], how="left", validate="many_to_one", indicator="savant_join")
        missing = {k: {"count": int(merged[k].isna().sum()), "fraction": round(float(merged[k].isna().mean()), 4)}
                   for k in SAVANT_FEATURES}
        unmatched = joined.loc[joined["mlbam_id"].isna(), "fg_id"].astype(str).tolist()
        not_savant = merged.loc[merged["savant_join"].eq("left_only") & merged["mlbam_id"].notna(), "fg_id"].astype(str).tolist()
        pa_disagreement = (merged["pa_fg"] - merged["pa_savant"]).abs().dropna()
        match_rows = merged.loc[merged["savant_join"].eq("both")]
        matched_missing = {k: round(float(match_rows[k].isna().mean()), 4) if len(match_rows) else None
                           for k in SAVANT_FEATURES}
        report["seasons"][str(year)] = {
            "fangraphs_all_rows": len(fg), "fangraphs_below_150_PA": below_150,
            "eligible_fg_PA_at_least_150": len(eligible), "savant_all_rows": len(sa),
            "matched_chadwick": int(joined["mlbam_id"].notna().sum()),
            "matched_savant": int(merged["savant_join"].eq("both").sum()),
            "n_PA_disagreements_over_5": int((pa_disagreement > 5).sum()),
            "join_rate": round(float(merged["savant_join"].eq("both").mean()), 4) if len(merged) else None,
            "savant_feature_missingness": missing,
            "savant_feature_missingness_among_matched_players": matched_missing,
            "unmatched_fangraphs_ids": unmatched[:30],
            "missing_savant_fangraphs_ids": not_savant[:30],
        }
        report["source_checksums"][fgpath.name] = sha(fgpath)
        report["source_checksums"][sapath.name] = sha(sapath)
    for suffix in "0123456789abcdef":
        path = raw_dir / f"chadwick_people-{suffix}.csv"
        report["source_checksums"][path.name] = sha(path)
    for a, b in transitions:
        if a not in fangraphs or b not in fangraphs:
            continue
        present = fangraphs[b][["fg_id", "pa_fg"]].rename(columns={"pa_fg": "next_pa"})
        cohort = fangraphs[a].loc[fangraphs[a]["pa_fg"] >= 150, ["fg_id"]]
        followed = cohort.merge(present, on="fg_id", how="left", validate="one_to_one")
        # No FG record in a COMPLETE next-season file = zero MLB plate appearances.
        plays = followed["next_pa"].fillna(0) >= 150
        report["transitions"][f"{a}->{b}"] = {
            "n_predictor_eligible": len(followed),
            "n_next_year_at_least_150_PA": int(plays.sum()),
            "rate_next_year_at_least_150_PA": round(float(plays.mean()), 4) if len(followed) else None,
            "n_no_next_season_fg_row": int(followed["next_pa"].isna().sum()),
            "n_next_year_below_150_PA": int((~plays).sum()),
        }
    return report


def to_markdown(result: dict) -> str:
    lines = ["# Phase 1 quantitative audit — actual CSV results",
             f"Generated UTC: {result['as_of_utc']}", "",
             "Numbers below come only from real supplied CSV files; they are not estimates.", "",
             "## Player-season data and join coverage",
             "| Season | FG rows (all PA) | Eligible ≥150 PA | Savant rows | Crosswalk matches | Savant matches | Overall join rate |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for year, rec in result["seasons"].items():
        lines.append(f"| {year} | {rec['fangraphs_all_rows']} | {rec['eligible_fg_PA_at_least_150']} | {rec['savant_all_rows']} | {rec['matched_chadwick']} | {rec['matched_savant']} | {rec['join_rate']} |")
    lines.extend(["", "## Playing-time selection (next season; 2026 intentionally excluded)",
                  "| Transition | Eligible in t | ≥150 PA in t+1 | Retention | No next-year row |",
                  "|---|---:|---:|---:|---:|"])
    for pair, rec in result["transitions"].items():
        lines.append(f"| {pair} | {rec['n_predictor_eligible']} | {rec['n_next_year_at_least_150_PA']} | {rec['rate_next_year_at_least_150_PA']} | {rec['n_no_next_season_fg_row']} |")
    lines.extend(["", "## Feature missingness by season (eligible merged FG cohort)",
                  "| Season | " + " | ".join(SAVANT_FEATURES) + " |",
                  "|---|" + "---:|" * len(SAVANT_FEATURES)])
    for year, rec in result["seasons"].items():
        lines.append("| " + str(year) + " | " + " | ".join(
            f"{100*rec['savant_feature_missingness'][f]['fraction']:.1f}%" for f in SAVANT_FEATURES) + " |")
    lines += ["", "**Interpretation warning:** Missing-feature rates include unmatched Savant player-seasons."
              " Disentangle absent Statcast measurements from ID mismatches before downstream imputation.",
              "", "**Holdout integrity:** 2026 outcomes were not opened by this audit script."]
    return "\n".join(lines) + "\n"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/audit"))
    parser.add_argument("--report", type=Path, default=Path("reports/phase1_quantitative_audit.md"))
    parser.add_argument("--years", nargs="+", type=int, default=PILOT_YEARS)
    args = parser.parse_args()
    if 2026 in args.years:
        raise SystemExit("2026 is the locked outcome holdout. Exclude it from exploratory audits.")
    results = compute(args.data_dir, args.years, TRANSITIONS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    markdown = to_markdown(results)
    (args.output_dir / "phase1_quantitative.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (args.output_dir / "phase1_quantitative.md").write_text(markdown, encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown, encoding="utf-8")
    print(markdown)

if __name__ == "__main__":
    main()
