"""Build leakage-safe transition data for Phase 3 baseline modeling."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "derived" / "phase3_transitions.csv"
TRANSITIONS = [
    (2015, 2016),
    (2016, 2017),
    (2017, 2018),
    (2018, 2019),
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
    (2024, 2025),
]
PA_THRESHOLD = 150
SAVANT_FEATURES = [
    "xwoba",
    "woba",
    "hard_hit_percent",
    "barrel_batted_rate",
    "sweet_spot_percent",
    "exit_velocity_avg",
]
FG_FEATURES = ["pa_fg", "wrc_plus", "wOBA", "ISO", "BB%", "K%"]


def load_fangraphs(year: int) -> pd.DataFrame:
    frame = pd.read_csv(RAW / f"fangraphs_batting_{year}.csv", low_memory=False)
    frame = frame.rename(
        columns={"PlayerId": "fg_id", "PA": "pa_fg", "wRC+": "wrc_plus"}
    )
    required = {"fg_id", *FG_FEATURES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{year} FanGraphs file is missing columns: {sorted(missing)}")
    return frame[["fg_id", *FG_FEATURES]].drop_duplicates("fg_id").set_index("fg_id")


def load_savant(year: int) -> pd.DataFrame:
    frame = pd.read_csv(RAW / f"savant_custom_{year}.csv", low_memory=False)
    frame = frame.rename(columns={"player_id": "mlbam_id", "year": "season"})
    required = {"mlbam_id", *SAVANT_FEATURES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{year} Savant file is missing columns: {sorted(missing)}")
    return frame[["mlbam_id", *SAVANT_FEATURES]].drop_duplicates("mlbam_id")


def load_register() -> pd.DataFrame:
    parts = []
    for suffix in "0123456789abcdef":
        parts.append(
            pd.read_csv(
                RAW / f"chadwick_people-{suffix}.csv",
                usecols=["key_fangraphs", "key_mlbam"],
            )
        )
    return (
        pd.concat(parts, ignore_index=True)
        .rename(columns={"key_fangraphs": "fg_id", "key_mlbam": "mlbam_id"})
        .dropna()
        .drop_duplicates("fg_id")
    )


def build_dataset() -> pd.DataFrame:
    register = load_register()
    fg_by_year = {year: load_fangraphs(year) for pair in TRANSITIONS for year in pair}
    savant_by_year = {year: load_savant(year).set_index("mlbam_id") for year, _ in TRANSITIONS}
    rows = []

    for current, next_year in TRANSITIONS:
        current_fg = fg_by_year[current]
        eligible = current_fg.loc[current_fg["pa_fg"] >= PA_THRESHOLD].reset_index()
        eligible = eligible.merge(register, on="fg_id", how="left", validate="many_to_one")
        eligible = eligible.join(
            savant_by_year[current], on="mlbam_id", how="left", validate="many_to_one"
        )

        next_fg = fg_by_year[next_year][["pa_fg", "wrc_plus"]].rename(
            columns={"pa_fg": "next_pa", "wrc_plus": "next_wrc_plus"}
        )
        eligible = eligible.join(next_fg, on="fg_id", how="left", validate="many_to_one")
        eligible["next_pa"] = eligible["next_pa"].fillna(0)
        eligible["next_eligible_150"] = (eligible["next_pa"] >= PA_THRESHOLD).astype(int)
        eligible["next_wrc_plus"] = eligible["next_wrc_plus"].where(
            eligible["next_eligible_150"].eq(1)
        )
        eligible.insert(1, "season", current)
        eligible.insert(2, "next_season", next_year)
        rows.append(eligible)

    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["season", "fg_id"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    return result


if __name__ == "__main__":
    dataset = build_dataset()
    print(f"Wrote {len(dataset):,} rows to {OUT}")
    print(dataset.groupby(["season", "next_season"]).size().to_string())
