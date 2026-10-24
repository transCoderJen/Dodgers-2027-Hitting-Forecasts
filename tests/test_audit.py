"""Synthetic fixtures only — NOT evidence about actual MLB samples."""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit import compute, read_fangraphs, read_savant


def make_fixture(directory: Path):
    directory.mkdir(exist_ok=True, parents=True)
    for i in "0123456789abcdef":
        pd.DataFrame(columns=["key_fangraphs", "key_mlbam"]).to_csv(
            directory / f"chadwick_people-{i}.csv", index=False)
    pd.DataFrame({"key_fangraphs": [1, 2, 3],
                  "key_mlbam": [101, 102, 103]}).to_csv(
        directory / "chadwick_people-a.csv", index=False)
    pd.DataFrame({"IDfg": [1, 2, 3, 4], "Season": [2015]*4,
                  "PA": [300, 170, 50, 200], "wRC+": [150, 90, 110, 80]}).to_csv(
        directory / "fangraphs_batting_2015.csv", index=False)
    pd.DataFrame({"IDfg": [2, 3, 4], "Season": [2016]*3,
                  "PA": [160, 120, 400], "wRC+": [104, 90, 100]}).to_csv(
        directory / "fangraphs_batting_2016.csv", index=False)
    for y in [2015, 2016]:
        pd.DataFrame({"player_id": [101, 102, 103], "year": [y]*3,
             "pa": [300, 170, 50] if y == 2015 else [1, 160, 120],
             "xwoba": [0.38, 0.29, 0.31], "woba": [0.37, 0.30, 0.30],
             "hard_hit_percent": [43.2, 30.5, 25.0],
             "barrel_batted_rate": [12.5, 5.5, 3.1],
             "sweet_spot_percent": [34, 31, 26],
             "exit_velocity_avg": [91.1, 88.2, 87.2]}).to_csv(
                  directory / f"savant_custom_{y}.csv", index=False)


def test_crosswalk_and_player_return(tmp_path):
    make_fixture(tmp_path)
    report = compute(tmp_path, [2015, 2016], [(2015, 2016)])
    s = report["seasons"]["2015"]
    assert s["fangraphs_all_rows"] == 4
    assert s["eligible_fg_PA_at_least_150"] == 3
    assert s["matched_chadwick"] == 2
    assert s["matched_savant"] == 2
    assert s["join_rate"] == pytest.approx(2/3, abs=0.0001)
    t = report["transitions"]["2015->2016"]
    assert t["n_predictor_eligible"] == 3
    assert t["n_next_year_at_least_150_PA"] == 2
    assert t["n_no_next_season_fg_row"] == 1


def test_fangraphs_duplicate_ids_rejected(tmp_path):
    make_fixture(tmp_path)
    path = tmp_path / "fangraphs_batting_2015.csv"
    df = pd.read_csv(path)
    pd.concat([df, df.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        read_fangraphs(path, 2015)


def test_missing_savant_feature_rejected(tmp_path):
    make_fixture(tmp_path)
    path = tmp_path / "savant_custom_2015.csv"
    df = pd.read_csv(path).drop(columns=["xwoba"])
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        read_savant(path, 2015)


def test_qualified_only_fangraphs_export_rejected(tmp_path):
    make_fixture(tmp_path)
    path = tmp_path / "fangraphs_batting_2015.csv"
    df = pd.read_csv(path)
    df = df.loc[df["PA"] >= 150]
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="qualified-only"):
        compute(tmp_path, [2015, 2016], [(2015, 2016)])


def test_conflicting_crosswalk_kept_out_of_joins(tmp_path):
    make_fixture(tmp_path)
    path = tmp_path / "chadwick_people-b.csv"
    pd.DataFrame({"key_fangraphs": [1], "key_mlbam": [999]}).to_csv(path, index=False)
    report = compute(tmp_path, [2015, 2016], [(2015, 2016)])
    assert report["chadwick"]["conflicted_fangraphs_ids"] == 1
    assert report["seasons"]["2015"]["matched_chadwick"] == 1
