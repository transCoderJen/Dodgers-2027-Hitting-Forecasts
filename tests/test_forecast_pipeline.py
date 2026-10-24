import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_2027_forecasts import build_2026_training_transition, build_2027_scoring_frame  # noqa: E402
from evaluate_baselines import DATASET  # noqa: E402
from evaluate_statcast_models import AUGMENTED_FEATURES  # noqa: E402


def test_2026_is_not_in_initial_development_transitions():
    development = pd.read_csv(DATASET)
    assert development["season"].max() == 2024
    assert development["next_season"].max() == 2025

    refit_transition = build_2026_training_transition()
    assert refit_transition["season"].eq(2025).all()
    assert refit_transition["next_season"].eq(2026).all()


def test_2027_scoring_frame_contains_no_future_outcomes():
    scoring = build_2027_scoring_frame()
    assert scoring["season"].eq(2026).all()
    assert scoring["forecast_season"].eq(2027).all()
    assert "next_pa" not in scoring.columns
    assert "next_wrc_plus" not in scoring.columns


def test_forecast_rows_have_unique_player_ids():
    forecasts = pd.read_csv(ROOT / "data" / "forecasts" / "player_forecasts_2027.csv")
    assert forecasts["fg_id"].notna().all()
    assert forecasts["fg_id"].is_unique
    assert forecasts["forecast_season"].eq(2027).all()


def test_statcast_features_are_present_for_forecast_population():
    forecasts = pd.read_csv(ROOT / "data" / "forecasts" / "player_forecasts_2027.csv")
    missing_rows = forecasts[AUGMENTED_FEATURES].isna().any(axis=1)
    assert missing_rows.sum() == forecasts["statcast_missing"].sum()
    assert missing_rows.mean() <= 0.10
    assert forecasts.loc[forecasts["review_candidate"], AUGMENTED_FEATURES].notna().all().all()


def test_dodgers_shortlist_is_ranked_and_labeled():
    shortlist = pd.read_csv(ROOT / "data" / "forecasts" / "dodgers_forecast_shortlist_2027.csv")
    assert len(shortlist) > 0
    assert shortlist["Team"].eq("LAD").all()
    assert shortlist["forecast_rank"].is_monotonic_increasing
