"""Refit selected models through 2026 and generate 2027 player forecasts."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor

from evaluate_baselines import DATASET, FEATURES, PA_THRESHOLD, RAW, model_pipeline
from evaluate_statcast_models import AUGMENTED_FEATURES, load_register, load_savant


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "forecasts"
REPORT = ROOT / "reports" / "phase4_2027_forecasts.md"


def xgb_classifier():
    return XGBClassifier(
        n_estimators=200, max_depth=2, learning_rate=0.03, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
        objective="binary:logistic", eval_metric="logloss", random_state=0, n_jobs=1,
    )


def xgb_regressor():
    return XGBRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.03, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
        objective="reg:squarederror", random_state=0, n_jobs=1,
    )


def load_fangraphs(year: int) -> pd.DataFrame:
    frame = pd.read_csv(RAW / f"fangraphs_batting_{year}.csv", low_memory=False)
    return frame.rename(columns={"PlayerId": "fg_id", "PA": "pa_fg", "wRC+": "wrc_plus"})[
        ["fg_id", *FEATURES]
    ].drop_duplicates("fg_id")


def build_2026_training_transition() -> pd.DataFrame:
    current = load_fangraphs(2025)
    register = load_register()
    current = current[current["pa_fg"] >= PA_THRESHOLD].merge(
        register, on="fg_id", how="left", validate="many_to_one"
    )
    current = current.merge(load_savant(2025), on="mlbam_id", how="left", validate="many_to_one")
    target = load_fangraphs(2026)[["fg_id", "pa_fg", "wrc_plus"]].rename(
        columns={"pa_fg": "next_pa", "wrc_plus": "next_wrc_plus"}
    )
    current = current.merge(target, on="fg_id", how="left", validate="one_to_one")
    current["next_pa"] = current["next_pa"].fillna(0)
    current["next_eligible_150"] = (current["next_pa"] >= PA_THRESHOLD).astype(int)
    current["next_wrc_plus"] = current["next_wrc_plus"].where(current["next_eligible_150"].eq(1))
    current.insert(1, "season", 2025)
    current.insert(2, "next_season", 2026)
    return current


def build_2027_scoring_frame() -> pd.DataFrame:
    raw = pd.read_csv(RAW / "fangraphs_batting_2026.csv", low_memory=False)
    current = raw.rename(columns={"PlayerId": "fg_id", "PA": "pa_fg", "wRC+": "wrc_plus"})[
        ["Name", "Team", "fg_id", *FEATURES]
    ].drop_duplicates("fg_id")
    register = load_register()
    current = current[current["pa_fg"] >= PA_THRESHOLD].merge(
        register, on="fg_id", how="left", validate="many_to_one"
    )
    current = current.merge(load_savant(2026), on="mlbam_id", how="left", validate="many_to_one")
    current.insert(1, "season", 2026)
    current.insert(2, "forecast_season", 2027)
    return current


def main() -> None:
    development = pd.read_csv(DATASET)
    holdout_transition = build_2026_training_transition()
    refit = pd.concat([development, holdout_transition], ignore_index=True)
    scoring = build_2027_scoring_frame()

    eligibility = model_pipeline(LogisticRegression(max_iter=2000, random_state=0))
    eligibility.fit(refit[FEATURES], refit["next_eligible_150"])
    scoring["predicted_eligibility_probability"] = eligibility.predict_proba(scoring[FEATURES])[:, 1]

    performance_train = refit[refit["next_eligible_150"].eq(1)]
    performance = model_pipeline(Ridge(alpha=1.0))
    performance.fit(performance_train[AUGMENTED_FEATURES], performance_train["next_wrc_plus"])
    scoring["predicted_wrc_plus"] = performance.predict(scoring[AUGMENTED_FEATURES])

    xgb_eligibility = xgb_classifier()
    xgb_eligibility.fit(refit[FEATURES], refit["next_eligible_150"])
    scoring["xgb_eligibility_probability"] = xgb_eligibility.predict_proba(scoring[FEATURES])[:, 1]
    xgb_performance = xgb_regressor()
    xgb_performance.fit(performance_train[AUGMENTED_FEATURES], performance_train["next_wrc_plus"])
    scoring["xgb_predicted_wrc_plus"] = xgb_performance.predict(scoring[AUGMENTED_FEATURES])

    scoring["predicted_wrc_plus_lift_vs_2026"] = scoring["predicted_wrc_plus"] - scoring["wrc_plus"]
    scoring["statcast_missing"] = scoring[AUGMENTED_FEATURES].isna().any(axis=1)
    scoring["review_candidate"] = (
        scoring["predicted_eligibility_probability"].ge(0.5)
        & scoring["predicted_wrc_plus"].ge(110)
        & scoring["predicted_wrc_plus_lift_vs_2026"].ge(10)
    )
    scoring = scoring.sort_values(
        ["review_candidate", "predicted_wrc_plus", "predicted_eligibility_probability"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    scoring.insert(0, "forecast_rank", scoring.index + 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scoring.to_csv(OUT_DIR / "player_forecasts_2027.csv", index=False)
    candidates = scoring[scoring["review_candidate"]].copy()
    candidates.to_csv(OUT_DIR / "review_candidates_2027.csv", index=False)
    dodgers = scoring[scoring["Team"].eq("LAD")].head(15).copy()
    dodgers.to_csv(OUT_DIR / "dodgers_forecast_shortlist_2027.csv", index=False)
    report = [
        "# Phase 4 2027 forecasting application",
        "",
        "Selected models were refit using development transitions through 2025 plus the previously held-out 2025→2026 transition. The 2026 holdout report was not modified.",
        "",
        "## Model choices",
        "",
        "- Playing-time eligibility: historical-statistics ridge-equivalent logistic regression.",
        "- Conditional performance: historical-plus-Statcast ridge regression.",
        "- Sensitivity: fixed XGBoost versions of both models.",
        "",
        "## Forecast population",
        "",
        f"- 2026 predictor population: {len(scoring):,} hitters with at least 150 PA.",
        f"- 2027 review candidates under the prespecified screen: {len(candidates):,}.",
        f"- Dodgers forecast shortlist: {len(dodgers):,} players, ranked by the primary predicted wRC+ after the candidate screen ordering.",
        f"- Statcast feature coverage: {int(scoring['statcast_missing'].sum()):,} of {len(scoring):,} rows have at least one missing Statcast feature; the selected pipeline imputes these values within the model pipeline and flags them here.",
        "- Candidate screen: predicted eligibility probability ≥ 0.50, predicted wRC+ ≥ 110, and predicted wRC+ at least 10 points above 2026 wRC+.",
        "",
        "These are model-screening candidates, not causal claims or final player evaluations. Baseball context, role, health, roster status, and playing-time opportunity require separate review.",
    ]
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {len(scoring):,} forecasts to {OUT_DIR / 'player_forecasts_2027.csv'}")
    print(f"Wrote {len(candidates):,} review candidates to {OUT_DIR / 'review_candidates_2027.csv'}")


if __name__ == "__main__":
    main()
