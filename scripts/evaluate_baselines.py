"""Evaluate fixed historical-statistics baselines with temporal validation."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATASET = ROOT / "data" / "derived" / "phase3_transitions.csv"
EVAL_DIR = ROOT / "data" / "evaluation"
REPORT = ROOT / "reports" / "phase3_baseline_evaluation.md"
FEATURES = ["pa_fg", "wrc_plus", "wOBA", "ISO", "BB%", "K%"]
PA_THRESHOLD = 150
TRANSITION_YEARS = [2015, 2016, 2017, 2018, 2021, 2022, 2023, 2024]
MIN_TRAIN_YEARS = 3


def model_pipeline(model):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        model,
    )


def load_fangraphs(year: int) -> pd.DataFrame:
    frame = pd.read_csv(RAW / f"fangraphs_batting_{year}.csv", low_memory=False)
    return frame.rename(columns={"PlayerId": "fg_id", "PA": "pa_fg", "wRC+": "wrc_plus"})[
        ["fg_id", *FEATURES]
    ].drop_duplicates("fg_id")


def evaluate_development() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(DATASET)
    folds = []
    prediction_rows = []
    train_years = []

    for test_year in TRANSITION_YEARS:
        if len(train_years) < MIN_TRAIN_YEARS:
            train_years.append(test_year)
            continue

        train = data[data["season"].isin(train_years)].copy()
        test = data[data["season"].eq(test_year)].copy()

        eligibility = model_pipeline(
            LogisticRegression(max_iter=2000, random_state=0)
        )
        eligibility.fit(train[FEATURES], train["next_eligible_150"])
        test_probability = eligibility.predict_proba(test[FEATURES])[:, 1]
        test_prediction = (test_probability >= 0.5).astype(int)

        performance_train = train[train["next_eligible_150"].eq(1)]
        performance_test = test[test["next_eligible_150"].eq(1)].dropna(
            subset=["next_wrc_plus"]
        )
        performance = model_pipeline(Ridge(alpha=1.0))
        performance.fit(performance_train[FEATURES], performance_train["next_wrc_plus"])
        wrc_prediction = performance.predict(performance_test[FEATURES])

        folds.append(
            {
                "test_transition": f"{test_year}->{test_year + 1}",
                "train_transitions": len(train_years),
                "eligibility_n": len(test),
                "eligibility_roc_auc": roc_auc_score(
                    test["next_eligible_150"], test_probability
                ),
                "eligibility_brier": brier_score_loss(
                    test["next_eligible_150"], test_probability
                ),
                "eligibility_accuracy": accuracy_score(
                    test["next_eligible_150"], test_prediction
                ),
                "performance_n": len(performance_test),
                "performance_mae": mean_absolute_error(
                    performance_test["next_wrc_plus"], wrc_prediction
                ),
                "performance_rmse": mean_squared_error(
                    performance_test["next_wrc_plus"], wrc_prediction
                )
                ** 0.5,
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "test_transition": f"{test_year}->{test_year + 1}",
                    "fg_id": test["fg_id"],
                    "actual_next_eligible_150": test["next_eligible_150"],
                    "predicted_next_eligible_probability": test_probability,
                }
            )
        )
        train_years.append(test_year)

    return pd.DataFrame(folds), pd.concat(prediction_rows, ignore_index=True), data


def evaluate_2026_holdout(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data.copy()
    current = load_fangraphs(2025)
    target = load_fangraphs(2026)[["fg_id", "pa_fg", "wrc_plus"]].rename(
        columns={"pa_fg": "next_pa", "wrc_plus": "next_wrc_plus"}
    )
    holdout = current[current["pa_fg"] >= PA_THRESHOLD].merge(
        target, on="fg_id", how="left", validate="one_to_one"
    )
    holdout["next_pa"] = holdout["next_pa"].fillna(0)
    holdout["actual_next_eligible_150"] = (holdout["next_pa"] >= PA_THRESHOLD).astype(int)

    eligibility = model_pipeline(LogisticRegression(max_iter=2000, random_state=0))
    eligibility.fit(train[FEATURES], train["next_eligible_150"])
    holdout["predicted_next_eligible_probability"] = eligibility.predict_proba(
        holdout[FEATURES]
    )[:, 1]

    performance_train = train[train["next_eligible_150"].eq(1)]
    performance = model_pipeline(Ridge(alpha=1.0))
    performance.fit(performance_train[FEATURES], performance_train["next_wrc_plus"])
    actual_performance = holdout[holdout["actual_next_eligible_150"].eq(1)].dropna(
        subset=["next_wrc_plus"]
    )
    holdout_predictions = performance.predict(actual_performance[FEATURES])
    summary = pd.DataFrame(
        [
            {
                "holdout": "2025->2026",
                "eligibility_n": len(holdout),
                "eligibility_roc_auc": roc_auc_score(
                    holdout["actual_next_eligible_150"],
                    holdout["predicted_next_eligible_probability"],
                ),
                "eligibility_brier": brier_score_loss(
                    holdout["actual_next_eligible_150"],
                    holdout["predicted_next_eligible_probability"],
                ),
                "performance_n": len(actual_performance),
                "performance_mae": mean_absolute_error(
                    actual_performance["next_wrc_plus"], holdout_predictions
                ),
                "performance_rmse": mean_squared_error(
                    actual_performance["next_wrc_plus"], holdout_predictions
                )
                ** 0.5,
            }
        ]
    )
    predictions = actual_performance[["fg_id", "next_pa", "next_wrc_plus"]].copy()
    predictions["predicted_next_wrc_plus"] = holdout_predictions
    return summary, predictions


def main() -> None:
    folds, eligibility_predictions, data = evaluate_development()
    holdout_summary, holdout_predictions = evaluate_2026_holdout(data)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(EVAL_DIR / "phase3_baseline_folds.csv", index=False)
    eligibility_predictions.to_csv(EVAL_DIR / "phase3_eligibility_predictions.csv", index=False)
    holdout_predictions.to_csv(EVAL_DIR / "phase3_2026_wrc_plus_predictions.csv", index=False)
    holdout_summary.to_csv(EVAL_DIR / "phase3_2026_holdout_summary.csv", index=False)

    development_summary = folds.drop(columns=["test_transition"]).mean(numeric_only=True)
    lines = [
        "# Phase 3 baseline evaluation",
        "",
        "Historical-statistics-only baselines evaluated with expanding temporal validation.",
        "The development folds exclude transitions involving 2020. The 2026 results are a final holdout and were not used during fitting or model selection.",
        "",
        "## Development-fold averages",
        "",
        folds.to_markdown(index=False),
        "",
        "## Final 2026 holdout",
        "",
        holdout_summary.to_markdown(index=False),
        "",
        "## Model definitions",
        "",
        "- Playing-time eligibility: standardized, median-imputed logistic regression predicting next-season PA >= 150.",
        "- Conditional performance: standardized, median-imputed ridge regression predicting next-season wRC+ among hitters who reach PA >= 150.",
        "- Predictors: current-season PA, wRC+, wOBA, ISO, BB%, and K%.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote baseline report to {REPORT}")
    print(f"Development folds: {len(folds)}; 2026 holdout rows: {len(holdout_predictions)}")
    print(development_summary.to_string())
    print(holdout_summary.to_string(index=False))


if __name__ == "__main__":
    main()
