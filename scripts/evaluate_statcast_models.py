"""Compare historical-statistics baselines with matched Statcast-feature models."""

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_baselines import (
    DATASET,
    FEATURES,
    PA_THRESHOLD,
    RAW,
    TRANSITION_YEARS,
    evaluate_2026_holdout,
    model_pipeline,
)


STATCAST_FEATURES = [
    "xwoba",
    "woba",
    "hard_hit_percent",
    "barrel_batted_rate",
    "sweet_spot_percent",
    "exit_velocity_avg",
]
AUGMENTED_FEATURES = FEATURES + STATCAST_FEATURES
ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "evaluation"
REPORT = ROOT / "reports" / "phase3_statcast_comparison.md"


def load_savant(year: int) -> pd.DataFrame:
    frame = pd.read_csv(RAW / f"savant_custom_{year}.csv", low_memory=False)
    frame = frame.rename(columns={"player_id": "mlbam_id"})
    return frame[["mlbam_id", *STATCAST_FEATURES]].drop_duplicates("mlbam_id")


def load_register() -> pd.DataFrame:
    parts = []
    for suffix in "0123456789abcdef":
        parts.append(pd.read_csv(RAW / f"chadwick_people-{suffix}.csv", usecols=["key_fangraphs", "key_mlbam"]))
    return (
        pd.concat(parts, ignore_index=True)
        .rename(columns={"key_fangraphs": "fg_id", "key_mlbam": "mlbam_id"})
        .dropna()
        .drop_duplicates("fg_id")
    )


def evaluate_fold(train, test, feature_set, label):
    eligibility = model_pipeline(LogisticRegression(max_iter=2000, random_state=0))
    eligibility.fit(train[feature_set], train["next_eligible_150"])
    probability = eligibility.predict_proba(test[feature_set])[:, 1]
    predicted = (probability >= 0.5).astype(int)

    performance_train = train[train["next_eligible_150"].eq(1)]
    performance_test = test[test["next_eligible_150"].eq(1)].dropna(subset=["next_wrc_plus"])
    performance = model_pipeline(Ridge(alpha=1.0))
    performance.fit(performance_train[feature_set], performance_train["next_wrc_plus"])
    wrc_prediction = performance.predict(performance_test[feature_set])
    return {
        "model": label,
        "eligibility_roc_auc": roc_auc_score(test["next_eligible_150"], probability),
        "eligibility_brier": brier_score_loss(test["next_eligible_150"], probability),
        "eligibility_accuracy": accuracy_score(test["next_eligible_150"], predicted),
        "performance_mae": mean_absolute_error(performance_test["next_wrc_plus"], wrc_prediction),
        "performance_rmse": mean_squared_error(performance_test["next_wrc_plus"], wrc_prediction) ** 0.5,
        "eligibility_n": len(test),
        "performance_n": len(performance_test),
    }


def build_2026_holdout():
    current = pd.read_csv(RAW / "fangraphs_batting_2025.csv", low_memory=False).rename(
        columns={"PlayerId": "fg_id", "PA": "pa_fg", "wRC+": "wrc_plus"}
    )[["fg_id", *FEATURES]]
    register = load_register()
    current = current[current["pa_fg"] >= PA_THRESHOLD].merge(register, on="fg_id", how="left", validate="many_to_one")
    current = current.merge(load_savant(2025), on="mlbam_id", how="left", validate="many_to_one")

    target = pd.read_csv(RAW / "fangraphs_batting_2026.csv", low_memory=False).rename(
        columns={"PlayerId": "fg_id", "PA": "next_pa", "wRC+": "next_wrc_plus"}
    )[["fg_id", "next_pa", "next_wrc_plus"]]
    holdout = current.merge(target, on="fg_id", how="left", validate="one_to_one")
    holdout["next_pa"] = holdout["next_pa"].fillna(0)
    holdout["next_eligible_150"] = (holdout["next_pa"] >= PA_THRESHOLD).astype(int)
    return holdout


def evaluate_holdout(train, holdout, feature_set, label):
    eligibility = model_pipeline(LogisticRegression(max_iter=2000, random_state=0))
    eligibility.fit(train[feature_set], train["next_eligible_150"])
    probability = eligibility.predict_proba(holdout[feature_set])[:, 1]
    performance_train = train[train["next_eligible_150"].eq(1)]
    actual = holdout[holdout["next_eligible_150"].eq(1)].dropna(subset=["next_wrc_plus"])
    performance = model_pipeline(Ridge(alpha=1.0))
    performance.fit(performance_train[feature_set], performance_train["next_wrc_plus"])
    prediction = performance.predict(actual[feature_set])
    return {
        "model": label,
        "eligibility_roc_auc": roc_auc_score(holdout["next_eligible_150"], probability),
        "eligibility_brier": brier_score_loss(holdout["next_eligible_150"], probability),
        "eligibility_accuracy": accuracy_score(holdout["next_eligible_150"], probability >= 0.5),
        "performance_mae": mean_absolute_error(actual["next_wrc_plus"], prediction),
        "performance_rmse": mean_squared_error(actual["next_wrc_plus"], prediction) ** 0.5,
        "eligibility_n": len(holdout),
        "performance_n": len(actual),
    }


def main() -> None:
    data = pd.read_csv(DATASET)
    folds = []
    train_years = []
    for test_year in TRANSITION_YEARS:
        if len(train_years) >= 3:
            train = data[data["season"].isin(train_years)]
            test = data[data["season"].eq(test_year)]
            for feature_set, label in [(FEATURES, "historical"), (AUGMENTED_FEATURES, "historical_plus_statcast")]:
                row = evaluate_fold(train, test, feature_set, label)
                row["test_transition"] = f"{test_year}->{test_year + 1}"
                folds.append(row)
        train_years.append(test_year)

    folds = pd.DataFrame(folds)
    holdout = build_2026_holdout()
    holdout_rows = pd.DataFrame(
        [
            evaluate_holdout(data, holdout, FEATURES, "historical"),
            evaluate_holdout(data, holdout, AUGMENTED_FEATURES, "historical_plus_statcast"),
        ]
    )
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(EVAL_DIR / "phase3_statcast_folds.csv", index=False)
    holdout_rows.to_csv(EVAL_DIR / "phase3_statcast_2026_holdout.csv", index=False)

    development = folds.groupby("model").mean(numeric_only=True)
    comparison = development.loc["historical_plus_statcast"] - development.loc["historical"]
    holdout_comparison = holdout_rows.set_index("model").loc["historical_plus_statcast"] - holdout_rows.set_index("model").loc["historical"]
    lines = [
        "# Phase 3 historical versus Statcast model comparison",
        "",
        "Both models use identical expanding temporal folds, preprocessing, outcomes, and the 2026 holdout.",
        "The historical model uses current-season PA, wRC+, wOBA, ISO, BB%, and K%. The augmented model adds six current-season Statcast features.",
        "",
        "## Development-fold averages",
        "",
        development.to_markdown(),
        "",
        "## Augmented minus historical development change",
        "",
        comparison.to_frame("change").to_markdown(),
        "",
        "## 2026 holdout",
        "",
        holdout_rows.to_markdown(index=False),
        "",
        "## Augmented minus historical 2026 change",
        "",
        holdout_comparison.to_frame("change").to_markdown(),
        "",
        "Interpretation: lower Brier score, MAE, and RMSE are better; higher ROC AUC and accuracy are better. The 2026 holdout was not used for model selection.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote comparison report to {REPORT}")
    print(development.to_string())
    print(holdout_rows.to_string(index=False))


if __name__ == "__main__":
    main()
