"""Evaluate fixed nonlinear XGBoost models on the Phase 3 transitions."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

from evaluate_baselines import DATASET, FEATURES, TRANSITION_YEARS
from evaluate_statcast_models import AUGMENTED_FEATURES, build_2026_holdout


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "evaluation"
REPORT = ROOT / "reports" / "phase3_xgboost_evaluation.md"
MODEL_FEATURES = {"historical_xgboost": FEATURES, "historical_plus_statcast_xgboost": AUGMENTED_FEATURES}


def classifier():
    return XGBClassifier(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.03,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=0,
        n_jobs=1,
    )


def regressor():
    return XGBRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.03,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        objective="reg:squarederror",
        random_state=0,
        n_jobs=1,
    )


def evaluate(train, test, features, label, split, transition):
    eligibility_model = classifier()
    eligibility_model.fit(train[features], train["next_eligible_150"])
    probability = eligibility_model.predict_proba(test[features])[:, 1]
    performance_train = train[train["next_eligible_150"].eq(1)]
    performance_test = test[test["next_eligible_150"].eq(1)].dropna(subset=["next_wrc_plus"])
    performance_model = regressor()
    performance_model.fit(performance_train[features], performance_train["next_wrc_plus"])
    prediction = performance_model.predict(performance_test[features])
    return {
        "split": split,
        "transition": transition,
        "model": label,
        "eligibility_n": len(test),
        "eligibility_roc_auc": roc_auc_score(test["next_eligible_150"], probability),
        "eligibility_brier": brier_score_loss(test["next_eligible_150"], probability),
        "eligibility_accuracy": accuracy_score(test["next_eligible_150"], probability >= 0.5),
        "performance_n": len(performance_test),
        "performance_mae": mean_absolute_error(performance_test["next_wrc_plus"], prediction),
        "performance_rmse": mean_squared_error(performance_test["next_wrc_plus"], prediction) ** 0.5,
    }


def main():
    data = pd.read_csv(DATASET)
    rows = []
    train_years = []
    for test_year in TRANSITION_YEARS:
        if len(train_years) >= 3:
            train = data[data["season"].isin(train_years)]
            test = data[data["season"].eq(test_year)]
            for label, features in MODEL_FEATURES.items():
                rows.append(evaluate(train, test, features, label, "development", f"{test_year}->{test_year + 1}"))
        train_years.append(test_year)
    development = pd.DataFrame(rows)

    holdout = build_2026_holdout()
    holdout_rows = pd.DataFrame([
        evaluate(data, holdout, features, label, "2026_holdout", "2025->2026")
        for label, features in MODEL_FEATURES.items()
    ])
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    development.to_csv(EVAL_DIR / "phase3_xgboost_development.csv", index=False)
    holdout_rows.to_csv(EVAL_DIR / "phase3_xgboost_2026_holdout.csv", index=False)

    summary = development.groupby("model", as_index=False).mean(numeric_only=True)
    report = [
        "# Phase 3 XGBoost evaluation",
        "",
        "Fixed, conservative XGBoost models are compared using the same expanding temporal folds and 2026 holdout as the linear models.",
        "No hyperparameters were tuned using 2026.",
        "",
        "## Development-fold averages",
        "",
        summary.to_markdown(index=False),
        "",
        "## 2026 holdout reference",
        "",
        holdout_rows.to_markdown(index=False),
        "",
        "## Fixed settings",
        "",
        "`n_estimators=200`, `max_depth=2`, `learning_rate=0.03`, `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=0.8`, and `reg_lambda=5`.",
    ]
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote XGBoost report to {REPORT}")
    print(summary.to_string(index=False))
    print(holdout_rows.to_string(index=False))


if __name__ == "__main__":
    main()
