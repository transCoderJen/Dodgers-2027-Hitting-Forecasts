"""Evaluate simple, prespecified benchmarks against the Phase 3 models."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression, Ridge

from evaluate_baselines import DATASET, FEATURES, TRANSITION_YEARS, model_pipeline
from evaluate_statcast_models import AUGMENTED_FEATURES, build_2026_holdout


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "evaluation"
REPORT = ROOT / "reports" / "phase3_simple_benchmarks.md"
PA_THRESHOLD = 150
SHRINKAGE_K = 300


def score_performance(actual, predicted):
    return {
        "n": len(actual),
        "mae": mean_absolute_error(actual, predicted),
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
    }


def score_fold(train, test, transition, split):
    rows = []
    train_rate = train["next_eligible_150"].mean()
    base_probability = np.repeat(train_rate, len(test))
    rows.append(
        {
            "split": split,
            "transition": transition,
            "outcome": "eligibility",
            "benchmark": "training_base_rate",
            "n": len(test),
            "brier": brier_score_loss(test["next_eligible_150"], base_probability),
            "accuracy_at_0_5": accuracy_score(test["next_eligible_150"], base_probability >= 0.5),
        }
    )

    actual = test.loc[test["next_eligible_150"].eq(1), "next_wrc_plus"].dropna()
    retained = test.loc[actual.index]
    persistence = retained["wrc_plus"]
    shrinkage_weight = retained["pa_fg"] / (retained["pa_fg"] + SHRINKAGE_K)
    shrinkage = 100 + shrinkage_weight * (retained["wrc_plus"] - 100)
    for benchmark, prediction in [("current_wrc_plus_persistence", persistence), ("pa_weighted_mean_shrinkage", shrinkage)]:
        scores = score_performance(actual, prediction)
        rows.append({"split": split, "transition": transition, "outcome": "conditional_wrc_plus", "benchmark": benchmark, **scores})

    for feature_set, label in [(FEATURES, "historical_ridge"), (AUGMENTED_FEATURES, "historical_plus_statcast_ridge")]:
        eligibility_model = model_pipeline(LogisticRegression(max_iter=2000, random_state=0))
        eligibility_model.fit(train[feature_set], train["next_eligible_150"])
        probability = eligibility_model.predict_proba(test[feature_set])[:, 1]
        rows.append(
            {
                "split": split,
                "transition": transition,
                "outcome": "eligibility",
                "benchmark": label,
                "n": len(test),
                "brier": brier_score_loss(test["next_eligible_150"], probability),
                "accuracy_at_0_5": accuracy_score(test["next_eligible_150"], probability >= 0.5),
            }
        )
        training_performance = train[train["next_eligible_150"].eq(1)]
        performance_model = model_pipeline(Ridge(alpha=1.0))
        performance_model.fit(training_performance[feature_set], training_performance["next_wrc_plus"])
        performance_prediction = performance_model.predict(retained[feature_set])
        scores = score_performance(actual, performance_prediction)
        rows.append(
            {
                "split": split,
                "transition": transition,
                "outcome": "conditional_wrc_plus",
                "benchmark": label,
                **scores,
            }
        )
    return rows


def main():
    data = pd.read_csv(DATASET)
    rows = []
    train_years = []
    for test_year in TRANSITION_YEARS:
        if len(train_years) >= 3:
            train = data[data["season"].isin(train_years)]
            test = data[data["season"].eq(test_year)]
            rows.extend(score_fold(train, test, f"{test_year}->{test_year + 1}", "development"))
        train_years.append(test_year)
    development = pd.DataFrame(rows)

    holdout = build_2026_holdout()
    holdout_rows = pd.DataFrame(score_fold(data, holdout, "2025->2026", "2026_holdout"))
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    development.to_csv(EVAL_DIR / "phase3_simple_benchmark_development.csv", index=False)
    holdout_rows.to_csv(EVAL_DIR / "phase3_simple_benchmark_2026.csv", index=False)

    development_summary = development.groupby(["outcome", "benchmark"], as_index=False).mean(numeric_only=True)
    report = [
        "# Phase 3 simple benchmark comparison",
        "",
        f"Simple benchmarks are prespecified and use shrinkage constant K={SHRINKAGE_K} PA; K was not tuned on the 2026 holdout.",
        "Development-fold averages are the basis for model comparison. The 2026 results are reported separately as a final reference.",
        "",
        "## Development-fold averages",
        "",
        development_summary.to_markdown(index=False),
        "",
        "## 2026 holdout reference",
        "",
        holdout_rows.to_markdown(index=False),
        "",
        "## Benchmark definitions",
        "",
        "- Training base rate: every player receives the training-fold eligibility rate.",
        "- Current wRC+ persistence: next-season wRC+ equals current-season wRC+.",
        "- PA-weighted mean shrinkage: `100 + PA / (PA + 300) * (current_wRC+ - 100)`.",
        "- Ridge rows are the learned historical and historical-plus-Statcast models for direct comparison.",
    ]
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote benchmark report to {REPORT}")
    print(development_summary.to_string(index=False))


if __name__ == "__main__":
    main()
