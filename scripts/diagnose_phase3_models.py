"""Quantify uncertainty, calibration, and subgroup behavior for Phase 3 models."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

from evaluate_baselines import DATASET, FEATURES, PA_THRESHOLD, RAW, TRANSITION_YEARS, model_pipeline
from evaluate_statcast_models import AUGMENTED_FEATURES, build_2026_holdout
from sklearn.linear_model import LogisticRegression, Ridge


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "evaluation"
REPORT = ROOT / "reports" / "phase3_model_diagnostics.md"
MODELS = {"historical": FEATURES, "historical_plus_statcast": AUGMENTED_FEATURES}
PA_BINS = [150, 250, 400, np.inf]
PA_LABELS = ["150-249", "250-399", "400+"]
N_BOOTSTRAPS = 1000


def fit_predictions(train, test, features, model_name, split, transition):
    eligibility_model = model_pipeline(LogisticRegression(max_iter=2000, random_state=0))
    eligibility_model.fit(train[features], train["next_eligible_150"])
    eligibility = test[["fg_id", "pa_fg", "next_eligible_150"]].copy()
    eligibility["model"] = model_name
    eligibility["split"] = split
    eligibility["transition"] = transition
    eligibility["predicted_probability"] = eligibility_model.predict_proba(test[features])[:, 1]

    performance_train = train[train["next_eligible_150"].eq(1)]
    performance_test = test[test["next_eligible_150"].eq(1)].dropna(subset=["next_wrc_plus"])
    performance_model = model_pipeline(Ridge(alpha=1.0))
    performance_model.fit(performance_train[features], performance_train["next_wrc_plus"])
    performance = performance_test[["fg_id", "pa_fg", "next_wrc_plus"]].copy()
    performance["model"] = model_name
    performance["split"] = split
    performance["transition"] = transition
    performance["predicted_wrc_plus"] = performance_model.predict(performance_test[features])
    return eligibility, performance


def collect_predictions(data, holdout):
    eligibility_parts = []
    performance_parts = []
    train_years = []
    for test_year in TRANSITION_YEARS:
        if len(train_years) >= 3:
            train = data[data["season"].isin(train_years)]
            test = data[data["season"].eq(test_year)]
            transition = f"{test_year}->{test_year + 1}"
            for model_name, features in MODELS.items():
                eligibility, performance = fit_predictions(
                    train, test, features, model_name, "development", transition
                )
                eligibility_parts.append(eligibility)
                performance_parts.append(performance)
        train_years.append(test_year)

    for model_name, features in MODELS.items():
        eligibility, performance = fit_predictions(
            data, holdout, features, model_name, "2026_holdout", "2025->2026"
        )
        eligibility_parts.append(eligibility)
        performance_parts.append(performance)
    return pd.concat(eligibility_parts, ignore_index=True), pd.concat(performance_parts, ignore_index=True)


def metric(values, metric_name):
    if metric_name == "roc_auc":
        return roc_auc_score(values["actual"], values["prediction"])
    errors = values["actual"] - values["prediction"]
    if metric_name == "mae":
        return np.abs(errors).mean()
    return np.sqrt(np.mean(errors**2))


def paired_bootstrap(eligibility, performance, split, metric_name, n_bootstraps=N_BOOTSTRAPS):
    rng = np.random.default_rng(20261023)
    deltas = []
    if metric_name == "roc_auc":
        source = eligibility[eligibility["split"].eq(split)].copy()
        key = ["fg_id", "transition"]
        hist = source[source.model.eq("historical")].rename(columns={"predicted_probability": "hist"})
        stat = source[source.model.eq("historical_plus_statcast")].rename(columns={"predicted_probability": "stat"})
        paired = hist.merge(stat[key + ["stat"]], on=key, validate="one_to_one")
        paired = paired.rename(columns={"next_eligible_150": "actual"})
        for _ in range(n_bootstraps):
            sample = paired.iloc[rng.integers(0, len(paired), len(paired))]
            if sample["actual"].nunique() < 2:
                continue
            deltas.append(
                roc_auc_score(sample["actual"], sample["stat"])
                - roc_auc_score(sample["actual"], sample["hist"])
            )
    else:
        source = performance[performance["split"].eq(split)].copy()
        key = ["fg_id", "transition"]
        hist = source[source.model.eq("historical")].rename(columns={"predicted_wrc_plus": "hist"})
        stat = source[source.model.eq("historical_plus_statcast")].rename(columns={"predicted_wrc_plus": "stat"})
        paired = hist.merge(stat[key + ["stat"]], on=key, validate="one_to_one")
        paired = paired.rename(columns={"next_wrc_plus": "actual"})
        for _ in range(n_bootstraps):
            sample = paired.iloc[rng.integers(0, len(paired), len(paired))]
            deltas.append(metric(sample.rename(columns={"stat": "prediction"}), metric_name))
            stat_value = metric(sample.rename(columns={"stat": "prediction"}), metric_name)
            hist_value = metric(sample.rename(columns={"hist": "prediction"}), metric_name)
            deltas[-1] = stat_value - hist_value

    return {
        "split": split,
        "metric": metric_name,
        "n_bootstraps": len(deltas),
        "statcast_minus_historical": np.mean(deltas),
        "ci_lower": np.quantile(deltas, 0.025),
        "ci_upper": np.quantile(deltas, 0.975),
    }


def calibration_table(eligibility, split):
    source = eligibility[eligibility["split"].eq(split)].copy()
    source["probability_bin"] = pd.cut(
        source["predicted_probability"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1], include_lowest=True
    )
    return (
        source.groupby(["model", "probability_bin"], observed=False)
        .agg(
            n=("next_eligible_150", "size"),
            mean_predicted=("predicted_probability", "mean"),
            observed_rate=("next_eligible_150", "mean"),
        )
        .reset_index()
    )


def subgroup_table(performance, split):
    source = performance[performance["split"].eq(split)].copy()
    source["pa_group"] = pd.cut(source["pa_fg"], bins=PA_BINS, labels=PA_LABELS, right=False)
    source["absolute_error"] = (source["next_wrc_plus"] - source["predicted_wrc_plus"]).abs()
    source["squared_error"] = (source["next_wrc_plus"] - source["predicted_wrc_plus"]) ** 2
    return (
        source.groupby(["model", "pa_group"], observed=False)
        .agg(
            n=("fg_id", "size"),
            mae=("absolute_error", "mean"),
            rmse=("squared_error", lambda x: np.sqrt(x.mean())),
        )
        .reset_index()
    )


def main() -> None:
    data = pd.read_csv(DATASET)
    holdout = build_2026_holdout()
    eligibility, performance = collect_predictions(data, holdout)

    uncertainty = pd.DataFrame(
        [
            paired_bootstrap(eligibility, performance, split, metric_name)
            for split in ["development", "2026_holdout"]
            for metric_name in ["roc_auc", "mae", "rmse"]
        ]
    )
    calibration_dev = calibration_table(eligibility, "development")
    calibration_holdout = calibration_table(eligibility, "2026_holdout")
    subgroup_dev = subgroup_table(performance, "development")
    subgroup_holdout = subgroup_table(performance, "2026_holdout")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    uncertainty.to_csv(EVAL_DIR / "phase3_paired_bootstrap.csv", index=False)
    calibration_dev.to_csv(EVAL_DIR / "phase3_calibration_development.csv", index=False)
    calibration_holdout.to_csv(EVAL_DIR / "phase3_calibration_2026.csv", index=False)
    subgroup_dev.to_csv(EVAL_DIR / "phase3_subgroups_development.csv", index=False)
    subgroup_holdout.to_csv(EVAL_DIR / "phase3_subgroups_2026.csv", index=False)

    report = [
        "# Phase 3 model diagnostics",
        "",
        "Diagnostics use the same fixed models and temporal splits as the baseline comparison.",
        "Bootstrap intervals describe paired resampling uncertainty in the Statcast-minus-historical metric difference; they do not represent new independent seasons.",
        "",
        "## Paired bootstrap intervals",
        "",
        uncertainty.to_markdown(index=False),
        "",
        "## Development eligibility calibration",
        "",
        calibration_dev.to_markdown(index=False),
        "",
        "## 2026 eligibility calibration",
        "",
        calibration_holdout.to_markdown(index=False),
        "",
        "## Development conditional-performance subgroups",
        "",
        subgroup_dev.to_markdown(index=False),
        "",
        "## 2026 conditional-performance subgroups",
        "",
        subgroup_holdout.to_markdown(index=False),
    ]
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote diagnostics report to {REPORT}")
    print(uncertainty.to_string(index=False))


if __name__ == "__main__":
    main()
