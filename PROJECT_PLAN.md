# Beyond the Box Score — Project Plan

## Project purpose

Build a research-quality MLB hitting analytics portfolio project for a Los Angeles Dodgers quantitative-analysis internship. The project will use public FanGraphs, Baseball Savant/Statcast, and Chadwick Register data to evaluate whether underlying hitting skill improves next-season wRC+ forecasts and to identify potentially undervalued players.

This is a statistical research project, not merely a machine-learning demonstration. It must emphasize baseball reasoning, temporal validation, reproducibility, uncertainty, interpretation, and the distinction between prediction and causation.

## Primary research question

Do Statcast measures of underlying hitting skill improve predictions of next-season wRC+ beyond traditional batting statistics, recent performance, and aging effects?

Compare a historical-statistics baseline with an otherwise comparable model that adds Statcast features. The null hypothesis is that Statcast features do not improve out-of-sample accuracy; the alternative is that they do. Compare RMSE and MAE with paired comparisons and uncertainty intervals. Do not assume complex models outperform simple baselines.

## Confirmed two-part framework

1. **Playing-time eligibility:** estimate `P(PA_(t+1) >= 150 | X_t)`.
2. **Conditional performance:** estimate `E[wRC+_(t+1) | X_t, PA_(t+1) >= 150]`.

Initial rules are provisional until the quantitative data audit is complete:

- Predictor population: hitters with at least 150 PA in season `t`.
- Classification outcome: whether the hitter reaches 150 PA in `t+1`.
- Regression population: hitters with at least 150 PA in `t+1`.
- Preserve lower-PA players in raw source files.
- Test alternative thresholds of 100, 200, and 300 PA.

## Data sources and constraints

- **FanGraphs:** wRC+, PA, wOBA, ISO, BB%, K%, BABIP, and historical batting statistics.
- **Baseball Savant:** xwOBA, average exit velocity, barrel rate, hard-hit rate, launch angle, sweet-spot rate, and related Statcast metrics. Statcast coverage begins in 2015.
- **Chadwick Register:** stable mapping from FanGraphs IDs to MLBAM IDs. Never join by names.

Document provider, URL, parameters, retrieval timestamp, schema, checksum, row count, errors, and usage limitations. Do not publish bulk raw FanGraphs or Savant data without permission. Statcast missingness may be non-random, and measurement systems changed over time, including the 2020 Hawk-Eye transition.

## Historical period and validation

Use seasons beginning in 2015. Pilot transitions:

`2015→2016`, `2018→2019`, `2021→2022`, and `2024→2025`.

Use expanding-window or rolling-origin validation. Never use information unavailable at the forecast date. Development predictors run through 2024 with development targets through 2025. The 2026 outcome is the locked final holdout and must not influence feature selection, tuning, or model selection. Future forecasting application: 2027.

Exclude transitions involving 2020 from the primary analysis initially; assess them only as a sensitivity analysis.

## Planned features

- Historical: previous-season and multi-year wRC+, wOBA, ISO, BB%, K%, BABIP, and PA.
- Statcast: xwOBA, barrel rate, hard-hit rate, average exit velocity, launch angle, and sweet-spot rate.
- Context: age, playing time, season effects, and sample-size reliability.
- Derived: `xwOBA - wOBA`, reliability-weighted multi-year performance, regression toward population means, and nonlinear aging interactions.

Feature importance must not be presented as causal evidence.

## Modeling sequence — only after Phase 1 approval

1. Persistence baseline.
2. Regression-to-the-mean and aging baseline.
3. Regularized linear regression using historical statistics.
4. Regularized regression with Statcast features.
5. XGBoost for nonlinear relationships.
6. Optional Bayesian hierarchical model with PyMC.

Evaluate RMSE, MAE, bias, classification discrimination and calibration, prediction-interval coverage, subgroup performance, and incremental Statcast value. Fit preprocessing, imputation, feature selection, and tuning inside temporal training folds only.

## Current phase: Phase 1 data feasibility audit

Do not train models, tune hyperparameters, conduct exploratory modeling, or inspect 2026 outcomes until the human researcher reviews and approves the empirical Phase 1 findings.

### Required Phase 1 work

1. Inspect and reuse the existing acquisition scripts, audit code, tests, and documentation.
2. Acquire all-player FanGraphs exports for 2015, 2016, 2018, 2019, 2021, 2022, 2024, and 2025 using permitted methods. Required fields include `IDfg`, `Season`, `PA`, and `wRC+`; do not use qualified-only exports.
3. Acquire the corresponding Savant files and all 16 Chadwick shards.
4. Audit missing identifiers, one-to-many mappings, duplicate player-season keys, unmatched records, and join rates.
5. Calculate player counts, 150-PA counts, feature coverage, missingness, duplicates, PA distributions, and missingness by season and playing time.
6. Validate both outcomes: next-season 150-PA eligibility and conditional next-season wRC+ population. Do not confuse an absent record with zero PA until the next-season export is verified exhaustive.
7. Write `reports/phase1_quantitative_audit.md` and machine-readable JSON/CSV results.
8. Run validation tests and document all limitations.

## Current evidence and blocker

Baseball Savant pilot files and all Chadwick shards have been acquired. FanGraphs exports remain the blocking dependency. Until they are available, report only source-level Savant and Chadwick observations; do not claim cross-source join rates, wRC+ outcomes, retention rates, or modeling feasibility.

## Collaboration checkpoint

At each major checkpoint:

1. Explain what was investigated.
2. Present observed findings with evidence.
3. Explain statistical implications and tradeoffs.
4. Identify unresolved decisions.
5. Recommend the next step.
6. Stop for human review before advancing phases.

## Approved methodological decisions

- **2020 treatment approved:** Include 2020 in descriptive EDA, but exclude the `2019→2020` and `2020→2021` transitions from the primary forecasting analysis.
- Treat transitions involving 2020 as sensitivity analyses because of the shortened COVID season, unusual playing-time environment, and Statcast tracking-system change.
- Preserve 2026 as the locked holdout and do not use it for EDA-driven decisions.

## Frozen model choices and 2027 application

- Playing-time eligibility: historical-statistics logistic ridge pipeline.
- Conditional next-season wRC+: historical-statistics plus Statcast ridge pipeline.
- XGBoost is retained as a fixed nonlinear sensitivity analysis, not the primary model.
- After the 2026 holdout evaluation, refit the selected models through 2026 and score the 2026 predictor population for 2027.
- A forecast-screened player is only a candidate for baseball review; the screen is not a causal or roster-value conclusion.

## Research standards

- Never invent data or results.
- Distinguish observations, assumptions, and decisions.
- Preserve reproducibility and deterministic pipelines where possible.
- Avoid temporal leakage and protect the 2026 holdout.
- Prefer interpretable baselines before complex models.
- Quantify uncertainty and explain assumptions.
- Avoid overclaiming player value or breakout potential.
