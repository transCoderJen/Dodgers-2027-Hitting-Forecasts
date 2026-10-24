# Phase 4 2027 forecasting application

Selected models were refit using development transitions through 2025 plus the previously held-out 2025→2026 transition. The 2026 holdout report was not modified.

## Model choices

- Playing-time eligibility: historical-statistics ridge-equivalent logistic regression.
- Conditional performance: historical-plus-Statcast ridge regression.
- Sensitivity: fixed XGBoost versions of both models.

## Forecast population

- 2026 predictor population: 407 hitters with at least 150 PA.
- 2027 review candidates under the prespecified screen: 2.
- Dodgers forecast shortlist: 13 players, ranked by the primary predicted wRC+ after the candidate screen ordering.
- Statcast feature coverage: 40 of 407 rows have at least one missing Statcast feature; the selected pipeline imputes these values within the model pipeline and flags them here.
- Candidate screen: predicted eligibility probability ≥ 0.50, predicted wRC+ ≥ 110, and predicted wRC+ at least 10 points above 2026 wRC+.

These are model-screening candidates, not causal claims or final player evaluations. Baseball context, role, health, roster status, and playing-time opportunity require separate review.
