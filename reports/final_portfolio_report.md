# Beyond the Box Score: 2027 MLB hitting forecasts

## Executive summary

This project tests whether current-season Statcast indicators improve next-season hitting forecasts beyond traditional batting statistics. The analysis separates two decisions:

1. Whether a hitter reaches 150 PA the following season.
2. Conditional next-season wRC+ among hitters who reach 150 PA.

The selected models are a historical-statistics logistic model for playing-time eligibility and a historical-plus-Statcast ridge model for conditional wRC+. XGBoost is retained as a nonlinear sensitivity analysis because its gains were negligible relative to the simpler ridge models.

The 2026 holdout produced 80.2% eligibility accuracy, 0.800 eligibility ROC AUC, 17.0 wRC+ MAE, and 21.4 wRC+ RMSE. These metrics are evaluation results, not guarantees about 2027 performance.

## Data acquisition and audit

Sources:

- FanGraphs member exports for all-player batting statistics, 2015–2026.
- Baseball Savant custom season exports for Statcast features, 2015–2026.
- Chadwick Register shards for FanGraphs-to-MLBAM ID crosswalks.

The Phase 1 audit used 2015–2025 for development and protected 2026 as the final holdout. Among eligible hitters with at least 150 PA, the audited FanGraphs/Savant joins were complete and no duplicate player-season keys were found. Primary non-2020 retention rates ranged from approximately 74% to 79%.

Full audit details are in [phase1_quantitative_audit.md](./phase1_quantitative_audit.md), with machine-readable results in [phase1_quantitative.json](../data/audit/phase1_quantitative.json). Source hashes and retrieval metadata are in [acquisition_manifest.json](../data/acquisition_manifest.json).

## EDA findings

- 2020 is materially different because of the shortened season and unusual measurement environment. It is included descriptively but excluded from primary transitions.
- Threshold sensitivity shows the tradeoff between cohort size and retention: average retention declines as the threshold increases from 100 to 300 PA. The 150 PA threshold remains the primary rule.
- Conditional next-season wRC+ distributions are centered near league average across development target seasons.
- Lower-PA hitters have larger conditional-performance errors than high-PA hitters.
- The 2026 scoring frame contains 407 hitters with at least 150 PA. Forty rows have at least one missing Statcast feature; the forecast pipeline imputes these values inside the model pipeline and flags them in the output.

The executed EDA is in [phase2_eda.ipynb](../notebooks/phase2_eda.ipynb).

## Model creation and comparison

The historical feature set uses current-season PA, wRC+, wOBA, ISO, BB%, and K%. The augmented feature set adds xwOBA, wOBA, hard-hit rate, barrel rate, sweet-spot rate, and average exit velocity.

Expanding-window development results:

| Model | Eligibility ROC AUC | Conditional wRC+ MAE |
|---|---:|---:|
| Historical ridge | 0.808 | 18.11 |
| Historical + Statcast ridge | 0.809 | 17.63 |
| Historical + Statcast XGBoost | 0.807 | 17.59 |

The Statcast addition provides a modest conditional-performance improvement, while playing-time eligibility is essentially unchanged. XGBoost does not provide a compelling enough gain to replace the simpler ridge model.

Details are in [phase3_statcast_comparison.md](./phase3_statcast_comparison.md), [phase3_model_diagnostics.md](./phase3_model_diagnostics.md), and [phase3_xgboost_evaluation.md](./phase3_xgboost_evaluation.md).

## 2026 holdout test

The fixed models were evaluated against actual 2026 outcomes only after the development comparisons were complete. The holdout visualization is in [phase3_2026_holdout_visualization.ipynb](../notebooks/phase3_2026_holdout_visualization.ipynb).

| Metric | 2026 result |
|---|---:|
| Eligibility accuracy | 80.2% |
| Eligibility ROC AUC | 0.800 |
| Eligibility Brier score | 0.146 |
| Conditional wRC+ MAE | 17.0 |
| Conditional wRC+ RMSE | 21.4 |

Because 2026 has been inspected, it should not be used for further feature selection or tuning. It is preserved as a one-time final evaluation in the project record.

## 2027 forecasting application

After the holdout evaluation, the selected models were refit through 2026 and applied to 2026 hitters with at least 150 PA to forecast 2027.

- Eligibility model: historical-statistics logistic model.
- Conditional performance model: historical-plus-Statcast ridge model.
- Sensitivity outputs: fixed XGBoost eligibility and wRC+ models.
- Review screen: eligibility probability at least 0.50, predicted wRC+ at least 110, and predicted wRC+ at least 10 points above 2026 wRC+.

The complete ranked output is [player_forecasts_2027.csv](../data/forecasts/player_forecasts_2027.csv). The Dodgers shortlist is [dodgers_forecast_shortlist_2027.csv](../data/forecasts/dodgers_forecast_shortlist_2027.csv). The strict screen produced two non-Dodger model candidates: Ketel Marte and Vladimir Guerrero Jr.

### Dodgers forecast highlights

| Player | 2026 PA | 2026 wRC+ | 2027 predicted wRC+ | Eligibility probability |
|---|---:|---:|---:|---:|
| Shohei Ohtani | 618 | 139.2 | 128.2 | 0.976 |
| Will Smith | 276 | 117.9 | 112.8 | 0.801 |
| Freddie Freeman | 657 | 119.9 | 112.6 | 0.971 |
| Max Muncy | 561 | 128.9 | 112.0 | 0.959 |
| Mookie Betts | 528 | 113.6 | 107.5 | 0.950 |
| Andy Pages | 574 | 122.8 | 106.6 | 0.960 |
| Kyle Tucker | 627 | 100.1 | 105.8 | 0.951 |

These are forecast outputs, not roster projections. The model does not know future trades, injuries, role changes, or lineup construction.

### Manual review of strict-screen candidates

**Ketel Marte — Arizona Diamondbacks, second baseman.** The forecast assigns a 96.0% probability of reaching 150 PA and predicts 113.6 wRC+, about 10.5 points above his 2026 value. MLB’s player page lists him as Arizona’s second baseman and records activation from the 10-day injured list on August 31, 2026 after left-knee inflammation; MLB also reported the preceding restricted-list and knee-inflammation episode. The baseball interpretation is therefore “strong offensive rebound candidate if available,” not an unconditional projection of everyday playing time. His expected role is a starting middle-infield role when healthy, but the 2026 availability and knee context are material risk flags. See [MLB player page](https://www.mlb.com/player/ketel-marte-606466), [return report](https://www.mlb.com/dbacks/news/ketel-marte-return-to-diamondbacks), and [injury report](https://www.mlb.com/news/ketel-marte-left-knee-injury).

**Vladimir Guerrero Jr. — Toronto Blue Jays, first baseman.** The forecast assigns a 93.8% probability of reaching 150 PA and predicts 110.2 wRC+, about 15.6 points above his 2026 value. MLB lists him as Toronto’s first baseman and records activation from the seven-day injured list on August 22, 2026 after a concussion. MLB also reported later lower-back tightness and a return to the lineup at first base. His long-term roster status is unusually secure because the MLB player profile records a 14-year extension through 2039. The baseball interpretation is “performance rebound with strong role security, but recent health needs monitoring.” See [MLB player page](https://www.mlb.com/player/vladimir-guerrero-jr-665489), [concussion IL report](https://www.mlb.com/bluejays/news/vladimir-guerrero-jr-out-of-blue-jays-lineup-vs-yankees-on-august-15-2026), and [back-tightness report](https://www.mlb.com/news/vladimir-guerrero-jr-out-of-lineup-against-orioles-lower-back-tightness).

The model cannot evaluate trade likelihood, clubhouse context, medical prognosis, or actual 2027 playing-time opportunity. Those remain human-review inputs.

## Limitations

- The development sample contains only eight non-2020 transitions and five expanding validation folds.
- 2020 is not exchangeable with ordinary full seasons.
- The 150 PA cutoff creates a conditional target population and does not model the full player population.
- Statcast missingness is not guaranteed to be random; 2026 imputation is flagged, not treated as observed measurement.
- The model uses no age, contract, roster, injury, park, role, or team-context features.
- Predictions are associative and predictive, not causal.
- 2027 outcomes are not yet available, so 2027 forecasts cannot be evaluated here.
- Public-source terms and redistribution restrictions apply to raw data.

## Reproducibility

From the project root:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/acquire_public.py --sources savant --years 2026
.venv\Scripts\python.exe scripts/record_fangraphs_manifest.py
.venv\Scripts\python.exe scripts/build_phase3_dataset.py
.venv\Scripts\python.exe scripts/evaluate_baselines.py
.venv\Scripts\python.exe scripts/evaluate_statcast_models.py
.venv\Scripts\python.exe scripts/diagnose_phase3_models.py
.venv\Scripts\python.exe scripts/evaluate_simple_benchmarks.py
.venv\Scripts\python.exe scripts/evaluate_xgboost.py
.venv\Scripts\python.exe scripts/build_2027_forecasts.py
.venv\Scripts\python.exe -m pytest -q
```

The forecast QA tests are in [test_forecast_pipeline.py](../tests/test_forecast_pipeline.py). The RPubs-ready slide source is [dodgers_phase4_2027.Rmd](../presentation/dodgers_phase4_2027.Rmd) with [dodgers_phase4_2027.css](../presentation/dodgers_phase4_2027.css).
