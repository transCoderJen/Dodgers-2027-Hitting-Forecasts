# Phase 3 baseline evaluation

Historical-statistics-only baselines evaluated with expanding temporal validation.
The development folds exclude transitions involving 2020. The 2026 results are a final holdout and were not used during fitting or model selection.

## Development-fold averages

| test_transition   |   train_transitions |   eligibility_n |   eligibility_roc_auc |   eligibility_brier |   eligibility_accuracy |   performance_n |   performance_mae |   performance_rmse |
|:------------------|--------------------:|----------------:|----------------------:|--------------------:|-----------------------:|----------------:|------------------:|-------------------:|
| 2018->2019        |                   3 |             392 |              0.770971 |            0.145477 |               0.788265 |             302 |           18.7699 |            23.7738 |
| 2021->2022        |                   4 |             411 |              0.814203 |            0.139661 |               0.785888 |             314 |           18.9021 |            24.6413 |
| 2022->2023        |                   5 |             417 |              0.815055 |            0.144887 |               0.784173 |             309 |           17.2972 |            22.2895 |
| 2023->2024        |                   6 |             404 |              0.826467 |            0.131122 |               0.819307 |             315 |           18.1191 |            23.4019 |
| 2024->2025        |                   7 |             410 |              0.815769 |            0.147592 |               0.768293 |             305 |           17.4638 |            22.1734 |

## Final 2026 holdout

| holdout    |   eligibility_n |   eligibility_roc_auc |   eligibility_brier |   performance_n |   performance_mae |   performance_rmse |
|:-----------|----------------:|----------------------:|--------------------:|----------------:|------------------:|-------------------:|
| 2025->2026 |             393 |              0.799894 |             0.14558 |             298 |           17.3794 |            22.2357 |

## Model definitions

- Playing-time eligibility: standardized, median-imputed logistic regression predicting next-season PA >= 150.
- Conditional performance: standardized, median-imputed ridge regression predicting next-season wRC+ among hitters who reach PA >= 150.
- Predictors: current-season PA, wRC+, wOBA, ISO, BB%, and K%.
