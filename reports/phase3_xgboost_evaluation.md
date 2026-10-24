# Phase 3 XGBoost evaluation

Fixed, conservative XGBoost models are compared using the same expanding temporal folds and 2026 holdout as the linear models.
No hyperparameters were tuned using 2026.

## Development-fold averages

| model                            |   eligibility_n |   eligibility_roc_auc |   eligibility_brier |   eligibility_accuracy |   performance_n |   performance_mae |   performance_rmse |
|:---------------------------------|----------------:|----------------------:|--------------------:|-----------------------:|----------------:|------------------:|-------------------:|
| historical_plus_statcast_xgboost |           406.8 |              0.806971 |            0.142269 |               0.792996 |             309 |           17.5944 |            22.4504 |
| historical_xgboost               |           406.8 |              0.805349 |            0.142787 |               0.784628 |             309 |           17.9787 |            23.0709 |

## 2026 holdout reference

| split        | transition   | model                            |   eligibility_n |   eligibility_roc_auc |   eligibility_brier |   eligibility_accuracy |   performance_n |   performance_mae |   performance_rmse |
|:-------------|:-------------|:---------------------------------|----------------:|----------------------:|--------------------:|-----------------------:|----------------:|------------------:|-------------------:|
| 2026_holdout | 2025->2026   | historical_xgboost               |             393 |              0.793995 |            0.147212 |               0.783715 |             298 |           17.3281 |            22.3227 |
| 2026_holdout | 2025->2026   | historical_plus_statcast_xgboost |             393 |              0.795832 |            0.146276 |               0.793893 |             298 |           16.9924 |            21.4509 |

## Fixed settings

`n_estimators=200`, `max_depth=2`, `learning_rate=0.03`, `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=0.8`, and `reg_lambda=5`.
