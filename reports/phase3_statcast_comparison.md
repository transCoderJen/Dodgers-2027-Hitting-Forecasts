# Phase 3 historical versus Statcast model comparison

Both models use identical expanding temporal folds, preprocessing, outcomes, and the 2026 holdout.
The historical model uses current-season PA, wRC+, wOBA, ISO, BB%, and K%. The augmented model adds six current-season Statcast features.

## Development-fold averages

| model                    |   eligibility_roc_auc |   eligibility_brier |   eligibility_accuracy |   performance_mae |   performance_rmse |   eligibility_n |   performance_n |
|:-------------------------|----------------------:|--------------------:|-----------------------:|------------------:|-------------------:|----------------:|----------------:|
| historical               |              0.808493 |            0.141748 |               0.789185 |           18.1104 |            23.256  |           406.8 |             309 |
| historical_plus_statcast |              0.809112 |            0.141636 |               0.789102 |           17.625  |            22.5166 |           406.8 |             309 |

## Augmented minus historical development change

|                      |       change |
|:---------------------|-------------:|
| eligibility_roc_auc  |  0.00061949  |
| eligibility_brier    | -0.000111405 |
| eligibility_accuracy | -8.3081e-05  |
| performance_mae      | -0.485401    |
| performance_rmse     | -0.739373    |
| eligibility_n        |  0           |
| performance_n        |  0           |

## 2026 holdout

| model                    |   eligibility_roc_auc |   eligibility_brier |   eligibility_accuracy |   performance_mae |   performance_rmse |   eligibility_n |   performance_n |
|:-------------------------|----------------------:|--------------------:|-----------------------:|------------------:|-------------------:|----------------:|----------------:|
| historical               |              0.799894 |            0.14558  |               0.801527 |           17.3794 |            22.2357 |             393 |             298 |
| historical_plus_statcast |              0.797951 |            0.146211 |               0.801527 |           17.0081 |            21.3949 |             393 |             298 |

## Augmented minus historical 2026 change

|                      |       change |
|:---------------------|-------------:|
| eligibility_roc_auc  | -0.00194278  |
| eligibility_brier    |  0.000630582 |
| eligibility_accuracy |  0           |
| performance_mae      | -0.371274    |
| performance_rmse     | -0.840861    |
| eligibility_n        |  0           |
| performance_n        |  0           |

Interpretation: lower Brier score, MAE, and RMSE are better; higher ROC AUC and accuracy are better. The 2026 holdout was not used for model selection.
