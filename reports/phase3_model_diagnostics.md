# Phase 3 model diagnostics

Diagnostics use the same fixed models and temporal splits as the baseline comparison.
Bootstrap intervals describe paired resampling uncertainty in the Statcast-minus-historical metric difference; they do not represent new independent seasons.

## Paired bootstrap intervals

| split        | metric   |   n_bootstraps |   statcast_minus_historical |    ci_lower |    ci_upper |
|:-------------|:---------|---------------:|----------------------------:|------------:|------------:|
| development  | roc_auc  |           1000 |                 0.000777265 | -0.00162832 |  0.00313818 |
| development  | mae      |           1000 |                -0.484158    | -0.735114   | -0.225399   |
| development  | rmse     |           1000 |                -0.739165    | -1.0161     | -0.454898   |
| 2026_holdout | roc_auc  |           1000 |                -0.00186166  | -0.00792668 |  0.00450112 |
| 2026_holdout | mae      |           1000 |                -0.366987    | -1.00597    |  0.306008   |
| 2026_holdout | rmse     |           1000 |                -0.836801    | -1.60266    | -0.00824315 |

## Development eligibility calibration

| model                    | probability_bin   |    n |   mean_predicted |   observed_rate |
|:-------------------------|:------------------|-----:|-----------------:|----------------:|
| historical               | (-0.001, 0.2]     |   11 |         0.176097 |        0.272727 |
| historical               | (0.2, 0.4]        |  141 |         0.322858 |        0.361702 |
| historical               | (0.4, 0.6]        |  327 |         0.502576 |        0.495413 |
| historical               | (0.6, 0.8]        |  501 |         0.708825 |        0.710579 |
| historical               | (0.8, 1.0]        | 1054 |         0.916858 |        0.92315  |
| historical_plus_statcast | (-0.001, 0.2]     |    9 |         0.173333 |        0.222222 |
| historical_plus_statcast | (0.2, 0.4]        |  140 |         0.318951 |        0.371429 |
| historical_plus_statcast | (0.4, 0.6]        |  316 |         0.50308  |        0.490506 |
| historical_plus_statcast | (0.6, 0.8]        |  502 |         0.708523 |        0.697211 |
| historical_plus_statcast | (0.8, 1.0]        | 1067 |         0.917683 |        0.924086 |

## 2026 eligibility calibration

| model                    | probability_bin   |   n |   mean_predicted |   observed_rate |
|:-------------------------|:------------------|----:|-----------------:|----------------:|
| historical               | (-0.001, 0.2]     |   1 |         0.154566 |        0        |
| historical               | (0.2, 0.4]        |  31 |         0.320356 |        0.483871 |
| historical               | (0.4, 0.6]        |  61 |         0.493436 |        0.409836 |
| historical               | (0.6, 0.8]        |  77 |         0.71365  |        0.766234 |
| historical               | (0.8, 1.0]        | 223 |         0.917913 |        0.892377 |
| historical_plus_statcast | (-0.001, 0.2]     |   2 |         0.181932 |        0        |
| historical_plus_statcast | (0.2, 0.4]        |  27 |         0.329613 |        0.444444 |
| historical_plus_statcast | (0.4, 0.6]        |  56 |         0.496576 |        0.428571 |
| historical_plus_statcast | (0.6, 0.8]        |  78 |         0.719152 |        0.75641  |
| historical_plus_statcast | (0.8, 1.0]        | 230 |         0.922469 |        0.882609 |

## Development conditional-performance subgroups

| model                    | pa_group   |   n |     mae |    rmse |
|:-------------------------|:-----------|----:|--------:|--------:|
| historical               | 150-249    | 231 | 20.6769 | 27.4091 |
| historical               | 250-399    | 378 | 18.0971 | 22.6041 |
| historical               | 400+       | 936 | 17.4845 | 22.424  |
| historical_plus_statcast | 150-249    | 231 | 19.3687 | 25.2657 |
| historical_plus_statcast | 250-399    | 378 | 17.7376 | 22.2455 |
| historical_plus_statcast | 400+       | 936 | 17.1467 | 21.9283 |

## 2026 conditional-performance subgroups

| model                    | pa_group   |   n |     mae |    rmse |
|:-------------------------|:-----------|----:|--------:|--------:|
| historical               | 150-249    |  38 | 20.866  | 27.0945 |
| historical               | 250-399    |  65 | 16.6952 | 21.3303 |
| historical               | 400+       | 195 | 16.928  | 21.4678 |
| historical_plus_statcast | 150-249    |  38 | 19.7614 | 24.1898 |
| historical_plus_statcast | 250-399    |  65 | 15.4908 | 19.9927 |
| historical_plus_statcast | 400+       | 195 | 16.9773 | 21.2663 |
