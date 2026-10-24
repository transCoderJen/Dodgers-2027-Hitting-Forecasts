# Phase 3 simple benchmark comparison

Simple benchmarks are prespecified and use shrinkage constant K=300 PA; K was not tuned on the 2026 holdout.
Development-fold averages are the basis for model comparison. The 2026 results are reported separately as a final reference.

## Development-fold averages

| outcome              | benchmark                      |     n |      brier |   accuracy_at_0_5 |      mae |     rmse |
|:---------------------|:-------------------------------|------:|-----------:|------------------:|---------:|---------:|
| conditional_wrc_plus | current_wrc_plus_persistence   | 309   | nan        |        nan        |  21.3613 |  27.4019 |
| conditional_wrc_plus | historical_plus_statcast_ridge | 309   | nan        |        nan        |  17.625  |  22.5166 |
| conditional_wrc_plus | historical_ridge               | 309   | nan        |        nan        |  18.1104 |  23.256  |
| conditional_wrc_plus | pa_weighted_mean_shrinkage     | 309   | nan        |        nan        |  18.384  |  23.5009 |
| eligibility          | historical_plus_statcast_ridge | 406.8 |   0.141636 |          0.789102 | nan      | nan      |
| eligibility          | historical_ridge               | 406.8 |   0.141748 |          0.789185 | nan      | nan      |
| eligibility          | training_base_rate             | 406.8 |   0.182698 |          0.759802 | nan      | nan      |

## 2026 holdout reference

| split        | transition   | outcome              | benchmark                      |   n |      brier |   accuracy_at_0_5 |      mae |     rmse |
|:-------------|:-------------|:---------------------|:-------------------------------|----:|-----------:|------------------:|---------:|---------:|
| 2026_holdout | 2025->2026   | eligibility          | training_base_rate             | 393 |   0.183365 |          0.75827  | nan      | nan      |
| 2026_holdout | 2025->2026   | conditional_wrc_plus | current_wrc_plus_persistence   | 298 | nan        |        nan        |  22.044  |  27.7802 |
| 2026_holdout | 2025->2026   | conditional_wrc_plus | pa_weighted_mean_shrinkage     | 298 | nan        |        nan        |  17.9484 |  22.8048 |
| 2026_holdout | 2025->2026   | eligibility          | historical_ridge               | 393 |   0.14558  |          0.801527 | nan      | nan      |
| 2026_holdout | 2025->2026   | conditional_wrc_plus | historical_ridge               | 298 | nan        |        nan        |  17.3794 |  22.2357 |
| 2026_holdout | 2025->2026   | eligibility          | historical_plus_statcast_ridge | 393 |   0.146211 |          0.801527 | nan      | nan      |
| 2026_holdout | 2025->2026   | conditional_wrc_plus | historical_plus_statcast_ridge | 298 | nan        |        nan        |  17.0081 |  21.3949 |

## Benchmark definitions

- Training base rate: every player receives the training-fold eligibility rate.
- Current wRC+ persistence: next-season wRC+ equals current-season wRC+.
- PA-weighted mean shrinkage: `100 + PA / (PA + 300) * (current_wRC+ - 100)`.
- Ridge rows are the learned historical and historical-plus-Statcast models for direct comparison.
