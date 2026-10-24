# Phase 1 quantitative audit — actual CSV results
Generated UTC: 2026-10-24T02:20:36.454299+00:00

Numbers below come only from real supplied CSV files; they are not estimates.

## Player-season data and join coverage
| Season | FG rows (all PA) | Eligible ≥150 PA | Savant rows | Crosswalk matches | Savant matches | Overall join rate |
|---|---:|---:|---:|---:|---:|---:|
| 2015 | 1348 | 389 | 964 | 389 | 389 | 1.0 |
| 2016 | 1353 | 394 | 969 | 394 | 394 | 1.0 |
| 2017 | 1358 | 391 | 957 | 391 | 391 | 1.0 |
| 2018 | 1379 | 392 | 990 | 392 | 392 | 1.0 |
| 2019 | 1410 | 411 | 990 | 411 | 411 | 1.0 |
| 2020 | 1289 | 203 | 581 | 203 | 203 | 1.0 |
| 2021 | 1508 | 411 | 1049 | 411 | 411 | 1.0 |
| 2022 | 1495 | 417 | 693 | 417 | 417 | 1.0 |
| 2023 | 1457 | 404 | 656 | 404 | 404 | 1.0 |
| 2024 | 1454 | 410 | 651 | 410 | 410 | 1.0 |
| 2025 | 1470 | 393 | 673 | 393 | 393 | 1.0 |

## Playing-time selection (next season; 2026 intentionally excluded)
| Transition | Eligible in t | ≥150 PA in t+1 | Retention | No next-year row |
|---|---:|---:|---:|---:|
| 2015->2016 | 389 | 309 | 0.7943 | 33 |
| 2016->2017 | 394 | 299 | 0.7589 | 45 |
| 2017->2018 | 391 | 306 | 0.7826 | 36 |
| 2018->2019 | 392 | 302 | 0.7704 | 31 |
| 2021->2022 | 411 | 314 | 0.764 | 32 |
| 2022->2023 | 417 | 309 | 0.741 | 37 |
| 2023->2024 | 404 | 315 | 0.7797 | 33 |
| 2024->2025 | 410 | 305 | 0.7439 | 31 |

## Feature missingness by season (eligible merged FG cohort)
| Season | xwoba | woba | hard_hit_percent | barrel_batted_rate | sweet_spot_percent | exit_velocity_avg |
|---|---:|---:|---:|---:|---:|---:|
| 2015 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2016 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2017 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2018 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2019 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2020 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2021 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2022 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2023 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2024 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2025 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

**Interpretation warning:** Missing-feature rates include unmatched Savant player-seasons. Disentangle absent Statcast measurements from ID mismatches before downstream imputation.

**Holdout integrity:** 2026 outcomes were not opened by this audit script.
