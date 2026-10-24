# Phase 1 — Data source feasibility findings (2026-10-22)

**Scope:** Documentary/endpoint feasibility audit. Not a completed population-wide data quality audit.

| Source | Verified through public documentation/site | Actual raw CSV rows successfully acquired in this runtime | Finding |
|---|---|---:|---|
| FanGraphs | wRC+ and PA in season-level batting leaderboards; `pybaseball` documents `batting_stats` | 0 | Retrieval method documented; access and source terms require care |
| Baseball Savant | Custom leaderboard has `Download CSV`, selectable contact metrics, seasons 2015+ | 0 | Promising official CSV interface; quantitative fetch not established here |
| Chadwick Register | 16 `people-*.csv` shards with `key_mlbam` and `key_fangraphs`; ODC-By license | 0 | Crosswalk schema verified through documentation |

The local runtime failed DNS resolution to each data host. The web browsing interface showed the source pages but could not return the raw CSV payloads. Therefore **no real cohort sizes, join rates, missingness statistics, or next-season outcomes can honestly be claimed yet**. The Python scripts have been validated with fictional test fixtures only. This report is intentionally explicit about the evidence boundary.

## Substantive methodological risks identified

1. **Selection bias:** next-season wRC+ is undefined for non-returners, so evaluation needs Part A (≥150 PA probability) plus Part B (wRC+ conditional on that event).
2. **Incomplete FanGraphs exports:** qualified-only exports would silently eliminate low-PA next-season players and invalidate Part A. Source next-year files must include the entire MLB hitter population; absent record implies 0 PA only then.
3. **Identifier mapping:** join on numeric FanGraphs ID → Chadwick MLBAM ID → Savant `player_id`; do not fuzzy-match names. Detect duplicate or conflicting IDs.
4. **Statcast missingness:** metrics needing batted-ball events may be undefined even when a player has 150+ PA. Measure missingness after the join, distinguishing join failure from metric absence.
5. **2015/2020 technology and season effects:** document launch of full Statcast in 2015 and Hawk-Eye in 2020; exclude 2020 transitions from the primary baseline audit.
6. **Holdout contamination:** 2026 is reserved for final evaluation. Do not inspect 2026 labels or calculate 2025→2026 return rates in pilot EDA.
7. **Usage rights:** Chadwick's public register has a stated attribution license. FanGraphs terms limit redistribution; using public leaderboards does not mean unrestricted republication or automated harvesting.

## Go / no-go

**Current decision: Conditional GO for targeted local ingestion, NO-GO for modeling.** The three data families look structurally compatible. Empirical completeness, access compliance, and join rates remain unverified until real CSVs are supplied and audited.

## Sources checked

- FanGraphs: https://www.fangraphs.com/leaders/major-league
- FanGraphs terms: https://www.fangraphs.com/about/terms-of-service
- pybaseball batting docs: https://github.com/jldbc/pybaseball/blob/master/docs/batting_stats.md
- Baseball Savant custom: https://baseballsavant.mlb.com/leaderboard/custom
- Baseball Savant expected stats: https://baseballsavant.mlb.com/leaderboard/expected_statistics
- Chadwick Register: https://github.com/chadwickbureau/register
