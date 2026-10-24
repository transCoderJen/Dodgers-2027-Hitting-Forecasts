# Dodgers 2027 Hitting Forecasts

This project develops a transparent, Statcast-informed forecasting workflow for Dodgers roster and player-development review. It estimates two linked outcomes for hitters:

1. The probability of reaching at least 150 plate appearances in the following season.
2. Conditional next-season wRC+ among hitters who reach that playing-time threshold.

The production application uses 2026 data to forecast 2027 Dodgers performance. The project is predictive decision support—not a transaction recommendation or causal analysis.

## What has been completed

- Acquired and audited FanGraphs batting data for 2015–2026 through permitted exports.
- Acquired season-level Baseball Savant features for 2015–2026.
- Built stable player joins using the Chadwick register.
- Completed exploratory analysis, including the 2020 shortened-season exception and playing-time threshold sensitivity.
- Evaluated historical and Statcast-augmented models with expanding temporal validation.
- Held out 2026 for final evaluation before refitting the selected models through 2026.
- Generated 407 2027 forecasts and a ranked Dodgers shortlist.
- Created automated forecast-pipeline tests, a reproducible analysis notebook, and an RPubs-ready presentation.

## Model choices and evaluation

The frozen production choices are:

- Historical ridge/logistic modeling for playing-time eligibility.
- Historical + Statcast ridge modeling for conditional wRC+.
- XGBoost retained as a nonlinear sensitivity analysis.
- 150 PA used as the primary eligibility threshold; 300 PA is treated as a stricter review sensitivity.

The untouched 2026 holdout produced approximately 80.2% eligibility accuracy, 0.800 eligibility ROC AUC, 0.146 Brier score, 17.0 wRC+ MAE, and 21.4 wRC+ RMSE. The holdout remains separate from model selection and is not changed by the final 2027 refit.

## Main outputs

- [Final portfolio report](reports/final_portfolio_report.md)
- [2027 forecast methodology](reports/phase4_2027_forecasts.md)
- [Complete ranked forecast table](data/forecasts/player_forecasts_2027.csv)
- [Dodgers forecast shortlist](data/forecasts/dodgers_forecast_shortlist_2027.csv)
- [2027 model and predictions notebook](notebooks/dodgers_2027_model_and_predictions.ipynb)
- [2026 holdout visualization notebook](notebooks/phase3_2026_holdout_visualization.ipynb)
- [R Markdown presentation source](presentation/dodgers_phase4_2027.Rmd)
- [Rendered presentation](presentation/dodgers_phase4_2027.html)

## Repository structure

```text
data/          Raw, derived, evaluation, and forecast data
notebooks/     EDA, holdout diagnostics, and 2027 forecast review
presentation/  RPubs-ready R Markdown presentation and CSS
reports/       Audit, model comparison, diagnostics, and forecast reports
scripts/       Acquisition, audit, evaluation, and forecast-building scripts
tests/         Automated data and forecast-pipeline tests
```

## Reproduce the Python workflow

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/build_2027_forecasts.py
pytest -q
```

The forecast-building script writes the production outputs to `data/forecasts/`. The notebook can be opened in VS Code with the Python and Jupyter extensions. Its execution dependency, `nbconvert`, is included in `requirements.txt`.

## Render the presentation

The presentation requires R, the R packages `rmarkdown`, `knitr`, `ggplot2`, and `scales`, plus Pandoc. From the project root, render with:

```powershell
Rscript -e "rmarkdown::render('presentation/dodgers_phase4_2027.Rmd', output_dir='presentation')"
```

## Important limitations

- 2027 outcomes are not yet available, so the application forecasts cannot be scored yet.
- 2020 is not treated as a normal season in primary year-to-year transitions.
- Statcast missingness is non-random; missing rows are retained and flagged in the forecast output.
- The model does not include contracts, injuries, roster moves, defensive value, park effects, or expected lineup role.
- Five expanding temporal development folds provide useful validation but limited evidence for long-run stability.
- Baseball context and organizational judgment remain necessary before acting on any forecast.

## Source and access notes

The project uses FanGraphs batting exports, Baseball Savant public season-level exports, and the Chadwick register. Raw provider exports and source-specific access details are documented in the acquisition and audit reports. The workflow does not bypass access controls or automate restricted FanGraphs access.

Project repository: <https://github.com/transCoderJen/Dodgers-2027-Hitting-Forecasts>
