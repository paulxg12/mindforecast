# MindForecast

Forecasting global mental-health prevalence to 2030 from WHO / GBD country-year
statistics.

One model is trained across **205 countries x 5 disorders x 30 years**. The
hard part of this dataset is that prevalence series are smooth, so a naive
"tomorrow equals today" baseline is already strong. Everything here is measured
against that baseline on a held-out time block.

## Results (backtest on 2015-2019, one-step-ahead)

| model | MAE | RMSE | MAPE |
|---|---|---|---|
| **ridge** | **0.0044** | **0.0144** | **0.244%** |
| naive (persistence) | 0.0051 | 0.0171 | 0.322% |
| hgb (gradient boosting) | 0.0123 | 0.0402 | 0.660% |

Ridge cuts MAE by **14.1%** vs persistence.

The interesting negative result: **gradient boosting loses badly**. Trees predict
piecewise-constant values and cannot extrapolate a trend outside the training
range, which is exactly what multi-year forecasting demands. This is why the
pipeline ships a linear model by default and reports hgb rather than hiding it.

Error by disorder for the best model:

| disorder | MAE | MAPE |
|---|---|---|
| Schizophrenia | 0.0006 | 0.21% |
| Eating | 0.0007 | 0.33% |
| Bipolar | 0.0010 | 0.19% |
| Depressive | 0.0094 | 0.25% |
| Anxiety | 0.0103 | 0.24% |

## How it works

1. **Tidy** — the wide GBD tables become a long panel of `(entity, code, year, disorder, prevalence)`. Regions without an ISO code are dropped so only real countries remain.
2. **Features** — lag-1..5 values, 5-year rolling mean/std, and a 5-year trend, per series. Disorder identity is passed as a categorical feature so one model serves all five series.
3. **Backtest** — train on years `<= 2014`, score on 2015-2019 against persistence.
4. **Forecast** — refit on all history, then recurse year by year to 2030, feeding predictions back in as observations.
5. **Report** — figures plus a self-contained `artifacts/report.html`.

## Usage

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

python run_all.py                                  # full pipeline
python -m mindforecast.cli run --cutoff 2012       # custom holdout
python -m mindforecast.cli forecast KEN --disorder Depressive
```

Python **3.12** is required — 3.14 has no matching wheels for the ML stack yet.

The dataset is fetched by `kagglehub` on first use and cached in
`~/.cache/kagglehub/`, never committed here.

## Outputs

```
artifacts/
├── report.html            # self-contained: tables + embedded figures
├── forecast.csv           # 11,275 rows, 2020-2030
├── metrics.csv
├── model_comparison.png
├── global_trend.png
└── spotlights.png
```

## Projected fastest risers (anxiety, countries above 0.5%)

| country | 2019 | 2030 | change |
|---|---|---|---|
| United States | 5.70 | 6.40 | +12.4% |
| Taiwan | 3.67 | 3.76 | +2.6% |
| United Arab Emirates | 4.24 | 4.35 | +2.5% |
| South Korea | 3.22 | 3.29 | +2.0% |
| Oman | 4.61 | 4.70 | +1.9% |

## Limitations

- Aggregate country-year statistics only — there are no individual-level records or diagnosis labels, so this cannot support patient-level classification.
- No population weighting: "global mean" here is an unweighted mean across countries.
- 11-year recursive forecasts assume historical trends continue; error compounds with horizon and is not represented as a confidence interval.
- These are statistical projections, not clinical or policy predictions.

## Data

[`imtkaggleteam/mental-health`](https://www.kaggle.com/datasets/imtkaggleteam/mental-health)
on Kaggle, derived from WHO / Global Burden of Disease study outputs.
