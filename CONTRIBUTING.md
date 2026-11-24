# Contributing

Thanks for looking. This repository holds two small, self-contained studies:

* **MindForecast** (`mindforecast/`) — panel forecasting of mental-health prevalence.
* **SkyClass** (`skyclass/`) — SDSS object classification across two honest regimes.

Fixes, clearer explanations, and additional baselines are all welcome. The
bar is reproducibility: a change should keep the numbers in either README
reproducible from a clean checkout.

## Setup

Python **3.12** is required (3.14 has no wheels for the ML stack yet).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

or, to get the console scripts and the linter:

```bash
pip install -e ".[dev]"
```

## Running the studies

```bash
python run_all.py                       # MindForecast: backtest + forecast + report
python -m mindforecast.cli forecast KEN --disorder Depressive

cd skyclass && python run_all.py        # SkyClass: both regimes + report
```

Both fetch their dataset with `kagglehub` on first use and cache it under
`~/.cache/kagglehub/`. Neither dataset is committed, and no Kaggle account is
needed for these public datasets.

## Checks

```bash
ruff check .
python -c "import mindforecast, skyclass"
```

CI runs exactly these. The full pipelines are a local step — they download data
and take minutes, so they are not part of CI.

## Guidelines

* Keep the honest-reporting style: if a baseline beats the model, say so.
* Do not commit datasets, model binaries or generated `artifacts/` beyond what
  is already tracked.
* Match the surrounding code: `from __future__ import annotations`, type hints
  on public functions, docstrings that explain *why*.
* Commit messages: `fix: ...`, `feat: ...`, `docs: ...`, with the reason in the
  body when it is not obvious.
* If a change alters a reported metric, update the table in the README and the
  regenerated `artifacts/` in the same commit.

## Reporting a problem

Open an issue with the command you ran, the seed/test size, and the full
traceback. Results depend on the split seed, so include it.
