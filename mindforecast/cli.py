"""Command line interface: run the full pipeline or forecast a single country."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_prevalence
from .features import add_features
from .forecast import recursive_forecast
from .models import backtest, fit_final
from .report import (
    movers,
    plot_global_trend,
    plot_model_comparison,
    plot_spotlights,
    write_html,
)

SPOTLIGHT = ["USA", "IND", "BRA", "JPN", "NGA", "DEU"]


def run(cutoff: int = 2014, horizon: int = 11, out_dir: Path = Path("artifacts")) -> dict:
    """Train, backtest, forecast to the horizon, and write figures plus an HTML report."""
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = load_prevalence()
    featured = add_features(panel)
    metrics, per_disorder, _ = backtest(featured, cutoff=cutoff)

    best = metrics.iloc[0]["model"]
    if best.startswith("naive"):
        best = "ridge"  # persistence cannot be fit as a model; fall back to the best learner
    model, disorder_codes = fit_final(featured, best)
    forecast = recursive_forecast(panel, model, disorder_codes, horizon=horizon)

    figures = [
        plot_model_comparison(metrics, out_dir / "model_comparison.png"),
        plot_global_trend(panel, forecast, out_dir / "global_trend.png"),
        plot_spotlights(panel, forecast, SPOTLIGHT, "Anxiety", out_dir / "spotlights.png"),
    ]

    movers_table = movers(panel, forecast, "Anxiety")
    report = write_html(
        out_dir / "report.html",
        figures,
        metrics,
        per_disorder,
        movers_table,
        best_model=best,
        n_rows=int(panel.shape[0]),
        n_countries=int(panel["code"].nunique()),
        cutoff=cutoff,
    )

    forecast.to_csv(out_dir / "forecast.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    return {
        "panel": panel,
        "metrics": metrics,
        "per_disorder": per_disorder,
        "forecast": forecast,
        "best_model": best,
        "report": report,
    }


def _print_summary(result: dict) -> None:
    metrics = result["metrics"]
    print("Backtest (one-step ahead):")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    naive = metrics[metrics.model.str.startswith("naive")].iloc[0]["mae"]
    best = metrics.iloc[0]["mae"]
    print(f"\nBest model: {result['best_model']} | MAE improvement vs naive: {(1 - best / naive) * 100:+.1f}%")
    fc = result["forecast"]
    print(f"Forecast: {len(fc):,} rows, {fc.year.min()}-{fc.year.max()}")
    print(f"Report: {result['report']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mindforecast", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="full pipeline: backtest + forecast + report")
    run_p.add_argument("--cutoff", type=int, default=2014, help="last training year (default 2014)")
    run_p.add_argument("--horizon", type=int, default=11, help="forecast years (default 11)")
    run_p.add_argument("--out-dir", type=Path, default=Path("artifacts"))

    fc_p = sub.add_parser("forecast", help="print a forecast for one country")
    fc_p.add_argument("country", help="ISO-3 code, e.g. KEN")
    fc_p.add_argument("--disorder", default="Anxiety", choices=["Anxiety", "Depressive", "Bipolar", "Schizophrenia", "Eating"])
    fc_p.add_argument("--horizon", type=int, default=11)

    args = parser.parse_args(argv)
    if args.command == "run":
        _print_summary(run(cutoff=args.cutoff, horizon=args.horizon, out_dir=args.out_dir))
        return 0

    panel = load_prevalence()
    model, codes = fit_final(add_features(panel), "ridge")
    table = recursive_forecast(panel, model, codes, horizon=args.horizon)
    rows = table[(table.code == args.country.upper()) & (table.disorder == args.disorder)]
    if rows.empty:
        print(f"No forecast for {args.country} / {args.disorder}. Try an ISO-3 code such as KEN.")
        return 1
    print(rows[["code", "disorder", "year", "prevalence"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
