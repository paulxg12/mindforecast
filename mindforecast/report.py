"""Figures and a self-contained HTML report."""

from __future__ import annotations

import base64
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DISORDER_COLORS = {
    "Anxiety": "#3b82f6",
    "Depressive": "#8b5cf6",
    "Schizophrenia": "#10b981",
    "Bipolar": "#f59e0b",
    "Eating": "#ef4444",
}


def plot_model_comparison(metrics: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = ["#94a3b8" if m.startswith("naive") else "#2563eb" for m in metrics["model"]]
    ax.bar(metrics["model"], metrics["mae"], color=colors)
    for i, (name, mae) in enumerate(zip(metrics["model"], metrics["mae"])):
        ax.text(i, mae, f"{mae:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("MAE (percentage points)")
    ax.set_title("One-step backtest: 2015-2019 holdout")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_spotlights(
    panel: pd.DataFrame, forecast: pd.DataFrame, codes: list[str], disorder: str, path: Path
) -> Path:
    n = len(codes)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), squeeze=False)

    for ax, code in zip(axes.ravel(), codes):
        hist = panel[(panel.code == code) & (panel.disorder == disorder)].sort_values("year")
        fc = forecast[(forecast.code == code) & (forecast.disorder == disorder)].sort_values("year")
        label = hist["entity"].iloc[0] if not hist.empty else code
        color = DISORDER_COLORS.get(disorder, "#2563eb")

        ax.plot(hist.year, hist.prevalence, color="#334155", lw=1.6, label="observed")
        if not fc.empty:
            ax.plot(fc.year, fc.prevalence, color=color, lw=2, ls="--", label="forecast")
        ax.set_title(f"{label} - {disorder}", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        ax.set_ylabel("% of population", fontsize=8)

    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Observed history and recursive forecast to 2030", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_global_trend(panel: pd.DataFrame, forecast: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ends: list[tuple[str, float, float, str]] = []
    for disorder, color in DISORDER_COLORS.items():
        hist = panel[panel.disorder == disorder].groupby("year")["prevalence"].mean()
        fc = forecast[forecast.disorder == disorder].groupby("year")["prevalence"].mean()
        ax.plot(hist.index, hist.values, color=color, lw=1.6)
        ax.plot([hist.index[-1], *fc.index], [hist.values[-1], *fc.values], color=color, lw=1.6, ls="--")
        ends.append((disorder, float(fc.index[-1]), float(fc.values[-1]), color))

    # Nudge labels apart so the low-prevalence disorders stay readable.
    ends.sort(key=lambda item: item[2])
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    min_gap = span * 0.045
    placed: list[float] = []
    for disorder, x, y, color in ends:
        y_label = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y_label)
        ax.annotate(
            disorder, (x, y_label), fontsize=8, color=color,
            xytext=(5, 0), textcoords="offset points", va="center",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.6, "alpha": 0.6}
            if abs(y_label - y) > 1e-9 else None,
        )

    ax.axvline(2019, color="#94a3b8", lw=1, ls=":")
    ax.text(2019.4, ax.get_ylim()[1] * 0.92, "forecast ->", fontsize=8, color="#64748b")
    ax.set_xlabel("year")
    ax.set_ylabel("unweighted mean across 205 countries (%)")
    ax.set_title("Global mean prevalence: history (solid) and forecast (dashed)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def movers(
    panel: pd.DataFrame, forecast: pd.DataFrame, disorder: str, base_year: int = 2019, top: int = 10
) -> pd.DataFrame:
    """Countries with the largest projected relative change from base_year to horizon end."""
    hist = panel[(panel.disorder == disorder) & (panel.year == base_year)]
    total = forecast[forecast.disorder == disorder].groupby("year")["year"].max()
    if total.empty:
        return pd.DataFrame()
    last = forecast[(forecast.disorder == disorder) & (forecast.year == total.max())]

    merged = hist[["code", "entity", "prevalence"]].merge(
        last[["code", "prevalence"]], on="code", suffixes=("_now", "_2030")
    )
    merged = merged[merged["prevalence_now"] > 0.5]  # drop tiny-base noise
    merged["change_pct"] = (merged["prevalence_2030"] / merged["prevalence_now"] - 1) * 100
    merged["change_abs"] = merged["prevalence_2030"] - merged["prevalence_now"]
    return merged.sort_values("change_pct", ascending=False).head(top).reset_index(drop=True)


def _img_tag(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{path.stem}">'


def write_html(
    path: Path,
    figures: list[Path],
    metrics: pd.DataFrame,
    per_disorder: pd.DataFrame,
    movers_table: pd.DataFrame,
    best_model: str,
    n_rows: int,
    n_countries: int,
    cutoff: int,
) -> Path:
    def table(df: pd.DataFrame) -> str:
        def cell(col: str, value: object) -> str:
            if isinstance(value, float):
                return f"{value:+.2f}" if col.startswith("change") else f"{value:.4f}"
            return str(value)

        head = "".join(f"<th>{c}</th>" for c in df.columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{cell(c, v)}</td>" for c, v in zip(df.columns, row)) + "</tr>"
            for row in df.itertuples(index=False)
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    best_mae = metrics.iloc[0]["mae"]
    naive_mae = metrics[metrics.model.str.startswith("naive")].iloc[0]["mae"]
    lift = (1 - best_mae / naive_mae) * 100
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>MindForecast - global mental-health prevalence projections</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.6 -apple-system, "Segoe UI", Roboto, sans-serif; max-width: 980px;
         margin: 2rem auto; padding: 0 1.25rem; color: #1e293b; background: #fff; }}
  h1 {{ font-size: 1.75rem; margin-bottom: .25rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2.25rem; border-bottom: 2px solid #e2e8f0;
        padding-bottom: .3rem; }}
  .sub {{ color: #64748b; margin-top: 0; }}
  .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0; }}
  .card {{ flex: 1 1 150px; border: 1px solid #e2e8f0; border-radius: 10px; padding: .9rem 1rem; }}
  .card b {{ display: block; font-size: 1.5rem; }}
  .card span {{ font-size: .8rem; color: #64748b; text-transform: uppercase;
                letter-spacing: .04em; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .88rem; margin: 1rem 0; }}
  th, td {{ text-align: right; padding: .4rem .6rem; border-bottom: 1px solid #eef2f7; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ background: #f8fafc; font-weight: 600; }}
  img {{ max-width: 100%; height: auto; display: block; margin: 1rem 0;
         border: 1px solid #e2e8f0; border-radius: 8px; }}
  code {{ background: #f1f5f9; padding: .1rem .35rem; border-radius: 4px; font-size: .85em; }}
  footer {{ margin-top: 3rem; font-size: .8rem; color: #94a3b8; }}
</style></head><body>
<h1>MindForecast</h1>
<p class="sub">Global mental-health prevalence: temporal backtest and projections to 2030.
Dataset <code>imtkaggleteam/mental-health</code> (WHO / GBD).</p>

<div class="cards">
  <div class="card"><span>countries</span><b>{n_countries}</b></div>
  <div class="card"><span>modeled rows</span><b>{n_rows:,}</b></div>
  <div class="card"><span>best model</span><b>{best_model}</b></div>
  <div class="card"><span>MAE vs naive</span><b>{lift:+.1f}%</b></div>
</div>

<h2>How it works</h2>
<p>Each (country, disorder) series is turned into lag-1..5 values plus rolling
mean/std and a 5-year trend. One model is trained across all series, with the
disorder passed as a categorical feature. The backtest holds out {cutoff + 1}-2019
and compares against a persistence baseline. The projection then feeds its own
predictions back in, one year at a time, to reach 2030.</p>

<h2>Backtest, {cutoff + 1}-2019</h2>
{table(metrics)}
<p>Persistence is a strong baseline: these series are smooth, so the honest test is
whether a model clears it. Best model improves MAE by <b>{lift:.1f}%</b>.</p>

<h2>Error by disorder (best model)</h2>
{table(per_disorder)}
<p>Depressive and anxiety disorders carry nearly all of the absolute error; the
other three are an order of magnitude smaller because their prevalence is far lower.</p>

<h2>Figures</h2>
{''.join(_img_tag(f) for f in figures)}

<h2>Fastest projected risers to 2030 (anxiety, countries above 0.5%)</h2>
{table(movers_table)}

<footer>Generated {stamp} by MindForecast. Aggregate country-year statistics only;
no individual-level data. Forecasts assume historical trends continue and are not
clinical predictions.</footer>
</body></html>
"""
    path.write_text(html, encoding="utf-8")
    return path
