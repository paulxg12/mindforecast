"""Figures and a self-contained HTML report for the SDSS classification study."""

from __future__ import annotations

import base64
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CLASS_COLORS = {"GALAXY": "#6366f1", "STAR": "#f59e0b", "QSO": "#0ea5e9"}


def plot_regime_comparison(results: dict[str, dict], path: Path) -> Path:
    """Grouped bars: macro F1 of every model under both feature regimes."""
    models = list(results["photometric (no redshift)"]["metrics"]["model"])
    regimes = list(results)
    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.2))
    palette = ["#94a3b8", "#2563eb"]
    for offset, (regime, color) in zip((-width / 2, width / 2), zip(regimes, palette)):
        table = results[regime]["metrics"].set_index("model").loc[models]
        bars = ax.bar(x + offset, table["macro_f1"], width, label=regime, color=color)
        for bar, value in zip(bars, table["macro_f1"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("macro F1 (held-out 20%)")
    ax.set_title("Object classification: imaging only vs. imaging + redshift")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_confusion(confusion: np.ndarray, labels: list[str], title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    counts = confusion.astype(int)
    im = ax.imshow(counts, cmap="Blues")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            share = counts[i, j] / max(counts[i].sum(), 1)
            ax.text(
                j, i, f"{counts[i, j]:,}\n{share:.0%}", ha="center", va="center",
                fontsize=8, color="white" if share > 0.5 else "#1e293b",
            )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_redshift_by_class(objects: pd.DataFrame, path: Path) -> Path:
    """Why the spectroscopic regime is easy: stars sit at redshift ~ 0."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bins = np.linspace(-0.02, 1.2, 90)
    for label, color in CLASS_COLORS.items():
        subset = objects.loc[objects["class"] == label, "redshift"]
        ax.hist(subset, bins=bins, histtype="step", linewidth=1.6, color=color,
                label=f"{label} (n={len(subset):,})")
    ax.set_yscale("log")
    ax.set_xlabel("redshift (clipped at 1.2 for readability)")
    ax.set_ylabel("objects (log scale)")
    ax.set_title("Redshift by class: nearly a perfect STAR / extragalactic split")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_class_profile(objects: pd.DataFrame, path: Path) -> Path:
    """Median colour of each class — the physical signal available without spectra."""
    colours = ["u_g", "g_r", "r_i", "i_z"]
    medians = objects.groupby("class")[colours].median().loc[list(CLASS_COLORS)]

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(colours))
    width = 0.26
    for offset, (label, color) in zip((-width, 0, width), CLASS_COLORS.items()):
        ax.bar(x + offset, medians.loc[label].to_numpy(), width, label=label, color=color)
    ax.set_xticks(x, colours)
    ax.set_xlabel("colour index (magnitude difference)")
    ax.set_ylabel("median value")
    ax.set_title("Median colours separate the three populations")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _img_tag(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{path.stem}">'


def _table(df: pd.DataFrame, pct_cols: tuple[str, ...] = ()) -> str:
    def fmt(col: str, value: object) -> str:
        if col in pct_cols and isinstance(value, (int, float)):
            return f"{value:.1%}"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{fmt(c, v)}</td>" for c, v in zip(df.columns, row)) + "</tr>"
        for row in df.itertuples(index=False)
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_html(
    path: Path,
    results: dict[str, dict],
    redshift_only: dict,
    figures: dict[str, Path],
    class_counts: pd.Series,
    n_objects: int,
    test_size: float,
    seed: int,
) -> Path:
    photo = results["photometric (no redshift)"]
    spec = results["spectroscopic (+ redshift)"]

    def best(table: pd.DataFrame) -> float:
        return float(table.loc[table["model"] != "majority (baseline)", "macro_f1"].max())

    photo_best = best(photo["metrics"])
    spec_best = best(spec["metrics"])
    base_f1 = float(photo["metrics"].query("model == 'majority (baseline)'")["macro_f1"].iloc[0])
    gap = (photo_best - base_f1) * 100
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    def regime_section(title: str, result: dict, blurb: str) -> str:
        return f"""
<h2>{title}</h2>
<p>{blurb}</p>
{_table(result["metrics"].copy().round(4))}
<h3>Per-class precision / recall (best model: {result["best_model"]})</h3>
{_table(result["per_class"].copy().round(4))}"""

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>SkyClass - SDSS object classification</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.6 -apple-system, "Segoe UI", Roboto, sans-serif; max-width: 980px;
         margin: 2rem auto; padding: 0 1.25rem; color: #1e293b; background: #fff; }}
  h1 {{ font-size: 1.75rem; margin-bottom: .25rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2.25rem; border-bottom: 2px solid #e2e8f0;
        padding-bottom: .3rem; }}
  h3 {{ font-size: .95rem; margin-top: 1.25rem; color: #334155; }}
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
  .note {{ background: #f8fafc; border-left: 3px solid #6366f1; padding: .75rem 1rem;
           margin: 1.25rem 0; font-size: .9rem; }}
  footer {{ margin-top: 3rem; font-size: .8rem; color: #94a3b8; }}
</style></head><body>
<h1>SkyClass</h1>
<p class="sub">Classifying SDSS objects as GALAXY, STAR or QSO from imaging photometry
and redshift. Dataset <code>fedesoriano/stellar-classification-dataset-sdss17</code>
(100,000 spectra, Sloan Digital Sky Survey DR17).</p>

<div class="cards">
  <div class="card"><span>objects</span><b>{n_objects:,}</b></div>
  <div class="card"><span>photometric macro F1</span><b>{photo_best:.3f}</b></div>
  <div class="card"><span>+redshift macro F1</span><b>{spec_best:.3f}</b></div>
  <div class="card"><span>gain vs majority</span><b>+{gap:.1f} pts</b></div>
</div>

<div class="note">
<b>The point of this project.</b> Redshift alone classifies SDSS objects almost
perfectly ({redshift_only["macro_f1"]:.3f} macro F1 from a single feature), because
stars have redshift ~ 0 and extragalactic objects do not. Reporting only that number
would be misleading. So the pipeline runs two experiments: a
<b>photometric</b> model that sees only what an imaging survey sees, and a
<b>spectroscopic</b> model that also gets redshift. The interesting result is the gap
between them.
</div>

<h2>Dataset</h2>
<p>{n_objects:,} spectra after dropping rows with unobserved bands. Classes:
{", ".join(f"<b>{k}</b> {v:,}" for k, v in class_counts.items())}.
Identifiers (<code>obj_ID</code>, <code>plate</code>, <code>MJD</code>, ...) are
dropped: they describe the observation, not the object. Right ascension is encoded as
sin/cos so the 0/360 wrap does not create a false distance, and four adjacent-band
colours are added.</p>
{_img_tag(figures["class_profile"])}
{_img_tag(figures["redshift"])}

{regime_section(
    "Experiment A - photometric only",
    photo,
    "Five band magnitudes, four colours and sky position. This is what a survey can do "
    "before any follow-up spectrum is taken, so it is the hard, useful problem.",
)}
{_img_tag(figures["confusion_photometric"])}

{regime_section(
    "Experiment B - with redshift",
    spec,
    "Adds the spectroscopic redshift. This is the near-separable version of the task and "
    "sets the ceiling for how much the imaging features contribute.",
)}
{_img_tag(figures["confusion_spectroscopic"])}

<h2>Model comparison</h2>
{_img_tag(figures["regime"])}
<p>All learners beat the majority-class baseline of {base_f1:.3f} macro F1. Tree ensembles
gain the most from redshift, and the logistic model does surprisingly well on colours
alone - the class boundaries in colour space are close to linear, which matches the
astrophysics.</p>

<h2>Reproducing</h2>
<p>Split: stratified {test_size:.0%} held-out, seed {seed}. Hyperparameters are fixed in
<code>skyclass/models.py</code>; nothing is tuned on the test split. Reported metrics are
single-split estimates, not cross-validated ones.</p>

<footer>Generated {stamp} by SkyClass. Public SDSS DR17 spectra via Kaggle; no
proprietary data. Model outputs are a teaching example, not a substitute for
spectroscopic classification.</footer>
</body></html>
"""
    path.write_text(html, encoding="utf-8")
    return path
