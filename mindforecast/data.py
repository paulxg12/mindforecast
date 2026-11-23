"""Loading and tidying the WHO/GBD mental-health dataset into panel form."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DISORDERS = ["Schizophrenia", "Depressive", "Anxiety", "Bipolar", "Eating"]

PREVALENCE_FILE = "1- mental-illnesses-prevalence.csv"
BURDEN_FILE = "2- burden-disease-from-each-mental-illness(1).csv"

_PREV_PREFIX = "(share of population) - Sex: Both - Age: Age-standardized"
_DALY_PREFIX = "DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: "


def _short(name: str) -> str:
    """"Depressive disorders" / "Bipolar disorder" -> "Depressive" / "Bipolar"."""
    return name.replace(" disorders", "").replace(" disorder", "").strip()


def dataset_dir() -> Path:
    """Locate the dataset, downloading it via kagglehub if it is missing."""
    override = Path.home() / ".cache/kagglehub/datasets/imtkaggleteam/mental-health/versions/1"
    if override.exists():
        return override

    import kagglehub  # imported lazily so training works fully offline

    return Path(kagglehub.dataset_download("imtkaggleteam/mental-health"))


def _to_panel(df: pd.DataFrame, value_cols: dict[str, str], value_name: str) -> pd.DataFrame:
    """Wide country-year frame -> long (entity, code, year, disorder, value) panel."""
    panel = df.melt(
        id_vars=["Entity", "Code", "Year"],
        value_vars=list(value_cols),
        var_name="disorder",
        value_name=value_name,
    )
    panel["disorder"] = panel["disorder"].map(value_cols)
    panel = panel.rename(columns={"Entity": "entity", "Code": "code", "Year": "year"})
    # Keep real countries only: GBD also ships regions ("World", "Global") without an ISO code.
    panel = panel[panel["code"].notna()]
    panel["code"] = panel["code"].astype(str)
    panel["year"] = panel["year"].astype(int)
    return panel.sort_values(["code", "disorder", "year"]).reset_index(drop=True)


def load_prevalence(directory: Path | None = None) -> pd.DataFrame:
    """Country x year x disorder share of population (percent), 1990-2019."""
    directory = directory or dataset_dir()
    df = pd.read_csv(directory / PREVALENCE_FILE)
    value_cols = {
        col: _short(col.replace(f" {_PREV_PREFIX}", "")) for col in df.columns[3:]
    }
    return _to_panel(df, value_cols, "prevalence")


def load_burden(directory: Path | None = None) -> pd.DataFrame:
    """Country x year x disorder DALY rate (per 100k)."""
    directory = directory or dataset_dir()
    df = pd.read_csv(directory / BURDEN_FILE)
    value_cols = {col: _short(col.replace(_DALY_PREFIX, "")) for col in df.columns[3:]}
    return _to_panel(df, value_cols, "dalys")
