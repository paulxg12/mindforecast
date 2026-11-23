"""Loading the SDSS stellar-classification dataset and cleaning its sentinels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

KAGGLE_DATASET = "fedesoriano/stellar-classification-dataset-sdss17"
DATA_FILE = "star_classification.csv"

CLASSES = ["GALAXY", "STAR", "QSO"]
BANDS = ["u", "g", "r", "i", "z"]

# SDSS writes -9999 when a band was not observed. Those magnitudes are not
# measurements, so they must not be learned as real (extremely bright) objects.
SENTINEL = -999.0


def dataset_dir() -> Path:
    """Locate the dataset, downloading it via kagglehub if it is missing."""
    cached = (
        Path.home()
        / ".cache/kagglehub/datasets"
        / KAGGLE_DATASET
        / "versions/1"
    )
    if cached.exists():
        return cached

    import kagglehub  # imported lazily so training works fully offline

    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def load_observations(directory: Path | None = None) -> pd.DataFrame:
    """One row per SDSS spectrum: astrometry, five band magnitudes, redshift, class."""
    directory = directory or dataset_dir()
    df = pd.read_csv(directory / DATA_FILE)

    unknown = set(df["class"]) - set(CLASSES)
    if unknown:
        raise ValueError(f"unexpected classes in dataset: {sorted(unknown)}")

    # Drop the handful of rows with unobserved bands rather than invent values.
    observed = (df[BANDS] > SENTINEL).all(axis=1)
    df = df[observed].copy()

    return df.reset_index(drop=True)
