"""Feature engineering for panel time-series: lag, rolling and trend features.

The same feature definition is used for training (vectorised) and for
multi-step forecasting (one row at a time), so both paths must agree on
FEATURES and on the lookback length.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 4, 5)
LOOKBACK = max(LAGS)

BASE_FEATURES = [f"lag_{lag}" for lag in LAGS] + ["roll_mean_5", "roll_std_5", "trend_5"]
# Column order the models are trained on; forecasting must reproduce it exactly.
MODEL_FEATURES = BASE_FEATURES + ["disorder_code"]


def add_features(panel: pd.DataFrame, value_col: str = "prevalence") -> pd.DataFrame:
    """Add lag/rolling features to a long panel, sorted by (code, disorder, year)."""
    df = panel.sort_values(["code", "disorder", "year"]).copy()
    grouped = df.groupby(["code", "disorder"], sort=False)[value_col]

    for lag in LAGS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    shifted = grouped.shift(1)
    df["roll_mean_5"] = shifted.groupby(
        [df["code"], df["disorder"]], sort=False
    ).transform(lambda s: s.rolling(5, min_periods=3).mean())
    df["roll_std_5"] = shifted.groupby(
        [df["code"], df["disorder"]], sort=False
    ).transform(lambda s: s.rolling(5, min_periods=3).std())
    df["trend_5"] = df["lag_1"] - df["lag_5"]

    df["disorder_code"] = pd.Categorical(df["disorder"]).codes
    return df


def model_features(df: pd.DataFrame) -> list[str]:
    """Feature names to feed the model (adds the disorder identity)."""
    return list(MODEL_FEATURES)


def series_features(history: np.ndarray) -> dict[str, float]:
    """Features for the next step of a single series, given its ordered history."""
    window = history[-LOOKBACK:]
    if len(window) < LOOKBACK:
        raise ValueError(f"need at least {LOOKBACK} observations, got {len(window)}")
    return {
        **{f"lag_{lag}": float(window[-lag]) for lag in LAGS},
        "roll_mean_5": float(window.mean()),
        "roll_std_5": float(window.std(ddof=1)),
        "trend_5": float(window[-1] - window[0]),
    }


def temporal_split(
    df: pd.DataFrame, cutoff: int = 2014
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a featured panel into train (year <= cutoff) and test (later years)."""
    train = df[df["year"] <= cutoff]
    test = df[df["year"] > cutoff]
    return train, test


def complete_rows(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    return df.dropna(subset=features + [target])
