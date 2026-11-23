"""Recursive multi-step forecasting beyond the last observed year."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import MODEL_FEATURES, LOOKBACK, series_features


def recursive_forecast(
    panel: pd.DataFrame,
    model,
    disorder_codes: dict[str, int],
    horizon: int = 11,
    value_col: str = "prevalence",
) -> pd.DataFrame:
    """Forecast each (code, disorder) series forward from its own history.

    Predictions are fed back in as observations, which is what makes the
    horizon multiple steps rather than a fixed one-step extrapolation.
    """
    out: list[dict[str, object]] = []

    for (code, disorder), group in panel.groupby(["code", "disorder"], sort=False):
        group = group.sort_values("year")
        history = group[value_col].to_numpy(dtype=float)
        if len(history) < LOOKBACK:
            continue

        year = int(group["year"].iloc[-1])
        dcode = disorder_codes.get(str(disorder))
        if dcode is None:
            continue

        for step in range(1, horizon + 1):
            feats = series_features(history)
            # A named frame keeps sklearn's feature-order checks happy.
            X = pd.DataFrame([{**feats, "disorder_code": dcode}], columns=MODEL_FEATURES)
            value = float(np.clip(model.predict(X)[0], 0.0, None))
            history = np.append(history, value)
            out.append(
                {
                    "code": code,
                    "disorder": disorder,
                    "year": year + step,
                    "step": step,
                    value_col: value,
                }
            )

    return pd.DataFrame(out)


def summarize(forecast: pd.DataFrame, value_col: str = "prevalence") -> pd.DataFrame:
    """Global mean forecast per disorder and year (population-weighted data is not available)."""
    return (
        forecast.groupby(["disorder", "year"])[value_col]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
