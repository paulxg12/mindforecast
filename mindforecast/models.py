"""Model training and honest temporal backtesting.

Every candidate is compared against a naive "tomorrow equals today" baseline,
because prevalence series are smooth and a model that cannot beat persistence
is not useful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .features import complete_rows, model_features, temporal_split

TARGET = "prevalence"


def candidate_models() -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=1.0),
        "hgb": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=0,
        ),
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    denom = np.clip(np.abs(y_true), 1e-9, None)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / denom) * 100),
    }


def backtest(
    featured: pd.DataFrame, cutoff: int = 2014
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """One-step-ahead backtest on years after `cutoff`.

    Returns (metrics by model, per-disorder metrics for the best model, fitted models).
    """
    features = model_features(featured)
    train, test = temporal_split(featured, cutoff)
    train = complete_rows(train, features, TARGET)
    test = complete_rows(test, features, TARGET)
    if test.empty:
        raise ValueError(f"no test rows after {cutoff}")

    y_test = test[TARGET].to_numpy()

    rows = [{"model": "naive (persistence)", **_metrics(y_test, test["lag_1"].to_numpy())}]

    fitted: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, model in candidate_models().items():
        model.fit(train[features], train[TARGET])
        pred = model.predict(test[features])
        fitted[name] = model
        predictions[name] = pred
        rows.append({"model": name, **_metrics(y_test, pred)})

    metrics = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)

    best_name = metrics.iloc[0]["model"]
    best_pred = test["lag_1"].to_numpy() if best_name.startswith("naive") else predictions[best_name]
    per_disorder = (
        test.assign(_pred=best_pred)
        .groupby("disorder")
        .apply(
            lambda g: pd.Series(_metrics(g[TARGET].to_numpy(), g["_pred"].to_numpy())),
            include_groups=False,
        )
        .reset_index()
        .sort_values("mae")
    )

    return metrics, per_disorder, fitted


def fit_final(featured: pd.DataFrame, model_name: str = "ridge") -> tuple[object, dict[str, int]]:
    """Fit the chosen model on all usable history; return it plus disorder->code map."""
    features = model_features(featured)
    data = complete_rows(featured, features, TARGET)
    model = candidate_models()[model_name]
    model.fit(data[features], data[TARGET])
    disorder_codes = (
        data[["disorder", "disorder_code"]].drop_duplicates().set_index("disorder")["disorder_code"].to_dict()
    )
    return model, {str(k): int(v) for k, v in disorder_codes.items()}
