"""Model training and evaluation.

Every learner is scored against a majority-class baseline. The two feature sets
from `features` are evaluated separately so the report can say plainly how much
of the task is solved by redshift alone and how much by pure imaging.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES_PHOTOMETRIC, FEATURES_SPECTROSCOPIC, TARGET, feature_matrix


def candidate_models() -> dict[str, object]:
    """Baseline plus three learners of increasing capacity and cost."""
    return {
        "majority (baseline)": DummyClassifier(strategy="most_frequent"),
        "logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0),
        ),
        "random forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=0,
        ),
        "hist gradient boosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=0,
        ),
    }


def _score(y_true, y_pred, y_proba, labels) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_proba, labels=labels)),
    }


def evaluate(
    featured: pd.DataFrame, features: list[str], test_size: float = 0.2, seed: int = 0
) -> dict[str, object]:
    """Train each candidate on `features` and score it on a held-out stratified split.

    Returns the metrics table, the per-class report for the best non-baseline
    model, that model's confusion matrix, and the fitted models.
    """
    from .features import stratified_split

    train, test = stratified_split(featured, test_size=test_size, seed=seed)
    X_train = feature_matrix(train, features)
    X_test = feature_matrix(test, features)
    y_train = train[TARGET]
    y_test = test[TARGET]
    labels = sorted(y_train.unique())

    rows: list[dict[str, object]] = []
    fitted: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}

    for name, model in candidate_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        fitted[name] = model
        predictions[name] = pred
        rows.append({"model": name, **_score(y_test, pred, proba, labels)})

    metrics = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)

    best_name = next(m for m in metrics["model"] if not m.startswith("majority"))
    y_pred = predictions[best_name]
    order = sorted(set(y_test) | set(y_pred))

    per_class = (
        pd.DataFrame(
            classification_report(
                y_test, y_pred, labels=order, output_dict=True, zero_division=0
            )
        )
        .T
        .loc[order, ["precision", "recall", "f1-score", "support"]]
        .reset_index()
        .rename(columns={"index": "class", "f1-score": "f1"})
    )
    per_class["support"] = per_class["support"].astype(int)

    return {
        "metrics": metrics,
        "per_class": per_class,
        "confusion": confusion_matrix(y_test, y_pred, labels=order),
        "labels": order,
        "best_model": best_name,
        "fitted": fitted,
        "y_test": y_test.reset_index(drop=True),
        "predictions": predictions,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "features": list(features),
    }


def evaluate_both_regimes(
    featured: pd.DataFrame, test_size: float = 0.2, seed: int = 0
) -> dict[str, dict[str, object]]:
    """Run the photometric and spectroscopic experiments side by side."""
    return {
        "photometric (no redshift)": evaluate(
            featured, FEATURES_PHOTOMETRIC, test_size=test_size, seed=seed
        ),
        "spectroscopic (+ redshift)": evaluate(
            featured, FEATURES_SPECTROSCOPIC, test_size=test_size, seed=seed
        ),
    }


def redshift_only_baseline(featured: pd.DataFrame, test_size: float = 0.2, seed: int = 0):
    """A one-feature reference: how far does redshift alone get you?"""
    from .features import stratified_split

    train, test = stratified_split(featured, test_size=test_size, seed=seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[["redshift"]], train[TARGET])
    pred = model.predict(test[["redshift"]])
    proba = model.predict_proba(test[["redshift"]])
    labels = sorted(train[TARGET].unique())
    return {"model": "redshift only (1 feature)", **_score(test[TARGET], pred, proba, labels)}
