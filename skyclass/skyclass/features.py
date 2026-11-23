"""Feature engineering for SDSS objects: sky position, colours and the two regimes.

Two feature sets are defined on purpose:

* ``FEATURES_PHOTOMETRIC`` uses only imaging (position + five bands + colours).
  This is the realistic question a survey asks of objects it has *not*
  spectroscopically followed up.
* ``FEATURES_SPECTROSCOPIC`` adds ``redshift``. Redshift all but determines the
  class, so this regime measures how much the other features actually matter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import BANDS, CLASSES

# Bookkeeping columns (object/plate/fiber identifiers, rerun flags, observation
# dates). They identify an observation, not the physics, and would let a model
# memorise the survey instead of learning it.
ID_COLUMNS = [
    "obj_ID",
    "spec_obj_ID",
    "run_ID",
    "rerun_ID",
    "cam_col",
    "field_ID",
    "plate",
    "MJD",
    "fiber_ID",
]

# Right ascension wraps at 360 degrees, so raw degrees make 359.9 and 0.1 look
# far apart. Encode the sky position on the unit circle instead.
POSITION_FEATURES = ["alpha_sin", "alpha_cos", "delta_sin", "delta_cos"]
COLORS = ["u_g", "g_r", "r_i", "i_z"]

FEATURES_PHOTOMETRIC = POSITION_FEATURES + BANDS + COLORS
FEATURES_SPECTROSCOPIC = FEATURES_PHOTOMETRIC + ["redshift"]

TARGET = "class"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add colour indices and circular sky coordinates; drop identifier columns."""
    out = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns]).copy()

    alpha = np.deg2rad(out["alpha"].to_numpy())
    delta = np.deg2rad(out["delta"].to_numpy())
    out["alpha_sin"] = np.sin(alpha)
    out["alpha_cos"] = np.cos(alpha)
    out["delta_sin"] = np.sin(delta)
    out["delta_cos"] = np.cos(delta)

    # A colour is a magnitude difference between adjacent bands: redder objects
    # have larger values. Colours are also extinction-robust, which is why they
    # carry most of the physical signal.
    for left, right in zip(BANDS, BANDS[1:]):
        out[f"{left}_{right}"] = out[left] - out[right]

    unknown = set(out[TARGET]) - set(CLASSES)
    if unknown:
        raise ValueError(f"unexpected classes: {sorted(unknown)}")
    return out


def feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Select the model columns in a fixed order, rejecting missing ones loudly."""
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"missing features: {missing}")
    return df[features]


def stratified_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by class so the rare QSO/STAR populations appear in both halves."""
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[TARGET]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
