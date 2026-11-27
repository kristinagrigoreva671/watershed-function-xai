"""Predicting hydrologic signatures from catchment attributes, with honest (spatial) validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def make_model(kind: str):
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-3, 3, 25)))
    if kind == "rf":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                                   max_features=0.4, n_jobs=-1, random_state=SEED))
    raise ValueError(kind)


def cross_validated_predictions(X: pd.DataFrame, y: pd.Series, kind: str, scheme: str,
                                groups: pd.Series | None = None) -> np.ndarray:
    """Out-of-fold predictions.

    scheme = "random"  -> 10-fold random CV (interpolation between gauged neighbours)
    scheme = "spatial" -> leave-region-out CV, grouped by 2-digit hydrologic unit (HUC-02);
                          every catchment is predicted by a model that never saw its region,
                          mimicking prediction in ungauged / poorly gauged regions.
    """
    pred = np.full(len(y), np.nan)
    if scheme == "random":
        splits = KFold(n_splits=10, shuffle=True, random_state=SEED).split(X)
    elif scheme == "spatial":
        splits = GroupKFold(n_splits=groups.nunique()).split(X, y, groups)
    else:
        raise ValueError(scheme)
    for train, test in splits:
        model = make_model(kind).fit(X.iloc[train], y.iloc[train])
        pred[test] = model.predict(X.iloc[test])
    return pred


def evaluate(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
             configs: dict[str, tuple[str, list[str]]]) -> pd.DataFrame:
    """R2 for each (config, CV scheme). configs maps name -> (model kind, feature list)."""
    rows = []
    for name, (kind, cols) in configs.items():
        for scheme in ("random", "spatial"):
            pred = cross_validated_predictions(X[cols], y, kind, scheme, groups)
            rows.append({"model": name, "cv": scheme, "r2": r2_score(y, pred),
                         "rmse": float(np.sqrt(np.mean((y - pred) ** 2)))})
    return pd.DataFrame(rows)
