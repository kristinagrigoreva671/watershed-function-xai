"""Catchment climate indices from daily basin-average forcing."""
from __future__ import annotations

import numpy as np
import pandas as pd


def climate_indices(p: pd.Series, pet: pd.Series, tmax: pd.Series, tmin: pd.Series) -> dict[str, float]:
    """Climate descriptors comparable to the CAMELS set.

    p, pet in mm/day; tmax, tmin in deg C. frac_snow is the share of precipitation that
    falls on days with mean air temperature below 0 deg C.
    """
    df = pd.concat({"p": p, "pet": pet, "t": (tmax + tmin) / 2}, axis=1).dropna()
    p_mean = df.p.mean()
    warm = df.index.month.isin([5, 6, 7, 8, 9])
    years = len(df) / 365.25
    return {
        "p_mean": float(p_mean),
        "pet_mean": float(df.pet.mean()),
        "aridity": float(df.pet.mean() / p_mean),
        "t_mean": float(df.t.mean()),
        "frac_snow": float(df.p[df.t < 0].sum() / df.p.sum()),
        "p_warm_share": float(df.p[warm].sum() / df.p.sum()),
        "high_prec_freq": float((df.p >= 5 * p_mean).sum() / years),
        "low_prec_freq": float((df.p < 1).sum() / years),
        "frost_days": float((df.t < 0).sum() / years),
    }
