"""Catchment climate indices from daily basin-average forcing."""
from __future__ import annotations

import numpy as np
import pandas as pd


def extraterrestrial_radiation(lat_deg: float, doy: np.ndarray) -> np.ndarray:
    """Daily extraterrestrial radiation Ra (MJ m-2 day-1), FAO-56 eq. 21. Works in polar night/day."""
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta)
                                          + np.cos(phi) * np.cos(delta) * np.sin(ws))
    return np.clip(ra, 0.0, None)


def hargreaves_pet(tmax: pd.Series, tmin: pd.Series, lat_deg: float) -> pd.Series:
    """Hargreaves-Samani potential evapotranspiration (mm/day)."""
    ra = extraterrestrial_radiation(lat_deg, tmax.index.dayofyear.to_numpy())
    tmean = (tmax + tmin) / 2
    pet = 0.0023 * 0.408 * ra * (tmean + 17.8) * np.sqrt((tmax - tmin).clip(lower=0))
    return pet.clip(lower=0)


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
