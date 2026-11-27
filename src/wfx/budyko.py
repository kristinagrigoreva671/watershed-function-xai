"""Budyko framework: Fu's (1981) equation and per-catchment estimation of the parameter omega.

    E/P = 1 + phi - (1 + phi**omega)**(1/omega),   phi = PET/P (aridity index)

omega summarises how a catchment partitions precipitation into evaporation and runoff
beyond what climate alone (phi) explains: higher omega -> more evaporation, less runoff.
It is commonly linked to vegetation, soil water storage, snow and seasonality.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

OMEGA_BOUNDS = (1.0 + 1e-6, 20.0)


def fu_evaporative_index(phi, omega):
    phi = np.asarray(phi, dtype=float)
    return 1.0 + phi - (1.0 + phi ** omega) ** (1.0 / omega)


def fit_omega(phi: float, evap_index: float) -> float:
    """Solve Fu's equation for omega given the aridity index and the observed E/P.

    Returns NaN when the catchment lies outside the Budyko limits (E/P <= 0 or above
    min(1, phi)), i.e. where water-balance closure assumptions are violated.
    """
    lo, hi = OMEGA_BOUNDS
    if not np.isfinite(phi) or not np.isfinite(evap_index) or evap_index <= 0:
        return np.nan
    f = lambda w: fu_evaporative_index(phi, w) - evap_index
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        return np.nan
    return float(brentq(f, lo, hi))


def add_budyko_columns(attrs):
    """Add evaporative index (1 - runoff ratio), Fu omega and a flag for points outside the limits."""
    out = attrs.copy()
    out["evap_index"] = 1.0 - out["runoff_ratio"]
    out["fu_omega"] = [fit_omega(a, e) for a, e in zip(out["aridity"], out["evap_index"])]
    out["outside_budyko_limits"] = out["fu_omega"].isna()
    return out
