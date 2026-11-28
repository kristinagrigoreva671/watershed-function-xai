"""HBV conceptual rainfall-runoff model (daily, lumped; HBV-light structure, Seibert & Vis 2012)
with an optional permafrost-confined variant.

Routines
    snow       degree-day melt and refreezing, liquid water holding capacity
    soil       beta function recharge, evapotranspiration limited below LP*FC
    response   upper zone (two outlets K0 above UZL, K1) and lower zone (K2) fed by percolation PERC
    routing    triangular weighting function of length MAXBAS

Permafrost variant ("HBV-PF")
    Percolation from the upper to the lower (groundwater) zone is scaled by (1 - f_pf), where f_pf
    is the permafrost fraction of the catchment: frozen ground acts as a confining layer, so
    recharge to deep storage is only possible through the unfrozen part (taliks, permafrost-free
    areas). No parameter is added - the hypothesis is tested, not tuned.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

PARAMS = ["TT", "CFMAX", "SFCF", "RFCF", "FC", "LP", "BETA", "PERC", "UZL", "K0", "K1", "K2", "MAXBAS"]
BOUNDS = np.array([
    [-2.5, 2.5],    # TT     threshold temperature (deg C)
    [0.5, 8.0],     # CFMAX  degree-day factor (mm/deg C/day)
    [0.5, 1.6],     # SFCF   snowfall correction
    [0.7, 1.3],     # RFCF   rainfall correction
    [30.0, 600.0],  # FC     field capacity (mm)
    [0.3, 1.0],     # LP     ET reduction threshold (-)
    [1.0, 6.0],     # BETA   shape of recharge function
    [0.0, 6.0],     # PERC   max percolation to lower zone (mm/day)
    [0.0, 100.0],   # UZL    upper-zone threshold (mm)
    [0.05, 0.9],    # K0     fast recession (1/day)
    [0.01, 0.5],    # K1     upper-zone recession (1/day)
    [0.0005, 0.15], # K2     lower-zone recession (1/day)
    [1.0, 7.0],     # MAXBAS routing length (days)
])
CFR, CWH = 0.05, 0.1


@njit(cache=True)
def _tri_cdf(x, m):
    """CDF of a symmetric triangular distribution on [0, m]."""
    h = m / 2.0
    if x <= h:
        return x * x / (m * h)
    return 1.0 - (m - x) * (m - x) / (m * h)


@njit(cache=True)
def _triangular_weights(maxbas):
    n = int(np.ceil(maxbas))
    w = np.zeros(n)
    for i in range(n):
        w[i] = _tri_cdf(min(float(i + 1), maxbas), maxbas) - _tri_cdf(float(i), maxbas)
    return w / w.sum()


@njit(cache=True)
def simulate(p, t, pet, theta, f_pf):
    """Run HBV for one parameter vector. Returns (q_sim, lower-zone outflow) in mm/day."""
    TT, CFMAX, SFCF, RFCF, FC, LP, BETA, PERC, UZL, K0, K1, K2, MAXBAS = theta
    n = len(p)
    snow = 0.0
    liquid = 0.0
    sm = 0.3 * FC
    suz = 0.0
    slz = 10.0
    q_gen = np.zeros(n)
    q_lz = np.zeros(n)
    perc_max = PERC * (1.0 - f_pf)
    for i in range(n):
        # snow routine
        if t[i] < TT:
            snow += p[i] * SFCF
            rain = 0.0
        else:
            rain = p[i] * RFCF
        if t[i] > TT:
            melt = min(CFMAX * (t[i] - TT), snow)
            snow -= melt
            liquid += melt + rain
        else:
            refreeze = min(CFR * CFMAX * (TT - t[i]), liquid)
            snow += refreeze
            liquid -= refreeze
            liquid += rain
        excess = max(liquid - CWH * snow, 0.0)
        liquid -= excess
        inflow = excess
        # soil routine
        recharge = inflow * (min(sm / FC, 1.0) ** BETA)
        sm += inflow - recharge
        et = pet[i] * min(sm / (LP * FC), 1.0)
        et = min(et, sm)
        sm -= et
        if sm > FC:
            recharge += sm - FC
            sm = FC
        # response routine
        suz += recharge
        perc = min(perc_max, suz)
        suz -= perc
        slz += perc
        q0 = K0 * max(suz - UZL, 0.0)
        q1 = K1 * suz
        suz -= q0 + q1
        q2 = K2 * slz
        slz -= q2
        q_gen[i] = q0 + q1 + q2
        q_lz[i] = q2
    w = _triangular_weights(MAXBAS)
    q = np.zeros(n)
    for i in range(n):
        for j in range(len(w)):
            if i + j < n:
                q[i + j] += w[j] * q_gen[i]
    return q, q_lz


@njit(cache=True)
def kge(sim, obs):
    """Kling-Gupta efficiency on days with observations (obs NaN skipped)."""
    n = 0
    so = 0.0
    ss = 0.0
    for i in range(len(obs)):
        if not np.isnan(obs[i]):
            n += 1
            so += obs[i]
            ss += sim[i]
    if n < 30:
        return np.nan
    mo, ms = so / n, ss / n
    vo = vs = cov = 0.0
    for i in range(len(obs)):
        if not np.isnan(obs[i]):
            do, ds = obs[i] - mo, sim[i] - ms
            vo += do * do
            vs += ds * ds
            cov += do * ds
    if vo <= 0 or vs <= 0 or mo <= 0:
        return -10.0
    r = cov / np.sqrt(vo * vs)
    alpha = np.sqrt(vs / vo)
    beta = ms / mo
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


@njit(parallel=True, cache=True)
def _population_loss(pop, p, t, pet, obs, f_pf):
    """pop: (n_params, S). Returns 1 - KGE for each member."""
    s = pop.shape[1]
    out = np.empty(s)
    for k in prange(s):
        q, _ = simulate(p, t, pet, pop[:, k], f_pf)
        out[k] = 1.0 - kge(q, obs)
    return out


def calibrate(p, t, pet, obs, f_pf=0.0, seed=0, maxiter=80, popsize=12):
    """Differential evolution on 1 - KGE. `obs` must be NaN during the warm-up period."""
    from scipy.optimize import differential_evolution

    p, t, pet, obs = (np.ascontiguousarray(x, dtype=np.float64) for x in (p, t, pet, obs))
    res = differential_evolution(
        lambda pop: _population_loss(np.atleast_2d(pop.T).T if pop.ndim == 1 else pop, p, t, pet, obs, f_pf),
        BOUNDS, seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-6, polish=False,
        vectorized=True, updating="deferred")
    return res.x, 1.0 - res.fun
