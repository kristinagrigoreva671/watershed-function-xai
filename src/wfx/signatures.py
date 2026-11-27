"""Hydrologic signatures computed from daily streamflow (and precipitation).

Definitions follow Addor et al. (2018, WRR, doi:10.1029/2018WR022606) and the CAMELS
attribute documentation, so the values can be compared with the published CAMELS ones.
All flows are in mm/day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def baseflow_lyne_hollick(q: np.ndarray, alpha: float = 0.925, passes: int = 3,
                          pad: int = 30) -> np.ndarray:
    """Baseflow by the Lyne-Hollick recursive digital filter (Ladson et al., 2013).

    Passes alternate forward/backward; the series is padded by reflecting `pad` days at
    both ends to reduce warm-up effects. Input must not contain NaN.
    """
    q = np.asarray(q, dtype=float)
    if np.isnan(q).any():
        raise ValueError("baseflow filter needs a gap-free series")
    n = len(q)
    pad = min(pad, n - 1)
    b = np.concatenate([q[pad:0:-1], q, q[-2:-pad - 2:-1]])
    for k in range(passes):
        s = b if k % 2 == 0 else b[::-1]
        quick = np.zeros_like(s)
        for t in range(1, len(s)):
            quick[t] = max(0.0, alpha * quick[t - 1] + 0.5 * (1 + alpha) * (s[t] - s[t - 1]))
        bf = np.clip(s - quick, 0.0, s)
        b = bf if k % 2 == 0 else bf[::-1]
    return b[pad:pad + n]


def _segments(q: pd.Series, max_gap: int = 5, min_len: int = 90) -> list[np.ndarray]:
    """Split a daily series into gap-free segments: gaps up to `max_gap` days are linearly
    interpolated, longer gaps split the series; segments shorter than `min_len` are dropped."""
    q = q.asfreq("D") if isinstance(q.index, pd.DatetimeIndex) else q
    filled = q.interpolate(limit=max_gap, limit_area="inside")
    valid = filled.notna().to_numpy()
    values = filled.to_numpy()
    segs, start = [], None
    for i, v in enumerate(np.append(valid, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                segs.append(values[start:i])
            start = None
    return segs


def baseflow_index(q: pd.Series) -> float:
    """Baseflow index; series with gaps are processed segment by segment."""
    segs = _segments(q)
    total = sum(s.sum() for s in segs)
    if not segs or total <= 0:
        return np.nan
    return float(sum(baseflow_lyne_hollick(s).sum() for s in segs) / total)


def seasonal_flow_share(q: pd.Series, months: list[int], min_coverage: float = 0.7) -> float:
    """Ratio of mean flow in `months` to mean annual flow, weighted by season length
    (e.g. months=[5, 6] gives the share of annual runoff passing in May-June).
    Returns NaN if fewer than `min_coverage` of the days in those months are observed."""
    in_season = q.index.month.isin(months)
    if q[in_season].notna().mean() < min_coverage or q.notna().mean() < min_coverage:
        return np.nan
    days = sum(pd.Period(f"2001-{m:02d}").days_in_month for m in months)
    return float(q[in_season].mean() * days / (q.mean() * 365.0))


def cold_region_signatures(q: pd.Series) -> dict[str, float]:
    """Signatures that characterise cold, snow- and frost-dominated regimes."""
    return {
        "winter_flow_ratio": seasonal_flow_share(q, [1, 2, 3]) * 365.0 / 90.0,  # Q_JFM / Q_mean
        "freshet_share": seasonal_flow_share(q, [5, 6]),                          # share of annual Q in May-June
    }


def runoff_ratio(q: pd.Series, p: pd.Series) -> float:
    df = pd.concat([q, p], axis=1).dropna()
    return float(df.iloc[:, 0].mean() / df.iloc[:, 1].mean())


def slope_fdc(q: pd.Series) -> float:
    """Slope of the flow duration curve between 33rd and 66th exceedance percentiles (log space)."""
    qv = q.dropna()
    q33 = np.quantile(qv, 1 - 0.33)  # flow exceeded 33% of the time
    q66 = np.quantile(qv, 1 - 0.66)  # flow exceeded 66% of the time
    if q33 <= 0 or q66 <= 0:
        return np.nan
    return float((np.log(q33) - np.log(q66)) / (0.66 - 0.33))


def _water_year(index: pd.DatetimeIndex) -> np.ndarray:
    return np.where(index.month >= 10, index.year + 1, index.year)


def stream_elasticity(q: pd.Series, p: pd.Series) -> float:
    """Streamflow-precipitation elasticity (Sankarasubramanian et al., 2001), median estimator
    over water years: median[(Q_y - Q_mean)/(P_y - P_mean)] * P_mean/Q_mean."""
    df = pd.concat([q.rename("q"), p.rename("p")], axis=1).dropna()
    wy = df.groupby(_water_year(df.index)).agg(["mean", "count"])
    wy = wy[wy[("q", "count")] >= 330]
    qa, pa = wy[("q", "mean")], wy[("p", "mean")]
    if len(qa) < 2:
        return np.nan
    dp = pa - pa.mean()
    ok = dp.abs() > 1e-9
    ratio = (qa[ok] - qa.mean()) / dp[ok]
    return float(np.median(ratio) * pa.mean() / qa.mean())


def _event_stats(mask: np.ndarray, n_years: float) -> tuple[float, float]:
    """Frequency (days/yr) and mean duration (days) of consecutive True runs."""
    freq = mask.sum() / n_years
    if mask.sum() == 0:
        return 0.0, 0.0
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    durations = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return float(freq), float(durations.mean())


def half_flow_date(q: pd.Series) -> float:
    """Mean day of the water year (from 1 October) on which cumulative flow passes half the annual total."""
    qv = q.dropna()
    days = []
    for _, g in qv.groupby(_water_year(qv.index)):
        if len(g) < 330:
            continue
        c = g.cumsum().to_numpy()
        days.append(int(np.searchsorted(c, c[-1] / 2.0)) + 1)
    return float(np.mean(days)) if days else np.nan


def compute_signatures(q: pd.Series, p: pd.Series) -> dict[str, float]:
    """Compute the CAMELS signature set for one catchment."""
    qv = q.dropna()
    n_years = len(qv) / 365.25
    median, mean = qv.median(), qv.mean()
    high_freq, high_dur = _event_stats((qv > 9 * median).to_numpy(), n_years)
    low_freq, low_dur = _event_stats((qv < 0.2 * mean).to_numpy(), n_years)
    return {
        "q_mean": float(mean),
        "runoff_ratio": runoff_ratio(q, p),
        "baseflow_index": baseflow_index(q),
        "slope_fdc": slope_fdc(q),
        "stream_elas": stream_elasticity(q, p),
        "q5": float(np.quantile(qv, 0.05)),
        "q95": float(np.quantile(qv, 0.95)),
        "high_q_freq": high_freq,
        "high_q_dur": high_dur,
        "low_q_freq": low_freq,
        "low_q_dur": low_dur,
        "zero_q_freq": float((qv == 0).mean()),
        "hfd_mean": half_flow_date(q),
    }
