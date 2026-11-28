"""Step 7 - process-based test: does treating permafrost as a confining layer improve a
conceptual model?

For every catchment with >= 10% permafrost in the Part 2 sample, plus 100 randomly drawn
catchments without permafrost, two HBV variants are calibrated (KGE, differential evolution):
    HBV     standard structure
    HBV-PF  percolation to the groundwater store scaled by (1 - permafrost fraction); no extra parameter
Warm-up 2008, calibration 2009-2014, independent evaluation 2015-2020. Forcing: ERA5-Land
precipitation and temperature, Hargreaves PET.

    python scripts/07_hbv_permafrost.py /path/to/Russia_HydroMeteo_Database_v04 [--daily-parquet file]

Writes results/hbv_permafrost.csv (one row per catchment and model). ~25 min on 2 CPU cores.
"""
import argparse
import time

import numpy as np
import pandas as pd

from wfx.climate import hargreaves_pet
from wfx.hbv import PARAMS, calibrate, simulate, kge
from wfx.io import REPO_ROOT
from wfx.northern_eurasia import load_table, read_daily_from_netcdf, select_sample
from wfx.signatures import baseflow_index, cold_region_signatures

SEED = 42
CAL = ("2009-01-01", "2014-12-31")
VAL = ("2015-01-01", "2020-12-31")
OUT = REPO_ROOT / "results" / "hbv_permafrost.csv"


def choose_catchments(table):
    sample = select_sample(table).dropna(subset=["t_mean"])
    pf = sample[sample.prm_pc_sse >= 10]
    ref = sample[sample.prm_pc_sse == 0].sample(100, random_state=SEED)
    return pd.concat([pf, ref])


def signatures_in(q, start, end):
    s = q.loc[start:end]
    out = {"bfi": baseflow_index(s)}
    out.update(cold_region_signatures(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db_dir")
    ap.add_argument("--daily-parquet", default=None)
    args = ap.parse_args()

    table = load_table()
    chosen = choose_catchments(table)
    if args.daily_parquet:
        daily = pd.read_parquet(args.daily_parquet)
        if "index" in daily.columns:
            daily["date"] = daily["date"].fillna(daily["index"])
    else:
        daily = read_daily_from_netcdf(args.db_dir)
    daily = daily[daily.gauge_id.astype(str).isin(chosen.index)]
    print(f"{len(chosen)} catchments ({(chosen.prm_pc_sse >= 10).sum()} with >=10% permafrost)")

    rows, t0 = [], time.time()
    for k, (gid, g) in enumerate(daily.groupby("gauge_id")):
        meta = chosen.loc[str(gid)]
        g = g.set_index(pd.to_datetime(g["date"])).sort_index().asfreq("D")
        p = g["prcp_e5l"].clip(lower=0).interpolate().fillna(0)
        t = ((g["t_max_e5l"] + g["t_min_e5l"]) / 2).interpolate()
        pet = hargreaves_pet(g["t_max_e5l"].interpolate(), g["t_min_e5l"].interpolate(), meta.lat)
        q = g["q_mm_day"].where(g["q_mm_day"] >= 0)
        obs_cal = q.where((q.index >= CAL[0]) & (q.index <= CAL[1]))
        obs_val = q.where((q.index >= VAL[0]) & (q.index <= VAL[1]))
        if obs_cal.notna().sum() < 3 * 300 or obs_val.notna().sum() < 3 * 300:
            continue
        obs_sig = signatures_in(q, *VAL)
        f_pf = float(meta.prm_pc_sse) / 100.0
        for model, f in [("HBV", 0.0), ("HBV-PF", f_pf)]:
            theta, kge_cal = calibrate(p.values, t.values, pet.values, obs_cal.values, f_pf=f, seed=SEED)
            q_sim, q_lz = simulate(p.values, t.values, pet.values, theta, f)
            q_sim = pd.Series(q_sim, index=q.index)
            sim_sig = signatures_in(q_sim.where(q.notna()), *VAL)  # same days as observed
            rows.append({"gauge_id": gid, "model": model, "prm_pc_sse": meta.prm_pc_sse,
                         "t_mean": meta.t_mean, "lat": meta.lat, "lon": meta.lon,
                         "kge_cal": kge_cal, "kge_val": kge(q_sim.values, obs_val.values),
                         **dict(zip(PARAMS, theta)),
                         **{f"obs_{k2}": v for k2, v in obs_sig.items()},
                         **{f"sim_{k2}": v for k2, v in sim_sig.items()}})
        if k % 25 == 0:
            print(f"{k + 1} done, {time.time() - t0:.0f} s", flush=True)
    out = pd.DataFrame(rows)
    out.round(5).to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} rows) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
