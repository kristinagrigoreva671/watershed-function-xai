"""Step 5 - build the per-catchment table for Northern Eurasia (2,164 catchments, 2008-2020).

Download Russia_HydroMeteo_Database_v04.zip from https://doi.org/10.5281/zenodo.8432070,
unzip it, then run:

    python scripts/05_build_northern_eurasia_table.py /path/to/Russia_HydroMeteo_Database_v04

Optional: --daily-parquet path to a long table (gauge_id, date, q_mm_day, prcp_e5l,
t_max_e5l, t_min_e5l, Ep) to skip reading the NetCDF files.
Writes data/northern_eurasia/catchment_table.csv (all catchments, with coverage diagnostics).
"""
import argparse
import time
import warnings

import pandas as pd

from wfx.northern_eurasia import NE_DIR, TABLE_PATH, build_table, select_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db_dir")
    ap.add_argument("--daily-parquet", default=None)
    args = ap.parse_args()
    t0 = time.time()
    daily = pd.read_parquet(args.daily_parquet) if args.daily_parquet else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        table = build_table(args.db_dir, daily)
    NE_DIR.mkdir(parents=True, exist_ok=True)
    table.round(5).to_csv(TABLE_PATH)
    sample = select_sample(table)
    print(f"{len(table)} catchments -> {len(sample)} in analysis sample  ({time.time() - t0:.0f} s)")
    bins = pd.cut(sample["prm_pc_sse"], [-1, 0, 10, 50, 90, 100], labels=["0", "<10", "10-50", "50-90", ">90"])
    print(bins.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
