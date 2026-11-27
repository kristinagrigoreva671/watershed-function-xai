"""Northern Eurasia extension: per-catchment signatures, climate and landscape attributes
from the Hydro-meteorological database for watersheds across the CIS.

    Abramov, D., Kurochkina, L. (2023). Hydro-meteorological database for watersheds across
    the CIS (v1.3) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8432070 (CC BY 4.0)

The database is not redistributed here (3.5 GB). `build_table` turns it into one row per
catchment; that derived table is shipped in data/northern_eurasia/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .climate import climate_indices
from .io import REPO_ROOT
from .signatures import cold_region_signatures, compute_signatures

NE_DIR = REPO_ROOT / "data" / "northern_eurasia"
TABLE_PATH = NE_DIR / "catchment_table.csv"

DAILY_VARS = ["q_mm_day", "prcp_e5l", "t_max_e5l", "t_min_e5l", "Ep"]

# Precipitation: ERA5-Land. Among the products in the database it gives the most plausible
# water balance for this sample (median Q/P = 0.44 vs 0.65 for MSWEP, which puts Q > P in a
# quarter of the catchments).

# HydroATLAS attributes aggregated to each catchment (see HydroATLAS technical documentation)
HYDROATLAS = {
    "prm_pc_sse": "Permafrost extent (%)",
    "for_pc_sse": "Forest extent (%)",
    "crp_pc_sse": "Cropland extent (%)",
    "pst_pc_sse": "Pasture extent (%)",
    "urb_pc_sse": "Urban extent (%)",
    "inu_pc_ult": "Inundation extent, long-term max (%)",
    "lka_pc_use": "Limnicity (lake area, %)",
    "kar_pc_sse": "Karst area (%)",
    "cly_pc_sav": "Clay fraction (%)",
    "slt_pc_sav": "Silt fraction (%)",
    "snd_pc_sav": "Sand fraction (%)",
    "gwt_cm_sav": "Groundwater table depth (cm)",
    "slp_dg_sav": "Terrain slope",
    "sgr_dk_sav": "Stream gradient",
    "ele_mt_sav": "Mean elevation (m)",
}


def read_daily_from_netcdf(db_dir: Path) -> pd.DataFrame:
    """Read all nc_concat/*.nc files into one long DataFrame (needs xarray + netCDF4)."""
    import xarray as xr

    frames = []
    for f in sorted((Path(db_dir) / "nc_concat").glob("*.nc")):
        with xr.open_dataset(f) as ds:
            df = ds[DAILY_VARS].to_dataframe().reset_index()
        if "date" not in df.columns:  # a few files name the time dimension "index"
            df = df.rename(columns={"index": "date"})
        df["gauge_id"] = f.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_table(db_dir: Path, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per catchment: location, HydroATLAS attributes, climate indices, signatures
    and data-coverage diagnostics."""
    import geopandas as gpd

    db_dir = Path(db_dir)
    static = pd.read_csv(db_dir / "attributes" / "russia_cis_static.csv", dtype={"gauge_id": str})
    gauges = gpd.read_file(db_dir / "geometry" / "russia_cis_gauges.gpkg")
    gauges["gauge_id"] = gauges["gauge_id"].astype(str)
    gauges["lon"], gauges["lat"] = gauges.geometry.x, gauges.geometry.y
    meta = static.merge(gauges[["gauge_id", "name_en", "lon", "lat"]], on="gauge_id").set_index("gauge_id")

    if daily is None:
        daily = read_daily_from_netcdf(db_dir)
    if "index" in daily.columns:  # time dimension named "index" in some source files
        daily = daily.assign(date=daily["date"].fillna(daily["index"]))
    daily = daily.assign(gauge_id=daily["gauge_id"].astype(str), date=pd.to_datetime(daily["date"]))

    rows = []
    for gid, g in daily.groupby("gauge_id"):
        g = g.set_index("date").sort_index().asfreq("D")
        q = g["q_mm_day"].where(g["q_mm_day"] >= 0)
        per_year = q.notna().groupby(q.index.year).sum()
        row = {"gauge_id": gid, "n_years_300d": int((per_year >= 300).sum()),
               "q_coverage": float(q.notna().mean()),
               "winter_coverage": float(q[q.index.month.isin([1, 2, 3])].notna().mean())}
        row.update(climate_indices(g["prcp_e5l"], g["Ep"], g["t_max_e5l"], g["t_min_e5l"]))
        if q.notna().sum() >= 365 * 3:
            sig = compute_signatures(q, g["prcp_e5l"])
            sig.update(cold_region_signatures(q))
            row.update(sig)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("gauge_id")
    out = meta.join(table, how="inner")
    out["log_area"] = np.log10(out["ws_area"])
    out["has_reservoir"] = out["rev_mc_usu"] > 0
    # Q > P over 2008-2020 means the precipitation forcing cannot close the water balance
    # (typically undercatch of mountain/snow precipitation); P-dependent signatures are unreliable there
    out["water_balance_ok"] = out["runoff_ratio"] < 1.0
    return out


def select_sample(table: pd.DataFrame, min_years: int = 8, area=(100, 50_000)) -> pd.DataFrame:
    """Catchments with >= `min_years` years of >= 300 observed days, area within bounds and no reservoir."""
    keep = ((table["n_years_300d"] >= min_years) & table["ws_area"].between(*area)
            & ~table["has_reservoir"] & table["runoff_ratio"].notna())
    return table[keep].copy()


def load_table() -> pd.DataFrame:
    return pd.read_csv(TABLE_PATH, dtype={"gauge_id": str}).set_index("gauge_id")
