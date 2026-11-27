"""Data loading for CAMELS-US catchment attributes and sample daily time series.

CAMELS attributes (671 catchments, contiguous USA):
    Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P. (2017). The CAMELS data set:
    catchment attributes and meteorology for large-sample studies. HESS, 21, 5293-5313.
    https://doi.org/10.5194/hess-21-5293-2017
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "camels_us"
ATTR_DIR = DATA_DIR / "attributes"
TS_DIR = DATA_DIR / "sample_timeseries"

ATTRIBUTE_FILES = ["name", "topo", "clim", "hydro", "vege", "soil", "geol"]

CFS_TO_M3S = 0.028316846592


def load_attributes(attr_dir: Path = ATTR_DIR) -> pd.DataFrame:
    """Merge all CAMELS attribute tables into one DataFrame indexed by gauge_id."""
    frames = []
    for name in ATTRIBUTE_FILES:
        df = pd.read_csv(attr_dir / f"camels_{name}.txt", sep=";", dtype={"gauge_id": str})
        df["gauge_id"] = df["gauge_id"].str.zfill(8)
        frames.append(df.set_index("gauge_id"))
    attrs = pd.concat(frames, axis=1)
    attrs["huc_02"] = attrs["huc_02"].astype(str).str.zfill(2)
    for col in ["dom_land_cover", "geol_1st_class", "geol_2nd_class", "gauge_name"]:
        if col in attrs:
            attrs[col] = attrs[col].astype(str).str.strip()
    attrs["log_area"] = np.log10(attrs["area_gages2"])
    return attrs


def load_forcing(gauge_id: str, ts_dir: Path = TS_DIR) -> tuple[pd.DataFrame, float]:
    """Read a Daymet basin-mean forcing file. Returns (daily DataFrame, catchment area in km2)."""
    path = next((ts_dir / "forcing_daymet").glob(f"{gauge_id}_lump_cida_forcing_leap.txt"))
    with open(path) as f:
        f.readline()  # latitude
        f.readline()  # elevation
        area_m2 = float(f.readline())
    df = pd.read_csv(path, sep=r"\s+", skiprows=3)
    df.index = pd.to_datetime(dict(year=df["Year"], month=df["Mnth"], day=df["Day"]))
    df = df.rename(columns={"prcp(mm/day)": "prcp", "tmax(C)": "tmax", "tmin(C)": "tmin",
                            "srad(W/m2)": "srad", "swe(mm)": "swe", "dayl(s)": "dayl", "vp(Pa)": "vp"})
    return df[["prcp", "tmax", "tmin", "srad", "swe", "dayl", "vp"]], area_m2 / 1e6


def load_streamflow(gauge_id: str, area_km2: float, ts_dir: Path = TS_DIR) -> pd.Series:
    """Read USGS daily streamflow (cfs) and convert to specific discharge in mm/day.

    Missing values (-999) become NaN.
    """
    path = ts_dir / "streamflow" / f"{gauge_id}_streamflow_qc.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["gauge_id", "year", "month", "day", "q_cfs", "flag"])
    idx = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    q = df["q_cfs"].where(df["q_cfs"] >= 0).to_numpy()
    q_mm = q * CFS_TO_M3S * 86400.0 / (area_km2 * 1e6) * 1000.0
    return pd.Series(q_mm, index=idx, name="q_mm")


def sample_gauges(ts_dir: Path = TS_DIR) -> list[str]:
    return sorted(p.name.split("_")[0] for p in (ts_dir / "streamflow").glob("*_streamflow_qc.txt"))
