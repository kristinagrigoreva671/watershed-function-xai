"""Feature and target definitions for the explainable-ML analysis."""
from __future__ import annotations

# Landscape and climate descriptors used as predictors. Location (lat/lon) and gauge
# identifiers are deliberately excluded so that models must learn from physical
# attributes rather than from geography, which is what matters for transfer to
# poorly gauged regions.
FEATURE_GROUPS: dict[str, list[str]] = {
    "Climate": ["p_mean", "pet_mean", "aridity", "p_seasonality", "frac_snow",
                "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur"],
    "Topography": ["elev_mean", "slope_mean", "log_area"],
    "Vegetation": ["frac_forest", "lai_max", "lai_diff", "gvf_max", "gvf_diff"],
    "Soil": ["soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity", "soil_conductivity",
             "max_water_content", "sand_frac", "silt_frac", "clay_frac"],
    "Geology": ["carbonate_rocks_frac", "geol_porostiy", "geol_permeability"],
}
FEATURES = [f for cols in FEATURE_GROUPS.values() for f in cols]
FEATURE_TO_GROUP = {f: g for g, cols in FEATURE_GROUPS.items() for f in cols}
CLIMATE_FEATURES = FEATURE_GROUPS["Climate"]

# Hydrologic signatures describing distinct aspects of watershed function.
TARGETS: dict[str, str] = {
    "runoff_ratio": "Runoff ratio (release vs. P)",
    "baseflow_index": "Baseflow index (storage/slow release)",
    "slope_fdc": "FDC slope (flashiness)",
    "stream_elas": "Streamflow elasticity to P",
    "hfd_mean": "Half-flow date (timing)",
    "fu_omega": "Budyko-Fu omega (partitioning)",
}

# Signatures used to define hydrologic functional types (clustering).
CLUSTER_SIGNATURES = ["runoff_ratio", "baseflow_index", "slope_fdc", "stream_elas",
                      "hfd_mean", "high_q_freq", "low_q_freq"]

LABELS = {
    "p_mean": "Mean precipitation", "pet_mean": "Mean PET", "aridity": "Aridity (PET/P)",
    "p_seasonality": "P seasonality", "frac_snow": "Snow fraction of P",
    "high_prec_freq": "Heavy-P frequency", "high_prec_dur": "Heavy-P duration",
    "low_prec_freq": "Dry-day frequency", "low_prec_dur": "Dry-spell duration",
    "elev_mean": "Mean elevation", "slope_mean": "Mean slope", "log_area": "log10 area",
    "frac_forest": "Forest fraction", "lai_max": "LAI max", "lai_diff": "LAI seasonal range",
    "gvf_max": "GVF max", "gvf_diff": "GVF seasonal range",
    "soil_depth_pelletier": "Depth to bedrock", "soil_depth_statsgo": "Soil depth (STATSGO)",
    "soil_porosity": "Soil porosity", "soil_conductivity": "Soil conductivity",
    "max_water_content": "Soil water capacity", "sand_frac": "Sand fraction",
    "silt_frac": "Silt fraction", "clay_frac": "Clay fraction",
    "carbonate_rocks_frac": "Carbonate rock fraction", "geol_porostiy": "Subsurface porosity",
    "geol_permeability": "Subsurface permeability",
}
