"""Step 4 - hydrologic functional types: cluster catchments by how they behave, not where they are.

Catchments are clustered on standardised signatures (k-means, k chosen by silhouette score).
Each type is then described by its signature profile and by the landscape it occupies.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from wfx.budyko import add_budyko_columns
from wfx.features import CLUSTER_SIGNATURES, LABELS
from wfx.io import REPO_ROOT, load_attributes
from wfx.plotting import CATEGORICAL, INK2, diverging_cmap, draw_us_states, save, set_style

SEED = 42
DESCRIBE = ["aridity", "frac_snow", "p_seasonality", "slope_mean", "elev_mean",
            "frac_forest", "clay_frac", "geol_permeability"]
SIG_LABELS = {"runoff_ratio": "Runoff ratio", "baseflow_index": "Baseflow index",
              "slope_fdc": "FDC slope", "stream_elas": "Elasticity",
              "hfd_mean": "Half-flow date", "high_q_freq": "High-flow freq.",
              "low_q_freq": "Low-flow freq."}


def name_types(summary: pd.DataFrame) -> dict[int, str]:
    """Attach descriptive names from medians (rule-based so they follow the data, not the run)."""
    names = {t: f"Type {t + 1}" for t in summary.index}
    if len(summary) != 4:
        return names
    snow = summary["frac_snow"].idxmax()
    rest = summary.drop(index=snow)
    dry = rest["baseflow_index"].idxmin()
    gw = rest["baseflow_index"].idxmax()
    humid = [t for t in rest.index if t not in (dry, gw)][0]
    names.update({snow: "Delayed release (snowmelt / wetland)", dry: "Dry, flashy, intermittent",
                  gw: "Groundwater-buffered, stable", humid: "Humid forested, responsive"})
    return names


def main():
    set_style()
    df = add_budyko_columns(load_attributes())
    d = df.dropna(subset=CLUSTER_SIGNATURES).copy()
    Z = d[CLUSTER_SIGNATURES].copy()
    for c in ["high_q_freq", "low_q_freq"]:  # heavily skewed counts
        Z[c] = np.log1p(Z[c])
    Z = StandardScaler().fit_transform(Z)

    scores = {k: silhouette_score(Z, KMeans(k, n_init=20, random_state=SEED).fit_predict(Z))
              for k in range(3, 9)}
    k = max(scores, key=scores.get)
    print("silhouette by k:", {kk: round(v, 3) for kk, v in scores.items()}, "-> k =", k)
    km = KMeans(k, n_init=50, random_state=SEED).fit(Z)
    # order types by baseflow index so labels are stable and interpretable
    centers = pd.DataFrame(km.cluster_centers_, columns=CLUSTER_SIGNATURES)
    order = centers["baseflow_index"].sort_values().index
    relabel = {old: new for new, old in enumerate(order)}
    d["type"] = [relabel[c] for c in km.labels_]
    centers = centers.loc[order].reset_index(drop=True)

    summary = d.groupby("type").agg(n=("type", "size"),
                                    **{c: (c, "median") for c in CLUSTER_SIGNATURES + DESCRIBE},
                                    dominant_land_cover=("dom_land_cover", lambda s: s.mode().iloc[0]))
    names = name_types(summary)
    summary.insert(0, "name", [names[t] for t in summary.index])
    summary.to_csv(REPO_ROOT / "results" / "functional_types_summary.csv")
    d[["gauge_name", "huc_02", "type"]].to_csv(REPO_ROOT / "results" / "functional_types.csv")
    print(summary.round(2).T.to_string())

    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    draw_us_states(ax)
    for t in range(k):
        s = d[d.type == t]
        ax.scatter(s.gauge_lon, s.gauge_lat, s=18, color=CATEGORICAL[t], edgecolor="white", lw=0.5,
                   zorder=3, label=f"{t + 1}. {names[t]} (n = {len(s)})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8.5, markerscale=1.4)
    ax.set_title(f"{k} hydrologic functional types from signatures alone (k-means, silhouette)")

    ax = fig.add_subplot(gs[1])
    m = centers.T.values
    lim = np.abs(m).max()
    im = ax.imshow(m, cmap=diverging_cmap(), vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(len(CLUSTER_SIGNATURES)))
    ax.set_yticklabels([SIG_LABELS[c] for c in CLUSTER_SIGNATURES])
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"Type {t + 1}" for t in range(k)])
    for t in range(k):  # colour key under each column; the text itself stays in ink
        ax.add_patch(plt.Rectangle((t - 0.5, len(CLUSTER_SIGNATURES) - 0.5), 1, 0.12,
                                   color=CATEGORICAL[t], clip_on=False, lw=0))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:+.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(m[i, j]) > 0.65 * lim else "#191919")
    ax.grid(False)
    ax.set_title("Signature profile (cluster centre, z-score)")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, aspect=30)
    save(fig, "fig6_functional_types.png")


if __name__ == "__main__":
    main()
