"""Step 1 - compute hydrologic signatures from raw daily data and compare with CAMELS.

The repository ships 3 years (2000-2002) of daily USGS streamflow and Daymet precipitation
for 4 CAMELS catchments. Published CAMELS signatures cover water years 1990-2009, so the
comparison checks that the implementation is consistent, not that values match exactly.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wfx.io import REPO_ROOT, load_attributes, load_forcing, load_streamflow, sample_gauges
from wfx.plotting import CATEGORICAL, INK2, save, set_style
from wfx.signatures import baseflow_lyne_hollick, compute_signatures

COMPARE = {"runoff_ratio": "Runoff ratio", "baseflow_index": "Baseflow index",
           "slope_fdc": "FDC slope", "q95": "Q95 (mm/d)", "q5": "Q5 (mm/d)",
           "high_q_freq": "High-flow days / yr"}


def main():
    set_style()
    attrs = load_attributes()
    rows, series = [], {}
    for gid in sample_gauges():
        forcing, area = load_forcing(gid)
        q = load_streamflow(gid, area)
        p = forcing["prcp"].reindex(q.index)
        sig = compute_signatures(q, p)
        series[gid] = (q, p)
        for key in COMPARE:
            rows.append({"gauge_id": gid, "signature": key, "computed": sig[key],
                         "camels": attrs.loc[gid, key]})
    table = pd.DataFrame(rows)
    out = REPO_ROOT / "results" / "signature_check.csv"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out, index=False)
    print(table.pivot(index="gauge_id", columns="signature", values=["computed", "camels"]).round(3))

    # Figure: hydrograph with baseflow for one catchment + computed vs. published signatures
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.05, 1], hspace=0.55, wspace=0.6)
    ax = fig.add_subplot(gs[0, :])
    gid = "01022500"
    q, _ = series[gid]
    qq = q.loc["2001"].interpolate()
    bf = baseflow_lyne_hollick(qq.to_numpy())
    ax.fill_between(qq.index, 0, qq.values, color=CATEGORICAL[0], alpha=0.25, lw=0, label="Quickflow")
    ax.fill_between(qq.index, 0, bf, color=CATEGORICAL[0], alpha=0.75, lw=0, label="Baseflow (Lyne-Hollick)")
    ax.plot(qq.index, qq.values, color=CATEGORICAL[0], lw=1.0)
    ax.set_ylabel("Streamflow (mm/day)")
    ax.set_title(f"{attrs.loc[gid, 'gauge_name']} ({gid}), 2001: baseflow separation")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    for i, (key, label) in enumerate(COMPARE.items()):
        a = fig.add_subplot(gs[1, i])
        d = table[table.signature == key]
        lo = min(d.computed.min(), d.camels.min())
        hi = max(d.computed.max(), d.camels.max())
        pad = 0.1 * (hi - lo + 1e-9)
        a.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2, lw=0.8, ls="--")
        for j, (_, r) in enumerate(d.iterrows()):
            a.scatter(r.camels, r.computed, s=46, color=CATEGORICAL[j], edgecolor="white", lw=1.5, zorder=3,
                      label=r.gauge_id if i == 0 else None)
        a.set_title(label, fontsize=9)
        if i == 0:
            a.set_ylabel("Computed, 2000-2002")
        a.tick_params(labelsize=7.5)
        if i == 0:
            handles, labels = a.get_legend_handles_labels()
    fig.text(0.5, 0.035, "Published CAMELS value (water years 1990-2009)", ha="center", color=INK2)
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), title=None)
    save(fig, "fig1_signature_check.png")

    r = np.corrcoef(table.computed, table.camels)[0, 1]
    print(f"Pearson r (all signatures pooled, standardised per signature is in CSV): {r:.3f}")


if __name__ == "__main__":
    main()
