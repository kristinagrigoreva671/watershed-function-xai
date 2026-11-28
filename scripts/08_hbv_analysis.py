"""Step 8 - analyse the HBV vs HBV-PF experiment (reads results/hbv_permafrost.csv)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from wfx.io import REPO_ROOT
from wfx.plotting import CATEGORICAL, INK2, diverging_cmap, save, set_style

RESULTS = REPO_ROOT / "results"
def main():
    set_style()
    r = pd.read_csv(RESULTS / "hbv_permafrost.csv", dtype={"gauge_id": str})
    r["cls"] = pd.cut(r.prm_pc_sse, [-0.1, 0.1, 30, 60, 100], labels=["0%", "10-30%", "30-60%", "60-100%"])
    w = r.pivot_table(index=["gauge_id", "cls"], columns="model",
                      values=["kge_val", "kge_cal", "obs_bfi", "sim_bfi", "obs_winter_flow_ratio",
                              "sim_winter_flow_ratio"], observed=True).reset_index()
    w.columns = ["_".join(c).strip("_") for c in w.columns]
    w["d_kge"] = w["kge_val_HBV-PF"] - w["kge_val_HBV"]
    for m in ["HBV", "HBV-PF"]:
        w[f"err_bfi_{m}"] = (w[f"sim_bfi_{m}"] - w[f"obs_bfi_{m}"])
        w[f"err_wfr_{m}"] = np.log10(w[f"sim_winter_flow_ratio_{m}"] / w[f"obs_winter_flow_ratio_{m}"])

    rows = []
    for cls, g in w.groupby("cls", observed=True):
        d = g.d_kge.dropna()
        p = wilcoxon(d).pvalue if len(d) > 5 and (d != 0).any() else np.nan
        rows.append({"class": cls, "n": len(g),
                     "kge_val_HBV": g["kge_val_HBV"].median(), "kge_val_HBV-PF": g["kge_val_HBV-PF"].median(),
                     "median_d_kge": d.median(), "share_improved": (d > 0).mean(), "wilcoxon_p": p,
                     "bfi_bias_HBV": g["err_bfi_HBV"].median(), "bfi_bias_HBV-PF": g["err_bfi_HBV-PF"].median(),
                     "winter_log_bias_HBV": g["err_wfr_HBV"].median(),
                     "winter_log_bias_HBV-PF": g["err_wfr_HBV-PF"].median()})
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "hbv_permafrost_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    std = r[r.model == "HBV"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4), gridspec_kw={"hspace": 0.45, "wspace": 0.3})
    classes = list(summary["class"])
    x = np.arange(len(classes))

    ax = axes[0, 0]
    for i, (m, c) in enumerate([("HBV", CATEGORICAL[0]), ("HBV-PF", CATEGORICAL[1])]):
        data = [w.loc[w.cls == k, f"kge_val_{m}"].dropna() for k in classes]
        bp = ax.boxplot(data, positions=x + (i - 0.5) * 0.36, widths=0.3, patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", lw=1.5))
        for b in bp["boxes"]:
            b.set(facecolor=c, edgecolor=c)
        for part in ("whiskers", "caps"):
            for l in bp[part]:
                l.set(color=c)
        ax.plot([], [], color=c, lw=6, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_xlabel("Permafrost extent")
    ax.set_ylabel("KGE, evaluation 2015-2020")
    ax.set_title("Model performance")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper center", ncol=2, fontsize=8)

    ax = axes[0, 1]
    ax.axhline(0, color=INK2, lw=0.8, ls="--")
    m = w.merge(std[["gauge_id", "prm_pc_sse", "t_mean"]], on="gauge_id")
    ax.scatter(m.prm_pc_sse, m.d_kge, c=m.t_mean, cmap=diverging_cmap(), vmin=-10, vmax=10, s=10, lw=0, alpha=0.75)
    med = summary.set_index("class").median_d_kge
    centers = {"0%": 0, "10-30%": 20, "30-60%": 45, "60-100%": 80}
    ax.plot([centers[k] for k in med.index], med.values, color="#191919", lw=2, marker="o", ms=5)
    ax.set_ylim(-0.4, 0.4)
    ax.set_xlabel("Permafrost extent (%)")
    ax.set_ylabel("ΔKGE (HBV-PF − HBV), evaluation")
    ax.set_title("Gain from the permafrost-confined structure")

    ax = axes[0, 2]
    for i, (m_, c) in enumerate([("HBV", CATEGORICAL[0]), ("HBV-PF", CATEGORICAL[1])]):
        data = [w.loc[w.cls == k, f"err_wfr_{m_}"].dropna() for k in classes]
        bp = ax.boxplot(data, positions=x + (i - 0.5) * 0.36, widths=0.3, patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", lw=1.5))
        for b in bp["boxes"]:
            b.set(facecolor=c, edgecolor=c)
        for part in ("whiskers", "caps"):
            for l in bp[part]:
                l.set(color=c)
    ax.axhline(0, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_xlabel("Permafrost extent")
    ax.set_ylabel("log10(simulated / observed)")
    ax.set_title("Winter flow (Jan-Mar) error, evaluation")

    for ax, par, label in zip(axes[1], ["K2", "PERC", "FC"],
                              ["K2, lower-zone recession (1/day)", "PERC, percolation (mm/day)",
                               "FC, soil storage capacity (mm)"]):
        ax.scatter(std.prm_pc_sse, std[par], c=std.t_mean, cmap=diverging_cmap(), vmin=-10, vmax=10, s=10, lw=0,
                   alpha=0.75)
        g = std.groupby(pd.cut(std.prm_pc_sse, [-0.1, 0.1, 30, 60, 100]), observed=True)[par].median()
        ax.plot([0, 20, 45, 80], g.values, color="#191919", lw=2, marker="o", ms=5)
        ax.set_xlabel("Permafrost extent (%)")
        ax.set_title(f"Calibrated {label}", fontsize=9.5)
        if par == "K2":
            ax.set_yscale("log")
    sm = plt.cm.ScalarMappable(cmap=diverging_cmap(), norm=plt.Normalize(-10, 10))
    fig.colorbar(sm, ax=axes[1].tolist(), shrink=0.85, pad=0.01, aspect=25).set_label("Mean air temperature (°C)")
    save(fig, "fig10_hbv_permafrost.png")


if __name__ == "__main__":
    main()
