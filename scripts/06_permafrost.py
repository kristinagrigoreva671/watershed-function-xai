"""Step 6 - does permafrost control watershed hydrologic function beyond climate?

Sample: ~1,280 catchments of Northern Eurasia (Abramov & Kurochkina database, 2008-2020),
permafrost extent from HydroATLAS. Four analyses:
  A. where the catchments are and how signatures change along the permafrost gradient;
  B. the confounding problem: is there any climatic overlap between permafrost and
     non-permafrost catchments?
  C. does adding permafrost extent to a climate + landscape model improve prediction in
     spatially withheld regions, and do the residuals of a model without permafrost trend
     with permafrost extent?
  D. SHAP: how the learned permafrost effect depends on air temperature.
"""
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors

from wfx.io import REPO_ROOT
from wfx.modeling import cross_validated_predictions, make_model
from wfx.northern_eurasia import load_table, select_sample
from wfx.plotting import AXIS, CATEGORICAL, INK2, blue_cmap, diverging_cmap, save, set_style

RESULTS = REPO_ROOT / "results"
SEED = 42

CLIMATE = ["p_mean", "pet_mean", "aridity", "t_mean", "frac_snow", "p_warm_share",
           "high_prec_freq", "low_prec_freq"]
LANDSCAPE = ["for_pc_sse", "crp_pc_sse", "pst_pc_sse", "urb_pc_sse", "inu_pc_ult", "lka_pc_use",
             "kar_pc_sse", "cly_pc_sav", "slt_pc_sav", "snd_pc_sav", "gwt_cm_sav", "slp_dg_sav",
             "sgr_dk_sav", "ele_mt_sav", "log_area"]
PERMAFROST = ["prm_pc_sse"]

TARGETS = {
    "baseflow_index": "Baseflow index",
    "winter_flow_ratio": "Winter flow ratio (Q_JFM / Q_mean)",
    "freshet_share": "Freshet share (May-June runoff)",
    "slope_fdc": "FDC slope (flashiness)",
    "hfd_mean": "Half-flow date",
    "runoff_ratio": "Runoff ratio",
}
SHORT = {k: v.split(" (")[0] for k, v in TARGETS.items()}


def draw_eurasia(ax, extent=(18, 165, 39, 72)):
    gj = json.loads((REPO_ROOT / "data" / "aux" / "countries.geo.json").read_text())
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            xs, ys = zip(*poly[0])
            ax.plot(xs, ys, color=AXIS, lw=0.4, zorder=1)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1 / np.cos(np.deg2rad(56)))
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def spatial_blocks(df, n=12):
    xy = np.c_[df.lat, df.lon * np.cos(np.deg2rad(df.lat))]
    return pd.Series(KMeans(n, n_init=20, random_state=SEED).fit_predict(xy), index=df.index)


def binned_median(x, y, edges):
    cats = pd.cut(x, edges)
    g = y.groupby(cats, observed=True)
    mids = [(iv.left + iv.right) / 2 for iv in g.median().index]
    return np.array(mids), g.median().values, g.quantile(0.25).values, g.quantile(0.75).values


def main():
    set_style()
    warnings.simplefilter("ignore", FutureWarning)
    df = select_sample(load_table()).dropna(subset=CLIMATE)
    df["block"] = spatial_blocks(df)
    print(f"sample: {len(df)} catchments; permafrost >= 50%: {(df.prm_pc_sse >= 50).sum()}")

    fig_gradient(df)
    fig_overlap(df)
    skill, oof, shap_store = model_experiments(df)
    fig_models(skill, oof, shap_store)


def fig_gradient(df):
    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1], hspace=0.22, wspace=0.35)
    ax = fig.add_subplot(gs[0, :])
    draw_eurasia(ax)
    order = df.sort_values("prm_pc_sse")
    sc = ax.scatter(order.lon, order.lat, c=order.prm_pc_sse, cmap=blue_cmap(), s=13, vmin=0, vmax=100,
                    edgecolor="#9c9c9c", lw=0.3, zorder=3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.01, aspect=25)
    cb.set_label("Permafrost extent (%)")
    ax.set_title(f"{len(df)} catchments, Northern Eurasia (2008-2020): permafrost extent (HydroATLAS)")

    edges = [-0.1, 0.1, 5, 15, 30, 50, 70, 90, 100]
    for i, key in enumerate(["baseflow_index", "winter_flow_ratio", "freshet_share", "slope_fdc"]):
        a = fig.add_subplot(gs[1, i])
        d = df.dropna(subset=[key])
        a.scatter(d.prm_pc_sse, d[key], c=d.t_mean, cmap=diverging_cmap(), vmin=-10, vmax=10, s=7,
                  alpha=0.65, lw=0)
        x, med, q1, q3 = binned_median(d.prm_pc_sse, d[key], edges)
        a.fill_between(x, q1, q3, color="#191919", alpha=0.12, lw=0)
        a.plot(x, med, color="#191919", lw=2, marker="o", ms=4)
        a.set_xlabel("Permafrost extent (%)")
        a.set_title(SHORT[key], fontsize=9.5)
        if key == "winter_flow_ratio":
            a.set_yscale("log")
    sm = plt.cm.ScalarMappable(cmap=diverging_cmap(), norm=plt.Normalize(-10, 10))
    cb = fig.colorbar(sm, ax=fig.axes[-4:], shrink=0.8, pad=0.01, aspect=20)
    cb.set_label("Mean air temperature (°C)")
    save(fig, "fig7_permafrost_gradient.png")


def fig_overlap(df):
    """Nearest climatic analogue without permafrost for every permafrost catchment."""
    cov = ["t_mean", "p_mean", "aridity", "frac_snow", "p_warm_share"]
    z = (df[cov] - df[cov].mean()) / df[cov].std()
    ctrl = df.prm_pc_sse <= 5
    nn = NearestNeighbors(n_neighbors=1).fit(z[ctrl])
    treated = df.prm_pc_sse >= 25
    dist, _ = nn.kneighbors(z[treated])
    # reference: typical distance between two non-permafrost catchments and their nearest neighbour
    ref, _ = NearestNeighbors(n_neighbors=2).fit(z[ctrl]).kneighbors(z[ctrl])
    out = pd.DataFrame({"prm": df.loc[treated, "prm_pc_sse"], "nn_dist": dist[:, 0]})
    out.to_csv(RESULTS / "permafrost_climate_analogues.csv")
    print(f"climate analogue distance: permafrost>=25% median {np.median(dist):.2f} "
          f"vs non-permafrost neighbours median {np.median(ref[:, 1]):.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    sc = ax.scatter(df.t_mean, df.prm_pc_sse, c=df.p_mean * 365, cmap=blue_cmap(), s=10, lw=0, alpha=0.8)
    ax.set_xlabel("Mean air temperature (°C)")
    ax.set_ylabel("Permafrost extent (%)")
    ax.set_title("Permafrost and cold climate are almost fully confounded")
    fig.colorbar(sc, ax=ax, pad=0.01).set_label("Mean precipitation (mm/yr)")
    ax = axes[1]
    bins = np.linspace(0, max(dist.max(), ref[:, 1].max()), 30)
    ax.hist(ref[:, 1], bins=bins, color=CATEGORICAL[0], alpha=0.75, label="Non-permafrost → nearest non-permafrost",
            density=True)
    ax.hist(dist[:, 0], bins=bins, color=CATEGORICAL[1], alpha=0.75, label="Permafrost ≥25% → nearest non-permafrost",
            density=True)
    ax.set_xlabel("Distance in standardised climate space (5 indices)")
    ax.set_ylabel("Density")
    ax.set_title("Permafrost catchments have no climatic analogues")
    ax.legend(fontsize=8)
    save(fig, "fig8_climate_overlap.png")


def model_experiments(df):
    """Leave-region-out predictions with and without permafrost extent, SHAP for the full model."""
    configs = {"Climate + landscape": CLIMATE + LANDSCAPE,
               "+ permafrost extent": CLIMATE + LANDSCAPE + PERMAFROST}
    rows, oof, shap_store = [], [], {}
    for target in TARGETS:
        d = df.dropna(subset=[target])
        if target == "runoff_ratio":
            d = d[d.water_balance_ok]
        y = d[target]
        if target == "winter_flow_ratio":
            y = np.log10(y)
        preds = {}
        for name, cols in configs.items():
            preds[name] = cross_validated_predictions(d[cols], y, "rf", "spatial", d["block"])
        frame = pd.DataFrame({"target": target, "prm": d.prm_pc_sse, "t_mean": d.t_mean,
                              "y": y, **{f"pred_{k}": v for k, v in preds.items()}})
        frame["sd"] = y.std()
        oof.append(frame)
        for name in configs:
            err = frame.y - frame[f"pred_{name}"]
            for group, mask in PRM_GROUPS.items():
                m = mask(frame.prm)
                rows.append({"target": target, "model": name, "group": group, "n": int(m.sum()),
                             "r2": r2_score(frame.y[m], frame[f"pred_{name}"][m]) if group == "all" else np.nan,
                             "rmse": float(np.sqrt((err[m] ** 2).mean())),
                             "bias_sd": float(err[m].mean() / y.std())})
        model = make_model("rf").fit(d[configs["+ permafrost extent"]], y)
        X = pd.DataFrame(model[0].transform(d[configs["+ permafrost extent"]]),
                         columns=configs["+ permafrost extent"], index=d.index)
        sv = shap.TreeExplainer(model[-1]).shap_values(X, check_additivity=False)
        shap_store[target] = (pd.DataFrame(sv, columns=X.columns, index=X.index), X)
    skill = pd.DataFrame(rows)
    skill.to_csv(RESULTS / "permafrost_model_skill.csv", index=False)
    oof = pd.concat(oof)
    oof.to_csv(RESULTS / "permafrost_oof_predictions.csv")
    print(skill.round(3).to_string(index=False))
    imp = pd.DataFrame({t: s.abs().mean() / s.abs().mean().sum() for t, (s, _) in shap_store.items()})
    imp.to_csv(RESULTS / "permafrost_shap_share.csv")
    print("permafrost share of mean |SHAP|:", imp.loc["prm_pc_sse"].round(3).to_dict())
    return skill, oof, shap_store


PRM_GROUPS = {"all": lambda p: p >= 0, "no permafrost (0%)": lambda p: p == 0,
              "sporadic/discontinuous (1-60%)": lambda p: (p > 0) & (p < 60),
              "extensive/continuous (>=60%)": lambda p: p >= 60}


def fig_models(skill, oof, shap_store):
    fig = plt.figure(figsize=(14, 8.8))
    gs = fig.add_gridspec(2, 3, hspace=0.48, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    targets = list(TARGETS)
    x = np.arange(len(targets))
    groups = ["sporadic/discontinuous (1-60%)", "extensive/continuous (>=60%)"]
    for i, group in enumerate(groups):
        change = []
        for t in targets:
            r = skill.query("target == @t and group == @group").set_index("model").rmse
            change.append(100 * (r["+ permafrost extent"] / r["Climate + landscape"] - 1))
        n = int(skill.query("target == 'baseflow_index' and group == @group").n.iloc[0])
        ax.barh(x + (i - 0.5) * 0.38, change, 0.36, color=CATEGORICAL[i + 2], label=f"{group}, n = {n}")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels([SHORT[t] for t in targets])
    ax.invert_yaxis()
    ax.set_xlabel("RMSE change when permafrost is added (%)")
    ax.set_title("Where permafrost exists, does it help?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), fontsize=8)

    edges = [-0.1, 0.1, 5, 15, 30, 50, 70, 90, 100]
    for k, target in enumerate(["baseflow_index", "winter_flow_ratio"]):
        ax = fig.add_subplot(gs[0, k + 1])
        r = oof[oof.target == target]
        for j, (name, color) in enumerate([("Climate + landscape", CATEGORICAL[0]),
                                           ("+ permafrost extent", CATEGORICAL[1])]):
            res = (r.y - r[f"pred_{name}"]) / r.sd
            if j == 0:
                ax.scatter(r.prm, res, color="#9c9c9c", s=5, lw=0, alpha=0.35)
            xm, med, q1, q3 = binned_median(r.prm, res, edges)
            ax.fill_between(xm, q1, q3, color=color, alpha=0.12, lw=0)
            ax.plot(xm, med, color=color, lw=2, marker="o", ms=4, label=name)
        ax.axhline(0, color=INK2, lw=0.8, ls="--")
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel("Permafrost extent (%)")
        ax.set_ylabel("Leave-region-out residual (SD)")
        ax.set_title(f"Residuals: {SHORT[target]}", fontsize=9.5)
        if k == 0:
            ax.legend(loc="lower left", fontsize=8)

    for k, target in enumerate(["baseflow_index", "winter_flow_ratio", "freshet_share"]):
        ax = fig.add_subplot(gs[1, k])
        sv, X = shap_store[target]
        ax.scatter(X.prm_pc_sse, sv.prm_pc_sse, c=X.t_mean, cmap=diverging_cmap(), vmin=-10, vmax=10, s=8,
                   lw=0, alpha=0.7)
        ax.axhline(0, color=INK2, lw=0.8, ls="--")
        ax.set_xlabel("Permafrost extent (%)")
        ax.set_ylabel("SHAP value of permafrost extent")
        ax.set_title(f"Learned permafrost effect: {SHORT[target]}", fontsize=9.5)
    sm = plt.cm.ScalarMappable(cmap=diverging_cmap(), norm=plt.Normalize(-10, 10))
    fig.colorbar(sm, ax=fig.axes[-3:], shrink=0.85, pad=0.01, aspect=25).set_label("Mean air temperature (°C)")
    save(fig, "fig9_permafrost_models.png")


if __name__ == "__main__":
    main()
