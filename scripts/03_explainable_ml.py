"""Step 3 - which landscape and climate attributes control each aspect of hydrologic function?

For each signature we
  1. compare three models (ridge on all features, random forest on climate only,
     random forest on all features) under random 10-fold CV and leave-region-out CV;
  2. fit the full random forest and explain it with SHAP (TreeExplainer);
  3. aggregate |SHAP| by attribute group (climate / topography / vegetation / soil / geology).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from wfx.budyko import add_budyko_columns
from wfx.features import CLIMATE_FEATURES, FEATURE_GROUPS, FEATURE_TO_GROUP, FEATURES, LABELS, TARGETS
from wfx.io import REPO_ROOT, load_attributes
from wfx.modeling import evaluate, make_model
from wfx.plotting import CATEGORICAL, GROUP_COLORS, INK2, blue_cmap, diverging_cmap, save, set_style

RESULTS = REPO_ROOT / "results"


def main():
    set_style()
    df = add_budyko_columns(load_attributes())
    configs = {"Ridge (all)": ("ridge", FEATURES),
               "RF (climate only)": ("rf", CLIMATE_FEATURES),
               "RF (all)": ("rf", FEATURES)}

    skill, importances, shap_store = [], {}, {}
    for target in TARGETS:
        d = df.dropna(subset=[target])
        X, y, groups = d[FEATURES], d[target], d["huc_02"]
        res = evaluate(X, y, groups, configs)
        res.insert(0, "target", target)
        skill.append(res)
        print(res.round(3).to_string(index=False))

        model = make_model("rf").fit(X, y)
        imputer, rf = model[0], model[-1]
        Xi = pd.DataFrame(imputer.transform(X), columns=FEATURES, index=X.index)
        sv = shap.TreeExplainer(rf).shap_values(Xi, check_additivity=False)
        shap_store[target] = (sv, Xi)
        importances[target] = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES)

    skill = pd.concat(skill, ignore_index=True)
    skill.to_csv(RESULTS / "model_skill.csv", index=False)
    imp = pd.DataFrame(importances)
    imp.to_csv(RESULTS / "shap_mean_abs.csv")

    plot_skill(skill)
    plot_controls(imp)
    plot_beeswarms(shap_store)


def plot_skill(skill):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    targets = list(TARGETS)
    x = np.arange(len(targets))
    width = 0.26
    for ax, scheme, title in zip(axes, ["random", "spatial"],
                                 ["Random 10-fold CV (gauged neighbours available)",
                                  "Leave-region-out CV (HUC-02; 'ungauged region')"]):
        for i, model in enumerate(skill.model.unique()):
            vals = [skill.query("target == @t and cv == @scheme and model == @model").r2.iloc[0] for t in targets]
            ax.bar(x + (i - 1) * width, vals, width - 0.03, color=CATEGORICAL[i], label=model)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([TARGETS[t].split(" (")[0] for t in targets], rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1)
    axes[0].set_ylabel("Out-of-sample R²")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    save(fig, "fig3_model_skill.png")


def plot_controls(imp):
    # normalise each signature's importances to shares that sum to 1
    share = imp / imp.sum(axis=0)
    order = share.mean(axis=1).sort_values(ascending=False).index[:15]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.55})
    ax = axes[0]
    m = share.loc[order]
    im = ax.imshow(m.values, cmap=blue_cmap(), aspect="auto", vmin=0, vmax=m.values.max())
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{LABELS[f]}  [{FEATURE_TO_GROUP[f][:4]}]" for f in order])
    ax.set_xticks(range(len(m.columns)))
    ax.set_xticklabels([TARGETS[t].split(" (")[0] for t in m.columns], rotation=30, ha="right")
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if v >= 0.08:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.6 * m.values.max() else "#191919")
    ax.set_title("Share of mean |SHAP| per signature (top 15 attributes)")
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, aspect=30, format=lambda v, _: f"{v:.0%}")

    ax = axes[1]
    grp = share.groupby(FEATURE_TO_GROUP).sum().loc[list(FEATURE_GROUPS)]
    left = np.zeros(grp.shape[1])
    ylabels = [TARGETS[t].split(" (")[0] for t in grp.columns]
    for g in grp.index:
        ax.barh(ylabels, grp.loc[g], left=left, color=GROUP_COLORS[g], label=g,
                edgecolor="white", linewidth=1.5)
        left += grp.loc[g].values
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Climate vs. landscape controls")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.25), ncol=5, fontsize=8, handlelength=1)
    ax.grid(axis="y", visible=False)
    save(fig, "fig4_controls.png")


def plot_beeswarms(shap_store):
    picks = ["baseflow_index", "fu_omega"]
    fig = plt.figure(figsize=(13, 5))
    for k, target in enumerate(picks):
        sv, Xi = shap_store[target]
        ax = fig.add_subplot(1, 2, k + 1)
        plt.sca(ax)
        shap.summary_plot(sv, Xi.rename(columns=LABELS), max_display=10, show=False,
                          plot_size=None, cmap=diverging_cmap(), color_bar=(k == 1))
        ax.set_title(TARGETS[target])
        ax.set_xlabel("SHAP value (effect on prediction)")
    fig.tight_layout()
    save(fig, "fig5_shap_beeswarm.png")


if __name__ == "__main__":
    main()
