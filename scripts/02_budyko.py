"""Step 2 - place 671 CAMELS catchments in Budyko space and estimate Fu's omega for each."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, NullFormatter

from wfx.budyko import add_budyko_columns, fu_evaporative_index
from wfx.io import REPO_ROOT, load_attributes
from wfx.plotting import INK2, blue_cmap, draw_us_states, save, set_style


def main():
    set_style()
    df = add_budyko_columns(load_attributes())
    out = REPO_ROOT / "results" / "budyko_omega.csv"
    df[["gauge_name", "huc_02", "aridity", "evap_index", "fu_omega", "outside_budyko_limits"]].to_csv(out)
    n_out = int(df.outside_budyko_limits.sum())
    print(f"{len(df)} catchments, {n_out} outside Budyko limits; "
          f"median omega = {df.fu_omega.median():.2f} (IQR {df.fu_omega.quantile(.25):.2f}-"
          f"{df.fu_omega.quantile(.75):.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    phi = np.linspace(0.01, 5.5, 400)
    ax.plot(phi, np.minimum(phi, 1), color=INK2, lw=1.0, label="Energy / water limit")
    for w, ls in [(1.5, ":"), (2.6, "-"), (5.0, "--")]:
        ax.plot(phi, fu_evaporative_index(phi, w), color="#9c9c9c", lw=1.0, ls=ls, label=f"Fu, ω = {w}")
    sc = ax.scatter(df.aridity, df.evap_index, c=df.frac_snow, cmap=blue_cmap(), s=14,
                    edgecolor="white", lw=0.4, vmin=0, vmax=0.8, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 5.5)
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Aridity index PET/P")
    ax.set_ylabel("Evaporative index E/P = 1 - Q/P")
    ax.set_title("671 CAMELS catchments in Budyko space")
    ax.legend(loc="lower right", fontsize=8)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Fraction of P falling as snow")

    ax = axes[1]
    draw_us_states(ax)
    ok = df[~df.outside_budyko_limits]
    sc = ax.scatter(ok.gauge_lon, ok.gauge_lat, c=ok.fu_omega.clip(upper=6), cmap=blue_cmap(), s=16,
                    edgecolor="white", lw=0.4, vmin=1, vmax=6, zorder=3)
    bad = df[df.outside_budyko_limits]
    ax.scatter(bad.gauge_lon, bad.gauge_lat, marker="x", color="#ec4647", s=14, lw=1.0, zorder=3,
               label=f"Outside Budyko limits (n = {n_out})")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Fu's ω: partitioning not explained by aridity alone")
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01, extend="max", aspect=25)
    cb.set_label("ω (higher = more evaporation, less runoff)")
    save(fig, "fig2_budyko.png")


if __name__ == "__main__":
    main()
