"""Shared figure style (static figures for the README)."""
from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt

from .io import REPO_ROOT

INK = "#191919"
INK2 = "#666666"
GRID = "#e7e7e7"
AXIS = "#c2c2c2"
CATEGORICAL = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
BLUE_RAMP = ["#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]
GROUP_COLORS = {"Climate": CATEGORICAL[0], "Topography": CATEGORICAL[1], "Vegetation": CATEGORICAL[5],
                "Soil": CATEGORICAL[3], "Geology": CATEGORICAL[6]}

FIG_DIR = REPO_ROOT / "figures"


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "axes.axisbelow": True,
    })


def blue_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)


def diverging_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list(
        "blue_red", ["#1757ac", "#84b9f3", "#efefef", "#f18a59", "#c22f30"])


def draw_us_states(ax, color=AXIS, lw=0.4):
    """Outline of the contiguous US states (PublicaMundi us-states.json, public domain)."""
    path = REPO_ROOT / "data" / "aux" / "us-states.json"
    gj = json.loads(path.read_text())
    for feat in gj["features"]:
        if feat["properties"]["name"] in ("Alaska", "Hawaii", "Puerto Rico"):
            continue
        geom = feat["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            xs, ys = zip(*poly[0])
            ax.plot(xs, ys, color=color, lw=lw, zorder=1)
    ax.set_xlim(-125.5, -66.5)
    ax.set_ylim(24, 50)
    ax.set_aspect(1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / name, facecolor="white")
    plt.close(fig)
