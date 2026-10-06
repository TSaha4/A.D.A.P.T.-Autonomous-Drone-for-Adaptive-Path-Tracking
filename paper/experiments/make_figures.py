"""Generate the paper's data figures from paper/experiments/results (no hand-made data).

fig_pipeline.pdf     Varanasi: input, current flood mask, obstacle map (current + predicted), planned mission
fig_ablation.pdf     Varanasi/Kanpur routes with and without predicted-flood obstacles
fig_sensitivity.pdf  HSV-margin sensitivity (flood-mask fraction, no-path legs)
fig_geodesy.pdf      resize consistency and flat-earth model error vs. extent
NOTE: panels showing the input map reproduce third-party imagery (see paper/TODO_for_authors.md).
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import cv2  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
R = common.RESULTS
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "pdf.fonttype": 42, "axes.linewidth": 0.5})
COL, TWO = 3.5, 7.16
C_CUR, C_PRED, C_ROUTE, C_FAIL, C_NN = "#1f4e9c", "#f28e2b", "#2ca02c", "#d62728", "#7f7f7f"


def load_run(m):
    rec = json.load(open(os.path.join(R, "e2e", f"{m}.json")))
    arr = np.load(os.path.join(R, "e2e", f"{m}.npz"))
    return rec, arr


def fallback_segments(full_path, drop_idx, factor=5):
    """Consecutive route points that are both stops and > 2 grid cells apart: straight-line fallback legs."""
    s, out = set(drop_idx), []
    for i in range(len(full_path) - 1):
        if i in s and (i + 1) in s and np.hypot(*np.subtract(full_path[i + 1], full_path[i])) > 2 * factor:
            out.append((full_path[i], full_path[i + 1]))
    return out


def draw_route(ax, img, mask, pred, rec, title, show_nn=True):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0.45)
    pred_only = (pred > 0) & (mask == 0)
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask > 0] = matplotlib.colors.to_rgba(C_CUR, 0.55)
    overlay[pred_only] = matplotlib.colors.to_rgba(C_PRED, 0.45)
    ax.imshow(overlay)
    fp = np.array(rec["full_path"])
    ax.plot(fp[:, 0], fp[:, 1], "-", color=C_ROUTE, lw=0.8, label="planned route")
    for a, b in fallback_segments(rec["full_path"], rec["drop_indices"]):
        ax.plot([a[0], b[0]], [a[1], b[1]], "-", color=C_FAIL, lw=1.0)
    if show_nn:
        op = np.array(rec["ordered_points"])
        ax.plot(op[:, 0], op[:, 1], "--", color=C_NN, lw=0.5, label="NN visiting order")
    sp = np.array(rec["safe_points"])
    ax.scatter(sp[:, 0], sp[:, 1], s=6, c="k", zorder=3, label="drop points")
    ax.scatter([rec["home"][0]], [rec["home"][1]], s=30, marker="*", c="yellow", edgecolors="k", lw=0.4, zorder=4,
               label="HOME")
    ax.set_title(title)
    ax.axis("off")


def fig_pipeline():
    rec, arr = load_run("varanasi")
    img, mask, pred = arr["display_image"], arr["mask"], arr["pred_mask"]
    fig, axs = plt.subplots(1, 4, figsize=(TWO, 2.05))
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("(a) input map (display 750x748)")
    axs[1].imshow(mask, cmap="gray_r")
    axs[1].set_title(f"(b) flood mask M ({100 * rec['mask_fraction']:.1f}%)")
    ob = np.full((*mask.shape, 3), 1.0)
    ob[(pred > 0) & (mask == 0)] = matplotlib.colors.to_rgb(C_PRED)
    ob[mask > 0] = matplotlib.colors.to_rgb(C_CUR)
    axs[2].imshow(ob)
    axs[2].set_title(f"(c) obstacles M + predicted ({100 * rec['obstacle_fraction']:.1f}%)")
    draw_route(axs[3], img, mask, pred, rec, "(d) drop points and planned route")
    for a in axs[:3]:
        a.axis("off")
    h, l = axs[3].get_legend_handles_labels()
    h.append(matplotlib.lines.Line2D([], [], color=C_FAIL, lw=1.0))
    l.append("no-path leg (straight fallback)")
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.3)
    fig.savefig(os.path.join(FIG, "fig_pipeline.pdf"), bbox_inches="tight")


def fig_ablation():
    fig, axs = plt.subplots(1, 4, figsize=(TWO, 2.0))
    ab = json.load(open(os.path.join(R, "sweeps", "ablation.json")))
    for j, m in enumerate(["varanasi", "kanpur"]):
        rec, arr = load_run(m)
        img, mask, pred = arr["display_image"], arr["mask"], arr["pred_mask"]
        for k, (tag, label) in enumerate([("nopred", "current only"), ("pred", "current+predicted")]):
            side = json.load(open(os.path.join(R, "sweeps", f"{m}_{tag}.sidecar.json")))
            r = dict(rec, full_path=side["full_path"], drop_indices=side["drop_indices"])
            key = "current-only" if tag == "nopred" else "current+predicted"
            x = ab[m][key]
            draw_route(axs[2 * j + k], img, mask, pred, r,
                       f"({'abcd'[2 * j + k]}) {m.capitalize()}, {label}\nno-path {x['no_path_legs']}; "
                       f"pred.-only {100 * x['route_fraction_in_predicted_only_area']:.1f}%", show_nn=False)
    fig.tight_layout(w_pad=0.2)
    fig.savefig(os.path.join(FIG, "fig_ablation.pdf"), bbox_inches="tight")


def fig_sensitivity():
    rows = list(csv.DictReader(open(os.path.join(R, "sweeps", "sensitivity.csv"))))
    # Hue margin has no effect for red annotations (red-wrap mode replaces the hue range), so the x-axis is the
    # S/V margin at the default hue margin 20; clicks 4/16 are shown at the default S/V margin 30.
    fig, axs = plt.subplots(1, 3, figsize=(TWO, 1.75))
    for j, m in enumerate(common.MAPS):
        ax = axs[j]
        sel = sorted([r for r in rows if r["map"] == m and r["hue_margin"] == "20" and r["n_clicks"] == "8"],
                     key=lambda r: int(r["sv_margin"]))
        ax.plot([int(r["sv_margin"]) for r in sel], [100 * float(r["mask_fraction"]) for r in sel], "s-", ms=3,
                lw=0.8, label="8 clicks")
        for c, mk in (("4", "v"), ("16", "^")):
            r = next(r for r in rows if r["map"] == m and r["hue_margin"] == "20" and r["sv_margin"] == "30"
                     and r["n_clicks"] == c)
            ax.plot([30], [100 * float(r["mask_fraction"])], mk, ms=4, mfc="none", label=f"{c} clicks")
            sel = sel + [r]
        for r in sel:
            ax.annotate(f"{r['regions']}/{r['no_path_legs']}", (int(r["sv_margin"]), 100 * float(r["mask_fraction"])),
                        fontsize=5, xytext=(3, 1), textcoords="offset points")
        ax.set_title(m.capitalize() + " (labels: regions/no-path legs)")
        ax.set_xlabel("S/V margin")
        ax.set_xticks([15, 30, 45])
        ax.set_xlim(10, 52)
        if j == 0:
            ax.set_ylabel("flood-mask fraction (%)")
        ax.grid(lw=0.3, alpha=0.5)
    axs[0].legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(FIG, "fig_sensitivity.pdf"), bbox_inches="tight")


def fig_geodesy():
    g = json.load(open(os.path.join(R, "geodesy", "geodesy_eval.json")))
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.6))
    rs = g["resize"]
    x = [r["metres_per_display_px"] for r in rs]
    axs[0].plot(x, [r["max_drop_dev_from_model_m"] for r in rs], "o", ms=3, label="max drop deviation")
    xx = np.linspace(min(x), max(x), 10)
    axs[0].plot(xx, xx, "--", lw=0.6, color="gray", label="1 display pixel")
    axs[0].set_xlabel("ground size of a display pixel (m)")
    axs[0].set_ylabel("deviation from model (m)")
    axs[0].set_title("(a) resize consistency")
    axs[0].legend(frameon=False)
    axs[0].grid(lw=0.3, alpha=0.5)
    ext = g["model_error_at_extents"]
    names = list(ext)
    axs[1].loglog([ext[n]["half_diagonal_km"] for n in names], [ext[n]["worst_model_error_m"] for n in names], "s-",
                  ms=3, lw=0.6)
    short = {names[0]: "configured\n(2 m/px)", names[1]: "Varanasi", names[2]: "Kanpur", names[3]: "Assam"}
    offs = {names[0]: (4, 2), names[1]: (-30, 4), names[2]: (4, -7), names[3]: (-22, -9)}
    for n in names:
        axs[1].annotate(short[n], (ext[n]["half_diagonal_km"], ext[n]["worst_model_error_m"]), fontsize=5,
                        xytext=offs[n], textcoords="offset points")
    axs[1].set_xlim(0.8, 1000)
    axs[1].set_xlabel("half-diagonal of map (km)")
    axs[1].set_ylabel("model vs WGS84 (m)")
    axs[1].set_title("(b) flat-earth model error")
    axs[1].grid(lw=0.3, alpha=0.5, which="both")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(FIG, "fig_geodesy.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    which = sys.argv[1:] or ["pipeline", "ablation", "sensitivity", "geodesy"]
    for name in which:
        globals()[f"fig_{name}"]()
        print("wrote", name)
