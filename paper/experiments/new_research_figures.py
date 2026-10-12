"""Generate compact journal figures from recorded new research results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "paper/experiments/results"
FIGURES = ROOT / "paper/figures"

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def ground_truth():
    benchmark = json.loads((RESULTS / "sen1floods11/test_metrics.json").read_text())
    methods = benchmark["methods"]
    labels = ["S1 Otsu", "S1 $k$-means", "S2 NDWI"]
    keys = ["S1_Otsu_VH", "S1_kmeans_VH", "S2_NDWI_0"]
    micros = [methods[k]["micro"]["iou"] for k in keys]
    macros = [methods[k]["macro"]["iou"] for k in keys]
    unet_path = RESULTS / "sen1floods11/unet_test.json"
    if unet_path.exists():
        data = json.loads(unet_path.read_text())["test"]
        labels.append("S1 U-Net")
        micros.append(data["iou"])
        macros.append(float(np.mean([x["iou"] for x in data["per_chip"] if x["iou"] is not None])))
    geo = json.loads((RESULTS / "georeference/error_grid.json").read_text())
    errors = [r["max_error_m"] for r in geo["per_chip"]]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.15), gridspec_kw={"width_ratios": [1.2, 1]})
    x = np.arange(len(labels))
    axes[0].bar(x - 0.17, micros, 0.34, label="Micro IoU", color="#2C5F8A")
    axes[0].bar(x + 0.17, macros, 0.34, label="Mean chip IoU", color="#90B7D0")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("IoU")
    axes[0].legend(frameon=False, loc="upper left", fontsize=6)
    axes[1].hist(errors, bins=12, color="#487B91", edgecolor="white")
    axes[1].axvline(np.median(errors), color="#A63A32", ls="--", lw=1,
                    label=f"median {np.median(errors):.0f} m")
    axes[1].set_xlabel("Maximum chip error (m)")
    axes[1].set_ylabel("GeoTIFF chips")
    axes[1].legend(frameon=False, fontsize=6)
    fig.tight_layout(pad=0.8)
    fig.savefig(FIGURES / "fig_ground_truth.pdf")
    plt.close(fig)


def scale():
    rows = json.loads((RESULTS / "fleet_sensitivity/sensitivity.json").read_text())
    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    for map_name, color, marker in (("varanasi", "#2C5F8A", "o"), ("kanpur", "#A63A32", "s")):
        group = [r for r in rows if r["map"] == map_name and r["variable"] == "scale"]
        ax.plot([r["meters_per_pixel"] for r in group], [r["visited"] for r in group],
                color=color, marker=marker, ms=3, lw=1.3, label=map_name.capitalize())
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("Ground scale (m per original pixel)")
    ax.set_ylabel("Visited drops")
    ax.legend(frameon=False, fontsize=6)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(pad=0.7)
    fig.savefig(FIGURES / "fig_scale_coverage.pdf")
    plt.close(fig)


def sitl():
    result = json.loads((RESULTS / "sitl/sitl_result.json").read_text())
    items = [line.split() for line in (RESULTS / "sitl/two_drop_reload.waypoints").read_text().splitlines()[1:]]
    traj = result["trajectory"]
    servo = [e for e in result["events"] if e["type"] == "SERVO_OUTPUT_RAW"]
    to_m = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
    lon0, lat0 = float(items[0][9]), float(items[0][8])
    x0, y0 = to_m.transform(lon0, lat0)
    route = [to_m.transform(float(r[9]), float(r[8])) for r in items[:15]]
    flown = [to_m.transform(t["lon"], t["lat"]) for t in traj]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.15), gridspec_kw={"width_ratios": [1.1, 1]})
    axes[0].plot([t["t_s"] for t in traj], [t["relative_alt_m"] for t in traj],
                 color="#2C5F8A", lw=1.1)
    for event in servo:
        if event["servo9_pwm"] == 2000:
            axes[0].axvline(event["t_s"], color="#A63A32", ls="--", lw=1,
                            label=f"servo 9: 2000 $\\mu$s at {event['alt_m']:.1f} m")
    axes[0].set_xlabel("Wall time (s); SITL speedup shown in caption")
    axes[0].set_ylabel("Relative altitude (m)")
    axes[0].legend(frameon=False, loc="upper right", fontsize=6)
    axes[1].plot([p[0] - x0 for p in route], [p[1] - y0 for p in route],
                 color="#A63A32", ls="--", lw=1.1, label="planned short mission")
    axes[1].plot([p[0] - x0 for p in flown], [p[1] - y0 for p in flown],
                 color="#2C5F8A", lw=1.1, label="SITL position")
    axes[1].scatter([route[4][0] - x0], [route[4][1] - y0], marker="o",
                    c="#A63A32", s=12, zorder=3)
    axes[1].set_xlabel("East of HOME (m)")
    axes[1].set_ylabel("North of HOME (m)")
    axes[1].axis("equal")
    axes[1].legend(frameon=False, fontsize=6)
    fig.tight_layout(pad=0.8)
    fig.savefig(FIGURES / "fig_sitl.pdf")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    ground_truth()
    scale()
    if (RESULTS / "sitl/sitl_result.json").exists():
        sitl()
