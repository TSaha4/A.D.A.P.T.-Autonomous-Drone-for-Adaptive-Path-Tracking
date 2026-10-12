"""Figures reconstructed only from results committed at research-validation fdb9cb3."""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "figures"
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})


def fleet_performance():
    records = json.loads((RESULTS / "multi_uav" / "comparison.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.25))
    for map_name, color in (("varanasi", "#1f77b4"), ("kanpur", "#d95f02")):
        rows = sorted((r for r in records if r["map"] == map_name and r["weather"] == "moderate"), key=lambda r: r["cap"])
        x = [r["cap"] for r in rows]
        axes[0].plot(x, [r["visited"] / r["drop_points"] * 100 for r in rows], "o-", color=color, label=map_name.capitalize())
        axes[1].plot(x, [r["est_completion_min"] for r in rows], "o-", color=color, label=map_name.capitalize())
    axes[0].set(ylabel="Drop coverage (%)", xlabel="Base cap K", xticks=[1, 2, 3, 4], ylim=(30, 90))
    axes[1].set(ylabel="Estimated completion (min)", xlabel="Base cap K", xticks=[1, 2, 3, 4], ylim=(18, 40))
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_fleet_performance.pdf", bbox_inches="tight")
    plt.close(fig)


def fleet_routes():
    paths = sorted(glob.glob(str(RESULTS / "multi_uav" / "sessions" / "kanpur_moderate_K4" / "*" / "base*.route.json")))
    image = np.load(RESULTS / "e2e" / "kanpur.npz")["display_image"][:, :, ::-1]
    fig, ax = plt.subplots(figsize=(4.5, 4.3))
    ax.imshow(image, alpha=0.55)
    colors = ["#0072b2", "#d55e00", "#009e73", "#cc79a7"]
    for i, path in enumerate(paths):
        route = json.loads(Path(path).read_text())
        pts = np.asarray(route["full_path"])
        ax.plot(pts[:, 0], pts[:, 1], color=colors[i], lw=0.8, label=f"UAV {i+1}: {len(route['drop_indices'])} drops")
        drops = pts[np.asarray(route["drop_indices"], dtype=int)]
        ax.scatter(drops[:, 0], drops[:, 1], s=7, color=colors[i])
        ax.scatter(*route["home"], s=35, marker="*", color=colors[i], edgecolor="black", linewidth=0.4)
    ax.set(xlim=(0, image.shape[1]), ylim=(image.shape[0], 0))
    ax.axis("off")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_fleet_routes.pdf", bbox_inches="tight")
    plt.close(fig)


def pipeline_pair():
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.45))
    for row, name in enumerate(("varanasi", "kanpur")):
        arr = np.load(RESULTS / "e2e" / f"{name}.npz")
        rec = json.loads((RESULTS / "e2e" / f"{name}.json").read_text())
        image, mask, pred = arr["display_image"][:, :, ::-1], arr["mask"], arr["pred_mask"]
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(mask, cmap="gray_r")
        axes[row, 2].imshow(image, alpha=0.48)
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask > 0] = (0.1, 0.3, 0.7, 0.38)
        overlay[(pred > 0) & (mask == 0)] = (1, 0.5, 0, 0.35)
        axes[row, 2].imshow(overlay)
        pts = np.asarray(rec["full_path"])
        axes[row, 2].plot(pts[:, 0], pts[:, 1], color="#009e73", lw=0.55)
        for a, b in zip(pts[:-1], pts[1:]):
            if np.linalg.norm(a-b) > 15:
                axes[row, 2].plot([a[0], b[0]], [a[1], b[1]], color="#d62728", lw=0.9)
        for col, title in enumerate((f"{name.capitalize()} input", "Flood mask", "Route and obstacles")):
            axes[row, col].set_title(title)
            axes[row, col].axis("off")
    fig.tight_layout(w_pad=0.15, h_pad=0.15)
    fig.savefig(FIGURES / "fig_pipeline_pair.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    fleet_performance()
    fleet_routes()
    pipeline_pair()
