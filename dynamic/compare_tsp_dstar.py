#!/usr/bin/env python3
"""TSP vs D* Lite comparison on REAL detected flood zones.

Selects the top-N largest flood blobs (the red zones) from the S1 mask,
uses the full flood mask as the obstacle field, then compares:
  - NN-TSP straight-line direct path (ignores water)
  - D* Lite actual path (avoids water)
Repo style: red dashed = TSP direct, blue solid = D* actual,
green star = home, cyan = drop points, gray = obstacle field.
"""
import os, sys, json, argparse
_here = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    if os.path.isdir(os.path.join(_c, "src")):
        sys.path.insert(0, _c)
        break
else:
    sys.exit("repo root with src/ not found — run from inside the A.D.A.P.T. repo")
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.routing.pathfinding import nearest_neighbor_tsp, compute_full_path

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="nepal2026")
ap.add_argument("--sel-min", type=int, default=300, help="min blob px for zone selection")
args_cli, _ = ap.parse_known_args()
BASE = os.path.join(os.path.dirname(__file__), "out", args_cli.name)
RUN = os.path.join(BASE, "pipeline") if os.path.exists(os.path.join(BASE, "pipeline", "flood_mask.png")) else BASE


def safe_boundary_point(mask, blob_mask, target_xy):
    """Boundary pixel of blob closest to target_xy."""
    cnts, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack([c.reshape(-1, 2) for c in cnts])
    d = np.hypot(pts[:, 0] - target_xy[0], pts[:, 1] - target_xy[1])
    return tuple(int(v) for v in pts[int(np.argmin(d))])


def polyline_len(pts):
    p = np.asarray(pts, dtype=float)
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def main(top_n=3):
    mask = cv2.imread(os.path.join(RUN, "flood_mask.png"), cv2.IMREAD_GRAYSCALE)
    sel_min = args_cli.sel_min
    style = cv2.imread(os.path.join(BASE, "style_map.png"))
    geo = json.load(open(os.path.join(RUN, "geo.json")))
    mpp = geo["meters_per_pixel"]
    h, w = mask.shape
    cy, cx = h // 2, w // 2

    # ---- select top-N flood blobs: largest, then greedy max-spread ----
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    big = [li for li in range(1, n) if stats[li, cv2.CC_STAT_AREA] >= sel_min]
    big.sort(key=lambda li: -stats[li, cv2.CC_STAT_AREA])
    cand = big[:25]
    chosen = [cand[0]]
    while len(chosen) < top_n and len(cand) > len(chosen):
        def dmin(li):
            cnts, _ = cv2.findContours((labels == li).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = np.vstack([cc.reshape(-1, 2) for cc in cnts]).mean(axis=0)
            return min(np.hypot(c[0]-cx0, c[1]-cy0) for cx0, cy0 in
                       [np.vstack([cv2.findContours((labels == k).astype(np.uint8),
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].reshape(-1, 2)]).mean(axis=0) for k in chosen])
        best = max([li for li in cand if li not in chosen], key=dmin)
        chosen.append(best)
    blobs = []
    for rank, li in enumerate(chosen):
        bm = (labels == li).astype(np.uint8) * 255
        anchor = (cx, cy) if rank == 0 else blobs[-1][1]
        sp = safe_boundary_point(mask, bm, anchor)
        blobs.append((bm, sp, int(stats[li, cv2.CC_STAT_AREA])))
        print(f"zone {rank+1}: area {stats[li, cv2.CC_STAT_AREA]} px "
              f"({stats[li, cv2.CC_STAT_AREA]*mpp*mpp/1e6:.2f} km2), safe pt {sp}")
    drops = [sp for _, sp, _ in blobs]
    home = drops[0]  # closest-to-center blob boundary point

    # ---- TSP direct (ignores water) ----
    order_idx, ordered = nearest_neighbor_tsp(drops, home=home)

    # ---- D* Lite actual (water = obstacle) ----
    full_path, drop_idx = compute_full_path(ordered, mask, downsample_factor=4)

    # ---- metrics ----
    tsp_pts = [home] + [drops[i] for i in order_idx] + [home]
    tsp_km = polyline_len(tsp_pts) * mpp / 1000
    d_km = polyline_len(full_path) * mpp / 1000
    print(f"\nTSP direct total : {tsp_km:6.1f} km")
    print(f"D* Lite actual   : {d_km:6.1f} km  ({(d_km/tsp_km-1)*100:+.0f}% detour for water avoidance)")

    # ---- plot A: repo style (obstacle field background) ----
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(255 - mask, cmap="gray", origin="upper")
    tsp_x = [p[0] for p in ordered]; tsp_y = [p[1] for p in ordered]
    ax.plot(tsp_x, tsp_y, "r--", label="TSP direct (ignores water)", alpha=0.7, lw=2)
    dx = [p[0] for p in full_path]; dy = [p[1] for p in full_path]
    ax.plot(dx, dy, "b-", label="D* Lite actual (avoids water)", lw=2)
    ax.plot(home[0], home[1], "g*", ms=16, label="HOME")
    ax.plot([p[0] for p in drops[1:]], [p[1] for p in drops[1:]], "co", ms=9, label="drop zones (top-3)")
    ax.set_title(f"TSP vs D* Lite — {top_n} detected flood zones | direct {tsp_km:.1f} km vs actual {d_km:.1f} km")
    ax.legend(loc="lower left"); ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(RUN, "tsp_vs_dstar.png"), dpi=120); plt.close(fig)

    # ---- plot B: same overlay on the OSM style map ----
    vis = cv2.resize(style, (w, h)) if style is not None else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    from src.weather.flood_spread import predict_spread
    WEATHER = {"precipitation": 10.0, "wind_speed_10m": 30.0, "wind_direction_10m": 270.0}
    pred = predict_spread(mask, WEATHER, 2.0)
    k = max(w, h) / 1400.0; th = max(1, int(round(2 * k))); fs = 0.55 * max(k, 0.6)
    pc, _ = cv2.findContours(pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, pc, -1, (0, 165, 255), max(1, th - 1))            # orange: predicted spread
    for i in range(len(ordered) - 1):
        cv2.line(vis, ordered[i], ordered[i + 1], (0, 0, 255), th)          # red: TSP direct
    for i in range(len(full_path) - 1):
        cv2.line(vis, tuple(map(int, full_path[i])), tuple(map(int, full_path[i + 1])), (255, 80, 0), th)  # blue: D*
    for p in drops[1:]:
        cv2.circle(vis, p, int(7 * k) + 2, (255, 255, 0), -1)
    cv2.drawMarker(vis, home, (0, 255, 0), cv2.MARKER_STAR, int(22 * k) + 6, max(2, th))
    msg = f"TSP direct {tsp_km:.0f} km (red)  vs  D* Lite {d_km:.0f} km (blue)  |  orange = predicted spread"
    cv2.putText(vis, msg, (12, h - int(24 * max(k, 0.7))), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), max(2, int(2.5 * k)), cv2.LINE_AA)
    cv2.putText(vis, msg, (12, h - int(24 * max(k, 0.7))), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(RUN, "tsp_vs_dstar_overlay.png"), vis)
    print("saved:", os.path.join(RUN, "tsp_vs_dstar.png"), "+ tsp_vs_dstar_overlay.png")


if __name__ == "__main__":
    main()
