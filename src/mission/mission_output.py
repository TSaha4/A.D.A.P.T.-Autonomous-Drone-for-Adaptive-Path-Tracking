import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from src.mission.coordinates import pixel_to_latlon

def generate_mission_file(full_path, drop_indices, home, image_center_px, geo_center, meters_per_pixel, filename="mission.waypoints", base_altitude=100, drop_altitude=10, servo_channel=9, servo_pwm=2000, reload_indices=None):
    # Mission:
    # Takeoff at home → visit waypoints → drop packages at drop indices → return home → land.
    # reload_indices (multi-base mode): indices of full_path where the UAV is back at HOME between sorties;
    # each becomes LAND then TAKEOFF at HOME (battery/payload reload). None (single-UAV mode) adds nothing.
    # Convert every point before opening the file, so an invalid coordinate cannot leave a partial mission
    home_lat, home_lon = pixel_to_latlon(home[0], home[1], image_center_px, geo_center, meters_per_pixel)
    path_latlon = [pixel_to_latlon(pt[0], pt[1], image_center_px, geo_center, meters_per_pixel) for pt in full_path]
    reload_set = set(reload_indices or [])
    for r in reload_set:
        if not (0 < r < len(full_path) - 1) or tuple(full_path[r]) != tuple(home) or r in drop_indices:
            raise ValueError(f"reload index {r} is not an intermediate return to HOME")

    with open(filename, "w") as f:
        f.write("QGC WPL 110\n")
        seq = 0

        # 0: Home location
        f.write(f"{seq}\t1\t3\t16\t0\t0\t0\t0\t{home_lat}\t{home_lon}\t0\t1\n")
        seq += 1

        # 1: Takeoff at home
        f.write(f"{seq}\t0\t3\t22\t0\t0\t0\t0\t{home_lat}\t{home_lon}\t{base_altitude}\t1\n")
        seq += 1

        for i, pt in enumerate(full_path):
            lat, lon = path_latlon[i]

            # If it's the home point (start or end of mission), skip unless it's a drop (usually not)
            if i == 0 and pt == home:
                continue

            if i in reload_set:
                # Land at HOME for the reload, then take off again
                f.write(f"{seq}\t0\t3\t21\t0\t0\t0\t0\t{home_lat}\t{home_lon}\t0\t1\n")
                seq += 1
                f.write(f"{seq}\t0\t3\t22\t0\t0\t0\t0\t{home_lat}\t{home_lon}\t{base_altitude}\t1\n")
                seq += 1
                continue

            # Fly to location at cruise altitude
            f.write(f"{seq}\t0\t3\t16\t0\t0\t0\t0\t{lat}\t{lon}\t{base_altitude}\t1\n")
            seq += 1
            
            # First/last entries are the HOME departure/return; any other occurrence of HOME is
            # the drop for the cluster whose safe point was chosen as HOME.
            if i in drop_indices and 0 < i < len(full_path) - 1:
                # Loiter 5 sec
                f.write(f"{seq}\t0\t3\t19\t5\t0\t0\t0\t{lat}\t{lon}\t{base_altitude}\t1\n")
                seq += 1
                # Descend to drop altitude
                f.write(f"{seq}\t0\t3\t16\t0\t0\t0\t0\t{lat}\t{lon}\t{drop_altitude}\t1\n")
                seq += 1
                # Drop package (MAV_CMD_DO_SET_SERVO: param1 = servo channel, param2 = PWM)
                f.write(f"{seq}\t0\t3\t183\t{servo_channel}\t{servo_pwm}\t0\t0\t{lat}\t{lon}\t{drop_altitude}\t1\n")
                seq += 1
                # Ascend back to cruise altitude
                f.write(f"{seq}\t0\t3\t16\t0\t0\t0\t0\t{lat}\t{lon}\t{base_altitude}\t1\n")
                seq += 1

        # Return to home
        f.write(f"{seq}\t0\t3\t20\t0\t0\t0\t0\t{home_lat}\t{home_lon}\t{base_altitude}\t1\n")
        seq += 1
        # Land at home
        f.write(f"{seq}\t0\t3\t21\t0\t0\t0\t0\t{home_lat}\t{home_lon}\t0\t1\n")

def display_path_on_map(image, contours, pred_mask, full_path, drop_indices, home=None, tsp_path=None):
    vis = image.copy()

    # Draw current contours
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    
    # Draw predicted spread
    if pred_mask is not None:
        pred_contours, _ = cv2.findContours(pred_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, pred_contours, -1, (0, 165, 255), 2) # Orange for predicted spread

    # Draw safe points (small blue dots)
    for idx in drop_indices:
        if idx < len(full_path):
            pt_int = tuple(map(int, full_path[idx]))
            cv2.circle(vis, pt_int, 5, (255, 0, 0), -1)

    # Draw HOME (green)
    if home:
        cv2.circle(vis, tuple(map(int, home)), 10, (0, 0, 0), -1)

    # Draw D* Lite path lines (blue) via cv2
    for i in range(len(full_path) - 1):
        pt1 = tuple(map(int, full_path[i]))
        pt2 = tuple(map(int, full_path[i + 1]))
        cv2.line(vis, pt1, pt2, (255, 0, 0), 2) 

    # Display
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    
    # Draw TSP direct path lines (red dashed) using matplotlib so it's clearly distinguishable
    if tsp_path:
        tsp_x = [pt[0] for pt in tsp_path]
        tsp_y = [pt[1] for pt in tsp_path]
        plt.plot(tsp_x, tsp_y, 'r--', linewidth=2, label="TSP Direct Path")
        plt.plot([], [], 'b-', linewidth=2, label="D* Lite Avoidance Path")
        plt.legend(loc="upper right")
        
    plt.title("Drone Path with Safe Drop Zones, Obstacles & Home")
    plt.axis("off")
    plt.show()


# --- Run-time mission check and multi-base map (multi-base mode; ported from the MAIN branch) ---

# MAV_CMD ids used in exported missions.
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_LOITER_TIME = 19
MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_NAV_LAND = 21
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_DO_SET_SERVO = 183
MAV_FRAME_GLOBAL = 0
MAV_FRAME_GLOBAL_RELATIVE_ALT = 3

# Map styling (BGR)
CONTOUR_GREEN_BGR = (0, 200, 0)
FORECAST_BROWN_BGR = (30, 75, 140)
# Unit house outline (image coordinates, y down): square body with a pitched roof.
HOUSE_OUTLINE = np.array([[-0.8, 0.8], [0.8, 0.8], [0.8, -0.1], [1.0, -0.1],
                          [0.0, -1.0], [-1.0, -0.1], [-0.8, -0.1]])


def validate_waypoints_file(filename):
    """Structural QGC WPL 110 check; returns a list of problems (empty when valid).

    Checks the header, 12 tab-separated fields per item, consecutive sequence
    numbers, a single current item (the home row), known frames and commands,
    coordinate ranges, autocontinue, and DO_SET_SERVO servo/PWM parameters.
    """
    known_commands = {MAV_CMD_NAV_WAYPOINT, MAV_CMD_NAV_LOITER_TIME, MAV_CMD_NAV_RETURN_TO_LAUNCH,
                      MAV_CMD_NAV_LAND, MAV_CMD_NAV_TAKEOFF, MAV_CMD_DO_SET_SERVO}
    with open(filename) as f:
        lines = f.read().splitlines()
    problems = []
    if not lines or lines[0] != "QGC WPL 110":
        problems.append(f"header is {lines[0]!r}, expected 'QGC WPL 110'" if lines else "empty file")
        return problems
    if len(lines) < 2:
        problems.append("no mission items")
    for n, line in enumerate(lines[1:]):
        fields = line.split("\t")
        if len(fields) != 12:
            problems.append(f"item {n}: {len(fields)} fields, expected 12")
            continue
        try:
            seq, current, frame, command = (int(v) for v in fields[:4])
            p1, p2, p3, p4, lat, lon, alt = (float(v) for v in fields[4:11])
            autocontinue = int(fields[11])
        except ValueError as exc:
            problems.append(f"item {n}: non-numeric field ({exc})")
            continue
        if seq != n:
            problems.append(f"item {n}: sequence number {seq}")
        if current != (1 if n == 0 else 0):
            problems.append(f"item {n}: current flag {current}")
        if frame not in (MAV_FRAME_GLOBAL, MAV_FRAME_GLOBAL_RELATIVE_ALT):
            problems.append(f"item {n}: frame {frame}")
        if command not in known_commands:
            problems.append(f"item {n}: unknown command {command}")
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0) or (lat == 0.0 and lon == 0.0):
            problems.append(f"item {n}: coordinates {lat}, {lon}")
        if autocontinue != 1:
            problems.append(f"item {n}: autocontinue {autocontinue}")
        if command == MAV_CMD_DO_SET_SERVO and not (p1 >= 1 and 800 <= p2 <= 2200):
            problems.append(f"item {n}: DO_SET_SERVO servo={p1:g} pwm={p2:g}")
    return problems


def draw_dashed_polyline(img, points, color, thickness=2, dash_len=10, gap_len=7):
    """Dashed polyline whose dash pattern runs continuously across vertices.

    Routed legs are made of many ~5 px grid steps; restarting the pattern on
    every step (as a per-segment dash would) renders the whole leg solid.
    """
    pts = [np.asarray(p, dtype=float) for p in points]
    period = dash_len + gap_len
    offset = 0.0  # arc length already consumed in the current dash period
    for a, b in zip(pts, pts[1:]):
        seg = float(np.linalg.norm(b - a))
        if seg < 1e-6:
            continue
        direction = (b - a) / seg
        s = 0.0
        while s < seg:
            phase = offset % period
            if phase < dash_len:
                step = min(dash_len - phase, seg - s)
                p0, p1 = a + direction * s, a + direction * (s + step)
                cv2.line(img, tuple(int(round(v)) for v in p0), tuple(int(round(v)) for v in p1),
                         color, thickness, cv2.LINE_AA)
            else:
                step = min(period - phase, seg - s)
            s += step
            offset += step


def arrow_positions(points, spacing, min_chord=12.0):
    """(tail, head) pairs placed every ``spacing`` px of arc length along a polyline.

    Each arrow spans a chord of ``min_chord`` px so its direction is not
    dominated by a single grid step.
    """
    pts = [np.asarray(p, dtype=float) for p in points]
    if len(pts) < 2:
        return []
    cumulative = [0.0]
    for a, b in zip(pts, pts[1:]):
        cumulative.append(cumulative[-1] + float(np.linalg.norm(b - a)))
    total = cumulative[-1]
    if total < min_chord:
        return []

    def at(s):
        i = int(np.searchsorted(cumulative, s, side="right")) - 1
        i = min(max(i, 0), len(pts) - 2)
        seg = cumulative[i + 1] - cumulative[i]
        t = 0.0 if seg < 1e-9 else (s - cumulative[i]) / seg
        return pts[i] + (pts[i + 1] - pts[i]) * t

    arrows = []
    n = max(1, int(total // spacing))
    for k in range(n):
        mid = (k + 0.5) * total / n
        tail, head = at(max(0.0, mid - min_chord / 2)), at(min(total, mid + min_chord / 2))
        arrows.append((tuple(tail), tuple(head)))
    return arrows


def basemap_sanity(rendered_bgr, source_bgr, max_dominant_fraction=0.5):
    """Check that a rendered map still shows the source basemap.

    Returns ``(ok, details)``. Fails when one colour covers more than
    ``max_dominant_fraction`` of the render (a corrupted solid/flat image), when
    shapes differ, or when the render no longer correlates with the source.
    """
    rendered = np.asarray(rendered_bgr)
    source = np.asarray(source_bgr)
    colours, counts = np.unique(rendered.reshape(-1, rendered.shape[-1]), axis=0, return_counts=True)
    dominant = float(counts.max() / counts.sum())
    details = {"dominant_colour": tuple(int(c) for c in colours[counts.argmax()]),
               "dominant_fraction": dominant, "shape_match": rendered.shape == source.shape,
               "correlation": None, "unchanged_fraction": None}
    if details["shape_match"]:
        r = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY).astype(float).ravel()
        s = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(float).ravel()
        details["correlation"] = float(np.corrcoef(r, s)[0, 1]) if r.std() > 0 and s.std() > 0 else 0.0
        details["unchanged_fraction"] = float(np.mean(np.all(rendered == source, axis=-1)))
    ok = (dominant <= max_dominant_fraction and details["shape_match"]
          and details["correlation"] is not None and details["correlation"] >= 0.5)
    return ok, details


# Sortie path colours: dark (L* <= 62), saturated, no greens/teals (hues 40-200 deg are
# excluded; green is reserved for flood contours), CIELAB dE >= 34 between any two
# and >= 35 from drop-point blue, BASE black, RELOAD cyan/yellow, contour green and
# forecast brown. Chosen by max-min dE search; more sorties than colours repeat (logged).
SORTIE_PALETTE_HEX = ["#d50000", "#1a237e", "#f200f2", "#d86191", "#3990bf", "#590016",
                      "#6248f2", "#d87700", "#6d87f2", "#a51897", "#f20054", "#59004b"]
DROP_POINT_BGR = (255, 0, 0)


def _hex_to_bgr(colour):
    return tuple(int(colour[i:i + 2], 16) for i in (5, 3, 1))


def split_sorties(full_path, drop_indices, reload_indices):
    """Sortie index ranges (outbound to last drop, then RTB) of one base route."""
    sorties, start = [], 0
    for r_idx in reload_indices:
        drops = [d for d in drop_indices if start <= d < r_idx]
        sorties.append({"start": start, "outward_end": max(drops) if drops else start,
                        "rtb_end": r_idx, "drops": drops})
        start = r_idx + 1
    if full_path and start < len(full_path):
        drops = [d for d in drop_indices if start <= d]
        sorties.append({"start": start, "outward_end": max(drops) if drops else start,
                        "rtb_end": len(full_path) - 1, "drops": drops})
    # Only sorties that deliver something are drawn and get legend entries.
    return [sortie for sortie in sorties if sortie["drops"]]


def display_multi_base_map(image, contours, pred_mask, bases, territories=None, save_path=None, show=True):
    """Render the mission map of the multi-base mode (one or more bases).

    ``bases`` is a list of dicts with keys ``home``, ``full_path``, ``drop_indices``,
    ``reload_indices`` and optionally ``number`` (the base number used in base<N>.waypoints;
    defaults to the 1-based list position). All sorties get distinct colours and, with more than
    one base, bases are labelled with their numbers. ``territories`` is an optional per-pixel
    territory-label image (-1 = none) whose cell boundaries are drawn.
    Ported from MAIN's display_path_on_map; the single-UAV mode keeps display_path_on_map.
    """
    import logging
    from pathlib import Path
    from matplotlib.patches import FancyArrowPatch
    vis = image.copy()
    bases = [dict(base) for base in bases]
    multi = len(bases) > 1
    for k, base in enumerate(bases, 1):
        base["number"] = int(base.get("number") or k)
        base["drop_indices"] = list(base.get("drop_indices") or [])
        base["reload_indices"] = sorted({int(i) for i in (base.get("reload_indices") or [])
                                         if 0 <= int(i) < len(base["full_path"])})
        base["sorties"] = split_sorties(base["full_path"], base["drop_indices"], base["reload_indices"])

    # Forecast flood spread: the whole forecast area hatched with brown lines,
    # outlined in brown (drawn first so the detected contours sit on top).
    if pred_mask is not None:
        forecast = np.asarray(pred_mask) > 0
        hatch = np.zeros(forecast.shape, dtype=np.uint8)
        diagonal = np.add.outer(np.arange(forecast.shape[0]), np.arange(forecast.shape[1]))
        hatch[(diagonal % 7) < 2] = 1
        vis[forecast & (hatch > 0)] = FORECAST_BROWN_BGR
        pred_contours, _ = cv2.findContours((forecast * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        pred_contours = [c for c in pred_contours if cv2.contourArea(c) >= 200]  # outline zones, not specks
        cv2.drawContours(vis, pred_contours, -1, FORECAST_BROWN_BGR, 1)

    # Detected flood zones: green borders on top
    cv2.drawContours(vis, contours, -1, CONTOUR_GREEN_BGR, 2, cv2.LINE_AA)

    # Base territories (multi-base): dotted boundaries between base cells
    if multi and territories is not None:
        t = np.asarray(territories)
        edge = np.zeros(t.shape, dtype=bool)
        edge[:, :-1] |= (t[:, :-1] != t[:, 1:]) & (t[:, :-1] >= 0) & (t[:, 1:] >= 0)
        edge[:-1, :] |= (t[:-1, :] != t[1:, :]) & (t[:-1, :] >= 0) & (t[1:, :] >= 0)
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
        dots = (np.add.outer(np.arange(t.shape[0]), np.arange(t.shape[1])) % 8) < 4
        vis[edge & dots] = (40, 40, 40)

    # Global sortie list, coloured from the palette in order
    drawn = [(base["number"], base, sortie) for base in bases for sortie in base["sorties"]]
    if len(drawn) > len(SORTIE_PALETTE_HEX):
        logging.warning("MAP PALETTE: %d sorties exceed the %d distinct path colours; colours repeat.",
                        len(drawn), len(SORTIE_PALETTE_HEX))
    for k, (b, base, sortie) in enumerate(drawn):
        colour = _hex_to_bgr(SORTIE_PALETTE_HEX[k % len(SORTIE_PALETTE_HEX)])
        path = base["full_path"]
        for i in range(sortie["start"], sortie["outward_end"]):
            cv2.line(vis, tuple(map(int, path[i])), tuple(map(int, path[i + 1])), colour, 4)
        # RTB: dashed, pattern continuous along the leg (6/11 renders ~11 px dash / ~6 px gap).
        draw_dashed_polyline(vis, path[sortie["outward_end"]:sortie["rtb_end"] + 1],
                             colour, thickness=3, dash_len=6, gap_len=11)

    # Drop points above the routes (blue dots)
    for base in bases:
        for idx in base["drop_indices"]:
            if idx < len(base["full_path"]):
                pt_int = tuple(map(int, base["full_path"][idx]))
                cv2.circle(vis, pt_int, 6, DROP_POINT_BGR, -1, cv2.LINE_AA)
                cv2.circle(vis, pt_int, 7, (255, 255, 255), 1, cv2.LINE_AA)

    def label(text, at, scale):
        cv2.putText(vis, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), 2, cv2.LINE_AA)

    any_reload = False
    for base in bases:
        b = base["number"]
        if base["home"] is None:
            continue
        home_pt = tuple(map(int, base["home"]))
        # Reloads at the base share one marker/label with BASE; any elsewhere get their own.
        reloads_at_base = 0
        for reload_idx in base["reload_indices"]:
            any_reload = True
            pt = tuple(map(int, base["full_path"][reload_idx]))
            if np.linalg.norm(np.subtract(pt, home_pt)) <= 15:
                reloads_at_base += 1
                continue
            cv2.circle(vis, pt, 12, (255, 255, 255), 3)
            cv2.drawMarker(vis, pt, (0, 255, 255), cv2.MARKER_STAR, 22, 3)
            label("RELOAD", (pt[0] + 14, pt[1] + 22), 0.55)
        # Base marker: black house with a white outline
        house = (HOUSE_OUTLINE * 11 + np.array(home_pt)).astype(np.int32)
        cv2.polylines(vis, [house.reshape(-1, 1, 2)], True, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.fillPoly(vis, [house.reshape(-1, 1, 2)], (0, 0, 0), cv2.LINE_AA)
        label(f"BASE {b}" if multi else "BASE", (home_pt[0] + 16, home_pt[1] - 10), 0.65)
        if reloads_at_base:
            star = (home_pt[0] + 22, home_pt[1] + 13)
            cv2.drawMarker(vis, star, (0, 0, 0), cv2.MARKER_STAR, 16, 4)
            cv2.drawMarker(vis, star, (255, 255, 0), cv2.MARKER_STAR, 16, 2)
            label(f"RELOAD x{reloads_at_base}", (home_pt[0] + 32, home_pt[1] + 18), 0.5)

    # Matplotlib High-Resolution Rendering with Directional Arrows
    fig, ax = plt.subplots(figsize=(11, 11 * vis.shape[0] / max(1, vis.shape[1])))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    # Directional arrows along the actual D* Lite paths, spaced by arc length
    for k, (b, base, sortie) in enumerate(drawn):
        colour_hex = SORTIE_PALETTE_HEX[k % len(SORTIE_PALETTE_HEX)]
        path = base["full_path"]
        legs = [(path[sortie["start"]:sortie["outward_end"] + 1], '-'),
                (path[sortie["outward_end"]:sortie["rtb_end"] + 1], '--')]
        for pts, style in legs:
            for tail, head in arrow_positions(pts, spacing=110):
                ax.add_patch(FancyArrowPatch(tail, head, arrowstyle='-|>,head_width=4,head_length=7',
                                             color=colour_hex, linewidth=1.5, linestyle=style, zorder=5))

    # Legend entries only for sorties that were actually rendered
    sortie_handles = []
    counters = {}
    for k, (b, base, sortie) in enumerate(drawn):
        counters[b] = counters.get(b, 0) + 1
        name = f"B{b} Sortie {counters[b]}" if multi else f"Sortie {counters[b]}"
        c_hex = SORTIE_PALETTE_HEX[k % len(SORTIE_PALETTE_HEX)]
        sortie_handles.append(Line2D([0], [0], color=c_hex, linestyle='-', linewidth=3.5,
                                     label=f"{name} Outbound"))
        sortie_handles.append(Line2D([0], [0], color=c_hex, linestyle='--', linewidth=2.5,
                                     dashes=(4, 3), label=f"{name} RTB"))

    from matplotlib.patches import Patch
    from matplotlib.path import Path as MarkerPath
    house_marker = MarkerPath(np.vstack([HOUSE_OUTLINE * [1, -1], HOUSE_OUTLINE[:1] * [1, -1]]))
    brown_hex = "#%02x%02x%02x" % FORECAST_BROWN_BGR[::-1]
    sortie_handles.extend([
        Line2D([0], [0], color="#%02x%02x%02x" % CONTOUR_GREEN_BGR[::-1], linewidth=2,
               label="Detected flood contour"),
        Patch(facecolor="none", edgecolor=brown_hex, hatch="////", label="Forecast flood spread"),
        Line2D([0], [0], color="blue", marker="o", markeredgecolor="white", linestyle="None",
               markersize=8, label="Drop point"),
        Line2D([0], [0], color="black", marker=house_marker, markeredgecolor="white", linestyle="None",
               markersize=14, label="BASE (numbered)" if multi else "BASE (Home)"),
    ])
    if multi:
        sortie_handles.append(Line2D([0], [0], color="#282828", linestyle=":", linewidth=1.5,
                                     label="Base territory"))
    if any_reload:
        sortie_handles.append(Line2D([0], [0], color="black", marker="*", markerfacecolor="#00ffff",
                                     markersize=12, linestyle="None", label="Reload stop"))
    # Legend outside the image so it never hides route segments.
    ax.legend(handles=sortie_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              framealpha=0.95, fontsize=9, prop={'weight': 'bold'})

    title = ("ADAPT Multi-Base Autonomous Drone Mission Route Map" if multi
             else "ADAPT Multi-Sortie Autonomous Drone Mission Route Map")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=160)
        cv2.imwrite(str(Path(save_path).parent / "mission_map_annotated.png"), vis)

    if show:
        plt.show()
    plt.close(fig)
    return vis
