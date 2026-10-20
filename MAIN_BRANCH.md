# ADAPT — MAIN Branch Reverse-Engineered Codebase Specification

**Repository:** `A.D.A.P.T.-Autonomous-Drone-for-Adaptive-Path-Tracking`
**Branch analysed:** `main` @ `ca27215` (full hash `ca2721542aad77f68cbdceba5f1e27979d99875f`, "Add multi-base routing, path-overlap repair, session output, updated palette", 2026-10-04). `origin/main` and `origin/routing-algo` point at the same commit.
**Analysis date:** 2026-10-08. Working tree was clean before and after the analysis; nothing in the repository was modified.

**How this document was produced:** every tracked source, test, config, script, and doc file was read in full. The test suite was executed (in a throwaway venv outside the repo), and `main.py` was run end-to-end on an exported copy of HEAD (outside the repo) for all three bundled maps. The section "Observed runs" lists the actual results. Statements tagged **UNCLEAR — requires verification** could not be settled from the code.

**How to get MAIN's source from the DEVELOPMENT checkout** (same git remote):
```bash
git fetch origin
git show ca27215:src/mission/multi_base.py          # any file
git diff 775165d ca27215 -- src/ main.py config/    # everything the multi-base commit changed
```
`775165d` is the merge base of `main` and the remote branches `research-validation`, `ajay`, `time-aware-routing` and `backup`. See §17.0.

---

## 0. EXECUTIVE SUMMARY: read this first

1. **"Multi-UAV" in MAIN is multi-base planning, not a swarm.** No class, variable, or file in MAIN represents a drone, a UAV ID, a fleet, or a swarm. The words "swarm", "UAV", "fleet", "drone_id" and "MAPF" appear only in docs (`README.md`, `Handoff.md`) and one docstring. The multi-vehicle capability works like this:
   - choose **N bases** (1 ≤ N ≤ `MAX_BASES` = 4), one UAV per base. The "one UAV per base" reading comes from `plan_base`'s docstring, *"Existing single-drone pipeline for one base"*.
   - **assign each drop point to exactly one base**,
   - **run the full single-drone pipeline independently per base**: connectivity filter → nearest-neighbour TSP → battery/payload multi-sortie planning → time-aware D* Lite → refinement loop.
   - **spatially deconflict** the resulting routes after the fact. If two bases' routes intersect, a 3-tier repair runs (reassign, then local reroute, then base nudge). If any crossing remains, the run exits with code 2.
   - export **one `.waypoints` file per base** (`base1.waypoints`, `base2.waypoints`, …) plus one combined map.
2. **There is no separate single-UAV code path in MAIN.** Single-UAV is the degenerate case: `select_bases` returns `mode="single"` with one base, and everything downstream runs with `len(plans) == 1`.
3. **The number of UAVs/bases is dynamic.** `select_bases` decides it automatically from battery reach, capped by `MAX_BASES` (environment variable, default 4). No CLI flag sets it.
4. **There is no MD5 or any other hashing anywhere.** The search covered the working tree and the full git history of all local and remote refs, with 0 matches. See §8.
5. **There is no simulator, MAVLink, SITL, or QGroundControl integration in code.** The only "mission" interface is a **file export** in QGC WPL 110 text format. "Simulation" in MAIN means the planner's internal battery/time model (`plan_mission_stops.simulate`).
6. **The porting implication:** the likely DEVELOPMENT ref (`origin/research-validation`) branched at `775165d`, before MAIN's single big commit `ca27215`. That commit added multi-base **and** the infrastructure multi-base depends on: `constraints.py` (battery model and home scoring), the routing-grid API in `pathfinding.py`, the time-indexed forecast API in `flood_spread.py`, connectivity filtering, reload-aware export, and session output. **Porting multi-base is not a 2-file copy.** It requires most of the `ca27215` mission-planning stack. See §17.

---

## 1. COMPLETE PROJECT INVENTORY

48 tracked files. There are no untracked files. `data/output/` does not exist in the working tree (gitignored), and the repository has no virtualenv.

```
.
├── .env.example                     config template (3 env vars)
├── .gitignore
├── Handoff.md                       authoritative engineering handoff (up to date with ca27215)
├── README.md                        OUTDATED: describes pre-D* Lite pipeline (Handoff §9 says so)
├── main.py                          CLI entry point + full orchestration (652 lines)
├── requirements.txt                 5 unpinned packages
├── debug_flood_spread.py            dev diagnostic script
├── dummy_path_test.png              output artefact of scripts/test_pathfinding_dummy.py (committed)
├── config/
│   ├── __init__.py                  "# Config package"
│   └── settings.py                  ALL tunable constants (71 lines)
├── data/input/
│   ├── assam.png  bhopal.png  kanpur.png  varanasi.png   flood maps (red flood overlay)
│   ├── dummy_weather.json            extreme preset  {precip 10, wind 30 km/h, dir 270}
│   └── dummy_weather_moderate.json   moderate preset {precip 3,  wind 12 km/h, dir 270}
├── src/
│   ├── __init__.py (empty)
│   ├── vision/      __init__.py, image_processing.py, clustering.py
│   ├── weather/     __init__.py, weather_api.py, flood_spread.py
│   ├── routing/     __init__.py, dstarlite.py, pathfinding.py
│   └── mission/     __init__.py, coordinates.py, safe_dropzone.py, constraints.py,
│                    mission_output.py, multi_base.py, overlap_repair.py
├── tests/           __init__.py + 10 unittest modules (84 tests)
├── scripts/test_pathfinding_dummy.py   dev plot (NOT a unit test despite the name)
└── scratch/                            3 throwaway debugging scripts
```

### 1.1 Per-file role table

Legend: **RT** = required at runtime by `main.py`; **Dev** = development/test only; **MU** = participates in multi-UAV (multi-base) functionality.

| File | Purpose | Imports (project) | Imported/called by | RT | Dev | MU |
|---|---|---|---|---|---|---|
| `main.py` | CLI, orchestration of the whole pipeline, per-base planning closures (`plan_base`, `plan_assigned`), territory partition, run summary | every `src.*` module, `config.settings` | user (CLI); tests import `create_session_dir`, `refine_route`, `map_routed_drop_distances`; `scratch/test_repair_on_172629.py` | ✔ | – | **✔ core** |
| `config/settings.py` | Constants + env overrides (`python-dotenv`) | – | `main`, `constraints`, `safe_dropzone`, `mission_output`, `overlap_repair`, `flood_spread`, `weather_api`, `pathfinding` (lazy) | ✔ | – | ✔ (`MAX_BASES`, `MIN_BASE_SEPARATION_PX`, `HOME_CLEARANCE_PX`) |
| `config/__init__.py` | package marker | – | – | ✔ | – | – |
| `src/vision/image_processing.py` | `ImageProcessor`: HSV sampling (interactive or auto), flood mask, contour filter | – | `main`, tests, `debug_flood_spread.py`, scratch | ✔ | – | – (shared upstream) |
| `src/vision/clustering.py` | `cluster_contours`: convex hull + centroid per contour | – | `main`, scratch | ✔ | – | – (shared upstream) |
| `src/weather/weather_api.py` | `get_weather_data`: Open-Meteo GET, `lru_cache`, fallback | `config.settings` | `main`, tests | ✔ (only without `--dummy-weather*`) | – | – |
| `src/weather/flood_spread.py` | `predict_spread` (time-indexed forecast frames), `forecast_at`/`obstacle_mask_at` (front interpolation) | `config.settings` | `main`, `constraints.plan_mission_stops` (lazy), `pathfinding.compute_full_path` (lazy), `overlap_repair.try_reroute_leg` (lazy), tests | ✔ | – | ✔ (used inside per-base planning and Tier-1 reroute) |
| `src/routing/dstarlite.py` | `DStarLite` grid planner | – | `pathfinding`, `overlap_repair`, tests, scratch | ✔ | – | ✔ (Tier-1 reroute instantiates it directly) |
| `src/routing/pathfinding.py` | NN-TSP, routing grid, component-aware snapping, distance fields, routed distance function, `compute_full_path`, `polyline_intersections` | `dstarlite`, `config.settings` (lazy), `flood_spread` (lazy) | `main`, `safe_dropzone` (lazy), `overlap_repair`, tests, scripts, scratch | ✔ | – | **✔** (`routed_distance_function`, `grid_distance_field`, `polyline_intersections`, `locate_cell` are multi-base critical) |
| `src/mission/coordinates.py` | `pixel_to_latlon` flat-earth conversion | – | `mission_output`, `overlap_repair` (imported, **unused**), tests | ✔ | – | indirect |
| `src/mission/safe_dropzone.py` | `find_safe_drop_points` (single/multi-point per zone), `filter_drops_by_home_connectivity` | `config.settings` (lazy), `pathfinding` (lazy) | `main`, tests, scratch | ✔ | – | ✔ (connectivity filter runs per base) |
| `src/mission/constraints.py` | Battery model, headwind, home-candidate scoring, `plan_mission_stops` (multi-sortie planner), `sortie_battery_report`, `PlannedRoute` | `config.settings`, `flood_spread` (lazy) | `main`, `multi_base`, `overlap_repair`, tests, scratch | ✔ | – | **✔** (`leg_battery_fraction` defines "reach"; `score_home_candidates` places bases) |
| `src/mission/mission_output.py` | QGC WPL 110 writer + validator; map rendering (single and multi-base, territories); `split_sorties`; `basemap_sanity` | `coordinates`, `config.settings` | `main`, `overlap_repair`, tests, scratch | ✔ | – | **✔** (multi-base map; per-base export; sortie splitting used by repair) |
| `src/mission/multi_base.py` | **Base placement + drop→base assignment** (`select_bases`, `single_drop_reach`) | `constraints` | `main`, `overlap_repair`, tests, scratch | ✔ | – | **✔ core** |
| `src/mission/overlap_repair.py` | **Airspace deconfliction**: reassign / reroute / nudge | `config.settings`, `coordinates`, `mission_output`, `constraints`, `multi_base`, `dstarlite`, `pathfinding`, `flood_spread` (lazy) | `main`, tests | ✔ | – | **✔ core** |
| `src/*/__init__.py` | empty package markers | – | – | ✔ | – | – |
| `tests/test_multi_base.py` | multi-base selection, intersections, distance fields, session dirs | `multi_base`, `pathfinding`, `main` | unittest | – | ✔ | **✔** |
| `tests/test_overlap_repair.py` | repair tiers | `overlap_repair`, `pathfinding`, `settings` | unittest | – | ✔ | **✔** |
| `tests/test_audit_regressions.py` | Oct-2026 audit regressions (rendering, palette, routing, drops, forecast, export, vision, refinement) | many | unittest | – | ✔ | partial (palette, `refine_route`) |
| `tests/test_mission_constraints.py` | battery / sortie / home scoring / drop placement | `constraints`, `mission_output`, `safe_dropzone`, `flood_spread` | unittest | – | ✔ | indirect |
| `tests/test_mission_output.py` | sortie colours, labels, `map_routed_drop_distances` | `mission_output`, `main` | unittest | – | ✔ | – |
| `tests/test_coordinates.py` | `pixel_to_latlon`, `REGION_PRESETS` | `coordinates`, `settings` | unittest | – | ✔ | – |
| `tests/test_dstarlite.py` | D* Lite, snapping, forecast selection | `dstarlite`, `pathfinding`, `flood_spread` | unittest | – | ✔ | – |
| `tests/test_flood_spread.py` | forecast frames + interpolation | `flood_spread` | unittest | – | ✔ | – |
| `tests/test_image_input.py` | trivial `Path.is_file` check | – | unittest | – | ✔ | – |
| `tests/test_weather_api.py` | Open-Meteo success/fallback with mocked `requests.get` | `weather_api` | unittest | – | ✔ | – |
| `scripts/test_pathfinding_dummy.py` | dev plot: TSP direct line vs D* Lite on a 100×100 dummy grid; writes `dummy_path_test.png` to CWD | `pathfinding` | manual | – | ✔ | – |
| `debug_flood_spread.py` | prints forecast frame stats for Varanasi | `image_processing`, `flood_spread` | manual | – | ✔ | – |
| `scratch/audit_western_cluster.py` | old single-home diagnostic (uses straight-line home scoring) | many | manual | – | ✔ | – |
| `scratch/debug_pixel.py` | prints rendered pixel colours | `mission_output` | manual | – | ✔ | – |
| `scratch/test_repair_on_172629.py` | reads Bhopal session 172629 waypoint files and counts crossings | many + `main` | manual (needs gitignored `data/output/session_20261004_172629/`) | – | ✔ | ✔ (debug) |
| `requirements.txt` | numpy, opencv-python, matplotlib, requests, python-dotenv (unpinned) | – | pip | ✔ | – | – |
| `.env.example` | `WEATHER_API_BASE_URL`, `MAX_SAFE_WIND_SPEED`, `MAX_SAFE_PRECIPITATION` | – | copied to `.env` manually | optional | – | – |
| `data/input/*.png` | input maps; `varanasi`/`kanpur`/`bhopal` have `REGION_PRESETS`; **`assam.png` has no preset** (needs `--lat/--lon`) | – | `main`, tests | ✔ (input) | – | – |
| `data/input/dummy_weather*.json` | weather presets | – | `main` (`--dummy-weather[-moderate]`) | ✔ (with flag) | – | – |
| `Handoff.md` | authoritative design notes, known limitations, audit history, multi-base description | – | humans | – | doc | ✔ (§8 describes multi-base) |
| `README.md` | **outdated** (pre-D* Lite, pre-multi-base; mentions DBSCAN, which is NOT implemented) | – | humans | – | doc | – |

**No** Dockerfile, `pyproject.toml`, `setup.py`, `environment.yml`, `package.json`, CI config, notebooks, ML models, datasets beyond the 4 PNGs, or deployment files exist.

### 1.2 Dead / unused code (verified by grep)
- `main.py`: `import re`, the `List, Tuple, Any` typing imports, and the `split_sorties` import (line 19) are unused in `main.py`.
- `overlap_repair.py`: `from src.mission.coordinates import pixel_to_latlon` is unused.
- `constraints.split_sorties(drop_count, per_drop_payload_kg, capacity_kg)` has **no callers**. It **shares its name** with `mission_output.split_sorties(full_path, drop_indices, reload_indices)`, which is a different function and is the one that is used.
- `constraints.plan_payload_batches` is only called by tests.
- `mission_output.draw_dashed_line` has no callers.
- `select_bases`' docstring mentions a `reach_counts` key, which is **not returned**.
- `settings.BATTERY_CAPACITY_WH`, `DRONE_RATED_PAYLOAD_KG` and `DRONE_MODEL_REFERENCE` are documentation-only constants with no readers.
- `clustering.cluster_contours` zero-moment branch `cx, cy = hull[0][0], hull[0][1]` would raise `IndexError` (hull shape is `(N,1,2)`). It is unreachable in practice because contours are filtered to area ≥ 200. DEVELOPMENT (`research-validation`) already fixed it; **keep DEVELOPMENT's version**.

---

## 2. COMPLETE ARCHITECTURE (actual)

```
CLI (main.py:main, argparse)
  │  --image | tkinter file picker      --lat/--lon | REGION_PRESETS[image stem]
  ▼
Geo anchor + SESSION dir  data/output/session_<YYYYmmdd_HHMMSS>[_n]/
  ▼
Weather:  dummy JSON  |  Open-Meteo get_weather_data(lat,lon)  (fallback = calm)
  ▼
Image load (cv2.imread) → resize to width 750 (INTER_AREA) if wider
  ▼
Flood segmentation  ImageProcessor
  sample points: --sample-points | auto_sample_points (--no-gui/--auto-sample) | mouse clicks
  compute_dynamic_hsv → mask_flood_areas (inRange [+ red wrap], 5×5 CLOSE+OPEN)
  ▼
MASK SANITY  0.1 % ≤ coverage ≤ 50 %   else exit 1
  ▼
Flood forecast  predict_spread(mask, weather, horizon=2h)
  → {0, 1/12, 1/6, 0.5, 1.0, 2.0 h: binary cumulative masks}
  combined_obstacle_mask = mask OR forecast[horizon]
  ▼
Zones  find_filtered_contours(area ≥ --min-area 200)  →  cluster_contours (hull, centroid)
  ▼
Drop points  find_safe_drop_points(hulls, centroids, obstacle_mask=combined, raw_contours)
  1 point per zone; 2–4 for zones with area ≥ 2500 px²; always dry at horizon
  ▼
Base candidates  ~40×40 spatial grid, dry in combined mask, distanceTransform clearance ≥ 15 px
  ▼
Routing grid  build_routing_grid(combined, factor 5)   routed_distance_function(grid, drops)
  ▼
████ MULTI-BASE LAYER ██████████████████████████████████████████████████████
█ select_bases → bases[], assignment{drop→base}, unreachable{drop→reason}
█   (single base if one site reaches all coverable drops, else greedy cover ≤ MAX_BASES)
█ for each base b:  plan_assigned(b) → plan_base(b+1, home, drops)   [single-drone pipeline]
█      filter_drops_by_home_connectivity → nearest_neighbor_tsp → refine_route(
█         plan_mission_stops (battery/payload/flood sorties)  ⇄  compute_full_path (time-aware D* Lite))
█      → compute_full_path(final) → sortie_battery_report → generate_mission_file base<N>.waypoints
█ execute_overlap_repairs  (Tier 0 reassign → Tier 1 reroute → Tier 2 nudge), ≤ 5 rounds
█ final pairwise polyline_intersections → any crossing ⇒ exit 2 (no map)
█ territory partition (multi-source grid_distance_field, argmin) + TERRITORY CHECK
████████████████████████████████████████████████████████████████████████████
  ▼
Combined map display_path_on_map(bases=[...], territories=...) → mission_route_map.png
                                                                  + mission_map_annotated.png
basemap_sanity → run_summary.json
```

**Stages that do NOT exist** (despite the README or the prompt template): DBSCAN or any learned clustering, ML/DL segmentation or forecasting (forecasting is a heuristic, not a trained model), 2-opt/VRP/MILP optimisation, MAVLink transport, SITL/Gazebo/QGC launching, GeoJSON/KML/CSV export, a database, a web UI or API.

---

## 3. MULTI-UAV / MULTI-BASE IMPLEMENTATION (critical)

### 3.1 Where it lives
| Concern | File:line | Symbol |
|---|---|---|
| Candidate base sites | `main.py:360-378` | inline in `main()` |
| Routed distance oracle | `src/routing/pathfinding.py:191` | `routed_distance_function` (uses `grid_distance_field` :161, `locate_cell` :128, `build_routing_grid` :90) |
| Per-site reach sets | `src/mission/multi_base.py:16` | `single_drop_reach` |
| Base selection + assignment | `src/mission/multi_base.py:47` | `select_bases` (helpers `_separated` :33, `_assign` :37) |
| Base re-placement scoring | `src/mission/constraints.py:42` | `score_home_candidates` (called through `distance_fn`) |
| Assignment table log | `main.py:402-416` | inline |
| Per-base planning | `main.py:418-509` | closure `plan_base(number, home, drops)` |
| Base → assigned drops | `main.py:511-519` | closure `plan_assigned(b)` |
| All bases planned | `main.py:521` | `plans_by_base = {b: plan_assigned(b) ...}` |
| Deconfliction | `src/mission/overlap_repair.py:292` | `execute_overlap_repairs` (+ `try_reroute_leg` :53, `try_nudge_base` :230, `sortie_drops` :44, `find_leg_index` :36) |
| Crossing detection | `src/routing/pathfinding.py:348` | `polyline_intersections` |
| Hard stop | `main.py:547-563` | inline, `sys.exit(2)` |
| Territories | `main.py:568-594` | inline (multi-source `grid_distance_field`) |
| Summary | `main.py:596-625`, `write_summary` :113 | inline |
| Multi-base map | `src/mission/mission_output.py:256` | `display_path_on_map(..., bases=, territories=)` |
| Per-base export | `src/mission/mission_output.py:32` | `generate_mission_file` → `session_dir/base<N>.waypoints` |

### 3.2 Data structures (actual; plain Python, no classes)
| Concept | Representation |
|---|---|
| **UAV / drone** | *Not represented.* A base (index `b`, 0-based; number `b+1` in logs, files and the map) implicitly has one drone. All drones share the global `settings` aircraft model (homogeneous fleet). |
| **UAV ID** | base number `b+1` → `"BASE n"` log tag, `base<n>.waypoints`, map label `BASE n`, legend `Bn Sortie k`. The waypoint files carry no MAVLink system ID. |
| **Home / base location** | `tuple[int,int]` pixel `(x, y)` in the *resized* image; list `bases = selection["bases"]` (mutated in place by Tier-2 nudge). |
| **Waypoint / drop point** | `tuple[int,int]` pixel; global list `safe_points`; **drop index** = position in `safe_points`. |
| **Task / cluster** | a drop index. Zones (contours) are only an upstream concept; assignment is per drop point, not per zone. A zone with 2–4 drops can be split across bases. |
| **Reach** | `list[set[int]]` parallel to `candidates`: drop indices servable by a single-drop sortie. |
| **Selection** | `dict`: `mode` ∈ {`"single"`,`"multi"`,`"none"`}, `bases: list[(x,y)]`, `assignment: dict[drop_idx → base_idx]`, `unreachable: dict[drop_idx → reason str]`, `best_single_reach: int`, `coverable: int`. |
| **Per-base plan** | `dict` from `plan_base`: `base` (1-based), `home`, `assigned`, `clearance_px`, `connectivity_excluded`, `refinement{status,iterations,cap}`, `full_path: list[(x,y)]` (all D* Lite cell centres), `drop_indices`, `reload_indices`, `stop_indices` (indices into `full_path`), `route_points` (planner stop list), `visited`, `unreachable`, `legs{total,dstar,same_cell,fallback}`, `route_flooded_px_arrival`, `route_flooded_px_current`, `planner_sortie_battery`, `routed_sortie_battery`, `waypoint_file`, `waypoint_problems`. `None` if planning failed. |
| **All plans** | `plans_by_base: dict[base_idx → plan | None]`; later `plans = [non-None plans sorted by base_idx]`. |
| **Route of one UAV** | `full_path` polyline plus `drop_indices` and `reload_indices`. Sorties are recovered by `mission_output.split_sorties` → `[{start, outward_end, rtb_end, drops}]`. |
| **Repair log** | `list[dict]`: `{"tier":0,"type":"reassign","drops":[...],"from_base":n,"to_base":n}`, `{"tier":1,"type":"reroute","base":n,"leg":i,"remaining_crossings":k}`, `{"tier":2,"type":"nudge","base":n,"from":(x,y),"to":(x,y)}`. |
| **Territory** | `np.int32` H×W label image: index into `plans` (**not** base number), `-1` = unreachable. |

### 3.3 How many UAVs, and how the number is chosen
- **Dynamic**: 1 … `settings.MAX_BASES` (env `MAX_BASES`, default 4).
- **Rule** (`select_bases`):
  1. `reach = single_drop_reach(candidates, drops, …)`; `coverable = ∪ reach`. Drops outside `coverable` are reported as unreachable.
  2. If some candidate reaches **all** coverable drops → `mode="single"`. The base is the minimum-cost site from `score_home_candidates` (routed distances) among those full-reach sites.
  3. Otherwise run the **greedy fewest-bases cover** (§7.11) with separation `MIN_BASE_SEPARATION_PX` (150 px). Each base is then re-placed at the minimum routed-cost valid site, bases left without drops are pruned, and the greedy pass repeats while drops are uncovered and bases < cap.
- Forcing single-UAV: set env `MAX_BASES=1`. Drops beyond that base's reach are then reported as "cap reached". No CLI switch exists.

### 3.4 How waypoints are divided (assignment)
`_assign(chosen, reach, candidates, drops, distance_fn)`: each coverable drop goes to the chosen base with the **lowest routed distance among bases whose reach set contains it**. Each drop goes to **exactly one** base. This is nearest-feasible-base assignment, effectively a battery-feasible routed Voronoi partition. There is no capacity or load balancing between bases.

The assignment can later change through **Tier-0 reassignment** in `execute_overlap_repairs`, which mutates `selection["assignment"]` in place.

### 3.5 Routes after assignment
Each base independently runs the **unchanged single-drone pipeline** (`plan_base`):
connectivity filter → NN-TSP from that base → `refine_route` loop (`plan_mission_stops` ⇄ `compute_full_path`) → final `compute_full_path` → `sortie_battery_report` → `generate_mission_file`.

- **Per-UAV optimisation**: yes; each base has its own TSP and D* Lite routing.
- **Global optimisation**: none jointly over routes. The only cross-base steps are placement (greedy cover + local re-placement scoring), assignment (nearest feasible) and post-hoc deconfliction.
- **Time**: each base's mission clock starts at t=0 (flood forecasts are queried from 0). This *implies* simultaneous launch, but nothing states it. **UNCLEAR — requires verification** of the intended operational concept. There is **no temporal deconfliction**: crossing checks are purely geometric and time-agnostic (stricter than needed).
- **UAV-specific constraints**: none. Every base uses the same `settings` aircraft parameters, battery model, payload, and the same weather/headwind.

### 3.6 Export and "simulation" of multi-UAV missions
- Export: N independent QGC WPL 110 files `session_dir/base1.waypoints … baseN.waypoints`. Each is a complete single-vehicle mission (home row, takeoff, path, drops, reload LAND/TAKEOFF, RTL, LAND). There is no combined multi-vehicle mission file and no vehicle/sysid binding.
- Repairs regenerate the affected files in place (same filename): Tier 0 through `plan_assigned`, Tier 1 inside `try_reroute_leg`, Tier 2 through `plan_base_fn`.
- Simulation: **none in code.** Files can be loaded manually into Mission Planner / QGroundControl (per the Handoff). There is no SITL, Gazebo, pymavlink, or MAVSDK.
- If an unresolved overlap remains, the run exits 2 **after** the waypoint files were already written. Those files stay in the session folder; `run_summary.json` records `path_overlaps`, and **no map is rendered**.

### 3.7 Single vs multi, explicitly

```text
SINGLE (select_bases → mode "single")         MULTI (mode "multi")
All drop points                                All drop points
   ↓ single_drop_reach (all candidates)           ↓ single_drop_reach
one site reaches all coverable drops           no single site does
   ↓ score_home_candidates(full-reach sites)      ↓ greedy cover ≤ MAX_BASES, separation 150 px
1 base; assignment = {i: 0 ∀ coverable}           ↓ _assign nearest-feasible  ↻ re-place (score_home_candidates) ×3
   ↓ plan_base(1, home, all assigned)          bases[0..N-1]; assignment{i: b}
   ↓ execute_overlap_repairs → no-op           per base: plan_base(b+1, bases[b], assigned_b)
   ↓ no territory, title "Multi-Sortie"           ↓ execute_overlap_repairs (≤5 rounds) → exit 2 if crossings remain
base1.waypoints                                   ↓ territories + TERRITORY CHECK
                                               base1..baseN.waypoints, title "Multi-Base"
```

---

## 4. MULTI-UAV DATA-FLOW TRACE (actual names)

```text
data/input/<map>.png ──cv2.imread──► image (H×W×3 BGR) ──resize w≤750──► image
  │
  ├─ImageProcessor.auto_sample_points(image) ─► proc.sample_points [(x,y)×3]
  │   .compute_dynamic_hsv() ─► proc.hsv_lower/upper, is_red_wrap
  │   .mask_flood_areas(image) ─► mask (uint8 0/255)
  │
  ├─predict_spread(mask, weather, 2.0) ─► forecasts {0.0,1/12,1/6,0.5,1.0,2.0: mask}
  │   pred_mask = forecasts[2.0];  combined_obstacle_mask = mask | pred_mask
  │
  ├─proc.find_filtered_contours(mask, 200) ─► contours
  │   cluster_contours(contours) ─► cluster_centers, cluster_edges (hulls)
  │   to_point_list(cluster_centers) ─► points
  │   find_safe_drop_points(cluster_edges, range(len(points)), points,
  │                         obstacle_mask=combined_obstacle_mask, raw_contours=contours)
  │        ─► safe_points  [(x,y)]   ← GLOBAL DROP LIST (index = drop id)
  │
  ├─grid sample stride=max(H,W)//40 ─► grid_points ─dry─► dry
  │   clearance = cv2.distanceTransform(combined==0)  ─► candidates (clearance ≥ 15)
  │
  ├─headwind = wind_kmh_to_mps(weather["wind_speed_10m"])
  │ routing_grid = build_routing_grid(combined_obstacle_mask, 5)          (RoutingGrid)
  │ routed_distance = routed_distance_function(routing_grid, safe_points, METERS_PER_PIXEL)
  │                    (one Dijkstra field per drop; distance(a,b) in metres)
  │
  ├─select_bases(candidates, safe_points, routed_distance, combined_obstacle_mask,
  │              headwind_mps, wind_from_deg, per_drop_kg, payload_capacity_kg,
  │              reserve, meters_per_pixel, max_bases, min_separation_px)
  │        ─► selection{mode, bases, assignment, unreachable, best_single_reach, coverable}
  │
  ├─plans_by_base = { b: plan_assigned(b) }            plan_assigned:
  │      assigned = [safe_points[i] for i,owner in assignment if owner==b]
  │      plan_base(b+1, bases[b], assigned):
  │         filter_drops_by_home_connectivity(drops, home, combined, 5) ─► drops, excluded
  │         nearest_neighbor_tsp(drops, home) ─► path_order ─► ordered_drops
  │         refine_route(plan, routed_distances, min(6, n+2)):
  │             plan(routed)  = plan_mission_stops(home, ordered_drops, forecasts, mask,
  │                              meters_per_pixel, headwind_mps, initial_battery=1.0,
  │                              routed_leg_distances_m=routed) ─► PlannedRoute
  │             routed_distances(route) = map_routed_drop_distances(route, ordered_drops,
  │                              compute_full_path(route.points, combined, forecast_masks=forecasts)[2], home)
  │             ─► route, status, iterations
  │         compute_full_path(route.points, combined, forecast_masks=forecasts, leg_report=[])
  │             ─► full_path, stop_indices, _
  │         drop_indices/reload_indices = stop_indices[route.drop_indices / reload_indices]
  │         sortie_battery_report(full_path, drop_indices, reload_indices, …) ─► sorties
  │         generate_mission_file(full_path, drop_indices, home, (W//2,H//2), (lat,lon),
  │                               METERS_PER_PIXEL, filename=session/base<n>.waypoints, reload_indices)
  │         validate_waypoints_file(...) ─► problems
  │         ─► plan dict
  │
  ├─coord_index = {safe_point → drop id};  base_reach = single_drop_reach(bases, safe_points, …)
  │ execute_overlap_repairs(plans_by_base, bases, selection, safe_points, base_reach, coord_index,
  │     combined, forecasts, clearance, routed_distance, headwind, plan_base, plan_assigned,
  │     session_dir, image.shape[:2], (lat,lon), max_rounds=5)   ─► repairs, remaining
  │     (mutates plans_by_base, selection["assignment"], bases)
  │
  ├─plans = non-None plans; pairwise polyline_intersections ─► overlaps → exit 2 if any
  ├─territory: fields = [grid_distance_field(routing_grid, cells(home + visited drops)) per plan]
  │            labels = argmin → resize ×5 (NEAREST) → pad ─► territory_labels
  ├─summary{...} ; display_path_on_map(image, contours, pred_mask, plans[0]…, bases=[…], territories)
  │            ─► mission_route_map.png (matplotlib, dpi 160) + mission_map_annotated.png (cv2)
  └─basemap_sanity(rendered, image) ─► summary["map"] ; write_summary ─► run_summary.json
```

---

## 5. SINGLE-UAV vs MULTI-UAV ARCHITECTURE

```text
                        COMMON PIPELINE (always runs)
  CLI/geo anchor/session → weather → image → ImageProcessor → MASK SANITY
  → predict_spread → contours → cluster_contours → find_safe_drop_points
  → base candidates (grid+clearance) → build_routing_grid → routed_distance_function
                                   │
                         select_bases  (MU-specific; also decides single)
                     ┌─────────────┴──────────────┐
               mode "single"                mode "multi"
                     │                            │
                     └────────► plan_base ◄───────┘   (common single-drone pipeline, per base)
         filter_drops_by_home_connectivity, nearest_neighbor_tsp, refine_route,
         plan_mission_stops, compute_full_path(DStarLite, forecasts), sortie_battery_report,
         generate_mission_file, validate_waypoints_file
                                   │
             execute_overlap_repairs (MU-specific; no-op for 1 base)
             pairwise polyline_intersections / exit 2 (MU-specific)
             territories + TERRITORY CHECK (MU-only, len(plans) > 1)
                                   │
             display_path_on_map(bases=…, territories=…)  (handles both)
             run_summary.json (multi-base schema in both modes)
```

| Category | Code |
|---|---|
| **Multi-UAV-specific** | `src/mission/multi_base.py` (all), `src/mission/overlap_repair.py` (all), `pathfinding.polyline_intersections`, `pathfinding.grid_distance_field` (multi-source form for territories), `main.py` lines 360-625 (candidates, selection, assignment table, `plan_assigned`, repairs, overlap exit, territories, aggregated summary), the multi-base branches of `display_path_on_map` (`bases`, `territories`, numbering, `Bn` legend, title), settings `MAX_BASES` and `MIN_BASE_SEPARATION_PX` |
| **Shared, but introduced in the same commit and required by multi-base** | `constraints.py` (all), `pathfinding.build_routing_grid/cell_of/cell_center/locate_cell/routed_distance_function/polyline_flood_pixels/compute_full_path(new)`, `flood_spread` time-indexed API, `safe_dropzone.filter_drops_by_home_connectivity` and `obstacle_mask` param, `mission_output` (`reload_indices`, `validate_waypoints_file`, `split_sorties`, palette), `main.refine_route`, `map_routed_drop_distances`, `create_session_dir`, `write_summary`, settings `HOME_CLEARANCE_PX` + battery/payload constants |
| **Single-UAV-specific** | none in MAIN. Single mode is `select_bases` returning 1 base, plus the `multi=False` cosmetics in `display_path_on_map` (title "Multi-Sortie", label "BASE", legend "Sortie k"). |
| **Upstream common (unchanged semantics)** | `ImageProcessor` (+ new `auto_sample_points`), `cluster_contours`, `get_weather_data`, `DStarLite`, `pixel_to_latlon` |

---

## 6. EVERY IMPORTANT FUNCTION / CLASS

### 6.1 `src/mission/multi_base.py`

**Function:** `single_drop_reach(candidates, drops, distance_fn, *, headwind_mps, per_drop_kg, reserve)` (`:16`)
- Purpose: for each candidate site, the set of drop indices a one-drop sortie can serve.
- Inputs: `candidates: list[(x,y)]`, `drops: list[(x,y)]`, `distance_fn(a,b)→metres (may be inf)`, scalars.
- Output: `list[set[int]]` parallel to `candidates`.
- Algorithm: `d = distance_fn(c, drop)`; skip non-finite; served iff `leg_battery_fraction(d, per_drop_kg, headwind) + leg_battery_fraction(d, 0, headwind) ≤ 1 − reserve`. Outbound loaded, return empty, the *scalar* headwind on both legs (worst case).
- Called by: `select_bases`, `main` (`base_reach`), `overlap_repair.try_nudge_base`. Calls: `constraints.leg_battery_fraction`.
- Assumes the same distance both directions. Side effects: none. Cost: O(|candidates|·|drops|) distance lookups.

**Function:** `_separated(point, others, min_separation_px)` (`:33`): Euclidean pixel distance ≥ min to all others.

**Function:** `_assign(chosen, reach, candidates, drops, distance_fn)` (`:37`): `{drop i → b}` minimising `distance_fn(candidates[chosen[b]], drop)` over bases with `i ∈ reach[chosen[b]]`. Ties are broken by lower `b` (tuple min).

**Function:** `select_bases(candidates, drops, distance_fn, obstacle_mask, *, headwind_mps, wind_from_deg, per_drop_kg, payload_capacity_kg, reserve, meters_per_pixel, max_bases, min_separation_px, refine_rounds=3)` (`:47`)
- Purpose: decide the number and positions of bases and assign drops.
- Output: `selection` dict (§3.2). `mode="none"` with empty `bases` when nothing is coverable.
- Algorithm: §7.11.
- Inner `score(positions, drop_indices)` → `score_home_candidates(..., payload_kg=0, return_details=True, distance_fn=distance_fn)["scored"]`, i.e. a list of `(hours + total_drain, (x,y), hours, max_sortie_battery)` ascending.
- Called by: `main.py:384`. Calls: `single_drop_reach`, `_separated`, `_assign`, `score_home_candidates`.
- Logs `BASE PLANNING:` lines. Note: `score_home_candidates` uses the **directional** headwind (`wind_from_deg` given), while `single_drop_reach` uses the **scalar worst-case** headwind. The inconsistency is real; it is not a documentation error.

### 6.2 `src/mission/overlap_repair.py`

**Function:** `find_leg_index(stop_indices, seg_index)` (`:36`): the leg `i` with `stop_indices[i] ≤ seg < stop_indices[i+1]`; otherwise the last leg.

**Function:** `sortie_drops(plan, segment, coord_index)` (`:44`): the global drop ids (via `coord_index`) of the sortie (from `mission_output.split_sorties`) whose `[start, rtb_end)` contains `segment`; `set()` if none.

**Function:** `try_reroute_leg(plan, other_plan, segs, combined_obstacle_mask, forecasts, headwind, session_dir, image_shape, geo_anchor, other_bases_plans=None)` (`:53`), Tier 1
- Inputs: two plan dicts; `segs = [(seg_in_plan, seg_in_other)]` from `polyline_intersections(..., return_segments=True)`.
- Output: `(updated_plan, leg_idx, new_hits)` or `None`.
- Algorithm, per crossing leg of `plan`:
  1. Collect the other plan's conflicting legs. Sample their polyline every ≤2 px → `pts_to_avoid`.
  2. Estimate arrival time from the elapsed path length / `DRONE_SPEED_MPS` → `leg_mask = obstacle_mask_at(forecasts, t, combined)`.
  3. `grid = build_routing_grid(leg_mask, 5)`. Locate start/goal via `locate_cell` in HOME's component.
  4. Soft obstacles = conflict cells (+ 1-cell buffer). Plan with `DStarLite(...).plan_path(grid.obstacles | buffered)`; if that fails, retry without the buffer.
  5. Reject if: no path, the routed leg touches flood (`polyline_flood_pixels>0`), any sortie exceeds `1-BATTERY_RESERVE_FRACTION` (`sortie_battery_report`), crossings with `other_plan` are not reduced, or a new crossing with any third base appears.
  6. Splice the new leg into `full_path`. Shift `stop_indices`/`drop_indices`/`reload_indices` by `delta`. **Rewrite** `session_dir/base<N>.waypoints`.
- Side effects: writes the waypoint file. `route_points` is **not** updated.
- Uses `image_shape` → centre `(W//2, H//2)` and `settings.METERS_PER_PIXEL` (**second georeference call-site**, see §17.H).

**Function:** `try_nudge_base(b, plans_by_base, selection, safe_points, clearance, routed_distance, headwind, plan_base_fn, image_shape, min_separation_px=None)` (`:230`), Tier 2
- Candidate offsets: rings r = 5…50 px, step 5 px, inside a disc, sorted by distance.
- Accept the first site that: is inside the image; has `clearance ≥ HOME_CLEARANCE_PX`; is ≥ separation from the other plans' homes; reaches all assigned drops (`single_drop_reach`); gets a non-`None` plan from `plan_base_fn(b+1, site, assigned)`; and whose new route crosses no other base.
- Output: `((x,y), new_plan)` or `None`. Side effect: `plan_base_fn` writes `base<N>.waypoints`. Each attempt re-runs the whole per-base pipeline, so this is expensive.
- `routed_distance` was built with fields only for drops, so `distance(newsite, drop)` uses the drop's field and is valid.

**Function:** `execute_overlap_repairs(plans_by_base, bases, selection, safe_points, base_reach, coord_index, combined_obstacle_mask, forecasts, clearance, routed_distance, headwind, plan_base_fn, plan_assigned_fn, session_dir, image_shape, geo_anchor, max_rounds=5)` (`:292`)
- Loop ≤ `max_rounds`: find all crossing pairs among live plans and take the **first** pair.
  - **Tier 0**: for each crossing segment pair, a move is possible if the crossing sortie's drop set ⊆ the other base's `base_reach`. Choose `min(moves)` (fewest drops, then lexicographic), reassign in `selection["assignment"]`, and re-plan **both** bases via `plan_assigned_fn`.
  - **Tier 1**: `try_reroute_leg(first around second)`, then `(second around first)`.
  - **Tier 2**: `try_nudge_base(first)`, then `(second)`; this mutates `bases[k]`.
  - Otherwise log an error and `break`.
- Returns `(repairs, find_crossings())`. `main` ignores the second value and recomputes overlaps itself.
- Caveats (observed in code): `base_reach` is computed once for the original bases and is **not recomputed after a Tier-2 nudge**. The `ASSIGNMENT TABLE` and `BASE SEPARATION` logs are printed **before** repairs and can be stale; `run_summary.json` has the final assignment. If a re-plan returns `None`, that base disappears from `plans`, but its drops remain "assigned" in the summary counts.

### 6.3 `main.py`

| Function | Line | Purpose / IO | Notes |
|---|---|---|---|
| `show_image_safe(title, img, wait)` | 32 | cv2 window, normalises dtype | GUI only |
| `to_point_list(cluster_centers)` | 59 | Nx2 → `[(int,int)]` | |
| `map_routed_drop_distances(route, ordered_drops, leg_distances, home)` | 72 | Converts per-leg D* Lite lengths into keys for `_routed_distance`: `("pair", a, b)` → m, plus legacy `drop_id` and `("home", drop_id)` | tested |
| `create_session_dir(root="data/output")` | 101 | `session_YYYYmmdd_HHMMSS`, suffix `_n` if it exists | **Race:** concurrent runs in the same second crash with `FileExistsError` (check-then-`mkdir`; observed in this analysis). |
| `write_summary(summary, session_dir)` | 113 | JSON dump, `default=str` | |
| `refine_route(plan, routed_distances, max_iterations)` | 122 | Alternates the planner and the router until the stop list repeats. Returns `(route, "converged" | "cycle" | "iteration_cap", n)`. Routed distances accumulate in `known`. | tested |
| `main()` | 147 | everything; closures `plan_base` (418) and `plan_assigned` (511) | exit codes: 0 ok / no zones / no drops; 1 bad image, mask sanity, sampling failure; 2 no candidates / no base / no plans / unresolved overlap |

`plan_base(number, home, drops)` (closure; it captures `clearance`, `combined_obstacle_mask`, `forecasts`, `mask`, `headwind`, `session_dir`, `image`, `args`):
`max_iterations = min(6, len(ordered_drops)+2)`. It returns `None` if connectivity leaves no drops or the planner yields no drops. Waypoint generation failures are captured into `waypoint_problems` and are not raised.

### 6.4 `src/mission/constraints.py`

| Function | Line | Summary |
|---|---|---|
| `leg_battery_fraction(distance_m, payload_kg=0, headwind_mps=0)` | 8 | `t_min = d / max(DRONE_SPEED_MPS, .1) / 60 × (1 + payload·0.08 + headwind·0.025)`; returns `t_min / 7.0` (fraction of the loaded endurance) |
| `wind_kmh_to_mps(kmh)` | 17 | `max(0, kmh) / 3.6` |
| `headwind_component_mps(w, from_deg, start, end)` | 22 | heading = `atan2(dx, -dy)` (image y down → compass); `max(0, w·cos(from − heading))` |
| `split_sorties(drop_count, per_drop, cap)` | 33 | **unused**; payload batching |
| `score_home_candidates(candidates, drops, obstacle_mask, meters_per_pixel=2.0, speed_mps=None, payload_kg=0, headwind_mps=0, per_drop_payload_kg=None, payload_capacity_kg=None, wind_from_deg=None, return_details=False, distance_fn=None)` | 42 | Per candidate: skip if flooded; NN order (by `distance_fn`); greedy sorties = longest prefix (≤ capacity) whose loaded outbound + RTB ≤ 0.8; a single drop that fails alone is "unreachable". `scored`: `(hours + Σdrain, cand, hours, max_sortie)`; candidates with any unreachable drop go to `rejected`. Returns the sorted `scored` list or a details dict (`sampled, dry_candidates, evaluated, rejected, scored, sortie_metrics`). |
| `plan_payload_batches` | 202 | test-only |
| `log_deviation(msg)` | 211 | `WARNING "FORCED MISSION DEVIATION: …"` |
| `PlannedRoute` (dataclass) | 215 | `points, drop_indices, reload_indices, unreachable, remaining_battery, deviations, sortie_battery` |
| `_routed_distance(routed, home, start, end, drop_id, leg_number, straight_m)` | 226 | Lookup order: `("pair", start, end)`, then legacy keys (only if no pair keys exist), then list form, then the straight-line distance. |
| `plan_mission_stops(home, drop_points, forecasts, current_mask, *, capacity, per_drop_kg, meters_per_pixel, speed_mps, headwind_mps, initial_battery, reserve, routed_leg_distances_m)` | 250 | Multi-sortie planner (§7.8). Returns `PlannedRoute`; `points` contain HOME at reloads (twice: arrival and re-departure). |
| `sortie_battery_report(full_path, drop_indices, reload_indices, *, meters_per_pixel, per_drop_kg, headwind_mps)` | 434 | Authoritative post-routing per-sortie battery on the exported geometry: `[{start_index, end_index, drops, distance_m, battery_fraction}]` |

### 6.5 `src/routing/pathfinding.py`

| Function | Line | Summary |
|---|---|---|
| `nearest_neighbor_tsp(points, home=None)` | 8 | Greedy NN from home (Euclidean). Returns `(order, [home] + ordered + [home])`. |
| `snap_to_nearest_free_cell(point, obstacles, w, h, max_search_radius=15, target_component=None, labels=None, accept=None)` | 38 | 8-neighbour BFS; prefers the target component; falls back to any free cell |
| `RoutingGrid` (namedtuple) | 87 | `occupancy, obstacles(set of (cx,cy)), labels(cv2 connectedComponents, 8-conn), factor, width, height, mask` |
| `build_routing_grid(mask, factor=5)` | 90 | Dilate 3×3, then block max-pool (a cell is blocked if **any** pixel is flooded); free-space components |
| `cell_of(pt, grid)` / `cell_center(cell, grid)` | 116/122 | px ↔ cell; centre = `cell·f + f//2` |
| `locate_cell(point, grid, target_component=None, max_search_radius=4)` | 128 | A free cell keeps its own component (never pulled across walls). A blocked cell snaps ≤ 4 cells to a target-component cell whose straight connector crosses no flood pixel. Returns `(cell, component or 0, snap_dist)`. |
| `grid_distance_field(grid, source_cell or list)` | 161 | Dijkstra over 8-connected free cells (1 / √2 costs); multi-source → min |
| `routed_distance_function(grid, targets, mpp)` | 191 | Precomputes one field per target. `distance(a,b)` = field of b at cell(a) (or the reverse) × factor × mpp; straight line if neither is a target; **inf** if unreachable |
| `polyline_flood_pixels(points, mask)` | 224 | Count of flooded pixels under a 1-px rasterised polyline |
| `compute_full_path(ordered_points, obstacle_mask, downsample_factor=5, *, forecast_masks=None, meters_per_pixel=2.0, speed_mps=None, leg_report=None)` | 234 | Per leg: arrival estimate → `obstacle_mask_at` → grid → `locate_cell` (HOME component) → D* Lite → validate. On failure, a **logged straight-line fallback**. Returns `(full_path, stop_indices, leg_distances_m)`. Elapsed time uses the routed length. |
| `polyline_intersections(a, b, return_segments=False)` | 348 | Vectorised segment-segment intersection incl. collinear overlap; zero-length (repeated home) segments ignored; returns points (rounded 0.1) or `(i, j)` segment pairs in original indexing |

### 6.6 `src/routing/dstarlite.py`
**Class `DStarLite(grid_shape=(rows, cols), start, goal)`**: state `g`, `rhs` dicts, heap `U` + `U_set` (lazy deletion), `km=0`, `obstacles` set. Chebyshev heuristic, Euclidean edge costs, 8-connected. `plan_path(obstacles)` adds obstacles incrementally (`update_vertex` on them and their neighbours), runs `compute_shortest_path`, then greedily descends `cost + g` from start to goal; returns `[]` if `g[start] = inf`. **Replanning with `km`/start movement is not used**: every leg creates a new instance (`compute_full_path`, `try_reroute_leg`). It is used as an incremental A*-equivalent, not for in-flight replanning (Handoff §6 confirms).

### 6.7 `src/mission/safe_dropzone.py`
- `find_safe_drop_points(clustered_contours, path_order, path_points, obstacle_mask=None, raw_contours=None, large_contour_area_threshold=None, large_contour_area_step=None, max_drops_per_contour=None, min_drop_separation_px=None)` (`:25`), §7.4. Output: `list[(x,y)]`, flat (the zone of origin is not kept).
- `filter_drops_by_home_connectivity(drop_points, home, obstacle_mask, downsample_factor=5)` (`:176`) → `(reachable_points, excluded_indices)`, using the same `build_routing_grid`/`locate_cell` as D* Lite.

### 6.8 `src/weather/flood_spread.py`
- `predict_spread(mask, weather_data, horizon_hours, timesteps=(1/12,1/6,0.5,1,2))` → `dict[float, uint8 mask]`, cumulative and binary (§7.3).
- `_bracket`, `interpolate_forecast_mask` (diagnostic grey blend), `signed_distance`, `interpolate_flood_front` (SDF interpolation), `forecast_at(forecasts, t, current) → (mask, details)`, `obstacle_mask_at(...) → mask`.

### 6.9 `src/mission/mission_output.py`
- `generate_mission_file(full_path, drop_indices, home, image_center_px, geo_center, meters_per_pixel, filename, base_altitude=100, drop_altitude=10, reload_indices=None)` (`:32`). Row format in §11.
- `validate_waypoints_file(filename) → list[str]` (`:93`): header, 12 fields, sequential seq, `current` only on row 0, frame ∈ {0,3}, known commands, lat/lon range and not (0,0), autocontinue=1, `DO_SET_SERVO` servo ≥ 1 and PWM 800–2200.
- `split_sorties(full_path, drop_indices, reload_indices)` (`:240`) → sorties with drops only.
- `display_path_on_map(image, contours, pred_mask, full_path, drop_indices, home=None, reload_indices=None, save_path=None, show=True, bases=None, territories=None)` (`:256`) → returns the cv2-annotated image `vis`. It saves the matplotlib figure to `save_path` and `mission_map_annotated.png` beside it.
- `basemap_sanity(rendered, source, max_dominant_fraction=0.5)` → `(ok, details)`; requires grayscale correlation ≥ 0.5.
- Constants: `MAV_CMD_*`, `MAV_FRAME_*`, `SORTIE_PALETTE_HEX` (12 colours), `CONTOUR_GREEN_BGR`, `FORECAST_BROWN_BGR`, `HOUSE_OUTLINE`, `DROP_POINT_BGR`.

### 6.10 Vision / weather / coordinates
- `ImageProcessor` (class, `image_processing.py:4`). State: `sample_points, hsv_lower, hsv_upper, is_red_wrap, image`. Methods: `select_sample_points` (cv2 window, 'c' to finish), `mouse_click`, `auto_sample_points(image, count=3, …)`, `compute_dynamic_hsv(sample_radius=5, hue_margin=20, sv_margin=30)`, `mask_flood_areas`, `find_filtered_contours(mask, min_area=50)`. Created by `main`; used by `main`, tests, debug scripts.
- `cluster_contours(contours) → (centers, hulls)`.
- `get_weather_data(lat, lon)` (`@lru_cache(maxsize=32)`) → `{precipitation, wind_speed_10m (km/h), wind_direction_10m, visibility, status: success|fallback}`.
- `pixel_to_latlon(px, py, (cx,cy), (lat,lon), mpp)` → `(lat + Δy·mpp/111320, lon + Δx·mpp/(111320·cos lat))`, with y inverted. MAIN has **no input validation**; DEVELOPMENT has a hardened version.

---

## 7. ALGORITHMS (only those actually implemented)

| # | Name | Where | Input → Output | Notes / why |
|---|---|---|---|---|
| 7.1 | Dynamic HSV thresholding + auto sampling | `ImageProcessor.compute_dynamic_hsv`, `auto_sample_points`, `mask_flood_areas` | image (+ sample pts) → binary mask | Min/max over 11×11 patches ± margins (H ±20, S/V ±30). Red wrap: if `lower_h ≤ 10` or `upper_h ≥ 170`, the hue range becomes the fixed [0,10] ∪ [170,179]. Auto sampling takes the dominant saturated hue (circular histogram ±6), then picks pixels with patch density ≈ 0.5, ≥ 40 px apart. Then 5×5 MORPH_CLOSE + MORPH_OPEN. |
| 7.2 | Contour extraction + "clustering" | `find_filtered_contours`, `cluster_contours` | mask → contours (area ≥ 200), hulls, centroids | `RETR_EXTERNAL`; moments centroid. **Not** DBSCAN or k-means: one "cluster" per contour. |
| 7.3 | Weather-conditioned flood spread | `predict_spread` | mask + weather → time-indexed frames | growth px = precip/5 · t · 1.0; elliptical dilation of radius ⌊g⌋ plus a fractional next ring; wind shift = wind/10 · t · 1.0 px downwind (direction + 180°), OR-ed; threshold 128; cumulative OR with the previous frame. A heuristic, not a learned model. |
| 7.4 | Front interpolation | `forecast_at` / `interpolate_flood_front` | frames, t → binary mask | Bracket frames; interpolate signed distance fields, `(1-α)·SDF0 + α·SDF1 < 0`. Midpoint switch if there is no front. |
| 7.5 | Safe drop point selection | `find_safe_drop_points` | hulls/contours + obstacle mask → drop pts | Dry band: dilate the contour polygon by r = 1… up to `min(20, ceil(0.03·max(H,W)))`; keep the first ring that has dry pixels. Large zone (area ≥ 2500): `target = min(4, 2 + ⌊(area-2500)/3000⌋)` anchors at equal perimeter arc length; for each anchor, the nearest dry pixel in the **full band** ≥ 40 px from the zone's other points. Small zone: the dry candidate nearest to the polyline through *cluster centroids* (`point_line_distance`; legacy behaviour, the path is the centroid list in contour order). |
| 7.6 | Connectivity pre-filter | `filter_drops_by_home_connectivity` | drops, home, mask → kept, excluded | Same grid and `locate_cell` as D* Lite, so filter and router agree (tested on 25 random maps). |
| 7.7 | Nearest-neighbour TSP | `nearest_neighbor_tsp` | home + drops → order | Greedy O(n²), Euclidean in pixels. No 2-opt. Used per base (main) and inside `score_home_candidates` (with `distance_fn`). |
| 7.8 | Battery/payload/flood multi-sortie planner | `plan_mission_stops` | home, TSP-ordered drops, forecasts → `PlannedRoute` | (1) Pre-pass: estimate the arrival at each drop in order; drops flooded at that estimate are moved to the front (once). (2) Greedy: for each drop, if adding it exceeds payload capacity, commit the sortie and reload. Simulate `sortie + [drop]` from HOME and back (payload decreasing, scalar headwind, routed distances if known). If remaining − total < reserve: if the drop alone fails from full → unreachable; else commit + reload and simulate alone. If it still fails: reload if not full, else unreachable. (3) Arrival flood check at the simulated arrival time (`forecast_at`); flooded → unreachable. (4) Commit appends drops then HOME. |
| 7.9 | Planner ⇄ router refinement | `main.refine_route` | plan fn, routed fn → converged route | Fixed-point iteration on the stop list (≤ min(6, n+2)); distances keyed by exact leg; cycle detection. |
| 7.10 | Time-aware D* Lite routing | `compute_full_path` + `DStarLite` | stop list, mask, forecasts → dense path | Per leg: arrival time from the elapsed **routed** length / 5 m/s → obstacle mask at that time → 5× downsampled grid (any-pixel blocked, 1-px dilation) → D* Lite on 8-connected cells. Validated: complete, not through obstacles. Otherwise an explicit straight-line fallback with the flood crossing count. |
| 7.11 | **Multi-base placement (greedy set cover + local re-placement)** | `select_bases` | candidates, drops, routed distance → bases, assignment | `reach` = single-drop battery feasibility on routed distances. If one site covers all → single (min `score_home_candidates`). Else repeat ≤ `max_bases` passes: **greedy**: add the separated site maximising (\|reach ∩ uncovered\|, −Σ routed dist) until covered or capped; **assign** each drop to the nearest feasible chosen base; **re-place** (≤ 3 rounds): for each base, among sites that reach all its drops and stay separated from the others, take the min routed-cost site (`score_home_candidates`); re-assign; **prune** bases without drops; stop if covered or no change. Leftovers get a reason (cap reached / separation). Greedy is an approximation (Handoff §9 states it is not a proven optimum). |
| 7.12 | Nearest-feasible assignment | `_assign` | → `{drop: base}` | Routed-distance argmin among bases that can reach the drop. |
| 7.13 | Segment intersection test | `polyline_intersections` | 2 polylines → hits | Parametric cross-product test, vectorised O(n·m); collinear overlap handled. |
| 7.14 | 3-tier deconfliction | `execute_overlap_repairs` | plans → repaired plans | Reassign sortie → soft-obstacle D* Lite local reroute (buffered, then exact) with flood/battery/no-new-crossing checks → base nudge (≤ 50 px, rings of 5 px) with full re-plan → hard error. |
| 7.15 | Territory partition (routed Voronoi) | `main.py:573-594` | final plans → label image | Multi-source `grid_distance_field` from each base's {home + visited drops}; argmin per cell; `TERRITORY CHECK` verifies each visited drop is in its own base's cell. Diagnostic and visual only; it does not affect planning. |
| 7.16 | Battery model | `leg_battery_fraction` | distance, payload, wind → fraction | Time-budget linear model on 7 min loaded endurance, 5 m/s; +8 %/kg, +2.5 %/(m/s headwind). |
| 7.17 | Headwind component | `headwind_component_mps` | → m/s | Used only in `score_home_candidates`. Planner and reach use the scalar worst case. |
| 7.18 | Coordinate conversion | `pixel_to_latlon` | px → lat/lon | Flat-earth, 111 320 m/deg, cos(lat) for longitude. |
| 7.19 | Base candidate sampling | `main.py:361-367` | → candidates | Stride = max(H,W)/40 grid; dry; `distanceTransform` (L2, 5) clearance ≥ 15 px. |
| 7.20 | Map sanity | `basemap_sanity` | rendered vs source → ok | Dominant colour ≤ 50 % and grayscale correlation ≥ 0.5. |

**Not implemented:** VRP/mTSP solver, MILP, auction/market allocation, k-means/DBSCAN, 2-opt, energy (Wh) model, MAPF or time-indexed conflict resolution, in-flight replanning, obstacle avoidance beyond flood masks, altitude planning (fixed 100 m cruise / 10 m drop).

---

## 8. MD5 / HASHING: investigated thoroughly

**Result: MD5 is NOT used anywhere in this project. No hashing of any kind is implemented.**

Searches performed:
- Working tree (all 48 files): `md5`, `MD5`, `hashlib`, `sha1`, `sha256`, `checksum`, `digest`, `hash(`, and the word `hash` (case-insensitive) → **0 matches**.
- Entire git history of **all refs** (`git log --all -p`, including `origin/research-validation`, `ajay`, `time-aware-routing`, `backup`, `routing-algo`): `md5|hashlib` → **0 matches**.
- `origin/research-validation` source (excluding PDF/bib): `md5|sha1|sha256|hashlib|checksum` → **0 matches**.

Consequently there is no MD5 definition, import, call site, hashed data, encoding, stored hash, comparison, API, config, filename, or CLI usage to document.

Mechanisms that could be *mistaken* for hashing or integrity checks:
| Mechanism | Where | What it actually is |
|---|---|---|
| `@lru_cache(maxsize=32)` | `weather_api.get_weather_data` | in-memory memoisation keyed by the `(lat, lon)` float args (Python's internal hashing of the args, not content hashing) |
| `fields` / `cells` dicts | `routed_distance_function` | per-run caches keyed by integer pixel tuples |
| `session_<timestamp>` dirs | `create_session_dir` | uniqueness by timestamp plus a `_n` suffix |
| `validate_waypoints_file` | `mission_output` | structural format validation, not a checksum |
| `basemap_sanity` | `mission_output` | statistical image check (dominant colour, correlation) |

```text
MD5 flow:  (none)  — Input ✗ → MD5 generation ✗ → value ✗ → storage/comparison ✗
```
If DEVELOPMENT or the paper mentions MD5, that content does not originate in MAIN. **UNCLEAR — requires verification** in DEVELOPMENT itself.

---

## 9. DEPENDENCIES

| Kind | Item | Used where |
|---|---|---|
| Language | Python 3 (tested here on **3.11.9**) | all |
| Dependency file | `requirements.txt` (unpinned): `numpy`, `opencv-python`, `matplotlib`, `requests`, `python-dotenv` | – |
| numpy | arrays everywhere | all modules |
| opencv-python (`cv2`) | imread/resize, HSV, inRange, morphology, contours, hull, moments, dilate, distanceTransform, connectedComponents, warpAffine, drawing, GUI windows | vision, flood_spread, pathfinding, safe_dropzone, mission_output, overlap_repair, main |
| matplotlib | final map figure, legend, arrows (`FancyArrowPatch`); `plt.show` unless `--no-gui` | mission_output, scripts; tests force the `Agg` backend |
| requests | Open-Meteo HTTP | weather_api |
| python-dotenv | `load_dotenv()` | config/settings |
| stdlib | argparse, logging, json, pathlib, datetime, heapq, collections (deque, namedtuple), dataclasses, functools.lru_cache, math, typing, unittest(+mock), tempfile, **tkinter** (optional file picker when `--image` is omitted) | – |
| Versions verified working in this analysis | numpy 2.4.6, opencv-python 5.0.0, matplotlib 3.11.2, requests 2.34.2 | – |
| External API | Open-Meteo `https://api.open-meteo.com/v1/forecast` (no key) | weather_api |
| **Absent** | pymavlink, dronekit, MAVSDK, SITL/Gazebo, QGC SDK, scipy, sklearn, torch/tensorflow, OR-Tools, Node, Docker | – |

---

## 10. CONFIGURATION

### 10.1 Sources
1. `config/settings.py` constants; some are overridable by **environment variables** (`.env` loaded by `python-dotenv` from the CWD).
2. CLI arguments (`main.py:148-161`).
3. Hard-coded literals inside functions (listed below).
4. Input JSON weather presets.

### 10.2 Environment-overridable settings
| Var | Default | Used by | Meaning |
|---|---|---|---|
| `WEATHER_API_BASE_URL` | Open-Meteo forecast URL | weather_api | endpoint |
| `MAX_SAFE_WIND_SPEED` | 40.0 (km/h) | main | warning only |
| `MAX_SAFE_PRECIPITATION` | 15.0 (mm) | main | warning only |
| `METERS_PER_PIXEL` | 2.0 | main, constraints, pathfinding, overlap_repair, export | **planning and export scale** (the true map scale is 100–270 m/px; documented limitation) |
| `LARGE_CONTOUR_AREA_THRESHOLD` | 2500 | safe_dropzone | multi-drop zone threshold |
| `LARGE_CONTOUR_AREA_STEP` | 3000 | safe_dropzone | +1 drop per step |
| `MAX_DROPS_PER_CONTOUR` | 4 | safe_dropzone | cap |
| `MIN_DROP_SEPARATION_PX` | 40 | safe_dropzone | within-zone spacing |
| `DROP_SERVO_CHANNEL` | 9 | export | DO_SET_SERVO param1 |
| `DROP_SERVO_PWM` | 2000 | export | DO_SET_SERVO param2 |
| `HOME_CLEARANCE_PX` | 15 | main (candidates), overlap_repair (nudge) | base flood clearance |
| **`MAX_BASES`** | **4** | main → select_bases | **max number of bases/UAVs** |
| **`MIN_BASE_SEPARATION_PX`** | **150** | main → select_bases; overlap_repair nudge | base spacing |

### 10.3 Fixed constants (`settings.py`, not env-overridable)
`DRONE_MODEL_REFERENCE` = "Garuda Aerospace Agri Kisan Drone (GA-AD)", `DRONE_RATED_PAYLOAD_KG` 8, `DRONE_LOADED_ENDURANCE_MIN` **7.0**, `DRONE_SPEED_MPS` **5.0**, `BATTERY_CAPACITY_WH` None, `PAYLOAD_CAPACITY_KG` **8.0**, `PAYLOAD_PER_DROP_KG` **0.25** (→ 32 drops per load), `PAYLOAD_TIME_FACTOR_PER_KG` 0.08, `WIND_PENALTY_PER_MPS` 0.025, `BATTERY_RESERVE_FRACTION` **0.20** (80 % usable per sortie), `INITIAL_BATTERY_FRACTION` 1.0, `FLOOD_GROWTH_PX_PER_5MM_HOUR` 1.0, `FLOOD_WIND_SHIFT_PX_PER_10KMH_HOUR` 1.0, `FLOOD_PRECIP_REFERENCE_MM_PER_H` 5.0, `FLOOD_WIND_REFERENCE_KMH` 10.0, `MASK_MIN_COVERAGE_FRACTION` 0.001, `MASK_MAX_COVERAGE_FRACTION` 0.50, `REGION_PRESETS` = {varanasi (25.14, 83.11, ~100 m/px), kanpur (26.45, 80.05, ~128), bhopal (23.37, 78.03, ~270)}.

### 10.4 CLI (`main.py`)
`--image PATH` (else tkinter picker), `--display-width/-w 750`, `--min-area 200`, `--lat`, `--lon` (image **centre**; required if the image stem has no preset, e.g. `assam.png`), `--horizon 2.0` h, `--dummy-weather`, `--dummy-weather-moderate` (takes precedence if both are given), `--no-gui`, `--auto-sample`, `--sample-points "x,y;x,y"`.

### 10.5 Hard-coded parameters inside code
| Value | Location |
|---|---|
| routing downsample factor **5** | `main.py:382,427`, `compute_full_path` default, `try_reroute_leg` |
| candidate grid **~40×40** | `main.py:361` |
| refinement cap `min(6, n+2)` | `main.py:446` |
| overlap repair `max_rounds=5` | `main.py:538` |
| `select_bases(refine_rounds=3)` | multi_base |
| nudge radius 5…50 px step 5 | `try_nudge_base` |
| reroute buffer 1 cell; avoid-sample every 2 px; ±15 segment window fallback | `try_reroute_leg` |
| `locate_cell` snap radius 4 cells | pathfinding |
| cruise altitude **100 m**, drop altitude **10 m**, loiter **5 s** | `generate_mission_file` |
| forecast timesteps (5 min, 10 min, 30 min, 1 h, 2 h) | `predict_spread` |
| drop band radius ≤ 20 px or 3 % of the image | safe_dropzone |
| HSV margins H20 / SV30, patch radius 5; auto-sample S ≥ 150, V ≥ 100, hue tol 6, window 11 | image_processing |
| `zones_with_point` tolerance −25 px | `main.py:355` |
| map palette 12 colours, arrows every 110 px, figure dpi 160 | mission_output |
| weather HTTP timeout 10 s | weather_api |

**Not configurable or absent:** UAV IDs, per-UAV home coordinates (homes are computed, never input), heterogeneous drone specs, waypoint count limits, altitude per base, mission start times.

---

## 11. EXTERNAL INTERFACES

| Interface | File / function | Input | Output | Protocol / format | Direction | Purpose |
|---|---|---|---|---|---|---|
| Open-Meteo | `weather_api.get_weather_data` | lat, lon | dict | HTTPS GET, query `current=precipitation,wind_speed_10m,wind_direction_10m,visibility&timezone=auto`, JSON | out → in | live weather |
| Weather presets | `main.py:204-223` | `data/input/dummy_weather*.json` | dict | JSON | in | offline weather |
| Map image | `main.py:241` | PNG/JPG/TIF/BMP | BGR array | file | in | flood input |
| **Mission files** | `generate_mission_file` | per-base path | `session/base<N>.waypoints` | **QGC WPL 110** text, tab-separated 12 fields: `seq current frame cmd p1 p2 p3 p4 lat lon alt autocontinue` | out | GCS upload (Mission Planner / QGroundControl, manual) |
| Route map | `display_path_on_map` | plans | `mission_route_map.png` (matplotlib, legend, arrows), `mission_map_annotated.png` (cv2 raw overlay) | PNG | out | visual verification |
| Run summary | `write_summary` | summary dict | `run_summary.json` | JSON | out | metrics / debug |
| Logs | `logging` INFO to stderr, `[LEVEL] msg` | – | verdict lines (Handoff §2) | text | out | grep-able verification |
| GUI | `ImageProcessor.select_sample_points`, `show_image_safe`, `plt.show`, tkinter dialog | user | – | OpenCV / Tk / matplotlib windows | in/out | interactive sampling and review |
| Env | `.env` | – | – | dotenv | in | config |

Waypoint file row sequence per base (from code, confirmed in output):
```
0  current=1 frame=0 cmd=16  home lat/lon alt 0            (QGC home row)
1  frame=3 cmd=22 TAKEOFF at home, alt 100
…  cmd=16 WAYPOINT alt 100 for every full_path point (each D* Lite cell centre, ~5 px apart)
   at a drop index: 19 LOITER_TIME(5 s) → 16 @10 m → 183 DO_SET_SERVO(p1=9, p2=2000) @10 m → 16 @100 m
   at a reload index: 21 LAND at home → 22 TAKEOFF at home (100 m)
…  20 RETURN_TO_LAUNCH → 21 LAND (alt 0)
```
**Not present:** MAVLink transport, telemetry, PX4/ArduPilot parameters, SITL launching, GeoJSON, KML, CSV.

---

## 12. TESTING (exists in MAIN)

Run: `python -m unittest discover tests` from the repo root. **Verified result in this analysis: `Ran 84 tests … OK (skipped=1)`** (Python 3.11.9). The skipped test needs the gitignored `data/output/session_20261004_172629/`.

Legend columns: S = single-UAV, M = multi-UAV, A = allocation, R = routing, X = simulation/export.

| File | Test | What it checks | S | M | A | R | X |
|---|---|---|---|---|---|---|---|
| test_multi_base | `test_single_base_when_one_site_reaches_everything` | mode single, 1 base, all → 0 | ✔ | | ✔ | | |
| | `test_two_far_clusters_get_two_separated_bases_and_exclusive_assignment` | 2 bases ≥ 150 px, each drop once, clusters not split | | ✔ | ✔ | | |
| | `test_separation_is_enforced_even_if_it_costs_coverage` | separation invariant | | ✔ | ✔ | | |
| | `test_unreachable_drops_are_reported_with_reason` | reason string, not assigned | | ✔ | ✔ | | |
| | `test_base_cap_is_respected_and_leftovers_reported` | ≤ max_bases; assigned + unreachable = all | | ✔ | ✔ | | |
| | `TestPathOverlap.test_crossing_and_overlap_detected_repeated_base_points_ignored` | `polyline_intersections` | | ✔ | | ✔ | |
| | `TestDistanceFields.test_multi_source_field_is_min_of_single_fields` | multi-source Dijkstra; wall → inf | | ✔ | | ✔ | |
| | `TestSessionFolders.test_each_run_gets_a_new_session_folder` | `create_session_dir` (sequential only) | | | | | ✔ |
| test_overlap_repair | `test_find_leg_index` | leg lookup | | ✔ | | | |
| | `test_tier1_reroute_resolves_crossing` | perpendicular paths; reroute → 0 hits (no forecasts) | | ✔ | | ✔ | ✔ (writes file) |
| | `test_tier2_nudge_resolves_crossing` | nudge with mock `plan_base` | | ✔ | ✔ | | |
| | `test_bhopal_session_172629_case_resolves_to_zero_overlap` | real crossing case (15 hits → 0) | | ✔ | | ✔ | **SKIPPED** (needs gitignored files) |
| | `test_hard_error_enforced_when_all_tiers_fail` | remaining crossings > 0 (the `exit 2` itself is not tested) | | ✔ | | | |
| test_audit_regressions | `TestMapRendering` ×5 | basemap kept, corrupt detection, dashed RTB, legend entries, label overprint (single base) | ✔ | | | | ✔ |
| | `TestMapPalette` ×1 | 12 colours, ΔE ≥ 30 pairwise / ≥ 35 vs fixed, no green | ✔ | ✔ | | | ✔ |
| | `TestRoutingConsistency` ×5 | no snap across walls; connector-clear snapping; filter ≡ D* Lite (25 maps); explicit fallback; D* Lite avoids flood | ✔ | | | ✔ | |
| | `TestDropPlacement` ×1 | large zone ≥ 3 separated dry points | ✔ | | | | |
| | `TestForecastInterpolation` ×3 | SDF front linear; frames binary + cumulative; moderate weather slow | | | | ✔ | |
| | `TestWaypointExport` ×2 | valid WPL incl. reload, servo params; validator rejects bad rows | ✔ | | | | ✔ |
| | `TestVisionSampling` ×3 | auto-sampling, reference Varanasi mask (48 828 px, 18 contours, `hsv_lower` [0,21,142]), bundled maps sane | ✔ | | | | |
| | `TestRouteRefinement` ×2 | converged / cycle / cap; distances accumulate | ✔ | | | ✔ | |
| test_mission_constraints | 27 tests | battery model, wind conversion and headwind, per-sortie reserve, home scoring (TSP order, payload, reachable tracking, rejected vs absent), `plan_mission_stops` (reload on battery and capacity, unreachable, routed detour, RTB in reserve, decreasing payload, flood reprioritisation, infeasible), `sortie_battery_report`, safe drop point dry/adjacent, connectivity filter, multi-point large contour | ✔ | indirect | | ✔ | ✔ (reload LAND/TAKEOFF) |
| test_mission_output | 2 tests | sortie colours (#d50000, #1a237e), BASE/RELOAD labels; `map_routed_drop_distances` across reload | ✔ | | | ✔ | ✔ |
| test_coordinates | 4 tests | centre, direction signs, metric scale, region presets in district boxes and unique | | | | | ✔ |
| test_dstarlite | 5 tests | forecast frame selection, 3-value `compute_full_path` contract, empty grid, replanning with obstacles, snapping | | | | ✔ | |
| test_flood_spread | 8 tests | zero weather, isotropic, wind direction, timesteps, sub-hour, interpolation | | | | | |
| test_image_input | 1 test | trivial `Path.is_file` | | | | | |
| test_weather_api | 2 tests | success + cache, fallback | | | | | |

Counts: audit 22, constraints 27, multi_base 8, overlap_repair 5, flood_spread 8, dstarlite 5, coordinates 4, mission_output 2, weather 2, image_input 1 = **84**.

**Untested important functionality:**
- `main.main()` end-to-end, `plan_base`, `plan_assigned`, the exit-2 paths, and the territory computation / `TERRITORY CHECK`.
- `execute_overlap_repairs` **Tier-0 reassignment and the success paths** (only the all-fail path is tested). This includes Tier-1 with real `forecasts` (tests pass `None`) and the stale `base_reach` after a nudge.
- `select_bases` re-placement loop correctness or optimality; `single_drop_reach` directly; `routed_distance_function` directly (inf handling).
- `display_path_on_map` with `bases=[…]` (>1) and `territories` (multi-base rendering is untested).
- Multiple waypoint files per session; `validate_waypoints_file` on multi-base output; `run_summary.json` schema.
- `create_session_dir` under concurrency (the race was observed in this analysis); `cluster_contours`; the `--sample-points` parser; weather threshold warnings; `get_weather_data` JSON `ValueError` path; `ImageProcessor.select_sample_points` (GUI).

---

## 13. EXECUTION

```text
Installation   python -m venv .venv ; .venv\Scripts\activate ; pip install -r requirements.txt
   ↓
Environment    optional: copy .env.example → .env  (MAX_BASES, MIN_BASE_SEPARATION_PX, … can be set here)
   ↓
Configuration  config/settings.py (+ env), CLI flags
   ↓
Command        (run from repo root — relative paths data/input, data/output, and `src`/`config` imports)
               python main.py --image data/input/varanasi.png --dummy-weather-moderate --no-gui
               python main.py --image data/input/assam.png --lat <lat> --lon <lon> --no-gui   (no preset)
               python main.py --image data/input/kanpur.png --dummy-weather --no-gui           (stress)
               python main.py --image data/input/bhopal.png                                     (live weather, GUI sampling)
               MAX_BASES=1 python main.py ...      (force single base; bash syntax)
   ↓
Entry point    main.py: main()
   ↓
Flow           §2 / §4
   ↓
Output         data/output/session_<ts>/ base1..N.waypoints, mission_route_map.png,
               mission_map_annotated.png, run_summary.json ; exit 0/1/2
Tests          python -m unittest discover tests
Dev scripts    python scripts/test_pathfinding_dummy.py ; python debug_flood_spread.py ;
               python scratch/*.py (scratch/test_repair_on_172629.py needs a gitignored session)
```
There is no server, no simulation startup, and no package entry point (`console_scripts`).

### 13.1 Observed runs (this analysis; exported HEAD copy, `--dummy-weather-moderate --no-gui`, Python 3.11.9)
| Map | Stage counts | Mode | Bases (px) | Min sep | Repairs | Coverage | Sorties ≤ 80 % | Fallback legs / flood px |
|---|---|---|---|---|---|---|---|---|
| bhopal | 25 drop points | MULTI_BASE_2 (best single 18/22) | (352,157), (277,457) | 309 px | none | 22 assigned / 3 unreachable / **22 visited** | 7 sorties, max 79.7 % | 0 / 0 |
| kanpur | 130 raw → 33 zones → 41 drops | MULTI_BASE_3 (best single 21/33) | (315,495), (351,63), (567,279) | 305 px | none | 32 / 9 / **32** | 12 sorties, max 78.3 % | 0 / 0 |
| varanasi | 102 raw → 22 zones → 28 drops | MULTI_BASE_4 (best single 16/21) | (225,315), (567,171), (369,81), (603,513) | 217 px | **2× Tier-0 reassign** (drops [16,20,27] B3→B1; [3] B1→B4) | 21 / 7 / **21** | all ≤ 80 % | 0 / 0 |

All three runs: exit 0, PATH OVERLAP CHECK clean, TERRITORY CHECK clean, all waypoint files valid, MAP SANITY ok.

---

## 14. FRONTEND / BACKEND / UI
**There is no web frontend, backend server, or API routes.** The UI consists of:
- **Input**: optional tkinter file dialog; an OpenCV window for clicking flood samples (`'c'` to finish); an OpenCV "Flood Mask" window (skipped with `--no-gui`).
- **Visualisation** (`display_path_on_map`), drawn in this order:
  1. forecast flood: brown diagonal hatch plus outline (forecast zones ≥ 200 px²);
  2. detected contours: green, 2 px;
  3. *multi only*: territory boundaries as dark grey dots;
  4. each sortie in a global palette colour: outbound solid 4 px, RTB dashed with a continuous dash pattern;
  5. drop points: blue dots with a white ring;
  6. bases: black house glyph, white outline, label `BASE n` (multi) or `BASE`;
  7. reloads at a base become a star with `RELOAD xk`; elsewhere a `RELOAD` star;
  8. matplotlib figure with direction arrows every 110 px (solid/dashed), legend outside the image (`Bn Sortie k Outbound/RTB`, flood, forecast, drop, base, territory, reload). Title "ADAPT Multi-Base…" or "ADAPT Multi-Sortie…".
  9. `MAP PALETTE` warning when there are more than 12 sorties.
- Connection to the planner: direct in-process function calls only.

## 15. DATABASE / STORAGE
**No database.** Persistent storage is files only:
- Reads: `data/input/*.png`, `data/input/dummy_weather*.json`, `.env`.
- Writes: `data/output/session_<ts>[_n]/` → `base<N>.waypoints` (one per base; rewritten in place by repairs), `mission_route_map.png`, `mission_map_annotated.png`, `run_summary.json`. `scripts/test_pathfinding_dummy.py` writes `dummy_path_test.png` in the CWD.
- Caches: in-memory only (`lru_cache` on weather; routed distance fields per run).

`run_summary.json` keys (verified): `image, geo_anchor, session_dir, weather, mask_coverage, home_candidates{grid_points,dry,with_clearance}, home_outcome (SINGLE_BASE|MULTI_BASE_n|NO_CANDIDATES_SAMPLED|NO_BASE_REACHES_ANY_DROP), bases[ per-plan dict minus full_path/drop_indices/reload_indices ], base_separation_min_px, base_separation_required_px, path_overlaps, overlap_repairs, territory_violations, assignment{drop: base#(1-based)}, unreachable_by_any_base{drop: reason}, home (first base, legacy), home_clearance_px (min), drops_visited, reloads, unreachable, connectivity_excluded, legs{total,dstar,same_cell,fallback}, route_flooded_px_arrival, route_flooded_px_current, planner_sortie_battery[], routed_sortie_battery[], waypoint_problems[], refinement{status, per_base[]}, stage_counts{raw_contours,zones,zones_with_drop_point,drop_points,unreachable_by_any_base,assigned,visited}, map{ok, dominant_colour, dominant_fraction, shape_match, correlation, unchanged_fraction}`.
Early-exit summaries contain only the keys populated up to that point.

---

## 16. FILE-TO-FILE DEPENDENCY MAP

```text
config/settings.py ◄──────────────── (almost everything)
src/routing/dstarlite.py ◄── src/routing/pathfinding.py ◄── src/mission/safe_dropzone.py (lazy)
                         ◄── src/mission/overlap_repair.py
src/weather/flood_spread.py ◄(lazy)── pathfinding.compute_full_path
                            ◄(lazy)── constraints.plan_mission_stops
                            ◄(lazy)── overlap_repair.try_reroute_leg
                            ◄──────── main
src/mission/coordinates.py ◄── mission_output ◄── overlap_repair ◄── main
src/mission/constraints.py ◄── multi_base ◄── overlap_repair ◄── main
                           ◄── overlap_repair (sortie_battery_report) ; ◄── main

Multi-base subsystem:
main.py
  ↓ imports  multi_base.select_bases, single_drop_reach
  ↓ imports  overlap_repair.execute_overlap_repairs
  ↓ imports  pathfinding.{build_routing_grid, routed_distance_function, polyline_intersections,
  │                       grid_distance_field, locate_cell, compute_full_path, nearest_neighbor_tsp,
  │                       polyline_flood_pixels}
  ↓ calls    select_bases ──► constraints.{leg_battery_fraction, score_home_candidates}
  ↓ produces selection{bases, assignment, unreachable}
  ↓ consumed by plan_assigned/plan_base (main) ──► safe_dropzone.filter_drops_by_home_connectivity
  │            ──► constraints.plan_mission_stops ──► flood_spread.{obstacle_mask_at, forecast_at}
  │            ──► pathfinding.compute_full_path ──► dstarlite.DStarLite, flood_spread.obstacle_mask_at
  │            ──► constraints.sortie_battery_report
  │            ──► mission_output.{generate_mission_file ──► coordinates.pixel_to_latlon, validate_waypoints_file}
  ↓ produces plans_by_base
  ↓ consumed by overlap_repair.execute_overlap_repairs
  │            ──► pathfinding.polyline_intersections, mission_output.split_sorties
  │            ──► try_reroute_leg ──► dstarlite, pathfinding.{build_routing_grid, locate_cell, cell_of,
  │                 cell_center, polyline_flood_pixels}, constraints.sortie_battery_report,
  │                 mission_output.{generate_mission_file, validate_waypoints_file}, flood_spread.obstacle_mask_at
  │            ──► try_nudge_base ──► multi_base.single_drop_reach, main.plan_base (callback)
  ↓ produces repaired plans_by_base, repairs
  ↓ consumed by main territory block (grid_distance_field, locate_cell) and
                mission_output.display_path_on_map(bases=, territories=), basemap_sanity
  ↓ produces base<N>.waypoints, PNGs, run_summary.json
```
Callback coupling: `overlap_repair` receives `plan_base` and `plan_assigned` as **function parameters** (`plan_base_fn`, `plan_assigned_fn`). Those are closures inside `main()` that capture the run state. Porting `overlap_repair.py` therefore requires DEVELOPMENT's `main` to expose equivalent callables with the signatures `plan_base_fn(number:int, home:(x,y), drops:list) -> plan|None` and `plan_assigned_fn(base_idx:int) -> plan|None`, returning plan dicts with at least `base, home, full_path, stop_indices, route_points, drop_indices, reload_indices` (keys read by repair).

---

## 17. HOW TO PORT MULTI-UAV FROM MAIN INTO DEVELOPMENT

### 17.0 Branch topology (observed from remote-tracking refs; verify against the actual DEVELOPMENT checkout)
```
4cdd48b ─ 186350c ─ 775165d ─┬─ ca27215  (main, origin/main, origin/routing-algo)   ← ALL multi-base work is this ONE commit
                             ├─ 0abbfef ─ 54bb27a  (origin/research-validation: geomapping, validation, paper)
                             ├─ 723381b  (origin/ajay: Sentinel-1/GIBS fetchers, mask-direct planner, TSP vs D* benchmark)
                             ├─ 607bc5f  (origin/time-aware-routing)
                             └─ (origin/backup = 775165d)
```
**UNCLEAR — requires verification:** whether DEVELOPMENT is `origin/research-validation`. Its description (fixes, tests, validation, research paper) matches. If it is, then at that ref:
- `src/mission/constraints.py`, `multi_base.py` and `overlap_repair.py` **do not exist**.
- `pathfinding.compute_full_path(ordered_points, obstacle_mask, downsample_factor=5)` returns **2 values** `(full_path, drop_indices)`, has no forecast or leg report, and there is no `RoutingGrid`/`locate_cell`/`routed_distance_function`/`polyline_intersections`.
- `flood_spread.predict_spread` returns **one ndarray** (not a dict of frames) plus `stretch_goal_convlstm_stub`.
- `safe_dropzone.find_safe_drop_points(clustered_contours, path_order, path_points)` has no obstacle mask and no connectivity filter.
- `main.py`: positional `image_path` (default varanasi), home = safe point nearest the image centre, a single route, a single output `data/output/enriched_drone_mission.waypoints` with stale-file removal (`remove_mission_file`), `meters_per_pixel = METERS_PER_PIXEL / scale_factor` and image centre `((W-1)/2, (H-1)/2)`.
- `coordinates.pixel_to_latlon` is hardened (finite checks, pole/antimeridian handling, raises `ValueError`); `generate_mission_file` pre-converts all points and takes `servo_channel`/`servo_pwm` params; `clustering` has the hull fix.
- Extra tests: `test_adversarial_coordinates`, `test_clustering`, `test_coordinates_groundtruth`, `test_coordinates_safety`, `test_geodesy_reference`, `test_image_centre`, `test_resize_consistency`, `test_stale_mission`, `test_wpl_validation`, `test_wpl_validator_adversarial`, `wpl_validator.py`, plus a **different `tests/test_mission_output.py`**.
- `paper/experiments/common.py` **monkeypatches `main.predict_spread`, `main.find_safe_drop_points`, `main.compute_full_path` and `main.generate_mission_file`** and depends on the old single-route `main` flow.

So, relative to that ref, MAIN's multi-base requires the whole `ca27215` planning stack, not just the two new modules.

### A. Files directly relevant to multi-UAV (new; copy and adapt)
- `src/mission/multi_base.py` (153 lines; depends only on `constraints`)
- `src/mission/overlap_repair.py` (442 lines)
- `tests/test_multi_base.py`, `tests/test_overlap_repair.py` (drop or adapt the Bhopal session test, which depends on gitignored files)
- The multi-base orchestration in `main.py` lines **360-625** (candidates → `select_bases` → assignment table → `plan_base`/`plan_assigned` → repairs → overlap hard-stop → territories → aggregated summary) and **627-645** (multi-base map call). Port this as code integrated into DEVELOPMENT's `main.py`, not by replacing the file.

### B. Shared infrastructure required by A (introduced in `ca27215`; likely missing in DEVELOPMENT)
| Needed | Source in MAIN | Required for |
|---|---|---|
| `constraints.py`: `leg_battery_fraction`, `wind_kmh_to_mps`, `headwind_component_mps`, `score_home_candidates`, `PlannedRoute`, `_routed_distance`, `plan_mission_stops`, `sortie_battery_report`, `log_deviation` | whole file | reach, placement, per-base sorties, repair battery checks |
| `pathfinding.py`: `RoutingGrid`, `build_routing_grid`, `cell_of`, `cell_center`, `locate_cell`, `grid_distance_field`, `routed_distance_function`, `polyline_flood_pixels`, `polyline_intersections`, new `compute_full_path` (3 returns, `forecast_masks`, `leg_report`), extended `snap_to_nearest_free_cell` | most of the file | routed distances, connectivity, routing, deconfliction, territories |
| `flood_spread.py`: dict-returning `predict_spread`, `_bracket`, `signed_distance`, `interpolate_flood_front`, `forecast_at`, `obstacle_mask_at`, `interpolate_forecast_mask` | whole file | time-aware routing and planner |
| `safe_dropzone.py`: `obstacle_mask`/`raw_contours`/multi-point params, `filter_drops_by_home_connectivity` | whole file | drops dry at horizon; per-base connectivity |
| `mission_output.py`: `reload_indices` in `generate_mission_file`, `validate_waypoints_file`, `split_sorties`, `MAV_*` constants, `_wp_line`, multi-base `display_path_on_map(bases=, territories=)`, `draw_dashed_polyline`, `arrow_positions`, `basemap_sanity`, `SORTIE_PALETTE_HEX` | merge, don't replace | export per base; repair uses `split_sorties` |
| `main.py` helpers: `refine_route`, `map_routed_drop_distances`, `create_session_dir`, `write_summary` | functions | per-base pipeline, output layout |
| `image_processing.auto_sample_points` | method | only if headless `--no-gui` runs are wanted |
| `settings.py` constants (§F) | – | everything above |

### C. Files that should NOT be copied blindly
| File | Why |
|---|---|
| `main.py` (whole) | DEVELOPMENT has its own CLI (positional image), stale-mission removal, resize-corrected m/px, the `(W-1)/2` centre convention, and the paper harness monkeypatching `main.*`. Port the orchestration block into it. |
| `src/mission/coordinates.py` | DEVELOPMENT's hardened version is authoritative; MAIN's has no validation. |
| `src/mission/mission_output.py` | Both diverged: DEVELOPMENT has pre-conversion, servo params and the drop-at-home semantics `0 < i < len-1`; MAIN has reload LAND/TAKEOFF, the validator, the multi-base map and the palette. Merge function by function. |
| `src/vision/clustering.py` | DEVELOPMENT has the hull fix. |
| `src/routing/pathfinding.py` | DEVELOPMENT has improved snap warnings in the old `compute_full_path`; MAIN rewrote the module. Reconcile, and keep DEVELOPMENT's logging intent. |
| `tests/test_mission_output.py` | Same filename, different content. Merge the tests; never overwrite. |
| `README.md`, `Handoff.md` | MAIN's README is outdated; Handoff is MAIN-specific. Merge text manually. |
| `scratch/*`, `debug_flood_spread.py`, `scripts/test_pathfinding_dummy.py`, `dummy_path_test.png` | dev artefacts |
| `tests/test_overlap_repair.py::test_bhopal_session_172629…` | depends on a gitignored local session folder |
| `config/settings.py` (whole) | add constants; do not replace DEVELOPMENT's file |

### D. Functions/classes to port
`select_bases`, `single_drop_reach`, `_separated`, `_assign`; `execute_overlap_repairs`, `try_reroute_leg`, `try_nudge_base`, `sortie_drops`, `find_leg_index`; `polyline_intersections`, `grid_distance_field` (multi-source), `routed_distance_function`, `build_routing_grid`, `RoutingGrid`, `locate_cell`, `cell_of`, `cell_center`, `polyline_flood_pixels`, `compute_full_path` (MAIN form); `leg_battery_fraction`, `wind_kmh_to_mps`, `headwind_component_mps`, `score_home_candidates`, `PlannedRoute`, `_routed_distance`, `plan_mission_stops`, `sortie_battery_report`, `log_deviation`; `predict_spread` (dict), `forecast_at`, `obstacle_mask_at`, `interpolate_flood_front`, `signed_distance`, `_bracket`; `filter_drops_by_home_connectivity`, the new `find_safe_drop_points`; `split_sorties` (mission_output), `validate_waypoints_file`, the `reload_indices` support, the multi-base `display_path_on_map`; `refine_route`, `map_routed_drop_distances`, `create_session_dir`, `write_summary`; and the main-closure logic `plan_base`/`plan_assigned` plus the territory block.

### E. Dependencies DEVELOPMENT must have
No new third-party packages. Multi-base uses numpy, opencv-python, matplotlib, requests and python-dotenv, which are already in DEVELOPMENT's `requirements.txt` (identical lists). DEVELOPMENT's `paper/experiments/requirements-paper.txt` is separate.

### F. Configuration changes required (add to DEVELOPMENT `config/settings.py`)
`MAX_BASES` (env, 4), `MIN_BASE_SEPARATION_PX` (env, 150), `HOME_CLEARANCE_PX` (env, 15); battery/aircraft: `DRONE_LOADED_ENDURANCE_MIN` 7.0, `DRONE_SPEED_MPS` 5.0, `PAYLOAD_CAPACITY_KG` 8.0, `PAYLOAD_PER_DROP_KG` 0.25, `PAYLOAD_TIME_FACTOR_PER_KG` 0.08, `WIND_PENALTY_PER_MPS` 0.025, `BATTERY_RESERVE_FRACTION` 0.20, `INITIAL_BATTERY_FRACTION` 1.0 (+ documentation constants); flood: `FLOOD_*` four constants; drops: `LARGE_CONTOUR_AREA_THRESHOLD`, `LARGE_CONTOUR_AREA_STEP`, `MAX_DROPS_PER_CONTOUR`, `MIN_DROP_SEPARATION_PX`; export: `DROP_SERVO_CHANNEL`, `DROP_SERVO_PWM` (DEVELOPMENT passes these as function params instead; pick one source of truth); sanity: `MASK_MIN/MAX_COVERAGE_FRACTION`; `REGION_PRESETS` (if DEVELOPMENT has its own georeferencing, reconcile). Optionally document the new env vars in `.env.example`.

### G. Integration points (where multi-base plugs into DEVELOPMENT)
1. **After drop-point generation, before home selection.** Replace DEVELOPMENT's "home = safe point nearest the image centre" with: candidate grid + clearance → `build_routing_grid` → `routed_distance_function` → `select_bases`. Keep DEVELOPMENT's upstream vision, georeferencing and validation.
2. **Per-base planning function.** Wrap DEVELOPMENT's existing single-route pipeline (or MAIN's richer one) in a `plan_base(number, home, drops) -> plan dict` with the keys `overlap_repair` reads (§16). This is the seam where DEVELOPMENT's route planner is reused.
3. **Mission export per base.** Call DEVELOPMENT's `generate_mission_file` (hardened) once per base with filename `base<N>.waypoints`, using DEVELOPMENT's image-centre and m/px conventions. The same must be done **inside `overlap_repair.try_reroute_leg`** (line ~206), which currently hard-codes `(W//2, H//2)` and `settings.METERS_PER_PIXEL`; parameterise it (e.g. pass `image_center_px` and `meters_per_pixel`, or an export callback).
4. **Output layout.** DEVELOPMENT writes one fixed file and deletes stale missions; MAIN writes session folders. Decide one policy. If session folders are kept, DEVELOPMENT's `test_stale_mission.py` semantics must be reconsidered rather than broken. With multiple bases, a stale `base3.waypoints` from an earlier run must never be confused with current output.
5. **Visualisation.** Extend DEVELOPMENT's `display_path_on_map` with the `bases=` and `territories=` parameters; keep its single-base call signature working.
6. **Paper harness.** `paper/experiments/common.py` patches `main.compute_full_path` etc. Any change to their signatures or call pattern breaks it. Keep the old symbols or adapt the harness deliberately.

### H. Potential conflicts
| Area | Conflict |
|---|---|
| **Existing fixes** | Coordinates hardening (DEVELOPMENT) vs MAIN's unvalidated version; clustering hull fix; `generate_mission_file` pre-conversion (no partial file) vs MAIN's convert-while-writing; drop-at-home semantics (`0<i<len-1` in DEVELOPMENT vs `tuple(pt) != tuple(home)` in MAIN); stale-mission removal. |
| **Georeference semantics** | DEVELOPMENT treats `METERS_PER_PIXEL` as the *original-image* scale and divides by the resize factor for export. MAIN uses it unchanged for **both** export **and** the battery/routing distances on resized pixels. Porting MAIN's battery model under DEVELOPMENT's semantics changes every reach/battery number. **Decision required** (UNCLEAR which is intended for planning distances). |
| **Image centre** | `(W//2, H//2)` in MAIN (2 call sites: `main.plan_base`, `overlap_repair.try_reroute_leg`) vs `((W-1)/2, (H-1)/2)` in DEVELOPMENT. DEVELOPMENT tests (`test_image_centre`, `test_resize_consistency`) will catch mismatches. |
| **Tests** | Same-named `tests/test_mission_output.py`; DEVELOPMENT's WPL validator tests (`tests/wpl_validator.py`) vs MAIN's `validate_waypoints_file`: keep both or unify, and run DEVELOPMENT's adversarial validator on every `base<N>.waypoints`. |
| **Route planner** | `compute_full_path` return arity (2 → 3) and new keyword args; `predict_spread` return type (ndarray → dict). Every DEVELOPMENT caller (main, paper scripts, tests) must be updated or given compatibility wrappers. |
| **Waypoint generator** | `generate_mission_file` signatures differ (`reload_indices` vs `servo_channel/servo_pwm`); merge into one signature. |
| **Research implementation** | Paper results (`paper/experiments/results/*`) were produced with the single-route pipeline. Multi-base changes the headline numbers (coverage, sorties). Keep the single-base path reproducible (e.g. `MAX_BASES=1`) and regenerate or label results. |
| **UI** | `display_path_on_map` signature: DEVELOPMENT still has `tsp_path=` (old) vs MAIN's `reload_indices, bases, territories`. |
| **Simulation** | None on either side in code; no conflict. |
| **Configuration** | DEVELOPMENT `settings.py` has only 4 constants; add MAIN's without renaming. `.env.example` differs (DEVELOPMENT +2 lines). |
| **Known MAIN defects to not import as-is** | `create_session_dir` race; `base_reach` not refreshed after Tier-2 nudge; pre-repair `ASSIGNMENT TABLE`/`BASE SEPARATION` logs can be stale; territory labels index `plans` not base numbers (misaligned if a base's plan is `None`); the unused `pixel_to_latlon` import in `overlap_repair`. |

---

## 18. PRESERVATION RULES FOR THE FUTURE MERGE (to be followed by the DEVELOPMENT session)
1. DEVELOPMENT is authoritative for all existing functionality.
2. MAIN is only the reference for the multi-UAV (multi-base) implementation and its strictly necessary dependencies.
3. Do not replace DEVELOPMENT implementations because MAIN differs; diff function by function first.
4. Preserve DEVELOPMENT's fixes (coordinates hardening, hull fix, export safety, stale-mission handling, resize-consistent georeferencing).
5. Preserve DEVELOPMENT's tests; never overwrite a same-named test file. Merge test cases.
6. Preserve DEVELOPMENT's research-paper work, `paper/experiments` harness included.
7. Preserve DEVELOPMENT's single-UAV behaviour: `MAX_BASES=1` (or an explicit flag) must reproduce it.
8. Port only multi-base plus its dependency closure (§17.B).
9. Where both branches implement a component, compare before editing.
10. Integrate into DEVELOPMENT's architecture; do not copy directories wholesale.
11. Remove no working functionality.
12. Overwrite no tests.
13. All existing DEVELOPMENT tests must still pass.
14. Add tests for the integrated multi-UAV functionality (see the untested list in §12).

---

## 19. RESEARCH-PAPER RELEVANCE
| Paper section | MAIN components |
|---|---|
| **System Architecture** | §2 pipeline; the multi-base layer as a wrapper around an unchanged single-drone planner; session outputs and verdict logs as verification instrumentation |
| **Methodology** | battery-reach definition (single-drop sortie on routed distance, 80 % usable); greedy fewest-bases cover with separation and re-placement (§7.11); nearest-feasible assignment (§7.12); per-base multi-sortie planning (§7.8); time-aware D* Lite with SDF-interpolated forecasts (§7.4, §7.10); planner⇄router fixed-point refinement (§7.9); 3-tier airspace deconfliction (§7.14); routed-Voronoi territories (§7.15) |
| **Algorithm** | pseudocode for `select_bases`, `plan_mission_stops`, `execute_overlap_repairs`; complexity: reach O(C·D) lookups after D Dijkstra fields on a (H/5)×(W/5) grid; greedy O(B·C·D) |
| **Optimisation objectives / constraints** | Objectives: fewest bases (greedy), then min routed cost (`score_home_candidates`: hours + Σ sortie drain). Constraints: battery reserve 20 %, payload 8 kg / 0.25 kg per drop, flood clearance 15 px (current and forecast), base separation 150 px, `MAX_BASES` 4, no route intersections, drops dry over the horizon, arrival-time flood check |
| **Experimental setup** | 3 maps × 2 weather presets (Handoff §8: 6-run matrix); aircraft GA-AD (7 min, 5 m/s); `METERS_PER_PIXEL` 2.0 (state the scale limitation); downsample 5; horizon 2 h |
| **Metrics** (`run_summary.json`) | `home_outcome`/#bases, `stage_counts`, assigned / unreachable / visited, `routed_sortie_battery`, legs (dstar/same_cell/fallback), `route_flooded_px_*`, `path_overlaps`, `overlap_repairs` by tier, `territory_violations`, `base_separation_min_px`, `refinement` status |
| **Results** | §13.1 table (single best-site reach vs multi-base coverage: bhopal 18→22, kanpur 21→32, varanasi 16→21 visited of coverable); repair tier usage |
| **Discussion / limitations** | independent bases (no MAPF, no temporal deconfliction), greedy not optimal, homogeneous fleet, heuristic flood model, linear battery, compressed georeference scale, mixed headwind treatment (directional in scoring, scalar elsewhere) |

---

## 20. DIAGRAMS

### Overall architecture
```text
[map.png]──►[ImageProcessor]──►mask──►[predict_spread]──►forecasts{t}
                                  │                         │
                                  ▼                         ▼
                        [contours/cluster]        combined_obstacle_mask
                                  └──►[find_safe_drop_points]──►safe_points
[weather]──────────────────────────────────────────────┐           │
                                                       ▼           ▼
                       [candidates]──►[routing grid + routed distance]──►[select_bases]
                                                                          │
                                         ┌────────────────────────────────┘
                                         ▼
                           [plan_base × N]──►[execute_overlap_repairs]──►[overlap gate]
                                         │                                     │
                                         ▼                                     ▼
                          base<N>.waypoints                 [territories]──►[map]──►run_summary.json
```

### Single-UAV pipeline (as executed in MAIN when mode = single)
```text
safe_points → select_bases(single) → home
 → filter_drops_by_home_connectivity → nearest_neighbor_tsp
 → refine_route{ plan_mission_stops ⇄ compute_full_path(D* Lite, forecasts) }
 → compute_full_path(final) → sortie_battery_report → generate_mission_file(base1.waypoints)
 → display_path_on_map (single) → run_summary.json
```

### Multi-UAV pipeline
```text
safe_points ─► single_drop_reach(all candidates) ─► greedy cover ─► _assign ─► re-place ×3 ─► prune
     ─► bases[0..N-1], assignment
     ─► ∀b plan_base(b+1, bases[b], drops_b)  (independent)
     ─► execute_overlap_repairs (≤5 rounds) ─► pairwise intersection gate ─(fail)► exit 2
     ─► territories + TERRITORY CHECK ─► combined map ─► run_summary.json
```

### Multi-UAV task allocation
```text
            drops (global ids)
                 │
     reach[c] = {i : out(loaded)+back(empty) ≤ 80%}   per candidate site c
                 │
   ┌─ one c covers all coverable? ──yes──► 1 base = argmin score_home_candidates
   │no
   ▼
 repeat ≤ MAX_BASES:
   greedy: add separated c maximising (|reach[c] ∩ uncovered|, −Σdist)
   assign: i → argmin_{b: i∈reach[b]} dist(base_b, i)
   re-place each b: argmin cost over sites ⊇ its drops & separated  (≤3 rounds)
   prune empty bases
 leftovers → unreachable{i: reason}
```

### Route planning (per base)
```text
home, drops ─► connectivity filter ─► NN-TSP order
   ─► plan_mission_stops(straight)  → PlannedRoute₀
   ─► compute_full_path(route) → leg lengths → map_routed_drop_distances → known
   ─► plan_mission_stops(known) → PlannedRoute₁ … until stop list repeats/cycles/cap
   ─► compute_full_path(final, leg_report): per leg t_arrival → obstacle_mask_at(t) → grid
          → locate_cell(HOME comp) → DStarLite → validate | explicit fallback
   ─► sortie_battery_report (authoritative ≤ 80%)
```

### MD5 flow
```text
Not applicable — no MD5 or hashing exists in MAIN (or in any branch's history).
```

### Simulation / export pipeline
```text
plan.full_path (px) ─► pixel_to_latlon(centre=(W//2,H//2), anchor=(lat,lon), m/px=2.0)
   ─► QGC WPL 110 rows (home, takeoff, waypoints, loiter/descend/servo/climb, reload land/takeoff, RTL, land)
   ─► session/base<N>.waypoints ─► validate_waypoints_file
   ─► [manual] load into Mission Planner / QGroundControl   (no SITL/MAVLink in code)
```

### Module dependency graph
```text
main ─┬─ vision.image_processing
      ├─ vision.clustering
      ├─ weather.weather_api ── settings
      ├─ weather.flood_spread ── settings
      ├─ routing.pathfinding ── routing.dstarlite ; (lazy) flood_spread, settings
      ├─ mission.safe_dropzone ── (lazy) pathfinding, settings
      ├─ mission.constraints ── settings ; (lazy) flood_spread
      ├─ mission.mission_output ── mission.coordinates, settings
      ├─ mission.multi_base ── constraints
      └─ mission.overlap_repair ── settings, coordinates(unused), mission_output, constraints,
                                   multi_base, dstarlite, pathfinding ; (lazy) flood_spread
```

---

## 21. FIGURE PLACEHOLDERS (insert real outputs later)

```text
[FIGURE PLACEHOLDER 1] Input flood/environment map
Suggested image: data/input/<varanasi|kanpur|bhopal>.png as resized to 750 px width.
Source: actual input file.

[FIGURE PLACEHOLDER 2] Detected flood regions
Suggested image: binary flood mask ("Flood Mask" window, run without --no-gui), or the green
contours on mission_map_annotated.png.
Source: actual run output (screenshot).

[FIGURE PLACEHOLDER 3] Generated safe drop points
Suggested image: mission_map_annotated.png crop showing blue drop dots beside green contours
and the brown forecast hatch.
Source: data/output/session_<ts>/mission_map_annotated.png.

[FIGURE PLACEHOLDER 4] Single-UAV route
Suggested image: mission_route_map.png from a run with MAX_BASES=1 (title "ADAPT Multi-Sortie …").
Source: actual simulation output, to be inserted manually.

[FIGURE PLACEHOLDER 5] Multi-UAV waypoint allocation
Suggested image: mission_route_map.png territory boundaries (dotted) + numbered bases, alongside
the ASSIGNMENT TABLE / run_summary.json "assignment".
Source: actual run (e.g. varanasi moderate → 4 bases).

[FIGURE PLACEHOLDER 6] Multi-UAV routes
Suggested image: full mission_route_map.png with "Bn Sortie k" legend.
Source: actual run output.

[FIGURE PLACEHOLDER 7] Simulation environment
Suggested image: Mission Planner / QGroundControl with base1..N.waypoints loaded.
Source: manual GCS screenshot (the code does not launch any simulator).

[FIGURE PLACEHOLDER 8] UAV mission execution
Suggested image: SITL/GCS flight trace for one base mission.
Source: future manual SITL run (not implemented in MAIN).

[FIGURE PLACEHOLDER 9] QGroundControl/SITL visualization
Suggested image: QGC plan view of all base<N>.waypoints overlaid.
Source: manual screenshot.

[FIGURE PLACEHOLDER 10] Single vs multi-UAV comparison
Suggested image: side-by-side mission_route_map.png (MAX_BASES=1 vs default) + table of
visited / unreachable / sorties / max battery from run_summary.json.
Source: two actual runs on the same map and weather.
```
Note: with `METERS_PER_PIXEL=2.0` the waypoints sit within about 1 km of the anchor (Handoff §5.3), so GCS screenshots will not overlay the true flood geography.

---

## 22. ONE-PAGE ARCHITECTURE SUMMARY

```text
PROJECT:            ADAPT — flood-relief drone mission planner (Python, OpenCV)
MAIN ENTRY POINT:   main.py:main()   →  python main.py --image <png> [--dummy-weather-moderate] [--no-gui]
CORE PIPELINE:      HSV flood mask → heuristic time-indexed flood forecast → contours/hulls →
                    dry drop points → base candidates → multi-base selection → per-base
                    single-drone planning → deconfliction → per-base QGC WPL 110 + map + JSON
SINGLE-UAV IMPL.:   no separate path; select_bases returns 1 base → plan_base once
MULTI-UAV IMPL.:    multi-base (1..MAX_BASES=4 bases, 1 implicit drone each):
                    src/mission/multi_base.py (select_bases, single_drop_reach),
                    main.py:360-645 (plan_base/plan_assigned, territories),
                    src/mission/overlap_repair.py (reassign/reroute/nudge, exit 2)
WAYPOINT GENERATION:find_safe_drop_points (1 per zone; 2-4 for area ≥ 2500 px²), dry at horizon;
                    dense path = D* Lite cell centres
TASK ALLOCATION:    battery reach (single-drop sortie ≤ 80%) on routed distances; greedy
                    fewest-bases cover with 150 px separation; nearest-feasible assignment;
                    re-placement via score_home_candidates
ROUTE OPTIMIZATION: nearest-neighbour TSP per base; multi-sortie battery/payload planner
                    (plan_mission_stops) with reloads; planner⇄router refinement loop
PATH PLANNING:      time-aware D* Lite on 5× downsampled grid, arrival-time flood masks
                    (SDF-interpolated), component-aware snapping, logged fallbacks
COORDINATE CONV.:   pixel_to_latlon flat-earth, centre (W//2,H//2), REGION_PRESETS anchor,
                    METERS_PER_PIXEL=2.0 (compressed scale; documented limitation)
MISSION EXPORT:     QGC WPL 110 file per base: base<N>.waypoints (DO_SET_SERVO 9/2000,
                    100 m cruise / 10 m drop, reload LAND+TAKEOFF), validate_waypoints_file
SIMULATION:         none in code (internal battery/time simulation only; manual GCS import)
MD5:                not used anywhere (0 matches in tree and full history)
TESTING:            84 unittest tests, all pass (1 skipped); multi-base: test_multi_base (8),
                    test_overlap_repair (5)
CONFIGURATION:      config/settings.py + env (MAX_BASES, MIN_BASE_SEPARATION_PX,
                    HOME_CLEARANCE_PX, METERS_PER_PIXEL, …) + CLI flags
KEY FILES:          main.py, src/mission/{multi_base,overlap_repair,constraints,mission_output,
                    safe_dropzone,coordinates}.py, src/routing/{pathfinding,dstarlite}.py,
                    src/weather/flood_spread.py, config/settings.py
KEY DEPENDENCIES:   numpy, opencv-python, matplotlib, requests, python-dotenv; Open-Meteo API
```

---

## 23. FINAL PORTING CHECKLIST (for the Claude Code session on DEVELOPMENT)

```text
[ ] 0. Confirm DEVELOPMENT's base: `git merge-base HEAD ca27215` (expect 775165d) and list
       what already exists (constraints.py? routing grid API? dict forecasts?).
[ ] 1. Run DEVELOPMENT's full test suite and record the baseline pass count.
[ ] 2. Read DEVELOPMENT main.py, coordinates.py, mission_output.py, pathfinding.py,
       flood_spread.py, safe_dropzone.py and paper/experiments/common.py.
[ ] 3. Diff each shared module against MAIN (`git diff <dev> ca27215 -- <file>`); classify every
       hunk: dev-fix (keep) / main-infra-needed-by-multibase (port) / main-only-cosmetic (optional).
[ ] 4. Decide the georeference semantics (METERS_PER_PIXEL original vs resized; centre convention)
       for planning distances AND export. Write the decision down.
[ ] 5. Add settings constants (§17.F) without removing DEVELOPMENT ones.
[ ] 6. Port constraints.py (new file) + its tests from test_mission_constraints.py.
[ ] 7. Port the routing-grid API into pathfinding.py (RoutingGrid, build_routing_grid, cell_of,
       cell_center, locate_cell, grid_distance_field, routed_distance_function,
       polyline_flood_pixels, polyline_intersections); upgrade compute_full_path, or add a
       new function and keep a compatibility wrapper for DEVELOPMENT/paper callers.
[ ] 8. Port the time-indexed flood_spread API, or wrap it so existing callers that expect an
       ndarray still work.
[ ] 9. Port safe_dropzone obstacle-mask + filter_drops_by_home_connectivity (keep old call form valid).
[ ] 10. Merge mission_output: keep DEVELOPMENT's hardened generate_mission_file and add reload_indices;
        add validate_waypoints_file (or reuse tests/wpl_validator.py), split_sorties, the multi-base
        display_path_on_map(bases=, territories=), and the palette.
[ ] 11. Port src/mission/multi_base.py unchanged (depends only on constraints).
[ ] 12. Port src/mission/overlap_repair.py; parameterise image centre / m/px (try_reroute_leg),
        drop the unused pixel_to_latlon import, optionally refresh base_reach after a nudge.
[ ] 13. Integrate into DEVELOPMENT main.py: candidates → select_bases → plan_base/plan_assigned
        (wrapping DEVELOPMENT's per-route pipeline) → execute_overlap_repairs → overlap gate
        → territories → combined map → summary. Keep the CLI and stale-output protections.
[ ] 14. Decide the output layout (session dirs vs fixed file) and keep test_stale_mission valid.
[ ] 15. Ensure MAX_BASES=1 reproduces DEVELOPMENT's single-UAV behaviour/results where intended.
[ ] 16. Keep paper/experiments harness working (monkeypatched symbols still exist and are used).
[ ] 17. Add tests: test_multi_base.py, test_overlap_repair.py (minus the gitignored-session test),
        plus new tests for Tier-0 reassignment success, multi-base map rendering, per-base export
        validated with DEVELOPMENT's WPL validator, the territory check, and the exit-2 gate.
[ ] 18. Run the full suite; every DEVELOPMENT test must still pass; then run the 3 maps × 2 weather
        matrix and compare with §13.1.
[ ] 19. Single-vs-multi comparison runs (MAX_BASES=1 vs default) → numbers for the paper.
[ ] 20. Validate the exported base<N>.waypoints in Mission Planner/QGC (manual).
[ ] 21. Update README/Handoff/paper methodology with the multi-base method and limitations.
```

---

### Final repository-wide sweep (performed)
Searched again for `swarm|uav|fleet|multi-agent|drone_id|mapf`, `mavlink|pymavlink|dronekit|mavsdk|sitl|gazebo|px4|ardupilot|qgroundcontrol`, `md5|hashlib|sha|checksum|digest|hash`, `dbscan|sklearn|scipy|torch|tensorflow`, `.waypoints|kml|geojson|csv`. Every multi-vehicle implementation is in `multi_base.py`, `overlap_repair.py`, `main.py:360-645`, `pathfinding.polyline_intersections/grid_distance_field`, and the multi-base branch of `display_path_on_map`. No other swarm, allocation, hashing, simulation, or mission-generation code exists.