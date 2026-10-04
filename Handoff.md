# A.D.A.P.T. — Autonomous Drone for Adaptive Path Tracking
## Project Handoff Context

**Project:** Drone Flood-Relief Mission Planner — final-year B.Tech CSE project, VIT Vellore.
Guide: Prof. Athira K.

---

## 1. What this project does

Given an image of a flood-affected area, the system plans a full drone relief mission:

1. **Detect flood zones** — OpenCV HSV thresholding on the input image extracts flood masks and contours.
2. **Predict flood spread over time** — a weather-conditioned model (not a trained neural net — see "Known limitations") outputs **time-indexed forecast frames** (flood extent at +30min, +1hr, +2hr etc.), using live/dummy weather (wind, precipitation) from the Open-Meteo API.
3. **Identify safe drop zones** — picks delivery points on dry ground near each flood contour. Large contours (above a configurable area threshold) now get **multiple drop points** spread along their perimeter, not just one.
4. **Filter by connectivity** — drop points not reachable from home in the (downsampled) obstacle grid are excluded *before* routing, not discovered as a routing failure later.
5. **Order drops** — nearest-neighbor TSP.
6. **Route around obstacles, time-aware** — D* Lite plans paths avoiding the flood contour **predicted for each leg's estimated arrival time**, not a single static snapshot.
7. **Battery + payload aware, multi-sortie** — the drone can't do the whole mission on one charge/one payload. The route splits into **sorties**: depart base → deliver to a batch of drops → return to base to reload/recharge → repeat. Each sortie is capped at 80% battery use (20% reserve) and 8kg total payload.
8. **Base placement (single or multi-base)** — candidate base sites (dry, >= `HOME_CLEARANCE_PX` from any current/forecast flood pixel) are scored by routed (around-flood) distance. If one site can reach every coverable drop point within battery range, one base is used; otherwise up to `MAX_BASES` bases are placed (see section 8).
9. **Output** — a QGC WPL 110-format `.waypoints` file (loadable in Mission Planner / QGroundControl) with real lat/lon coordinates, plus a visualized route map (color-coded sorties, dashed return-to-base legs, directional arrows, BASE/RELOAD markers).

---

## 2. Pipeline, file by file

```
main.py                          — entry point, orchestrates the full pipeline
config/settings.py                — ALL tunable constants live here (see section 4)

src/vision/
  image_processing.py             — HSV flood mask + contour detection
  clustering.py                   — clusters contours into candidate zones

src/weather/
  weather_api.py                  — Open-Meteo integration, caching, fallback
  flood_spread.py                 — time-indexed flood spread prediction (forecast frames)

src/mission/
  safe_dropzone.py                — safe drop-point selection, multi-point for large contours,
                                     home-connectivity filtering
  constraints.py                  — battery/payload modeling, sortie splitting, home-candidate scoring
  mission_output.py               — QGC WPL 110 file generation + route map visualization
  coordinates.py / calibration.py — pixel-to-lat/lon conversion (currently fixed-scale/CLI-based;
                                     see section 5, georeferencing NOT yet autonomous)

src/routing/
  pathfinding.py                  — TSP ordering, D* Lite invocation, full-path assembly
  dstarlite.py                    — D* Lite algorithm (grid-based, time-aware obstacle queries)

tests/                            — pytest/unittest suite, one file per module above
data/input/                       — test images (varanasi.png, kanpur.png, bhopal.png),
                                     dummy_weather.json (stress test), dummy_weather_moderate.json (realistic)
data/output/session_<timestamp>/  — one folder per run: base<N>.waypoints (one per base),
                                     mission_route_map.png, mission_map_annotated.png, run_summary.json
```

**Run it:**
```bash
python main.py --image data/input/varanasi.png --dummy-weather-moderate --no-gui
```
(`--dummy-weather` = stress test / extreme conditions; `--dummy-weather-moderate` = realistic demo conditions; omit both to use live Open-Meteo data.)

- `--no-gui` samples the flood colour automatically (dominant saturated overlay hue); `--sample-points "x,y;x,y"` overrides it.
- `--lat/--lon` are the **image-centre** coordinates. For the three bundled maps they default to `REGION_PRESETS` in `config/settings.py`; for any other image they are required (the run refuses to guess a city).
- Every run writes a new `data/output/session_<YYYYmmdd_HHMMSS>/` folder (never overwrites an earlier session; a same-second run gets a `_2` suffix) and logs explicit verdict lines: `SESSION OUTPUT`, `MASK SANITY`, `STAGE COUNTS`, `BASE SELECTION`, `ASSIGNMENT TABLE`, `BASE SEPARATION`, `BASE n ROUTE REFINEMENT`, `BASE n D* LITE ROUTING`, `BASE n ROUTED SORTIE k`, `BASE n WAYPOINT VALIDATION`, `PATH OVERLAP (REPAIR|CHECK)`, `TERRITORY CHECK`, `MISSION COVERAGE`, `MAP SANITY`. Grep for these first when debugging.

---

## 3. Drone spec used for battery/payload modeling

**Garuda Aerospace Agri Kisan Drone (GA-AD)** — chosen as a real, India-available platform:
- Payload capacity: **8 kg**
- Rated speed: **5 m/s**
- Loaded endurance: **7 minutes** (manufacturer-published; this is the *only* performance figure Garuda publishes — no Wh/battery-capacity spec exists, so the drain model uses endurance-as-time-budget rather than an energy/Wh model)
- Per-drop payload: **0.25 kg** (configurable in `settings.py`)

Battery % per leg = (estimated flight time for that leg) / (7-minute loaded endurance) × 100, adjusted for payload weight and headwind component. **This is a first-order linear estimate, not a calibrated aircraft energy model** — state this plainly if asked, it's a documented, deliberate simplification, not an oversight.

**Important real-world implication:** 7 minutes at 5 m/s is a very short range (~2.1km round trip fully loaded). This is why multi-sortie planning with reload stops is essential — a single-sortie mission covering a realistically-sized flood zone will usually be infeasible with this drone spec. If you want a longer-range demo, consider swapping in a longer-endurance platform spec (document the swap and cite the new source).

---

## 4. Config — everything tunable lives in `config/settings.py`

Don't hunt through logic files for constants — they've all been centralized here, including:
- `BATTERY_RESERVE_FRACTION` (currently 20%, i.e. 80% max usable per sortie)
- `PAYLOAD_CAPACITY_KG`, `PAYLOAD_PER_DROP_KG`
- `METERS_PER_PIXEL` (coordinate scale — see section 5)
- `LARGE_CONTOUR_AREA_THRESHOLD`, `LARGE_CONTOUR_AREA_STEP`, `MAX_DROPS_PER_CONTOUR`, `MIN_DROP_SEPARATION_PX` (multi-point delivery tuning)
- Drone speed, headwind/payload drain multipliers

---

## 5. Known limitations — stated deliberately, not hidden bugs

These are the honest, documented simplifications for the paper's Limitations section — don't "discover" these as bugs, they're known and intentional for this project phase:

1. **Flood-spread prediction is a weather-conditioned heuristic, not a trained ConvLSTM/deep model.** No labeled flood-progression dataset was available in the project timeline. It does produce genuinely time-indexed forecast frames (interpolated between discrete timesteps), which is what makes the time-aware D* Lite routing meaningful — just isn't a learned model.
2. **Battery drain is a linear estimate**, not calibrated aircraft telemetry — see section 3.
3. **Georeferencing (pixel → lat/lon) is NOT yet autonomous, and the scale is not real.** The image centre is anchored per region (`REGION_PRESETS`, hand-estimated from map labels, about ±5 km), but `METERS_PER_PIXEL = 2.0` while the bundled maps are really about 100 m/px (Varanasi), 128 m/px (Kanpur) and 270 m/px (Bhopal). Waypoints therefore sit within about 1 km of the region centre, not over the actual flood pixels (offsets compressed 50-135x); every run logs a `GEOREFERENCE SCALE` warning. Using the true scale would make every drop unreachable with the 7-minute endurance. Current implementation uses a fixed-scale conversion anchored to a CLI-provided center lat/lon — this was deliberately deprioritized to focus on the core routing algorithm first. An autonomous approach (geocoding the region name via OpenStreetMap Nominatim, or full interactive calibration) was scoped but not implemented — **this is the most impactful thing to pick up next** if continuing toward a真正 field-deployable system, since accurate real-world coordinates are what make the Mission Planner simulation meaningful.
4. **Flood reprioritization at each drop's arrival time uses a single-pass reorder heuristic**, not exhaustive search over all orderings.
5. **Reload stops are represented as standard land/takeoff commands** in the mission file — there's no mechanism to verify a human actually reloads the drone before it continues.
6. **Pixel-to-GPS conversion assumes flat-earth approximation** — fine at the scale of a single city/district, not for larger areas.

---

## 6. What's solid vs. what needs work (from the last engineering pass)

**Solid — don't rebuild from scratch, extend carefully:**
- `safe_dropzone.py` — multi-point perimeter generation, connectivity filtering
- `pathfinding.py` — D* Lite with time-aware obstacle queries, conservative dilation to prevent obstacle-grid clipping
- `mission_output.py` — multi-sortie visualization, QGC WPL 110 export
- `constraints.py` — sortie-splitting battery/payload logic (this took ~8 debugging rounds to get right — read the git history/commit messages before touching it, several subtle bugs here already got found and fixed: km/h↔m/s unit bug, payload double-counting, cumulative-vs-per-sortie battery comparison, pre-commit vs post-commit reserve checks)

**Needs further work — legitimate next milestones, not bugs:**
- **Autonomous georeferencing** (see section 5.3) — highest-impact next step
- **In-flight dynamic replanning** — current D* Lite plans the whole route upfront; real-time obstacle injection during flight (SITL/MAVLink integration) is stubbed, not implemented
- **True 2D wind-vector costs in D* Lite** — wind currently applies as a uniform leg-level drain penalty, not directional per-grid-cell cost
- **Real-time multi-agent coordination (MAPF)** — multi-base planning (section 8) assumes bases operate independently with no live coordination. That is valid because bases are separated, serve mutually exclusive drop sets, and routes are verified not to intersect; collision-aware multi-agent path finding (time-indexed conflicts, live deconfliction) was not attempted and is future work

---

## 7. Debugging history worth knowing

The battery/sortie logic went through many rounds of real bugs before stabilizing — if something looks broken in `constraints.py`, check this list before assuming it's new:
- Wind speed unit mismatch (Open-Meteo returns km/h, model expected m/s) — fixed
- Payload counted twice in per-leg drain calculation — fixed
- Sortie-splitting compared *cumulative total route drain* against the *single-sortie* 80% threshold (category error — total naturally exceeds 100% across multiple sorties) — fixed
- Sortie battery check fired *after* committing a leg instead of *before* — allowed sorties up to 192% battery use — fixed
- Flood-forecast interpolation between timesteps was a no-op (always returned the lower-bound frame) — fixed
- Early forecast frames contained non-binarized intermediate pixel values — fixed
- Drop points sometimes landed in a disconnected region of the downsampled obstacle grid, causing D* Lite to correctly fail — now filtered before routing, not discovered mid-plan
- Map visualization was drawing a misleading straight-line "TSP Direct Path" overlay on top of the real D* Lite route, making correct routes look like they cut through flood zones — removed
- D* Lite node snapping previously checked obstacle presence without verifying connected component membership, occasionally snapping a goal node across a boundary into an isolated pocket in a different component. Fixed: snapping now verifies connected-component identity with start node's component and includes graceful direct fallback.

**October 2026 stabilization audit:** each item below was reproduced before fixing and is covered by `tests/test_audit_regressions.py` or `tests/test_mission_constraints.py`.
- `--no-gui` used Varanasi-calibrated sample pixels on every image (on Kanpur they hit grey land). The HSV threshold itself is the original min/max-over-patch algorithm (an audit-era percentile variant made interactive masks far too sparse and was reverted; `test_original_hsv_algorithm_reproduces_reference_varanasi_mask` pins it). `--no-gui` now auto-picks 3 sample points the way an operator clicks: on flood-overlay pixels whose 11x11 sampling patch is about half overlay. A mask-coverage guard (0.1-50%) aborts on a near-empty or near-total mask.
- Contouring is the original algorithm (HSV mask -> 5x5 MORPH_CLOSE + MORPH_OPEN -> `findContours(RETR_EXTERNAL)` -> area >= `--min-area` (200) -> per-contour convex hull + centroid in `cluster_contours`). Density zones, exact-dot contours and hull grouping were tried in Oct 2026 and reverted; small isolated flood spots below 200 px^2 are intentionally not zones.
- Large zones took drop candidates only from the first dry ring, which can be a single notch, so all perimeter points collapsed to one. They now search the whole band.
- `--lat/--lon` defaulted to Varanasi for every image, so both weather and waypoints were placed in Varanasi. Replaced by per-image presets; unknown images require explicit coordinates.
- Sortie reserve check ignored the return-to-base leg, and carried payload *increased* along the sortie. Routed sorties reached 103-134% battery in the baseline. `plan_mission_stops` now evaluates whole sorties (decreasing payload, RTB included) like the home scorer, and `sortie_battery_report` re-checks the exported geometry.
- `obstacle_mask_at` blended binary frames and thresholded at 128, so every new flood pixel appeared at alpha = 0.5 (a step function). It now interpolates signed distance fields, so the flood front advances proportionally. Forecast frames are now also cumulative (never recede).
- Snapping moved points that sat in a *free* cell on the far side of an obstacle into HOME's component, so routes ended on the wrong side of a wall. `locate_cell` (shared by the connectivity filter and D* Lite) now snaps only blocked cells, within 4 cells, with a flood-free connector.
- The routing grid sampled one pixel per 5x5 block, which let thin flood strips slip through. Cells are now blocked if any (1-px dilated) pixel in the block is flooded.
- Drop points were placed against the t=0 mask but filtered and routed against the horizon mask, so many sat inside forecast flooding. They are now placed against the horizon mask (dry for the whole forecast).
- D* Lite failures fell back to an unlogged straight line. Fallbacks are now logged as `FALLBACK STRAIGHT-LINE SEGMENT (not a D* Lite route)` with a flooded-pixel crossing count; partial or blocked D* Lite paths are rejected.
- DO_SET_SERVO rows had servo=0, PWM=0 (2000 sat in param3). Now param1 = `DROP_SERVO_CHANNEL`, param2 = `DROP_SERVO_PWM`; the home row uses frame 0; `validate_waypoints_file` checks the format.
- Map: RTB dashes restarted on every 5 px grid step and rendered solid; RELOAD labels overprinted BASE; the legend hid route segments; contours were the same green as sortie 3. All fixed. `basemap_sanity` (no colour >50%, correlation with the source image) runs every time and is unit-tested against solid and striped corruption.
- HOME could land in a dry pocket enclosed by flood: straight-line scoring favoured it and the connectivity filter then removed almost every drop (Kanpur moderate flew one sortie with 4 drops). HOME is now restricted to the routing-grid free region holding the most drops, and candidates are scored with routed (around-flood) distances (`routed_distance_function`); Kanpur moderate now flies 7 sorties, 21 drops.
- Home candidates were taken every 18th dry pixel in raster order (about 27k candidates, about 80 s per run) instead of a 40x40 spatial grid (about 1.6k).
- The refinement loop logged "stabilized" for both true convergence and cycle detection; it now distinguishes converged / cycle / iteration cap. Routed distances are keyed by exact leg and accumulate across iterations.

If you hit something that looks like deja vu from this list, check git blame/history first — it was probably already fixed once and may have regressed.

---

## 8. Multi-base planning, map palette and session output (October 2026)

**When it triggers.** `src/mission/multi_base.py::select_bases` computes, for every candidate base site, which drop points a single-drop sortie can serve (routed distance out with one package and back empty, worst-case headwind on both legs, within the 80% usable battery: the same test `plan_mission_stops` applies). If one site reaches every coverable drop point, a single base is used (minimum routed cost via the existing `score_home_candidates`). Otherwise multi-base planning runs.

**Placement criteria (all enforced):**
1. *Separation:* no two bases closer than `MIN_BASE_SEPARATION_PX` (default 150); violating sites are rejected during selection and re-placement.
2. *Non-overlapping paths:* each drop point is assigned to exactly one base (lowest routed distance among bases that can reach it). After routing, every pair of bases' routes is checked for segment intersections (`pathfinding.polyline_intersections`). If crossings exist:
   - *Stage 0 (Reassignment):* if the other base can reach the crossing sortie's drops within battery, reassign and re-plan both.
   - *Tier 1 (Local Reroute):* if reassignment fails, locally re-run D* Lite for the specific crossing legs with the conflicting route segments added as temporary soft obstacles, detouring around the conflict while verifying flood clearance and sortie battery limits.
   - *Tier 2 (Base Nudging):* if rerouting fails, perform a local search around the conflicting base position (within safe territory, flood clearance >= 15 px, base separation >= 150 px) and re-plan.
   - *Hard Error (sys.exit(2)):* if all repair tiers fail to eliminate crossings, the mission blocks execution with exit code 2 rather than shipping overlapping paths.
3. *Shortest-distance coverage:* greedy fewest-bases cover (each new base is the separated site covering the most uncovered drops, ties broken by routed distance), then each base is re-placed at the minimum routed-cost valid site for its assigned drops using the existing routed home scorer; this repeats while drops remain uncovered and bases remain under `MAX_BASES` (default 4).
4. *Flood clearance:* every candidate site, single or multi-base, keeps `HOME_CLEARANCE_PX` (15 px) from any current or forecast flood pixel.

**Assignment table and unreachable drops.** The log lists every drop point with its base and routed distance. Drops no base can serve are logged as `UNREACHABLE BY ANY BASE` with the reason (out of battery range from every flood-clear site; every reaching site is within `MIN_BASE_SEPARATION_PX` of an existing base; or the base cap was reached). Nothing is dropped silently; `run_summary.json` has `assignment` and `unreachable_by_any_base`.

**Per-base planning.** Each base runs the unchanged single-drone pipeline (connectivity filter, TSP, time-aware D* Lite, battery/payload sortie splitting with reloads, refinement) on only its drops, and writes `base<N>.waypoints`. The combined map shows every base (numbered black houses), all sorties, and territory boundaries: each grid cell belongs to the base whose service points (base + its visited drops) are routed-closest, so territories cannot overlap, and the run checks that every visited drop lies in its own base's territory (`TERRITORY CHECK`).

**Map palette.** Sortie paths use 12 dark, saturated colours: `#d50000` red, `#1a237e` navy, `#f200f2` magenta, `#d86191` rose, `#3990bf` steel blue, `#590016` wine, `#6248f2` violet, `#d87700` orange, `#6d87f2` periwinkle, `#a51897` purple, `#f20054` crimson, `#59004b` plum. No greens or teals (green is reserved for detected flood contours). Each pair differs by at least CIELAB dE 34, and each is at least dE 35 from drop-point blue, BASE black, RELOAD cyan/yellow, contour green and forecast brown (`TestMapPalette` enforces this). More than 12 sorties in one map would repeat colours, and the run logs `MAP PALETTE` if so.

**Validation at handoff:** 84/84 unit tests (`python -m unittest discover tests`); 3 images x 2 weather presets (6-run matrix) all pass crash-free with ZERO path overlaps, mask sane, bases found, route valid (0 fallback legs, 0 route pixels in flood), battery sane (all sorties <= 80%), map renders, waypoints valid (one per base), base separation, no path overlap, territories consistent, all drop points assigned or reported unreachable.

## 9. Still open after the audit (known, not hidden)

- **Home scoring uses straight-line distances.** A home can be "feasible" for every drop, yet routed detours later make some drops battery-unreachable (1-7 per run in the audit matrix). Those drops are reported as `unreachable_battery`, never flown over budget.
- **Coverage is limited by the 7-minute endurance** even with multi-base: 3-17 drop points per run remain unreachable by any base (logged with reasons). Greedy base placement is an approximation of the fewest-bases cover, not a proven optimum.
- **The 1-pixel flood advance in moderate weather** necessarily flips at one instant (around alpha 0.5); interpolation is only visibly gradual for multi-pixel advances.
- **HOME keeps `HOME_CLEARANCE_PX` (15 px) from any current or forecast flood pixel** when such a point exists (16-21 px in the audit runs); otherwise it falls back to any dry point with a warning.
- **Overlapping sorties** to the same corridor draw on top of each other on the map.
- **README.md is outdated** (describes the pre-D* Lite pipeline). `scripts/test_pathfinding_dummy.py` still deliberately plots a labelled "TSP Direct Path" comparison line; it is a dev plot, not the mission map.

*Handoff updated October 2026 after the stabilization audit and path overlap repair fix (84/84 unit tests passing via `python -m unittest discover tests`). Main open item: autonomous georeferencing and real map scale.*