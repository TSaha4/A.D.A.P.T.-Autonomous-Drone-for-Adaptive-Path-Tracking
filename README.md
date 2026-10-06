# A.D.A.P.T. — Autonomous Drone for Adaptive Path Tracking

A.D.A.P.T. converts a flood-annotated map image and current weather into a UAV relief mission in
QGroundControl `QGC WPL 110` format. The mission takes off from a home point, visits one drop point
on the boundary of every flooded region, releases a payload at each, and returns home.

> **Status: research prototype — not flight-ready.** The geospatial and mission-generation code is
> extensively verified in software. However, the bundled sample maps are **not georeferenced**, and no
> mission has been tested in Mission Planner, in SITL simulation, or in flight. Do not fly generated
> missions without the checks listed under [Safety](#safety).

## What it does

```
flood map image ──► display scaling ──► operator HSV sampling + morphology ──► flood mask M
                                                                          │
weather (Open-Meteo or file) ──► weather-driven spread rule ──► predicted mask M̂ ──► obstacles O = M ∪ M̂
                                                                          │
flood regions ──► boundary drop points + HOME ──► nearest-neighbour visit order ──► per-leg D* Lite on O
                                                                          │
                              pixel → latitude/longitude (with validity checks) ──► QGC WPL 110 mission
```

1. **Display scaling.** Images wider than `--display-width` are downscaled. All later stages work
   in display pixels.
2. **Flood segmentation.** The operator clicks flood pixels. HSV thresholds are taken from 11×11
   patches around the clicks (hue margin 20, S/V margin 30). Red hues use fixed wrap-around bands
   `[0,10] ∪ [170,179]`. A 5×5 morphological close then open cleans the mask.
3. **Regions.** External contours of area ≥ `--min-area` px², with their convex hulls and centroids.
4. **Weather.** Current precipitation, wind speed and wind direction from the
   [Open-Meteo](https://open-meteo.com) API, or a fixed file with `--dummy-weather`. If the API
   fails, calm weather is used and a warning is printed.
5. **Flood-spread prediction.** A heuristic, terrain-free rule run for `--horizon` hours:
   - growth of 1 px per 5 mm of precipitation in all directions (dilation);
   - a shift of 1 px per 10 km/h of wind, downwind.

   It is used only to keep the route away from areas that may flood soon. It is **not** a
   hydrological model and is uncalibrated.
6. **Drop points and HOME.** One convex-hull vertex per region is chosen as its drop point. HOME is
   the drop point nearest the image centre, and its region still receives a delivery.
7. **Visit order.** A greedy nearest-neighbour tour from HOME and back. It is not optimal.
8. **Path planning.**
   - The obstacle mask is downsampled 5× to a grid.
   - Endpoints that fall inside obstacles are moved to the nearest free cell (breadth-first search,
     radius 10).
   - Each leg is planned with D* Lite.
   - If a leg has no path, the mission flies it as a **straight segment** and an error is logged.
9. **Geographic conversion.** A flat-earth (equirectangular) conversion about the image centre,
   using `--lat/--lon` and `METERS_PER_PIXEL`. Invalid inputs are rejected (see
   [Geographic conversion](#geographic-conversion)).
10. **Mission.** HOME, then TAKEOFF to 100 m, then waypoints. At each drop point: LOITER 5 s, descend
    to 10 m, `DO_SET_SERVO` (channel 9, 2000 µs), climb back to 100 m. The mission ends with RTL and
    LAND.

## Repository layout

```
main.py                     CLI entry point (runs the whole pipeline)
config/settings.py          configuration from environment / .env
src/vision/                 image_processing.py (HSV sampling, mask, contours), clustering.py (hulls, centroids)
src/weather/                weather_api.py (Open-Meteo client + fallback), flood_spread.py (spread rule)
src/mission/                safe_dropzone.py, coordinates.py (pixel -> lat/lon), mission_output.py (WPL writer, plot)
src/routing/                pathfinding.py (NN ordering, snapping, per-leg planning), dstarlite.py (D* Lite)
data/input/                 sample maps (varanasi.png, kanpur.png, assam.png) and dummy_weather.json
data/output/                generated mission (enriched_drone_mission.waypoints)
tests/                      test suite, independent geodesy reference and mission validator
scripts/                    test_pathfinding_dummy.py (visual routing demo on a synthetic grid)
paper/                      research manuscript, evaluation scripts and results (see paper/README.md)
```

## Setup

The pipeline requires Python 3.8+. The test suite and the paper's evaluation scripts need Python
3.11+. All results reported here were produced with Python 3.11.9.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # bash: source .venv/Scripts/activate (Windows) or .venv/bin/activate
pip install -r requirements.txt
pip install pytest                  # to run the tests
```

If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once. Note that the
activation command has no trailing backslash.

To reproduce the exact package versions used in the paper, install from
`paper/experiments/requirements-paper.txt` instead.

On a server without a display, install `opencv-python-headless`. Note that the colour-sampling
step needs a window.

## Configuration

Copy `.env.example` to `.env` and edit it. Every variable is optional.

| Variable | Default | Meaning |
|---|---|---|
| `WEATHER_API_BASE_URL` | `https://api.open-meteo.com/v1/forecast` | Weather endpoint |
| `MAX_SAFE_WIND_SPEED` | `40.0` | km/h; a warning is printed above this (the run is not stopped) |
| `MAX_SAFE_PRECIPITATION` | `15.0` | mm; a warning is printed above this (the run is not stopped) |
| `METERS_PER_PIXEL` | `2.0` | Ground metres per pixel of the **original** input image |

**`METERS_PER_PIXEL` and `--lat/--lon` determine where the mission is placed in the world.** Set them
correctly for each map:

- `--lat/--lon` is the ground position of the **image centre**;
- `METERS_PER_PIXEL` is the map's true scale.

The defaults (Varanasi city, 2.0 m/px) do not describe the bundled maps. Landmark-based estimates
put their scales at roughly 93, 130 and 633 m/px.

## Usage

```powershell
python main.py data/input/varanasi.png --dummy-weather
python main.py path/to/map.png --lat 26.4499 --lon 80.3319 --horizon 3
```

| Argument | Default | Description |
|---|---|---|
| `image_path` | `data/input/varanasi.png` | Flood-annotated map image |
| `--display-width`, `-w` | `750` | Display width in pixels (images wider than this are downscaled; must be > 0) |
| `--min-area` | `200` | Minimum flood-region area (display px²) |
| `--lat`, `--lon` | `25.3176`, `82.9739` | Image-centre latitude/longitude; used for the weather query **and** mission coordinates |
| `--horizon` | `2.0` | Flood-spread horizon in hours (whole hours are used, minimum 1) |
| `--dummy-weather` | off | Use `data/input/dummy_weather.json` (10 mm, 30 km/h, from 270°) instead of the API |

**Interactive step:**

1. A window titled `Select flood points (click) and press 'c'` opens.
2. Click several pixels inside the flooded (red) areas, then press `c`.
3. A flood-mask window follows; press any key to continue.
4. The final route plot opens in matplotlib.

**Exit codes:**

| Code | Meaning |
|---|---|
| `0` | Success, or nothing to deliver (no regions found) |
| `1` | Image could not be read, point selection failed, a later stage failed, or an earlier mission file could not be removed |
| `2` | Invalid `--display-width`, `--lat/--lon` or `METERS_PER_PIXEL`. These are checked before any work is done, and an earlier mission is left untouched |

## Output

### Mission file

The mission is written to `data/output/enriched_drone_mission.waypoints`. It is tab-separated
`QGC WPL 110`, and every positional item uses frame 3 (altitude relative to home).

| Items | Command (id) | Altitude | Notes |
|---|---|---|---|
| 0 | `NAV_WAYPOINT` (16) | 0 m | HOME |
| 1 | `NAV_TAKEOFF` (22) | 100 m | at HOME |
| each route point | `NAV_WAYPOINT` (16) | 100 m | cruise |
| each drop point (extra) | `NAV_LOITER_TIME` (19), `NAV_WAYPOINT` (16), `DO_SET_SERVO` (183), `NAV_WAYPOINT` (16) | 100 → 10 → 10 → 100 m | 5 s loiter; servo param1 = channel 9, param2 = 2000 µs |
| last two | `NAV_RETURN_TO_LAUNCH` (20), `NAV_LAND` (21) | 100 m, 0 m | at HOME |

The output file is managed as follows:

- **Converted before writing:** every coordinate is converted and checked before the file is opened,
  so a failure never leaves a partial mission.
- **Previous mission removed:** once the input image has loaded, the mission from any earlier run is
  deleted, so a run that later fails cannot leave an old mission that looks new.

### Plot

The plot shows:

- current flood contours in green;
- predicted spread in orange;
- drop points as blue dots;
- HOME as a black dot;
- the planned D* Lite path as blue lines;
- the nearest-neighbour visiting order as red dashed lines.

## Geographic conversion

```
E = (x − cx)·k,   N = (cy − y)·k,   k = METERS_PER_PIXEL / display scale
lat = lat0 + N / 111320,   lon = lon0 + E / (111320 · cos lat0)
```

`(cx, cy) = ((W'−1)/2, (H'−1)/2)` is the geometric centre of the display image.

The conversion raises an error, and no mission is written, for any of the following:

- non-finite inputs;
- `METERS_PER_PIXEL ≤ 0`;
- a reference latitude of exactly ±90° or outside that range, or a longitude outside ±180°;
- an offset that would cross a pole;
- an east–west offset spanning half a parallel or more.

Longitudes are wrapped across ±180° only when they fall outside that range.

The model assumes the image is north-up, with square pixels of uniform scale. Its accuracy against
WGS84:

- **Scale error from the 111,320 m/° constant:** about 0.5 % north–south at 25° latitude.
- **Error over large areas:** grows with map size, for example about 345 m across a 90 km map.
- **Web-map screenshots:** these use Web Mercator, which violates the uniform-scale assumption.

## Testing and verification

```powershell
python -m pytest            # 126 tests + 494 subtests, about 3 s
```

The suite covers the following:

- **Coordinate conversion**
  - **Ground truth:** synthetic points checked against an independent WGS84 (Vincenty) reference in
    `tests/geodesy_reference.py`, itself checked against a published geodesic.
  - **Edge cases:** poles, the antimeridian, NaN/inf, and zero or negative scale.
  - **Round trips:** randomised round trips through a test-only inverse.
- **Resize consistency:** the same map at multiple display widths must give the same ground
  positions.
- **Image centre:** the reference pixel is checked for even and odd image sizes.
- **Mission structure:** HOME-region drop, servo parameter slots, and the absence of partial or
  stale missions.
- **D* Lite and snapping, clustering, flood spread and the weather client.**

Missions can also be checked with the independent validator. It does not import the production
code; it re-derives every row from the route when a sidecar file is supplied.

```powershell
python -m tests.wpl_validator data/output/enriched_drone_mission.waypoints --servo-channel 9 --servo-pwm 2000
python -m tests.wpl_validator MISSION.waypoints --sidecar ROUTE.json --table items.csv
```

The validator is itself exercised by 36 adversarial tests built from corrupted missions.

## Evaluation summary

Results come from the bundled maps with fixed weather and a simulated operator. Details, the
scripts and the raw data are in `paper/experiments/`.

| | Varanasi | Kanpur | Assam |
|---|---|---|---|
| Flood-mask fraction | 8.1 % | 15.1 % | 83.1 % (segmentation failure: white background passes the red test) |
| Regions = deliveries | 16 | 33 | 1 |
| Legs with no path (flown straight) | 8 of 16 | 6 of 33 | 0 |
| Mission items | 364 | 1145 | 10 |
| Validator | pass | pass | pass |

Other findings:

- **Predicted-flood obstacles are a trade-off.** They reduce the share of the route over predicted
  flooding, from 33.8 % to 17.9 % (Varanasi) and from 47.9 % to 11.6 % (Kanpur). But they cause
  every no-path leg.
- **Route order:** nearest-neighbour ordering is 3.2 % longer than optimal on Varanasi.
- **D* Lite speed:** D* Lite returns the same optimal paths as A* but, as used here (rebuilt per
  leg), is 61–112× slower.
- **Coordinate accuracy:** mission rows match their source pixels to within 4×10⁻¹⁵°.
- **Mutation testing:** 22 of 24 injected bugs are detected by the test suite.

## Known limitations

- **Geography:**
  - The sample maps carry no georeferencing.
  - The default reference point and scale are wrong for them, so the generated coordinates do not
    correspond to the flooded places.
- **Segmentation:**
  - Operator-dependent, and works only for colour-annotated maps (not raw imagery).
  - Fails on thematic maps with white backgrounds.
  - No ground-truth accuracy has been measured.
- **Flood-spread rule:**
  - Heuristic, terrain-free, uncalibrated and defined in pixels.
  - Open-Meteo's current precipitation is an accumulation over a short interval but is treated as
    an hourly rate.
- **Drop points:** chosen by geometry alone, with no terrain, structure or population safety check.
- **Planning:**
  - Endpoints moved off obstacles can land in enclosed pockets; those legs are flown straight,
    possibly over water.
  - The grid samples one pixel per 5×5 block.
  - D* Lite's incremental replanning is unused.
- **Mission:**
  - The servo is never reset between drops.
  - The LAND after RTL is redundant on ArduCopter.
  - Waypoint yaw is written as 0, which PX4 interprets as "face north".
  - Mission length and endurance are not checked against the vehicle.
- **Validation:** no Mission Planner, SITL, HITL or flight testing has been done.

## Safety

Before any real flight, at minimum:

1. Use georeferenced inputs and check every waypoint against a base map.
2. Load the mission in a ground station and rehearse it in SITL.
3. Configure geofences, altitude limits and failsafes for link loss, battery and GNSS.
4. Fly with a safety pilot.
5. Confirm drop points are clear of people and structures.
6. Ground-test the release mechanism with the configured servo channel and PWM.
7. Follow local aviation regulations.

## Research paper

`paper/` contains an IEEE-format manuscript describing and evaluating this system. It also holds
all evaluation scripts, their raw results, and the audit files (`paper_facts.md`,
`claim_evidence_map.csv`, `literature_matrix.csv`). Build and reproduction instructions are in
`paper/README.md`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Activate.ps1 is not recognized` | Remove any trailing `\`; set the execution policy as described in Setup |
| `Could not open requirements file` | Check the spelling: `requirements.txt` |
| `No significant flood areas detected` | Click more representative flood pixels, or lower `--min-area` |
| Most of the map becomes flood | Clicks picked up background; click the centre of large flood patches. On white-background thematic maps this is a known failure |
| `No path found … straight-line fallback` | Expected when a drop point is enclosed by obstacles; inspect those legs before flight |
| `Invalid --lat/--lon or METERS_PER_PIXEL` (exit 2) | Provide a valid image-centre latitude/longitude and a positive scale |
| `Could not remove mission file` (exit 1) | The earlier mission is read-only, or a folder sits at the output path; remove it manually |
| `Weather API unavailable` warning | No network or API error; calm weather is used, so the predicted spread is minimal |
| Display errors on a headless machine | Install `opencv-python-headless`; note that the click step needs a GUI |

## Future work

- Run Mission Planner/QGroundControl load tests and ArduPilot/PX4 SITL.
- Support georeferenced inputs (e.g. GeoTIFF) and correct map projections.
- Collect segmentation ground truth.
- Handle disconnected drop points explicitly instead of flying straight legs.
- Use a calibrated or terrain-aware flood model.
- Add endurance-aware and multi-UAV routing.
