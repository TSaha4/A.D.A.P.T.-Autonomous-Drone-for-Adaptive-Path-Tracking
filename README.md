# Drone Flood-Relief Mission Planner

A computer-vision and path-planning pipeline that ingests annotated flood maps, infers safe supply drop locations, and exports an autonomous drone mission in QGroundControl waypoint format.


## Abstract

Large-scale floods in dense urban regions create severe logistical challenges for emergency responders. This project delivers an AI-powered workflow that transforms high-resolution satellite imagery into ready-to-fly drone missions. Adaptive HSV segmentation isolates floodwater, geometric clustering extracts actionable regions, safe drop zones are identified along flood boundaries, and a nearest-neighbor Traveling Salesman heuristic produces an energy-aware closed loop that begins and ends at a verified home base. The final output is a MAVLink WPL 110 waypoint file, validated on flood events in Assam, Kolkata, and Varanasi, enabling rapid deployment through standard ground-control stations.


## Why It Matters
Floods often render roads unusable, making aerial delivery one of the fastest ways to reach stranded 
communities. This project automates the tedious work of plotting mission plans by turning a single 
annotated map into a complete flight path, including safe drop zones, takeoff/landing instructions, and 
servo triggers for payload release.

- **Faster situational awareness** – Automates the mapping-to-mission pipeline so operators can focus on relief logistics.
- **Safety-aware routing** – Targets accessible perimeter points instead of risky centroids deep inside floodwater.
- **Operational compatibility** – Produces industry-standard waypoint files consumable by ArduPilot and PX4 ecosystems.


## Core Capabilities

- **Adaptive flood segmentation** powered by interactive HSV sampling and wrap-around handling for red hues.
- **Geometric clustering** with contour moments, convex hulls, and optional DBSCAN merging of adjacent inundated regions.
- **Safe drop-point discovery** along contour boundaries using point-to-segment distance minimization.
- **Time-indexed flood forecasting** – a weather-conditioned ConvLSTM (or the physics baseline) predicts the flood contour *as a sequence over future timestamps* (e.g. t+30 min, t+1 h, t+2 h), so routing can ask "what does the flood look like when the drone will actually be there".
- **Time-aware D\* Lite routing** – each flight leg is planned against the predicted obstacle contour valid at that leg's estimated arrival time, not a single mission-start snapshot.
- **Battery & payload constraints** – linear, documented (DJI Mavic-class) energy model per leg (distance, payload, headwind), payload depletion per drop, greedy return-to-base *reload* legs and battery-reserve reroutes.
- **Nearest-neighbor TSP routing** that yields a closed flight loop anchored by an automatically selected home site.
- **Home/base optimisation** – samples a grid of safe, non-flooded candidate pixels and picks the one minimising a weighted (flight time + battery) cost across the full route; alternatives are logged, so the choice is visible.
- **Mission export** as `QGC WPL 110` waypoints including takeoff, loiter, payload release, return, and landing commands (one file per *sortie* when reloads split the mission).
- **Visualization overlays** highlighting flood contours, safe zones, the home base, and the computed flight trajectory.


## Project Structure

- `main.py` – CLI entry point coordinating preprocessing, clustering, safe-point selection, forecasting, routing, home optimisation, and mission export.
- `image_processing.py` – Interactive HSV sampling, flood masking, morphological cleaning, contour filtering.
- `clustering.py` – Convex-hull and centroid computation for each flood cluster.
- `safe_dropzone.py` – Edge-based safe drop point search using distance-to-path metrics.
- `pathfinding.py` – Greedy nearest-neighbor TSP solver and the D\* Lite leg planners (single-snapshot + time-aware).
- `dstarlite.py` – D\* Lite implementation over a 2-D grid.
- `mission_planner.py` – Mission orchestrator: flood-threat ordering, capacity batches, battery/payload gates, sortie building, home-base optimisation.
- `constraints.py` – DroneSpec battery/payload model and wind helpers (sourced DJI Mavic-class anchors).
- `flood_predictor.py` – `FloodTimeline`: time-indexed contour sequence with per-ETA obstacle queries.
- `flood_spread.py` – Cellular-automata spread model (kept as the *fallback baseline* + DL training-data generator).
- `convlstm.py` – Optional PyTorch weather-conditioned ConvLSTM forecaster (Task 1 DL layer).
- `mission_output.py` – MAVLink WPL 110 file writer and matplotlib/OpenCV visualization utilities.
- `*.png` – Sample flood maps captured from Assam, Kanpur, and Varanasi events.

## Algorithms & Techniques

- **Dynamic HSV thresholding** – Samples operator clicks, expands hue/saturation/value margins, accounts 
for red wrap-around.
- **Morphological filtering** – `cv2.morphologyEx` with close/open operations to denoise masks.
- **Contour analysis** – `cv2.findContours`, area filtering, convex hull construction, image moment 
centroids.
- **Point-to-segment geometry** – Fast projection math to keep drop points near the flight path.
- **Nearest Neighbor TSP** – Greedy O(n²) heuristic that is fast, simple, and returns to the starting point.
- **Mission sequencing** – Generates takeoff, transit, loiter, descent, servo trigger, climb, RTL, and land 
commands.


## Requirements

| Component | Version/Notes |
|-----------|---------------|
| Python    | 3.8+ recommended |
| OpenCV    | `pip install opencv-python` |
| NumPy     | `pip install numpy` |
| Matplotlib| `pip install matplotlib` (for visualization) |

> Optional: Install `opencv-python-headless` on servers without display support and run with `--display-width` to skip GUI scaling.


## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install opencv-python numpy matplotlib
```

If you already have these dependencies system-wide, you can skip the virtual environment step.


## Running the Pipeline

```bash
python main.py path/to/flood_map.png \
    --display-width 750 \
    --min-area 200
```

### CLI Arguments

- `image_path` *(optional)* – Path to the annotated flood map (default: `varanasi.png`).
- `--display-width` – Resize width for the interactive window (default: 750 px).
- `--min-area` – Minimum contour area in pixels to keep (default: 200).
- `--forecast-min` – Comma-separated future timestamps (minutes) at which flood contours are predicted (default: `30,60,120`).
- `--flood-backend` – `auto` (ConvLSTM if weights exist, else physics baseline), `dl`, or `ca` (default: `auto`).
- `--package-kg` – Payload weight of each relief package (default: `0.5` kg).
- `--home-candidates` – Max home/base candidate pixels evaluated (default: `16`).
- `--path-downsample` – Downsample factor for the D\* Lite planning grid (default: `5`).
- `--horizon` – *(legacy)* forecast horizon in hours, appended to `--forecast-min`.
- `--lat` / `--lon` – Geographic centre used for pixel→lat/lon (also the weather-API query point).
- `--dummy-weather` – Use bundled extreme weather to force visible flood spread.

Most knobs can instead be set in `.env` – see `.env.example`.

### Interactive Sampling Workflow

1. An OpenCV window opens: `Select flood points (click) and press 'c'`.
2. Click multiple pixels inside the flood (red) regions to collect HSV samples.
3. Press `c` to finalize sampling and start automatic processing.

Processing logs include the derived HSV bounds, contour counts, cluster centroids, safe drop points, the time-indexed flood forecast, the chosen home base vs. alternatives, and every constraint deviation (payload reload, battery reroute, unreachable zone).

## Methodology

1. **Input & Resizing** – High-resolution satellite flood maps are read and optionally scaled to the requested display width while preserving aspect ratio.
2. **Adaptive HSV Sampling** – Operator-selected flood pixels calibrate hue, saturation, and value ranges with configurable margins and explicit red wrap-around handling.
3. **Mask Generation & Morphology** – `cv2.inRange`, followed by morphological closing/opening with a `5×5` kernel, produces a clean binary flood mask.
4. **Contour Extraction & Filtering** – External contours are identified, filtered by `--min-area`, and converted to convex hulls; optional DBSCAN clustering can merge nearby hulls.
5. **Centroid & Edge Representation** – Image moments yield centroids; hull vertices capture boundary geometry for safe-zone analysis.
6. **Safe Drop Point Search** – Each cluster's hull points are evaluated against inter-point segments of the evolving route to choose edge locations closest to the flight path, then snapped to the nearest *dry* pixel.
7. **Time-Indexed Flood Forecast** – A sequence of flood contours is predicted at `--forecast-min` timestamps (weather-conditioned ConvLSTM, falling back to the cellular-automata baseline when no weights exist).
8. **Home Base Optimisation** – A grid of safe, non-flooded candidate pixels is scored with the full time-aware route (see below) under a weighted time + battery objective; the cheapest candidate is chosen and the top alternatives are logged.
9. **Route Planning (Nearest Neighbor TSP)** – A greedy O(n²) heuristic visits the closest unserved safe point at each iteration; zones that predicted flood would reach before their ETA are pulled earlier or flagged unreachable.
10. **Time-Aware D\* Lite Legs** – Each leg is planned against the flood contour predicted for the leg's estimated arrival time (interpolating/rounding between forecast frames) using the cumulative flight time.
11. **Battery & Payload Gates** – A documented (DJI Mavic-class) linear energy model drains the battery per leg as a function of distance, carried payload, and headwind; payload depletes per drop. When the next drop would exceed the remaining load or the projected route would exceed the 80 % energy reserve, the mission returns to base – splitting the plan into *sorties* (payload reload / battery swap), or dropping the zone with a clear log when even a fresh round trip is impossible.
12. **MAVLink Mission Encoding** – Each sortie is translated into a WPL 110 file: waypoint travel, loiter, descent, `DO_SET_SERVO` payload release, climb, RTL, and land.
13. **Visualization** – OpenCV and matplotlib render the flood mask, cluster outlines, safe drops, home base, and flight path for mission validation.


## Datasets & Preprocessing

- **Sources** – Satellite flood maps sourced from Assam (2020), Kolkata (2025 monsoon), and Varanasi case studies.
- **Formats** – PNG/TIFF rasters at ≥1000×1000 resolution with floodwater annotated in red hues.
- **Preprocessing** – Rescaling to 750 px width, operator-guided HSV calibration, and morphological smoothing ensure robust cluster extraction despite lighting or sensor variability.


## Outputs

- `enriched_drone_mission.waypoints` – MAVLink WPL 110 mission file ready for upload into QGroundControl or any MAVLink GCS.  When payload reloads or battery swaps split the flight, one file per *sortie* is written (`sortie_01.waypoints`, `sortie_02.waypoints`, ...).  The WPL 110 row schema is unchanged.
- **Visualization Windows** –
  - Flood mask (binary view of thresholded regions).
  - Annotated mission map (green contours, blue safe points, black home base, red path).

Import the waypoint file into QGroundControl, verify altitude and servo parameters, and synchronize with the UAV.


## Testing

The suite is plain `unittest` (no extra test deps):

```bash
python -m unittest discover -s tests -v
```

- `test_flood_spread.py` – cellular-automata baseline + time-indexed prediction (`FloodTimeline`, per-ETA queries, interpolation).
- `test_dstarlite.py` – D\* Lite replanning + time-aware leg queries against forecast slices.
- `test_mission_constraints.py` – battery/payload model, reload & battery reroute, flood-threatened zones, home-point optimisation, graceful degradation.
- `test_weather_api.py`, `test_coordinates.py` – existing behaviour guards (pixel↔lat/lon untouched).

## Training the DL flood forecaster (optional)

The ConvLSTM needs PyTorch and a fitted weight file; until then `--flood-backend auto` transparently uses the physics baseline:

```bash
pip install torch
python scripts/train_flood_forecast.py --grid 64 --epochs 30 --out data/models/flood_convlstm.pt
```


## Evaluation & Results

- **Test Region** – Varanasi flood map used as the primary demonstration scenario.
- **Path Visualization** – Generated mission overlays show flood contours (green), safe perimeter drop zones (blue), home base (black), and the optimized closed path (red).
- **Mission File Excerpt** – The produced WPL 110 file contains the expected command sequence: takeoff (`22`), waypoint navigation (`16`), timed loiter (`19`), descent for delivery, servo trigger (`183`), return-to-launch (`20`), and land (`21`).
- **Operational Impact** – Compared with centroid-only routing, boundary-based drops reduce the risk of delivering into inundated zones and maintain efficient flight distance.


## Discussion

- **Strengths** – Practical hybrid of human-in-the-loop calibration and automation; standardized MAVLink output; safety-aware drop selection across real satellite datasets.
- **Limitations** – Relies on static imagery; HSV sampling can misclassify under complex illumination; assumes UAV hardware with sufficient range and payload.


## Future Enhancements

- **Alternate color schemes** – Adjust margins or add presets for non-red flood annotations.
- **Advanced TSP solvers** – Integrate heuristics (2-opt, simulated annealing) for larger point sets.
- **Terrain awareness** – Incorporate elevation maps or no-fly zones for safer routing.
- **Real GPS mapping** – Replace pixel-to-GPS heuristic with georeferenced map projections.
- **Batch processing** – Automate mission generation for multiple maps or live drone feeds.
- **Real-time data ingestion** – Stream satellite or aerial imagery (e.g., Sentinel-2, PlanetScope, drone video) into the pipeline for continuous updates.
- **Simulation integration** – Feed generated waypoint files into SITL, Gazebo, or Mission Planner simulators for rehearsal and validation.
- **Multi-UAV coordination** – Extend to swarm planning with energy-aware task allocation and dynamic replanning.


## Troubleshooting

- **No contours detected** – Ensure sample clicks cover representative flood pixels; lower `--min-area` as needed.
- **Image display errors** – On headless servers, install `opencv-python-headless` or disable visualization calls.
- **Unwanted drop points** – Increase morphological kernel size or adjust HSV margins to tighten segmentation.
