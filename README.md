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
- **Nearest-neighbor TSP routing** that yields a closed flight loop anchored by an automatically selected home site.
- **Mission export** as `QGC WPL 110` waypoints including takeoff, loiter, payload release, return, and landing commands.
- **Visualization overlays** highlighting flood contours, safe zones, the home base, and the computed flight trajectory.


## Project Structure

- `main.py` – CLI entry point coordinating preprocessing, clustering, safe-point selection, routing, and mission export.
- `image_processing.py` – Interactive HSV sampling, flood masking, morphological cleaning, contour filtering.
- `clustering.py` – Convex-hull and centroid computation for each flood cluster.
- `safe_dropzone.py` – Edge-based safe drop point search using distance-to-path metrics.
- `pathfinding.py` – Greedy nearest-neighbor TSP solver that returns closed loops.
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

### Interactive Sampling Workflow

1. An OpenCV window opens: `Select flood points (click) and press 'c'`.
2. Click multiple pixels inside the flood (red) regions to collect HSV samples.
3. Press `c` to finalize sampling and start automatic processing.

Processing logs include the derived HSV bounds, contour counts, cluster centroids, safe drop points, and travel order.


## Methodology

1. **Input & Resizing** – High-resolution satellite flood maps are read and optionally scaled to the requested display width while preserving aspect ratio.
2. **Adaptive HSV Sampling** – Operator-selected flood pixels calibrate hue, saturation, and value ranges with configurable margins and explicit red wrap-around handling.
3. **Mask Generation & Morphology** – `cv2.inRange`, followed by morphological closing/opening with a `5×5` kernel, produces a clean binary flood mask.
4. **Contour Extraction & Filtering** – External contours are identified, filtered by `--min-area`, and converted to convex hulls; optional DBSCAN clustering can merge nearby hulls.
5. **Centroid & Edge Representation** – Image moments yield centroids; hull vertices capture boundary geometry for safe-zone analysis.
6. **Safe Drop Point Search** – Each cluster's hull points are evaluated against inter-point segments of the evolving route to choose edge locations closest to the flight path.
7. **Home Base Selection** – The safe drop point nearest to the map center is marked as the takeoff/landing site for mission symmetry.
8. **Route Planning (Nearest Neighbor TSP)** – A greedy O(n²) heuristic visits the closest unserved safe point at each iteration and returns to home, balancing rapid computation with efficient flight distance.
9. **MAVLink Mission Encoding** – The ordered points are translated into WPL 110 commands: waypoint travel, loiter, descent, `DO_SET_SERVO` payload release, climb, RTL, and land.
10. **Visualization** – OpenCV and matplotlib render the flood mask, cluster outlines, safe drops, home base, and flight path for mission validation.


## Datasets & Preprocessing

- **Sources** – Satellite flood maps sourced from Assam (2020), Kolkata (2025 monsoon), and Varanasi case studies.
- **Formats** – PNG/TIFF rasters at ≥1000×1000 resolution with floodwater annotated in red hues.
- **Preprocessing** – Rescaling to 750 px width, operator-guided HSV calibration, and morphological smoothing ensure robust cluster extraction despite lighting or sensor variability.


## Outputs

- `enriched_drone_mission.waypoints` – MAVLink WPL 110 mission file ready for upload into QGroundControl or any MAVLink GCS.
- **Visualization Windows** –
  - Flood mask (binary view of thresholded regions).
  - Annotated mission map (green contours, blue safe points, black home base, red path).

Import the waypoint file into QGroundControl, verify altitude and servo parameters, and synchronize with the UAV.


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
