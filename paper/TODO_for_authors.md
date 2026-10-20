# TODO for the authors (must be resolved before submission)

## 1. Placeholders in the manuscript

| Where | Placeholder | Action |
|---|---|---|
| `main.tex` author block | `[Author 1]`, `[Author 2]`, `[Faculty Advisor]`, department, institution, city, country, e-mail | Fill in; add ORCID iDs if the venue requests them |
| `main.tex` first footnote | `[Funding statement]` | Provide, or write "This work received no external funding." |
| `sections/backmatter.tex` | `[REPOSITORY URL]` | Public repository URL; archive a release (e.g. Zenodo) and cite its DOI |
| `sections/backmatter.tex` | Acknowledgment: funding/institutions; extent of the authors' own review of AI-drafted content | Complete. IEEE requires disclosure of the AI system and of which sections it produced (already listed) |
| `main.tex` | `\markboth{Draft manuscript --- not peer reviewed}` | Replace when porting to the venue template |

## 1b. Figure placeholders for your own simulation screenshots

The manuscript contains nine empty, framed placeholders drawn with the `\simplaceholder{TAG}{width}{height}`
macro (`main.tex`). To insert an image, put the file in `figures/` and replace the whole
`\simplaceholder{...}{...}{...}` call by `\includegraphics[width=...]{figures/<file>}`; keep the caption,
but delete "[To be inserted by the authors.]" and adjust the wording to what the image shows.

| Tag (search for it) | Figure label | File | How to produce the image |
|---|---|---|---|
| INPUT FLOOD MAP | `fig:sim-input` (a) | `sections/experiments.tex` | The input PNG (e.g. `data/input/varanasi.png`), or the window shown after loading |
| FLOOD REGION DETECTION | `fig:sim-input` (b) | `sections/experiments.tex` | The "Flood Mask" window that `main.py` shows after you click the flood colour |
| SAFE DROP POINTS | `fig:sim-input` (c) | `sections/experiments.tex` | Single-UAV mode: the final map window (blue dots); multi-base mode: `mission_route_map.png` |
| SINGLE-UAV OPTIMIZED ROUTE (Approach A: one drone) | `fig:sim-single` | `sections/experiments.tex` | `python main.py data/input/varanasi.png --dummy-weather` → final map window. The caption must keep saying the order is heuristic, not optimal |
| MULTI-UAV TASK ALLOCATION (Approach B: drone fleet; show each drone's cluster) | `fig:sim-multi` (a) | `sections/experiments.tex` | `python main.py data/input/varanasi.png --mode multi` → `ASSIGNMENT TABLE` in the log and `run_summary.json` (`assignment`, `unreachable_by_any_base`); e.g. a screenshot of the log, or a plot of the bases and their drops |
| MULTI-UAV ROUTES | `fig:sim-multi` (b) | `sections/experiments.tex` | Same run: `data/output/session_<time>/mission_route_map.png` |
| SINGLE VS MULTI UAV | `fig:sim-compare` | `sections/experiments.tex` | `--mode multi --max-bases 1` and `--mode multi --max-bases 4` on the same map and weather (or `python paper/experiments/multi_uav_comparison.py --keep-maps`); place the two maps side by side |
| UAV SIMULATION | `fig:sim-exec` (a) | `sections/experiments.tex` | ArduPilot SITL (see below) flying an exported mission |
| MISSION EXECUTION | `fig:sim-exec` (b) | `sections/experiments.tex` | QGroundControl / Mission Planner during the SITL flight (mission view, a drop) |

**Important:** Figs. `fig:sim-exec` (a, b) correspond to work that has **not** been done. When you insert
them, also update `sections/experiments.tex` (subsection "Simulation and Mission Execution"), the lower
block of Table `tab:verif` ("Software-in-the-loop … not performed"), the abstract ("No ground-station,
simulation or flight validation has been performed") and `sections/limitations.tex` (operational
validity), so that the text reports what the simulation actually showed.

### Simulation steps (both modes)

1. Generate missions:
   - single-UAV: `python main.py data/input/varanasi.png --dummy-weather` →
     `data/output/enriched_drone_mission.waypoints`;
   - multi-UAV: `python main.py data/input/varanasi.png --dummy-weather --mode multi [--max-bases 4]` →
     `data/output/session_<time>/base<N>.waypoints` (one per UAV), `base<N>.route.json`, `run_summary.json`,
     `mission_route_map.png`.
2. Check every file independently first:
   `python -m tests.wpl_validator <file>.waypoints --sidecar <file>.route.json --servo-channel 9 --servo-pwm 2000`
   (multi-base files contain reload stops; the sidecar declares them).
3. Ground-station load test: open each `.waypoints` file in QGroundControl or Mission Planner and record
   item counts and warnings.
4. ArduPilot SITL: one vehicle per base. Start each with its own instance and home location, e.g.
   `sim_vehicle.py -v ArduCopter -I <n> --custom-location=<lat>,<lon>,0,0` using the HOME row (item 0) of
   `base<n>.waypoints`, connect QGroundControl, upload the mission, arm, and switch to AUTO.
5. Record: mission acceptance, servo-9 PWM at each drop, drop altitude, behaviour at each reload stop
   (LAND then TAKEOFF: does the vehicle disarm after landing and must AUTO be restarted?), RTL/LAND at the
   end, and, for several vehicles, their minimum distance during the flight (the planner guarantees only
   that routes do not cross, not time or altitude separation).
6. The missions are not geographically meaningful for the bundled maps (they sit near the configured
   reference); this is fine for SITL behaviour checks but must be stated in the captions.

## 2. Missing metrics, assumptions and decisions

- **Segmentation accuracy (E6):** no ground-truth masks exist, so IoU, precision and recall are
  missing.
  1. Annotate flood masks for the three maps, or better, use a public dataset.
  2. Add a script `paper/experiments/segmentation_metrics.py` that compares `results/e2e/<map>.npz`
     `mask` against your masks.
- **Image rights (blocking for the figures and for data availability):**
  - Varanasi: printed "processed by India Today", on a web basemap.
  - Kanpur: ESA attribution printed, on a web basemap.
  - Assam: no source printed.
  - Obtain permission, or replace the maps with openly licensed products before publication.
    Figs. 2–3 reproduce these images.
- **Kanpur caption:** the image itself says "Sentinel-2 SAR", which is internally inconsistent
  (Sentinel-2 is optical). The paper quotes it with "(sic)". Confirm the true source.
- **Geographic configuration:** decide whether the paper should also report runs with
  landmark-estimated or properly georeferenced scales and centres. Currently all runs use the
  repository default (2.0 m/px, Varanasi reference) **by design**, to describe the system as it is.
  Do not change `METERS_PER_PIXEL` silently.
- **Target autopilot:** state whether ArduCopter or PX4 is the intended target. The mission semantics
  differ (waypoint yaw, RTL→LAND).
- **Servo:** confirm channel 9 / PWM 2000 for your release mechanism, and whether a reset command is
  needed.
- **Weather:** the paper uses the fixed weather file. If you want live-weather results, record the
  API responses with timestamps so they are reproducible.
- **Template:** port to the official IEEE Access LaTeX template (download from the IEEE Access
  author page). Re-check page layout and figure sizes afterwards.
- **Literature:** consider a systematic (PRISMA-style) search to strengthen the gap claim. Re-verify
  that the 2026 references (Gioia et al.; Dong et al.) are in their final published form.

## 3. Experiments reviewers will most likely demand (ranked, cheapest first)

1. **Ground-station load test (hours).** Install Mission Planner and/or QGroundControl, load the
   three `.waypoints` files, and record screenshots, item counts and any warnings. Report them as a
   ground-station load test, not as a flight.
2. **ArduPilot SITL (1–2 days).** Run `sim_vehicle.py -v ArduCopter`, upload each mission, fly it in
   AUTO, and log mission acceptance, servo-9 PWM at each drop, drop altitude, RTL/LAND behaviour and
   mission-storage limits. Repeat with PX4 SITL if PX4 is claimed. (Free, with no hardware risk.)
3. **Segmentation ground truth (1–3 days).** Annotate the three maps and report IoU, precision and
   recall. Evaluate on a FloodNet subset (colour-annotated masks are not the same task — document the
   adaptation).
4. **One georeferenced input (1–3 days).** Run the pipeline on a GeoTIFF flood product, e.g. a
   Sentinel-1 flood map, with a correctly computed scale and centre. Compare the generated
   coordinates with GIS overlays.
5. **Multiple human operators (1 day).** Have three to five people perform the click step on each
   map, and report the variance in mask, regions and no-path legs.
6. **Weather scenarios (hours).** Sweep (P, V, θ, T) in the ablation, including calm weather and
   recorded real events.
7. **Repeated timing runs (hours).** Run five or more repetitions and report mean ± std for the
   runtime claims.
8. **Planner variants (evaluation only; frozen architecture).** Report how many no-path legs
   disappear with current-only obstacles, or with a finer grid, as a sensitivity analysis — not as a
   change to A.D.A.P.T.
9. **Multi-UAV evaluation (days).** Repeat the single- vs. multi-UAV comparison with recorded weather
   and a calibrated battery model; measure the minimum distance between vehicles in a multi-vehicle SITL
   run; if a workload-balancing allocation is added later, compare it with the current nearest-feasible
   rule on the same missions.
10. **HITL / field trial (weeks; requires approvals).** Only after 1–4, under local aviation rules and
   with a safety pilot.

## 4. Assumptions made while drafting (verify)

- The README's claim of validation on "Kolkata" is not supported by the repository (there is no
  Kolkata image), so the paper does not repeat it.
- The 2026 date and venue status of the cited 2026 papers were taken from Crossref on 2026-10-06.
- The paper describes the current working tree, including the verified fixes, which are not yet
  committed. Commit before sharing, so that the code matches the paper.
- The multi-base mode was integrated from the MAIN branch (commit `ca27215`, branch `routing-algo`).
  The paper calls it "a reference implementation developed in parallel with this work"; name the branch
  or the contributors if you prefer. The equivalence experiment needs an export of that commit
  (`git archive ca27215 | tar -x -C <dir>`; read-only, outside the repository).
- The battery coefficients come from a comment in `config/settings.py` that names a commercial
  agricultural drone (8 kg payload, 5 m/s, 7 min loaded endurance). The paper does not cite the
  manufacturer's specification; add a citation, or keep the values described as assumptions.
