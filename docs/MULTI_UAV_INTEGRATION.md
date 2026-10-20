# Multi-UAV / Multi-Base Integration Notes

This document records how the multi-base mission planner from MAIN (`origin/routing-algo`, commit
`ca27215`) was ported into the DEVELOPMENT branch (`research-validation`). DEVELOPMENT is the source
of truth: MAIN was used only as the reference for functionality DEVELOPMENT did not have.

Branch topology (read-only inspection): both branches descend from `775165d`. MAIN added the whole
multi-base capability in one commit (`ca27215`); DEVELOPMENT added the geospatial/export fixes, the
verification tooling, the tests and the paper (`0abbfef`, `54bb27a`).

## 1. What "multi-UAV" means in MAIN (and therefore here)

MAIN does not contain a swarm, a fleet model, drone IDs or any inter-vehicle coordination at run
time. It is a **multi-base planner**: it chooses between 1 and `MAX_BASES` launch sites, assigns
every drop point to exactly one site, plans each site independently with the single-vehicle
planner, and then checks the resulting routes for geometric crossings. One base corresponds to
one UAV mission file (`base<N>.waypoints`). The port reproduces exactly that, nothing more.

## 2. Integration matrix

| MAIN component | In DEVELOPMENT before? | Action | Reason |
|---|---|---|---|
| `src/mission/multi_base.py` (`single_drop_reach`, `select_bases`) | No | **Ported verbatim** | Missing; assignment algorithm (nearest feasible base by routed distance) kept unchanged |
| `src/mission/overlap_repair.py` (Tier 0 reassign, Tier 1 reroute, Tier 2 nudge) | No | **Ported, adapted** | Missing. Export removed from the repair loop (files are written once, after deconfliction); scale passed explicitly; reach of a nudged base refreshed (MAIN defect) |
| `src/mission/constraints.py` (battery/payload model, `plan_mission_stops`, `score_home_candidates`, `sortie_battery_report`) | No | **Ported verbatim** except the unused `split_sorties(drop_count, …)` | Required infrastructure; the unused helper duplicated the name of `mission_output.split_sorties` |
| `pathfinding.py`: `RoutingGrid`, `build_routing_grid`, `cell_of`, `cell_center`, `locate_cell`, `grid_distance_field`, `routed_distance_function`, `polyline_flood_pixels`, `polyline_intersections` | No | **Ported verbatim** (added) | Required by base selection, connectivity filtering, routing and overlap detection |
| `pathfinding.compute_full_path` (MAIN rewrite: time-indexed obstacles, component-aware snapping, leg report, 3-tuple return) | DEVELOPMENT has the original 2-tuple version | **Ported as a new function `compute_routed_path`**; DEVELOPMENT's `compute_full_path` untouched | Single-UAV mode, its tests and the paper harness depend on the original function and its return contract |
| `pathfinding.snap_to_nearest_free_cell` (MAIN adds `target_component`, `labels`, `accept`, default radius 15) | Yes (radius 10) | **Merged**: new keyword arguments added, DEVELOPMENT default radius 10 kept | Without the new keywords the behaviour is identical to DEVELOPMENT's; MAIN never relies on the default radius |
| `pathfinding.nearest_neighbor_tsp` | Yes, identical | Keep DEVELOPMENT | Same code |
| `src/routing/dstarlite.py` | Yes, identical | Keep DEVELOPMENT | Same code; reused by every router |
| `flood_spread.predict_spread` (MAIN: new fractional, time-indexed growth model) | DEVELOPMENT has the hourly CA model | **Keep DEVELOPMENT model**; not ported | Changing the flood model is out of scope (frozen architecture) and the paper evaluates DEVELOPMENT's model |
| `flood_spread` time-indexed interface (`forecast_at`, `obstacle_mask_at`, `interpolate_flood_front`, `interpolate_forecast_mask`, `signed_distance`, `_bracket`) | No | **Ported verbatim** | The battery/arrival-time planner and the router query obstacles at arrival time |
| Forecast frames (`{hours: mask}`) | No | **New adapter `forecast_frames`** built on DEVELOPMENT's `predict_spread` | Frame *k* hours = `predict_spread(mask, w, k)` (one CA iteration per hour, DEVELOPMENT semantics); the last frame equals the single-mode predicted mask |
| `safe_dropzone.find_safe_drop_points` (MAIN: dry band outside the contour, multi-point drops for large zones) | DEVELOPMENT has the hull-vertex version | **Merged**: MAIN behaviour only when `obstacle_mask` is passed; the legacy call is unchanged | MAIN's battery planner rejects flooded drops, and hull vertices lie on the flood mask, so the multi-base planner needs dry drops |
| `safe_dropzone.filter_drops_by_home_connectivity` | No | **Ported verbatim** | Per-base connectivity filter |
| `mission_output.generate_mission_file` (MAIN: `reload_indices`, settings-based servo) | DEVELOPMENT has pre-conversion, servo kwargs, HOME-region drop rule | **Merged**: `reload_indices=None` keyword added; everything else stays DEVELOPMENT | Without `reload_indices` the output is byte-identical |
| `mission_output.validate_waypoints_file` | No (DEVELOPMENT has the independent `tests/wpl_validator.py`) | **Ported** as the run-time check | Production code must not import the independent test oracle |
| `mission_output.display_path_on_map` (MAIN multi-base renderer) | DEVELOPMENT has its own single-route renderer | **Ported as `display_multi_base_map`** with helpers (`split_sorties`, `draw_dashed_polyline`, `arrow_positions`, `basemap_sanity`, palette); DEVELOPMENT renderer untouched | Base labels use the real base number (MAIN numbered by list position) |
| `main.py` orchestration (candidates, `select_bases`, `plan_base`, `refine_route`, `map_routed_drop_distances`, repairs, overlap gate, territories, session folder, `run_summary.json`) | No | **Ported into `main.run_multi_base`**, reached with `--mode multi` | Single-UAV pipeline in `main()` unchanged |
| `create_session_dir` | No | Ported, race fixed | MAIN crashed when two runs started in the same second |
| `config/settings.py` battery/payload/base constants | No | **Ported**: `DRONE_*`, `PAYLOAD_*`, `WIND_PENALTY_PER_MPS`, `BATTERY_RESERVE_FRACTION`, `INITIAL_BATTERY_FRACTION`, `LARGE_CONTOUR_*`, `MAX_DROPS_PER_CONTOUR`, `MIN_DROP_SEPARATION_PX`, `HOME_CLEARANCE_PX`, `MAX_BASES`, `MIN_BASE_SEPARATION_PX` | Same configuration system (`settings.py` + `.env`) |
| `FLOOD_*` constants, `MASK_*` sanity bounds, `REGION_PRESETS`, `DROP_SERVO_*` | No | Not ported | Belong to MAIN's flood model / other features, not to multi-base planning |
| `clustering.py` | DEVELOPMENT has the zero-moment fix | **Keep DEVELOPMENT** | MAIN still has the bug |
| `coordinates.py` | DEVELOPMENT hardened and verified | **Keep DEVELOPMENT** | Used unchanged for every base file |
| `image_processing.auto_sample_points`, `--no-gui/--auto-sample/--sample-points`, region presets, `--dummy-weather-moderate` | No | Not ported | Not part of multi-UAV capability |
| MD5 / hashing | Absent in both branches | Nothing to port | Verified by search in both trees |
| MAIN tests (`test_multi_base`, `test_overlap_repair`, `test_mission_constraints`, relevant parts of `test_audit_regressions`) | No | Ported and adapted into new files | DEVELOPMENT tests kept untouched |

## 3. Conventions reconciled

* **Ground scale.** DEVELOPMENT defines `METERS_PER_PIXEL` per *original* image pixel and works
  with `k = METERS_PER_PIXEL / s` metres per *display* pixel (`s` = resize factor). MAIN used
  `METERS_PER_PIXEL` directly as metres per display pixel. The multi-base mode uses `k` everywhere
  (battery distances, routed distance fields, arrival times and export), so planning and the
  exported coordinates describe the same ground distances. For an image that is not resized the
  two conventions coincide.
* **Image centre.** Export uses DEVELOPMENT's `((W-1)/2, (H-1)/2)` (MAIN used `W//2, H//2`).
* **Coordinate conversion and export safety.** DEVELOPMENT's hardened `pixel_to_latlon` and the
  pre-converting writer are used for every base file; a file that fails generation or run-time
  validation is removed.
* **Flood model.** DEVELOPMENT's hourly model, exposed through MAIN's time-indexed interface.
* **HOME row frame.** DEVELOPMENT writes the HOME row in frame 3 (MAIN: frame 0); both are accepted
  by QGC and by both validators. DEVELOPMENT's convention is kept.

## 4. MAIN defects corrected during the port (algorithms unchanged)

| Defect in MAIN | Correction |
|---|---|
| Waypoint files were written inside `plan_base` and Tier 1, so a rejected Tier 2 candidate could overwrite a base's file; files were also left on disk when the run then exited with status 2 | Planning writes no files; every base is exported once, after the final crossing check, and nothing is exported when the check fails |
| The reach set of a nudged base was not updated | `execute_overlap_repairs` recomputes it after a Tier 2 move |
| Map labels used the list position ("BASE 2") rather than the base number of `base<N>.waypoints` | `display_multi_base_map` uses the real base number |
| Two runs started in the same second crashed while creating the same session folder | `create_session_dir` creates the folder atomically and retries with a suffix |
| Some assignment logs were printed before repairs and could be stale | The final assignment table is logged again after repairs |
| A mission that failed run-time validation stayed on disk | It is deleted (DEVELOPMENT's no-invalid-mission rule) |
| `generate_mission_file` accepted any reload index | Reload indices must be intermediate returns to HOME and not drops; otherwise nothing is written |
| Unused imports (`math`, `cv2`, `pixel_to_latlon`) and an unused `split_sorties` in `constraints.py` | Removed |

## 5. Validation

| Check | Result |
|---|---|
| DEVELOPMENT baseline before integration | 126 passed + 494 subtests |
| Full suite after integration | 220 passed + 494 subtests: 94 new tests, and no existing `test_*.py` file changed. The independent validator `tests/wpl_validator.py` gained an opt-in reload extension, enabled only by sidecar `reload_indices` or `--allow-reloads`, and is otherwise unchanged |
| Single-UAV mode unchanged | Missions and route sidecars of Varanasi, Kanpur and Assam (paper harness) are byte-identical to the pre-integration commit |
| Port fidelity vs MAIN (`paper/experiments/multi_uav_equivalence.py`) | Identical bases, assignment, every route point, drop/reload indices and repairs on Varanasi (calm, 2 bases), Kanpur (calm, 4 bases) and Varanasi (moderate, 4 bases, two Tier 0 repairs), with matched inputs |
| Adversarial QA pass (`docs/QA_REPORT.md`, `qa/`) | 10 defects found and fixed (QA-01 to QA-10), with regression tests in `tests/test_robustness.py`; suite now 231 passed + 520 subtests; 500 fuzz scenarios without an invariant violation |
| Mutation testing (`paper/experiments/mutation_eval.py`) | 31/33 killed: the 24 original mutants give 22 killed, as before; all 9 multi-base mutants are killed |
| Multi-UAV experiment (`paper/experiments/multi_uav_comparison.py`) | 24 runs, 20 feasible, 43 missions: 0 validator errors, 0 crossings, 0 fallback legs, max sortie battery 79.6 % |

New test files (MAIN tests are ported into new files, so DEVELOPMENT's files of the same name are untouched):

| File | Content |
|---|---|
| `tests/test_multi_uav.py` | **End-to-end `main.py --mode multi` on synthetic maps:** single UAV (`--max-bases 1`), two and three UAVs, base cap, battery limits, unreachable drops sealed off by flood, crossing gate (exit 2, no files), empty and single-zone maps, no base site, scale convention, CLI validation. **Unit tests:** Tier 0 success, reach refresh after nudging, repairs write no files, reload validation, forecast frames. **Preservation:** single-mode unchanged |
| `tests/test_multi_base.py` | MAIN's base-selection, crossing and distance-field tests |
| `tests/test_overlap_repair.py` | MAIN's repair tests, adapted to the new signatures |
| `tests/test_mission_constraints.py` | MAIN's battery/payload/sortie tests, using DEVELOPMENT's flood model |
| `tests/test_multi_base_regressions.py` | MAIN's rendering, routing-consistency, drop-placement, interpolation, export and refinement regressions |

MAIN tests that were not ported:

- The two automatic-HSV-sampling tests, because that feature was not ported.
- The Bhopal-session test, which needs MAIN's local output folder and was always skipped.
- MAIN's own `predict_spread` frame tests, because MAIN's flood model was not ported.

## 6. End-to-end trace (Varanasi, moderate weather, `--max-bases 4`)

Input: `data/input/varanasi.png`, 972×970, shown at 750×748 (s = 0.7716).

1. **Scale and centre.** Metres per display pixel: k = 2.592; image centre (374.5, 373.5).
2. **Segmentation.** Simulated operator, 8 clicks; 16 flood zones.
3. **Forecast frames.** Frames at 0, 1 and 2 h (DEVELOPMENT spread rule); O = frame at 2 h.
4. **Drop points.** 22 dry drop points, with every zone having at least one.
5. **Base candidates.** 1764 grid points, 1613 of them dry; 1308 have at least 15 px clearance.
6. **Reach.** 18 drops are coverable; 4 are reported as "unreachable by any base: no flood-clear
   base site can serve it within battery". The best single site reaches 10, so multi-base selection
   is used.
7. **Bases.** Three bases: (279, 243), (711, 189), (549, 513). Assignment: 8, 7 and 3 drops.
8. **Per-base planning.** Refinement converged in 2, 3 and 1 passes. Routing used 35 legs (28 D* Lite,
   7 same-cell, 0 fallback); 7 reloads; sortie battery at most 79.6 %.
9. **Crossing check.** No crossings, so no repair. Every drop lies in its own base's territory.
10. **Export.** `base1/2/3.waypoints` were validated at run time and by `tests/wpl_validator.py`
    against the sidecars, with 0 errors. 18 drops served in total; the routes total 12.9 km.

## 7. Remaining limitations (honest)

- **Not a swarm.** There is no in-flight coordination, and deconfliction is geometric only: routes
  must not cross, but there is no time or altitude separation and no minimum route-to-route distance.
- **No load balancing.** The assignment is nearest-feasible, as in MAIN, with no mission-time
  objective.
- **Uncalibrated battery model.** It ignores climb, descent, hover and reload time, and applies the
  full wind as a headwind on every leg.
- **Untested reload stops.** LAND followed by TAKEOFF has not been tested on an autopilot.
- **Scale-dependent coverage.** Coverage depends on `METERS_PER_PIXEL`. At the maps'
  landmark-estimated scales, no drop is reachable (exit 2).
- **Different drop rules in the two modes.** The single-UAV mode uses boundary hull vertices and HOME
  = a drop point, while the multi-UAV mode uses dry drops and a flood-clear base. `--max-bases 1` is
  therefore the controlled single-UAV baseline for multi-UAV comparisons; it is not identical to
  `--mode single`.
- **No simulation.** No SITL, ground-station or flight test has been performed for either mode.

## 8. How to run and generate the simulation figures

See `README.md` ("Multi-UAV mode") and `paper/TODO_for_authors.md` (section 1b). The latter lists,
for each figure placeholder in the paper, the exact command and output file to use, and gives the
SITL procedure.
