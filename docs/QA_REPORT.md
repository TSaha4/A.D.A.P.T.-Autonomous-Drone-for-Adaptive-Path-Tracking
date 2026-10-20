# ADAPT Comprehensive Test Report

Adversarial QA of the DEVELOPMENT branch (`research-validation`, HEAD `54bb27a` plus the uncommitted
multi-UAV integration), October 2026. Every result below comes from a command that was actually run; the
QA suite is in `qa/` (see `qa/README.md`). Nothing was committed: all changes are in the working tree.

## 1. Environment

| | |
|---|---|
| OS | Windows 11 Home (build 10.0.26200) |
| Hardware | Intel Core i5-12500H, 31.7 GiB RAM |
| Python | 3.11.9 (venv `.venv`) |
| Dependencies | numpy 2.4.6, opencv-python 5.0.0.93, matplotlib 3.11.2, requests 2.34.2, python-dotenv 1.2.4 |
| Test framework | pytest 9.1.1 (unittest-style tests); QA additions, venv only: hypothesis 6.168.5, coverage 7.16.2, ruff 0.16.10 |
| Node | not used by the project |
| Simulation environment | **none available** (no ArduPilot/PX4 SITL, QGroundControl, Mission Planner or pymavlink installed) |

## 2. Baseline (before this QA pass)

| Tests | Passed | Failed | Skipped | Errors | Warnings | Time | Branch coverage |
|---|---|---|---|---|---|---|---|
| 220 + 494 subtests | 220 + 494 | 0 | 0 | 0 | 0 | 60 s (230 s under coverage) | 85 % |

Static analysis: `ruff` with bug-focused rules (`E9,F,B,PLE,S`) found no security findings and no real defects;
26 findings were unused imports/variables (pre-existing in `image_processing.py`, `clustering.py`,
`dstarlite.py`, `flood_spread.py`) and style rules (`zip` without `strict=`). A `B023` finding in
`tests/test_adversarial_coordinates.py` is harmless (the lambda is used within the same loop iteration).
`compileall` passes. No type annotations to check (mypy not run).

Coverage gaps at baseline: interactive GUI code (`image_processing.select_sample_points`,
`display_path_on_map`, `show_image_safe`; not testable headless), multi-mode error branches in `main.py`
(no safe drop, base planning exception, export failure), alternative branches of Tier 1/Tier 2 repair.

## 3. Testing performed

| Category | How | Volume |
|---|---|---|
| Unit / boundary / property tests | `qa/qa_units.py` (Hypothesis) | 69 tests, ~2 300 generated cases |
| Coordinate systems | all hemispheres, equator, prime meridian, antimeridian, poles, invalid lat/lon/scale; independent formula | 150 random + 14 fixed |
| Clustering | 0/1/2 points, collinear, identical points, 300 random contours | 7 |
| Drop points | legacy and dry selection; random blob maps; fully flooded map; connectivity filter edge inputs | 40 random maps + 6 |
| Route ordering | TSP permutation property, empty, duplicates, home None | 150 random |
| D* Lite | random grids 1×1 to 25×25 up to 60 % obstacles vs. Dijkstra (reachability and exact length), blocked start/goal, 1×1 and 1×2 grids | 40 random + 4 |
| Both planners | stop mapping, tiny/thin masks, out-of-image points, duplicates, fully flooded, leg-length consistency | 80 random + 7 |
| Crossing test | random polylines vs. brute-force segment intersection | 150 random |
| Flood forecast | weather extremes, non-numeric, non-finite horizon, monotone frames, last frame = single-mode prediction, query at past/future/boundary times | 40 random + 12 |
| Battery / payload | edge distances, NaN, capacity below one package; sortie planner invariants (no drop lost/duplicated, per-sortie battery and payload limits, report = plan) | 40 random + 3 |
| Base selection | invariants on random instances (partition, reach, nearest feasible, separation, cap, no empty base), duplicate candidates | 40 random + 1 |
| Repair loop | `qa/qa_repairs.py`: random crossing plans, coincident drops; Tier 0 and Tier 1 exercised | 120 random |
| Mission export | random routes → run-time check + independent validator with sidecar; empty path; malformed files | 40 random + 6 |
| End-to-end, both modes | `qa/qa_pipeline.py`: bad files, CLI values, weather inputs, environment variables, session root, non-ASCII path, 9 pathological map geometries × modes × caps, noise, tiny/odd images, display widths, S1–S12, M1–M14, determinism, state leakage | 108 runs |
| Fuzzing | `qa/fuzz.py`, seeded random maps/scales/weather/caps/modes, all invariants | 240 before fixes, 500 after |
| Stress | `qa/stress.py`, 10–600 flood zones, caps 1–16 | 17 + 5 configurations |
| Real maps | invariant check of the 26 multi-base experiment sessions (Varanasi, Kanpur, Assam) | 26 |
| Security review | grep + ruff `S` rules; file writes, network, deserialisation, subprocess, secrets, hashing | whole code base |

## 4. Single-UAV results

S1 one region, S2 two regions, S3 normal maps, S4 100 regions, S9 obstacles (ring enclosure, comb corridors),
S10 strong weather (12 mm, 35 km/h, 4 h), S11 RTL/LAND last, S12 empty map (no file written): all pass; every
mission passes the independent validator, with one DO_SET_SERVO per region. S5 route optimisation: the
nearest-neighbour order is a heuristic (not optimal, see paper). **S7 battery and S8 payload: not applicable**: the
single-UAV mode has no battery or payload model (documented limitation; the multi-base mode with `--max-bases 1`
is the battery-aware single-UAV planner). S6 unreachable: an enclosed drop is reached by a logged straight
fallback leg (documented behaviour of this mode). Single-mode missions on the three bundled maps are
byte-identical before and after the fixes, and identical to the pre-integration commit.

## 5. Multi-UAV results

| Case | Result |
|---|---|
| M1 `--max-bases 1` | one base, one mission, far drops reported with "cap reached" |
| M2 two UAVs (two clusters, dominant cluster, uneven, unreachable region, symmetric) | pass; ≤ 2 bases; invariants hold |
| M3 three UAVs | 3 bases, all drops served |
| M4/M5 at and above the default maximum (4, 5, 50) | accepted; only as many bases as separated clusters need (≤ 4) |
| M6 more UAVs than tasks (2 drops, cap 10) | one base, no empty missions |
| M7 100 drops, 3 UAVs | pass |
| M8 uneven workload | reproduced and documented: nearest-feasible assignment does not balance (e.g. 17/7/6 drops) |
| M9 battery-constrained allocation | drops are only assigned to bases that reach them (property test) |
| M10 no feasible base | exit 2, `NO_BASE_REACHES_ANY_DROP`, no files |
| M11 duplicate bases | with separation 0, two candidates at the same spot are never both chosen |
| M12 bases outside the valid area | not applicable: bases are generated inside the image on dry pixels with clearance (verified) |
| M13 crossing routes | repaired or rejected (exit 2, nothing exported) |
| M14 independent routes | left unchanged (no repairs) |
| M15 empty task group | bases without drops are removed; a base whose planning fails is now reported (QA-08) |

Invariants 1–10 (exclusive assignment, no silent loss, valid drops, battery and payload per sortie, valid
geometry, missions match assignments, base cap changes do not corrupt the drop set, single-UAV unchanged) were
checked on every multi-base run of the end-to-end suite, the fuzzer and the 26 real-map sessions. Before the
fixes the fuzzer found two violations (QA-08, QA-09); after them, 0 of 500 scenarios violate any invariant.

## 6. Edge-case results

No flood, fully flooded map, 15-px region, 650-px region, 100 disconnected regions, thin lines, regions touching
all borders, enclosed island, comb of corridors, 600-speck noise, 1×1/2×2/5×5/12×9/40×3/3×40 images, RGBA and
grayscale PNGs, display widths 1/3/100 000, horizons 0 and −3: all handled (valid mission or graceful exit). Base
clearance near a noise speck that morphology removes is by design.

## 7. Exceptional-case results

Missing/empty/corrupt/non-image file and a directory as image: exit 1 with a message in both modes. Invalid
lat/lon/scale: exit 2 before any work. Failures found and fixed: QA-01 to QA-07 (section 14).

## 8. Stress-test results

Wall time is dominated by the D*-Lite router (rebuilt per leg; in multi mode also re-run during sortie
refinement). Measured (single runs, 750×750 synthetic maps, 2 m/px):

| Mode | Flood zones | Cap K | Drop points | Bases | Visited | Total time | Router share | Peak memory |
|---|---|---|---|---|---|---|---|---|
| single | 10 | – | 10 | – | 10 | 2.6 s | 98 % | – |
| single | 50 | – | 50 | – | 50 | 18.6 s | 99 % | – |
| single | 100 | – | 100 | – | 100 | 35.6 s | 99 % | 89 MB |
| single | 300 | – | 300 | – | 300 | 110 s | 98 % | 90 MB |
| single | 600 | – | 600 | – | 600 | 242 s | 96 % | 92 MB |
| multi | 10 | 1 / 4 | 20 | 1 | 19 | 27 s | 84 % | – |
| multi | 50 | 1 | 50 | 1 | 49 | 207 s | 93 % | – |
| multi | 50 | 4 | 50 | 2 | 50 | 98 s | 87 % | – |
| multi | 100 | 1 | 100 | 1 | 90 | 413 s | 94 % | – |
| multi | 100 | 4 / 8 / 16 | 100 | 3 | 100 | 391–395 s | 92 % | 344 MB |
| multi | 300 | 4 | 300 | 4 | 300 | 1 138 s (19 min) | 91 % | 382 MB |

"Router share" is the time spent in `compute_full_path` (single) or `compute_routed_path` (multi). Memory was
measured as the peak working set of the run's process (only for the second batch). In the single mode the run time
grows roughly linearly with the number of drops, because each leg builds a new D* Lite instance on the whole
grid; in the multi mode each base's route is routed again in every refinement pass. With `--max-bases 1` some
drops are reported as unreachable (90 of 100 visited), as expected from the battery range.

No crash or invalid output occurred at any size. The cap (4 → 16) does not change run time once the needed
bases are placed.

## 9. Random / fuzz results

Before fixes: 240 scenarios, 5 with problems (seeds 31, 71, 77, 99, 132 → QA-08, QA-09). After fixes: 500
scenarios (seeds 0–499: 341 multi-mode, 159 single-mode; 1–6 bases; 3 268 multi-mode drop points),
**0 with problems**. Repair-loop property test: 120 random cases, 0 violations. No uncaught exception in any
fuzzed run.

## 10. Simulation results

```text
NOT EXECUTED
Reason: no SITL (ArduPilot/PX4), QGroundControl, Mission Planner or MAVLink library is installed on this machine.
Alternative validation performed: every exported mission (single and multi) was checked by the run-time WPL check
and by the independent specification-based validator against its route file; mission semantics were reviewed
against documentation only.
```
No QGroundControl, SITL, MAVLink or hardware validation is claimed.

## 11. Mission-export results

Run-time check and independent validator: 0 errors on all missions of the regression suite, the end-to-end QA
runs, 500 fuzz scenarios, the 26 real-map sessions and 40 random export round trips. Empty path, one waypoint,
1 000+ item missions, reload stops, southern/western/antimeridian references: valid. Malformed WPL files are
rejected by both checkers. Invalid coordinates are rejected before a file is opened.

## 12. Performance results

See section 8. Other measured costs: forecast frames for a 168 h horizon took 7.5 s before QA-10 and now need
one spread iteration per hour; weather/segmentation/export stages are below 0.1 s on 750×750 maps; base
selection < 1 s for 100 drops.

## 13. Coverage

Branch coverage of `main.py`, `src/`, `config/`: 85 % before, 86 % after (regression suite only; the QA suite
exercises more). Well covered (≥ 90 %): coordinates, clustering, multi_base, constraints, pathfinding,
dstarlite, flood_spread, settings. Poorer: `main.py` 78 % (GUI windows, rare multi-mode error branches),
`overlap_repair.py` 81 % (Tier 1 fallbacks, Tier 2 for the second base), `mission_output.py` 84 % (the legacy
interactive plot), `image_processing.py` 68 % (interactive click window). Not covered by automated tests: the
interactive GUI path (needs a display and a person).

## 14. Bugs found

| ID | Severity | Component | Problem | Status |
|---|---|---|---|---|
| QA-01 | P2 | `main.py` CLI | `--horizon nan/inf` crashed with ValueError/OverflowError (both modes) | Fixed |
| QA-02 | P2 | weather API → `main.py` | a null/NaN/missing weather field crashed the run (TypeError/KeyError/ValueError) | Fixed |
| QA-03 | P3 | `main.py` dummy weather | weather file without all fields, or not an object, crashed (KeyError/TypeError) | Fixed |
| QA-04 | P3 | `flood_spread.predict_spread` | precipitation ≥ ~10⁴ mm built a huge dilation element (1e6 mm → 160 GB, OpenCV OOM) | Fixed |
| QA-05 | P3 | `config/settings.py` | a non-numeric environment variable gave a bare `float()`/`int()` traceback at import | Fixed |
| QA-06 | P3 | `main.py` multi mode | an existing file as `--session-root` failed with FileExistsError after the whole run | Fixed |
| QA-07 | P3 | `main.py` image loading | images under non-ASCII paths could not be read on Windows ("Error loading image file") | Fixed |
| QA-08 | P2 | `main.run_multi_base` | drops assigned to a base that produced no route vanished from `run_summary.json` (4/240 fuzz runs) | Fixed |
| QA-09 | P2 | multi-base repair bookkeeping | coincident drop points (two adjacent regions, one dry pixel): Tier 0 moved only one index, the point stayed with two bases and a repairable plan ended as infeasible | Fixed |
| QA-10 | P3 | `flood_spread.forecast_frames` | O(N²) spread iterations in the horizon (168 h: 7.5 s) | Fixed |
| QA-11 | P3 latent | `weather_api` | the cached weather dict is shared; a caller mutating it would change later results | Open (no caller mutates it) |
| QA-12 | P3 latent | `constraints.leg_battery_fraction` | a NaN distance counts as zero battery | Open (unreachable: scale validated, distances finite or ∞) |
| QA-13 | P3 latent | `constraints.plan_mission_stops` | payload capacity below one package raises AssertionError | Open (constants not user-configurable) |
| QA-14 | P3 latent | `flood_spread.predict_spread` | called directly with NaN/None weather or a non-finite horizon it raises ValueError/TypeError | Open (main validates both first) |

Details (reproduction, root cause, fix, regression test) for the fixed bugs:

- **QA-01.** `python main.py map.png --horizon nan` → `int(nan)` in `predict_spread`. Root cause: the horizon was
  never validated. Fix: `main` rejects a non-finite horizon with a usage error (exit 2) before any work.
  Regression: `tests/test_robustness.py::test_non_finite_horizon_is_rejected_before_any_work`.
- **QA-02.** Open-Meteo may return `null` for a field; `main` compared `None > 40.0`. Root cause: values passed
  through unchecked. Fix: `weather_api.sanitize_weather` (used by the API client and by `main` for both sources)
  turns a missing/non-finite field into 0.0 (calm, as for an unavailable API), logs a warning naming the field,
  and marks the status "partial". Regression: `test_incomplete_or_non_numeric_weather_does_not_crash`,
  `test_api_nulls_become_calm_values_with_a_warning`.
- **QA-03.** Same root cause for `data/input/dummy_weather.json`; same fix. Regression:
  `test_malformed_dummy_weather_file_does_not_crash`.
- **QA-04.** Fix: the dilation radius is capped at the image diagonal + 2 px. Growth of at least the diagonal
  already floods every pixel, so results are unchanged (verified against an explicit reference). Regression:
  `test_extreme_precipitation_does_not_exhaust_memory`.
- **QA-05.** Fix: `settings._env` parses every numeric variable and stops with
  "Configuration error: NAME='value' … is not a number" (no traceback). Regression:
  `test_invalid_numeric_environment_variable_gives_a_clear_message`.
- **QA-06.** Fix: `main` rejects a `--session-root` that exists and is not a folder before any work. Regression:
  `test_session_root_that_is_a_file_is_rejected_before_any_work`.
- **QA-07.** `cv2.imread` cannot open non-ASCII paths on Windows. Fix: only when `imread` fails and the file
  exists, the bytes are decoded with `cv2.imdecode` (errors handled). Regression:
  `test_image_under_a_non_ascii_path_is_read`.
- **QA-08.** Fuzz seed 71: base selection judges reach on the 5-px routing grid (quantised, snapped cells), the
  sortie planner on exact pixel distances; at coarse scales (~28 m per display pixel) a drop judged reachable
  needs 86 % battery, the base gets no route, and its drops were reported nowhere. The algorithms are unchanged
  (MAIN's behaviour); fix: such drops are listed with a reason in the new `not_planned` field of
  `run_summary.json`, logged, and counted in `stage_counts`. Regression:
  `test_drops_of_a_base_without_a_plan_are_reported`.
- **QA-09.** Fuzz seed 31 (two regions sharing a dry drop pixel); bug confirmed with a constructed crossing.
  Root cause: the point→drop map kept one index per point. Fix: `main.drop_index_by_point` maps a point to all
  its drop indices and `overlap_repair.sortie_drops` returns all of them. Regression:
  `test_coincident_drop_points_are_reassigned_together`.
- **QA-10.** Fix: frame k is one further spread iteration of frame k−1 (identical masks, verified by the
  existing frame tests). Regression: `test_frames_need_one_spread_iteration_per_hour`.

## 15. Fixes made

Production changes (all minimal, at the root cause): `main.py` (horizon and session-root validation, weather
sanitising, non-ASCII image fallback, `drop_index_by_point`, `not_planned` reporting), `src/weather/weather_api.py`
(`sanitize_weather`), `src/weather/flood_spread.py` (dilation radius cap, incremental frames),
`config/settings.py` (`_env` parser), `src/mission/overlap_repair.py` (`sortie_drops` returns all coincident
drops). New regression tests: `tests/test_robustness.py` (11 tests, 26 subtests). No algorithm was changed:
single-mode missions on the bundled maps are byte-identical before and after; regenerated multi-mode
experiment sessions are byte-identical; the multi-base plans still equal the reference implementation on all
three equivalence cases.

## 16. Remaining issues

QA-11 to QA-14 (latent, not reachable through the program's inputs; documented by strict `xfail` tests in
`qa/`). Base-selection reach and sortie planning use different distance estimates (QA-08 root cause); the
consequence is now reported, the inconsistency itself is MAIN's design and was not changed.

## 17. Known limitations (not bugs)

Single-UAV mode has no battery/payload model and flies straight fallback legs to enclosed drops; nearest-feasible
assignment without workload balancing; geometric deconfliction only (same altitude, no timing); reload stops
(LAND/TAKEOFF) untested on an autopilot; coverage depends on `METERS_PER_PIXEL` (at the maps' true scale no drop is
reachable); two adjacent regions can share a drop point (two packages at one spot); D* Lite is rebuilt per leg,
which dominates run time (minutes for 100+ drops in multi mode); the click step needs a display; no georeferencing
of the bundled maps.

## 18. Final regression status

| | Before QA | After fixes |
|---|---|---|
| Regression suite (`python -m pytest`) | 220 passed + 494 subtests, 0 failed | 231 passed + 520 subtests, 0 failed, 0 warnings |
| Branch coverage | 85 % | 86 % |
| Adversarial unit + end-to-end QA suites | 34 failed (8 unit, 26 end-to-end) | 0 failed; 8 documented latent issues as strict xfail |
| Fuzzing | 5 / 240 scenarios with problems | 0 / 500 |
| Mutation testing | 31 / 33 detected (220-test suite) | 31 / 33 detected (231-test suite; same 2 behaviour-equivalent survivors) |
| Single-mode outputs (3 maps) | — | byte-identical to the pre-integration commit |
| Multi-mode equivalence with the reference (3 cases) | identical | identical |

## 19. Final acceptance

```text
PASS WITH KNOWN LIMITATIONS
```
Every acceptance criterion that can be checked on this machine is met with evidence. Not met, because not
executable here: "Simulation works where available": no simulator exists on this machine (section 10), so
operational behaviour (SITL, ground-station load, reload stops on an autopilot) remains unvalidated. Known
limitations are listed in section 17; no P0/P1 issue was found and all P2 issues are fixed.
