# TODO for the authors (must be resolved before submission)

## 1. Placeholders in the manuscript

| Where | Placeholder | Action |
|---|---|---|
| `main.tex` author block | `[Author 1]`, `[Author 2]`, `[Faculty Advisor]`, department, institution, city, country, e-mail | Fill in; add ORCID iDs if the venue requests them |
| `main.tex` first footnote | `[Funding statement]` | Provide, or write "This work received no external funding." |
| `sections/backmatter.tex` | `[REPOSITORY URL]` | Public repository URL; archive a release (e.g. Zenodo) and cite its DOI |
| `sections/backmatter.tex` | Acknowledgment: funding/institutions; extent of the authors' own review of AI-drafted content | Complete. IEEE requires disclosure of the AI system and of which sections it produced (already listed) |
| `main.tex` | `\markboth{Draft manuscript --- not peer reviewed}` | Replace when porting to the venue template |

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
9. **HITL / field trial (weeks; requires approvals).** Only after 1–4, under local aviation rules and
   with a safety pilot.

## 4. Assumptions made while drafting (verify)

- The README's claim of validation on "Kolkata" is not supported by the repository (there is no
  Kolkata image), so the paper does not repeat it.
- The 2026 date and venue status of the cited 2026 papers were taken from Crossref on 2026-10-06.
- The paper describes the current working tree, including the verified fixes, which are not yet
  committed. Commit before sharing, so that the code matches the paper.
