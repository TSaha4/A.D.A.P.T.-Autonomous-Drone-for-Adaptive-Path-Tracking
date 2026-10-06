# Research Audit — A.D.A.P.T. paper

Companion files: `paper_facts.md` (single source of truth), `literature_matrix.csv`,
`claim_evidence_map.csv`, `PEER_REVIEW_SIMULATION.md`, `TODO_for_authors.md`.

## 1. Literature searched

- **Search channels:** web search (keyword queries per theme), the Crossref REST API (title search
  and DOI lookup), OpenAlex (abstracts), DataCite (software DOI), and publisher/documentation pages
  (MAVLink, ArduPilot, PX4, Open-Meteo, IEEE Access author guidelines).
- **Themes:**
  - UAV disaster response;
  - flood mapping and segmentation (SAR, UAV, colour-based);
  - flood modelling (raster / cellular-automata);
  - path planning (A*, D*, D* Lite);
  - route optimisation and relief logistics;
  - safe landing/drop-site selection;
  - MAVLink / mission generation;
  - geodesy and map projections;
  - weather constraints;
  - integrated hazard-aware mission planning.
- **Considered:** about 60 candidates.
  - **Verified:** 48 DOI records (47 Crossref + 1 DataCite), plus 1 non-DOI conference paper
    (D* Lite, AAAI 2002, CMU page) and 6 documentation sources.
- **Rejected:**
  - one recalled DOI that resolved to an unrelated paper (10.1109/ICUAS.2014.6842296, a hexacopter
    design paper);
  - arXiv-only preprints (not peer-reviewed);
  - six verified but redundant references removed during page trimming (two path-planning surveys,
    one civil-UAV survey, one CNN flood-detection paper, one strategic relief model, one flash-flood
    monitoring note). They remain in `references.bib` but are not cited.
- **Final reference list:** 50 cited entries.
  - 41 peer-reviewed journal/conference papers (40 with DOI + the AAAI D* Lite paper);
  - 1 USGS professional paper;
  - 1 software DOI (Open-Meteo);
  - 1 standard non-DOI software citation (OpenCV);
  - 6 web documentation pages (MAVLink file format, common messages, mission protocol; ArduPilot;
    PX4; Open-Meteo docs).
  - Bibliographic metadata of every DOI entry was generated from the DOI registry
    (`experiments/fetch_bibtex.py`) and only reformatted (`experiments/normalize_bib.py`).
    The two corrections made are listed in that script, with reasons.
- **Limitation:** the search was targeted, not systematic (no PRISMA protocol). Absence claims in the
  paper are therefore bounded ("in the reviewed literature").

## 2. Research gap (derived from the literature matrix)

Each of the following is well developed on its own:

- perception (flood masks);
- hazard modelling (DEM-based CA / raster models);
- path planning (given obstacle maps);
- route optimisation (given nodes);
- landing-site selection (landing, not delivery).

In the reviewed set, none chains *imagery → delivery points → hazard-inflated obstacles → planned
route → executable autopilot mission*, and none reports verification of the image-to-geodetic and
mission-encoding steps. The closest works are:

- Dong et al. 2026: flood-aware multi-UAV delivery routing, but with demand nodes and network states
  given, and schedules as output;
- Li et al. 2025: UAV cluster deployment for urban floods, in simulation;
- Gioia et al. 2026: a review that identifies fragmentation of hazard-evolution / mission-planning
  methods.

## 3. Novelty assessment

**Defensible claim:** an open, training-free, DEM-free integration of standard components that
produces an executable QGC WPL 110 relief mission from an annotated flood map and current weather,
together with an independent verification methodology for its geospatial and mission layers and an
empirical characterisation that includes negative results.

**Deliberately NOT claimed:**

- that it is the first such system;
- algorithmic novelty of any component;
- real-time operation;
- flood-prediction accuracy;
- segmentation accuracy;
- drop-point safety;
- geographic accuracy for the supplied maps;
- flight readiness;
- any Mission Planner, SITL or flight validation.

**Title alternatives considered:**

1. "From Flood Maps to Flight Plans: …"
2. "A Training-Free Image-to-Mission Pipeline for UAV Flood Relief"
3. "Verifiable Generation of UAV Relief Missions from Annotated Flood Maps"

The chosen title states the inputs, the verification emphasis and the output without marketing
language.

## 4. Evidence for every major contribution

See `claim_evidence_map.csv` (35 entries, each classified as experimentally demonstrated, verified by
software tests, mathematically derived, supported by literature/documentation, or stated limitation).

## 5. Experimental evidence (all in `paper/experiments/results`)

| ID | Experiment | Script |
|---|---|---|
| E1 | End-to-end runs | `e2e_maps.py` |
| E2 | Ablation of predicted-flood obstacles | `ablation_and_sensitivity.py` |
| E3 | HSV / click / scale sensitivity | `ablation_and_sensitivity.py` |
| E4 | NN vs. Held–Karp optimum | `tsp_optimality.py` |
| E5 | D* Lite vs. A* baseline | `planner_baseline.py` |
| E6 | Segmentation IoU | **not possible — no ground-truth masks (TODO)** |
| E7 | Geodesy | `geodesy_eval.py` |
| E8 | Mutation testing | `mutation_eval.py` |
| E9 | Landmark scale plausibility | `map_scale_estimate.py` |
| E10 | Environment + test suite | `env_and_tests.py` |
| — | Figures | `make_figures.py` |
| — | Manuscript audit | `audit_manuscript.py` |

All experiments call the unmodified pipeline (`main.main()` or `src/*`). The only substitutions are
the simulated operator for colour clicks, suppressed GUI windows, a redirected output path, and the
explicit ablation switches.

## 6. Methodology/code consistency audit

Each statement in Sec. IV was checked against the source:

- `main.py`: order of stages; preflight; stale-mission handling; reference pixel; k = ρ/s;
- `image_processing.py`: 11×11 patch; margins 20/30; red-wrap rule; close-then-open with 5×5;
- `clustering.py`;
- `weather_api.py`;
- `flood_spread.py`: ⌊P/5⌋, ⌊V/10⌋, θ+180°, N = max(1, ⌊T⌋), ellipse kernel, warpAffine shift;
- `safe_dropzone.py`: centroid polyline, first-vertex fallback;
- `pathfinding.py`: factor 5, INTER_NEAREST, BFS radius 10, new D* Lite per leg, straight fallback,
  (5gx, 5gy);
- `dstarlite.py`: Chebyshev heuristic, Euclidean costs, key definition, greedy extraction;
- `coordinates.py`: equations and rejection rules;
- `mission_output.py`: item sequence and parameters.

Two wording errors found in the draft were corrected:

- "sampled corner" → "top-left corner; sampled row may differ by ≤ 2 px";
- the unverified "may exceed mission storage" → a statement that capacity is not checked.

## 7. Unsupported claims removed or reworded during drafting

- The BFS-reachability claim (supported only by a scratchpad check) was replaced by the in-repository
  A* evidence.
- "0.56 m" was corrected to 0.57 m (0.5647 m).
- "5 km offsets" → "≤ 7.1 km offsets".
- "Halves or better" → exact reductions of 47 % and 76 %.
- The safe-landing works' "delivery-point selection" mark in Table I was downgraded to partial.
- Simantiris et al.'s accuracy figures (from a search summary, not the verified abstract) are not
  quoted in the paper.
- The 15-minute precipitation interval is stated "for example" and attributed to the Open-Meteo
  documentation.

## 8. Limitations (summary; full list in Sec. VI of the paper)

- **Geography:** maps not georeferenced; scale ×46–317 off; default reference used for all maps;
  flat-earth model error; web-Mercator mismatch; resize anisotropy ≤ 0.061 %.
- **Perception:** colour extraction from annotated maps only; operator-dependent; Assam at 83 %
  flood; no ground truth.
- **Prediction:** uncalibrated; pixel units; interval-sum precipitation treated as hourly.
- **Selection:** no safety model.
- **Ordering:** up to 44.5 % above optimal.
- **Planning:** 14 straight fallback legs; D* Lite rebuilt per leg (61–112× slower than A*);
  1-in-25 grid sampling.
- **Mission:** servo never reset; LAND after RTL redundant; PX4 yaw 0; capacity and endurance not
  checked (263 km at the estimated Varanasi scale).
- **Validation:** no GCS, SITL, HITL or flight.

## 9. Unresolved issues and recommended future validation (ranked)

See `TODO_for_authors.md` §3.
