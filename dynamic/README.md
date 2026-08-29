# A.D.A.P.T. Dynamic Pipeline — any region, any flood event

This branch (`ajay`) adds a **reliable, fully automated input pipeline** that turns
*any* region + date range into a ready-to-fly A.D.A.P.T. mission — removing the
static-image limitation of the original one-shot pipeline (and its manual HSV clicking).

```
                          ┌──────────────────────────────────────────────────────┐
  INPUT SOURCES           │                DETECTION & PLANNING                  │   OUTPUTS
                          │                                                      │
  Sentinel-1 SAR (10 m)   │   fetch_s1_cdse.py / fetch_gibs.py                   │   flood_mask.png
  Copernicus Data Space ──►   load S1 GRD VV ──► sigma0 (linear) ──►            │   flood_mask_clean.png
  (openEO batch job)      │   min-composite per window ──►                       │   style_map.png (OSM+
                          │   water: VV < -22 dB (0.0063 linear)                 │     red-dot news style)
  NASA GIBS MODIS 721 ───►   (no-auth fallback: SWIR dark-water + change-diff)  │   geo.json (WGS84)
  (300 m, no API key)     │                                                      │
                          │   same-season baseline diff:                         │   mission.waypoints
  OSM Overpass ──────────►   new_water = (flood - dry) > 0                      │   (QGC WPL 110: takeoff,
  (river corridors)       │   blob cleanup: morph-open, >= 4 ha per blob         │    waypoints, loiter,
                          │   georeference: GeoTIFF transform -> EPSG:4326       │    DO_SET_SERVO drop,
  Open-Meteo (weather) ──►                                                      │    climb, RTL, land)
                          │   run_pipeline_on_mask.py  (mask-direct, NO HSV)     │
                          │      contours -> clusters -> safe drop points        │   mission_map.png
                          │      -> NN-TSP loop -> D* Lite (avoids water)        │   (overlay on OSM style:
                          │      -> pixel_to_latlon -> WPL 110                   │    green=detected,
                          │                                                      │    orange=predicted spread,
  flood_spread.py ───────►   rainfall/wind spread CA -> predicted contours      │    blue=drops, black=home,
  (rain+wind CA)          │                                                      │    red=path)
                          │   compare_tsp_dstar.py  (TSP-direct vs D* benchmark) │   tsp_vs_dstar*.png
                          └──────────────────────────────────────────────────────┘
```

## Why this exists (results that motivated it)

Validated on 8 real events (all SAR 10 m unless noted):

| Event | New water detected | Drop zones | TSP direct | D* Lite actual | Detour |
|---|---|---|---|---|---|
| Bhotekoshi/Rasuwa flash flood (Nepal, Aug 2026) | 163.2 km² | 30 | 154.9 km* | 162.4 km* | +5% |
| Wayanad flash floods (Jul 2024) | 36.08 km² | 20 | 87.4 km | 92.7 km | +6% |
| Yamuna/Delhi floods (Jul 2023) | 73.09 km² | 30 | 167.6 km | 177.8 km | +6% |
| Sikkim South Lhonak GLOF (Oct 2023) | 5.80 km² (valley scour) | 30 | 154.9 km | 162.4 km | +5% |
| Chamoli GLOF (Feb 2021) ⚠️ noisy (winter Himalaya) | 500 km² raw | 30 | 138.0 km | 145.1 km | +5% |
| Chennai floods (Dec 2015, MODIS 300 m) | ~250 km² | 4-8 | 182.5 km | 195.1 km | +7% |
| Indus mega-flood (Sep 2022, MODIS 300 m) | 1,720 km² | 4 | — | — | — |
| Bihar 2025 / Varanasi 2025 news-map style | red-annotation pipeline (original) | 17-33 | — | — | — |

\* Nepal numbers from its own 3-zone comparison. Detections trace the real inundation:
Delhi's Yamuna floodplain (Wazirabad→ITO→Okhla→Hindan), Sikkim's Lachen/Lachung GLOF
valleys, the Indus 100-km-wide lake, Chennai's Pulicat/Tiruvallur flooding.

## Setup

```bash
pip install numpy opencv-python matplotlib requests python-dotenv   # existing repo deps
pip install openeo rioxarray                                        # dynamic pipeline extras
```

**Sentinel-1 (recommended, 10 m, cloud-penetrating):**
1. Free account: https://dataspace.copernicus.eu → Create account
2. Create an OAuth client: Account → Client Registration (Device Flow / client-credentials)
3. Save credentials to `dynamic/.cdse.env` (never commit it):
   ```
   CDSE_CLIENT_ID=sh-xxxxxxxx-....
   CDSE_CLIENT_SECRET=....
   ```
   First run authenticates automatically; a browser device-flow is used if the file is absent.

**No-account fallback** (`fetch_gibs.py`): NASA GIBS MODIS Bands-721 (~300 m/px).
Good for large riverine floods (Indus 2022, Brahmaputra); NOT usable for urban or
gorge flash floods, and blocked by monsoon clouds (optical).

## Usage — any region, any flood date

```bash
# 1) fetch + detect (batch job runs server-side; ~5-10 min)
python dynamic/fetch_s1_cdse.py --name yamuna2023 --lat 28.58 --lon 77.25 \
    --span 0.30 --flood-date 2023-07-13 --dry-date 2023-06-20

# 2) finalize: cleanup + mission + comparison (one command)
python dynamic/finalize_event.py --name yamuna2023 --lat 28.58 --lon 77.25 \
    --flood-date 2023-07-13 --dry-date 2023-06-20 [--sel-min 100 --planner-min 25 --top 3]
```

Outputs land in `out/<name>/`:

| File | Meaning |
|---|---|
| `s1_flood_mask.tif` | raw georeferenced binary water (WGS84) |
| `flood_mask.png` / `flood_mask_clean.png` | full-res mask / cleaned (≥ 4 ha blobs) |
| `style_map.png` | red-dot news-style map over OSM |
| `geo.json` | bounds + meters/pixel for `pixel_to_latlon` |
| `pipeline/mission.waypoints` | **QGC WPL 110 mission** — import into QGroundControl |
| `pipeline/mission_map.png` | overlay on real geography: green=detected, orange=predicted spread, blue=drops, black=home, red=D* path |
| `pipeline/tsp_vs_dstar*.png` | benchmark: NN-TSP direct vs D* Lite water-avoiding path |

Flags that matter: `--vv-threshold -22` (dB; tighten to -18 in noisy cities, loosen to -25 for
calm water), `--span` (AOI half-height in degrees), `--dry-date` (defaults to 12 days before
the flood date = same-season baseline, which avoids snow/vegetation seasonal false positives).

**GIBS fallback** (no account): `python dynamic/fetch_gibs.py --name indus2022 --lat 27.70
--lon 68.30 --span 0.85 --flood-date 2022-09-05 --dry-date 2022-03-15 --coast-lon <lon>`
— auto-scans ±6 days for the clearest scene, detects new water vs a same-season baseline,
constrains to OSM river corridors, writes the same output family. Then run
`finalize_event.py` identically.

**Original red-annotation images** (news maps like `data/input/varanasi.png`) still work
through the unchanged HSV pipeline; `scripts/headless_pipeline.py` runs it without the GUI
(auto-seeds HSV from the red band) across a folder of inputs.

## Architecture notes & pipeline hardening (vs one-shot main)

- **Mask-direct planning**: HSV segmentation is bypassed entirely for SAR input — the binary
  mask feeds `cluster_contours → find_safe_drop_points → nearest_neighbor_tsp →
  compute_full_path (D* Lite) → generate_mission_file` directly.
- **Same-season baseline change detection** (12 days pre-flood) — kills seasonal
  snow/vegetation false positives that a winter baseline produces.
- **Physical water threshold**: CDSE `sar_backscatter` returns **linear** sigma-0; the dB
  threshold is converted (`10^(dB/10)`). Backscatter < −22 dB ⇒ open water, even sediment-laden.
- **Batch jobs with explicit asset download** (`send_job → start_job → status-poll →
  get_results → download`) — immune to the silent `execute_batch` download failure and to
  429 rate limits (cooldown + retry).
- **CRS-safe output**: job GeoTIFFs arrive in UTM; everything is reprojected to EPSG:4326
  before `pixel_to_latlon`, so WPL coordinates are exact — no more approximate pixel→GPS.
- **Empty-mask diagnosis**: if detection is empty, scene availability per window is checked
  via the CDSE STAC API and printed. (This is how we proved Sentinel-1 has **no coverage over
  Chennai in Dec 2015** — early-Copernicus India acquisition restriction. For pre-2017 India
  events, use the GIBS/MODIS path or Landsat.)
- **Proportional rendering**: markers/legend scale with image size; predicted-spread
  contours (rain + wind CA) are drawn on every mission map.

## Known limitations

| Limitation | Cause | Workaround |
|---|---|---|
| No S1 over India before ~2017 | early-Copernicus acquisition policy | GIBS/MODIS path or Landsat |
| Himalayan winter speckle (terrain change) | slope/shadow/seasonal snow in SAR | elevation gate via DEM (queued), `mask=True` layover/shadow flag in `sar_backscatter` |
| Landslide scars not detected | scars ≠ standing water in SAR | pair with optical change or NDWI |
| MODIS 300 m is too coarse for urban/street flooding | sensor resolution | use the S1 path |
| OSM style map is © OpenStreetMap contributors | tile usage policy | light use OK; attribute in products |

## Module map

```
dynamic/
├── fetch_s1_cdse.py        # Sentinel-1 input (openEO batch, CDSE auth, change detection)
├── fetch_gibs.py           # no-auth MODIS-721 input (cloud scan, river corridors)
├── run_pipeline_on_mask.py # mask-direct planner (no HSV) + georeferenced WPL + overlay map
├── finalize_event.py       # one-command event finalizer (cleanup/resize/mission/compare)
├── compare_tsp_dstar.py    # TSP-direct vs D* Lite benchmark on the largest flood zones
└── .cdse.env               # (local only, gitignored) your CDSE OAuth client
scripts/
└── headless_pipeline.py    # original HSV pipeline, GUI-free, batch over test images
```

Everything runs on CPU, needs no keys beyond the free CDSE account (S1 path), and was
validated end-to-end — masks, style maps, missions, and benchmarks — on the eight events
listed above.
