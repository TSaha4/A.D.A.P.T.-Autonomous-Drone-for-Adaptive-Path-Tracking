# Reproduce the faculty-paper results

Run commands from the repository root. The article source, compiled PDFs, test code, recorded JSON/CSV/NumPy outputs, plots, and generated WPL mission files are tracked in this branch. The three original demonstration maps are in `data/input/`. No frontend simulation files are needed.

## Environment

Use Python 3.11 for one shared environment. The experiment runs used Python 3.11.15 for the U-Net and SITL tools and Python 3.14.7 for other baselines. Recorded package versions are in [requirements-paper.txt](requirements-paper.txt). A CPU-only setup is:

```sh
python3.11 -m venv .venv-paper
. .venv-paper/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r paper/experiments/requirements-paper.txt
```

Re-running under a different OS or numerical-library build can change runtimes, optimisation tie-breaking, and the trained model's last digits. The recorded outputs and trained checkpoint remain available for direct inspection.

## Source data and provenance

[The input manifest](input_manifest.json) records byte lengths and SHA-256 checksums for 956 external files (about 988 MiB), seven repository inputs, the exact ArduCopter source commit, and the source URLs. The large third-party files are retrieved into the ignored `paper/experiments/cache/` directory. They are:

- [Sen1Floods11 v1.1](https://github.com/cloudtostreet/Sen1Floods11): official hand-labelled train, validation, and test split CSV files plus the referenced Sentinel-1, Sentinel-2, and label GeoTIFFs, fetched from the project's public Google Cloud Storage bucket.
- [Copernicus EMSR504 AOI01](https://mapping.emergency.copernicus.eu/activations/EMSR504/): the initial delineation and two monitoring vector packages, fetched from the product links on the official activation page.
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api): the exact retrospective hourly response used for EMSR504 is committed as [emsr504_weather.json](reference/emsr504_weather.json). Its query, coordinates and interval are in the manifest. The committed snapshot avoids changes in reanalysis data affecting the published comparison.

The trained U-Net weights are committed as [unet_best_state.pt](models/unet_best_state.pt), and the training script and seed are available. The original maps are third-party graphics already present in this repository; their provenance and lack of ground truth are disclosed in the article.

```sh
python paper/experiments/fetch_external_inputs.py
python paper/experiments/fetch_external_inputs.py --verify-only
```

The second command checks every downloaded input and the tracked source snapshots against the manifest. A mismatch fails rather than silently accepting different data.

## Tests and numerical experiments

```sh
python -m pytest -q tests
python paper/experiments/e2e_maps.py
python paper/experiments/sen1_benchmark.py
python paper/experiments/sen1_unet.py
python paper/experiments/sen1_qualitative.py
python paper/experiments/geotiff_georeference.py
python paper/experiments/geotiff_mission_demo.py
python paper/experiments/observed_flood_forecast.py
python paper/experiments/route_ordering_baselines.py
python paper/experiments/operator_robustness.py
python paper/experiments/assam_gate.py
python paper/experiments/fleet_balancing.py
python paper/experiments/fleet_ortools_allocation.py
python paper/experiments/fleet_sensitivity.py
```

Some fleet and operator sweeps can take several minutes. Run them serially to avoid changing the reported wall times through contention. The U-Net script retrains the model; `sen1_qualitative.py` uses the newly trained cache checkpoint when present and otherwise uses the committed checkpoint. The scripts write into `paper/experiments/results/`, where the measured outputs are already archived. `e2e_maps.py` uses deterministic simulated clicks; it does not collect human-operator measurements.

To regenerate the figures after the numerical outputs exist:

```sh
python paper/experiments/results_figures.py
python paper/experiments/new_research_figures.py
python paper/experiments/faculty_figures.py
```

The manuscript also uses source figures in `paper/figures/`; the corresponding TeX files and generation scripts are tracked. From `paper/`, build the article and supplement with `pdflatex main.tex`, `bibtex main`, two more `pdflatex main.tex` passes, and `pdflatex supplement.tex`. The compiled [main.pdf](../main.pdf) and [supplement.pdf](../supplement.pdf) are tracked.

## SITL replay

The observed run used [ArduCopter 4.7.1](https://github.com/ArduPilot/ardupilot/tree/Copter-4.7.1), source commit `dbe792162d06cab66c3475fd5556bf7a120f119e`, MAVProxy 1.8.75, and pymavlink 2.4.50. Build ArduPilot's SITL `copter` target from that commit using its documented `waf` workflow. The recorded parameter defaults are [arducopter_sitl.parm](reference/arducopter_sitl.parm). Start the binary with:

```sh
arducopter --wipe --home 25.3185895796,82.9734749884,0,0 --model quad --speedup 5 --defaults paper/experiments/reference/arducopter_sitl.parm
python paper/experiments/sitl_check.py
python paper/experiments/mavproxy_groundstation.py
```

The binary must be on `PATH`, or replace `arducopter` with its absolute path. `mavproxy_groundstation.py` accepts `MAVPROXY_BIN` if the command is elsewhere. Both scripts connect to SITL TCP port 5760; run them one after the other. The archived [SITL results](results/sitl/) contain the WPL inputs and downloaded mission, item acknowledgements, trajectory, servo and battery telemetry, and the MAVProxy terminal transcript. The terminal transcript has only local path prefixes normalized; no telemetry or mission command was changed.

The SITL timings and locations may vary with the simulator build and host. The full map mission was uploaded, not flown. The short reload mission was flown only in simulation, and the second physical payload release was never demonstrated.
