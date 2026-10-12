# A.D.A.P.T. research manuscript

The revised paper is [main.pdf](main.pdf) (14 A4 pages); the editable source is [main.tex](main.tex) with section files in `sections/`. It follows the faculty's proposed paper structure and uses only committed evidence from the pushed `research-validation` branch at `fdb9cb3`. The other pushed development line is separate: `main` is `ca27215`, `time-aware-routing` is `607bc5f`, and the branches diverge at `775165d`. The results have not been rerun on `main`. Local uncommitted work in other checkouts was excluded.

## Build

The paper is typeset with the installed `extarticle` class and standard TeX packages. From this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`experiments/faculty_figures.py` regenerates the Varanasi/Kanpur pipeline, Kanpur fleet-route and fleet-performance figures from committed JSON, NPZ and route records. Other existing data figures remain in `figures/`; the paper includes only figures that correspond to recorded results. `experiments/audit_manuscript.py` checks citations, labels and the abstract length.

## Evidence boundary

The paper reports software verification, three demonstration maps, two recorded fleet weather settings, and route/mission behaviour for `fdb9cb3`. The maps lack flood ground truth and georeferencing. No SITL, ground-station, telemetry or flight result exists in this pushed branch. The requested segmentation accuracy, forecast accuracy, battery calibration, statistical intervals, VRP comparison and balancing improvement require new datasets, runs or implementation. See [FACULTY_REVISION_STATUS.md](FACULTY_REVISION_STATUS.md) for item-by-item status.

Before submission, confirm third-party map redistribution rights, complete the author/funding/interest declarations, and select the venue's required template. The manuscript does not claim operational flight readiness.
