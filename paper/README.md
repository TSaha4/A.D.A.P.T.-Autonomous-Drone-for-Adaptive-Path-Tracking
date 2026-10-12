# A.D.A.P.T. journal manuscript

The editable manuscript is [main.tex](main.tex) with section files in `sections/`; [main.pdf](main.pdf) is the compiled 14-page draft. The paper follows the proposed journal structure and reports only measured results available in the accompanying experiment records. It does not present ground-station, SITL, field-flight, segmentation-ground-truth or flood-forecast accuracy results as completed experiments.

## Build

From this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Run `python paper/experiments/results_figures.py` from the repository root to recreate the Varanasi/Kanpur pipeline, Kanpur multi-UAV route and fleet-performance figures from recorded data. Run `python paper/experiments/audit_manuscript.py` to check citations and labels. Before submission, confirm permissions for third-party map graphics and complete the funding, competing-interest and author-contribution declarations.
