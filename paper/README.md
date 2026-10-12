# A.D.A.P.T. journal manuscript

The editable journal article is [main.tex](main.tex) with section files in `sections/`; [main.pdf](main.pdf) is the compiled 14-page draft. [supplement.tex](supplement.tex) and [supplement.pdf](supplement.pdf) hold evidence figures S1–S7. The article reports held-out flood-segmentation baselines, two observed flood transitions, a georeferenced diagnostic mission, route and fleet comparisons, software verification, and ArduCopter SITL with MAVProxy upload/readback. The three original demonstration maps remain ungeoreferenced, and the short SITL flight does not establish physical payload release or field readiness.

## Build

From this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
```

The optional `--geotiff-reference` runtime path requires `pip install -r requirements-geotiff.txt` from the repository root. The reference GeoTIFF must match the original input image dimensions and registration. `--max-mask-fraction 0.8` rejects an oversized detected mask before route planning and export. These options do not turn raw satellite images into the red-annotation input expected by the current HSV stage.

Experiment scripts and measured results are in `experiments/`; `experiments/faculty_figures.py` regenerates S1–S7 from those results. Run `python paper/experiments/audit_manuscript.py` from the repository root to check labels and citations. Before submission, confirm permissions for third-party map graphics and complete the funding, competing-interest and author-contribution declarations.
