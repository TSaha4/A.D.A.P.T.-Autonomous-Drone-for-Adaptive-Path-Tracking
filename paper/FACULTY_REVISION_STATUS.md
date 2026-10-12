# Faculty revision status — 12 October 2026

## Pushed branch provenance

GitHub reports `research-validation` at `fdb9cb3` and `main` at `ca27215`. The latter contains the pushed `time-aware-routing` commit `607bc5f`; `research-validation` is a separate line with merge base `775165d`. GitHub comparison reports `research-validation` three commits ahead and one behind `main`. This revision uses the committed paper experiments and code at `fdb9cb3` only. It does not use uncommitted files from either local checkout, nor does it present `main` behaviour as experimentally measured.

## Changes made to the manuscript

| Faculty request | Status in revised paper |
|---|---|
| 10–14 pages; abstract, related work by theme, problem, method, verification, setup, results, limitations, conclusion, declarations | Done: 14-page PDF with the requested sequence and a shortened abstract. |
| Testable contributions; remove cost-analysis framing | Done: three bounded contributions and runtime/resource language. No unsupported priority claim. |
| Symbols table and complexity analysis | Done. |
| Carry over system overview, Varanasi stages; add Kanpur stages | Done from committed records. |
| Real route/fleet output; coverage and makespan versus K | Done from committed route and comparison records. Completion time is explicitly an optimistic estimate. |
| Replace screenshot placeholders | Done: placeholders removed; absent SITL screenshots are not represented as results. |
| Independent coordinate oracle, WPL validator, mutation/fuzz tests | Done: methods and existing measured outcomes reported. |
| Consolidate limitations and state threats to validity | Done. |
| Data/code availability and author declarations | Code URL and exact evaluated commit included; author, funding and interest fields await the authors. |

## Evidence still required for the faculty's critical experiments

| Requested change | Why it is not reported as a completed result |
|---|---|
| ArduPilot SITL, ground-station item acceptance, servo, RTL, reload, battery and telemetry figures | No SITL run or telemetry exists at the evaluated pushed commit. The verification section gives the test protocol. |
| Georeferenced flood product, WGS84 overlay, labelled mask and IoU/F1/precision/recall | The three committed maps contain neither usable geographic control nor labelled flood truth. Synthetic geodetic tests cannot substitute for these. |
| HSV versus Otsu, Lab k-means, NDWI and trained U-Net; Assam mitigation | No common ground-truth benchmark or multispectral data is committed. Assam is documented as a failure, not a corrected outcome. |
| Forecast against two observed flood snapshots and no-prediction baseline | Only the current-mask ablation exists. No observed future mask is committed, so prediction skill cannot be scored. |
| 2-opt/Or-opt, OR-Tools TSP and multi-depot VRP baselines | The committed evaluations include nearest-neighbour versus Held–Karp on small cases and D* Lite versus A*; the requested additional solvers are not implemented or measured. |
| Multiple operator samples, seeds, weather scenarios and confidence intervals | The recorded pipeline uses one deterministic operator and limited weather presets. No sampling distribution supports a confidence interval. |
| Workload-balanced reassignment, before/after Gini and makespan | Current assignments and per-UAV workloads are recorded, but no balancing method is present or measured. |
| Battery coefficient calibration and ±20% sensitivity | Coefficients are assumptions; no logs or sensitivity sweep is committed. |
| Coverage versus metres-per-pixel curve | Results exist for the configured scale and two landmark-estimated failure points, not a full curve. |
| Runtime scaling | Only single-run timings on a few map instances are recorded. |

These items need new experiments and, for several, new source data. The paper keeps their absence visible instead of fabricating numbers or screenshots.
