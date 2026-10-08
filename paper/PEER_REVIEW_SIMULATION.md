# Simulated Peer Review — A.D.A.P.T. manuscript (internal, pre-submission)

This is an adversarial self-review written as five independent IEEE Access–style reviewers would.
It is **not** a real review and predicts nothing about acceptance. For each issue, the last column says
whether the manuscript was revised ("Fixed"), whether the issue is disclosed but cannot be fixed
without new evidence ("Disclosed"), or whether it needs new work by the authors ("Open → TODO").

## Reviewer 1 — Technical correctness

**Strengths**

- The methodology is specified as implemented: equations (1)–(9), Algorithms 1–2, and the stated
  failure semantics match the code (checked line by line; see RESEARCH_AUDIT.md §6).
- The geodetic conversion has a stated validity domain and an independent WGS84 oracle, which was
  itself validated against a published geodesic.
- The planner is cross-checked against A*: costs are equal and reachability agrees.

| Concern | Severity | Response |
|---|---|---|
| Calling a dilation+shift rule a "cellular automaton" may overstate it | minor | Fixed: it is called a "weather-driven spread rule", with the CA reading stated as an interpretation and its heuristic, uncalibrated nature stated three times |
| Why D* Lite if it is rebuilt per leg? | major | Fixed (partly): the code-documented rationale is cited, and Sec. IV-I/V-G state plainly that the incremental property is unused and costs 61–112× runtime. Design change → future work |
| Straight-line fallback on no-path legs is unsafe | major | Disclosed: reported as a failure mode, quantified (14 legs), shown in red in Figs. 2–3, listed in Safety and Future Work. Fixing it would change the algorithm (frozen) |
| Drop points are chosen relative to a centroid polyline in discovery order, not the route | minor | Disclosed in Sec. IV-G and VI |
| 111,320 m/deg constant and uniform scale | major for geographic use | Disclosed and quantified (0.49 % at 25°N; 345 m–7.7 km at real extents) |
| Grid samples one pixel per 5×5 block | minor | Disclosed (Sec. IV-I, VI) |
| Timing from a single run | minor | Fixed: stated in Threats to Validity; the claims are ratios of 60–110×, which single runs support |

## Reviewer 2 — Novelty

**Strengths**

- The integration is clearly delimited, and the paper explicitly disclaims algorithmic novelty.
- The gap table is qualified and conservative.

| Concern | Severity | Response |
|---|---|---|
| Every component is standard; novelty is limited to integration | major (likely main rejection risk) | Disclosed: the contributions are framed as integration + verification methodology + characterisation including negative results. Cannot be fixed without new methods (frozen architecture) |
| The literature search is not systematic; integrated UAV flood systems may be missed | moderate | Fixed: Threats to Validity states the targeted-search limitation; absence claims are worded "in the reviewed literature". Open → TODO: a PRISMA-style search by the authors |
| Gap table rows group works coarsely | minor | Disclosed in the caption ("stated scope; not exhaustive") |

## Reviewer 3 — Experimental validity

**Strengths**

- Negative findings are reported (ablation trade-off, Assam failure, 61–112× planner overhead, NN gap).
- Every number is traceable to a result file.

| Concern | Severity | Response |
|---|---|---|
| Only three convenience images, all third-party news/thematic maps; no benchmark | major | Disclosed; Open → TODO (FloodNet, Sen1Floods11, own georeferenced products) |
| No segmentation ground truth (IoU/precision/recall) | major | Disclosed; Open → TODO (annotate the masks; script slot provided) |
| Simulated operator instead of humans | moderate | Disclosed; the sensitivity study (Table VII) shows its effect. Open → TODO: a user study with several operators |
| One fixed weather setting; prediction never validated | major | Disclosed (spread rule uncalibrated; calm-weather equivalence noted) |
| No Mission Planner / SITL / flight | major for any UAV venue | Disclosed in the abstract and Table IX. Open → TODO (cheapest high-value experiment) |
| Missions are geographically meaningless for the supplied maps | major | Disclosed prominently (Sec. V-I); the paper separates software correctness from geographic validity |

## Reviewer 4 — Writing quality

**Strengths**

- Restrained language; no overselling vocabulary (automated scan).
- Clear separation of verification levels.

| Concern | Severity | Response |
|---|---|---|
| Limitations are repeated across abstract, results, discussion and conclusion | minor | Accepted as deliberate for a hostile audience; repetition kept short |
| Long title | minor | Kept: it states scope precisely; alternatives are listed in RESEARCH_AUDIT.md |
| 13 pages including references (~11 pages of body) | minor | Within the 10–12-page body target; IEEE Access has no page limit |
| The template is IEEEtran, not the official IEEE Access template | administrative | Open → TODO: port before submission (README) |

## Reviewer 5 — Reproducibility

**Strengths**

- Scripts, raw results and figure generators are included; the runs are deterministic.
- Exact package versions are recorded (paper/experiments/requirements-paper.txt).

| Concern | Severity | Response |
|---|---|---|
| The project's requirements.txt is unpinned | minor | Fixed for the paper (pinned file in paper/); the project file is left unchanged by policy |
| Third-party images may not be redistributable | major for data availability | Open → TODO (permissions or replacement) |
| Tested on Windows only | minor | Disclosed (environment.json); Open → TODO: a Linux CI run |
| MiKTeX latexmk is broken on the authoring machine | build note | README documents the manual pdflatex/bibtex sequence |

## Likely editorial outcome (an honest estimate, not a prediction)

At a broad venue such as IEEE Access, the most probable outcome without SITL and benchmark experiments
is **major revision or rejection for limited novelty and validation**. Adding (1) ground-station + SITL
runs, (2) segmentation ground truth on at least one public dataset, and (3) one georeferenced input
would address the three most likely rejection reasons.

## Addendum — multi-UAV (multi-base) mode, added after the integration of the multi-base planner

| Concern | Severity | Response |
|---|---|---|
| "Multi-UAV" could be read as a swarm or cooperative system | major if unaddressed | Fixed: Sec. V states it is a pre-flight planner (one mission per base, no in-flight coordination); abstract, introduction and conclusion use the same wording |
| Nearest-feasible assignment does not balance workload; no mission-time objective | major | Disclosed with evidence (Sec. VI-H: Kanpur 17 / 17,8 / 17,7,6 / 15,7,6,6 drops per UAV); future work names VRP-style allocation |
| Coverage gains depend on the configured 2 m/px scale | major | Disclosed and measured: at landmark-estimated scales no drop is reachable (exit 2) |
| Battery coefficients are uncalibrated | major | Disclosed (Sec. V-C, Sec. VII); calibration listed as future work |
| Deconfliction is geometric only (no time/altitude separation) | major for any flight use | Disclosed (Sec. V-G, Sec. VII safety); SITL with several vehicles listed in TODO |
| Reload encoded as LAND + TAKEOFF untested on an autopilot | major for operational claims | Disclosed (Sec. VI-K); SITL check listed in TODO |
| Was the integrated code faithful to the original multi-base implementation? | minor | Fixed: identical plans on three cases (results/multi_uav/equivalence.json) |
| Completion time ignores take-off, landing, drops and reloads | minor | Disclosed as a lower bound in the table caption and text |
