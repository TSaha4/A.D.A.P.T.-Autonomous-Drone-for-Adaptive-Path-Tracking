# Adversarial QA suite

Evidence for `docs/QA_REPORT.md`. These files are deliberately **not** named `test_*.py`, so a plain
`python -m pytest` (the fast regression suite in `tests/`) does not collect them; several of them run the full
pipeline hundreds of times. Run them from the repository root with the venv active
(`pip install hypothesis` is needed for the property tests).

| Command | What it does | Time (laptop) |
|---|---|---|
| `python -m pytest qa/qa_units.py` | Adversarial unit and property tests (Hypothesis): coordinates in all hemispheres, clustering of degenerate contours, dry drop points, TSP, D* Lite vs. Dijkstra on random grids, both planners, crossing test vs. brute force, flood frames, battery model, sortie planner, base selection, export round trips | ~1.5 min |
| `python -m pytest qa/qa_repairs.py` | Property test of the crossing-repair loop on random crossing plans | ~1 min |
| `python -m pytest qa/qa_pipeline.py` | End-to-end `main.main()` in both modes: bad files, bad CLI values, incomplete weather, bad environment variables, pathological maps, tiny/odd images, single-UAV cases S1–S12, multi-UAV cases M1–M14, determinism and state leakage | ~35 min |
| `python qa/fuzz.py 0 500 fuzz.json 8` | Seeded random end-to-end scenarios (map, scale, weather, base cap, mode) checked against all multi-base invariants (`harness.check_multi_session`); every failure is reproducible from its seed | ~40 min (8 workers) |
| `python qa/stress.py stress.json` | Growing numbers of flood zones in both modes: wall time, peak memory, per-stage time | > 1 h |

Tests marked `xfail(strict=True)` document latent issues QA-11 to QA-14 of the report (not reachable through the
program's inputs, therefore not fixed); if one of them is fixed the test will fail as "XPASS" and the marker must
be removed.

Raw results of the QA pass reported in `docs/QA_REPORT.md` are in `qa/results/`:

| File | Content |
|---|---|
| `fuzz_before_fixes_0_240.json` / `fuzz_after_fixes_0_500.json` | fuzzer records before and after the fixes |
| `stress_batch1.json` / `stress_batch2.json` | stress measurements |
| `qa_suites_before_fixes.txt` / `qa_suites_after_fixes.txt` | pytest output of the end-to-end suite before the fixes, and of the unit and end-to-end suites after them (the files had a `test_qa_` prefix then) |
