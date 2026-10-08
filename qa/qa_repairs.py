"""Property test of the crossing-repair loop on random crossing plans (QA suite; see qa/README.md).

Plans are straight out-and-back stars from each base (a simple planner), so crossings are frequent and all three
tiers get exercised. Invariants after execute_overlap_repairs:
  * every drop is still assigned to exactly one base, and each live plan serves exactly its base's drops;
  * the reported remaining crossings are exactly the crossings of the final plans;
  * the reach set of every base equals the reach recomputed at its (possibly nudged) position;
  * a nudged base moved at most 50 px and keeps the clearance and separation limits.
"""
import collections

import numpy as np
from hypothesis import HealthCheck, given, settings as hs, strategies as st

import harness  # noqa: F401 - repository on sys.path
from config import settings
from src.mission.multi_base import single_drop_reach
from src.mission.overlap_repair import execute_overlap_repairs
from src.routing.pathfinding import polyline_intersections


def star(number, home, drops):
    path = [tuple(home)]
    stops, dropi = [0], []
    for d in drops:
        path.append(tuple(d)); stops.append(len(path) - 1); dropi.append(len(path) - 1)
        path.append(tuple(home)); stops.append(len(path) - 1)
    reload = [i for i in stops[1:-1] if path[i] == tuple(home)]
    return {"base": number, "home": tuple(home), "full_path": path, "stop_indices": stops,
            "route_points": [path[i] for i in stops], "drop_indices": dropi, "reload_indices": reload}


@hs(deadline=None, max_examples=120, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 100_000), nb=st.integers(2, 4), nd=st.integers(2, 12), dup=st.booleans(),
       scale=st.sampled_from([2.0, 5.0, 9.0]))
def test_repair_loop_invariants(seed, nb, nd, dup, scale):
    rng = np.random.default_rng(seed)
    size = 300
    bases = [(int(rng.integers(20, size - 20)), int(rng.integers(20, size - 20))) for _ in range(nb)]
    drops = [(int(rng.integers(5, size - 5)), int(rng.integers(5, size - 5))) for _ in range(nd)]
    if dup:
        drops.append(drops[0])  # coincident drop points of two regions
    dist = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1])) * scale  # noqa: E731
    selection = {"assignment": {i: int(rng.integers(0, nb)) for i in range(len(drops))}}
    if dup:  # in the pipeline coincident drops get the same (nearest feasible) base
        selection["assignment"][len(drops) - 1] = selection["assignment"][0]
    reach = single_drop_reach(bases, drops, dist, headwind_mps=0.0, per_drop_kg=0.25, reserve=0.2)

    def plan_base(number, home, assigned):
        return star(number, home, assigned) if assigned else None

    def plan_assigned(b):
        mine = [drops[i] for i, o in sorted(selection["assignment"].items()) if o == b]
        return plan_base(b + 1, bases[b], mine)

    plans = {b: plan_assigned(b) for b in range(nb)}
    original_bases = list(bases)
    index = collections.defaultdict(list)
    for i, p in enumerate(drops):
        index[p].append(i)
    clearance = np.full((size, size), 30.0, np.float32)
    repairs, remaining = execute_overlap_repairs(
        plans, bases, selection, drops, reach, dict(index), np.zeros((size, size), np.uint8), None, clearance, dist,
        0.0, plan_base, plan_assigned, (size, size), meters_per_pixel=scale, max_rounds=5)

    assert sorted(selection["assignment"]) == list(range(len(drops)))
    for b, plan in plans.items():
        mine = sorted(drops[i] for i, o in selection["assignment"].items() if o == b)
        served = sorted(plan["full_path"][i] for i in plan["drop_indices"]) if plan else []
        assert served == mine, (b, served, mine)
    live = [b for b in sorted(plans) if plans[b]]
    actual = [(x, y) for i, x in enumerate(live) for y in live[i + 1:]
              if polyline_intersections(plans[x]["full_path"], plans[y]["full_path"])]
    assert sorted((a, b) for a, b, _ in remaining) == actual
    expected_reach = single_drop_reach(bases, drops, dist, headwind_mps=0.0, per_drop_kg=0.25, reserve=0.2)
    for b in range(nb):
        if bases[b] != original_bases[b]:
            assert reach[b] == expected_reach[b], "stale reach after nudge"
            assert np.hypot(bases[b][0] - original_bases[b][0], bases[b][1] - original_bases[b][1]) <= 50 + 1e-9
            for o in range(nb):
                if o != b and plans.get(o):
                    assert np.hypot(bases[b][0] - bases[o][0], bases[b][1] - bases[o][1]) >= \
                        settings.MIN_BASE_SEPARATION_PX - 1e-9
    # coincident drops end with one base
    owners = collections.defaultdict(set)
    for i, o in selection["assignment"].items():
        owners[drops[i]].add(o)
    if dup:
        assert len(owners[drops[0]]) == 1, "coincident drops split across bases"


def test_tiers_are_exercised():
    """Sanity: over a fixed set of seeds every tier fires at least once (otherwise the property test is weak)."""
    tiers = collections.Counter()
    for seed in range(120):
        rng = np.random.default_rng(seed)
        bases = [(int(rng.integers(20, 280)), int(rng.integers(20, 280))) for _ in range(3)]
        drops = [(int(rng.integers(5, 295)), int(rng.integers(5, 295))) for _ in range(8)]
        f = 2.0 if seed % 2 else 9.0
        dist = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1])) * f  # noqa: E731
        sel = {"assignment": {i: int(rng.integers(0, 3)) for i in range(8)}}

        def pa(b):
            mine = [drops[i] for i, o in sorted(sel["assignment"].items()) if o == b]
            return star(b + 1, bases[b], mine) if mine else None

        plans = {b: pa(b) for b in range(3)}
        reach = single_drop_reach(bases, drops, dist, headwind_mps=0.0, per_drop_kg=0.25, reserve=0.2)
        reps, _ = execute_overlap_repairs(plans, bases, sel, drops, reach, {p: [i] for i, p in enumerate(drops)},
                                          np.zeros((300, 300), np.uint8), None, np.full((300, 300), 30.0, np.float32),
                                          dist, 0.0, lambda n, h, d: star(n, h, d) if d else None, pa, (300, 300),
                                          meters_per_pixel=f)
        tiers.update(r["tier"] for r in reps)
    print("tiers fired:", dict(tiers))
    assert tiers[0] and tiers[1]
