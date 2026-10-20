"""Stress test: growing numbers of flood zones (= drop points) in both modes; wall time, peak memory, stage times.

Usage: python stress.py OUT.json
Each configuration runs in its own process so that peak memory (psutil not available: Windows job counters via
ctypes GetProcessMemoryInfo) is per run.
"""
import ctypes
import ctypes.wintypes
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_mb():
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(PMC), ctypes.wintypes.DWORD]
    c = PMC()
    c.cb = ctypes.sizeof(c)
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return None
    return round(c.PeakWorkingSetSize / 2 ** 20, 1)


def one(mode, n, k, mpp):
    sys.path.insert(0, HERE)
    import logging
    logging.disable(logging.CRITICAL)
    import harness as Hn
    import main as M
    from unittest import mock
    times = {}

    def timed(name, fn):
        def w(*a, **kw):
            t = time.perf_counter()
            r = fn(*a, **kw)
            times[name] = times.get(name, 0.0) + time.perf_counter() - t
            return r
        return w

    side = int(n ** 0.5 + 0.999)
    step = 740 // side
    shapes = [("rect", 5 + step * (i % side), 5 + step * (i // side), 5 + step * (i % side) + max(4, step // 3),
               5 + step * (i // side) + max(4, step // 3)) for i in range(n)]
    with tempfile.TemporaryDirectory() as tmp:
        img = os.path.join(tmp, "f.png")
        Hn.make_map(img, shapes)
        args = [img, "--mode", mode, "--min-area", "50"] +(["--max-bases", str(k), "--session-root", os.path.join(tmp, "s")] if mode == "multi" else [])
        patches = [mock.patch.object(M, n_, timed(n_, getattr(M, n_))) for n_ in
                   ("select_bases", "execute_overlap_repairs", "generate_mission_file", "compute_full_path",
                    "compute_routed_path", "find_safe_drop_points", "routed_distance_function", "nearest_neighbor_tsp",
                    "display_multi_base_map")]
        t = time.perf_counter()
        code, exc = Hn.run_main(args, tmp, mpp=mpp, extra=patches)
        total = time.perf_counter() - t
        res = {"mode": mode, "zones": n, "k": k, "mpp": mpp, "exit": code, "exc": repr(exc) if exc else None,
               "total_s": round(total, 2), "peak_mb": peak_mb(),
               "stage_s": {k_: round(v, 2) for k_, v in times.items()}}
        if mode == "multi":
            s = Hn.latest_session(os.path.join(tmp, "s"))
            if s and os.path.exists(os.path.join(s, "run_summary.json")):
                sm = json.load(open(os.path.join(s, "run_summary.json")))
                res.update(drop_points=len(sm.get("drop_points", [])), bases=len(sm.get("bases", [])),
                           visited=sm.get("drops_visited"), outcome=sm.get("outcome"))
        print(json.dumps(res))


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
        sys.exit(0)
    results = []
    configs = [tuple(json.loads(c)) for c in sys.argv[2:]]
    configs = configs or [("single", n, 1, 2.0) for n in (10, 50, 100, 300, 600)] + \
              [("multi", n, k, 2.0) for n in (10, 50, 100, 300, 600) for k in (1, 4)] + \
              [("multi", 100, k, 2.0) for k in (8, 16)]
    for cfg in configs:
        r = subprocess.run([sys.executable, __file__, "--one", *map(str, cfg)], capture_output=True, text=True,
                           timeout=3600)
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        rec = json.loads(line[-1]) if line else {"config": cfg, "error": r.stderr[-800:]}
        print(rec, flush=True)
        results.append(rec)
    json.dump(results, open(sys.argv[1], "w"), indent=1)
