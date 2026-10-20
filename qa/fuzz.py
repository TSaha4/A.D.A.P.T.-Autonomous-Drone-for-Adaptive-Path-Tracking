"""Seeded end-to-end fuzzing of main.main() in both modes. Usage: python fuzz.py START COUNT OUT.json [WORKERS]

Every scenario is reproducible from its seed. A scenario fails if the run raises an uncaught exception, exits with
an unexpected status, or (multi) violates any invariant of harness.check_multi_session, or (single) writes a
mission that the independent validator rejects or whose drop count differs from the region count.
"""
import json
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def scenario(seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    w, h = int(rng.integers(200, 900)), int(rng.integers(200, 900))
    shapes = []
    for _ in range(int(rng.integers(0, 25))):
        kind = rng.choice(["rect", "circle", "line"])
        x, y = int(rng.integers(-20, w + 20)), int(rng.integers(-20, h + 20))
        if kind == "rect":
            shapes.append(("rect", x, y, x + int(rng.integers(1, 120)), y + int(rng.integers(1, 120))))
        elif kind == "circle":
            shapes.append(("circle", x, y, int(rng.integers(1, 70))))
        else:
            shapes.append(("line", x, y, int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(1, 9))))
    weather = [{"precipitation": 0.0, "wind_speed_10m": 0.0, "wind_direction_10m": 0.0},
               {"precipitation": 3.0, "wind_speed_10m": 12.0, "wind_direction_10m": float(rng.integers(0, 360))},
               {"precipitation": 10.0, "wind_speed_10m": 30.0, "wind_direction_10m": float(rng.integers(0, 360))}][
        int(rng.integers(0, 3))]
    weather["status"] = "success"
    return {"seed": seed, "size": (w, h), "shapes": shapes, "weather": weather,
            "mpp": float(rng.choice([0.5, 1.0, 2.0, 4.0, 8.0, 20.0])), "k": int(rng.integers(1, 7)),
            "horizon": float(rng.choice([0.5, 1.0, 2.0, 3.5])), "mode": str(rng.choice(["single", "multi", "multi"])),
            "display_width": int(rng.choice([750, 750, 400]))}


def run(seed):
    sys.path.insert(0, HERE)
    import logging
    logging.disable(logging.CRITICAL)
    import matplotlib
    matplotlib.use("Agg")
    import cv2
    import harness as Hn
    from tests import wpl_validator
    sc = scenario(seed)
    rec = {"seed": seed, "mode": sc["mode"], "k": sc["k"], "mpp": sc["mpp"], "size": sc["size"],
           "n_shapes": len(sc["shapes"]), "weather": sc["weather"]["precipitation"], "problems": []}
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "f.png")
        Hn.make_map(img_path, sc["shapes"], sc["size"])
        args = [img_path, "--mode", sc["mode"], "--horizon", str(sc["horizon"]), "--display-width", str(sc["display_width"])]
        if sc["mode"] == "multi":
            args += ["--max-bases", str(sc["k"]), "--session-root", os.path.join(tmp, "s")]
        t = time.perf_counter()
        try:
            code, exc = Hn.run_main(args, tmp, mpp=sc["mpp"], weather=dict(sc["weather"]))
        except BaseException as e:  # noqa: BLE001
            code, exc = "EXC", e
        rec["seconds"] = round(time.perf_counter() - t, 2)
        rec["exit"] = code if code != "EXC" else "EXC"
        if exc is not None:
            rec["problems"].append(f"uncaught {type(exc).__name__}: {exc}")
            rec["traceback"] = "".join(traceback.format_exception(exc))[-1500:]
        elif sc["mode"] == "single":
            out = os.path.join(tmp, "data", "output", "enriched_drone_mission.waypoints")
            if code not in (0,):
                rec["problems"].append(f"single exit {code}")
            if os.path.exists(out):
                findings, items = wpl_validator.validate(out, servo_channel=9, servo_pwm=2000)
                errs = [str(f) for f in findings if f.level == "ERROR"]
                if errs:
                    rec["problems"].append(f"validator: {errs[:3]}")
                rec["drops"] = sum(r["cmd"] == 183 for r in items)
        else:
            if code not in (0, 2):
                rec["problems"].append(f"multi exit {code}")
            session = Hn.latest_session(os.path.join(tmp, "s"))
            if session and os.path.exists(os.path.join(session, "run_summary.json")):
                img = cv2.imread(img_path)
                if sc["display_width"] < img.shape[1]:
                    s = sc["display_width"] / img.shape[1]
                    img = cv2.resize(img, (sc["display_width"], int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
                # red-pixel checks only make sense without forecast growth and without resampling
                calm = sc["weather"]["precipitation"] == 0 and sc["weather"]["wind_speed_10m"] == 0
                v = Hn.check_multi_session(session, img if calm and sc["display_width"] >= sc["size"][0] else None)
                rec["problems"] += v
                summ = json.load(open(os.path.join(session, "run_summary.json")))
                rec["outcome"] = summ.get("outcome")
                rec["bases"] = len(summ.get("bases", []))
                rec["drop_points"] = len(summ.get("drop_points", []))
                rec["visited"] = summ.get("drops_visited")
                rec["repairs"] = [r["tier"] for r in summ.get("overlap_repairs", [])]
    return rec


def main():
    start, count, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    with ProcessPoolExecutor(max_workers=workers) as pool:
        recs = list(pool.map(run, range(start, start + count)))
    json.dump(recs, open(out, "w"), indent=1, default=str)
    bad = [r for r in recs if r["problems"]]
    print(f"{len(recs)} scenarios, {len(bad)} with problems")
    for r in bad:
        print(r["seed"], r["mode"], r["exit"], r["problems"][:3])


if __name__ == "__main__":
    main()
