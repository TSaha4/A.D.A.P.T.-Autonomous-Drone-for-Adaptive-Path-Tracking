"""
Independent validator for QGC WPL 110 missions produced by A.D.A.P.T.

Standard library only; deliberately imports nothing from src/, so it is an independent check
of src/mission/mission_output.py and src/mission/coordinates.py.

Usage:
    python -m tests.wpl_validator MISSION.waypoints [--sidecar ROUTE.json] [--table OUT.csv]

The optional sidecar JSON holds what the pipeline passed to generate_mission_file
(full_path, drop_indices, home, image_center_px, geo_center, meters_per_pixel) and, optionally,
expected_drops (number of flood regions). With it, every mission row is re-derived
independently from its pixel coordinate and compared.

Exit status: 0 if no ERROR findings, 1 otherwise.
"""
import argparse
import csv
import json
import math
import sys

HEADER = "QGC WPL 110"
NAV_WAYPOINT, NAV_LOITER_TIME, NAV_RTL, NAV_LAND, NAV_TAKEOFF, DO_SET_SERVO = 16, 19, 20, 21, 22, 183
KNOWN_COMMANDS = {NAV_WAYPOINT: "NAV_WAYPOINT", NAV_LOITER_TIME: "NAV_LOITER_TIME", NAV_RTL: "NAV_RETURN_TO_LAUNCH",
                  NAV_LAND: "NAV_LAND", NAV_TAKEOFF: "NAV_TAKEOFF", DO_SET_SERVO: "DO_SET_SERVO"}
FRAME_GLOBAL_RELATIVE_ALT = 3
METERS_PER_DEGREE = 111320.0  # constant of the documented flat-earth model
COORD_TOL_DEG = 1e-9          # ~0.1 mm
MIN_DECIMALS = 7              # 1e-7 deg ~ 1.1 cm


class Finding:
    def __init__(self, level, check, message):
        self.level, self.check, self.message = level, check, message

    def __repr__(self):
        return f"[{self.level}] {self.check}: {self.message}"


def parse_wpl(path):
    with open(path) as f:
        lines = f.read().splitlines()
    items = []
    for n, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        items.append({"line": n, "raw": fields})
    return (lines[0] if lines else ""), items


def _num(s):
    return float(s)


def _decimals(s):
    s = s.strip().lower()
    if "e" in s:
        mantissa, exp = s.split("e")
        frac = mantissa.split(".")[1] if "." in mantissa else ""
        return len(frac) - int(exp)
    return len(s.split(".")[1]) if "." in s else 0


def _haversine_m(lat1, lon1, lat2, lon2, r=6371008.8):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


def expected_latlon(px, py, image_center_px, geo_center, meters_per_pixel):
    cx, cy = image_center_px
    lat0, lon0 = geo_center
    north = (cy - py) * meters_per_pixel
    east = (px - cx) * meters_per_pixel
    return lat0 + north / METERS_PER_DEGREE, lon0 + east / (METERS_PER_DEGREE * math.cos(math.radians(lat0)))


def expected_items(sidecar, cruise_alt, drop_alt):
    """Independently re-derives the intended mission (cmd, lat, lon, alt) from the route."""
    path = [tuple(p) for p in sidecar["full_path"]]
    drops = set(sidecar["drop_indices"])
    home = tuple(sidecar["home"])
    conv = lambda p: expected_latlon(p[0], p[1], sidecar["image_center_px"], sidecar["geo_center"],
                                     sidecar["meters_per_pixel"])
    h_lat, h_lon = conv(home)
    reloads = set(sidecar.get("reload_indices") or [])
    out = [(NAV_WAYPOINT, h_lat, h_lon, 0), (NAV_TAKEOFF, h_lat, h_lon, cruise_alt)]
    for i, pt in enumerate(path):
        if i == 0 and pt == home:
            continue
        if i in reloads:
            # Multi-base mode: back at HOME between sorties -> land for the reload, then take off again
            out += [(NAV_LAND, h_lat, h_lon, 0), (NAV_TAKEOFF, h_lat, h_lon, cruise_alt)]
            continue
        lat, lon = conv(pt)
        out.append((NAV_WAYPOINT, lat, lon, cruise_alt))
        # Intended design: every drop index except the HOME departure (first) and return (last) is a delivery
        if i in drops and 0 < i < len(path) - 1:
            out += [(NAV_LOITER_TIME, lat, lon, cruise_alt), (NAV_WAYPOINT, lat, lon, drop_alt),
                    (DO_SET_SERVO, lat, lon, drop_alt), (NAV_WAYPOINT, lat, lon, cruise_alt)]
    out += [(NAV_RTL, h_lat, h_lon, cruise_alt), (NAV_LAND, h_lat, h_lon, 0)]
    return out


def validate(path, sidecar=None, cruise_alt=100.0, drop_alt=10.0, servo_channel=None, servo_pwm=None,
             allow_reloads=False):
    """servo_channel / servo_pwm: the payload-release output the mission is meant to drive; when given,
    every DO_SET_SERVO must use exactly these values.
    allow_reloads: accept multi-base reload stops, i.e. a LAND at HOME (alt 0) immediately followed by a
    TAKEOFF at HOME to cruise altitude, before the final RTL/LAND. Implied when the sidecar lists
    reload_indices. Otherwise any LAND before the end of the mission is an error, as before."""
    allow_reloads = allow_reloads or bool(sidecar and sidecar.get("reload_indices"))
    findings = []
    err = lambda c, m: findings.append(Finding("ERROR", c, m))
    warn = lambda c, m: findings.append(Finding("WARN", c, m))
    info = lambda c, m: findings.append(Finding("INFO", c, m))

    header, raw_items = parse_wpl(path)
    if header.strip() != HEADER:
        err("header", f"first line is {header!r}, expected {HEADER!r}")

    items = []
    for it in raw_items:
        f = it["raw"]
        if f == [""]:
            warn("blank", f"line {it['line']}: blank line ignored")
            continue
        if len(f) != 12:
            err("columns", f"line {it['line']}: {len(f)} fields, expected 12")
            continue
        try:
            rec = {"line": it["line"], "seq": int(f[0]), "current": int(f[1]), "frame": int(f[2]),
                   "cmd": int(f[3]), "p": [_num(x) for x in f[4:8]], "lat": _num(f[8]), "lon": _num(f[9]),
                   "alt": _num(f[10]), "auto": int(f[11]), "lat_s": f[8], "lon_s": f[9]}
        except ValueError as e:
            err("parse", f"line {it['line']}: {e}")
            continue
        items.append(rec)
    if not items:
        err("empty", "no mission items")
        return findings, items

    # 4. sequence numbers
    seqs = [r["seq"] for r in items]
    if seqs != list(range(len(items))):
        bad = next(i for i, s in enumerate(seqs) if s != i)
        err("sequence", f"sequence numbers not contiguous from 0: item {bad} has seq {seqs[bad]}")

    # 5. format-level checks
    for r in items:
        if r["cmd"] not in KNOWN_COMMANDS:
            err("command", f"seq {r['seq']}: unexpected command {r['cmd']}")
        # Altitudes of positional items are only meaningful as "metres above HOME" in frame 3. The HOME row
        # may also use frame 0 (Mission Planner's convention); RTL and DO_SET_SERVO carry no position.
        if r["seq"] == 0:
            if r["frame"] not in (0, FRAME_GLOBAL_RELATIVE_ALT):
                err("frame", f"seq 0: frame {r['frame']} for HOME (expected 0 or 3)")
        elif r["cmd"] in (NAV_WAYPOINT, NAV_LOITER_TIME, NAV_LAND, NAV_TAKEOFF) and r["frame"] != FRAME_GLOBAL_RELATIVE_ALT:
            err("frame", f"seq {r['seq']}: frame {r['frame']}; altitude would not be relative to HOME (expected 3)")
        if r["auto"] != 1:
            err("autocontinue", f"seq {r['seq']}: autocontinue={r['auto']}")
        if r["current"] != (1 if r["seq"] == 0 else 0):
            warn("current", f"seq {r['seq']}: current flag {r['current']}")

    # 1-3. coordinate validity and precision
    for r in items:
        lat, lon = r["lat"], r["lon"]
        if not (math.isfinite(lat) and math.isfinite(lon)):
            err("coordinate", f"seq {r['seq']}: non-finite coordinate ({r['lat_s']}, {r['lon_s']})")
            continue
        if not -90.0 <= lat <= 90.0:
            err("latitude", f"seq {r['seq']}: latitude {lat} outside [-90, 90]")
        if not -180.0 <= lon <= 180.0:
            err("longitude", f"seq {r['seq']}: longitude {lon} outside [-180, 180]")
        if r["cmd"] != DO_SET_SERVO and lat == 0.0 and lon == 0.0:
            warn("coordinate", f"seq {r['seq']}: (0, 0) position")
        # A short decimal can be exact (e.g. a waypoint lying on the reference meridian), so from the file
        # alone truncation cannot be proven; with a sidecar the route comparison below decides it.
        if sidecar is None:
            for name, s in (("latitude", r["lat_s"]), ("longitude", r["lon_s"])):
                if _decimals(s) < MIN_DECIMALS:
                    warn("precision", f"seq {r['seq']}: {name} {s} has fewer than {MIN_DECIMALS} decimals; "
                                      f"cannot confirm it is untruncated without the route sidecar")

    # 7. HOME / takeoff / return / land consistency
    home = items[0]
    same = lambda a, b: abs(a["lat"] - b["lat"]) <= COORD_TOL_DEG and abs(a["lon"] - b["lon"]) <= COORD_TOL_DEG
    if home["cmd"] != NAV_WAYPOINT or home["alt"] != 0:
        err("home", f"item 0 should be the HOME row (cmd 16, alt 0); got cmd {home['cmd']} alt {home['alt']}")
    if len(items) < 4:
        err("structure", "mission shorter than HOME, TAKEOFF, RTL, LAND")
        return findings, items
    if items[1]["cmd"] != NAV_TAKEOFF:
        err("takeoff", f"item 1 is cmd {items[1]['cmd']}, expected NAV_TAKEOFF (22)")
    elif not same(items[1], home) or items[1]["alt"] != cruise_alt:
        err("takeoff", "takeoff not at HOME position or not to cruise altitude")
    # 9. return
    if [items[-2]["cmd"], items[-1]["cmd"]] != [NAV_RTL, NAV_LAND]:
        err("return", f"mission does not end with RTL, LAND: {[items[-2]['cmd'], items[-1]['cmd']]}")
    else:
        if not same(items[-1], home):
            err("return", "LAND position differs from HOME")
        if items[-1]["alt"] != 0:
            err("return", f"LAND altitude {items[-1]['alt']} (expected 0)")
        info("return", "RTL is followed by LAND; ArduCopter's RTL normally lands by itself, so the LAND item may never run")
    if not allow_reloads:
        if any(r["cmd"] in (NAV_RTL, NAV_LAND) for r in items[:-2]):
            err("return", "RTL/LAND appears before the end of the mission")
    else:
        if any(r["cmd"] == NAV_RTL for r in items[:-2]):
            err("return", "RTL appears before the end of the mission")
        for k, r in enumerate(items[:-2]):
            if r["cmd"] != NAV_LAND:
                continue
            nxt = items[k + 1]
            if not same(r, home) or r["alt"] != 0:
                err("reload", f"seq {r['seq']}: intermediate LAND is not at HOME with altitude 0")
            if nxt["cmd"] != NAV_TAKEOFF or not same(nxt, home) or nxt["alt"] != cruise_alt:
                err("reload", f"seq {r['seq']}: intermediate LAND is not followed by TAKEOFF at HOME to cruise altitude")
        n_reloads = sum(r["cmd"] == NAV_LAND for r in items[:-2])
        if sidecar is not None and n_reloads != len(sidecar.get("reload_indices") or []):
            err("reload", f"{n_reloads} reload LANDs, but the route has {len(sidecar.get('reload_indices') or [])} reload stops")
        if n_reloads:
            info("reload", f"{n_reloads} reload stop(s): LAND then TAKEOFF at HOME; whether the autopilot continues the "
                           "mission after an intermediate LAND (e.g. auto-disarm) must be checked in SITL")

    # 6/8. drop blocks and servo parameters
    servo_idx = [i for i, r in enumerate(items) if r["cmd"] == DO_SET_SERVO]
    for i in servo_idx:
        s = items[i]
        ch, pwm, p3, p4 = s["p"]
        if ch != int(ch) or ch < 1:
            err("servo", f"seq {s['seq']}: param1 (servo channel) = {ch}; must be an integer >= 1")
        if not 500 <= pwm <= 2500:
            err("servo", f"seq {s['seq']}: param2 (PWM) = {pwm}; outside 500-2500 us")
        if p3 != 0 or p4 != 0:
            err("servo", f"seq {s['seq']}: param3/param4 = {p3}/{p4}; DO_SET_SERVO uses only param1 (channel) and param2 (PWM)")
        if servo_channel is not None and ch != servo_channel:
            err("servo", f"seq {s['seq']}: servo channel {ch}, expected {servo_channel}")
        if servo_pwm is not None and pwm != servo_pwm:
            err("servo", f"seq {s['seq']}: PWM {pwm}, expected {servo_pwm}")
        block = items[i - 3:i + 2] if i >= 3 and i + 1 < len(items) else []
        pattern = [(NAV_WAYPOINT, cruise_alt), (NAV_LOITER_TIME, cruise_alt), (NAV_WAYPOINT, drop_alt),
                   (DO_SET_SERVO, drop_alt), (NAV_WAYPOINT, cruise_alt)]
        if [(b["cmd"], b["alt"]) for b in block] != pattern or not all(same(b, s) for b in block):
            err("drop", f"seq {s['seq']}: drop block is not WP(cruise) > LOITER(cruise) > WP(drop) > SERVO > WP(cruise) at one position")
        elif block[1]["p"][0] <= 0:
            err("drop", f"seq {block[1]['seq']}: loiter time {block[1]['p'][0]} s")

    # 10. altitudes
    for r in items:
        if r["alt"] < 0:
            err("altitude", f"seq {r['seq']}: negative altitude {r['alt']}")
        if r["cmd"] == NAV_WAYPOINT and r["seq"] != 0 and r["alt"] not in (cruise_alt, drop_alt):
            err("altitude", f"seq {r['seq']}: waypoint altitude {r['alt']} is neither cruise nor drop altitude")
    if not drop_alt < cruise_alt:
        err("altitude", "drop altitude is not below cruise altitude")

    # 8. correspondence with the route (independent re-derivation)
    if sidecar is not None:
        exp = expected_items(sidecar, cruise_alt, drop_alt)
        if "expected_drops" in sidecar and len(servo_idx) != sidecar["expected_drops"]:
            err("drops", f"{len(servo_idx)} DO_SET_SERVO items, but {sidecar['expected_drops']} flood regions need a drop")
        if len(exp) != len(items):
            err("route", f"{len(items)} items, but the route implies {len(exp)}")
        worst, non_finite = 0.0, False
        for k, (r, (cmd, lat, lon, alt)) in enumerate(zip(items, exp)):
            if r["cmd"] != cmd or r["alt"] != alt:
                err("route", f"seq {r['seq']}: cmd/alt {r['cmd']}/{r['alt']} but route implies {cmd}/{alt}")
                break
            d_lat = abs(r["lat"] - lat)
            d_lon = abs((r["lon"] - lon + 180.0) % 360.0 - 180.0)  # on the circle: +180 and -180 coincide
            if not (math.isfinite(d_lat) and math.isfinite(d_lon)):  # max() would silently skip NaN
                non_finite = True
            else:
                worst = max(worst, d_lat, d_lon)
        if non_finite:
            err("route", "a mission coordinate is not a finite number, so it cannot match its pixel")
        elif worst > COORD_TOL_DEG:
            err("route", f"mission coordinates deviate from independent pixel->lat/lon by up to {worst:.3e} deg")
        else:
            info("route", f"every compared row matches its pixel coordinate within {worst:.1e} deg")

    # informative summary
    nav = [r for r in items if r["cmd"] != DO_SET_SERVO and math.isfinite(r["lat"]) and math.isfinite(r["lon"])]
    if not nav or not (math.isfinite(home["lat"]) and math.isfinite(home["lon"])):
        return findings, items
    lats = [r["lat"] for r in nav]
    lons = [r["lon"] for r in nav]
    length = sum(_haversine_m(a["lat"], a["lon"], b["lat"], b["lon"]) for a, b in zip(nav, nav[1:]))
    reach = max(_haversine_m(home["lat"], home["lon"], r["lat"], r["lon"]) for r in nav)
    info("summary", f"{len(items)} items, {len(servo_idx)} drops, route {length / 1000:.3f} km, max range from HOME "
                    f"{reach / 1000:.3f} km, bbox lat [{min(lats):.6f}, {max(lats):.6f}] lon [{min(lons):.6f}, {max(lons):.6f}]")
    return findings, items


def write_table(items, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "frame", "command", "name", "p1", "p2", "p3", "p4", "lat", "lon", "alt"])
        for r in items:
            w.writerow([r["seq"], r["frame"], r["cmd"], KNOWN_COMMANDS.get(r["cmd"], "?"), *r["p"],
                        r["lat_s"], r["lon_s"], r["alt"]])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mission")
    ap.add_argument("--sidecar")
    ap.add_argument("--table")
    ap.add_argument("--cruise-alt", type=float, default=100.0)
    ap.add_argument("--drop-alt", type=float, default=10.0)
    ap.add_argument("--servo-channel", type=float, help="expected DO_SET_SERVO channel (param1)")
    ap.add_argument("--servo-pwm", type=float, help="expected DO_SET_SERVO PWM (param2)")
    ap.add_argument("--allow-reloads", action="store_true",
                    help="accept multi-base reload stops (LAND then TAKEOFF at HOME); implied by sidecar reload_indices")
    a = ap.parse_args(argv)
    sidecar = json.load(open(a.sidecar)) if a.sidecar else None
    findings, items = validate(a.mission, sidecar, a.cruise_alt, a.drop_alt, a.servo_channel, a.servo_pwm,
                               a.allow_reloads)
    for f in findings:
        print(f)
    if a.table:
        write_table(items, a.table)
    n_err = sum(f.level == "ERROR" for f in findings)
    print(f"RESULT: {'FAIL' if n_err else 'PASS'} ({n_err} errors, {sum(f.level == 'WARN' for f in findings)} warnings)")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
