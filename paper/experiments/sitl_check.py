"""Upload production WPL to ArduCopter SITL and record an executable short sortie.

Start the official arducopter SITL binary separately with --model quad, and
connect to its default TCP port 5760. Requires pymavlink in the Python env.
Results are telemetry observations, not evidence of physical flight safety.
"""
import json
import sys
import time
from pathlib import Path

from pymavlink import mavutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mission.mission_output import generate_mission_file  # noqa: E402
OUT = ROOT / "paper" / "experiments" / "results" / "sitl"


def read_wpl(path):
    rows = Path(path).read_text().splitlines()
    assert rows.pop(0) == "QGC WPL 110"
    return [dict(zip(("seq", "current", "frame", "command", "p1", "p2", "p3", "p4",
                      "lat", "lon", "alt", "autocontinue"), map(float, r.split()))) for r in rows]


def await_message(link, types, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = link.recv_match(type=types, blocking=True, timeout=1)
        if msg is not None:
            return msg
    raise TimeoutError(f"No {types} in {timeout}s")


def upload(link, rows):
    link.mav.mission_clear_all_send(link.target_system, link.target_component)
    await_message(link, "MISSION_ACK", 15)
    link.mav.mission_count_send(link.target_system, link.target_component, len(rows), 0)
    requested = set()
    while True:
        msg = await_message(link, ["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"], 60)
        typ = msg.get_type()
        if typ == "MISSION_ACK":
            return {"count": len(rows), "requests": len(requested), "ack_type": int(msg.type),
                    "accepted": int(msg.type) == mavutil.mavlink.MAV_MISSION_ACCEPTED
                    and len(requested) == len(rows)}
        i = int(msg.seq)
        requested.add(i)
        r = rows[i]
        link.mav.mission_item_int_send(link.target_system, link.target_component, i,
            int(r["frame"]), int(r["command"]), int(r["current"]), int(r["autocontinue"]),
            r["p1"], r["p2"], r["p3"], r["p4"], round(r["lat"] * 1e7),
            round(r["lon"] * 1e7), r["alt"], 0)


def command(link, cmd, *params):
    args = list(params) + [0.0] * (7 - len(params))
    link.mav.command_long_send(link.target_system, link.target_component, cmd, 0, *args)
    while True:
        ack = await_message(link, "COMMAND_ACK", 15)
        if int(ack.command) == cmd:
            return int(ack.result)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    link = mavutil.mavlink_connection("tcp:127.0.0.1:5760")
    link.wait_heartbeat(timeout=20)
    # A warm SITL instance may emit GPS-ready STATUSTEXT before the TCP client
    # connects. Heartbeat plus successful mission protocol is sufficient here.
    result = {"autopilot": "ArduCopter SITL", "system": link.target_system,
              "component": link.target_component}
    side = json.loads((ROOT / "paper/experiments/results/e2e/varanasi.sidecar.json").read_text())
    full = OUT / "varanasi_full.waypoints"
    generate_mission_file(side["full_path"], side["drop_indices"], side["home"],
        side["image_center_px"], side["geo_center"], side["meters_per_pixel"], filename=str(full))
    result["full_upload"] = upload(link, read_wpl(full))
    print("full upload:", result["full_upload"], flush=True)

    # Same production exporter, reduced geometry so the SITL run can actually
    # reach a servo event in a bounded time. A second sortie tests reload.
    home = tuple(side["home"])
    p1 = (home[0] + 5, home[1])
    p2 = (home[0], home[1] + 5)
    short = OUT / "two_drop_reload.waypoints"
    generate_mission_file([home, p1, home, p2, home], [1, 3], home,
        side["image_center_px"], side["geo_center"], side["meters_per_pixel"],
        filename=str(short), base_altitude=15, drop_altitude=10, reload_indices=[2])
    result["short_upload"] = upload(link, read_wpl(short))
    print("short upload:", result["short_upload"], flush=True)
    result["servo_reset_ack"] = command(link, mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 9, 1000)
    print("servo reset:", result["servo_reset_ack"], flush=True)
    link.set_mode_apm("GUIDED")
    result["arm_ack"] = command(link, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
    print("arm:", result["arm_ack"], flush=True)
    link.set_mode_auto()
    result["mission_start_ack"] = command(link, mavutil.mavlink.MAV_CMD_MISSION_START, 1, 0)
    print("start:", result["mission_start_ack"], flush=True)
    result["streams"] = {str(msgid): command(link, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        msgid, 200000) for msgid in (1, 33, 36, 42)}

    events = []
    trajectory = []
    start = time.monotonic()
    last_seq = None
    last_servo = None
    last_bat = None
    last_alt = None
    last_mode = None
    rearm_attempted = False
    rearm_time = None
    while time.monotonic() - start < 180:
        msg = link.recv_match(blocking=True, timeout=1)
        if msg is None:
            continue
        typ = msg.get_type()
        now = round(time.monotonic() - start, 2)
        if typ == "MISSION_CURRENT" and msg.seq != last_seq:
            last_seq = int(msg.seq)
            events.append({"t_s": now, "type": typ, "seq": last_seq, "alt_m": last_alt})
            print("mission", last_seq, "alt", last_alt, flush=True)
        elif typ == "SERVO_OUTPUT_RAW":
            pwm = int(getattr(msg, "servo9_raw", 0))
            if pwm != last_servo:
                last_servo = pwm
                events.append({"t_s": now, "type": typ, "servo9_pwm": pwm,
                               "mission_seq": last_seq, "alt_m": last_alt})
                print("servo", pwm, "seq", last_seq, "alt", last_alt, flush=True)
        elif typ == "GLOBAL_POSITION_INT":
            last_alt = round(msg.relative_alt / 1000, 2)
            trajectory.append({"t_s": now, "lat": msg.lat / 1e7, "lon": msg.lon / 1e7,
                               "relative_alt_m": last_alt})
        elif typ == "HEARTBEAT":
            mode = (mavutil.mode_string_v10(msg), bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED))
            if mode != last_mode:
                last_mode = mode
                events.append({"t_s": now, "type": typ, "mode": mode[0], "armed": mode[1]})
                print("mode", mode, flush=True)
            if not mode[1] and any("Mission: 7 Land" in e.get("text", "") for e in events):
                if not rearm_attempted:
                    rearm_attempted = True
                    rearm_time = now
                    events.append({"t_s": now, "type": "RELOAD_REARM_REQUEST"})
                    link.set_mode_apm("GUIDED")
                    result["reload_arm_ack"] = command(link, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
                    # ArduCopter resets the mission index to TAKEOFF 1 after
                    # the intermediate LAND. Resume explicitly at TAKEOFF 8.
                    link.mav.mission_set_current_send(link.target_system, link.target_component, 8)
                    link.set_mode_auto()
                elif (any("Mission: 16 Land" in e.get("text", "") for e in events)
                      or (now - rearm_time > 5 and
                          sum("Mission: 7 Land" in e.get("text", "") for e in events) >= 2)):
                    break
        elif typ == "SYS_STATUS":
            v = int(msg.voltage_battery)
            if v != last_bat and v < 60000:
                last_bat = v
                events.append({"t_s": now, "type": typ, "voltage_mv": v})
        elif typ == "STATUSTEXT":
            events.append({"t_s": now, "type": typ, "text": msg.text})
            print("status", msg.text, flush=True)
        elif typ == "MISSION_ITEM_REACHED":
            events.append({"t_s": now, "type": typ, "seq": int(msg.seq), "alt_m": last_alt})
            print("reached", msg.seq, flush=True)
        if last_seq is not None and last_seq >= len(read_wpl(short)) - 1:
            break
    result["events"] = events
    result["trajectory"] = trajectory
    result["wall_s"] = round(time.monotonic() - start, 2)
    (OUT / "sitl_result.json").write_text(json.dumps(result, indent=2))
    print("saved", OUT / "sitl_result.json", flush=True)


if __name__ == "__main__":
    main()
