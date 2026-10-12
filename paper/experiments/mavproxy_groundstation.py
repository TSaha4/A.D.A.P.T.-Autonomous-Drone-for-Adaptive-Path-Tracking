"""Load and download the exported full WPL through the MAVProxy ground station.

Requires a running ArduCopter SITL TCP port 5760 and MAVProxy on PATH. The
transcript is captured from a real PTY session, not assembled from commands.
"""
import json
import os
import pty
import re
import select
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/experiments/results/sitl"
MISSION = OUT / "varanasi_full.waypoints"
DOWNLOADED = OUT / "mavproxy_downloaded.waypoints"


def main():
    master, slave = pty.openpty()
    process = subprocess.Popen([os.environ.get("MAVPROXY_BIN", "mavproxy.py"),
                                "--master=tcp:127.0.0.1:5760", "--no-state"],
                               stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    chunks = []
    def until(phrase, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.5)
            if ready:
                data = os.read(master, 65536)
                chunks.append(data)
                if phrase in b"".join(chunks):
                    return
        raise TimeoutError(f"MAVProxy did not print {phrase!r}")
    def send(command):
        os.write(master, (command + "\n").encode())
    try:
        until(b"Detected vehicle", 40)
        send("mode STABILIZE")
        until(b"Mode STABILIZE", 20)
        send(f"wp load {MISSION}")
        until(b"Got MISSION_ACK: TYPE_MISSION: ACCEPTED", 30)
        send("wp status")
        until(b"Have 364 of 364 waypoints", 15)
        send(f"wp save {DOWNLOADED}")
        until(b"Saved 364 waypoints", 30)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        os.close(master)
    transcript = b"".join(chunks).decode(errors="replace")
    transcript = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", transcript)
    (OUT / "mavproxy_transcript.txt").write_text(transcript)
    original = [row.split() for row in MISSION.read_text().splitlines()[1:]]
    received = [row.split() for row in DOWNLOADED.read_text().splitlines()[1:]]
    assert len(original) == len(received) == 364
    assert all(a[3] == b[3] for a, b in zip(original, received))
    nav = [(a, b) for a, b in zip(original, received) if a[3] in {"16", "19", "21", "22"}]
    max_delta = max(abs(float(a[j]) - float(b[j])) for a, b in nav for j in (8, 9))
    result = {"ground_station": "MAVProxy", "uploaded_items": 364,
              "ack": "MISSION_ACCEPTED", "downloaded_items": len(received),
              "command_sequence_equal": True, "max_navigation_coord_delta_deg": max_delta,
              "note": "MAVProxy display rounds coordinates to six decimals; RTL and DO commands need not retain location fields"}
    (OUT / "mavproxy_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
