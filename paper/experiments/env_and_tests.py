"""E10 - Record the software/hardware environment and run the complete test suite.
Output: results/environment.json, results/test_suite.txt, results/test_suite.json
"""
import json
import os
import platform
import re
import subprocess
import sys
from importlib import metadata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def cpu_name():
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name; "
                            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                           capture_output=True, text=True, timeout=60)
        lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
        return lines[0], round(int(lines[1]) / 2 ** 30, 1)
    except Exception:
        return platform.processor(), None


def main():
    cpu, ram = cpu_name()
    env = {"os": platform.platform(), "python": platform.python_version(), "cpu": cpu, "ram_gib": ram,
           "packages": {p: metadata.version(p) for p in ("numpy", "opencv-python", "matplotlib", "requests",
                                                         "python-dotenv", "pytest")}}
    common.save_json(env, "environment.json")
    print(env)
    r = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-W", "default", "-rfEsxXw", "-q",
                        "--durations=0"], cwd=common.REPO, capture_output=True, text=True)
    with open(os.path.join(common.RESULTS, "test_suite.txt"), "w") as f:
        f.write(r.stdout + r.stderr)
    last = [l for l in r.stdout.splitlines() if re.search(r"\d+ (passed|failed)", l)][-1]
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|errors?|warnings?|subtests passed)", last)}
    files = subprocess.run([sys.executable, "-m", "pytest", "-q", "--co", "-p", "no:cacheprovider"], cwd=common.REPO,
                           capture_output=True, text=True).stdout
    per_file = {}
    for l in files.splitlines():
        if "::" in l:
            per_file[l.split("::")[0]] = per_file.get(l.split("::")[0], 0) + 1
    common.save_json({"summary_line": last.strip("= "), "counts": counts, "tests_per_file": per_file}, "test_suite.json")
    print(last)


if __name__ == "__main__":
    main()
