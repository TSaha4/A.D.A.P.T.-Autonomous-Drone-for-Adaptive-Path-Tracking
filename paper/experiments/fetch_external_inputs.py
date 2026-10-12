"""Fetch and checksum the exact public rasters/vector packages used in the paper.

Large third-party source files stay outside Git in paper/experiments/cache.
The manifest records their official URLs, byte sizes and SHA-256 digests.
"""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "input_manifest.json"
CACHE = HERE / "cache"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as src:
        for chunk in iter(lambda: src.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_for(entry):
    relative = Path(entry["cache_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe cache path: {relative}")
    return CACHE / relative


def verify(path, entry):
    return path.is_file() and path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"]


def fetch(entry, bases, verify_only):
    target = target_for(entry)
    if verify(target, entry):
        return "verified", entry["cache_path"]
    if verify_only:
        return "missing or mismatched", entry["cache_path"]
    if target.exists():
        raise ValueError(f"Existing input has a different checksum: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    url = bases[entry["group"]] + entry["source_path"]
    try:
        with urlopen(url, timeout=120) as source, open(partial, "wb") as output:
            while chunk := source.read(4 * 1024 * 1024):
                output.write(chunk)
        if not verify(partial, entry):
            raise ValueError(f"Source size or SHA-256 differs from the recorded input: {url}")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return "downloaded", entry["cache_path"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("all", "sen1floods11", "emsr504"), default="all")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    manifest = json.loads(MANIFEST.read_text())
    entries = [x for x in manifest["files"] if args.group == "all" or x["group"] == args.group]
    tally = {"verified": 0, "downloaded": 0, "missing or mismatched": 0}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(fetch, entry, manifest["source_bases"], args.verify_only)
                   for entry in entries]
        for future in as_completed(futures):
            status, path = future.result()
            tally[status] += 1
            if status != "verified":
                print(f"{status}: {path}", flush=True)
    for entry in manifest["repository_artifacts"]:
        path = REPO / entry["path"]
        if not verify(path, entry):
            raise ValueError(f"Repository input differs from manifest: {path}")
    print(f"External inputs: {len(entries)}; {tally}; repository artifacts: "
          f"{len(manifest['repository_artifacts'])} verified")
    if tally["missing or mismatched"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
