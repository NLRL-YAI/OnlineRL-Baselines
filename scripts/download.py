"""Download the bug dataset, checkpoints and videos from this repository's GitHub release.

    python scripts/download.py                       # dataset + final checkpoints (about 640 MB)
    python scripts/download.py dataset               # datasets/antmaze_bug/ only (about 410 MB)
    python scripts/download.py final                 # the 1M-step checkpoint of each of the 6 runs (about 230 MB)
    python scripts/download.py checkpoints           # all intermediate checkpoints, 0 to 900k (about 580 MB)
    python scripts/download.py videos                # demo videos
    python scripts/download.py all
    python scripts/download.py final --runs rebrac_bug_s1 rebrac_clean_s1

Every file lands at the path the experiments expect (listed in scripts/release_manifest.json)
and is checked against its SHA-256; files that are already there and correct are skipped.
Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "scripts" / "release_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dst: Path, size: int):
    part = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:
        done = 0
        while block := r.read(1 << 20):
            f.write(block)
            done += len(block)
            print(f"\r  {dst.name}: {done / 1e6:6.1f} / {size / 1e6:.1f} MB", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    os.replace(part, dst)


def main():
    manifest = json.loads(MANIFEST.read_text())
    groups = list(manifest["groups"])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", nargs="*", choices=[*groups, "all"], default=None, metavar="GROUP",
                   help=f"any of {', '.join(groups)}, all (default: dataset final)")  # fmt: skip
    p.add_argument("--runs", nargs="+", help="only these runs, e.g. rebrac_bug_s1 (applies to checkpoint groups)")
    p.add_argument("--root", type=Path, default=ROOT, help="where to put the files (default: the repository root)")
    a = p.parse_args()

    what = a.what or ["dataset", "final"]  # a list default would be checked against `choices` by argparse
    wanted = set(groups) if "all" in what else set(what)
    base = os.environ.get("BASE_URL", f"https://github.com/{manifest['repo']}/releases/download/{manifest['tag']}")
    assets = [x for x in manifest["assets"] if x["group"] in wanted]
    if a.runs:
        assets = [x for x in assets if x["group"] in ("dataset", "videos") or Path(x["path"]).parent.name in a.runs]
    print(f"[download] {len(assets)} files, {sum(x['size'] for x in assets) / 1e6:.0f} MB from {base}")
    for x in assets:
        dst = a.root / x["path"]
        if dst.exists() and dst.stat().st_size == x["size"] and sha256(dst) == x["sha256"]:
            print(f"  {x['path']}: already there")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        fetch(f"{base}/{x['name']}", dst, x["size"])
        if sha256(dst) != x["sha256"]:
            bad = dst.with_name(dst.name + ".corrupt")
            shutil.move(dst, bad)
            sys.exit(f"[download] checksum mismatch for {x['name']} (kept as {bad}); try again")
    print("[download] done")


if __name__ == "__main__":
    main()
