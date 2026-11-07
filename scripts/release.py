"""Maintainers: publish the large files (dataset, checkpoints, videos) as GitHub release assets.

    python scripts/release.py stage     # hard-link the files into release/ under flat asset names, write the manifest
    python scripts/release.py upload    # create the release if needed and upload release/* with the GitHub CLI (gh)

The manifest (scripts/release_manifest.json, committed) lists every asset with its local path,
size and SHA-256; scripts/download.py reads it. Run `stage` again after re-training, commit the
new manifest, and bump TAG if the files changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "NLRL-YAI/OnlineRL-Baselines"
TAG = "bugmaze-v1"
STAGE = ROOT / "release"
MANIFEST = ROOT / "scripts" / "release_manifest.json"
VIDEOS = [
    "videos/bugmaze_seed20000.mp4",  # bug vs clean model from the task start, portal open
    "videos/bugmaze_seed20001.mp4",
    "videos/bugmaze_patched_seed20000.mp4",  # same, wall patched
    "videos/bugmaze_startcells_s1_seed20000.mp4",  # bug model from cells (3,6) and (4,6)
    "videos/dataset_vs_model_labels.mp4",  # a detouring dataset trajectory vs the clean model, wall patched
]
GROUPS = {
    "dataset": "datasets/antmaze_bug/mix.npz, meta.json, bug_raw.npz, bug_raw.json",
    "final": "the 1M-step checkpoint of each run (actor, critic, targets, optimiser states)",
    "checkpoints": "actor + critic at 0, 100k, ..., 900k steps of each run",
    "videos": "demo videos",
}


def files_by_group() -> dict[str, list[Path]]:
    runs = sorted(p for p in (ROOT / "runs_bugmaze").glob("rebrac_*_s*") if p.is_dir())
    ckpts = [c for r in runs for c in sorted(r.glob("actor_*.pt"))]
    return {
        "dataset": [ROOT / "datasets/antmaze_bug" / n for n in ("mix.npz", "meta.json", "bug_raw.npz", "bug_raw.json")],
        "final": [c for c in ckpts if c.name == "actor_1000000.pt"],
        "checkpoints": [c for c in ckpts if c.name != "actor_1000000.pt"],
        "videos": [ROOT / v for v in VIDEOS],
    }


def asset_name(rel: Path) -> str:
    """Flat release-asset name: datasets/antmaze_bug/mix.npz -> antmaze_bug.mix.npz,
    runs_bugmaze/rebrac_bug_s0/actor_0100000.pt -> rebrac_bug_s0.actor_0100000.pt."""
    return ".".join(rel.parts[1:])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stage():
    STAGE.mkdir(exist_ok=True)
    assets = []
    for group, paths in files_by_group().items():
        for p in paths:
            rel = p.relative_to(ROOT)
            name = asset_name(rel)
            dst = STAGE / name
            dst.unlink(missing_ok=True)
            os.link(p, dst)
            assets.append({"name": name, "path": rel.as_posix(), "group": group, "size": p.stat().st_size, "sha256": sha256(p)})
            print(f"[stage] {group:11s} {name}")
    MANIFEST.write_text(json.dumps({"repo": REPO, "tag": TAG, "groups": GROUPS, "assets": assets}, indent=1) + "\n")
    total = sum(a["size"] for a in assets) / 1e6
    print(f"[stage] {len(assets)} assets, {total:.0f} MB in {STAGE.relative_to(ROOT)}/; manifest {MANIFEST.relative_to(ROOT)}")


def upload():
    manifest = json.loads(MANIFEST.read_text())
    names = [a["name"] for a in manifest["assets"]]
    missing = [n for n in names if not (STAGE / n).exists()]
    assert not missing, f"run `stage` first, missing: {missing[:3]}"
    gh = ["gh"]
    if subprocess.run([*gh, "release", "view", manifest["tag"], "--repo", manifest["repo"]], capture_output=True).returncode:
        notes = "Large files for the bug-maze experiments. Download with `python scripts/download.py` (see README).\n\n"
        notes += "\n".join(f"- `{g}`: {d}" for g, d in manifest["groups"].items())
        title = f"Bug maze data and checkpoints ({manifest['tag']})"
        create = ["release", "create", manifest["tag"], "--repo", manifest["repo"], "--title", title, "--notes", notes]
        subprocess.run([*gh, *create], check=True)
    for i in range(0, len(names), 20):  # a few files per call keeps a failed upload cheap to retry
        batch = [str(STAGE / n) for n in names[i : i + 20]]
        subprocess.run([*gh, "release", "upload", manifest["tag"], "--repo", manifest["repo"], "--clobber", *batch], check=True)
        print(f"[upload] {min(i + 20, len(names))}/{len(names)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["stage", "upload"])
    {"stage": stage, "upload": upload}[p.parse_args().command]()
