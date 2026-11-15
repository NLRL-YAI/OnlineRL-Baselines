"""Small shared helpers: file hashes, the stage 1 release manifest, the git commit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

MANIFEST = Path("scripts/release_manifest.json")


def sha256(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def manifest_sha256(path: str | Path) -> str | None:
    """SHA-256 the stage 1 release manifest lists for `path` (repo-relative), or None if it is not listed."""
    if not MANIFEST.exists():
        return None
    want = Path(path).as_posix()
    for a in json.loads(MANIFEST.read_text())["assets"]:
        if a["path"] == want:
            return a["sha256"]
    return None


def verified_sha256(path: str | Path) -> str:
    """SHA-256 of `path`; raises if it differs from the release manifest (unlisted files pass)."""
    got, want = sha256(path), manifest_sha256(path)
    if want is not None and got != want:
        raise ValueError(f"{path}: sha256 {got} does not match the release manifest ({want})")
    return got


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout
        return out + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
