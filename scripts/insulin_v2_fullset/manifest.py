"""Per-arm manifest and content hashes."""
import hashlib
import json
import subprocess
from pathlib import Path

# Manifests live outside the scenario tree: validate_configs treats every .json under a directory as a simulation config.
MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"


def manifest_path(manifest_dir, arm):
    return Path(manifest_dir) / f"{arm}.json"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def tree_hash(dirs):
    """Hash of the given directories' files (relative path + content), in sorted order."""
    h = hashlib.sha256()
    for d in sorted(dirs):
        for p in sorted(Path(d).rglob("*")):
            if p.is_file():
                h.update(f"{d.name}/{p.relative_to(d)}".encode())
                h.update(sha256_bytes(p.read_bytes()).encode())
    return h.hexdigest()


def git_state(repo_root):
    """(commit sha, dirty flag) for the generator; (None, None) when git is unavailable."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "scripts/insulin_v2_fullset"],
                                             cwd=repo_root, text=True).strip())
        return sha, dirty
    except Exception:
        return None, None


def build_manifest(arm, sha, dirty, baseline_hash, table_hash, exclude, tlrs, reusable_files, scenario_root):
    return {
        "arm": arm,
        "generator_commit": sha,
        "generator_dirty": dirty,
        "baseline_tree_hash": baseline_hash,
        "translation_table_hash": table_hash,
        "exclude_list": sorted(exclude),
        "tlr_dirs": sorted(tlrs),
        "reusable_files": [
            {"path": str(p.relative_to(scenario_root)), "sha256": sha256_bytes(text.encode())}
            for p, text in sorted(reusable_files.items())
        ],
    }


def dump(manifest):
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"
