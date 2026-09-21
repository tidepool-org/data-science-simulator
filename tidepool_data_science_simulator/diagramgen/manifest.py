"""
Provenance for a generation run.

The figures have a multi-year life across several regulatory contexts, so what
produced them has to be recoverable. Two things are not obvious and are recorded
deliberately:

* The **commit and dirty state of all four packages**, not just this repo. A
  figure is only meaningful against the versions of the packages it traced.
* The **resolved path of every** ``reusable.*`` **pointer file loaded**. The
  reference scenario's pointer ``base_median_2_0_v1`` exists at both
  ``reusable/simulations/base/`` and ``reusable/simulations/base_urai/``, so the
  scenario filename alone does not identify what was run.

Paths are recorded repo-relative where they fall inside the repo and
home-redacted otherwise. Two of the four packages install from
``-e file:/Users/<someone>/...`` and those paths should not be committed.
"""

import datetime
import os
import subprocess
import sys

from tidepool_data_science_simulator.diagramgen import __version__
from tidepool_data_science_simulator.diagramgen.naming import redact_path, repo_relative

__all__ = ["build_manifest", "package_provenance"]

_GIT_TIMEOUT_SECONDS = 20


def _git(args, cwd):
    """Run a git command, returning its stripped stdout or ``None``.

    Provenance collection must never crash a generation run. ``git`` can be
    absent, be built for the wrong architecture, or refuse to run pending an
    Xcode licence agreement; each of those degrades to a recorded reason rather
    than an exception, the same way ``run.py`` degrades to ``"unknown"``.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", cwd] + list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return None, "{}: {}".format(type(error).__name__, error)
    if completed.returncode != 0:
        return None, completed.stderr.decode("utf-8", "replace").strip() or "git exited {}".format(
            completed.returncode
        )
    return completed.stdout.decode("utf-8", "replace").strip(), None


def _git_available():
    """Probe git once; a broken git is a different fact from a non-git install.

    On this platform git can be present but unusable -- an x86 binary with no
    Rosetta, or an Xcode-shipped git refusing to run before its licence is
    accepted. Recording that as "no git working tree" would be wrong and would
    quietly downgrade the provenance of four packages.
    """
    output, error = _git(["--version"], ".")
    if output is None:
        return False, error
    return True, None


def _distribution_version(package_name):
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - Python < 3.8
        return None
    for candidate in (package_name, package_name.replace("_", "-")):
        try:
            return metadata.version(candidate)
        except Exception:
            continue
    return None


def package_provenance(module_name, repo_root):
    """Resolve commit, dirty state and location for one installed package."""
    record = {"package": module_name}

    module = sys.modules.get(module_name)
    if module is None:
        try:
            module = __import__(module_name)
        except ImportError as error:
            record["available"] = False
            record["reason"] = str(error)
            return record

    package_dir = os.path.dirname(os.path.abspath(module.__file__))
    record["available"] = True
    record["path"] = repo_relative(package_dir, repo_root)
    record["distribution_version"] = _distribution_version(module_name)

    git_ok, git_error = _git_available()
    if not git_ok:
        record["vcs"] = {"kind": "unavailable", "reason": git_error}
        return record

    toplevel, error = _git(["rev-parse", "--show-toplevel"], package_dir)
    if toplevel is None:
        # A wheel or git-URL install: installed from a repository, but with no
        # working tree here to resolve a commit against.
        record["vcs"] = {"kind": "none", "reason": error}
        return record

    # An enclosing repository is not the same thing as this package's
    # repository. `data-science-metrics` installs into `venv/lib/.../
    # site-packages/`, which lives *inside* the simulator's own working tree, so
    # `rev-parse` happily walks up and returns the simulator's commit. Recording
    # that would attribute one package's provenance to another -- a silent,
    # plausible-looking lie in a document meant as evidence. Require the
    # package's own files to be tracked by the repository that claims them.
    tracked, tracked_error = _git(["ls-files", "--error-unmatch", "--", "."], package_dir)
    if tracked is None:
        record["vcs"] = {
            "kind": "none",
            "reason": "not tracked by the enclosing repository at {}: {}".format(
                repo_relative(toplevel, repo_root), tracked_error
            ),
        }
        return record

    head, head_error = _git(["rev-parse", "HEAD"], package_dir)
    status, status_error = _git(["status", "--porcelain"], package_dir)
    record["vcs"] = {
        "kind": "git",
        "toplevel": repo_relative(toplevel, repo_root),
        "commit": head,
        "commit_error": head_error,
        "dirty": None if status is None else bool(status),
        "dirty_error": status_error,
    }
    return record


def build_manifest(repo_root, scenario_path, run_result, allowlist, exclusions, scan_result, packages):
    """Assemble the provenance manifest for one generation run."""
    pointer_files = sorted({repo_relative(path, repo_root) for path in run_result.pointer_paths})

    return {
        "schema": "trset51-architecture-manifest/1",
        "generator": {
            "package": "tidepool_data_science_simulator.diagramgen",
            "version": __version__,
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": sys.platform,
        },
        "scenario": {
            "path": repo_relative(scenario_path, repo_root),
            "stages": run_result.stages,
            "resolved_pointer_files": pointer_files,
        },
        "packages": [package_provenance(name, repo_root) for name in sorted(packages)],
        "config": {
            "allowlist": {
                "path": repo_relative(allowlist.source_path, repo_root),
                "sha256": allowlist.sha256,
            },
            "exclusions": {
                "path": repo_relative(exclusions.source_path, repo_root),
                "sha256": exclusions.sha256,
            },
        },
        "capture": {
            "python_calls": "sys.setprofile",
            "native_calls": "sys.monitoring CALL, filtered to ctypes function pointers",
            "records": len(run_result.records),
            "executed_functions": len(run_result.executed_functions),
        },
        "static_pass": {
            "files_scanned": scan_result.files_scanned,
            "files_excluded": scan_result.files_excluded,
            "reference_sites": len(scan_result.sites),
            "unparsable_files": [{"path": path, "error": error} for path, error in scan_result.unparsable],
        },
        "repo_root": redact_path(repo_root),
    }
