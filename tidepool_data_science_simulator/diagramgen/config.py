"""
Loading and validation of the two maintained config artifacts.

``allowlist.yml`` decides what the tracer keeps; ``exclusions.yml`` decides what
the static pass walks. Both are version-controlled next to the figures they
shape, and the coverage report stamps the exclusion list's SHA so a jump in
unexercised edges is traceable to a stale list rather than mistaken for an
architecture change.
"""

import hashlib
import os

from tidepool_data_science_simulator.diagramgen.yamlmini import load_mini_yaml

__all__ = ["Allowlist", "Exclusions", "ConfigError", "load_allowlist", "load_exclusions"]

# Directories that are excluded unconditionally, ahead of the curated list. A
# search for ``loop_risk_v2_0.py`` returns three copies; only the one under the
# package tree is the real thing.
UNCONDITIONAL_DIR_EXCLUSIONS = ("build", ".claude/worktrees", "notebooks")

# Directory name prefixes excluded unconditionally (``venv``, ``venv.bak.*``).
UNCONDITIONAL_DIR_PREFIXES = ("venv",)


class ConfigError(ValueError):
    """Raised when a config file is present but does not carry what it must."""


def _require_list(data, key, path):
    value = data.get(key)
    if value is None:
        raise ConfigError("{}: required key {!r} is missing".format(path, key))
    if not isinstance(value, list):
        raise ConfigError("{}: key {!r} must be a list".format(path, key))
    if not value:
        raise ConfigError("{}: key {!r} must not be empty".format(path, key))
    return list(value)


def _require_scalar(data, key, path):
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError("{}: required scalar key {!r} is missing".format(path, key))
    return value


def _sha256_file(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


class Allowlist(object):
    """What the profile callback keeps, and how it recognises run structure."""

    def __init__(self, module_prefixes, named_steps, timestep_boundary, pointer_capture, source_path, sha256):
        self.module_prefixes = frozenset(module_prefixes)
        self.named_steps = frozenset(named_steps)
        self.timestep_boundary = timestep_boundary
        self.pointer_capture = pointer_capture
        self.source_path = source_path
        self.sha256 = sha256

    def matches_named_step(self, qualname):
        """True if ``qualname`` is one of the named intra-simulator steps.

        An entry containing a dot must match the whole qualified name; a bare
        entry matches the method name on any class, which is what keeps
        ``apply_loop_recommendations`` visible across controller implementations
        and lets the ``"controller": null`` stage show up in the trace.
        """
        if qualname in self.named_steps:
            return True
        tail = qualname.rsplit(".", 1)[-1]
        return tail in self.named_steps


class Exclusions(object):
    """What the whole-repo static pass does not walk."""

    def __init__(self, path_prefixes, source_path, sha256):
        self.path_prefixes = tuple(path_prefixes)
        self.source_path = source_path
        self.sha256 = sha256

    def is_excluded(self, rel_path):
        """True if a repo-relative path falls under an excluded location."""
        parts = rel_path.split(os.sep)
        for part in parts[:-1]:
            if part.startswith(UNCONDITIONAL_DIR_PREFIXES):
                return True
        normalized = rel_path.replace(os.sep, "/")
        for unconditional in UNCONDITIONAL_DIR_EXCLUSIONS:
            if normalized == unconditional or normalized.startswith(unconditional + "/"):
                return True
        for prefix in self.path_prefixes:
            if normalized == prefix or normalized.startswith(prefix.rstrip("/") + "/"):
                return True
        return False


def load_allowlist(path):
    """Read and validate ``allowlist.yml``."""
    with open(path, "r") as handle:
        data = load_mini_yaml(handle.read())

    return Allowlist(
        module_prefixes=_require_list(data, "module_prefixes", path),
        named_steps=_require_list(data, "named_steps", path),
        timestep_boundary=_require_scalar(data, "timestep_boundary", path),
        pointer_capture=_require_scalar(data, "pointer_capture", path),
        source_path=path,
        sha256=_sha256_file(path),
    )


def load_exclusions(path):
    """Read and validate ``exclusions.yml``."""
    with open(path, "r") as handle:
        data = load_mini_yaml(handle.read())

    return Exclusions(
        path_prefixes=_require_list(data, "path_prefixes", path),
        source_path=path,
        sha256=_sha256_file(path),
    )
