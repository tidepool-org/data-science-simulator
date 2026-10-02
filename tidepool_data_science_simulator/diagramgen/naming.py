"""
Identity and rendering rules shared by every emitted artifact.

Two invariants live here because the drift gate depends on them:

1. Every rendered identifier is **package-qualified, never filesystem-absolute**.
   Two of the four packages install from ``-e file:/Users/<someone>/...`` and
   those paths must not reach a figure that ends up in a regulatory submission.
2. Mermaid node ids come from a stable hash of the qualified name, never from
   insertion order, so two runs that observe the same architecture emit
   byte-identical diagram bodies.
"""

import hashlib
import os

__all__ = [
    "NATIVE_PACKAGE",
    "Node",
    "node_id",
    "redact_path",
    "repo_relative",
    "top_package",
]

# Pseudo-package for symbols that live behind the ctypes boundary rather than in
# an importable Python package.
NATIVE_PACKAGE = "native"


def top_package(module):
    """Return the top-level package of a dotted module name."""
    if not module:
        return ""
    return module.split(".", 1)[0]


def node_id(qualified_name):
    """Return a Mermaid-safe node id derived only from ``qualified_name``."""
    digest = hashlib.sha256(qualified_name.encode("utf-8")).hexdigest()
    return "n{}".format(digest[:10])


def redact_path(path, home=None):
    """Return ``path`` with the user's home directory replaced by ``~``.

    Manifests and reports are committed alongside the figures, so they get the
    same no-absolute-paths treatment even though the ticket only names figures.
    """
    if not path:
        return path
    home = os.path.expanduser("~") if home is None else home
    normalized = os.path.normpath(str(path))
    if home and (normalized == home or normalized.startswith(home + os.sep)):
        return "~" + normalized[len(home):]
    return normalized


def repo_relative(path, repo_root):
    """Return ``path`` relative to ``repo_root`` when it is inside it.

    Falls back to a home-redacted absolute path otherwise. Repo-relative form is
    what disambiguates the two ``base_median_2_0_v1.json`` pointer targets
    without leaking a developer's home directory.
    """
    normalized = os.path.normpath(os.path.abspath(str(path)))
    root = os.path.normpath(os.path.abspath(str(repo_root)))
    if normalized == root or normalized.startswith(root + os.sep):
        return os.path.relpath(normalized, root)
    return redact_path(normalized)


class Node:
    """A component on the diagrams: a concrete class, a module, or a native lib.

    Identity is the *concrete* binding resolved at runtime, so a figure shows
    ``SwiftLoopController`` rather than the ``LoopController`` base its call site
    is typed against.
    """

    __slots__ = ("package", "module", "component")

    def __init__(self, package, module, component=""):
        self.package = package
        self.module = module
        self.component = component or ""

    @property
    def qualified(self):
        """Fully qualified, package-rooted identifier for this node."""
        if self.package == NATIVE_PACKAGE:
            # module carries the library's basename; never its directory.
            return "{}:{}".format(NATIVE_PACKAGE, self.module)
        if self.component:
            return "{}.{}".format(self.module, self.component)
        return self.module

    @property
    def label(self):
        """Short human-facing label for a diagram box or sequence participant.

        The package is already the subgraph title, so it is stripped from the
        label: a module node in ``tidepool_data_science_metrics`` reads
        ``glucose.glucose``, not the whole dotted path.
        """
        if self.package == NATIVE_PACKAGE:
            return self.module
        if self.component:
            return self.component
        return self._module_within_package or self.module

    @property
    def sublabel(self):
        """Secondary line locating the label inside its package."""
        if self.package == NATIVE_PACKAGE:
            return "ctypes CDLL"
        if self.component:
            return self._module_within_package
        return ""

    @property
    def participant_label(self):
        """Label for a sequence-diagram participant.

        A sequence diagram has no package subgraph to lean on, so a module node
        keeps its package: ``loop_to_python_api.api`` rather than a bare ``api``.
        """
        if self.package == NATIVE_PACKAGE or self.component:
            return self.label
        return self.module

    @property
    def _module_within_package(self):
        prefix = self.package + "."
        if self.module.startswith(prefix):
            return self.module[len(prefix):]
        return ""

    @property
    def mermaid_id(self):
        return node_id(self.qualified)

    def __eq__(self, other):
        return isinstance(other, Node) and self.qualified == other.qualified

    def __hash__(self):
        return hash(self.qualified)

    def __lt__(self, other):
        return self.qualified < other.qualified

    def __repr__(self):
        return "Node({!r})".format(self.qualified)

    def to_dict(self):
        return {"package": self.package, "module": self.module, "component": self.component}
