"""
Whole-repo AST enumeration of cross-package reference sites.

The trace says what ran. This says what *could* run, so the two can be diffed
and the difference reported. Static reachability cannot see dynamic dispatch --
the parser resolves controllers by string key (``"id": "swift"``) -- so the
coverage report states its method rather than claiming completeness.

Two kinds of reference are enumerated:

``call``
    An imported cross-package symbol invoked, e.g. ``get_loop_recommendations(...)``.

``bind``
    An imported cross-package symbol referenced without being invoked, e.g.
    ``metabolism_model=SimpleMetabolismModel`` in ``ScenarioParserV2``. That
    binding happens at construction and is what puts the class on the data flow
    diagram; its per-step invocation is a separate, traced edge that belongs on
    the sequence diagram.
"""

import ast
import os

from tidepool_data_science_simulator.diagramgen.naming import top_package

__all__ = ["ReferenceSite", "ScanResult", "scan_repo"]


class ReferenceSite(object):
    """One place where this repo names a symbol from another package."""

    __slots__ = ("rel_path", "module", "enclosing", "target_module", "target_symbol", "kind", "lineno")

    def __init__(self, rel_path, module, enclosing, target_module, target_symbol, kind, lineno):
        self.rel_path = rel_path
        self.module = module
        self.enclosing = enclosing
        self.target_module = target_module
        self.target_symbol = target_symbol
        self.kind = kind
        self.lineno = lineno

    @property
    def sort_key(self):
        return (self.rel_path, self.lineno, self.target_module, self.target_symbol, self.kind)

    def to_dict(self):
        return {
            "path": self.rel_path,
            "module": self.module,
            "enclosing": self.enclosing,
            "target_module": self.target_module,
            "target_symbol": self.target_symbol,
            "kind": self.kind,
            "line": self.lineno,
        }


class ScanResult(object):
    def __init__(self, sites, files_scanned, files_excluded, unparsable):
        self.sites = sites
        self.files_scanned = files_scanned
        self.files_excluded = files_excluded
        self.unparsable = unparsable


class _ReferenceCollector(ast.NodeVisitor):
    """Collects cross-package references, tracking the enclosing qualified name."""

    def __init__(self, rel_path, module, packages):
        self.rel_path = rel_path
        self.module = module
        self.packages = packages
        self.own_package = top_package(module)
        # local name -> (target module, target symbol)
        self.imported = {}
        self.sites = []
        self._scope = []
        self._called_nodes = set()

    # -- scope tracking ----------------------------------------------------

    def _visit_scoped(self, node):
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    visit_ClassDef = _visit_scoped
    visit_FunctionDef = _visit_scoped
    visit_AsyncFunctionDef = _visit_scoped

    @property
    def enclosing(self):
        return ".".join(self._scope) if self._scope else "<module>"

    # -- imports -----------------------------------------------------------

    def visit_Import(self, node):
        for alias in node.names:
            package = top_package(alias.name)
            if package in self.packages and package != self.own_package:
                local = alias.asname or alias.name.split(".", 1)[0]
                self.imported[local] = (alias.name, "")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.level:  # relative import: same package by construction
            self.generic_visit(node)
            return
        source = node.module or ""
        package = top_package(source)
        if package in self.packages and package != self.own_package:
            for alias in node.names:
                local = alias.asname or alias.name
                self.imported[local] = (source, alias.name)
        self.generic_visit(node)

    # -- references --------------------------------------------------------

    def visit_Call(self, node):
        self._record(node.func, kind="call")
        self._mark_call_target(node.func)
        self.generic_visit(node)

    def _mark_call_target(self, func):
        """Mark the whole attribute chain of a call target as already recorded.

        ``m.models.foo()`` is one call, not a call plus two bindings of ``m``
        and ``m.models``. Marking only the outermost node would leave the
        traversal to record each inner link as a separate `bind` site and
        inflate the coverage report with references that do not exist.
        """
        while isinstance(func, ast.Attribute):
            self._called_nodes.add(id(func))
            func = func.value
        if isinstance(func, ast.Name):
            self._called_nodes.add(id(func))

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and id(node) not in self._called_nodes:
            self._record(node, kind="bind")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load) and id(node) not in self._called_nodes:
            self._record(node, kind="bind")
        self.generic_visit(node)

    def _record(self, node, kind):
        resolved = self._resolve(node)
        if resolved is None:
            return
        target_module, target_symbol = resolved
        self.sites.append(
            ReferenceSite(
                rel_path=self.rel_path,
                module=self.module,
                enclosing=self.enclosing,
                target_module=target_module,
                target_symbol=target_symbol,
                kind=kind,
                lineno=node.lineno,
            )
        )

    def _resolve(self, node):
        """Map an expression back to (target module, target symbol), or None."""
        if isinstance(node, ast.Name):
            entry = self.imported.get(node.id)
            if entry is None:
                return None
            target_module, target_symbol = entry
            return (target_module, target_symbol or node.id)

        if isinstance(node, ast.Attribute):
            root = node
            parts = []
            while isinstance(root, ast.Attribute):
                parts.append(root.attr)
                root = root.value
            if not isinstance(root, ast.Name):
                return None
            entry = self.imported.get(root.id)
            if entry is None:
                return None
            parts.reverse()
            target_module, target_symbol = entry
            if target_symbol:
                # `from x import y` then `y.z(...)`: the module is x, the symbol
                # the dotted path from y.
                return (target_module, ".".join([target_symbol] + parts))
            # `import x.y` then `x.y.z(...)`: everything but the last part is
            # module path, the last is the symbol.
            full = ".".join([root.id] + parts)
            return (full.rsplit(".", 1)[0], full.rsplit(".", 1)[-1])

        return None


def _module_name_for(rel_path):
    """Dotted module name for a repo-relative ``.py`` path."""
    without_ext = os.path.splitext(rel_path)[0]
    parts = [part for part in without_ext.split(os.sep) if part]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def scan_repo(repo_root, exclusions, packages):
    """Enumerate cross-package reference sites across the whole repository.

    Parameters
    ----------
    repo_root: str
        Absolute path to the repository root.
    exclusions: Exclusions
        Curated exclusion list; unconditional directory exclusions are applied
        inside it.
    packages: iterable of str
        Top-level package names that count as "another package".

    Returns
    -------
    ScanResult
    """
    packages = frozenset(packages)
    sites = []
    scanned = 0
    excluded = 0
    unparsable = []

    for dirpath, dirnames, filenames in os.walk(repo_root):
        rel_dir = os.path.relpath(dirpath, repo_root)
        rel_dir = "" if rel_dir == "." else rel_dir
        # Prune hidden directories except the ones we explicitly care about, and
        # prune anything the exclusion rules already reject, so a `venv` tree is
        # never descended into at all.
        keep = []
        for name in sorted(dirnames):
            candidate = os.path.join(rel_dir, name) if rel_dir else name
            if name.startswith(".") and name != ".claude":
                continue
            if exclusions.is_excluded(candidate + os.sep + "__probe__.py"):
                excluded += _count_python_files(os.path.join(dirpath, name))
                continue
            keep.append(name)
        dirnames[:] = keep

        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            rel_path = os.path.join(rel_dir, filename) if rel_dir else filename
            if exclusions.is_excluded(rel_path):
                excluded += 1
                continue
            full_path = os.path.join(dirpath, filename)
            try:
                with open(full_path, "r", encoding="utf-8") as handle:
                    tree = ast.parse(handle.read(), filename=rel_path)
            except (SyntaxError, UnicodeDecodeError, OSError) as error:
                unparsable.append((rel_path, str(error)))
                continue
            scanned += 1
            collector = _ReferenceCollector(rel_path, _module_name_for(rel_path), packages)
            collector.visit(tree)
            sites.extend(collector.sites)

    sites.sort(key=lambda site: site.sort_key)
    return ScanResult(sites=sites, files_scanned=scanned, files_excluded=excluded, unparsable=sorted(unparsable))


def _count_python_files(directory):
    total = 0
    for _dirpath, _dirnames, filenames in os.walk(directory):
        total += sum(1 for name in filenames if name.endswith(".py"))
    return total
