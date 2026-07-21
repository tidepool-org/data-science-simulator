"""
Unit test for setup.py's packages=[...] list.

Guards against the regression found 2026-07-21: 'projects' and 'validation'
were omitted, so a non-editable install (a real wheel/sdist build, as opposed
to this repo's editable dev install) would silently exclude loop_risk_v2_0.py
and the whole validation/ package. Verified against a real `bdist_wheel`
build that this fix resolves it; this test guards the declared list only
(fast, no build step) so a future edit can't silently drop these again.
"""

import ast
import os


def _get_setup_packages():
    """Parse setup.py's AST to extract the literal packages=[...] list."""
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    setup_path = os.path.join(repo_root, "setup.py")
    with open(setup_path) as fh:
        tree = ast.parse(fh.read(), filename=setup_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup":
            for keyword in node.keywords:
                if keyword.arg == "packages":
                    return ast.literal_eval(keyword.value)
    raise AssertionError("Could not find packages=[...] in setup.py")


def test_projects_risk_package_is_declared():
    packages = _get_setup_packages()
    assert "tidepool_data_science_simulator.projects" in packages
    assert "tidepool_data_science_simulator.projects.risk" in packages


def test_validation_package_is_declared():
    packages = _get_setup_packages()
    assert "tidepool_data_science_simulator.validation" in packages
