"""
Locate the file a reusable pointer resolves to. Mirrors ``ScenarioParserV2.load_pointer`` (first match in a fixed
folder order). tests/test_insulin_v2_fullset_generator.py asserts it agrees with the parser for every baseline
pointer, so any drift in the parser's search list is caught.
"""
import os
from pathlib import Path

SIMULATION_SUBDIRS = ["base", "suspend", "loop_versions", "specialized", "versions", "1xComparator", "custom_presets"]
METABOLISM_SUBDIRS = ["profiles", "suspensions", "presets", "versions", "types"]


def split_pointer(pointer):
    """'reusable.a.b.name' -> ('reusable.a.b', 'name')."""
    segs = pointer.split(".")
    return ".".join(segs[:-1]), segs[-1]


def search_dirs(pointer, root):
    """Directories searched for this pointer, in order."""
    segs = pointer.split(".")
    folder = "/".join(segs[:-1])
    subdirs = []
    if "simulations" in folder:
        subdirs = SIMULATION_SUBDIRS
    elif "metabolism_settings" in folder:
        subdirs = METABOLISM_SUBDIRS
    return [Path(root) / folder] + [Path(root) / folder / s for s in subdirs]


def locate(pointer, root):
    """Path of the .json file the parser would load for this pointer, or None."""
    name = pointer.split(".")[-1] + ".json"
    for d in search_dirs(pointer, root):
        if (d / name).is_file():
            return d / name
    return None


def new_pointer(pointer, new_name):
    prefix, _ = split_pointer(pointer)
    return f"{prefix}.{new_name}"
