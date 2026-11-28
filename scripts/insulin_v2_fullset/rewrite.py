"""
Text-level rewriting for one arm: suffixed profile/preset/base/settings copies, and the pointer rewrites in the
copied TLR configs. Only the intended values change; every other byte is preserved.
"""
import json

from . import jsonspans as js
from .locate import locate, new_pointer, search_dirs, split_pointer
from .translation import ARMS

FIASP_SETTINGS_POINTER = "reusable.loop_settings.fiasp_v2"
PATIENT_MS = ["patient", "patient_model", "metabolism_settings"]


class GenerationError(Exception):
    pass


def arm_tag(arm):
    return arm.replace("loop_risk_v2_", "")  # e.g. rai_main


class Registry:
    """New reusable files for one run: path (relative to the scenario root) -> text."""

    def __init__(self, root):
        self.root = root
        self.files = {}      # Path (absolute) -> text
        self.by_arm = {}     # arm -> set of Paths

    def add(self, arm, pointer, source_path, name, text):
        """Register ``name`` in the folder of ``source_path``. Returns the new pointer."""
        target = source_path.parent / f"{name}.json"
        for d in search_dirs(pointer, self.root):
            clash = d / f"{name}.json"
            if clash != target and (clash.is_file() or clash in self.files):
                raise GenerationError(f"new file {target.name} would share a name with {clash} in its resolver search path")
        if target in self.files and self.files[target] != text:
            raise GenerationError(f"two different contents generated for {target}")
        self.files[target] = text
        self.by_arm.setdefault(arm, set()).add(target)
        return new_pointer(pointer, name)


def _read(path):
    return path.read_text()


def set_keys_text(text, updates):
    """Set top-level keys of a JSON object text; updates = {key: value}. Existing keys are replaced in place."""
    for k, v in updates.items():
        root = js.parse(text)
        text = js.apply(text, [js.set_member_edit(text, root, k, v)])
    return text


def profile_pointer(reg, arm, pointer, ptype, timing):
    """Pointer to a suffixed copy of a patient metabolism file with only the insulin type/timing changed."""
    src = locate(pointer, reg.root)
    if src is None:
        raise GenerationError(f"cannot locate {pointer}")
    stem = pointer.split(".")[-1]
    nominal = ARMS[arm][:2]
    name = f"{stem}_{arm_tag(arm)}" if (ptype, timing) == nominal else f"{stem}_{arm_tag(arm)}__{ptype}_{timing or 'na'}"
    updates = {"patient_insulin_type": ptype}
    if timing is not None:
        updates["insulin_timing"] = timing
    return reg.add(arm, pointer, src, name, set_keys_text(_read(src), updates))


def settings_pointer(reg, arm, pointer, model):
    """Pointer to controller settings equal to ``pointer`` but with the given model."""
    src = locate(pointer, reg.root)
    if src is None:
        raise GenerationError(f"cannot locate {pointer}")
    fiasp = locate(FIASP_SETTINGS_POINTER, reg.root)
    if model == "fiasp" and fiasp is not None:
        a, b = json.loads(_read(src)), json.loads(_read(fiasp))
        a.pop("model", None), b.pop("model", None)
        if a == b:  # differs from fiasp_v2 only in model: use the existing fiasp_v2 mechanism
            return FIASP_SETTINGS_POINTER
    name = f"{pointer.split('.')[-1]}_{arm_tag(arm)}__{model}"
    return reg.add(arm, pointer, src, name, set_keys_text(_read(src), {"model": model}))


def base_text_for(reg, arm, base_pointer, base_target, base_model_now, cache):
    """Suffixed copy of a base simulation file: patient metabolism pointer and (URAI) controller settings pointer."""
    key = (arm, base_pointer)
    if key in cache:
        return cache[key]
    src = locate(base_pointer, reg.root)
    if src is None:
        raise GenerationError(f"cannot locate base {base_pointer}")
    text = _read(src)
    root = js.parse(text)
    ptype, timing, model = base_target
    edits = []
    ms = js.get(root, PATIENT_MS)
    if ms is None:
        raise GenerationError(f"{base_pointer}: no patient.patient_model.metabolism_settings")
    if ms.kind == "str":
        edits.append(js.replace_edit(ms, profile_pointer(reg, arm, ms.value(text), ptype, timing)))
    elif ms.kind == "obj":
        raise GenerationError(f"{base_pointer}: inline patient metabolism settings are not supported")
    ctrl = js.get(root, ["controller"])
    if ctrl is not None and ctrl.kind == "obj" and model is not None and model != base_model_now:
        st = js.get(ctrl, ["settings"])
        if st is None or st.kind != "str":
            raise GenerationError(f"{base_pointer}: controller settings must be a pointer to change the model")
        edits.append(js.replace_edit(st, settings_pointer(reg, arm, st.value(text), model)))
    new_text = js.apply(text, edits)
    name = f"{base_pointer.split('.')[-1]}_{arm_tag(arm)}"
    ptr = reg.add(arm, base_pointer, src, name, new_text)
    cache[key] = ptr
    return ptr


def config_text_for(reg, arm, text, new_base_pointer, items):
    """
    Copy of a TLR config with the base pointer swapped and per-override patient pointers rewritten.

    ``items`` is a list (one per override) of dicts: {"patient": (ptype, timing) | None, "corrections": {...}}.
    """
    root = js.parse(text)
    bc = js.get(root, ["base_config"])
    text = js.apply(text, [js.replace_edit(bc, new_base_pointer)])
    for i, item in enumerate(items):
        text = _rewrite_item(reg, arm, text, i, item)
    return text


def _item_node(text, i):
    return js.get(js.parse(text), ["override_config", i])


def _rewrite_item(reg, arm, text, i, item):
    ptype, timing = item["patient"]
    # 1. patient-side preset repoint -> suffixed copy
    ms = js.get(_item_node(text, i), PATIENT_MS)
    if ms is not None:
        if ms.kind == "str":
            ptr = profile_pointer(reg, arm, ms.value(text), ptype, timing)
            text = js.apply(text, [js.replace_edit(ms, ptr)])
        elif ms.kind == "obj":
            upd = {"patient_insulin_type": ptype, **({"insulin_timing": timing} if timing is not None else {})}
            for k, v in upd.items():
                node = js.get(_item_node(text, i), PATIENT_MS)
                text = js.apply(text, [js.set_member_edit(text, node, k, v)])
    # 2. corrections (added only after verification found a mismatch)
    corr = item.get("corrections") or {}
    if "patient" in corr and ms is None:
        p, t = corr["patient"]
        val = {"patient_insulin_type": p, **({"insulin_timing": t} if t is not None else {})}
        text = _ensure_path(text, i, ["patient", "patient_model", "metabolism_settings"], val)
    if "model" in corr:
        text = _correct_model(reg, arm, text, i, corr["model"])
    return text


def _ensure_path(text, i, path, leaf_value):
    """Add ``path`` (missing tail) under override item ``i`` holding ``leaf_value``."""
    node = _item_node(text, i)
    depth = 0
    while depth < len(path) and js.get(node, [path[depth]]) is not None:
        node = js.get(node, [path[depth]])
        depth += 1
    if depth == len(path):  # leaf exists: merge keys
        for k, v in leaf_value.items():
            node = js.get(_item_node(text, i), path)
            text = js.apply(text, [js.set_member_edit(text, node, k, v)])
        return text
    nested = leaf_value
    for key in reversed(path[depth + 1:]):
        nested = {key: nested}
    return js.apply(text, [js.set_member_edit(text, node, path[depth], nested)])


def _correct_model(reg, arm, text, i, model):
    ctrl = js.get(_item_node(text, i), ["controller"])
    if ctrl is None or ctrl.kind != "obj":
        return _ensure_path(text, i, ["controller", "settings"], {"model": model})
    st = js.get(ctrl, ["settings"])
    if st is None or st.kind == "obj":
        return _ensure_path(text, i, ["controller", "settings"], {"model": model})
    if st.kind == "str":
        return js.apply(text, [js.replace_edit(st, settings_pointer(reg, arm, st.value(text), model))])
    raise GenerationError(f"override {i}: cannot correct controller model")
