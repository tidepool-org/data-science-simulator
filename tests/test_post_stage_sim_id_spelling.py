"""TRSET-59 drift guard: the post-mitigation stage spelling in scenario configs.

Canonical spelling is ``post-Loop_WithMitigations_``. ``84329fd`` moved the tree to
it; the retired ``post-Loop-WithMitigations_`` must not come back anywhere.

The in-scope collection, ``loop_risk_v2_2_0_full``, still carries a frozen set of
non-canonical TLR directories. Normalising them is a separate sim_id change (see the
TRSET-59 change record), so they are named here rather than ignored: a new
non-canonical directory fails, and so does an allow-listed one that gets fixed
without being removed from the list.
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent / "scenario_configs" / "tidepool_risk_v2" / "loop_risk_v2_0"
IN_SCOPE = ROOT / "loop_risk_v2_2_0_full"

CANONICAL = "post-Loop_WithMitigations_"
RETIRED_HYPHEN = re.compile(r"^post-Loop-WithMitigations")
POST = re.compile(r"^post")

# TLR directories in loop_risk_v2_2_0_full known to carry a non-canonical post spelling.
ALLOWED_NON_CANONICAL = {
    # post-LoopWithMitigations_ (no separator)
    "TLR-676", "TLR-742", "TLR-745", "TLR-808", "TLR-822", "TLR-826",
    # post-Loop_withMitigations_ (lowercase w)
    "TLR-590", "TLR-604",
    "TLR-799_Step1_DO_NOT_USE_FOR_RISK", "TLR-799_USE_STEP1_TMP_USE_FOR_RISK_EVAL",
    # post_Loop_WithMitigations_ (underscore after 'post')
    "TLR-912",
}


def _post_sim_ids(path):
    ids = []

    def walk(node):
        if isinstance(node, dict):
            v = node.get("sim_id")
            if isinstance(v, str) and POST.match(v):
                ids.append(v)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(json.loads(path.read_text()))
    return ids


def _configs(base):
    return sorted(base.rglob("Simulation-Configuration-*.json"))


@pytest.mark.skipif(not IN_SCOPE.is_dir(), reason="scenario_configs not present")
def test_retired_hyphen_spelling_absent_everywhere():
    offenders = [
        str(p.relative_to(ROOT)) for p in _configs(ROOT)
        if any(RETIRED_HYPHEN.match(s) for s in _post_sim_ids(p))
    ]
    assert not offenders, f"retired 'post-Loop-WithMitigations' found in: {offenders[:10]}"


@pytest.mark.skipif(not IN_SCOPE.is_dir(), reason="scenario_configs not present")
def test_in_scope_collection_post_spelling_matches_baseline():
    non_canonical = {
        p.relative_to(IN_SCOPE).parts[0]
        for p in _configs(IN_SCOPE)
        if any(not s.startswith(CANONICAL) for s in _post_sim_ids(p))
    }
    assert non_canonical - ALLOWED_NON_CANONICAL == set(), "new non-canonical spelling"
    assert ALLOWED_NON_CANONICAL - non_canonical == set(), "fixed: remove from allow-list"


def test_guard_detects_retired_spelling(tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"a": [{"sim_id": "post-Loop-WithMitigations_t1_median"}]}))
    assert RETIRED_HYPHEN.match(_post_sim_ids(f)[0])
