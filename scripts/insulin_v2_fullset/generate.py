"""
Phase 2 generator for TRSET-84: builds the arm directories, verifies them by difference, then writes.

Nothing is written unless every arm verifies. Verification runs against a staging clone of ``reusable/`` so the
real tree is untouched until the end.

Usage:
    python -m scripts.insulin_v2_fullset.generate [--tlr NAME ...] [--arms ARM ...] [--force] [--dry-run]
        [--scenario-root DIR] [--baseline DIR] [--out-root DIR]
"""
import argparse
import copy
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2

from . import manifest as mf
from .locate import locate
from .resolve import (DEFAULT_BASELINE, NO_CONTROLLER, REPO_ROOT, SCENARIO_ROOT, extract_fields, load_exclude_list,
                      resolve_config_file, tlr_dirs)
from .rewrite import GenerationError, Registry, base_text_for, config_text_for
from .translation import ARMS, load_table, pair_key, table_hash, unresolved_rows
from .verify import compare, target_for


class UnmappedPairError(GenerationError):
    pass


class VerificationError(GenerationError):
    pass


class OverwriteError(GenerationError):
    pass


def _clone_reusable(src_root, dst_root):
    dst_root.mkdir(parents=True, exist_ok=True)
    src, dst = Path(src_root) / "reusable", Path(dst_root) / "reusable"
    try:  # APFS clone: fast and copy-on-write
        subprocess.run(["cp", "-Rc", str(src), str(dst)], check=True, capture_output=True)
    except Exception:
        shutil.copytree(src, dst)


def _row_for(table_rows, ptype, model):
    key = pair_key(ptype, model)
    if key not in table_rows:
        raise UnmappedPairError(f"baseline pair has no translation-table row: {key}")
    return table_rows[key]


def _base_info(scenario_root, base_pointer):
    """Baseline fields of a base simulation file with pointers resolved (no overrides)."""
    src = locate(base_pointer, scenario_root)
    if src is None:
        raise GenerationError(f"cannot locate base {base_pointer}")
    parser = ScenarioParserV2()
    parser.pointer_object_dir = str(scenario_root)
    d = json.loads(src.read_text())
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        parser.resolve_pointers(d)
    return extract_fields(d)


def run(scenario_root=SCENARIO_ROOT, baseline=DEFAULT_BASELINE, out_root=None, arms=None, tlrs=None, force=False,
        dry_run=False, exclude=None, table=None, sha=None, dirty=None, log=print):
    scenario_root, baseline = Path(scenario_root), Path(baseline)
    out_root = Path(out_root) if out_root else baseline.parent
    arms = list(arms or ARMS)
    exclude = load_exclude_list() if exclude is None else exclude
    table = load_table() if table is None else table
    for a in arms:
        if a not in ARMS:
            raise GenerationError(f"unknown arm {a}")
    bad = unresolved_rows(table)
    if bad:
        raise UnmappedPairError(f"translation table has unresolved rows: {bad}")
    rows = {r["key"]: r for r in table["rows"]}

    dirs = tlr_dirs(baseline, exclude)
    if tlrs:
        missing = sorted(set(tlrs) - {d.name for d in dirs})
        if missing:
            raise GenerationError(f"--tlr names not in scope: {missing}")
        dirs = [d for d in dirs if d.name in set(tlrs)]
    cfgs = [(d, c) for d in dirs for c in sorted(d.glob("*.json"))]

    # 1. Resolve every baseline sim; fail before writing if any pair is unmapped.
    log(f"Resolving {len(cfgs)} baseline configs in {len(dirs)} TLR dirs ...")
    base_recs = {}
    for d, c in cfgs:
        recs = resolve_config_file(c, scenario_root)
        base_recs[c] = recs
        for r in recs:
            _row_for(rows, r["patient_insulin_type"], r["controller_model"])

    tmp = Path(tempfile.mkdtemp(prefix="insulin_v2_stage_"))
    try:
        stage_root = tmp / "root"
        _clone_reusable(scenario_root, stage_root)
        reg = Registry(scenario_root)
        out_files = {}   # Path (final location) -> text
        staged = set()

        def stage_registry():
            for p, text in reg.files.items():
                if p not in staged:
                    sp = stage_root / p.relative_to(scenario_root)
                    sp.parent.mkdir(parents=True, exist_ok=True)
                    sp.write_text(text)
                    staged.add(p)

        for arm in arms:
            base_cache, info_cache = {}, {}
            corrections = {}   # cfg path -> {override index: {...}}
            for attempt in range(3):
                gen_texts = {}
                for d, c in cfgs:
                    text = c.read_text()
                    bptr = json.loads(text)["base_config"]
                    if not isinstance(bptr, str):
                        raise GenerationError(f"{c}: inline base_config is not supported")
                    if bptr not in info_cache:
                        info_cache[bptr] = _base_info(scenario_root, bptr)
                    f = info_cache[bptr]
                    row = _row_for(rows, f["patient_insulin_type"], f["controller_model"])
                    btarget = target_for(row, arm, f)
                    new_bptr = base_text_for(reg, arm, bptr, btarget, f["controller_model"], base_cache)
                    items = []
                    for i, r in enumerate(base_recs[c]):
                        t = target_for(_row_for(rows, r["patient_insulin_type"], r["controller_model"]), arm, r)
                        items.append({"patient": (t[0], t[1]), "corrections": corrections.get(c, {}).get(i)})
                    gen_texts[c] = config_text_for(reg, arm, text, new_bptr, items)
                stage_registry()
                # 2. verification by difference
                new_corr, hard = {}, []
                for d, c in cfgs:
                    sp = tmp / "cfg" / arm / d.name / c.name
                    sp.parent.mkdir(parents=True, exist_ok=True)
                    sp.write_text(gen_texts[c])
                    try:
                        gen = resolve_config_file(sp, stage_root)
                    except Exception as e:
                        hard.append(f"{d.name}/{c.name}: generated config does not resolve: {type(e).__name__}: {e}")
                        continue
                    base = base_recs[c]
                    if len(gen) != len(base):
                        hard.append(f"{d.name}/{c.name}: sim count {len(base)} != {len(gen)}")
                        continue
                    for i, (b, g) in enumerate(zip(base, gen)):
                        t = target_for(_row_for(rows, b["patient_insulin_type"], b["controller_model"]), arm, b)
                        for p in compare(b, g, t):
                            if p["kind"] == "hard":
                                hard.append(f"{d.name}/{c.name} [{b['sim_id']}]: {p['detail']}")
                            elif p["kind"] == "patient":
                                new_corr.setdefault(c, {}).setdefault(i, {})["patient"] = (t[0], t[1])
                                log(f"  correction {d.name}/{c.name} [{b['sim_id']}] patient: got {p['got']}, need {p['expected']}")
                            else:
                                new_corr.setdefault(c, {}).setdefault(i, {})["model"] = t[2]
                                log(f"  correction {d.name}/{c.name} [{b['sim_id']}] controller model: got {p['got']}, need {p['expected']}")
                if hard:
                    raise VerificationError(f"{arm}: {len(hard)} sims failed verification by difference:\n  " +
                                            "\n  ".join(hard[:20]))
                if not new_corr:
                    break
                n = sum(len(v) for v in new_corr.values())
                log(f"{arm}: applying minimal corrections to {n} override items (attempt {attempt + 1})")
                if attempt == 2:
                    raise VerificationError(f"{arm}: corrections did not converge")
                for c, per in new_corr.items():
                    for i, v in per.items():
                        corrections.setdefault(c, {}).setdefault(i, {}).update(v)
            for d, c in cfgs:
                out_files[out_root / arm / d.name / c.name] = gen_texts[c]
            log(f"{arm}: verified {sum(len(base_recs[c]) for _, c in cfgs)} sims in {len(cfgs)} configs")

        # 3. manifests
        base_hash = mf.tree_hash(dirs)
        thash = table_hash(table)
        manifests = {}
        for arm in arms:
            m = mf.build_manifest(arm, sha, dirty, base_hash, thash, exclude, [d.name for d in dirs],
                                  {p: reg.files[p] for p in reg.by_arm.get(arm, set())}, scenario_root)
            manifests[out_root / arm / mf.MANIFEST_NAME] = mf.dump(m)

        # 4. write (all-or-nothing on conflicts)
        everything = {**reg.files, **out_files, **manifests}
        conflicts = [p for p, t in everything.items() if p.exists() and p.read_text() != t]
        if conflicts and not force:
            raise OverwriteError(f"{len(conflicts)} existing files differ from generated content (use --force): "
                                 f"{[str(p) for p in conflicts[:5]]}")
        if dry_run:
            log(f"dry run: would write {len(everything)} files ({len(reg.files)} reusable)")
            return everything
        written = 0
        for p, t in sorted(everything.items()):
            if p.exists() and p.read_text() == t:
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(t)
            written += 1
        log(f"wrote {written} files ({len(everything)} generated, {len(reg.files)} reusable)")
        return everything
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario-root", default=str(SCENARIO_ROOT))
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--tlr", nargs="*", default=None, help="limit to these TLR directory names")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    sha, dirty = mf.git_state(REPO_ROOT)
    try:
        run(a.scenario_root, a.baseline, a.out_root, a.arms, a.tlr, a.force, a.dry_run, sha=sha, dirty=dirty)
    except GenerationError as e:
        print(f"FAILED ({type(e).__name__}): {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
