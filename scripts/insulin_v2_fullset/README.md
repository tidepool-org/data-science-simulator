# TRSET-84: full-set scenario configs for the v1 insulin models

Builds three arm directories next to `loop_risk_v2_2_0_full`:

| Arm directory | Patient insulin type | `insulin_timing` | Controller model |
|---|---|---|---|
| `loop_risk_v2_rai_main` | `rapid_acting_v1` | `main` | baseline, unchanged |
| `loop_risk_v2_urai_main` | `ultra_rapid_v1` | `main` | `fiasp` |
| `loop_risk_v2_urai_old` (A2) | `fiasp` (old model) | n/a | `fiasp` |

Rows for sims whose baseline pair is not the plain one come from `translation_table.json`. A1 (the old RAI reference)
is a plain re-run of `loop_risk_v2_2_0_full` and needs no configs. `faster`/`slower` arms are out of scope (Rev 1);
reinstating them means re-adding them to `ARMS` in `translation.py`.

## Phase 1: inventory (read-only)

```bash
python -m scripts.insulin_v2_fullset.inventory [--write-table]
```

Resolves every in-scope baseline sim through `ScenarioParserV2` without building or simulating, and reports each
(patient insulin type, controller model) pair with its TLRs and sim IDs (`output/inventory_report.*`, gitignored).
`--write-table` adds rows for new pairs and never touches existing rows.

## Phase 2: generation

```bash
python -m scripts.insulin_v2_fullset.generate [--tlr NAME ...] [--arms ARM ...] [--dry-run] [--force]
```

1. Fails, writing nothing, if any in-scope baseline pair has no usable row in the table.
2. Per arm, copies each TLR config with the `base_config` pointer (and any patient-side metabolism pointer in an
   override) rewritten to suffixed copies (`_rai_main`, `_urai_main`, `_urai_old`) in the same folder as the original.
   Copies change only `patient_insulin_type` and `insulin_timing` (and controller `model`, through `fiasp_v2` when the
   settings differ only in model). Edits are made on the original text, so other bytes are preserved. Pump-side
   pointers are never changed.
3. Verifies by difference: every generated sim, resolved through `ScenarioParserV2`, must equal its baseline sim in
   every field except the three insulin fields, and those must match the table. Where an override undoes the arm's
   setting (for example a full controller-settings pointer carrying its own model), a minimal correction is added to
   that override item, then re-verified. A table `insulin_timing` of `null` means "n/a": absent, or `main` for a non-v1
   patient type (the model treats `main` as the default for old types).
4. Verification runs against a staged clone of `reusable/`; files are written only when every arm passes.
5. Refuses to overwrite an existing file whose content differs unless `--force`. Reruns are byte-identical.
6. Writes `_manifest.json` in each arm directory (generator commit, baseline tree hash, table hash, exclude list,
   created reusable files with hashes). The runner only lists directories containing `TLR-`, so the manifest is ignored.

`exclude_list.json` names TLR directories left out of every arm. Names that match no directory are an error.

## Tests

`tests/test_insulin_v2_fullset_{inventory,generator,integration}.py`. The end-to-end Loop run in the integration test is
local only: set `TRSET84_E2E=1`.
