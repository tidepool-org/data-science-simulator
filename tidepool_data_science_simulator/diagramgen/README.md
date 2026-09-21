# Trace-based architecture diagram generator (TRSET-51)

Instruments a real simulation run and emits architecture figures that are
**evidence of executed behavior**, not hand-maintained assertions about
permitted structure. The figures are for the TRSET position paper, which has a
multi-year life across the AINN response, partner discussions and EU MDR, so
they have to be regenerable against current code.

## Running it

```bash
python -m tidepool_data_science_simulator.diagramgen
```

Writes five artifacts into `.docs/architecture/`:

| File | What it is |
| --- | --- |
| `data_flow.mmd` | Cross-package data flow, Mermaid `flowchart` |
| `timestep_sequence.mmd` | One fully-exercised control cycle, Mermaid `sequenceDiagram` |
| `manifest.json` | Provenance: package commits, resolved pointer files, config SHAs |
| `trace.jsonl` | The filtered trace the figures were derived from — **gitignored** |
| `coverage_report.md` | Static cross-package reference sites diffed against what ran |

Four of those five are committed. `trace.jsonl` is not: it is roughly 7 MB per
run (11,384 records for the full reference scenario) and nothing downstream
reads it back — the figures and the coverage report are derived in the same run
that writes it. It stays on disk as a local diagnostic for anyone who wants to
ask a question the figures do not answer, and `.gitignore` keeps it out of
history.

Two further files live in the same directory and are **inputs**, maintained by
hand: `allowlist.yml` (what the tracer keeps) and `exclusions.yml` (what the
static pass skips).

### Drift gate

```bash
python -m tidepool_data_science_simulator.diagramgen --check
```

Regenerates into a scratch directory and exits non-zero if a committed figure's
normalized body has drifted. Suitable for a pre-commit hook.

The gate covers the two `.mmd` figures and nothing else. `manifest.json`
carries commit SHAs and a generation timestamp by design, and `trace.jsonl` is
not committed at all.
`coverage_report.md` carries source line numbers, which move whenever anything
above them is edited — gating it would fire the architecture drift gate on a
reformatted comment, and a gate that cries wolf gets switched off. Its volatile
measures (files scanned, files excluded, functions executed) live in its
provenance header for the same reason.

CI wiring is a
separate ticket: this repo has no `.github/`, and `conda-environment.yml`
installs two packages from absolute local editable paths, so the environment is
not reproducible on a runner today.

`--check` compares bodies with the `%%` provenance header stripped. The
manifest's SHAs change on every commit to any of four packages; a gate that
fired on those would fire constantly and be turned off within a week.

macOS only. The Swift algorithm ships as `libLoopAlgorithmToPython.dylib`.

## Execution model

`Simulation` subclasses `multiprocessing.Process` and `run_simulations()` always
spawns. Under macOS/Python 3.12 `spawn`, a parent-side tracer observes nothing
of the timestep. So the generator builds simulations with the real
`ScenarioParserV2` and calls `Simulation.run()` **in-process**. It never calls
`sim.start()` and never goes through `run_simulations()`. Every cross-package
edge then lives in one traceable process, and **simulator execution code needs
no change at all** — this package is a pure consumer.

Three consumer-side adjustments are made to the built objects. None is a change
to simulator code:

- `sim.multiprocess = False`, because `build_sim_from_config()` hard-codes
  `True`, which would make `run()` reconfigure the root logger onto a file under
  `DATA_DIR/logs` and push results through a `multiprocessing.Queue`.
- `sim.controller.loop_algo_io_dir = <scratch dir>`, because the Swift
  controller writes two JSON files per control cycle — roughly 1,100 for a full
  reference run — and falls back to the process working directory.
- The traced block runs with the working directory moved to that same scratch
  directory, because `Simulation.__init__` already runs one control cycle at t=0
  before the attribute above can be set.

## Capture

Two mechanisms, because neither alone sees the whole picture.

**`sys.setprofile`** is the workhorse. It reports every Python call and return,
giving the callee frame and — via `frame.f_back` — the caller, so an edge is
observed with its concrete bindings on both ends. Filtering happens *inside* the
callback against `allowlist.yml`, not post-hoc over a full trace, which avoids a
second traversal of roughly 1,100 control cycles.

**`sys.monitoring`** (PEP 669, Python 3.12+), narrowly scoped, captures the
native boundary.

> The originating ticket assumed `setprofile`'s `c_call` event would expose the
> `ctypes.CDLL` → `libLoopAlgorithmToPython.dylib` edge. Measured against a real
> run, it does not. `swift_lib.getLoopRecommendations` is a ctypes `_FuncPtr`
> (`PyCFuncPtr`), not a `PyCFunction`, and CPython emits no `c_call` for it — the
> only `_ctypes` events on the wire come from `POINTER` during `restype` setup.
> `sys.monitoring`'s `CALL` event does report it. Returning
> `sys.monitoring.DISABLE` for every non-ctypes callable retires the event at
> that bytecode offset, so the probe converges to firing only at genuine native
> call sites and costs nothing in steady state.

A `_FuncPtr`'s library is identified exactly, not guessed from the symbol name:
`ctypes.CDLL` builds a fresh `_FuncPtr` subclass per library, so
`type(funcptr) is cdll._FuncPtr`. Only the library's basename is kept — the
dylib lives under an editable install path that must not reach a figure.

### Concrete bindings

Resolved from the frame's `self`, so the figures say `SwiftLoopController` and
`SimpleMetabolismModel` rather than `LoopController` or a base class.

`ScenarioParserV2` imports `SimpleMetabolismModel` directly and hands the class
to `VirtualPatient`. That is a binding, not a call, so no call event carries it;
it is recovered by the static pass and rendered as a dashed
`binds (construction)` edge, restricted to bindings whose enclosing function
actually executed during the traced run. Its *per-step invocation* is a separate,
traced edge and appears on the sequence diagram.

### Sequence diagram timestep

The first control cycle of `post-Loop_WithMitigations_t1_median` at which the
Swift controller returned a non-empty recommendation — observed as
`apply_loop_recommendations` being reached, since `Simulation.update` only calls
it when the recommendation is truthy. That is the first fully-exercised cycle,
past warm-up. The `Simulation.init()` call at t=0 is excluded for free: it runs
during the construction phase, before the loop, and has a different shape.

## Determinism

- Mermaid node ids come from a stable hash of the qualified name, never
  insertion order.
- Edges are emitted sorted by `(caller, callee)`; symbol lists inside labels are
  sorted too.
- All rendered identifiers are package-qualified, never filesystem-absolute. Two
  of the four packages install from `-e file:/Users/<someone>/...`; those paths
  must not leak into a figure. Manifest and report paths get the same treatment:
  repo-relative inside the repo, home-redacted to `~` outside it.
- Timestamps, durations and PIDs appear only in `trace.jsonl`.

## Diagram generation only

The generator never calls `save_df()`, never writes to `DATA_DIR/results/...`,
and never emits a TSV or `<sim_id>.json`. Metrics return values are discarded at
the call site.

`trace.jsonl` records call identity and argument **shape** only — module,
qualname, resolved class, arity, type names. Never argument or return
**values**. No glucose array and no LBGI/DKAI scalar reaches disk. Enforced by
`tests/test_diagramgen_integration.py::test_trace_records_shape_but_never_values`.

## One attributed edge

The `data-science-metrics` edge lives in the results path, inside
`run_simulations()`, which cannot be used here because it spawns. The generator
reproduces that call so the edge is genuinely exercised, and records the caller
as `tidepool_data_science_simulator.run` with `"attributed": true` on every such
trace record. The figure labels it `results path`. The callee execution is
observed; the caller attribution is declared, and says so.

## Maintained artifacts

Three things must stay in step with the figures. This is a known, accepted cost,
change-controlled alongside them:

1. `.docs/architecture/allowlist.yml`
2. `.docs/architecture/exclusions.yml`
3. `tests/test_data/diagramgen/Simulation-Configuration-TRSET51-fixture.json`

The fixture is a short-duration structural twin of the TLR-552 median reference
scenario. It lives under `tests/` rather than `scenario_configs/` so a
`loop_risk_v2_0.py` directory walk cannot sweep it into a risk run.

## Config file format

`allowlist.yml` and `exclusions.yml` are read by a deliberately tiny reader
(`yamlmini.py`) that accepts flat mappings of a key to a scalar or a list of
scalars, and rejects everything else loudly. PyYAML is not a dependency of this
repo, and adding one to an environment that is already hard to reproduce buys
little for two files of this shape. The files remain valid YAML.

## What the coverage report does not claim

Static reachability cannot see dynamic dispatch — `ScenarioParserV2` resolves
controllers by string key (`"id": "swift"`). The report states its method rather
than claiming completeness. See its "Method" section.
