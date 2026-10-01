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
| `timestep_sequence.mmd` | One control cycle (see [Sequence diagram timestep](#sequence-diagram-timestep)), Mermaid `sequenceDiagram` |
| `manifest.json` | Provenance: package commits, resolved pointer files, config SHAs |
| `trace.jsonl` | The filtered trace the figures were derived from — **gitignored** |
| `coverage_report.md` | Static cross-package reference sites diffed against what ran |

Four of those five are committed. `trace.jsonl` is not: it is roughly 7 MB per
run (11,384 records for the full reference scenario) and nothing downstream
reads it back — the figures and the coverage report are derived in the same run
that writes it. It stays on disk as a local diagnostic for anyone who wants to
ask a question the figures do not answer, and `.gitignore` keeps it out of
history.

Five further files live in the same directory and are **inputs**, maintained by
hand: `allowlist.yml` (what the tracer keeps), `exclusions.yml` (what the static
pass skips), and the three render configs `render.json`,
`mermaid_config_png.json` and `mermaid_config_svg.json` (see
[Rendering the figures](#rendering-the-figures)).

The render step adds four more committed artifacts — `data_flow.png`,
`data_flow.svg`, `timestep_sequence.png` and `timestep_sequence.svg` — and is a
separate, explicitly-invoked command, not part of generation.

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

## Rendering the figures

```bash
python -m tidepool_data_science_simulator.diagramgen.render
```

Turns the two committed `.mmd` files into a **PNG for the position paper** and
an **SVG for the repository**, using a digest-pinned `mermaid-cli` container,
and records the whole render toolchain in `manifest.json`.

This is a separate entry point on purpose. `generate()` stays runnable on a host
with no Docker, and the render step is a consumer of the *committed* figures —
it never regenerates a `.mmd`, never imports the tracer, and never touches the
simulator. Same relationship `diagramgen` has to the simulator, one level down.

### Docker only, and nothing in the conda environment

Nothing is added to `conda-environment.yml`: no Node, no Puppeteer, no
Chrome-for-Testing, no npm. There is no host `mmdc` fallback either, and that
is deliberate — a host install would silently defeat the pin this step exists to
provide, and the figure would still come out looking fine.

Docker earns its place on four grounds. It keeps a ~280 MB browser out of an
environment already documented as not runner-reproducible. It fixes the font
set, which is the largest uncontrolled variable in Mermaid layout. It collapses
four version strings into one digest to record. And it confines the renderer to
a single mount.

### The pin

`render.json` carries the image as `…/mermaid-cli@sha256:<digest>`. A tag-only
reference is **rejected**, not warned about: the pin must not be losable by
editing one string. The digest fixes `@mermaid-js/mermaid-cli`, the resolved
`mermaid`, `puppeteer`, the Chromium build and the Alpine font set together.

Tags and packaged versions genuinely disagree upstream — ghcr tag `11.17.1`
ships `mermaid-cli-11.17.0.tgz` — so every version in the manifest is read out
of the running container rather than restated from config. What the pinned
digest currently resolves to:

| | |
| --- | --- |
| `@mermaid-js/mermaid-cli` | 11.17.0 |
| `mermaid` | 12.0.0 |
| `puppeteer` / `puppeteer-core` | 25.11.0 |
| browser | Chromium 152.0.7977.82 |
| Node | v22.23.2 |
| base | Alpine Linux v3.24 |

Note `mermaid` 12.0.0 — a major version ahead of the CLI's own 11.x, and not
something any tag would have told you. This is the reason the probe exists.

### Font

`themeVariables.fontFamily` is pinned in both Mermaid configs and **verified
present in the image before every render**. A family the image does not ship is
an error, because fontconfig would otherwise fall back silently: the figure
still renders, with different metrics, and nothing anywhere says so. The image
installs `ttf-dejavu`, `ttf-freefont`, `ttf-linux-libertine`, `ttf-inconsolata`,
`terminus-font`, `ttf-font-awesome`, `font-noto-cjk` and `font-noto-emoji`;
DejaVu Sans is the pinned choice.

### Width

**mmdc has no flag that sets an output width.** Measured against the pinned
image:

```
output_px = min(natural_layout_width, --width viewport) x --scale
```

`--width` only *caps*: at `-w 1800` and `-w 2700` a figure whose natural width
is 1418 px comes out 1418 px both times. So the render step measures each
figure and derives its scale:

1. Render behind a viewport wider than any figure (20,000 px) to read the
   natural width without clamping it.
2. Render again at `scale = target_width / natural_width`.

The configured pixel width is then hit whatever the traced graph does, and the
scale is **derived per run and recorded**, never a constant in config. That is
the distinction that matters: a fixed scale factor would give a different width,
and a different effective DPI, every time the architecture changed — which is
exactly the failure the originating request set out to avoid. It rules out a
config-level scale factor, not the flag.

Current measurements:

| Figure | Natural | Target | Scale | Output |
| --- | --- | --- | --- | --- |
| `data_flow.png` | 1922 px | 2700 px | 1.4048 | 2700 x 930 |
| `timestep_sequence.png` | 1737 px | 2000 px | 1.1514 | 2000 x 1302 |

Targets are per-figure because the two figures have different shapes: 2,700 px
for the landscape `flowchart LR`, 2,000 px for the portrait sequence diagram,
both aimed at 300 DPI print. Width and direction are config-driven, so splitting
`data_flow` into subfigures or switching it to `flowchart TB` needs no code
change.

SVG carries **no** width. mermaid-cli emits a responsive root — `width="100%"`
plus a `viewBox` — so a vector has no pixel width to pin, and a `width_px` on an
SVG output is rejected as a config error. Its intrinsic `viewBox` size is
recorded instead.

### Two outputs, two configs

| Output | `htmlLabels` | Why |
| --- | --- | --- |
| `*.png` | `true` | The deliverable. Chrome rasterizes its own `foreignObject`, so `<br/>` labels render. PNG, never JPEG — lossy compression degrades thin strokes and small text, which is most of what these figures are. White background, so the figure survives printing and dark-mode viewers. |
| `*.svg` | `false` | The repository copy. resvg, CairoSVG, librsvg and the Word and Docs importers do not implement `foreignObject`; a browser-only SVG looks empty in all of them. Future-proofing, not a condition on the deliverable. |

Both are committed files with recorded SHA-256, not inline CLI strings.
`securityLevel` is stated as `strict` and `themeCSS`/`--cssFile` go unused —
those are the surfaces the published Mermaid CSS-injection advisories exercise.

### Browser sandbox — read this before assuming isolation

The published image's `ENTRYPOINT` is `mmdc -p /puppeteer-config.json`, and that
baked-in file sets `--no-sandbox`. **This render step does not override it.**
Chromium's own sandbox needs user namespaces Docker denies by default, which is
why upstream ships it this way; overriding the entrypoint to drop the flag makes
the render fail rather than making it safer.

So the isolation boundary here is **the container and its single mount, not the
browser sandbox**. `manifest.json` records that explicitly under
`render.isolation` rather than leaving a reader to assume otherwise. On top of
it the step adds `--network none`, and mounts only a scratch directory holding
the `.mmd` sources and the Mermaid configs — the trace, the manifest and the two
maintained config files are never visible to the container.

This differs from the constraint stated in the originating code request, which
asked that `--no-sandbox` not be adopted. It is not adoptable or avoidable
separately: it is a property of the pinned image. Recorded here so the security
re-review has it in front of it.

### Failure behavior

Missing `docker`, an unpullable digest, a missing input `.mmd`, a missing
`manifest.json` or a non-zero `mmdc` exit produces a clear message on stderr and
a non-zero exit, and **leaves `.docs/architecture/` exactly as it found it**.
Rendering happens in a scratch directory; results move into place only once
every output has been produced and validated.

### What the render step does *not* gate

`GATED_FILES` stays the two `.mmd` files. Neither the PNG nor the SVG enters
`--check`. Raster and renderer output are not byte-stable across browser, font
or Mermaid version changes the way a normalized `.mmd` body is, so gating them
would fire the *architecture* drift gate on an unrelated toolchain bump — the
same reasoning that correctly kept `coverage_report.md` out.
`tests/test_diagramgen_render.py::test_rendered_figures_stay_out_of_the_drift_gate`
exists so a later, well-meaning addition trips.

### Reproducibility

**Measured byte-identical.** Five renders of each committed figure on macOS
arm64 with Docker 29.8.0 and the pinned digest produced exactly one distinct
SHA-256 per figure, so `render.json` sets `reproducibility_level: "bytes"` and
the suite asserts at that level.

The setting exists because that claim is host-dependent and should not be taken
on faith anywhere it has not been checked:

- `bytes` — identical SHA-256. The current, measured claim.
- `dimensions` — identical pixel dimensions only. Drop to this if a host is
  found where byte-stability does not hold.

The test reports the measured answer on every run either way, so the evidence is
captured rather than assumed.

### Tests

Real Docker, real pinned image, real render — the render boundary is not mocked,
on the same reasoning that kept the Swift boundary unmocked. Those tests carry
`@pytest.mark.docker` and **skip with a stated reason** when Docker is absent,
rather than passing quietly; a test that silently no-ops is no evidence, which
in a validation record is worse than a failure.

```bash
pytest tests/test_diagramgen_render.py -rs      # see the skips and their reasons
pytest tests/test_diagramgen_render.py -m docker  # demand the real render
```

The pin validation, config cross-checks, dimension readers, manifest
degradation contract, drift-gate exclusion and the Docker-unavailable failure
path all run without a daemon.

### Scope of the security assessment

The assessment behind this step holds **for local execution only**. No CI
wiring, no shared-server render, no service exposure without re-assessment. CI
is a separate ticket for the generator as a whole; it is also the trigger for
re-reviewing this.

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

`--sequence-cycle` selects which control cycle of `--sequence-stage` the
sequence diagram is cut from. The rule used is stamped into the `%%` header
(`sequence cycle rule:` and `control cycle:` lines) and into
`manifest.json` as `sequence_diagram.cycle_rule`.

| Rule | Cycle selected |
| --- | --- |
| `first-recommendation` (default) | The first cycle of the stage at which the controller returned a non-empty recommendation — observed as `apply_loop_recommendations` being reached, since `Simulation.update` only calls it when the recommendation is truthy. That is the first fully-exercised cycle, past warm-up. |
| `first-cycle` | The lowest run-phase timestep index in the stage, whether or not anything was recommended. |

The `Simulation.init()` call at t=0 is excluded for free under both rules: it
runs during the construction phase, before the loop, and has a different shape.

A `"controller": null` stage resolves to `DoNothingController`, whose
`get_loop_recommendations` returns `None`, so `apply_loop_recommendations` is
never reached. `first-recommendation` on such a stage exits non-zero with
"No control cycle in stage ... reached apply_loop_recommendations" and writes
nothing; there is deliberately no silent fallback. Ask for `first-cycle`
explicitly to draw it.

### TLR-549 figures (TRSET-57)

Committed under `.docs/architecture/TLR-549/`, one directory per figure set,
each with the `.mmd` sources, `manifest.json`, `coverage_report.md`, PNG and
SVG. They are **not** part of the drift gate: `GATED_FILES` and the scope of
`--check` are unchanged, and the TLR-552 figures in `.docs/architecture/` are
untouched. Each run regenerates `data_flow.mmd` and `coverage_report.md`, so
those are duplicated across the two directories by design.

```bash
S=scenario_configs/tidepool_risk_v2/loop_risk_v2_0/loop_risk_v2_2_0_full/TLR-549/Simulation-Configuration-TLR-549_30_median_profile_v1.json

# noLoop cycle: DoNothingController, no Loop API, no dylib
python -m tidepool_data_science_simulator.diagramgen --scenario $S \
  --sequence-stage pre-noLoop_t1_median --sequence-cycle first-cycle \
  --output-dir .docs/architecture/TLR-549/noloop

# Loop cycle with physical activity configured (stage ID per the branch's config)
python -m tidepool_data_science_simulator.diagramgen --scenario $S \
  --sequence-stage post-Loop-WithMitigations_t1_median \
  --output-dir .docs/architecture/TLR-549/loop_pa

# Render. --render-config is required for a non-default --output-dir
python -m tidepool_data_science_simulator.diagramgen.render \
  --output-dir .docs/architecture/TLR-549/noloop --render-config .docs/architecture/render.json
python -m tidepool_data_science_simulator.diagramgen.render \
  --output-dir .docs/architecture/TLR-549/loop_pa --render-config .docs/architecture/render.json
```

Render config resolution: `--render-config` defaults to
`<output-dir>/render.json`, and the two `mermaid_config_*.json` files it names
are resolved relative to *that file's* directory. The new directories hold no
render config, so pass the pinned `.docs/architecture/render.json`. The allowlist
and exclusions default to `.docs/architecture/` regardless of `--output-dir`.

Once the stage-ID rename lands (`post-Loop-` to `post-Loop_`), use the
underscore form for `--sequence-stage`.

**Physical activity and the Loop figure.** The generator was not changed to
trace PA processing, and the allowlist is unchanged. Compared with the
committed TLR-552 figure, the `loop_pa` sequence has one additional
`SimpleMetabolismModel()` / `run` pair in `VirtualPatient.update`. That pair is
the `abs_insulin_amount != 0` branch (`patient.py`), which depends on delivered
insulin and not on activity: an ablation run of the short fixture with the
activity removed and a non-zero basal shows the same extra pair. So this is not
evidence that PA adds a participant or message. Why TLR-552's cycle does not
take that branch was not investigated.

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
4. `.docs/architecture/render.json` and the two `mermaid_config_*.json` files —
   the image digest in particular. Bumping it is a toolchain change with the
   same standing as a package version bump, and it changes the figures.
5. `tests/test_data/diagramgen/render_flowchart.mmd` and `render_sequence.mmd`,
   the render fixtures. Small structural twins of the two real figures,
   exercising `flowchart LR`, `sequenceDiagram` and the `<br/>` labels that make
   the PNG and SVG configs differ.

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

## Manifest schema

`manifest.json` declares `trset51-architecture-manifest/2`.

`/1` → `/2` is additive and narrow: `/2` may carry a `render` block describing
the pinned toolchain that produced the PNG and SVG. Everything `/1` carried is
unchanged and in the same place.

For a consumer: a **missing** `render` block — in a `/1` manifest, or in a `/2`
one that has not been through the render step — means *this figure was not
produced by a pinned toolchain*. It is not an error and should not be treated as
one. Inside the block, an unresolved value and an unattempted probe are kept
distinguishable: a version that could not be read is `null` with a reason under
`toolchain_errors`, while `toolchain_probe.attempted` says whether the probe ran
at all. Provenance that is absent and provenance that failed must not look
alike.
