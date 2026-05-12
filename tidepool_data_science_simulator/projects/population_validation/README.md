# Population Validation Experiments

This project area contains the reusable code for the Tidepool Simulator
population-validation workflow.

## Purpose

This workflow supports population-level validation of the Tidepool Simulator
against real-world data (RWD) under a paired strategy-comparison design.

The central question is not whether the simulator exactly reproduces each
individual user's glucose trace. The primary question is whether the simulator
produces a credible **within-user change in glycemic outcomes** when the dosing
strategy changes from temporary basal to automated bolus.

In practice, this means the workflow is organized around:

- a cohort of user-specific day-level scenario inputs
- paired simulated runs under temp-basal and autobolus strategies
- scenario variants that progressively remove or perturb input information
- cohort-level comparison of the resulting strategy-effect distributions

## What lives here

- `run_population_validation_config.py`
  - a configurable runner for scenario-based population-validation experiments
- `README.md`
  - this overview

## What does not live here

This repository is public, so real-user-derived experiment inputs are
intentionally excluded from version control. That includes:

- scenario JSON exports derived from real users
- observed cohort summary files derived from real users
- any intermediate files that may contain PHI or be close enough to real user
  data to create privacy risk

If you are working in a private environment, you may maintain local files such
as:

- `scenarios/`
- `scenario_tir.csv`

These are expected local inputs for private analyses and should not be committed
to the public repository.

## Expected private inputs

To reproduce the experiment in concept, a private environment should provide:

- one top-level scenario configuration file describing the experiment batch
- one or more user/day scenario JSON files
- user-specific pump settings and patient settings for each scenario
- carbohydrate event records for the active comparison window
- optional bolus event records, depending on the scenario definition
- observed CGM values for aligned observed-versus-simulated comparisons

The exact source and governance of those files depends on the study context and
must be handled outside the public repository.

## Experiment structure

The population-validation workflow is organized around:

- one top-level scenario configuration file
- a reusable pointer directory for shared config fragments
- one or more per-user override configurations
- optional observed CGM alignment for observed-versus-simulated comparisons

The runner supports both:

- single-scenario smoke tests
- larger batch experiments with saved resolved configs, plots, and summary
  outputs

## Conceptual run design

The intended experiment design is:

1. Build a cohort of user/day scenario files in a private environment.
2. For each user, run paired simulations under:
   - temp basal
   - autobolus
3. Repeat that pairing across a scenario ladder that changes input fidelity.
4. Save per-run outputs and cohort summaries.
5. Compare the simulated strategy effect to the observed strategy effect from
   the corresponding RWD cohort.

The scenario ladder typically follows this shape:

- Scenario 1: full-fidelity replay-style configuration
- Scenario 2: meals only, with bolus replay removed
- Scenario 3: unannounced meals, with bolus replay removed
- Scenario 4: Scenario 3 plus settings noise

The primary quantity of interest is usually the within-user strategy effect,
such as:

- `TIR_autobolus - TIR_temp_basal`

Supporting metrics can include:

- mean BG
- time above range
- time below range
- time above 250
- time below 54

## What the runner does

`run_population_validation_config.py` is a general experiment runner that:

- loads a top-level batch configuration
- resolves reusable config pointers through `ScenarioParserV2`
- optionally saves resolved per-simulation configs
- runs one or more simulations, optionally in parallel
- can enable standard simulator plots
- can align observed CGM onto simulation timestamps
- computes summary glucose quantities of interest for simulated and observed
  traces when observed CGM is available

The runner is intentionally generic enough to support:

- one-user smoke tests
- small-batch debugging
- larger cohort reruns

## Outputs

Depending on the flags used, a run can produce:

- simulation result tables
- resolved configs
- standard simulator plots
- observed-versus-simulated summary rows
- cohort-level summary tables derived from the run outputs

The public repo does not include private result artifacts, but the runner is
designed so those artifacts can be produced in a private analysis environment.

## Typical private usage

In a private environment with local scenario inputs available:

```bash
conda activate tidepool-data-science-simulator-swift
python tidepool_data_science_simulator/projects/population_validation/run_population_validation_config.py \
  path/to/top_level_population_validation_config.json \
  --limit 1
```

Once a smoke test passes, remove `--limit` and scale up to the intended batch
run.

## Reproducibility guidance

Someone with authorized access to the private cohort data should be able to
reproduce the experiment in concept by:

1. preparing the user/day scenario files and any top-level batch config
2. selecting the controller strategy conditions to compare
3. running the batch through `run_population_validation_config.py`
4. aggregating the resulting per-user metrics into the intended cohort-level
   summaries
5. comparing simulated strategy-effect distributions with the observed cohort

This README is intentionally focused on that conceptual reproducibility without
including any real-user-derived files or paths.

## Notes for future contributors

If you are extending this workflow, keep the public repo focused on:

- reusable runner code
- generic documentation
- public-safe helpers

Keep any real-user-derived inputs, outputs, and study-specific analysis
artifacts in an appropriate private environment.
