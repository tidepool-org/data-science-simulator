# data-science-simulator

Simulation framework for insulin algorithm testing. Supports FDA risk analysis and Tidepool Loop performance evaluation.

## Environment

- Python via conda; environments defined in `conda-environment.yml` (main), `conda-environment-dev.yml`, `conda-environment-swift.yml`.
- Tests: `pytest` (config in `pytest.ini`, `tox.ini`).

## Layout

- `tidepool_data_science_simulator/` — package source
  - `projects/` — top-level experiment subprojects (e.g. `risk/`, `icgm/`, `loop_guardrails/`, `insulin_algorithm_testing_framework/`)
- `scenario_configs/` — scenario JSON/YAML (FDA scenarios under `tidepool_risk_v2/loop_risk_v2_0`)
- `tests/` — pytest suite
- `notebooks/`, `reports/`, `results/` — exploratory + output artifacts

## Conventions

- Simulation results use a streaming Parquet writer; both TSV and Parquet outputs are supported.
- Metrics calculation from combined parquet files runs in parallel via `calculate_metrics_from_parquet` (multiprocessing.Pool) in `insulin_algorithm_testing_framework/core/metrics_calculator.py`.
- Controllers selected via config: `"controller": {"id": "pyloopkit_v1"}` or `{"id": "swift"}` (SwiftLoopController requires LoopAlgorithmToPython dylib on macOS).
- PHI must never be committed; `.gitignore` enforces this.

## Entry points

- `projects/risk/run_single_scenario.py` — run a single JSON scenario; supports both wrapper (metadata/base_config/override_config) and flat configs; overlays `actual_cgm` on the BG plot when present.
- `projects/insulin_algorithm_testing_framework/experiments/run_icgm_510k_analysis.py` — iCGM 510(k) sweep; prefers parquet output and parallel metrics processing.

## Databricks

- `databricks.yml` defines the asset bundle (target `dev` → `tidepool-dev.cloud.databricks.com`); `.databricks/` is git-ignored.

## Related repos (sibling checkouts)

- `PyLoopKit`, `LoopAlgorithm`, `LoopAlgorithmToPython` — algorithm implementations consumed here
- `data-science-models`, `data-science-metrics` — model and metric libraries
- `data-science-insights` — FDA 510(k) RWD analyses (Databricks)
