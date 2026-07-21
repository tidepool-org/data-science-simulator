"""
Single validated entry point for running a Tidepool Loop risk assessment
in-process. GUI and other consumers (ad hoc scripts) call only this module;
none of them reach into simulator internals directly.

Wraps loop_risk_v2_0.build_risk_sim_generator + run_simulations +
severity_model.build_assessment. Library-browsing/resolution (listing
available config collections, mapping a chosen name to its path) is
deliberately NOT handled here -- it lives in the GUI's view layer, so a
future "configure parameters directly" mode can hand this module a
freshly-written temp directory with no changes required here.
"""

import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from tidepool_data_science_simulator.utils import PROJECT_ROOT_DIR
from tidepool_data_science_simulator.projects.risk.loop_risk_v2_0 import (
    build_risk_sim_generator,
    create_save_dir,
    get_timestamp,
)
from tidepool_data_science_simulator.run import run_simulations
from tidepool_data_science_simulator.validation.config_validator import ConfigValidator
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results

_POST_PROCESSING_DIR = os.path.join(PROJECT_ROOT_DIR, "post_processing")
if _POST_PROCESSING_DIR not in sys.path:
    sys.path.insert(0, _POST_PROCESSING_DIR)

from severity_model import build_assessment, SeverityAssessment  # noqa: E402


@dataclass
class ConfigValidationResult:
    """Aggregated (is_valid, errors, warnings) across every config file in scope."""
    is_valid: bool
    errors_by_file: dict
    warnings_by_file: dict


@dataclass
class RiskDirRunResult:
    """Outcome for one TLR-* directory. assessment is None when build_assessment
    found no usable data for it -- surfaced explicitly, never silently dropped.
    png_paths has one entry per scenario config file run in this directory."""
    risk_dir_name: str
    assessment: Optional[SeverityAssessment]
    png_paths: list = field(default_factory=list)


@dataclass
class RunResult:
    save_dir: str
    risk_dir_results: list = field(default_factory=list)
    cancelled: bool = False


def _find_pointer_object_dir(config_dir: str, max_levels_up: int = 4) -> Optional[str]:
    """Walk upward from config_dir looking for a directory containing a 'reusable'
    subdir, without hardcoding the library's nesting depth. Returns None if not
    found within max_levels_up (ConfigValidator degrades to skipping reference
    resolution in that case, matching its own no-pointer-dir behavior)."""
    current = os.path.abspath(config_dir)
    for _ in range(max_levels_up):
        if os.path.isdir(os.path.join(current, "reusable")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def _list_target_risk_dirs(config_dir: str, target_risk_dir: Optional[str]) -> list:
    """Mirrors build_risk_sim_generator's own TLR-* filtering so the total count
    for progress reporting is known before the generator starts."""
    return [
        d for d in sorted(os.listdir(config_dir))
        if os.path.isdir(os.path.join(config_dir, d))
        and "TLR-" in d
        and (target_risk_dir is None or target_risk_dir in d)
    ]


def validate_config_dir(config_dir: str, target_risk_dir: Optional[str] = None) -> ConfigValidationResult:
    """Validate every scenario config file that would actually run, given
    target_risk_dir. Call this and check is_valid before run_risk_assessment."""
    pointer_object_dir = _find_pointer_object_dir(config_dir)
    validator = ConfigValidator(pointer_object_dir=pointer_object_dir)

    if target_risk_dir is None:
        results = validator.validate_directory(config_dir, recursive=True)
    else:
        results = {}
        for risk_dir_name in _list_target_risk_dirs(config_dir, target_risk_dir):
            results.update(
                validator.validate_directory(os.path.join(config_dir, risk_dir_name), recursive=True)
            )

    errors_by_file = {path: errs for path, (_, errs, _) in results.items() if errs}
    warnings_by_file = {path: warns for path, (_, _, warns) in results.items() if warns}
    is_valid = all(is_valid for is_valid, _, _ in results.values()) if results else True
    return ConfigValidationResult(is_valid, errors_by_file, warnings_by_file)


def run_risk_assessment(
    config_dir: str,
    target_risk_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> RunResult:
    """
    Run a risk assessment over every TLR-* directory in config_dir (or just
    target_risk_dir, if given), returning one RiskDirRunResult per directory.

    Raises ValueError if any in-scope config file has validation errors --
    callers should still call validate_config_dir directly to surface errors
    in the UI before offering to run; this is a guard for callers that skip that.
    """
    validation_result = validate_config_dir(config_dir, target_risk_dir)
    if not validation_result.is_valid:
        raise ValueError(
            f"Cannot run: {len(validation_result.errors_by_file)} config file(s) "
            "have validation errors. Call validate_config_dir first and resolve them."
        )

    save_dir = create_save_dir()
    timestamp = get_timestamp()
    total = len(_list_target_risk_dirs(config_dir, target_risk_dir))

    risk_dir_results = []
    completed_count = 0
    previous_risk_dir_name = None
    current_png_paths = []

    def _finalize(risk_dir_name, png_paths):
        nonlocal completed_count
        risk_result_dirpath = os.path.join(save_dir, risk_dir_name)
        assessment = build_assessment(risk_result_dirpath, timestamp)
        risk_dir_results.append(RiskDirRunResult(risk_dir_name, assessment, png_paths))
        completed_count += 1
        if progress_callback is not None:
            progress_callback(completed_count, total, risk_dir_name)

    sim_suite_generator = build_risk_sim_generator(
        config_dir, override_config_save_dir=save_dir, target_risk_dir=target_risk_dir
    )

    for risk_dir_name, scenario_json_name, sim_suite in sim_suite_generator:
        if previous_risk_dir_name is not None and risk_dir_name != previous_risk_dir_name:
            # Finalize the just-completed dir BEFORE checking cancellation --
            # otherwise a cancel landing exactly on this transition would
            # silently drop a risk dir that had already fully finished.
            _finalize(previous_risk_dir_name, current_png_paths)
            current_png_paths = []

        if cancel_event is not None and cancel_event.is_set():
            return RunResult(save_dir=save_dir, risk_dir_results=risk_dir_results, cancelled=True)

        risk_result_dirpath = os.path.join(save_dir, risk_dir_name)
        os.makedirs(risk_result_dirpath, exist_ok=True)

        loop_algo_io_dir = os.path.join(risk_result_dirpath, "loop_algo_io")
        os.makedirs(loop_algo_io_dir, exist_ok=True)
        for sim in sim_suite.values():
            if hasattr(sim.controller, "loop_algo_io_dir"):
                sim.controller.loop_algo_io_dir = loop_algo_io_dir

        full_results_dict, _ = run_simulations(
            sim_suite,
            save_dir=risk_result_dirpath,
            save_results=True,
            num_procs=4,
            name=scenario_json_name,
        )

        figure_filepath = os.path.join(
            risk_result_dirpath, f"{risk_dir_name}_{scenario_json_name}_{get_timestamp()}.png"
        )
        plot_sim_results(full_results_dict, save=True, save_path=figure_filepath)
        current_png_paths.append(figure_filepath)

        previous_risk_dir_name = risk_dir_name

    if previous_risk_dir_name is not None:
        _finalize(previous_risk_dir_name, current_png_paths)

    return RunResult(save_dir=save_dir, risk_dir_results=risk_dir_results, cancelled=False)
