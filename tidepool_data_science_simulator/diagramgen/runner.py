"""
Execution of the reference scenario under the tracer.

Execution model -- and why it is not the production one:

``Simulation`` subclasses ``multiprocessing.Process`` and ``run_simulations()``
always spawns. Under macOS/Python 3.12 ``spawn``, a parent-side tracer observes
nothing of the timestep, so the generator builds simulations with the real
``ScenarioParserV2`` and calls ``Simulation.run()`` **in-process**. It never
calls ``sim.start()`` and never goes through ``run_simulations()``. Every
cross-package edge then lives in one traceable process, and simulator execution
code needs no change whatsoever.

Two consumer-side attributes are set on the built objects, neither of which is a
change to simulator code:

``sim.multiprocess = False``
    ``build_sim_from_config()`` hard-codes ``multiprocess=True``. Left alone,
    ``Simulation.run()`` reconfigures the root logger onto a file under
    ``DATA_DIR/logs`` and pushes results to a ``multiprocessing.Queue`` -- both
    wrong for a generator that must write only to its own output directory.

``sim.controller.loop_algo_io_dir = <scratch dir>``
    ``SwiftLoopController`` writes ``loop_algo_input_*.json`` /
    ``loop_algo_output_*.json`` per control cycle, falling back to the process
    working directory when this is unset. A full reference run emits roughly
    1,100 of them; they go to a temporary directory that is deleted afterwards.

    That attribute can only be set *after* the simulation object exists, and
    ``Simulation.__init__`` already runs one control cycle via ``init()`` at
    t=0. Those first two files would land in the process working directory, so
    the whole traced block also runs with the working directory moved to the
    same scratch directory. Every path the generator itself uses is resolved to
    an absolute path before that happens.

The generator never calls ``save_df()``, never writes to ``DATA_DIR/results/...``
and never emits a TSV or ``<sim_id>.json``. Metrics return values are discarded
at the call site.
"""

import contextlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np

from tidepool_data_science_simulator.diagramgen.naming import Node
from tidepool_data_science_simulator.diagramgen.tracer import (
    CallTracer,
    PHASE_CONSTRUCTION,
    PHASE_METRICS,
    PHASE_RUN,
)

__all__ = ["RunResult", "run_traced"]

# The results-path metrics call lives inside ``run_simulations()``, which cannot
# be used here because it spawns. The generator reproduces that call so the edge
# is exercised, and attributes it to its real runtime call site.
_METRICS_CALL_SITE = Node("tidepool_data_science_simulator", "tidepool_data_science_simulator.run", "")
_METRICS_CALL_QUALNAME = "run_simulations"


class RunResult(object):
    """Everything a downstream emitter needs from one traced generation run."""

    def __init__(self, records, pointer_paths, executed_functions, loaded_modules, stages, scenario_path):
        self.records = records
        self.pointer_paths = pointer_paths
        self.executed_functions = executed_functions
        self.loaded_modules = loaded_modules
        self.stages = stages
        self.scenario_path = scenario_path


@contextlib.contextmanager
def _working_directory(path):
    """Temporarily move the process working directory, restoring it after."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _build_sims(parser_cls, scenario_path):
    """Build simulations with the real parser, muting its diagnostic printing.

    The parser prints an override diagnosis block to stdout on every config it
    resolves. That is useful when running risk simulations and pure noise here,
    and capturing it keeps the generator's own output readable.
    """
    parser = parser_cls(path_to_json_config=scenario_path)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        sims = parser.get_sims()
    return sims


def _compute_metrics(results_df):
    """Exercise the ``data-science-metrics`` edge exactly as ``run.py`` does.

    Return values are discarded: this call exists to put the edge on the data
    flow diagram, not to produce numbers. Nothing derived from it is written.
    """
    from tidepool_data_science_metrics.glucose.glucose import (
        blood_glucose_risk_index,
        lbgi_risk_score,
        percent_values_ge_70_le_180,
        percent_values_gt_180,
        percent_values_gt_250,
        percent_values_lt_40,
        percent_values_lt_54,
    )
    from tidepool_data_science_metrics.insulin.insulin import dka_index, dka_risk_score

    metrics_df = results_df[results_df["active"] == 1]
    clipped = np.array([min(401, max(1, value)) for value in metrics_df["bg"]])

    lbgi, _hbgi, _brgi = blood_glucose_risk_index(clipped)
    lbgi_risk_score(lbgi)
    dka_index_value = dka_index(metrics_df["iob"], metrics_df["sbr"].values[0])
    dka_risk_score(dka_index_value)
    percent_values_lt_40(clipped)
    percent_values_lt_54(clipped)
    percent_values_gt_180(clipped)
    percent_values_gt_250(clipped)
    percent_values_ge_70_le_180(clipped)


def run_traced(scenario_path, allowlist, scratch_dir=None):
    """Run every stage of ``scenario_path`` in-process under the tracer.

    Parameters
    ----------
    scenario_path: str
        Absolute path to the scenario configuration JSON.
    allowlist: Allowlist
        Capture filter, already loaded and validated.
    scratch_dir: str or None
        Where the Swift controller's per-cycle I/O files go. A temporary
        directory is created and removed when this is None.

    Returns
    -------
    RunResult
    """
    # Import here rather than at module scope: the tracer indexes loaded ctypes
    # libraries when it installs, so the Swift API module must already be in
    # ``sys.modules`` by then, and importing the parser is what pulls it in.
    from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2

    scenario_path = os.path.abspath(scenario_path)
    owns_scratch = scratch_dir is None
    scratch_dir = tempfile.mkdtemp(prefix="trset51-loopio-") if owns_scratch else scratch_dir

    stages = []
    try:
        with _working_directory(scratch_dir), CallTracer(allowlist) as tracer:
            tracer.begin_stage(stage=None, phase=PHASE_CONSTRUCTION)
            sims = _build_sims(ScenarioParserV2, scenario_path)

            # `get_sims()` inserts in `override_config` order, so iterating the
            # dict keeps the scenario's declared stage order and is
            # deterministic across runs.
            for stage_id, sim in sims.items():
                sim.multiprocess = False
                if hasattr(sim.controller, "loop_algo_io_dir"):
                    sim.controller.loop_algo_io_dir = scratch_dir

                tracer.begin_stage(stage=stage_id, phase=PHASE_RUN)
                sim.run()
                # Read the control-cycle count before the phase switch resets it.
                steps = int(tracer.timestep + 1) if tracer.timestep >= 0 else 0
                results_df = sim.get_results_df()

                tracer.begin_stage(
                    stage=stage_id,
                    phase=PHASE_METRICS,
                    attributed_caller=_METRICS_CALL_SITE,
                    attributed_caller_qualname=_METRICS_CALL_QUALNAME,
                )
                _compute_metrics(results_df)

                stages.append(
                    {
                        "sim_id": stage_id,
                        "controller": type(sim.controller).__name__,
                        "duration_hrs": sim.duration_hrs,
                        "steps": steps,
                    }
                )

        return RunResult(
            records=tracer.records,
            pointer_paths=tracer.pointer_paths,
            executed_functions=tracer.executed_functions,
            loaded_modules=frozenset(sys.modules),
            stages=stages,
            scenario_path=scenario_path,
        )
    finally:
        if owns_scratch:
            shutil.rmtree(scratch_dir, ignore_errors=True)
