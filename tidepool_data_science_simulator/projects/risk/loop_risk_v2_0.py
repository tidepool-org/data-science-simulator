__author__ = "Cameron Summers"

"""
This file builds and runs the Tidepool Loop Risk simulations. Scenarios in JSON format
specify how to build the simulations. 
"""

import os
import datetime
import pandas as pd
import subprocess
import json
import logging
import sys
import traceback

# CRITICAL: Configure logging BEFORE importing project modules
# Some imported modules may configure logging, and basicConfig only works on first call
from tidepool_data_science_simulator.utils import PROJECT_ROOT_DIR, DATA_DIR

LOGS_DIR = os.path.join(DATA_DIR, 'logs')
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

LOG_FILE_PATH = os.path.join(LOGS_DIR, 'simulator_debug.log')

# Force reconfiguration of root logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=LOG_FILE_PATH,
    filemode='w',
    force=True  # Python 3.8+ forces reconfiguration
)
print(f"Debug logging enabled. Log file: {LOG_FILE_PATH}")

# Now import other project modules
from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.utils import timing
from tidepool_data_science_simulator.run import run_simulations, SimulationRunError


THIS_DIR = os.path.abspath(__file__)
TIDEPOOL_RISK_SCENARIOS_DIR = os.path.join(PROJECT_ROOT_DIR, "scenario_configs/tidepool_risk_v2/loop_risk_v2_0/loop_risk_v2_2_0_full")

RESULTS_SAVE_DIR = os.path.join(DATA_DIR, "results/tidepool_loop_risk_v2_0")


FAILURES_LOG_FILENAME = "failures.log"

EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_NONE_SUCCEEDED = 2


def make_failure_record(risk_name, scenario_json_name, stage, exc, sim_id=None):
    """
    Describe an exception caught while handling one scenario JSON. Call from the except block.

    :param stage: one of 'parse', 'run', 'metrics', 'plot', 'save'
    :return: dict with the keys written to metadata.json, plus 'traceback' for failures.log
    """
    return {
        "tlr": risk_name,
        "json": scenario_json_name,
        "sim_id": sim_id,
        "stage": stage,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


@timing
def build_risk_sim_generator(scenario_json_filepath, override_config_save_dir=None, target_risk_dir=None,
                             failures=None):
    """
    Build a generator of suites of related simulations for processing.

    :param target_risk_dir: substring to filter to a single TLR-* directory (e.g. "TLR-HF").
        If None, all TLR-* directories under scenario_json_filepath are processed.
    :param failures: list to append a 'parse' failure record to for each JSON that cannot be
        parsed into sims; the generator then carries on with the next JSON. If None, a parse
        error is raised.
    """
    print(f"build_risk_sim_generator called with scenario_json_filepath: {scenario_json_filepath}")
    print(f"TIDEPOOL_RISK_SCENARIOS_DIR: {TIDEPOOL_RISK_SCENARIOS_DIR}")

    # List TLR-* directories directly in scenario_json_filepath (one flat level)
    risk_dirs = [risk_dir for risk_dir in os.listdir(scenario_json_filepath)
                 if os.path.isdir(os.path.join(scenario_json_filepath, risk_dir)) and "TLR-" in risk_dir]
    print(f"Found risk directories: {risk_dirs}")

    for risk_dir_name in risk_dirs:
        print(f"Processing risk directory: {risk_dir_name}")
        if target_risk_dir is not None and target_risk_dir not in risk_dir_name:
            print(f"Skipping {risk_dir_name}")
            continue
        print(f"Processing: {risk_dir_name}")

        risk_dir_path = os.path.join(scenario_json_filepath, risk_dir_name)
        try:
            scenario_json_filenames = [filename for filename in os.listdir(risk_dir_path) if ".json" in filename]
        except Exception as e:
            if failures is None:
                raise
            logging.warning("Failed to list %s: %s", risk_dir_name, e)
            failures.append(make_failure_record(risk_dir_name, None, "parse", e))
            continue
        print(f"JSON files found in {risk_dir_name}: {scenario_json_filenames}")

        for scenario_json_name in scenario_json_filenames:
            print(f"Processing JSON file: {scenario_json_name}")
            scenario_json_path = os.path.join(risk_dir_path, scenario_json_name)
            try:
                parser = ScenarioParserV2(path_to_json_config=scenario_json_path)
                print(f"Parsing: {scenario_json_path}")
                sim_suite = parser.get_sims(override_json_save_dir=override_config_save_dir)
            except Exception as e:
                if failures is None:
                    raise
                logging.warning("Failed to parse %s/%s: %s", risk_dir_name, scenario_json_name, e)
                failures.append(make_failure_record(risk_dir_name, scenario_json_name, "parse", e))
                continue
            print(f"Yielding: {risk_dir_name}, {scenario_json_name}")
            yield risk_dir_name, scenario_json_name, sim_suite

    print("build_risk_sim_generator completed")


def create_save_dir(results_dir=None):

    timestamp = get_timestamp()
    run_save_dir = os.path.join(results_dir or RESULTS_SAVE_DIR, "Risk_Run_{}".format(timestamp))
    os.makedirs(run_save_dir)
    return run_save_dir


def get_timestamp():

    return datetime.datetime.now().isoformat()


def _metadata_in_order(timestamp, risk_run_metadata, failures):
    """
    Build metadata.json content in its documented order: timestamp, the run-level fields
    in the order they were set, failures_log, and failures last.
    """
    run_level = {key: value for key, value in risk_run_metadata.items()
                 if key not in ("timestamp", "failures_log", "failures")}
    metadata = {"timestamp": timestamp}
    metadata.update(run_level)
    metadata["failures_log"] = FAILURES_LOG_FILENAME
    metadata["failures"] = [
        {key: value for key, value in failure.items() if key != "traceback"} for failure in failures
    ]
    return metadata


def write_metadata(run_save_dir, risk_run_metadata, failures, timestamp=None):
    """
    Write metadata.json for a run. failures is always written, as an empty list when nothing failed.
    """
    metadata = _metadata_in_order(timestamp or get_timestamp(), risk_run_metadata, failures)
    with open(os.path.join(run_save_dir, "metadata.json"), "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)


def write_failures_log(run_save_dir, failures):
    """
    Write failures.log with the full traceback of every failure. Written on every run,
    regardless of the debug-logging setting; empty of entries when nothing failed.
    """
    with open(os.path.join(run_save_dir, FAILURES_LOG_FILENAME), "w") as log_file:
        log_file.write("{} failure(s)\n".format(len(failures)))
        for failure in failures:
            log_file.write("\n{}\nTLR: {}\nJSON: {}\nsim_id: {}\nstage: {}\n{}: {}\n{}".format(
                "=" * 80, failure.get("tlr"), failure.get("json"), failure.get("sim_id"),
                failure["stage"], failure["exception_type"], failure["message"], failure["traceback"]))


def write_run_summary(all_risk_results, run_save_dir, risk_run_metadata, failures=()):
    """
    Write one Risk_Results_<timestamp>.csv and one metadata.json for a run.

    The CSV filename and the metadata timestamp share a single get_timestamp() call.
    A failed concat or save is logged and raised; the CSV is written before
    metadata.json so a failure never leaves metadata.json without its results.

    :return: path of the CSV written
    """
    try:
        all_risk_results_df = pd.concat(all_risk_results)
        print(f"Final dataframe shape: {all_risk_results_df.shape}")

        timestamp = get_timestamp()

        csv_path = os.path.join(run_save_dir, "Risk_Results_{}.csv".format(timestamp))
        all_risk_results_df.to_csv(csv_path)
        print("Results saved to CSV")

        write_metadata(run_save_dir, risk_run_metadata, failures, timestamp=timestamp)
    except Exception:
        logging.exception("Failed to write run summary to %s", run_save_dir)
        raise

    return csv_path


def run_risk_scenarios(scenarios_dir, run_save_dir, failures, num_procs=4, target_risk_dir=None):
    """
    Run every scenario JSON under scenarios_dir, each on its own, and keep going past failures.

    A failure at any stage of one JSON is appended to failures and the run carries on with
    the next JSON. A JSON with any failure contributes no rows to the returned results.

    :return: (all_risk_results, scenarios_attempted, scenarios_succeeded)
    """
    all_risk_results = []
    attempted = 0
    succeeded = 0

    sim_suite_generator = build_risk_sim_generator(scenarios_dir, override_config_save_dir=run_save_dir,
                                                   target_risk_dir=target_risk_dir, failures=failures)
    for risk_name, scenario_json_name, sim_suite in sim_suite_generator:
        attempted += 1
        failures_before = len(failures)
        try:
            risk_result_dirpath = os.path.join(run_save_dir, risk_name)
            if not os.path.exists(risk_result_dirpath):
                os.mkdir(risk_result_dirpath)

            # Create subdirectory for Loop algorithm I/O files
            # This should always happen, not just when the risk directory is created
            loop_algo_io_dir = os.path.join(risk_result_dirpath, "loop_algo_io")
            os.makedirs(loop_algo_io_dir, exist_ok=True)

            # Set the directory on each simulation's controller
            for sim_id, sim in sim_suite.items():
                if hasattr(sim.controller, 'loop_algo_io_dir'):
                    sim.controller.loop_algo_io_dir = loop_algo_io_dir

            full_results_dict, summary_results_df = run_simulations(sim_suite,
                                                                    save_dir=risk_result_dirpath,
                                                                    save_results=True,
                                                                    num_procs=num_procs,
                                                                    name=scenario_json_name)
        except SimulationRunError as e:
            for failure in e.failures:
                failures.append({"tlr": risk_name, "json": scenario_json_name, **failure})
            print(f"Failed: {risk_name}, {scenario_json_name}: {e}")
            continue
        except Exception as e:
            failures.append(make_failure_record(risk_name, scenario_json_name, "run", e))
            print(f"Failed: {risk_name}, {scenario_json_name}: {e}")
            continue

        try:
            summary_results_df["scenario_name"] = scenario_json_name
            summary_results_df["risk_name"] = risk_name

            # Save figure
            figure_filepath = os.path.join(risk_result_dirpath,
                                           "{}_{}_{}.png".format(risk_name, scenario_json_name, get_timestamp()))
            plot_sim_results(full_results_dict, save=True, save_path=figure_filepath)
        except Exception as e:
            failures.append(make_failure_record(risk_name, scenario_json_name, "plot", e))
            print(f"Failed: {risk_name}, {scenario_json_name}: {e}")
            continue

        all_risk_results.append(summary_results_df)
        succeeded += 1
        print(f"Processed: {risk_name}, {scenario_json_name}")

    # JSONs that failed to parse never reach the loop above but were still attempted
    parse_failures = {(f["tlr"], f["json"]) for f in failures if f["stage"] == "parse"}
    return all_risk_results, attempted + len(parse_failures), succeeded


def exit_code_for(scenarios_attempted, scenarios_succeeded, n_failures):
    """0 if everything succeeded, 1 if some failed with partial results, 2 if nothing succeeded."""
    if scenarios_succeeded == 0:
        return EXIT_NONE_SUCCEEDED
    if n_failures:
        return EXIT_PARTIAL
    return EXIT_OK


def failed_tlr_counts(failures):
    """TLR directory name to the number of JSONs that failed in it."""
    failed_jsons = {(f["tlr"], f["json"]) for f in failures if f.get("tlr")}
    counts = {}
    for tlr, _ in failed_jsons:
        counts[tlr] = counts.get(tlr, 0) + 1
    return counts


def main(scenarios_dir=None, results_dir=None, num_procs=4):
    """
    Run the risk scenarios and write the results, metadata.json, failures.log and severity summaries.

    :return: process exit code. 0 everything succeeded, 1 some failures with partial results,
        2 nothing succeeded or a fatal error.
    """
    scenarios_dir = scenarios_dir or TIDEPOOL_RISK_SCENARIOS_DIR
    failures = []
    risk_run_metadata = {}

    try:
        # Create place to save results
        run_save_dir = create_save_dir(results_dir)
    except Exception:
        logging.exception("Could not create the results directory")
        traceback.print_exc()
        return EXIT_NONE_SUCCEEDED

    try:
        all_risk_results, attempted, succeeded = run_risk_scenarios(
            scenarios_dir, run_save_dir, failures, num_procs=num_procs)
    except Exception:
        # Fatal before results could be gathered, e.g. the scenarios directory cannot be listed
        logging.exception("Fatal error while running the scenarios")
        traceback.print_exc()
        return EXIT_NONE_SUCCEEDED

    print(f"Total scenarios processed: {len(all_risk_results)}")

    risk_run_metadata["scenarios_attempted"] = attempted
    risk_run_metadata["scenarios_succeeded"] = succeeded

    exit_code = exit_code_for(attempted, succeeded, len(failures))
    try:
        write_failures_log(run_save_dir, failures)
        if all_risk_results:
            write_run_summary(all_risk_results, run_save_dir, risk_run_metadata, failures)
        else:
            write_metadata(run_save_dir, risk_run_metadata, failures)
    except Exception:
        # The write functions have already logged; per-sim results are on disk but the run's summary is not
        traceback.print_exc()
        exit_code = EXIT_NONE_SUCCEEDED

    if succeeded:
        write_severity_summaries(run_save_dir, risk_run_metadata, failures)

    print(f"{succeeded} of {attempted} scenarios succeeded; {attempted - succeeded} failed; "
          f"see {os.path.join(run_save_dir, FAILURES_LOG_FILENAME)}")
    return exit_code


def write_severity_summaries(run_save_dir, risk_run_metadata, failures):
    """
    Write the RTF severity summaries, as the GUI run does, skipping every TLR that has a failure.

    post_processing/ is not part of the installed package, so it is put on sys.path here.
    A failure is reported, not swallowed, but does not undo the simulation results saved above.
    The skipped TLRs are recorded in metadata.json.
    """
    _post_processing_dir = os.path.join(PROJECT_ROOT_DIR, "post_processing")
    if _post_processing_dir not in sys.path:
        sys.path.insert(0, _post_processing_dir)
    from create_severity_summary import process_results_directory, SeveritySummaryError

    try:
        summary_result = process_results_directory(run_save_dir, failed_tlrs=failed_tlr_counts(failures))
        for tlr_dir, reason in summary_result.skipped:
            print(f"Severity summary skipped {tlr_dir}: {reason}")
        print(f"Wrote {len(summary_result.written)} severity summary document(s); "
              f"skipped {len(summary_result.skipped)} directory(ies).")
        with open(os.path.join(run_save_dir, "metadata.json")) as metadata_file:
            timestamp = json.load(metadata_file)["timestamp"]
        risk_run_metadata["severity_summaries_skipped"] = [
            {"tlr": os.path.basename(tlr_dir), "reason": reason} for tlr_dir, reason in summary_result.skipped
        ]
        write_metadata(run_save_dir, risk_run_metadata, failures, timestamp=timestamp)
    except SeveritySummaryError as e:
        print(f"Severity summary not written: {e}")


if __name__ == "__main__":
    sys.exit(main())
