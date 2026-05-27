"""
Run a single JSON scenario config through the simulator.

Usage:
    python -m tidepool_data_science_simulator.projects.risk.run_single_scenario \\
        path/to/scenario.json [--save-dir DIR] [--num-procs N] \\
        [--save-format {tsv,parquet}] [--no-save]
"""

import argparse
import datetime
import json
import logging
import os
import sys

import matplotlib.pyplot as plt

from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2
from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.run import run_simulations
from tidepool_data_science_simulator.utils import DATA_DIR, PROJECT_ROOT_DIR
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results


SCENARIO_JSON = os.path.join(
    PROJECT_ROOT_DIR,
    "scenario_configs/tidepool_risk_v2/loop_risk_v2_0/TLR-000-base/"
    "Simulation-Configuration-TLR-000-base_median_profile_v1.json",
)
SCENARIO_JSON = '/Users/mconn/Downloads/rwd_user_0050_day_01.json'

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a single JSON scenario config.")
    parser.add_argument(
        "scenario_json",
        nargs="?",
        default=SCENARIO_JSON,
        help="Path to the JSON scenario config file. Defaults to SCENARIO_JSON.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Output directory. Defaults to <DATA_DIR>/results/single_scenario/<scenario_stem>_<timestamp>/.",
    )
    parser.add_argument("--num-procs", type=int, default=1, help="Multiprocessing worker count.")
    parser.add_argument(
        "--save-format",
        choices=("tsv", "parquet"),
        default="tsv",
        help="Per-simulation result file format.",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not write result files.")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving the results plot.")
    return parser.parse_args(argv)


def _overlay_actual_cgm(ax_bg, cfg, full_results):
    actual_cgm = cfg.get("actual_cgm")
    if not actual_cgm:
        return
    first_ts = next(iter(full_results.values())).index[0]
    reference = datetime.datetime(first_ts.year, first_ts.month, first_ts.day, 12, 0, 0)
    times_hours = []
    values = []
    for entry in actual_cgm:
        t = datetime.datetime.strptime(entry["time"], "%m/%d/%Y %H:%M:%S")
        times_hours.append((t - reference).total_seconds() / 3600.0)
        values.append(entry["value"])
    ax_bg.plot(times_hours, values, color="tab:green", linewidth=1.0,
               alpha=0.8, label="actual cgm", zorder=5)
    ax_bg.legend(prop={"size": 6}, loc="upper right")


def default_save_dir(scenario_path):
    stem = os.path.splitext(os.path.basename(scenario_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    return os.path.join(DATA_DIR, "results", "single_scenario", f"{stem}_{timestamp}")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    scenario_path = os.path.abspath(args.scenario_json)
    if not os.path.isfile(scenario_path):
        logger.error("Scenario JSON not found: %s", scenario_path)
        return 2

    save_dir = os.path.abspath(args.save_dir) if args.save_dir else default_save_dir(scenario_path)
    os.makedirs(save_dir, exist_ok=True)

    logger.info("Scenario: %s", scenario_path)
    logger.info("Save dir: %s", save_dir)

    with open(scenario_path) as f:
        cfg = json.load(f)

    if {"metadata", "base_config", "override_config"} <= cfg.keys():
        parser = ScenarioParserV2(path_to_json_config=scenario_path)
        sims = parser.get_sims(override_json_save_dir=save_dir)
    else:
        parser = ScenarioParserV2()
        sim_start_time, duration_hrs, vp, controller = parser.build_components_from_config(cfg)
        sim = Simulation(
            sim_start_time,
            duration_hrs=duration_hrs,
            virtual_patient=vp,
            controller=controller,
            multiprocess=True,
            sim_id=cfg["sim_id"],
        )
        sims = {sim.sim_id: sim}

    full_results, summary_df = run_simulations(
        sims,
        save_dir=save_dir,
        save_results=not args.no_save,
        num_procs=args.num_procs,
        save_format=args.save_format,
    )

    if not args.no_plot:
        plot_path = os.path.join(save_dir, "sim_results.png")
        fig, ax = plot_sim_results(full_results, save=False, save_path=plot_path)
        _overlay_actual_cgm(ax[0], cfg, full_results)
        fig.savefig(plot_path)
        logger.info("Plot saved: %s", plot_path)
        plt.show()

    logger.info("Done. Ran %d simulation(s).", len(sims))
    return 0


if __name__ == "__main__":
    sys.exit(main())
