import argparse
import datetime
import json
import os
import types

import numpy as np
import pandas as pd

from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2
from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.run import run_simulations
from tidepool_data_science_simulator.utils import PROJECT_ROOT_DIR
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_metrics.glucose.glucose import (
    blood_glucose_risk_index,
    percent_values_ge_70_le_180,
    percent_values_gt_180,
    percent_values_gt_250,
    percent_values_lt_40,
    percent_values_lt_54,
)


def make_results_dir(config_path, output_dir=None):
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_T_%H_%M_%S")
    config_stem = os.path.splitext(os.path.basename(config_path))[0]

    if output_dir is None:
        output_dir = os.path.join(
            PROJECT_ROOT_DIR,
            "data",
            "results",
            "population_validation",
            config_stem,
            timestamp,
        )

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser("run_population_validation_config")
    parser.add_argument("config_path", help="Path to top-level scenario JSON config")
    parser.add_argument(
        "--pointer-dir",
        default=os.path.join(PROJECT_ROOT_DIR, "scenario_configs"),
        help="Root directory used to resolve reusable.* pointers",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for results. Defaults to repo-local data/results/population_validation/<config>/<timestamp>",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N override configs for smoke testing",
    )
    parser.add_argument(
        "--num-procs",
        type=int,
        default=1,
        help="Number of simulations to run in parallel",
    )
    parser.add_argument(
        "--save-resolved-configs",
        action="store_true",
        help="Save resolved per-sim override configs alongside results for inspection",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save the standard simulator over-time plot PNG(s) into the results directory",
    )
    parser.add_argument(
        "--accept-recommended-bolus",
        action="store_true",
        help="Allow the virtual patient to accept recommended manual boluses from the controller",
    )
    parser.add_argument(
        "--recommendation-accept-prob",
        type=float,
        default=1.0,
        help="Acceptance probability to use with --accept-recommended-bolus",
    )
    parser.add_argument(
        "--min-bolus-rec-threshold",
        type=float,
        default=0.0,
        help="Minimum recommended bolus size to accept with --accept-recommended-bolus",
    )
    parser.add_argument(
        "--recommendation-meal-attention-minutes",
        type=float,
        default=1e12,
        help="Meal-attention window used with --accept-recommended-bolus",
    )
    return parser.parse_args()


def is_top_level_batch_config(config_obj):
    return all(key in config_obj for key in ("metadata", "base_config", "override_config"))


def normalize_controller_settings_for_swift(sim_config):
    controller = sim_config.get("controller")
    if controller is None:
        return sim_config

    controller_id = controller.get("id", "")
    if "swift" not in controller_id:
        return sim_config

    settings = controller.get("settings", {})

    # Real-data exports use this name, while the current Swift controller expects
    # the Tidepool risk config name.
    if "glucose_safety_limit" in settings and "suspend_threshold" not in settings:
        settings["suspend_threshold"] = settings["glucose_safety_limit"]

    # The Swift controller also expects these settings to exist.
    settings.setdefault("partial_application_factor", 0.0)
    settings.setdefault("use_mid_absorption_isf", False)

    return sim_config


def attach_observed_cgm(result_df, sim_config):
    actual_cgm = sim_config.get("actual_cgm")
    if not actual_cgm:
        return result_df

    observed_index = [
        datetime.datetime.strptime(entry["time"], "%m/%d/%Y %H:%M:%S")
        for entry in actual_cgm
    ]
    observed_values = [entry["value"] for entry in actual_cgm]
    observed_series = pd.Series(observed_values, index=observed_index, name="observed_cgm")
    result_df["observed_cgm"] = observed_series.reindex(result_df.index)
    # Keep a second comparator series aligned to the simulation timestamps so
    # observed-vs-sim overlays and downstream metric calculations are denser and
    # easier to interpret than exact timestamp matches alone.
    result_df["observed_cgm_aligned"] = observed_series.reindex(
        result_df.index,
        method="nearest",
        tolerance=pd.Timedelta("5min"),
    )
    return result_df


def compute_glucose_qois(glucose_values):
    values = np.asarray(glucose_values, dtype=float)
    values = values[~np.isnan(values)]
    values = np.array([min(401, max(1, val)) for val in values])

    if len(values) == 0:
        return {}

    lbgi, hbgi, brgi = blood_glucose_risk_index(values)
    mean_bg = float(np.mean(values))
    std_bg = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    return {
        "n_points": int(len(values)),
        "mean_bg": mean_bg,
        "median_bg": float(np.median(values)),
        "std_bg": std_bg,
        "cv_bg": float(std_bg / mean_bg) if mean_bg else np.nan,
        "lbgi": float(lbgi),
        "hbgi": float(hbgi),
        "brgi": float(brgi),
        "percent_lt_40": float(percent_values_lt_40(values)),
        "percent_lt_54": float(percent_values_lt_54(values)),
        "percent_in_70_180": float(percent_values_ge_70_le_180(values)),
        "percent_gt_180": float(percent_values_gt_180(values)),
        "percent_gt_250": float(percent_values_gt_250(values)),
    }


def compute_observed_vs_sim_qoi_row(sim_id, result_df):
    if "observed_cgm_aligned" not in result_df.columns:
        return None

    active_df = result_df[result_df["active"] == 1].copy()
    paired_df = active_df[active_df["observed_cgm_aligned"].notna()].copy()
    if paired_df.empty:
        return None

    sim_metrics = compute_glucose_qois(paired_df["bg_sensor"])
    observed_metrics = compute_glucose_qois(paired_df["observed_cgm_aligned"])
    raw_match_count = (
        int(active_df["observed_cgm"].notna().sum())
        if "observed_cgm" in active_df.columns
        else 0
    )

    row = {
        "sim_id": sim_id,
        "active_start": paired_df.index.min().isoformat(),
        "active_end": paired_df.index.max().isoformat(),
        "active_row_count": int(len(active_df)),
        "paired_row_count": int(len(paired_df)),
        "raw_observed_match_count": raw_match_count,
        "paired_fraction_of_active": float(len(paired_df) / len(active_df)),
    }

    for prefix, metrics in [("sim", sim_metrics), ("observed", observed_metrics)]:
        row.update({f"{prefix}_{key}": value for key, value in metrics.items()})

    row["delta_mean_bg_sim_minus_observed"] = (
        row["sim_mean_bg"] - row["observed_mean_bg"]
    )
    row["delta_median_bg_sim_minus_observed"] = (
        row["sim_median_bg"] - row["observed_median_bg"]
    )
    row["delta_percent_in_70_180_sim_minus_observed"] = (
        row["sim_percent_in_70_180"] - row["observed_percent_in_70_180"]
    )
    row["delta_percent_lt_54_sim_minus_observed"] = (
        row["sim_percent_lt_54"] - row["observed_percent_lt_54"]
    )
    row["delta_percent_gt_180_sim_minus_observed"] = (
        row["sim_percent_gt_180"] - row["observed_percent_gt_180"]
    )

    return row


def main():
    args = parse_args()
    results_dir = make_results_dir(args.config_path, args.output_dir)
    config_obj = json.load(open(args.config_path))

    parser = ScenarioParserV2(pointer_object_dir=args.pointer_dir)

    if args.accept_recommended_bolus:
        original_get_patient_config = parser.get_patient_config

        def get_patient_config_with_bolus_acceptance(self):
            patient_config = original_get_patient_config()
            patient_config.recommendation_accept_prob = args.recommendation_accept_prob
            patient_config.min_bolus_rec_threshold = args.min_bolus_rec_threshold
            patient_config.recommendation_meal_attention_time_minutes = (
                args.recommendation_meal_attention_minutes
            )
            return patient_config

        parser.get_patient_config = types.MethodType(
            get_patient_config_with_bolus_acceptance, parser
        )

    resolved_dir = None
    if args.save_resolved_configs:
        resolved_dir = os.path.join(results_dir, "resolved_configs")
        os.makedirs(resolved_dir, exist_ok=True)

    if is_top_level_batch_config(config_obj):
        parser.metadata = config_obj["metadata"]
        parser.base_sim_config = config_obj["base_config"]
        parser.override_configs = config_obj["override_config"]

        if args.limit is not None:
            parser.override_configs = parser.override_configs[: args.limit]

        sims = parser.get_sims(override_json_save_dir=resolved_dir)
    else:
        sim_config = normalize_controller_settings_for_swift(config_obj)

        if resolved_dir is not None:
            resolved_path = os.path.join(resolved_dir, f"{sim_config['sim_id']}_resolved.json")
            with open(resolved_path, "w") as fh:
                json.dump(sim_config, fh, indent=2)

        sim_start_time, duration_hrs, virtual_patient, controller = parser.build_components_from_config(sim_config)
        sim = Simulation(
            sim_start_time,
            duration_hrs=duration_hrs,
            virtual_patient=virtual_patient,
            controller=controller,
            multiprocess=True,
            sim_id=sim_config["sim_id"],
        )
        sims = {sim.sim_id: sim}

    full_results, summary_results_df = run_simulations(
        sims,
        save_dir=results_dir,
        save_results=True,
        compute_summary_metrics=True,
        num_procs=args.num_procs,
    )

    if not is_top_level_batch_config(config_obj):
        sim_id = sim_config["sim_id"]
        full_results[sim_id] = attach_observed_cgm(full_results[sim_id], sim_config)
        full_results[sim_id].to_csv(os.path.join(results_dir, f"{sim_id}.tsv"), sep="\t")

    comparison_rows = []
    for sim_id, sim_df in full_results.items():
        row = compute_observed_vs_sim_qoi_row(sim_id, sim_df)
        if row is not None:
            comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    if not comparison_df.empty:
        comparison_df.set_index("sim_id", inplace=True)
        comparison_path = os.path.join(results_dir, "observed_vs_sim_qois.csv")
        comparison_df.to_csv(comparison_path)
    else:
        comparison_path = None

    if args.plot:
        combined_plot_path = os.path.join(results_dir, "all_sims_plot.png")
        plot_sim_results(full_results, save=True, save_path=combined_plot_path)

        for sim_id, sim_df in full_results.items():
            sim_plot_path = os.path.join(results_dir, f"{sim_id}_plot.png")
            plot_sim_results({sim_id: sim_df}, save=True, save_path=sim_plot_path)

    manifest = {
        "config_path": os.path.abspath(args.config_path),
        "pointer_dir": os.path.abspath(args.pointer_dir),
        "results_dir": os.path.abspath(results_dir),
        "num_sims": len(sims),
        "sim_ids": list(sims.keys()),
        "plots_saved": args.plot,
        "accept_recommended_bolus": args.accept_recommended_bolus,
        "observed_vs_sim_qois_path": comparison_path,
    }
    manifest_path = os.path.join(results_dir, "run_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("Results saved to:", os.path.abspath(results_dir))
    print("Simulations run:", len(sims))
    print("Summary rows:", len(summary_results_df))
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
