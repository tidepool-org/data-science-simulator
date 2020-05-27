import pandas as pd
import numpy as np
import datetime

from pyloopkit.dose import DoseType

# from tidepool_data_science_simulator.models.simple_metabolism_model import get_iob_from_sbr, simple_metabolism_model
from tidepool_data_science_simulator.legacy.risk_metrics_ORIG import get_bgri, lbgi_risk_score, hbgi_risk_score


# %% create pandas dataframes from the input data
def dict_inputs_to_dataframes(input_data):
    # define the dataframes to store the data in
    df_basal_rate = pd.DataFrame()
    df_carb = pd.DataFrame()
    df_carb_ratio = pd.DataFrame()
    df_dose = pd.DataFrame()
    df_glucose = pd.DataFrame()
    df_last_temporary_basal = pd.DataFrame()
    df_misc = pd.DataFrame()
    df_sensitivity_ratio = pd.DataFrame()
    df_settings = pd.DataFrame()
    df_target_range = pd.DataFrame()

    for k in input_data.keys():
        if type(input_data[k]) != dict:
            if "basal_rate" in k:
                df_basal_rate[k] = input_data.get(k)
            elif "carb_ratio" in k:
                df_carb_ratio[k] = input_data.get(k)
            elif "carb" in k:
                df_carb[k] = input_data.get(k)
            elif "dose" in k:
                df_dose[k] = input_data.get(k)
            elif "glucose" in k:
                df_glucose[k] = input_data.get(k)
            elif "last_temporary_basal" in k:
                # TODO: change how this is dealt with in pyloopkit
                df_last_temporary_basal[k] = input_data.get(k)
            elif "sensitivity_ratio" in k:
                df_sensitivity_ratio[k] = input_data.get(k)
            elif "target_range" in k:
                df_target_range[k] = input_data.get(k)
            else:
                if np.size(input_data.get(k)) == 1:
                    if type(input_data[k]) == list:
                        df_misc.loc[k, 0] = input_data.get(k)[0]
                    else:
                        df_misc.loc[k, 0] = input_data.get(k)
        else:
            if "settings_dictionary" in k:
                settings_dictionary = input_data.get("settings_dictionary")
                for sk in settings_dictionary.keys():
                    if np.size(settings_dictionary.get(sk)) == 1:
                        if type(settings_dictionary[sk]) == list:
                            df_settings.loc[sk, "settings"] = settings_dictionary.get(
                                sk
                            )[0]
                        else:
                            df_settings.loc[sk, "settings"] = settings_dictionary.get(
                                sk
                            )
                    else:
                        if sk in ["model", "default_absorption_times"]:
                            # TODO: change this in the loop algorithm
                            # to take 2 to 3 inputs instead of 1
                            df_settings.loc[sk, "settings"] = str(
                                settings_dictionary.get(sk)
                            )

    return (
        df_basal_rate,
        df_carb,
        df_carb_ratio,
        df_dose,
        df_glucose,
        df_last_temporary_basal,
        df_misc,
        df_sensitivity_ratio,
        df_settings,
        df_target_range,
    )


def dataframe_inputs_to_dict(dfs, df_misc, df_settings):
    # write the dataframes back to one dictionary
    input_dictionary = dict()
    input_dictionary = df_misc.to_dict()[0]
    for df in dfs:
        for col in df.columns:
            if "units" not in col:
                input_dictionary[col] = df[col].tolist()
            else:
                input_dictionary[col] = df[col].unique()[0]

    input_dictionary["settings_dictionary"] = df_settings.to_dict()["settings"]

    # set the format back for the edge cases
    input_dictionary["settings_dictionary"]["model"] = np.safe_eval(
        input_dictionary["settings_dictionary"]["model"]
    )
    input_dictionary["settings_dictionary"]["default_absorption_times"] = np.safe_eval(
        input_dictionary["settings_dictionary"]["default_absorption_times"]
    )

    input_dictionary["offset_applied_to_dates"] = int(
        input_dictionary["offset_applied_to_dates"]
    )

    return input_dictionary


def input_dict_to_one_dataframe(input_data):
    # get dataframes from input
    (
        df_basal_rate,
        df_carb,
        df_carb_ratio,
        df_dose,
        df_glucose,
        df_last_temporary_basal,
        df_misc,
        df_sensitivity_ratio,
        df_settings,
        df_target_range,
    ) = dict_inputs_to_dataframes(input_data)

    # combine the dataframes into one big dataframe,
    # put glucose at end since that trace is typically long
    combined_df = pd.DataFrame()
    combined_df = pd.concat([combined_df, df_settings])
    combined_df = pd.concat([combined_df, df_misc])

    dfs = [
        df_basal_rate,
        df_carb,
        df_carb_ratio,
        df_dose,
        df_last_temporary_basal,
        df_sensitivity_ratio,
        df_target_range,
        df_glucose,
    ]

    for df in dfs:
        combined_df = pd.concat([combined_df, df.T])

    # move settings back to the front of the dataframe
    combined_df = combined_df[np.append("settings", combined_df.columns[0:-1])]

    return combined_df


def str2bool(string_):
    return string_.lower() in ("yes", "true", "t", "1")


def input_table_to_dict(input_df):
    dict_ = dict()

    # first parse and format the settings
    all_settings = input_df["settings"].dropna()
    dict_["settings_dictionary"] = all_settings.to_dict()

    for k in dict_["settings_dictionary"].keys():
        if k in ["dynamic_carb_absorption_enabled", "retrospective_correction_enabled"]:

            dict_["settings_dictionary"][k] = str2bool(dict_["settings_dictionary"][k])
        else:
            dict_["settings_dictionary"][k] = np.safe_eval(
                dict_["settings_dictionary"][k]
            )
    if "suspend_threshold" not in dict_["settings_dictionary"].keys():
        dict_["settings_dictionary"]["suspend_threshold"] = None

    # then parse and format the rest
    input_df_T = input_df.drop(columns=["settings"]).dropna(axis=0, how="all").T

    input_df_columns = input_df_T.columns
    for col in input_df_columns:
        if "units" in col:
            dict_[col] = input_df_T[col].dropna().unique()[0]
        elif "offset" in col:
            dict_[col] = int(np.safe_eval(input_df_T[col].dropna()[0]))
        elif "time_to_calculate" in col:
            dict_[col] = datetime.datetime.fromisoformat(
                pd.to_datetime(input_df_T[col].dropna()[0]).isoformat()
            )
        else:
            temp_df = input_df_T[col].dropna()
            temp_array = []
            for v in temp_df.values:
                if ":" in v:
                    if len(v) == 7:
                        obj = datetime.time.fromisoformat(
                            pd.to_datetime(v).strftime("%H:%M:%S")
                        )
                    elif len(v) == 8:
                        obj = datetime.time.fromisoformat(v)
                    elif len(v) > 8:
                        obj = datetime.datetime.fromisoformat(
                            pd.to_datetime(v).isoformat()
                        )
                    else:
                        obj = np.safe_eval(v)
                elif "DoseType" in v:
                    obj = DoseType.from_str(v[9:])
                else:
                    obj = np.safe_eval(v)

                temp_array = np.append(temp_array, obj)

            dict_[col] = list(temp_array)

    return dict_


def create_contiguous_ts(date_min, date_max, freq="1s"):
    date_range = pd.date_range(date_min, date_max, freq=freq)

    contig_ts = pd.DataFrame(date_range, columns=["datetime"])
    contig_ts["time"] = contig_ts["datetime"].dt.start_time

    return contig_ts


def get_setting(current_time, df, setting_value_name, setting_time_name):
    continguous_ts = create_contiguous_ts(
        current_time.date(), current_time.date() + datetime.timedelta(days=1), freq="1s"
    )
    df_ts = pd.merge(
        continguous_ts, df, left_on="time", right_on=setting_time_name, how="left"
    )
    df_ts[setting_value_name].fillna(method="ffill", inplace=True)
    setting_value_at_current_time = df_ts.loc[
        df_ts["datetime"] == current_time, setting_value_name
    ].values[0]
    setting_value_at_current_time

    return setting_value_at_current_time


def transform_input_scenario_to_simulation_df(
    scenario_filepath, simulation_duration_hours
):
    # LOAD & VIEW SCENARIO INPUTS FROM GOOGLE DRIVE
    # worksheet = gc.open(scenario_file_names[scenario_number]).sheet1
    # rows = worksheet.get_all_values()
    # col_headings = rows[0]
    data = pd.read_csv(scenario_filepath, sep="\t")
    custom_table_df = data.set_index("setting_name")

    # create output dataframes
    metab_dur_mins = 8 * 60  # 8 hours

    # +CS - Simulation lasts as long as simulation is specified or metabolism model requires
    sim_dur_mins = np.max([simulation_duration_hours * 60, metab_dur_mins])

    # +CS - Why is this length sim_dur_mins * 2 and sim_df is sim_dur_mins?
    delta_bgs_df = pd.DataFrame(index=np.arange(0, sim_dur_mins * 2, 5))
    iob_df = delta_bgs_df.copy()
    sim_df = pd.DataFrame(index=np.arange(0, sim_dur_mins, 5))
    scenario_results = pd.DataFrame()

    # show inputs
    custom_table_df

    # %% RUN INITIAL SCENARIO THROUGH DIABETES METABOLISM MODEL

    # get inputs from custom scenario
    # NOTE: this line next line is needed bc we are pulling from gsheet instead of .csv
    custom_table_df[custom_table_df == ""] = np.nan
    inputs_from_file = input_table_to_dict(custom_table_df)

    # convert inputs to dataframes
    (
        basal_rates,
        carb_events,
        carb_ratios,
        dose_events,
        cgm_df,
        df_last_temporary_basal,
        df_misc,
        isfs,
        df_settings,
        df_target_range,
    ) = dict_inputs_to_dataframes(inputs_from_file)

    print("running scenario through simple diabetes metabolism model...")
    t0 = inputs_from_file.get("time_to_calculate_at")
    bg_t0_actual = cgm_df.loc[
        cgm_df["glucose_dates"] == t0, "actual_blood_glucose"
    ].values[0]

    bg_t0_loop = cgm_df.loc[cgm_df["glucose_dates"] == t0, "glucose_values"].values[0]

    # get actual and loop carb amounts
    carb_amount_actual = carb_events.loc[
        carb_events["carb_dates"] == t0, "actual_carbs"
    ].values[0]
    carb_amount_loop = carb_events.loc[
        carb_events["carb_dates"] == t0, "carb_values"
    ].values[0]

    # get actual and loop insulin amounts
    insulin_amount_actual = dose_events.loc[
        dose_events["dose_start_times"] == t0, "actual_doses"
    ].values[0]
    insulin_amount_loop = dose_events.loc[
        dose_events["dose_start_times"] == t0, "dose_values"
    ].values[0]

    # get actual and loop cir
    cir_index = carb_ratios[
        t0.time() >= carb_ratios["carb_ratio_start_times"]
    ].index.values.min()
    cir_actual = carb_ratios.loc[cir_index, "actual_carb_ratios"]
    cir_loop = carb_ratios.loc[cir_index, "carb_ratio_values"]

    # get actual and loop isf
    isf_index = isfs[
        t0.time() >= isfs["sensitivity_ratio_start_times"]
    ].index.values.min()
    isf_actual = isfs.loc[isf_index, "actual_sensitivity_ratios"]
    isf_loop = isfs.loc[isf_index, "sensitivity_ratio_values"]

    (
        delta_bg_array_metab,
        ts_array_metab,
        carbs_consumed_array_metab,
        insulin_delivered_array_metab,
        iob_array_metab,
    ) = simple_metabolism_model(
        carb_amount=carb_amount_actual,
        insulin_amount=insulin_amount_actual,
        CIR=cir_actual,
        ISF=isf_actual,
    )

    delta_bgs_df["initial_scenario"] = np.nan

    # +CS - these aren't bg_times. they are a boolean array mask for bgs indices up to metabolism duration
    bg_metab_mask = (delta_bgs_df.index >= 0) & (delta_bgs_df.index < metab_dur_mins)
    delta_bgs_df.loc[bg_metab_mask, "initial_scenario"] = delta_bg_array_metab

    # get scheduled basal rate
    sbr_index = basal_rates[
        t0.time() >= basal_rates["basal_rate_start_times"]
    ].index.values.min()
    sbr_loop_scalar = basal_rates.loc[sbr_index, "basal_rate_values"]
    sbr_actual_scalar = basal_rates.loc[sbr_index, "actual_basal_rates"]

    # calculate the amount of insulin onboard from scheduled basal rate
    iob_from_sbr_array_metab = get_iob_from_sbr(sbr_loop_scalar)

    # capture the insulin that will be onboard for the next 8 hours
    iob_df["initial_scenario"] = np.nan
    iob_df.loc[bg_metab_mask, "initial_scenario"] = (
        iob_array_metab + iob_from_sbr_array_metab
    )

    bg_timeseries = bg_t0_actual + np.cumsum(delta_bg_array_metab)
    sim_df.loc[bg_metab_mask, "pump_bgs"] = bg_timeseries
    pump_LBGI, pump_HBGI, pump_BGRI = get_bgri(bg_timeseries)

    scenario_results.loc["LBGI", "pumpValue"] = pump_LBGI
    scenario_results.loc["LBGI", "pumpRiskScore"] = lbgi_risk_score(pump_LBGI)
    scenario_results.loc["HBGI", "pumpValue"] = pump_HBGI
    scenario_results.loc["HBGI", "pumpRiskScore"] = hbgi_risk_score(pump_HBGI)
    scenario_results.loc["BGRI", "pumpValue"] = pump_BGRI

    return scenario_results, inputs_from_file
