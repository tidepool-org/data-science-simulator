__author__ = "Cameron Summers"

import os
import re

import pandas as pd


def load_jaeb_issue_report_time_series_data(data_dir, num_reports=10):

    print("Loading jaeb issue report time series data...")
    population_data = []
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for file in sorted(files):
            if re.search("LOOP-\d+.*.csv", file):

                filepath = os.path.join(root, file)
                df = pd.read_csv(filepath)
                if "cgm" in df.columns:
                    population_data.append(df)

                if len(population_data) > num_reports:
                    break

    return population_data