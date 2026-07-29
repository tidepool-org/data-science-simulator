"""
Regression test for a real bug found during Phase 3 manual verification
(2026-07-21): plot_sim_results(save=True) used the global plt.savefig(),
which saves whatever pyplot considers the "current figure" rather than the
specific figure just built, and never closed figures afterward. Calling it
repeatedly in the same process (exactly what gui_runner.run_risk_assessment
does once per scenario file in a directory) produced blank PNGs for roughly
every other call. Fixed by using fig.savefig() and closing the figure after
saving. This test would have caught it -- the approved Phase 3 integration
test plan didn't assert on PNG pixel content, only on the assessment object,
so this slipped through automated coverage until a manual browser pass.
"""

import datetime

import numpy as np
import pandas as pd
from PIL import Image

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results


def _mock_results():
    t0 = datetime.datetime(2019, 8, 15, 12, 0, 0)
    times = pd.date_range(start=t0, periods=10, freq="5min")
    df = pd.DataFrame({
        "time": times,
        "bg": np.random.uniform(80, 180, 10),
        "bg_sensor": np.random.uniform(80, 180, 10),
        "sbr": np.random.uniform(0.5, 1.5, 10),
        "temp_basal": np.random.uniform(0, 2, 10),
        "true_bolus": np.zeros(10),
        "reported_bolus": np.zeros(10),
        "iob": np.random.uniform(0, 3, 10),
        "ei": np.random.uniform(0, 0.2, 10),
        "true_carb_value": np.zeros(10),
        "reported_carb_value": np.zeros(10),
    }, index=times)
    return {"test_sim": df}


def _is_blank(png_path):
    im = Image.open(png_path).convert("RGB")
    return len(im.getcolors(maxcolors=1000000)) <= 1


def test_repeated_saved_calls_each_produce_non_blank_png(tmp_path):
    paths = [tmp_path / f"plot_{i}.png" for i in range(4)]
    for path in paths:
        plot_sim_results(_mock_results(), save=True, save_path=str(path))

    blank = [str(p) for p in paths if _is_blank(p)]
    assert blank == [], f"Blank PNG(s) produced on repeated calls: {blank}"
