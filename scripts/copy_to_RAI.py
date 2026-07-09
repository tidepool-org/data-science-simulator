"""
copy_to_RAI.py  —  Run once from any directory:
    python3 /tmp/copy_to_RAI.py
"""
import shutil
from pathlib import Path

SOURCE = Path("/Users/shawnfoster/PycharmProjects/data-science-simulator-v2/data-science-simulator/scenario_configs/tidepool_risk_v2/loop_risk_v2_0/loop_risk_v2_2_0_full")
DEST   = Path("/Users/shawnfoster/PycharmProjects/data-science-simulator-v2/data-science-simulator/scenario_configs/tidepool_risk_v2/loop_risk_v2_0/insulin_model_compare/RAI")
DEST.mkdir(parents=True, exist_ok=True)

dirs = [
    "TLR-1011","TLR-1023","TLR-1032","TLR-1034","TLR-1053","TLR-1062",
    "TLR-1065","TLR-1066","TLR-1078","TLR-1116",
    "TLR-1117_bike","TLR-1117_jog","TLR-1117_str_training","TLR-1117_walk",
    "TLR-1118_bike","TLR-1118_jog","TLR-1118_str_training","TLR-1118_walk",
    "TLR-1120","TLR-1121","TLR-1130","TLR-1131","TLR-1136","TLR-1142","TLR-1143","TLR-1147",
    "TLR-549","TLR-552","TLR-553","TLR-554","TLR-555","TLR-556","TLR-558","TLR-561","TLR-562",
    "TLR-566","TLR-568","TLR-576","TLR-577","TLR-578","TLR-579","TLR-586","TLR-587","TLR-590",
    "TLR-596","TLR-604","TLR-605","TLR-606","TLR-607","TLR-613","TLR-615","TLR-616","TLR-627",
    "TLR-629","TLR-660","TLR-664","TLR-668","TLR-675","TLR-676","TLR-682","TLR-684","TLR-687",
    "TLR-688","TLR-689","TLR-690","TLR-696","TLR-703","TLR-704","TLR-710","TLR-723","TLR-725",
    "TLR-726","TLR-727","TLR-731_cb",
    "TLR-736_180","TLR-736_270","TLR-736_360",
    "TLR-739","TLR-742","TLR-788","TLR-789","TLR-789_hf","TLR-790","TLR-792","TLR-793",
    "TLR-822","TLR-826","TLR-843",
    "TLR-845_10","TLR-845_10_corr","TLR-845_10_wmeal",
    "TLR-845_15","TLR-845_15_corr","TLR-845_15_wmeal",
    "TLR-845_20","TLR-845_20_corr","TLR-845_20_wmeal",
    "TLR-845_30","TLR-845_30_corr","TLR-845_30_wmeal",
    "TLR-845_40","TLR-845_40_corr","TLR-845_40_wmeal",
    "TLR-846_130","TLR-846_130_corr","TLR-846_130_wmeal",
    "TLR-846_70","TLR-846_70_corr","TLR-846_70_wmeal",
    "TLR-847","TLR-847_corr","TLR-847_wmeal",
    "TLR-861",
    "TLR-899_01_025","TLR-899_01_075","TLR-899_01_090","TLR-899_01_095",
    "TLR-899_01_105","TLR-899_01_110","TLR-899_01_125","TLR-899_01_175","TLR-899_01_250",
    "TLR-899_10_105","TLR-899_10_110","TLR-899_10_125","TLR-899_10_175",
    "TLR-899_10_25","TLR-899_10_250","TLR-899_10_75","TLR-899_10_90","TLR-899_10_95",
    "TLR-899_1_105","TLR-899_1_110","TLR-899_1_125","TLR-899_1_175",
    "TLR-899_1_25","TLR-899_1_250","TLR-899_1_75","TLR-899_1_90","TLR-899_1_95",
    "TLR-901","TLR-911","TLR-912","TLR-950",
    "TLR-969_30_3","TLR-969_30_5","TLR-969_3_30","TLR-969_3_5","TLR-969_5_3","TLR-969_5_half",
]

copied, missing = [], []
for d in dirs:
    src = SOURCE / d
    if src.is_dir():
        shutil.copytree(src, DEST / d, dirs_exist_ok=True)
        copied.append(d)
        print(f"  ✓ {d}")
    else:
        missing.append(d)
        print(f"  ✗ {d} (not in source)")

print(f"\nDone. Copied: {len(copied)}, Missing from source: {len(missing)}")
if missing:
    print(f"Missing: {missing}")
