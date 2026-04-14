import os
import pandas as pd

import proloaf.datahandler as dh
from proloaf.cli import parse_basic
from proloaf.confighandler import read_config
from proloaf.event_logging import create_event_logger

MAIN_PATH = os.path.dirname(os.path.realpath(__file__))
logger = create_event_logger("preprocess")

if __name__ == "__main__":
    ARGS = parse_basic()
    config_file = os.path.join(MAIN_PATH, "targets", ARGS.station, "preprocessing.json")
    PAR = read_config(config_path=config_file)

    # 输入输出路径
    if PAR["local"]:
        INPATH = os.path.join(MAIN_PATH, PAR["raw_path"])
    else:
        INPATH = PAR["raw_path"]

    OUTFILE = os.path.join(MAIN_PATH, PAR["data_path"])

    # 这里只处理一个 csv 文件
    csv_cfg = PAR["csv_files"][0]
    infile = os.path.join(INPATH, csv_cfg["file_name"])

    logger.info(f"Reading raw csv: {infile}")

    # 读原始 csv
    df = pd.read_csv(
        infile,
        sep=csv_cfg.get("sep", ","),
        usecols=csv_cfg.get("use_columns"),
    )

    # 解析时间列
    date_column = csv_cfg.get("date_column", "utc_timestamp")
    df[date_column] = pd.to_datetime(
        df[date_column],
        dayfirst=csv_cfg.get("dayfirst", False),
        utc=(csv_cfg.get("time_zone", "UTC").upper() == "UTC"),
    )

    # 设为时间索引，并统一命名为 Time
    df = df.set_index(date_column)
    df.index.name = "Time"

    # 排序
    df = df.sort_index()

    # 强制对齐为 30min 频率
    df = df.resample("30min").mean()

    # 去掉最前面关键列不完整的部分
    feature_cols = [c for c in df.columns]
    first_full_idx = df[feature_cols].dropna().index.min()
    if first_full_idx is not None:
        df = df.loc[first_full_idx:].copy()

    # 补缺失并检查
    dh.fill_if_missing(df, periodicity=48)
    dh.check_continuity(df)
    dh.check_nans(df)

    # 这里先不加 aux features，因为 train.py 里还会加
    if PAR.get("add_aux_features", False):
        df = dh.add_cyclical_features(df)
        df = dh.add_onehot_features(df)

    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    df.to_csv(OUTFILE, sep=";", index=True)

    logger.info(f"Saved prepared data to: {OUTFILE}")
    logger.info(f"Final shape: {df.shape}")
    print(df.head())