"""
A script to select tb samples to use from Farhat et al 2019,
https://www.nature.com/articles/s41467-019-10110-6#Tab1
"""

import logging
from typing import List

import pandas as pd
import numpy as np

logging.basicConfig()
logging.root.setLevel(logging.INFO)


def load_raw_data(abs_list: List[str]) -> pd.DataFrame:
    df = pd.read_excel("data/tb/41467_2019_10110_MOESM4_ESM.xlsx", engine="openpyxl")
    df = df[df.columns[:32]]  # remove empty cols created during import
    df = df.loc[
        (df["Country of Isolation"] == "Peru") & (df["DST or MIC"] == "MIC")
    ]  # majority of samples from peru
    samples_missing_tests = df[abs_list].isna().any(axis=1)
    logging.warning(f"{samples_missing_tests.sum()}/{len(df)} samples will be dropped")
    return df.loc[~samples_missing_tests]


def make_MIC_numeric(x: str) -> float:
    try:
        x = sorted([float(i) for i in x.split("-")])  # type: ignore
        return x[0] + (x[1] - x[0]) / 2  # type: ignore
    except ValueError:
        for i in range(1, 3):
            x_ = x[i:]
            try:
                return float(x_)
            except ValueError:
                pass
        raise ValueError


def parse_MIC_data(df: pd.DataFrame, abs_list: List[str]) -> pd.DataFrame:
    df = df.assign(**{i: df[i].apply(make_MIC_numeric) for i in abs_list})
    df = df.assign(**{i: df[i].apply(np.log2) for i in abs_list})


if __name__ == "__main__":
    first_line_abs = ["INH", "PZA", "EMB", "RIF"]
    df = load_raw_data(first_line_abs)
