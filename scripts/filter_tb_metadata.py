"""
A script to select tb samples to use from Farhat et al 2019,
https://www.nature.com/articles/s41467-019-10110-6#Tab1
"""

import logging
import shutil
import urllib.request as request
from os import path
from contextlib import closing
from typing import List

import numpy as np
import pandas as pd

logging.basicConfig()
logging.root.setLevel(logging.INFO)


def load_raw_data(abs_list: List[str]) -> pd.DataFrame:
    df = pd.read_excel("data/tb/41467_2019_10110_MOESM4_ESM.xlsx", engine="openpyxl")
    df = df[df.columns[:32]]  # remove empty cols created during import
    df = df.loc[
        (df["Country of Isolation"] == "Peru") & (df["DST or MIC"] == "MIC")
    ]  # majority of samples from peru
    df = df.loc[~df["Phylogenetic lineage"].isna()]
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
    return df.assign(**{f"log2_{i}_mic": df[i].apply(np.log2) for i in abs_list})


def _assign_clade(family: str) -> int:
    if family == "4.3":
        return 1
    elif family == "4.1":
        return 2
    else:
        return 3


def group_into_clades(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(Clade=df["Phylogenetic lineage"].apply(_assign_clade))


def download_single_assembly(acc: str) -> bool:
    try:
        if acc.startswith("M"):
            fasta_name = f"{acc[:4]}01.fasta.gz"
            save_name = f"{acc}.fasta.gz"
            ftp_path = f"ftp://ftp.ebi.ac.uk/pub/databases/ena/wgs/public/{acc[:3].lower()}/{fasta_name}"
        elif acc.startswith("C"):
            save_name = f"{acc}.fasta"
            ftp_path = (
                f"https://www.ebi.ac.uk/ena/browser/api/fasta/{acc}.1?download=true"
            )
        else:
            raise ValueError(f"Unknown accession type: {acc}")
        destination_dir = "/home/bj515/OneDrive/work_stuff/WGS_AMR_prediction/graph_learning/data/tb/assemblies"
        destination = path.join(destination_dir, save_name)
        with closing(request.urlopen(ftp_path)) as r:
            with open(destination, "wb") as f:
                shutil.copyfileobj(r, f)
        return True
    except Exception as e:
        logging.warning(e)
        return False


def download_assemblies(df: pd.DataFrame):
    success = df["WGSAccessionNumber OR Run Accession"].apply(download_single_assembly)
    if not success.all():
        logging.error(f"Failed to download {(~success).sum()} samples")
    else:
        logging.info("Successfully downloaded all samples")


if __name__ == "__main__":
    first_line_abs = ["INH", "PZA", "EMB", "RIF"]
    df = load_raw_data(first_line_abs)
    df = df.loc[df.BioProject == "PRJNA343736\xa0"]  # only this proj has assemblies
    df = parse_MIC_data(df, first_line_abs)
    df = group_into_clades(df)
    df = df.assign(id=df["WGSAccessionNumber OR Run Accession"])
    df.to_csv("data/tb/filtered_accessions.tsv", sep="\t", index=False)
