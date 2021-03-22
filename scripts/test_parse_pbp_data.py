import pandas as pd

from parse_pbp_data import parse_cdc, parse_pmen

cdc = pd.read_csv("data/pneumo_pbp/cdc_seqs_df.csv")
pmen = pd.read_csv("data/pneumo_pbp/pmen_pbp_profiles_extended.csv")
pbp_patterns = ["a1", "b2", "x2"]


def check_data(data):
    for pbp in pbp_patterns:
        assert (
            all(
                data.groupby(data[f"{pbp}_type"]).apply(
                    lambda df: df[f"{pbp}_seq"].nunique() == 1
                )
            )
            is True
        )  # one sequence per type
        assert (
            all(
                data.groupby(data[f"{pbp}_seq"]).apply(
                    lambda df: df[f"{pbp}_type"].nunique() == 1
                )
            )
            is True
        )  # one type per sequence


def test_parse_cdc():
    cdc_seqs = parse_cdc(cdc, pbp_patterns)
    check_data(cdc_seqs)


def test_parse_men():
    cdc_seqs = parse_cdc(cdc, pbp_patterns)
    pmen_seqs = parse_pmen(pmen, cdc_seqs, pbp_patterns)
    check_data(pmen_seqs)
