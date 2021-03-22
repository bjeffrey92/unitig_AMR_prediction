import pandas as pd
from typing import List


def get_pbp_sequence(
    pbp_pattern: str, df: pd.DataFrame, cols: pd.Series
) -> pd.Series:
    pbp_cols = cols[cols.str.startswith(pbp_pattern)]
    return df[pbp_cols].sum(axis=1)


def parse_pmen(
    pmen: pd.DataFrame, cdc: pd.DataFrame, pbp_patterns: List[str]
) -> pd.DataFrame:
    cols = pmen.columns.to_series()
    pbp_seqs = {pbp: get_pbp_sequence(pbp, pmen, cols) for pbp in pbp_patterns}
    df = pd.DataFrame(pbp_seqs)

    df["id"] = pmen.id
    df["mic"] = pmen.mic
    df_cols = df.columns.to_list()  # reorder columns
    df = df[df_cols[-2:] + df_cols[:-2]]

    df = df.loc[~pd.isna(df.mic)]  # drop samples with missing mic


def parse_cdc(cdc: pd.DataFrame, pbp_patterns: List[str]) -> pd.DataFrame:
    cols = cdc.columns.to_series()
    pbp_seqs = {pbp: get_pbp_sequence(pbp, cdc, cols) for pbp in pbp_patterns}
    df = pd.DataFrame(pbp_seqs)

    df["isolate"] = cdc.isolate
    df["mic"] = cdc.mic
    df = df.loc[~pd.isna(df.mic)]  # drop samples with missing mic
    df.reindex()
    df["id"] = "cdc_" + df.index.astype(str)

    df_cols = df.columns.to_list()  # reorder columns
    df = df[df_cols[-1:] + df_cols[-3:-1] + df_cols[:3]]

    return df


if __name__ == "__main__":
    cdc = pd.read_csv("data/pneumo_pbp/cdc_seqs_df.csv")
    pmen = pd.read_csv("data/pneumo_pbp/pmen_pbp_profiles_extended.csv")

    pbp_patterns = ["a1", "b2", "x2"]

    cdc = parse_cdc(cdc, pbp_patterns)

    cdc_isolates = cdc.isolate.str.split("_", expand=True)[1]
    cdc_a1 = cdc_isolates.str.split("-", expand=True)[0]
    cdc_b2 = cdc_isolates.str.split("-", expand=True)[1]
    cdc_x2 = cdc_isolates.str.split("-", expand=True)[2]

    cdc_seqs = pd.DataFrame(
        {
            "isolates": cdc_isolates,
            "a1_type": cdc_a1,
            "b2_type": cdc_b2,
            "x2_type": cdc_x2,
            "a1_seq": cdc.a1,
            "b2_seq": cdc.b2,
            "x2_seq": cdc.x2,
        }
    )

    def check_non_duplicated(i):
        return cdc_seqs.loc[cdc_seqs.a1_type == i].a1_type.nunique()

    a1_counts = cdc_a1.value_counts(ascending=True)
    a1_counts = a1_counts[a1_counts > 1]
