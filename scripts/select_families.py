import pandas as pd 
import random

df = pd.read_csv('data/metadata_accessions_in_rtab.csv')

Abs = ['log2_azm_mic',
    'log2_cip_mic',
    'log2_cro_mic',
    'log2_cfx_mic'] #abs to use, last two are missing in lots of samples

#subset metadata so all ab cols are complete 
for i in Abs[:-2]:
    df = df[~pd.isna(df[i])]

#countries with more than 40 representatives
families = df.Family.value_counts()[df.Family.value_counts() > 40].index

def select_samples(d, families):
    if not d.Family.iloc[0] in families:
        return None
    elif len(d) <= 100:
        return d
    else:
        indices = random.sample(d.index.to_list(), 100)
        return d.loc[indices]

#get metadata where each country will have at most 100 and at least 40 samples
familiy_df = df.groupby(df.Family).apply(lambda x: select_samples(x, families))

familiy_df.to_csv('data/family_normalised_metadata.csv', index = False)