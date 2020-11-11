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
countries = df.Country.value_counts()[df.Country.value_counts() > 40].index

def select_samples(d, countries):
    if not d.Country.iloc[0] in countries:
        return None
    elif len(d) <= 100:
        return d
    else:
        indices = random.sample(d.index.to_list(), 100)
        return d.loc[indices]

#get metadata where each country will have at most 100 and at least 40 samples
country_df = df.groupby(df.Country).apply(lambda x: select_samples(x, countries))

country_df.to_csv('data/country_normalised_metadata.csv', index = False)