import pandas as pd
import csv
import numpy as np
import torch
import os
import logging
import sys
import tempfile
from itertools import compress

logging.basicConfig()
logging.root.setLevel(logging.INFO)


def parse_metadata(metadata_file, rtab_file, outcome_column):
    metadata_df = pd.read_csv(metadata_file)

    #drop everything without measurement for outcome
    metadata_df = metadata_df.loc[~metadata_df[outcome_column].isna()] 

    with open(rtab_file, 'r') as a:
        reader = csv.reader(a, delimiter = '\t')
        input_files = next(reader)[1:]
    
    assert all([i.endswith('.contigs_velvet') for i in input_files])    
    accessions = [i.strip('.contigs_velvet') for i in input_files]

    accessions = pd.DataFrame(accessions)
    df = metadata_df.merge(accessions, 
                        left_on = 'Sanger_lane', right_on = 0)
    df = df.rename(columns = {0:'Filename'})
    df['Filename'] = df['Filename'] + '.contigs_velvet' #so can identify relevant inputs in the rtab

    diff = len(accessions) - len(df)
    if diff > 0:
        logging.warning(f'{diff} entries in {rtab_file} could not be mapped to entries in {metadata_file}')
        input_files = [i for i in input_files if i in df.Filename.values] #get all which are present

    df.set_index('Filename', inplace = True)
    df = df.loc[input_files] #order metadata by order that files are present in the rtab

    return df


def split_training_and_testing(rtab_file, 
                                files_to_include,
                                training_rtab_file,
                                testing_rtab_file,
                                num_unitigs,
                                freq_filt = (0.01, 0.99),
                                training_split = 0.7):
    '''
    create training and testing rtab files so features can be generated in most memory efficient way
    '''

    #get training and testing data as separate lists
    with open(rtab_file, 'r') as a:
        reader = csv.reader(a, delimiter = '\t')
        header = next(reader)
        filt = [i in files_to_include for i in header] #which to include
        filt[0] = True #add True at start to include pattern_id column
        header = list(compress(header, filt))
        total_samples = len(header) - 1

        training_n = round(total_samples * training_split)
        testing_n = total_samples - training_n

        #memory allocation
        training_rows = [header[:training_n + 1]] + \
            ([[None] * (training_n + 1)] * num_unitigs)
        testing_rows = [header[:1] + header[training_n + 1:]] + \
            ([[None] * (testing_n + 1)] * num_unitigs)

        i = 1
        j = 1
        for row in reader:
            sys.stdout.write(f'\rprocessing {j} of {num_unitigs} unitigs')
            sys.stdout.flush()
            row = list(compress(row, filt))
            j += 1

            frequency = sum([1 for i in row[1:] if i == '1'])/len(row[1:])
            if frequency < freq_filt[0] or frequency > freq_filt[1]: continue #only include intermediate frequency unitigs

            training_rows[i] = row[:training_n + 1]
            testing_rows[i] = row[:1] + row[training_n + 1:]
            
            i += 1
        sys.stdout.write('')
        sys.stdout.flush()

    def has_header(rtab_file):
        rtab_file.seek(0)
        row_count = 0
        for row in rtab_file:
            row_count += 1
            if row_count == 1:
                return True
        return False

    if has_header(training_rtab_file):
        training_rows = training_rows[1:]
    if has_header(testing_rtab_file):
        testing_rows = testing_rows[1:]

    training_rows = [('\t'.join(x) + '\n').encode() for x in training_rows[:i]]
    training_rtab_file.writelines(training_rows)

    testing_rows = [('\t'.join(x) + '\n').encode() for x in testing_rows[:i]]
    testing_rtab_file.writelines(testing_rows)


def load_features(rtab_file):
    '''
    parses rtab file as sparse matrix of features
    this was the most memory efficient way i could find to do this
    '''
    num_unitigs = sum(1 for line in open(rtab_file)) - 1

    with open(rtab_file, 'r') as a:
        reader = csv.reader(a, delimiter = '\t')
        header = next(reader)
        num_samples = len(header) - 1
    
        x_idx = []
        y_idx = []
        values = []

        i = 0
        for row in reader:
            for j in range(1, len(row)): #first element of row is unitig number
                if row[j] == '1':
                    x_idx.append(j - 1)
                    y_idx.append(i)
                    values.append(1)
            i += 1
            sys.stdout.write(f'\r{i}/{num_unitigs} unitigs processed') # \r adds on same line
            sys.stdout.flush()
        sys.stdout.write('')
        sys.stdout.flush()

    indices = torch.LongTensor([x_idx, y_idx])
    
    #delete these to free up RAM
    del x_idx
    del y_idx

    shape = (num_samples, num_unitigs)
    values_tensor = torch.FloatTensor(values)

    del values

    features = torch.sparse_coo_tensor(indices, values_tensor, shape)
    return features


def order_metadata(metadata, rtab_file):
    
    with open(rtab_file, 'r') as a:
        reader = csv.reader(a, delimiter = '\t')
        input_files = next(reader)[1:]

    metadata = metadata.loc[input_files] #order dataframe 
    
    return metadata


def load_labels(metadata, label_column):
    return torch.FloatTensor(metadata[label_column].values)


def load_countries(metadata, countries = countries):
    country_tensors = {}
    for i in countries:
        country_tensors[i] = \
            torch.FloatTensor([(lambda x: 1 if x == i else 0)(x) for x in countries])

    def parse_country(country):
        return country_tensors[country]

    return torch.stack(metadata.Country.apply(parse_country).to_list())


if __name__ == '__main__':

    rtab_file = 'data/gonno_unitigs/gonno.rtab'
    metadata_file = 'data/metadata.csv'
    outcome_column = 'log2_cip_mic'

    #maps entries in rtab to metadata
    metadata = parse_metadata(metadata_file, rtab_file, outcome_column)

    #alphabetical list of countries
    countries = metadata.Country.unique()
    countries = countries.sort().tolist()
    
    #if don't wish to specify countries
    # countries = []

    with tempfile.TemporaryFile() as training_rtab_file, \
        tempfile.TemporaryFile() as testing_rtab_file:
    
        num_unitigs = sum(1 for line in open(rtab_file)) - 1

        if countries:
            for country in countries:
                to_include = metadata[metadata.Country == country].index
                split_training_and_testing(rtab_file, to_include, 
                                        training_rtab_file, testing_rtab_file,
                                        num_unitigs)
        else:
            split_training_and_testing(rtab_file, metadata.index, 
                                        training_rtab_file, testing_rtab_file,
                                        num_unitigs)

        #reads in rtab as sparse feature tensor
        training_features = load_features(training_rtab_file)
        testing_features = load_features(testing_rtab_file)

        #ensure metadata is in same order as features for label extraction
        training_metadata = order_metadata(metadata, training_rtab_file)
        testing_metadata = order_metadata(metadata, testing_rtab_file)

    #parse training and testing labels as tensors
    training_labels = load_labels(training_metadata, outcome_column)
    testing_labels = load_labels(testing_metadata, outcome_column)

    #countries of training and testing data as tensor of 1 and 0
    training_countries = load_countries(training_metadata)
    testing_countries = load_countries(testing_metadata)