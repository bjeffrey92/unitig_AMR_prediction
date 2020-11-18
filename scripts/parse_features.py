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
        reader = csv.reader(a, delimiter = ' ')
        input_files = next(reader)[1:]
    
    accessions = pd.DataFrame(input_files)
    df = metadata_df.merge(accessions, 
                        left_on = 'Sanger_lane', right_on = 0)
    df = df.rename(columns = {0:'Filename'})

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
                                metadata = None, 
                                country_split = False,
                                freq_filt = (0.05, 0.95),
                                training_split = 0.7):
    '''
    create training and testing rtab files so features can be generated in most memory efficient way
    '''

    num_unitigs = sum(1 for line in open(rtab_file)) - 1

    #get training and testing data as separate lists
    with open(rtab_file, 'r') as a:
        reader = csv.reader(a, delimiter = ' ')
        header = next(reader)
        filt = [i in files_to_include for i in header] #which to include
        filt[0] = True #add True at start to include pattern_id column
        header = list(compress(header, filt))
        total_samples = len(header) - 1

        if country_split:
            #get indices of samples in the rtab file header so that countries 
            #are equally represented in the training and testing data
            #train test split may not be exact
            training_indices = []
            testing_indices = []            
            country_freq = metadata.Country.value_counts().to_dict()
            for c in country_freq:
                c_training_n = round(country_freq[c] * training_split)
                c_samples = \
                    metadata[metadata.Country == c].index.to_list()
                training_indices += [header.index(i) \
                                            for i in c_samples[:c_training_n]]
                testing_indices += [header.index(i) \
                                            for i in c_samples[c_training_n:]]
                training_n = len(training_indices)
                testing_n = len(testing_indices)
        else:
            #split randomly
            training_n = round(total_samples * training_split)
            testing_n = total_samples - training_n
            training_indices = list(range(1, training_n + 1))
            testing_indices = list(range(training_n + 1, len(header)))

        #memory allocation
        header_array = np.array(header)
        training_rows = [header[:1] + header_array[training_indices].tolist()] + \
            ([[None] * (training_n + 1)] * num_unitigs)
        testing_rows = [header[:1] + header_array[testing_indices].tolist()] + \
            ([[None] * (testing_n + 1)] * num_unitigs)

        included_unitigs = [None] * num_unitigs
        i = 1
        j = 1
        for row in reader:
            sys.stdout.write(f'\rprocessing {j} of {num_unitigs} unitigs')
            sys.stdout.flush()
            row = list(compress(row, filt))
            j += 1

            frequency = sum([1 for i in row[1:] if i == '1'])/len(row[1:])
            if frequency < freq_filt[0] or frequency > freq_filt[1]: continue #only include intermediate frequency unitigs

            included_unitigs[i-1] = row[0]
            row_array = np.array(row)
            training_rows[i] = row[:1] + row_array[training_indices].tolist()
            testing_rows[i] = row[:1] + row_array[testing_indices].tolist()
            
            i += 1
        sys.stdout.write('')
        sys.stdout.flush()

    with open(training_rtab_file, 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = '\t')
        for row in training_rows[:i]:
            writer.writerow(row)

    with open(testing_rtab_file, 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = '\t')
        for row in testing_rows[:i]:
            writer.writerow(row)

    return [i for i in included_unitigs if i is not None]

def load_features(rtab_file, mapping_dict):
    '''
    parses rtab file as sparse matrix of features
    this was the most memory efficient way i could find to do this
    params:
        rtab_file: output of unitig-counter split as training or testing
        mapping_dict: maps pattern id in rtab file to relevant nodes in the graph
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
            graph_nodes = mapping_dict[row[0]]
            for j in range(1, len(row)): #first element of row is unitig number
                if row[j] == '1':
                    for node in graph_nodes:
                        x_idx.append(j - 1)
                        y_idx.append(int(node))
                        values.append(1)
            i += 1
            sys.stdout.write(f'\r{i}/{num_unitigs} unitigs processed') # \r adds on same line
            sys.stdout.flush()
        sys.stdout.write('')
        sys.stdout.flush()

    indices = torch.LongTensor([x_idx, y_idx])
    shape = (num_samples, max(y_idx) + 1)
    
    #delete these to free up RAM
    del x_idx
    del y_idx

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


def load_countries(metadata, countries):
    country_tensors = {}
    for i in countries:
        country_tensors[i] = \
            torch.FloatTensor([(lambda x: 1 if x == i else 0)(x) for x in countries])

    def parse_country(country):
        return country_tensors[country]

    return torch.stack(metadata.Country.apply(parse_country).to_list())


def load_families(metadata, families):
    family_tensors = {}
    for i in families:
        family_tensors[i] = \
            torch.FloatTensor([(lambda x: 1 if x == i else 0)(x) for x in families])

    def parse_family(family):
        return family_tensors[family]

    return torch.stack(metadata.Family.apply(parse_family).to_list())
    

def save_data(out_dir, training_features, testing_features, 
                training_labels, testing_labels, 
                training_countries = None, testing_countries = None):
    
    torch.save(training_features, os.path.join(out_dir, 'training_features.pt'))
    torch.save(testing_features, os.path.join(out_dir, 'testing_features.pt'))
    torch.save(training_labels, os.path.join(out_dir, 'training_labels.pt'))
    torch.save(testing_labels, os.path.join(out_dir, 'testing_labels.pt'))

    if training_countries is not None:
        torch.save(training_countries, 
                os.path.join(out_dir, 'training_countries.pt'))
    if testing_countries is not None:
        torch.save(testing_countries, 
                os.path.join(out_dir, 'testing_countries.pt'))



if __name__ == '__main__':

    rtab_file = 'data/gonno_unitigs/unitigs.unique_rows.Rtab'
    metadata_file = 'data/metadata.csv'
    # metadata_file = 'data/country_normalised_metadata.csv'
    outcome_column = 'log2_cip_mic'

    #maps entries in rtab to metadata
    metadata = parse_metadata(metadata_file, rtab_file, outcome_column)

    #alphabetical list of countries
    # countries = metadata.Country.unique()
    # countries.sort()
    # countries = countries.tolist()
    
    #if don't wish to specify countries
    countries = []

    with tempfile.NamedTemporaryFile() as a, tempfile.NamedTemporaryFile() as b:
        training_rtab_file = a.name
        testing_rtab_file = b.name

        if countries:
            included_unitigs = split_training_and_testing(rtab_file, 
                                        metadata.index, 
                                        training_rtab_file, testing_rtab_file,
                                        metadata, country_split = True)
        else:
            included_unitigs = split_training_and_testing(rtab_file, 
                                        metadata.index, 
                                        training_rtab_file, testing_rtab_file)

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
    if countries:
        training_countries = load_countries(training_metadata, countries)
        testing_countries = load_countries(testing_metadata, countries)
    else:
        training_countries = None
        testing_countries = None

    out_dir = os.path.join('data/model_inputs/freq_5_95', outcome_column)
    save_data(out_dir, training_features, testing_features, training_labels, 
                testing_labels, training_countries, testing_countries)