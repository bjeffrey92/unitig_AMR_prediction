import pandas as pd
import csv
import numpy as np
import torch
import logging 

rtab_file = 'data/gonno_unitigs/gonno.rtab'
metadata_file = 'data/metadata.csv'

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
        # values = []

        i = 0
        for row in reader:
            for j in range(1, len(row)): #first element of row is unitig number
                if row[j] == '1':
                    x_idx.append(j - 1)
                    y_idx.append(i)
                    # values.append(1)
            print(f'{i}/{num_unitigs} unitigs processed')
            i += 1

    shape = (num_samples, num_unitigs)
    indices = torch.LongTensor([x_idx, y_idx])
    values = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(indices, 
                                    values,
                                    shape)

def load_labels(metadata_file, rtab_file):
    metadata_df = pd.read_csv(metadata_file)

    with open(rtab_file, 'r') as a:
        reader = csv.reader(a, delimiter = '\t')
        input_files = next(reader)[1:]
    
    assert all([i.endswith('.contigs_velvet') for i in input_files])    
    accessions = [i.strip('.contigs_velvet') for i in input_files]

