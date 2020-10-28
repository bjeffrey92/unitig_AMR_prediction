import pandas as pd
import csv
import numpy as np
import torch
import os
import logging

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)


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
        values = []

        i = 0
        for row in reader:
            for j in range(1, len(row)): #first element of row is unitig number
                if row[j] == '1':
                    x_idx.append(j - 1)
                    y_idx.append(i)
                    values.append(1)
            logging.info(f'{i}/{num_unitigs} unitigs processed')
            i += 1

    indices = torch.LongTensor([x_idx, y_idx])
    
    #delete these to free up RAM
    del x_idx
    del y_idx

    shape = (num_samples, num_unitigs)
    values_tensor = torch.FloatTensor(values)

    del values

    features = torch.sparse.FloatTensor(indices, values_tensor, shape)
    return features


def split_training_and_testing(rtab_file, training_split = 0.7):
    '''
    create training and testing rtab files so features can be generated in most memory efficient way
    '''

    num_unitigs = sum(1 for line in open(rtab_file)) - 1

    #get training and testing data as separate lists
    with open(rtab_file, 'r') as a:
        reader = csv.reader(a, delimiter = '\t')
        header = next(reader)
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
            logging.info(f'processing {j} of {num_unitigs} unitigs')
            j += 1

            frequency = sum([int(i) for i in row[1:] if i == '1'])/len(row[1:])
            if frequency < 0.01 or frequency > 0.99: continue #only include intermediate frequency unitigs

            training_rows[i] = row[:training_n + 1]
            testing_rows[i] = row[:1] + row[training_n + 1:]
            
            i += 1

    training_data_file = os.path.join(os.path.dirname(rtab_file), 
                            'training_data_' + os.path.basename(rtab_file))
    testing_data_file = os.path.join(os.path.dirname(rtab_file), 
                            'testing_data_' + os.path.basename(rtab_file))

    logging.info(f'training data being written to {training_data_file}')
    with open(training_data_file, 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = '\t')
        for row in training_rows[:i]:
            writer.writerow(row)

    logging.info(f'testing data being written to {testing_data_file}')
    with open(testing_data_file, 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = '\t')
        for row in testing_rows[:i]:
            writer.writerow(row)

    return training_data_file, testing_data_file


def load_labels(metadata_file, rtab_file):
    metadata_df = pd.read_csv(metadata_file)

    with open(rtab_file, 'r') as a:
        reader = csv.reader(a, delimiter = '\t')
        input_files = next(reader)[1:]
    
    assert all([i.endswith('.contigs_velvet') for i in input_files])    
    accessions = [i.strip('.contigs_velvet') for i in input_files]

