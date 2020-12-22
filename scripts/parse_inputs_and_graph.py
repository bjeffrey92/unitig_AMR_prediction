import math
import sys
import logging
import csv
import pandas as pd
import tempfile
import torch
import os
import numpy as np
import networkx as nx
import pickle
from itertools import compress
from torch_sparse import SparseTensor
from scipy.sparse import identity, csr_matrix

logging.basicConfig()
logging.root.setLevel(logging.INFO)


def parse_graph_adj_matrix(edges_file, nodes_file, mapping_dict, norm = True):
    '''
    norm is whether or not to normalised the adjacency matrix by multiplication
    with symmetric normalised degree matrix
    '''

    G = nx.Graph()
    # Add nodes first
    node_list = []
    with open(nodes_file, 'r') as node_file:
        for node in node_file:
            (node_id, node_seq) = node.rstrip().split("\t")
            if node_id in mapping_dict:
                node_list.append((int(node_id), 
                                    dict(seq=node_seq, 
                                    seq_len=len(node_seq))))
    G.add_nodes_from(node_list)

    # add edges
    edge_list = []
    with open(edges_file, 'r') as edge_file:
        for edge in edge_file:
            (start, end, label) = edge.rstrip().split("\t")
            if start in mapping_dict and end in mapping_dict:
                edge_list.append((int(start), int(end)))
    G.add_edges_from(edge_list)

    adj_matrix = nx.adjacency_matrix(G)
    I = identity(adj_matrix.shape[0])
    adj_matrix = adj_matrix + I #so every node is connected to itself

    if norm:
        degs = np.array([v + 1 for k,v in dict(nx.degree(G)).items()]) #degree of each node
        normed_deg = np.array([1/math.sqrt(i) for i in degs]) #equivalent to raising degree matrix to power -1/2
        row = np.array(range(len(normed_deg)))
        col = row 
        deg_matrix = csr_matrix((normed_deg, (row, col)), 
                                    shape = (len(normed_deg), len(normed_deg)))
        adj_matrix = deg_matrix * adj_matrix * deg_matrix

    return adj_matrix


def convert_to_tensor(matrix, torch_sparse_coo = True):
    
    shape = matrix.shape
    
    row = torch.LongTensor(matrix.row)
    col = torch.LongTensor(matrix.col)
    value = torch.Tensor(matrix.data)
    
    sparse_tensor = SparseTensor(row = row, col = col, 
                        value = value, sparse_sizes = shape)

    if torch_sparse_coo:
        return sparse_tensor.to_torch_sparse_coo_tensor()
    else:
        return sparse_tensor


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

        i = 1
        j = 1
        for row in reader:
            sys.stdout.write(f'\rprocessing {j} of {num_unitigs} unitigs')
            sys.stdout.flush()
            row = list(compress(row, filt))
            j += 1

            frequency = sum([1 for i in row[1:] if i == '1'])/len(row[1:])
            if frequency < freq_filt[0] or frequency > freq_filt[1]: continue #only include intermediate frequency unitigs

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


def load_features(rtab_file, mapping_dict, adj_tensor):
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
    shape = (num_samples, adj_tensor.shape[1])
    
    #delete these to free up RAM
    del x_idx
    del y_idx

    values_tensor = torch.FloatTensor(values)

    del values

    features = torch.sparse_coo_tensor(indices, values_tensor, shape)
    return features


def filter_unitigs(training_features, testing_features, adj):
    '''
    Removes features which are all 0 because they were filtered out and 
    transforms adjacency matrix so it has the correct dimensionality 
    '''

    training_features_trans = training_features.to_dense().transpose(0,1)
    testing_features_trans = testing_features.to_dense().transpose(0,1)
    
    #create list of every feature which is represented at least once
    present_unitigs = []
    for i in range(len(testing_features_trans)):
        if testing_features_trans[i].tolist().count(1) != 0 or \
            training_features_trans[i].tolist().count(1) != 0:
            present_unitigs.append(i)

    #check which indices are in present and map them to new id  
    adj = adj.coalesce()
    x_idx = adj.indices()[0].tolist()
    y_idx = adj.indices()[1].tolist()
    values = adj.values().tolist()

    #convert to dataframes to allow merging
    df = pd.DataFrame({'x':x_idx, 'y':y_idx, 'values': values})
    present_unitigs_df = pd.DataFrame({'graph_node':present_unitigs,
                                'unitig_no':list(range(len(present_unitigs)))})

    #sequential merges to map graph nodes to position in the features matrix
    merged_df = df.merge(present_unitigs_df, left_on = 'x', 
                        right_on = 'graph_node')[['x', 'y', 
                                                'unitig_no', 'values']]
    merged_df = merged_df.merge(present_unitigs_df, left_on = 'y', 
                                right_on = 'graph_node')

    #build new adjacency tensor
    indices = torch.FloatTensor([merged_df.unitig_no_x.to_list(), 
                                  merged_df.unitig_no_y.to_list()])
    values = torch.FloatTensor(merged_df['values'].to_list())
    adj_tensor = torch.sparse_coo_tensor(indices, values)
    adj_tensor = adj_tensor.type(torch.FloatTensor)

    #build new feature tensors
    filtered_training_features = torch.stack([training_features_trans[i] \
                                                for i in present_unitigs])
    filtered_testing_features = torch.stack([testing_features_trans[i] \
                                                for i in present_unitigs])
    filtered_training_features = \
            filtered_training_features.transpose(0,1).to_sparse()
    filtered_testing_features = \
            filtered_testing_features.transpose(0,1).to_sparse()

    return adj_tensor, filtered_training_features, filtered_testing_features


def get_distances(adj, n = 20):
    '''
    adj: sparse adjacency tensor
    n: max number of steps between nodes to consider
    '''
    #construct networkx graph and use this to build distance matrix
    G = nx.Graph()

    #nodes first then edges
    node_list = list(range(adj.shape[0]))
    G.add_nodes_from(node_list)
    adj = adj.coalesce()
    indices = adj.indices().tolist()
    edge_list = [(indices[0][i], indices[1][i]) for i in range(len(indices[0]))]
    G.add_edges_from(edge_list)

    #shorted distance between each pair of connected nodes
    shortest_path_lengths = nx.all_pairs_shortest_path_length(G)

    def _max_n_steps(distances, n):
        return {k:v for k,v in distances.items() if v <= n}
    #return as dictionary keeping only distance of max n steps
    return {node:_max_n_steps(distances, n) for node, distances in shortest_path_lengths}


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
            torch.FloatTensor([(lambda x: 1 if x == i else 0)(x) \
                                                    for x in countries])

    def parse_country(country):
        return country_tensors[country]

    return torch.stack(metadata.Country.apply(parse_country).to_list())


def load_families(metadata, families):
    family_tensors = {}
    for i in families:
        family_tensors[i] = \
            torch.FloatTensor([(lambda x: 1 if x == i else 0)(x) \
                                                        for x in families])

    def parse_family(family):
        return family_tensors[family]

    return torch.stack(metadata.Family.apply(parse_family).to_list())
    

def save_data(out_dir, training_features, testing_features, 
                training_labels, testing_labels, adjacency_matrix,
                distances, training_metadata, testing_metadata,
                training_countries = None, testing_countries = None):
    
    torch.save(training_features, os.path.join(out_dir, 'training_features.pt'))
    torch.save(testing_features, os.path.join(out_dir, 'testing_features.pt'))
    torch.save(training_labels, os.path.join(out_dir, 'training_labels.pt'))
    torch.save(testing_labels, os.path.join(out_dir, 'testing_labels.pt'))
    torch.save(adjacency_matrix, os.path.join(out_dir, 
                                            'unitig_adjacency_tensor.pt'))

    with open(os.path.join(out_dir, 'distances_dict.pkl'), 'wb') as a:
        pickle.dump(distances, a)

    training_metadata.to_csv(os.path.join(out_dir, 'training_metadata.csv'),
                            index = False)
    testing_metadata.to_csv(os.path.join(out_dir, 'testing_metadata.csv'),
                            index = False)

    if training_countries is not None:
        torch.save(training_countries, 
                os.path.join(out_dir, 'training_countries.pt'))
    if testing_countries is not None:
        torch.save(testing_countries, 
                os.path.join(out_dir, 'testing_countries.pt'))


if __name__ == '__main__':

    edges_file = 'data/gonno_unitigs/graph.edges.dbg'
    nodes_file = 'data/gonno_unitigs/graph.nodes'
    unique_rows_file = 'data/gonno_unitigs/unitigs.unique_rows_to_all_rows.txt'
    rtab_file = 'data/gonno_unitigs/unitigs.unique_rows.Rtab' #to filter unitigs based on frequency
    metadata_file = 'data/metadata.csv'
    outcome_columns = ['log2_azm_mic', 'log2_cfx_mic', 
                    'log2_cip_mic', 'log2_cro_mic']

    logging.info('Mapping pattern ids to graph nodes')
    #dicts will contain all intermediate frequency graph nodes mapped to their pattern id
    node_to_pattern_id = {}
    pattern_id_to_node = {}
    with open(unique_rows_file, 'r') as a:
        reader = csv.reader(a, delimiter = ' ')
        for row in reader:
            for node in row[2:-1]:
                node_to_pattern_id[node] = row[0]
            pattern_id_to_node[row[0]] = row[2:-1]

    logging.info('Constructing graph adjacency matrix')
    #component_nodes is set of all the nodes in the largest component
    adj_matrix = parse_graph_adj_matrix(edges_file, nodes_file, 
                                        node_to_pattern_id, norm = True)  
    
    adj_tensor = convert_to_tensor(adj_matrix.tocoo())

    for outcome_column in outcome_columns:
        metadata = parse_metadata(metadata_file, rtab_file, outcome_column) #to know which files to include

        #alphabetical list of countries
        # countries = metadata.Country.unique()
        # countries.sort()
        # countries = countries.tolist()
        
        #if don't wish to specify countries
        countries = []

        with tempfile.NamedTemporaryFile() as a, \
                tempfile.NamedTemporaryFile() as b:
            training_rtab_file = a.name
            testing_rtab_file = b.name

            if countries:
                split_training_and_testing(rtab_file, 
                                            metadata.index, 
                                            training_rtab_file, 
                                            testing_rtab_file,
                                            metadata, 
                                            country_split = True,
                                            freq_filt = (0.05, 0.95))
            else:
                split_training_and_testing(rtab_file, 
                                            metadata.index, 
                                            training_rtab_file, 
                                            testing_rtab_file,
                                            freq_filt = (0.05, 0.95))

            #reads in rtab as sparse feature tensor
            training_features = load_features(training_rtab_file, 
                                            pattern_id_to_node, 
                                            adj_tensor)
            testing_features = load_features(testing_rtab_file,
                                            pattern_id_to_node, 
                                            adj_tensor)

            #removes all unitigs which were filtered out and reshapes adjacency tensor
            adj, training_features, testing_features = \
                 filter_unitigs(training_features, testing_features, adj_tensor)

            distances = get_distances(adj)

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
                    testing_labels, adj, distances, training_metadata, testing_metadata,
                    training_countries, testing_countries)