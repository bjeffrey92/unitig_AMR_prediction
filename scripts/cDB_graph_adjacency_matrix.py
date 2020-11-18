import sys
import logging
import csv
import pandas as pd
from functools import lru_cache
from itertools import compress
import torch
import numpy as np
import networkx as nx
from torch_sparse import SparseTensor
from scipy import sparse 
from bloom_filter import BloomFilter

from scripts.parse_features import parse_metadata

logging.basicConfig()
logging.root.setLevel(logging.INFO)


def filter_unitigs(rtab_file, files_to_include, filt = (0.01, 0.99)):

    num_unitigs = sum(1 for line in open(rtab_file)) - 1

    with open(rtab_file, 'r') as a:
        reader = csv.reader(a, delimiter = ' ')
        header = next(reader)
        file_filter = [i in files_to_include for i in header] #which to include
        intermediate_unitigs = []
        j = 1
        for row in reader:
            sys.stdout.write(f'\rprocessing {j} of {num_unitigs} unitigs')
            sys.stdout.flush()
            pattern_id = row[0]
            row = list(compress(row, file_filter))
            frequency = sum([1 for i in row if i == '1'])/len(row)
            if frequency >= filt[0] and frequency <= filt[1]:
                intermediate_unitigs.append(pattern_id)
            j += 1
        sys.stdout.write('')
        sys.stdout.flush()
    
    return intermediate_unitigs

@lru_cache(maxsize = None)
def is_intermediate(item: str)-> bool:
    if item not in intermediate_unitigs_bf:
        return False
    else:
        return item in intermediate_unitigs


def parse_graph_adj_matrix(edges_file, nodes_file, mapping_dict):
    unitig_ids = {}    
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
                unitig_ids[node_seq] = node_id
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

    return adj_matrix


def order_adjacency_dict(adjacency_dict):
    def index(l):
        l2 = [None] * len(l)
        l3 = [None] * len(l)
        j = -1
        for i in range(len(l)): 
            if l[i] not in l3: 
                j += 1 
            l2[i] = j 
            l3[i] = l[i] 
        return l2

    keys = list(adjacency_dict.keys())
    keys.sort()
    indexed_keys = index(keys)

    renamed_dict = {}
    i = 0
    for k,v in adjacency_dict.items():
        renamed_values = []
        for node in v:
            if node in keys:
                renamed_node = indexed_keys[keys.index(node)]
                renamed_values.append(renamed_node)
        if renamed_values:
            renamed_dict[indexed_keys[keys.index(k)]] = renamed_values

    return renamed_dict


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


if __name__ == '__main__':

    edges_file = 'data/gonno_unitigs/graph.edges.dbg'
    nodes_file = 'data/gonno_unitigs/graph.nodes'
    unique_rows_file = 'data/gonno_unitigs/unitigs.unique_rows_to_all_rows.txt'
    rtab_file = 'data/gonno_unitigs/unitigs.unique_rows.Rtab' #to filter unitigs based on frequency
    metadata_file = 'data/metadata.csv'
    outcome_column = 'log2_cip_mic'

    #applies frequency filter to unitigs
    metadata = parse_metadata(metadata_file, rtab_file, outcome_column) #to know which files to include
    intermediate_unitigs = filter_unitigs(rtab_file, metadata.index, 
                                        filt = (0.05, 0.95)) 
    
    #to speed up search
    intermediate_unitigs_bf = BloomFilter() 
    for i in intermediate_unitigs:
        intermediate_unitigs_bf.add(i)

    #dict will contain all intermediate frequency graph nodes mapped to their pattern id
    node_to_pattern_id = {}
    pattern_id_to_node = {}
    with open(unique_rows_file, 'r') as a:
        reader = csv.reader(a, delimiter = ' ')
        for row in reader:
            for node in row[2:-1]:
                node_to_pattern_id[node] = row[0]
            pattern_id_to_node[row[0]] = row[2:-1]

    adj_matrix = parse_graph_adj_matrix(edges_file, nodes_file, 
                                        node_to_pattern_id)
    adj_tensor = convert_to_tensor(adj_matrix.tocoo())




    logging.info('Identifying edges between intermediate frequency nodes, can be very slow')
    last_line = subprocess.check_output(['tail', '-1', rtab_file])
    n = int(last_line.decode().split(' ')[0])
    adjacency_dict = {i:[] for i in range(n)} #empty dict of max possible size

    total_lines = total_lines = sum(1 for line in open(edges_file)) #lines in edges file
    with open(edges_file, 'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        i = 0
        for row in reader:
            if int(row[0]) in adjacency_dict:
                if not is_intermediate(row[0]):
                    del adjacency_dict[int(row[0])]
                elif is_intermediate(row[1]):
                    adjacency_dict[int(row[0])].append(int(row[1]))
            i += 1
            sys.stdout.write(f'\rprocessing {i} of {total_lines} edges')
            sys.stdout.flush()

    intermediate_unitigs_integers = list(map(int, intermediate_unitigs))
    filtered_adj_dict = {k:v for k,v in adjacency_dict.items() 
                                        if k in intermediate_unitigs_integers}

    ordered_adj_dict = order_adjacency_dict(filtered_adj_dict)
    dims = len(ordered_adj_dict)
    adj_matrix = sparse.dok_matrix((dims, dims)) #empty sparse matrix     
    deg_matrix = sparse.dok_matrix((dims, dims)) #empty sparse matrix    

    logging.info('Constructing sparse adjacency and degree matrix')
    for node, neighbours in order_adjacency_dict.items():
        for nbr in neighbours:
            adj_matrix[node, nbr] = 1
        deg_matrix[node, node] = len(neighbours)

    logging.info('Converting adjacency and degree matrices to sparse tensors')
    adj_tensor = convert_to_tensor(adj_matrix.tocoo())
    deg_tensor = convert_to_tensor(deg_matrix.tocoo())