import pyfrost #https://github.com/broadinstitute/pyfrost
import torch
import numpy as np
import csv
from scipy import sparse 
from itertools import compress
from scripts.parse_features import parse_metadata
import logging

logging.basicConfig()
logging.root.setLevel(logging.INFO)

def map_nodes(node_list, nodes_dict):    
    node_uuids = []
    for n in node_list:
        try:
            node_uuids.append(int(nodes_dict[n]) - 1)
        except KeyError: pass
    return node_uuids


def filter_unitigs(rtab_file, files_to_include, filt = (0.01, 0.99)):

    with open(rtab_file, 'r') as a:
        reader = csv.reader(a, delimiter = '\t')
        header = next(reader)
        file_filter = [i in files_to_include for i in header] #which to include
        intermediate_unitigs = []
        for row in reader:
            pattern_id = row[0]
            row = list(compress(row, file_filter))
            frequency = sum([1 for i in row if i == '1'])/len(row)
            if frequency >= filt[0] and frequency <= filt[1]:
                intermediate_unitigs.append(pattern_id)

    return intermediate_unitigs


def convert_to_tensor(adj_matrix):
    values = adj_matrix.data
    indices = np.vstack((adj_matrix.row, adj_matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = adj_matrix.shape

    return torch.sparse_coo_tensor(i, v, torch.Size(shape))


if __name__ == '__main__':

    gfa_file = 'data/gonno_unitigs/gonno_unitigs.gfa'
    unitigs_fasta = 'data/gonno_unitigs/gonno_unitigs_unitigs.fasta'
    rtab_file = 'data/gonno_unitigs/gonno.rtab' #to filter unitigs based on frequency
    metadata_file = 'data/metadata.csv'

    g = pyfrost.load(gfa_file) #path to bfg_colours file is inferred

    #applies frequency filter to unitigs
    metadata = parse_metadata(metadata_file, rtab_file) #to know which files to include
    intermediate_unitigs = filter_unitigs(rtab_file, metadata.index) 

    logging.info('Identifying intermediate frequency nodes and mapping them to uuids, this could take a while')
    #map unitig sequences to their uuid
    nodes_dict = {}
    with open(unitigs_fasta, 'r') as a:
        for line in a:
            if line.startswith('>'):
                kmer = line.strip('>*\n')
                continue
            else: 
                if kmer in intermediate_unitigs:
                    nodes_dict[line.strip('\n')] = kmer

    logging.info('Extracting adjacency')
    #get adjacency of all intermediate nodes in unitigs.fasta
    adjacency_dict = {}
    for n, neighbours in g.adj.items():
        node = g.nodes[n]['unitig_sequence']
        if node not in nodes_dict: continue #graph includes all kmers
        adjacency_dict[node] = []
        for nbr in neighbours:
            adjacency_dict[node].append(g.nodes[nbr]['unitig_sequence'])

    #dictionary of unitig uuids, value is list of adjacent unitigs
    unitig_adjacency = {int(nodes_dict[k]) - 1 :map_nodes(v, nodes_dict) \
        for k, v in adjacency_dict.items()}

    dims = len(unitig_adjacency)
    adj_matrix = sparse.dok_matrix((dims, dims)) #empty sparse matrix 

    logging.info('Constructing sparse adjacency matrix')
    #fill in sparse matrix
    unitig_order = list(unitig_adjacency.keys())
    for unitig, neighbours in unitig_adjacency.items():
        unitig_pos = unitig_order.index(unitig)
        for nbr in neighbours:
            nbr_pos = unitig_order.index(nbr)
            adj_matrix[unitig_pos, nbr_pos] = 1
            adj_matrix[nbr_pos, unitig_pos] = 1 

    logging.info('Converting adjacency matrix to sparse tensor')
    adj_tensor = convert_to_tensor(adj_matrix.tocoo())
