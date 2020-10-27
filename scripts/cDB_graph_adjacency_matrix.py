import pyfrost #https://github.com/broadinstitute/pyfrost
# import tensorflow as tf
# import numpy as np
from scipy import sparse 

gfa_file = 'data/gonno_unitigs/gonno_unitigs.gfa'
unitigs_fasta = 'data/gonno_unitigs/gonno_unitigs_unitigs.fasta'

g = pyfrost.load(gfa_file) #path to bfg_colours file is inferred


def map_nodes(node_list, nodes_dict):    
    node_uuids = []
    for n in node_list:
        try:
            node_uuids.append(int(nodes_dict[n]) - 1)
        except KeyError: pass
    return node_uuids


if __name__ == '__main__':
    #map unitig sequences to their uuid
    nodes_dict = {}
    with open(unitigs_fasta, 'r') as a:
        for line in a:
            if line.startswith('>'):
                kmer = line.strip('>*\n')
                continue
            else: 
                nodes_dict[line.strip('\n')] = kmer

    #get adjacency of all nodes in unitigs.fasta
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

    #fill in sparse matrix
    for unitig, neighbours in unitig_adjacency.items():
        for nbr in neighbours:
            adj_matrix[unitig, nbr] = 1
            adj_matrix[nbr, unitig] = 1 

    adj_matrix = adj_matrix.tocoo()
    sparse.save_npz('cDB_graph_adjacency_matrix.npz', adj_matrix)

