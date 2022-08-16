import numpy as np
import networkx as nx
import torch
from torch_sparse import SparseTensor
from sklearn.cluster import KMeans
import scipy.sparse

from GNN_model.utils import load_adjacency_matrix


def build_graph(adj):
    """
    Get networkx graph from adjacency tensor
    """
    row = adj.indices()[0].tolist()
    col = adj.indices()[1].tolist()

    # initialise graph with all nodes
    G = nx.Graph()
    G.add_nodes_from(set(row))

    # add all edges
    edges_list = [None] * len(row)
    for i in range(len(row)):
        edges_list[i] = (row[i], col[i])
    G.add_edges_from(edges_list)

    return G


def get_degree_matrix(adj):
    row = np.array(adj.indices()[0].tolist())

    # counts of each item in adjacency matrix gives the degree
    degs = np.unique(row, return_counts=True)[1]

    # build matrix
    row = np.array(range(len(degs)))
    col = row

    # csr matrix format
    deg_matrix = scipy.sparse.csr_matrix(
        (degs, (row, col)), shape=(len(degs), len(degs))
    )

    return deg_matrix


def cluster_connected(G, label):
    G_label = nx.Graph()

    node_list = []
    for node in G.nodes:
        if G.nodes[node]["attr"] == label:
            node_list.append(node)
    G_label.add_nodes_from(node_list)

    edge_list = []
    for edge in G.edges:
        if edge[0] in node_list and edge[1] in node_list:
            edge_list.append(edge)
    G_label.add_edges_from(edge_list)

    return nx.is_connected(G_label)


if __name__ == "__main__":
    Ab = "log2_azm_mic"
    data_dir = f"data/model_inputs/freq_5_95/{Ab}"

    adj_tensor = load_adjacency_matrix(data_dir, False)
    adj_tensor = adj_tensor.coalesce()

    G = build_graph(adj_tensor)

    adj = nx.adjacency_matrix(G)  # csr format
    deg = get_degree_matrix(adj_tensor)

    laplacian = adj - deg

    laplacian = laplacian.asfptype()  # change to floating point

    # returns as complex number
    vals, vecs = scipy.sparse.linalg.eigs(laplacian, 6, sigma=0)

    # extra real part from complex plane
    vals = vals.real
    vecs = vecs.real

    vecs = vecs[:, np.argsort(vals)]
    vals = vals[np.argsort(vals)]

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(vecs[:, 1:4])

    for i in range(len(G)):
        node = list(G.nodes)[i]
        G.nodes[node]["attr"] = kmeans.labels_[i]

    for label in np.unique(kmeans.labels_):
        print(label, cluster_connected(G, label))
