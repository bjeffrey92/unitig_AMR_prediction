import pickle
import random
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def colour_code(N, max_attr):
    if N == 'present':
        return 'green'
    i = 255/max_attr
    hex_code = '#%02x%02x%02x' % (int(i * N), int(255 - i * N), 255) 
    return hex_code

def plot_graph(graph, layout, max_attr, with_labels = False):
    G = nx.Graph(graph) #shallow copy
    colour_map = [colour_code(G.nodes[node]['attr'], max_attr) for node in G]
    nx.draw(G, node_size = 150, pos = layout, node_color = colour_map,  
            with_labels = with_labels)
    plt.show()

def get_attribute_vector(graph):
    return np.array([graph.nodes[node]['attr'] for node in graph])

def set_attributes(graph, attributes):
    assert len(attributes) == len(graph)
    i = 0
    for node in graph:
        graph.nodes[node]['attr'] = float(attributes[i])
        i += 1
    return graph

def convolve(graph, adj, deg = None):
    attributes = get_attribute_vector(graph)
    if deg is None:
        new_attributes = np.linalg.multi_dot([adj, attributes])
    else:
        new_attributes = np.linalg.multi_dot([deg, adj, deg, attributes])
    return set_attributes(graph, new_attributes)

if __name__ == '__main__':
    with open('smaller_graph.pkl', 'rb') as a:
        G = pickle.load(a)

    adj = nx.adj_matrix(G)
    adj = adj.toarray()
    I = np.eye(adj.shape[0])
    adj = adj + I #add self loops to each node when convolving

    degs = np.array([v for k,v in dict(nx.degree(G)).items()]) #degree of each node
    normed_deg = np.array([1/math.sqrt(i) for i in degs]) #equivalent to raising degree matrix to power -1/2
    deg = np.diag(normed_deg)

    #label nodes with 1/5 probability of being labelled 1
    for node in G:
        G.nodes[node]['attr'] = np.random.binomial(1, 0.2)

    layout = nx.spring_layout(G) #to keep layout constant
    
    G_1 = convolve(nx.Graph(G), adj)
    G_2 = convolve(nx.Graph(G), adj, deg)

    max_attr = max([G.nodes[node]['attr'] for node in G] + 
                    [G_1.nodes[node]['attr'] for node in G_1] + 
                    [G_2.nodes[node]['attr'] for node in G_2])

    plot_graph(G, layout, max_attr)
    plot_graph(G_1, layout, max_attr)
    plot_graph(G_2, layout, max_attr)