import pickle
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def colour_code(node, max_attr):
    '''
    Converts node value to a hex code with colour relative to max node value 
    '''
    N = 255/max_attr
    hex_code = '#%02x%02x%02x' % (int(N * node), int(255 - N * node), 255) 
    return hex_code


def coloured_plotting(G):
    color_map = [colour_code(G.nodes[node]['attr']) for node in G]
    nx.draw(G, node_color = color_map, with_labels = False)
    plt.show()

def get_attribute_vector(G):
    attributes = []
    for node in G:
        attributes.append(G.nodes[node]['attr'])
    return np.array(attributes)

def set_attributes(G, attributes):
    assert len(attributes) == len(G)
    i = 0
    for node in G:
        G.nodes[node]['attr'] = float(attributes[i])
        i += 1
    return G

def convolve(G, adj = adj):
    attributes = get_attribute_vector(G)
    new_attributes = np.linalg.linalg.matmul(adj, attributes)
    # normed_attributes = new_attributes/max(new_attributes)
    G = set_attributes(G, new_attributes)
    return G

if __name__ == '__main__':
    with open('small_graph.pkl', 'rb') as a:
        G = pickle.load(a)

    adj = nx.adj_matrix(G)
    adj = adj.toarray()

    #label nodes with 1/5 probability of being labelled 1
    for node in G:
        G.nodes[node]['attr'] = np.random.binomial(1, 0.2)

    coloured_plotting(G)

    G = convolve(G, adj)