import math
import torch
import numpy as np
from torch.nn.modules.module import Module


class SparsePrePrunedLayer(Module):
    def __init__(self, distances):
        super(SparsePrePrunedLayer, self).__init__()

        self.distances = distances
        self.weights = self._build_weights_matrix()
        b = torch.zeros(len(distances))
        torch.nn.init.normal_(b)
        self.bias = b

    def _build_weights_matrix(self):
        total_values = sum([len(v) for v in self.distances.values()])
        indices = [[None] * total_values] * 2

        n = 0
        for key, value in self.distances.items():
            n_values = len(value)
            indices[0][n : n + n_values] = [key] * n_values
            indices[1][n : n + n_values] = list(value.keys())
            n += n_values

        values = torch.zeros(total_values)
        torch.nn.init.normal_(values)
        return torch.sparse_coo_tensor(indices, values)

    def forward(self, layer_input):
        x = torch.sparse.mm(self.weights, layer_input).squeeze(1)
        return x + self.bias


class PreSelectionConvolution(Module):
    def __init__(self, n_nodes, n_neighbours, sum_dimension=1):
        super(PreSelectionConvolution, self).__init__()
        self.n_nodes = n_nodes
        self.n_neighbours = n_neighbours
        self.sum_dimension = sum_dimension

        w = torch.zeros(n_nodes, n_neighbours)
        b = torch.zeros(n_nodes)
        torch.nn.init.xavier_uniform_(w)
        torch.nn.init.normal_(b)
        self.weight = torch.nn.Parameter(w.to_sparse())
        self.bias = torch.nn.Parameter(b.to_sparse())

    def sparse_dense_mul(self, s, d):
        """
        Element wise multiplication
        """
        i = s._indices()
        v = s._values()
        dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
        return torch.sparse.FloatTensor(i, v * dv, s.size())

    def forward(self, layer_input):
        x = layer_input * self.weight
        x = torch.sparse.sum(x, self.sum_dimension)
        return self.bias + x

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_nodes)
            + ","
            + str(self.n_neighbours)
            + " -> "
            + str(self.n_nodes)
            + ")"
        )


class GraphConvolution(Module):
    # arxiv.org/abs/1609.02907
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin = torch.nn.Linear(
            in_features, out_features
        )  # encodes weights and bias

    def forward(self, layer_input, adj):
        x = self.lin(layer_input)  # Y = WX^T + B
        x = torch.sparse.mm(adj, x)
        # output = torch.stack(tuple(map(torch.mean, torch.unbind(x))))
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GraphConvolutionPerNode(Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionPerNode, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin = torch.nn.Linear(
            in_features, out_features
        )  # encodes weights and bias

    def forward(self, layer_input, adj):
        x = torch.sparse.mm(adj, layer_input).transpose(
            0, 1
        )  # more efficient than storing dense matrix in memory
        output = self.lin(x)  # Y = WX^T + B
        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
