import math
import torch
from torch.nn.modules.module import Module

class GraphConvolution(Module):

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin = torch.nn.Linear(in_features, out_features) #encodes weights and bias

    def forward(self, layer_input, adj):
        x = self.lin(layer_input) #Y = WX^T + B
        x = torch.sparse.mm(adj, x)
        # output = torch.stack(tuple(map(torch.mean, torch.unbind(x))))
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolutionPerNode(Module):

    def __init__(self, in_features, out_features):
        super(GraphConvolutionPerNode, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin = torch.nn.Linear(in_features, out_features) #encodes weights and bias

    def forward(self, layer_input, adj):
        x = torch.sparse.mm(adj, layer_input).transpose(0,1) #more efficient than storing dense matrix in memory
        output = self.lin(x) #Y = WX^T + B
        return output 

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'