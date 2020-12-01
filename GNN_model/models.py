import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN_model.layers import GraphConvolution, GraphConvolutionPerNode


class GCNGlobalNode(nn.Module):
    def __init__(self, n_feat, conv_1, n_hid, out_dim, dropout):
        super(GCNGlobalNode, self).__init__()

        self.gc1 = GraphConvolution(n_feat, conv_1)
        # self.gc2 = GraphConvolution(conv_1, conv_2)
        self.linear1 = nn.Linear(conv_1, n_hid)
        self.linear2 = nn.Linear(n_hid, out_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.leaky_relu(self.gc1(x, adj))[-1] #select last node which is the global node
        # x = F.leaky_relu(self.gc2(x, adj))[-1] 
        x = F.leaky_relu(self.linear1(x))
        F.dropout(x, self.dropout, inplace = True, training = True)
        out = F.leaky_relu(self.linear2(x))[0]
        return out


class GCNMaxPooling(nn.Module):
    def __init__(self, n_feat, conv_1, conv_2, n_hid_1, n_hid_2, 
                out_dim, dropout):
        super(GCNMaxPooling, self).__init__()

        self.gc1 = GraphConvolution(n_feat, conv_1)
        
        #if more than one convolutional layer
        if conv_2 > 0:
            self.gc2 = GraphConvolution(n_feat, conv_2)
        else:
            self.gc2 = None
        
        self.linear1 = nn.Linear(n_feat, n_hid_1)
        
        #if more than one densely connected hidden layer
        if n_hid_2 > 0:
            self.linear2 = nn.Linear(n_hid_1, n_hid_2)
            self.linear3 = nn.Linear(n_hid_2, out_dim)
        else:
            self.linear2 = nn.Linear(n_hid_1, out_dim)
            self.linear3 = None
        
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.leaky_relu(self.gc1(x, adj))
        
        if self.gc2 is not None:
            x, _ = torch.max(x, 1) #global max pooling
            x = x.unsqueeze(1)
            x = F.leaky_relu(self.gc2(x, adj))
        
        x, _ = torch.max(x, 1) #global max pooling
        x = x.unsqueeze(1)
        
        x = F.leaky_relu(self.linear1(x))
        F.dropout(x, self.dropout, inplace = True, training = True)
        
        x = F.leaky_relu(self.linear2(x))
        if self.linear3 is not None:
            F.dropout(x, self.dropout, inplace = True, training = True)
            x = F.leaky_relu(self.linear3(x))

        return x[0][0]


class GCNPerNode(nn.Module):
    def __init__(self, n_feat, n_hid_1, n_hid_2, out_dim, dropout, 
                n_hid_3 = None):
        super(GCNPerNode, self).__init__()

        self.gc = GraphConvolutionPerNode(n_feat, n_hid_1)
        self.linear1 = nn.Linear(n_hid_1, n_hid_2)
        if n_hid_3:
            self.linear2 = nn.Linear(n_hid_2, n_hid_3)
            self.linear3 = nn.Linear(n_hid_3, out_dim)
        else:
            self.linear2 = nn.Linear(n_hid_2, out_dim)
            self.linear3 = None
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.leaky_relu(self.gc(x, adj))
        # F.dropout(x, self.dropout, inplace = True, training = True)
        x = F.leaky_relu(self.linear1(x))
        F.dropout(x, self.dropout, inplace = True, training = True)
        if self.linear3 is not None:
            x = F.leaky_relu(self.linear2(x))
            F.dropout(x, self.dropout, inplace = True, training = True)
            out = F.leaky_relu(self.linear3(x)[0][0])
        else:
            out = F.leaky_relu(self.linear2(x)[0][0])
        return out


class VanillaNN(nn.Module):
    def __init__(self, n_feat, n_hid_1, n_hid_2, out_dim, dropout):
        super(VanillaNN, self).__init__()

        self.linear1 = nn.Linear(n_feat, n_hid_1)
        self.linear2 = nn.Linear(n_hid_1, n_hid_2)
        self.linear3 = nn.Linear(n_hid_2, out_dim)
        self.dropout = dropout

    def forward(self, x, adj = None): #add unused adj argument so same training functions can be used for now
        x = F.leaky_relu(self.linear1(x.transpose(0,1)))
        F.dropout(x, self.dropout, inplace = True, training = True)
        x = F.leaky_relu(self.linear2(x))
        F.dropout(x, self.dropout, inplace = True, training = True)
        out = F.leaky_relu(self.linear3(x)[0][0])
        return out

