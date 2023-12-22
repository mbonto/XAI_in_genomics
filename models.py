import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import load_npz
import os
import json
from torch_geometric.nn import SGConv, GCNConv, global_mean_pool, Sequential, graclus, max_pool_x
from torch_geometric.nn.pool.pool import pool_batch, pool_edge
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.max_pool import _max_pool_x
from torch_geometric.utils.convert import from_scipy_sparse_matrix 
                        

    
def load_model(model_name, n_feat, n_class, softmax, device, save_path, n_layer=None, n_hidden_feat=None, graph_name=None, graph_features=None):
    # Hyperparameters
    if model_name in ['LR', 'DiffuseLR']:
        dropout = 0.2
    elif model_name in ['MLP', 'DiffuseMLP']:
        batch_norm = True
        dropout = 0

    # Models
    if model_name == 'LR':
        model = LogisticRegression(n_feat, n_class, softmax, dropout)
    elif model_name == 'MLP':
        model = MLP(n_layer, n_feat, n_hidden_feat, n_class, batch_norm, softmax, dropout)
    elif model_name == 'DiffuseLR':
        edge_index, edge_weight = load_graph(os.path.join(save_path, 'graph'), 'pearson_correlation.npy', 0.5, device)
        model = DiffuseLR(n_feat, n_class, edge_index, edge_weight, softmax, device, dropout)
    elif model_name == 'DiffuseMLP':
        edge_index, edge_weight = load_graph(os.path.join(save_path, 'graph'), 'pearson_correlation.npy', 0.5, device)
        model = DiffuseMLP(n_layer, n_feat, n_hidden_feat, n_class, edge_index, edge_weight, device, batch_norm, softmax, dropout)
    elif model_name == 'GCN':
        edge_index, edge_weight = load_graph(save_path, graph_name, device, graph_features)
        model = GCN(n_layer, n_feat, 1, n_hidden_feat, n_class, edge_index, edge_weight, softmax)  # here, n_feat is the number of nodes. There is 1 feature per node.

    model.to(device)
    return model
    
    

# Models
class LogisticRegression(nn.Module):
    
    def __init__(self, nb_feat, nb_classes, softmax, dropout=0):
        super(LogisticRegression, self).__init__()
        self.name = 'LR'
        self.variables = {"nb_classes": nb_classes, "nb_feat": nb_feat}
        self.drop = nn.Dropout(p=dropout)
        self.softmax = softmax
        if nb_classes == 2:
            self.fc = nn.Linear(nb_feat, 1)
        else:
            self.fc = nn.Linear(nb_feat, nb_classes)
    
    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x)
        if self.softmax:
            if self.variables["nb_classes"] == 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        else:
            if self.variables["nb_classes"] == 2:
                x = x.reshape(-1)
        return x
    
    

class GCN(nn.Module):

    def __init__(self, n_layer, n_node, n_node_feat, n_hidden_node_feat, n_class, edge_index, edge_weight, softmax):
        super(GCN, self).__init__()
        self.name = 'GCN'
        self.variables = {"nb_classes": n_class, "nb_node": n_node, "nb_node_feat": n_node_feat, "nb_hidden_node_feat": n_hidden_node_feat, "nb_layers": n_layer}
        self.n_layer = n_layer
        self.softmax = softmax

        # Graph
        self.n_node = n_node
        self.n_edge = edge_index.shape[1]

        
        # Coarsening layers
        self.cluster = {}
        self.perm = {}
        # self.batch = {}
        self.edge_index = {}
        self.edge_weight = {}
        self.n_cluster = {}

        # Initial graph
        self.edge_index[0] = edge_index
        self.edge_weight[0] = edge_weight
        # self.batch[0] = torch.zeros(self.n_node)

        # Initial number of clusters = number of nodes in the graph
        self.n_cluster[0] = self.n_node
        
        # Next pooling layers
        for c in range(1, n_layer+1):
            # Graclus reduces by roughly half the number of nodes.
            # It associates a random unmarked node with its closest neighour (if no neighbour, no association). Then, both nodes are marked.
            self.cluster[c], self.perm[c] = consecutive_cluster(graclus(self.edge_index[c-1], num_nodes=self.n_cluster[c-1]))
            # cluster associates each index to a cluster.
            # perm associates each cluster to a node of the cluster (used to determine the batch to which the new nodes belong).
            # print("cluster (n_feat,)", self.cluster.shape)
            # print("perm (n_cluster)", self.perm.shape)

            # All nodes within the same cluster will be represented as one node whose features are the maximum features across these nodes.
            # self.batch[c] = pool_batch(self.perm[c], self.batch[c-1])

            # Edge indices are defined to be the union of the edge indices of all nodes within the same cluster.
            self.edge_index[c], self.edge_weight[c] = pool_edge(self.cluster[c], self.edge_index[c-1], self.edge_weight[c-1])
            # print("edge_index (2, n_edge)", edge_index.shape)
            # print("edge_weight (n_edge,)", edge_weight.shape)

            # Number of clusters after pooling 
            self.n_cluster[c] = torch.unique(self.cluster[c]).shape[0]
            print(f"Number of clusters at layer {c}", self.n_cluster[c])

        
        # Hidden layers
        feat_list = [n_node_feat] + [n_hidden_node_feat] * n_layer 
        layers = []
        for c in range(n_layer):
            layers.append((GCNConv(feat_list[c], feat_list[c+1]), "x, batch_edge_index, batch_edge_weight -> x"))
            layers.append((get_batch_cluster(self.cluster[c+1], self.perm[c+1], self.edge_index[c+1], self.edge_weight[c+1], self.n_cluster[c+1], self.n_cluster[c]), "batch_size, batch -> batch_cluster, batch, batch_edge_index, batch_edge_weight"))
            layers.append((_max_pool_x, "batch_cluster, x -> x"))
            layers.append((nn.ReLU(), "x -> x"))
        self.layers = Sequential("x, batch_edge_index, batch_edge_weight, batch_size, batch", layers)
        

        # Output
        if n_class == 2:
            self.fc = nn.Linear(self.n_cluster[n_layer] * feat_list[-1], 1)
        else:
            self.fc = nn.Linear(self.n_cluster[n_layer] * feat_list[-1], n_class)


    def forward(self, x):
        batch_size = x.shape[0]
        n_node = x.shape[1]
        # print("x (batch_size, n_feat)", x.shape)
        
        # Preparation of the batch for PyG
        batch_edge_index, batch_edge_weight, batch = get_batch_edge_index(self.edge_index[0], self.edge_weight[0], n_node, batch_size)
        x = reshape_batch(x)
        # print("batch_edge_index (2, n_edge x batch_size)", batch_edge_index.shape)
        # print("batch_edge_weight (n_edge x batch_size,)", batch_edge_weight.shape)
        # print("x (batch_size x n_feat, 1)", x.shape)

        # 1. Node embedding + coarsening
        x = self.layers(x, batch_edge_index, batch_edge_weight, batch_size, batch)
        # x = global_mean_pool(x, batch)
        
        # 3. Final classifier
        x = reverse_reshape_batch(x, batch_size)
        #if torch.all(x == 0):
        #    print("All features degenerate to 0!")
        x = self.fc(x)
        
        if self.softmax:
            if self.variables["nb_classes"] == 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        else:
            if self.variables["nb_classes"] == 2:
                x = x.reshape(-1)
        return x
        


class get_batch_cluster(nn.Module):
    def __init__(self, cluster, perm, edge_index, edge_weight, n_cluster, n_cluster_initial):
        super(get_batch_cluster, self).__init__()
        self.cluster = cluster
        self.perm = perm
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.n_cluster = n_cluster
        self.n_cluster_initial = n_cluster_initial
        self.batch_size = None

    def forward(self, batch_size, batch):
        if batch_size != self.batch_size:
            self.batch_cluster = self.cluster
            self.batch_perm = self.perm
            self.batch_edge_index = self.edge_index
            self.batch_edge_weight = self.edge_weight
            for b in range(1, batch_size):
                self.batch_cluster = torch.cat((self.batch_cluster, self.cluster + self.n_cluster * b))
                self.batch_perm = torch.cat((self.batch_perm, self.perm + self.n_cluster_initial * b))
                self.batch_edge_index = torch.cat((self.batch_edge_index, self.edge_index + self.n_cluster * b), dim=1)
                self.batch_edge_weight = torch.cat((self.batch_edge_weight, self.edge_weight))
            self.batch = pool_batch(self.batch_perm, batch)
            self.batch_size = batch_size
        return self.batch_cluster, self.batch, self.batch_edge_index, self.batch_edge_weight




class MLP(nn.Module):
    def __init__(self, nb_layers, nb_feat, nb_hidden_feat, nb_classes, batch_norm=False, softmax=False, dropout=0):
        super(MLP, self).__init__()
        self.name = 'MLP'
        self.variables = {"nb_classes": nb_classes, "nb_feat": nb_feat, "nb_hidden_feat": nb_hidden_feat, 
                         "nb_layers": nb_layers}
        self.drop = nn.Dropout(p=dropout)
        self.softmax = softmax
        
        # Hidden layers
        feat_list = [nb_feat] + [nb_hidden_feat] * nb_layers 
        layers = []       
        for i in range(nb_layers):
            layers.append(nn.Linear(feat_list[i], feat_list[i+1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(feat_list[i+1]))
            layers.append(nn.ReLU())  
        self.layers = nn.Sequential(*layers)
        
        # Output
        if nb_classes == 2:
            self.fc = nn.Linear(feat_list[-1], 1)
        else:
            self.fc = nn.Linear(feat_list[-1], nb_classes)
    
    def forward(self, x):
        x = self.drop(x)
        x = self.layers(x)
        x = self.fc(x)
        if self.softmax:
            if self.variables["nb_classes"] == 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        else:
            if self.variables["nb_classes"] == 2:
                x = x.reshape(-1)
        return x
    
    

class DiffuseMLP(nn.Module):
    
    def __init__(self, nb_layers, nb_feat, nb_hidden_feat, nb_classes, edge_index, edge_weight, device, batch_norm=False, softmax=False, dropout=0):
        super(DiffuseMLP, self).__init__()
        self.name = 'DiffuseMLP'
        self.variables = {"nb_classes": nb_classes, "nb_feat": nb_feat, "nb_hidden_feat": nb_hidden_feat, 
                         "nb_layers": nb_layers}
        self.drop = nn.Dropout(p=dropout)
        self.softmax = softmax
        
        # Diffusion
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.n_node = nb_feat
        self.conv = SGConv(1, 1, bias=False)
        self.conv.lin.weight = nn.Parameter(torch.ones(1))
        self.conv.lin.weight.requires_grad = False
        self.device = device
        
        # Hidden layers
        feat_list = [nb_feat] + [nb_hidden_feat] * nb_layers 
        layers = []       
        for i in range(nb_layers):
            layers.append(nn.Linear(feat_list[i], feat_list[i+1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(feat_list[i+1]))
            layers.append(nn.ReLU())  
        self.layers = nn.Sequential(*layers)
        
        # Output
        self.fc = nn.Linear(feat_list[-1], nb_classes)
    
    def forward(self, x):
        # Diffusion
        x, batch_size = reshape_batch(x)
        batch_edge_index, batch_edge_weight, batch = get_batch_edge_index(self.edge_index, self.edge_weight, self.n_node, batch_size)
        batch = batch.to(self.device)
        x = self.conv(x, batch_edge_index, batch_edge_weight)
        # MLP
        x = reverse_reshape_batch(x, batch_size)
        x = self.drop(x)
        x = self.layers(x)
        x = self.fc(x)
        if self.softmax:
            x = F.softmax(x, dim=1)
        return x



class DiffuseLR(nn.Module):
    
    def __init__(self, nb_feat, nb_classes, edge_index, edge_weight, softmax, device, dropout=0):
        """
        # The shape of x is (batch_size, nb_feat). Here, each feature is seen as a node in a graph.
        # To diffuse infomation on the graph with layers from torch_geometric, we need to reshape x.
        # Indeed, torch_geometric does not consider batches. The shape of the input of a layer is 
        # (nb_node, nb_feat_per_node). Here, nb_feat_per_node is 1.
        # Thus, we merge the examples contain in the batch in one big graph (nb_feat*batch_size, nb_feat_per_node).
        # Warning: make sure the edge list is modified accordingly.
        
        Parameters:
            nb_feat  -- int, number of features of each example
            nb_node_feat  -- int, number of node features after a GCN layer
            nb_classes  -- int, number of classes
            edge_index  -- torch tensor of size (2, n_edge)
            edge_weight  --  torch tensor of size (n_edge)
        """
        super(DiffuseLR, self).__init__()
        self.name = 'DiffuseLR'
        self.variables = {"nb_classes": nb_classes, "nb_feat": nb_feat}
        self.drop = nn.Dropout(p=dropout)
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.n_node = nb_feat
        self.softmax = softmax
        self.conv = SGConv(1, 1, bias=False)
        self.conv.lin.weight = nn.Parameter(torch.ones(1))
        self.conv.lin.weight.requires_grad = False
        self.fc = nn.Linear(nb_feat, nb_classes)
        self.device = device
    
    def forward(self, x):
        # Diffusion
        x, batch_size = reshape_batch(x)
        batch_edge_index, batch_edge_weight, batch = get_batch_edge_index(self.edge_index, self.edge_weight, self.n_node, batch_size)
        batch = batch.to(self.device)
        x = self.conv(x, batch_edge_index, batch_edge_weight)
        # LR
        x = reverse_reshape_batch(x, batch_size)
        x = self.drop(x)
        x = self.fc(x)
        if self.softmax:
            x = F.softmax(x, dim=1)
        return x
    


def get_batch_edge_index(edge_index, edge_weight, nb_feat, batch_size):
    """
    From (batch_size, nb_feat), get (nb_feat x batch_size, nb_feat_per_node (here 1))
    Parameters:
        edge_index  --  torch tensor of size (2, n_edge)
        edge weight  --  torch tensor of size (n_edge)
        nb_feat  --  int, here, number of features and number of nodes in a single graph
        batch_size  --  int, number of examples
    """
    batch_edge_index = edge_index
    batch_edge_weight = edge_weight
    batch = torch.zeros(size=[nb_feat]).type(torch.int64)
    for b in range(1, batch_size):
        batch_edge_index = torch.cat((batch_edge_index, edge_index + nb_feat * b), dim=1)
        batch_edge_weight = torch.cat((batch_edge_weight, edge_weight))
        batch = torch.cat((batch, torch.ones(size=[nb_feat]).type(torch.int64) * b))
    return batch_edge_index, batch_edge_weight, batch.to(edge_index.device)


def edge_index_from_adjacency(A):
    """
    Return a torch tensor edge_index of size (2, n_edge) and a torch tensor edge_weight of size (n_edge).
    
    Parameter:
        A  -- scipy sparse square matrix, adjacency matrix of a graph.
    """
    return from_scipy_sparse_matrix(A)


def reshape_batch(x):
    """Reshape x from (batch_size, n_node) to (batch_size x n_node, n_feat_per_node (here 1)).
    """
    return torch.reshape(x, [-1, 1])  # , x.shape[0]


def reverse_reshape_batch(x, batch_size):
    """Reshape x from (batch_size x n_node, n_feat_per_node) to (batch_size, n_node * n_feat_per_node).
    """
    return torch.reshape(x, (batch_size, -1))


def load_graph(save_path, name, device, graph_features):
    # Adjacency matrix
    A = load_npz(os.path.join(save_path, "graph", name))
    # Special case: select particular features
    if graph_features is not None:
        feat_name = np.array(json.load(open(os.path.join(save_path, "genesIds.txt"))))
        assert len(feat_name) == A.shape[0]
        indices = [np.argwhere(feat == feat_name)[0, 0] for feat in graph_features]
        A = A[:, indices]
        A = A[indices, :]
        feat_name = feat_name[indices]
        assert (feat_name == graph_features).all()
    # PyG format
    edge_index, edge_weight = edge_index_from_adjacency(A)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(dtype=torch.float32).to(device)
    return edge_index, edge_weight
