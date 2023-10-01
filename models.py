import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import load_npz
import os
from torch_geometric.nn import SGConv, GCNConv, global_mean_pool, Sequential, graclus, max_pool_x
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.max_pool import _max_pool_x
# from torch_geometric.utils import scatter
from torch_geometric.utils.convert import from_scipy_sparse_matrix 
                        

    
def load_model(model_name, n_feat, n_class, softmax, device, save_path, n_layer=None, n_hidden_feat=None, graph_name=None):
    # Hyperparameters
    if model_name in ['LR', 'DiffuseLR']:
        dropout = 0.2
    elif model_name in ['MLP', 'DiffuseMLP']:
        batch_norm = True
        dropout = 0
    elif model_name in ['GCN']:
        pass

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
        edge_index, edge_weight = load_graph(os.path.join(save_path, 'graph'), graph_name, device)
        model = GCN(n_layer, 1, n_hidden_feat, n_class, edge_index, edge_weight, softmax)

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
        self.fc = nn.Linear(nb_feat, nb_classes)
    
    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x)
        if self.softmax:
            x = F.softmax(x, dim=1)
        return x
    
    

#def _max_pool_x(
#    cluster: Tensor,
#    x: Tensor,
#    size: Optional[int] = None,
#) -> Tensor:
#    return scatter(x, cluster, dim=0, dim_size=size, reduce='max')



class GCN(nn.Module):

    def __init__(self, n_layer, n_node_feat, n_hidden_node_feat, n_class, edge_index, edge_weight, softmax):
        super(GCN, self).__init__()
        self.name = 'GCN'
        self.variables = {"nb_classes": n_class, "nb_node_feat": n_node_feat, "nb_hidden_node_feat": n_hidden_node_feat, "nb_layers": n_layer}
        self.softmax = softmax

        # Graph
        self.edge_index = edge_index
        self.edge_weight = edge_weight

        # Hidden layers
        feat_list = [n_node_feat] + [n_hidden_node_feat] * n_layer 
        layers = []       
        for i in range(n_layer):
            layers.append((GCNConv(feat_list[i], feat_list[i+1]),
                           "x, edge_index, edge_weight -> x")
                           )
            layers.append(nn.ReLU())  
        self.layers = Sequential("x, edge_index, edge_weight", layers)
        
        # Output
        self.fc = nn.Linear(feat_list[-1], n_class)

    def forward(self, x):
        # Preparation of the batch for PyG
        batch_size = x.shape[0]
        n_node = x.shape[1]
        batch_edge_index, batch_edge_weight, batch = get_batch_edge_index(self.edge_index, self.edge_weight, n_node, batch_size)
        print("x (batch_size, n_feat)", x.shape)
        x = reshape_batch(x)
        print("x (batch_size x n_feat, 1)", x.shape)

        # 1. Node embedding
        x = self.layers(x, batch_edge_index, batch_edge_weight)
        print("x (batch_size x n_feat, n_hidden_dim)", x.shape)

        # 2. Coarsening layer
        ## All nodes within the same cluster will be represented as one node.
        ## Final node features are defined by the maximum features of all nodes within the same cluster.
        ## Edge indices are defined to be the union of the edge indices of all nodes within the same cluster.
        # Algorithm associating a node with its best neighour (if no neighbour, no association)
        # An unmarked node is chosen at random. After the association, the node and its neighbour are marked. 
        # Reduce by roughly half the number of nodes
        cluster = graclus(batch_edge_index, num_nodes=n_node)
        print("cluster (batch_size x n_feat)", cluster.shape)
        print("    first elements", cluster[:10])
        print("    number of clusters", torch.unique(cluster).shape[0])
        # cluster associates each index to a cluster
        # perm associates each cluster to a node of the cluster (used to determine the batch to which the new nodes belong)
        cluster, perm = consecutive_cluster(cluster)
        print("cluster (batch_size x n_feat)", cluster.shape)
        print("perm (n_cluster)", perm.shape)
        # All nodes within the same cluster will be represented as one node.
        # Final node features are defined by the maximum features of all nodes within the same cluster.
        x = _max_pool_x(cluster, x)
        print("x (n_cluster, n_hidden_dim)", x.shape)
        # batch_edge_index, batch_edge_weight = pool_edge(cluster, batch_edge_index, batch_edge_weight)
        # batch = pool_batch(perm, data.batch)
        # x, batch = max_pool_x(cluster, x, batch)
        # x = global_mean_pool(x, batch)
        if torch.all(x == 0):
            print("All features degenerate to 0!")

        # 3. Final classifier
        x = self.fc(x)
        
        if self.softmax:
            x = F.softmax(x, dim=1)
        return x
        


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
        self.fc = nn.Linear(feat_list[-1], nb_classes)
    
    def forward(self, x):
        x = self.drop(x)
        x = self.layers(x)
        x = self.fc(x)
        if self.softmax:
            x = F.softmax(x, dim=1)
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
    return torch.reshape(x, [batch_size, -1])


def load_graph(save_path, name, device):
    A = load_npz(os.path.join(save_path, name))
    edge_index, edge_weight = edge_index_from_adjacency(A)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(dtype=torch.float32).to(device)
    return edge_index, edge_weight
