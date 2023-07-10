import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch_geometric.nn import SGConv
from torch_geometric.utils import dense_to_sparse
                        

    
def load_model(model_name, n_feat, n_class, softmax, device, save_path, n_layer=None, n_hidden_feat=None):
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
    From (batch_size, nb_feat), get (nb_node, nb_feat_per_node (here 1))
    Parameters:
        edge_index  --  torch tensor of size (2, n_edge)
        edge weight  --  torch tensor of size (n_edge)
        n_node  --  int, number of nodes in the graph
        batch_size  --  int, number of examples
    """
    batch_edge_index = edge_index
    batch_edge_weight = edge_weight
    batch = torch.zeros(size=[nb_feat]).type(torch.LongTensor)
    for b in range(1, batch_size):
        batch_edge_index = torch.cat((batch_edge_index, edge_index + nb_feat * b), dim=1)
        batch_edge_weight = torch.cat((batch_edge_weight, edge_weight))
        batch = torch.cat((batch, torch.ones(size=[nb_feat]).type(torch.LongTensor) * b))
    return batch_edge_index, batch_edge_weight, batch


def edge_index_from_adjacency(A):
    """
    Return a torch tensor edge_index of size (2, n_edge) and a torch tensor edge_weight of size (n_edge).
    
    Parameter:
        A  -- square matrix, adjacency matrix of a graph.
    """
    return dense_to_sparse(A)


def reshape_batch(x):
    """Reshape x from (batch_size, n_node) to (batch_size x n_node, n_feat_per_node (here 1)).
    """
    return torch.reshape(x, [-1, 1]), x.shape[0]


def reverse_reshape_batch(x, batch_size):
    """Reshape x from (batch_size x n_node, n_feat_per_node) to (batch_size, n_node * n_feat_per_node).
    """
    return torch.reshape(x, [batch_size, -1])


def load_graph(save_path, name, min_value, device):
    A = np.load(os.path.join(save_path, name), allow_pickle=True)
    A = torch.from_numpy(A)
    A = A.type('torch.FloatTensor')
    A = nn.functional.relu(A)
    A = (A > min_value) * A
    edge_index, edge_weight = edge_index_from_adjacency(A)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    return edge_index, edge_weight