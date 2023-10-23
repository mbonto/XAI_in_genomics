# Paths
def set_path():
    path = '/local/mbontono/data/'  # absolute path of the data folder
    return path


# Sets of variables
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR


def get_hyperparameters(name, model_name):
    if model_name == "LR_L1_penalty":
        if name in ["ttg-breast", "BRCA-pam", "BRCA", "KIRC"]:
            C = 0.1
        elif name in ["ttg-all", "pancan"]:
            C = 1.
        return C
    else:
        n_layer = None
        n_hidden_feat = None
        graph_name = None
        if model_name == "MLP":
            if name in ["pancan", "KIRC", "BRCA", "SIMU1", "SIMU2", "SimuA", "SimuB", "SimuC", "demo", "demo1", "ttg-all", "BRCA-pam"]:
                n_layer = 1
                n_hidden_feat = 20
            elif name in ["ttg-breast"]:
                n_layer = 1
                n_hidden_feat = 10
        elif model_name == "DiffuseMLP":
            n_layer = 1
            n_hidden_feat = 20
        elif model_name == "GCN":
            n_layers = {"BRCA": 5, "BRCA-pam": 5, "ttg-breast": 3, "ttg-all": 1, "pancan": 7}
            ks = {"BRCA": 2, "BRCA-pam": 10, "ttg-breast": 2, "ttg-all": 10, "pancan": 10}
            n_layer = n_layers[name]
            n_hidden_feat = 1
            k = ks[name]
            graph_name = f"pearson_correlation_{k}_variables.npz"
        return n_layer, n_hidden_feat, graph_name
            

def set_optimizer(name, model):
    n_epoch = 25
    weight_decay = 1e-4
    lr_gamma = 0.1
    if model.name == "GCN":
        n_epoch = 15
        criterion = nn.CrossEntropyLoss() # weight=torch.tensor([0.5, 5]))
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[int(0.5 * n_epoch), int(0.9 * n_epoch)], gamma=lr_gamma)
    return criterion, optimizer, scheduler, n_epoch


def get_save_path(name, code_path):
    if name == 'pancan':
        save_path = os.path.join(code_path, 'Pancan', 'Results')
    elif name == 'BRCA':
        save_path = os.path.join(code_path, 'Gdc', 'Results', 'BRCA')
    elif name == 'KIRC':
        save_path = os.path.join(code_path, 'Gdc', 'Results', 'KIRC')
    elif name == 'LUAD':
        save_path = os.path.join(code_path, 'Gdc', 'Results', 'LUAD')
    elif name == 'BRCA-pam':
        save_path = os.path.join(code_path, 'Legacy', 'Results', 'BRCA-pam')
    elif name == 'ttg-all':
        save_path = os.path.join(code_path, 'TTG', 'Results', 'all')
    elif name == 'ttg-breast':
        save_path = os.path.join(code_path, 'TTG', 'Results', 'breast')
    else:
        save_path = os.path.join(code_path, 'Simulation', 'Results', name)
    return save_path


def get_data_path(name):
    data_path = set_path()
    if name in ["pancan", "BRCA", "KIRC", "ttg-all", "ttg-breast", "BRCA-pam", "tcga"]:
        data_path = os.path.join(data_path, 'tcga')
    else:
        data_path = os.path.join(data_path, 'simulation')
    return data_path


def get_TCGA_setting(name):
    assert name in ["pancan", "BRCA", "KIRC", "ttg-all", "ttg-breast", "BRCA-pam"], "Name should be pancan, ttg-all, ttg-breast, BRCA-pam, BRCA or KIRC"
    if name == "pancan":
        database = "pancan"
        label_name = "type"
    elif name in ["ttg-all", "ttg-breast"]:
        database = "ttg"
        label_name = "_sample_type"
    elif name in ["BRCA", "KIRC"]:
        database = "gdc"
        label_name = "sample_type.samples"
    elif name == "BRCA-pam":
        database = "legacy"
        label_name = "PAM50Call_RNAseq"
        name = "BRCA"
    return database, name, label_name
    

def get_data_normalization_parameters(name):
    # Default
    use_mean = True
    use_std = True
    log2 = False
    reverse_log2 = False
    divide_by_sum = False
    factor = 10**6  # only used when divide_by_sum is True. The expression of each gene is divided by the total expression in the sample and multiplied by `factor`.
    
    if name == 'pancan':
        # divide_by_sum = True
        # log2 = True
        pass
    elif name in ['ttg-all', 'ttg-breast', 'BRCA-pam']:
        # reverse_log2 = True
        # divide_by_sum = True
        # log2 = True
        pass
    elif name in ['BRCA', 'KIRC']:
        # reverse_log2 = True
        # divide_by_sum = True
        # log2 = True
        pass
    return use_mean, use_std, log2, reverse_log2, divide_by_sum, factor


def get_split_dataset_setting(name):
    test_size = 0.4
    random_state = 43
    return test_size, random_state


def get_loader_setting(name):
    batch_size = 32
    return batch_size


def get_XAI_hyperparameters(name, n_class):
    """
    Return `base_class` and `studied_class`, two hyperparameters required by the XAI method called Integrated Gradients (IG).
    `base_class` is either a number between 0 and n_class - 1 or None. If it is a number, the baseline used by IG is the
    average of the training examples of the `base_class` class. Otherwise, the baseline used is the null tensor.
    studied_class is a list of numbers between 0 and n_class - 1. It contains the classes of the examples for which the
    IG scores will be computed.    
    """
    if name in ["BRCA", "KIRC", "SimuA", "SimuB", "SimuC", "demo"]:
        base_class = 1
        studied_class = [0,]
    elif name in ["ttg-all", "ttg-breast"]:
        base_class = 0
        studied_class = [1,]
    elif name in ["BRCA-pam"]:
        base_class = 4
        studied_class = [0, 1, 2, 3]
    elif name.split("_")[0] == "syn":
        base_class = None
        studied_class = [0,]
    else:
        base_class = None
        studied_class = list(np.arange(n_class))
    return base_class, studied_class
