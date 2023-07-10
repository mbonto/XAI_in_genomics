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
    n_layer = None
    n_hidden_feat = None
    if model_name == "MLP":
        if name in ["pancan", "KIRC", "BRCA", "SIMU1", "SIMU2", "SimuA", "SimuB", "SimuC", "demo"]:
            n_layer = 1
            n_hidden_feat = 20
        else:
            n_layer = 1
            n_hidden_feat = 20
    elif model_name == "DiffuseMLP":
        n_layer = 1
        n_hidden_feat = 20
    return n_layer, n_hidden_feat
        

def set_optimizer(name, model):
    n_epoch = 25
    criterion = nn.CrossEntropyLoss()
    lr = 0.1
    weight_decay = 1e-4
    lr_gamma = 0.1
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
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
    else:
        save_path = os.path.join(code_path, 'Simulation', 'Results', name)
    return save_path


def get_data_path(name):
    data_path = set_path()
    if name in ["pancan", "BRCA", "KIRC",]:
        data_path = os.path.join(data_path, 'tcga')
    else:
        data_path = os.path.join(data_path, 'simulation')
    return data_path


def get_TCGA_setting(name):
    assert name in ["pancan", "BRCA", "KIRC"], "Name should be pancan, BRCA or KIRC"
    if name == "pancan":
        database = "pancan"
        label_name = "type"
    elif name in ["BRCA", "KIRC"]:
        database = "gdc"
        label_name = "sample_type.samples"
    return database, label_name
    

def get_data_normalization_parameters(name):
    # Default
    use_mean = True
    use_std = True
    log2 = False
    reverse_log2 = False
    divide_by_sum = False
    factor = None
    
    if name == 'pancan':
        log2 = True
    elif name in ['BRCA', 'KIRC']:
        log2 = True
        reverse_log2 = True
        divide_by_sum = True
        factor = 10**6
        use_std = False
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
    if name in ['BRCA', 'KIRC', 'SimuA', 'SimuB', 'SimuC', 'demo']:
        base_class = 1
        studied_class = [0,]
    elif name.split("_")[0] == "syn":
        base_class = None
        studied_class = [0,]
    else:
        base_class = None
        studied_class = list(np.arange(n_class))
    return base_class, studied_class
