# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
import random
import torch
from joblib import load
from collections import OrderedDict
import argparse
from setting import *
from utils import *
from loader import *
from evaluate import *
from models import *
from XAI_method import *
set_pyplot()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, GCN, LR_L1_penalty)")
argParser.add_argument("--exp", type=int, help="experiment number", default=1)
args = argParser.parse_args()
name = args.name
model_name = args.model
exp = args.exp


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)
save_name = os.path.join(model_name, f"exp_{exp}")


# Seed
random.seed(exp)
np.random.seed(exp)
torch.manual_seed(exp)
if device == 'cuda':
    torch.cuda.manual_seed_all(exp)


# Dataset
train_loader, test_loader, n_class, n_feat, class_name, feat_name, transform, n_sample = load_dataloader(data_path, name, device)

    
# Model
if model_name in ['LR', 'MLP', 'GCN']:
    softmax = True
    n_layer, n_hidden_feat, graph_name = get_hyperparameters(name, model_name)
    model = load_model(model_name, n_feat, n_class, softmax, device, save_path, n_layer, n_hidden_feat, graph_name)
    checkpoint = torch.load(os.path.join(save_path, save_name, 'checkpoint.pt'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

else:
    model = load(os.path.join(save_path, save_name, "checkpoint.joblib"))
    checkpoint = {}
    with open(os.path.join(save_path, save_name, "accuracy.csv"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(', ')
            if line[0] == 'train':
                checkpoint["train_acc"] = float(line[1])
            elif line[0] == 'test':
                checkpoint["test_acc"] = float(line[1])
    checkpoint['state_dict'] = OrderedDict()
    checkpoint['state_dict']['fc.weight'] = torch.tensor(model.coef_).to(device)
    checkpoint['state_dict']['fc.bias'] = torch.tensor(model.intercept_).to(device)
    # Convert the model to PyTorch
    softmax = True
    n_layer, n_hidden_feat, graph_name = get_hyperparameters(name, "LR")
    model = load_model("LR", n_feat, n_class, softmax, device, save_path, n_layer, n_hidden_feat, graph_name)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    
# Assert that the model and the data are coherent
assert compute_accuracy_from_model_with_dataloader(model, train_loader, transform, device) == checkpoint['train_acc']
assert compute_accuracy_from_model_with_dataloader(model, test_loader, transform, device) == checkpoint['test_acc']


# Baseline
base_class, studied_class = get_XAI_hyperparameters(name, n_class)
baseline = get_baseline(train_loader, device, n_feat, transform, base_class)
baseline_pred = model(baseline).detach().cpu().numpy()
print(f"The output of the baseline is {baseline_pred}")


    
