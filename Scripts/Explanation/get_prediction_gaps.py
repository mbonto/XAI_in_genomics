# Libraries
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)

import numpy as np
import torch
import argparse

from setting import *
from dataset import *
from loader import *
from evaluate import *
from models import *
from XAI_method import *
from XAI_interpret import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, DiffuseLR, DiffuseMLP)")
argParser.add_argument("--set", type=str, help="set (train or test)")
argParser.add_argument("--gap", type=int, help="prediction gaps are computed every `gap` features removed", default=10)
argParser.add_argument("--exp", type=int, help="experiment number", default=1)
args = argParser.parse_args()
name = args.name
model_name = args.model
set_name = args.set
gap = args.gap
exp = args.exp
print('Model    ', model_name)
XAI_method = "Integrated_Gradients"


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)
save_name = os.path.join(model_name, f"exp_{exp}")


# Load a dataset
train_loader, test_loader, n_class, n_feat, class_name, feat_name, transform, n_sample = load_dataloader(data_path, name, device)


# Load a model
softmax = True
n_layer, n_hidden_feat = get_hyperparameters(name, model_name)
model = load_model(model_name, n_feat, n_class, softmax, device, save_path, n_layer, n_hidden_feat)

# Parameters
checkpoint = torch.load(os.path.join(save_path, save_name, 'checkpoint.pt'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()


# Assert that the model and the data are coherent
assert compute_accuracy_from_model_with_dataloader(model, train_loader, transform, device) == checkpoint['train_acc']
assert compute_accuracy_from_model_with_dataloader(model, test_loader, transform, device) == checkpoint['test_acc']


# Baseline
base_class, studied_class = get_XAI_hyperparameters(name, n_class)
baseline = get_baseline(train_loader, device, n_feat, transform, base_class)
default_output = model(baseline).detach().cpu().numpy()


# Data of interest
if set_name == 'train':
    loader = train_loader
elif set_name == 'test':
    loader = test_loader
    

# Load the attribution scores
attr, y_pred, y_true, labels, features, _, _ = load_attributions(XAI_method, os.path.join(save_path, save_name, XAI_method), set_name=set_name)

# Normalize them
attr = transform_data(attr, transform='divide_by_norm')


# Local prediction gap ...
print("Local PGs")

# ... on unimportant features
PGU = prediction_gap_with_dataloader(model, loader, transform, attr, gap, baseline, studied_class, None, "unimportant", y_true, y_pred)
print('Average PGU', np.round(np.mean(list(PGU.values())) * 100, 2))
adj_PGU = {}
for c in studied_class:
    adj_PGU[c] = PGU[c] / (1 - default_output[0, c])
print('    Adjusted', np.round(np.mean(list(adj_PGU.values())) * 100, 2))

# ... on important features
PGI = prediction_gap_with_dataloader(model, loader, transform, attr, gap, baseline, studied_class, None, "important", y_true, y_pred)
print('Average PGI', np.round(np.mean(list(PGI.values())) * 100, 2))

# Save
with open(os.path.join(save_path, save_name, XAI_method, "figures", f"local_XAI_{set_name}.csv"), "w") as f:
    f.write(f"PGU, {np.round(np.mean(list(PGU.values())) * 100, 2)}\n")
    f.write(f"PGU_adjusted, {np.round(np.mean(list(adj_PGU.values()))* 100, 2)}\n")
    f.write(f"PGI, {np.round(np.mean(list(PGI.values())) * 100, 2)}\n")
    
    
# Global prediction gaps ...
print('\nGlobal PGs')

# ... where variables are ordered by summing the attributions of all examples, by summing their ranks or by using median ranks 
types = ["sum", "rank", "rank-median"]
global_PG = {}
for _type in types:
    global_PG[_type] = {}
    print(_type)
    for order in ["increasing", "decreasing"]:
        indices = get_features_order(attr, _type + "_" + order)
        PG = prediction_gap_with_dataloader(model, loader, transform, attr, gap, baseline, studied_class, indices, None, y_true, y_pred)
        global_PG[_type][order] = np.round(np.mean(list(PG.values())) * 100, 2)
        if order == "increasing":
            print(f'    Average PGU', global_PG[_type][order])
        else:
            print(f'    Average PGI', global_PG[_type][order])
        if order == "increasing":
            adj_PG = {}
            for c in studied_class:
                adj_PG[c] = PG[c] / (1 - default_output[0, c])
            global_PG[_type][order + "_adjusted"] = np.round(np.mean(list(adj_PG.values())) * 100, 2)
            print('        Adjusted', global_PG[_type][order + "_adjusted"])

# ... random order
PGRs = []
for t in range(30):
    indices = get_features_order(attr, _type="random")
    PGR = prediction_gap_with_dataloader(model, loader, transform, attr, gap, baseline, studied_class, indices, None, y_true, y_pred)
    PGRs.append(np.round(np.mean(list(PGR.values())) * 100, 2))
print('Average PGR', np.round(np.mean(PGRs), 2))


# Save
with open(os.path.join(save_path, save_name, XAI_method, "figures", f"global_XAI_{set_name}.csv"), "w") as f:
    for _type in global_PG.keys():
        for order in ["increasing", "decreasing"]:
            if order == "increasing":
                f.write(f"{_type}, PGU, {global_PG[_type][order]}\n")
                f.write(f"{_type}, PGU_adjusted, {global_PG[_type][order + '_adjusted']}\n")
            else:
                f.write(f"{_type}, PGI, {global_PG[_type][order]}\n")
    f.write(f"PGR, {np.round(np.mean(PGRs), 2)}\n")
    f.write(f"list_PGR, {PGRs}\n")
