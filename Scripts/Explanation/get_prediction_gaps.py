# Libraries
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
import random
import torch
import argparse
from setting import *
from loader import *
from evaluate import *
from models import *
from XAI_method import *
from XAI_interpret import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, GNN)")
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


# Seed
random.seed(exp)
np.random.seed(exp)
torch.manual_seed(exp)
if device == 'cuda':
    torch.cuda.manual_seed_all(exp)


# Load a dataset
train_loader, test_loader, n_class, n_feat, class_name, feat_name, transform, n_sample = load_dataloader(data_path, name, device)


# Load a model
softmax = True
n_layer, n_hidden_feat, graph_name = get_hyperparameters(name, model_name)
model = load_model(model_name, n_feat, n_class, softmax, device, save_path, n_layer, n_hidden_feat, graph_name)


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
attr, y_pred, y_true, class_name, feat_name, _, _ = load_attributions(XAI_method, os.path.join(save_path, save_name, XAI_method), set_name=set_name)
feat_name = np.array(feat_name)


# Normalize them
attr = transform_data(attr, transform='divide_by_norm')


# Local prediction gap ...
print("Local PGs")

# ... on unimportant features
indices = np.argsort(np.abs(attr), axis=1)
PGU = prediction_gap_with_dataloader(model, loader, transform, gap, baseline, studied_class, indices, y_true, y_pred)
PGU = np.round(np.mean(list(PGU.values())) * 100, 2)
print('    Average PGU', PGU)
# adj_PGU = {}
# for c in studied_class:
#     adj_PGU[c] = PGU[c] / (1 - default_output[0, c])
# print('    Adjusted', np.round(np.mean(list(adj_PGU.values())) * 100, 2))

# ... on important features
indices = np.argsort(-np.abs(attr), axis=1)
PGI = prediction_gap_with_dataloader(model, loader, transform, gap, baseline, studied_class, indices, y_true, y_pred)
PGI = np.round(np.mean(list(PGI.values())) * 100, 2)
print('    Average PGI', PGI)

# Save
with open(os.path.join(save_path, save_name, XAI_method, "figures", f"local_XAI_{set_name}.csv"), "w") as f:
    f.write(f"PGU, {PGU}\n")
    # f.write(f"PGU_adjusted, {np.round(np.mean(list(adj_PGU.values()))* 100, 2)}\n")
    f.write(f"PGI, {PGI}\n")
    
    
# Global prediction gaps ...
print('\nGlobal PGs')
ordered_feat_name = np.load(os.path.join(save_path, "order", f"order_IG_{model_name}_set_train_exp_{exp}.npy"), allow_pickle=True)
ordered_indices = np.array([np.argwhere(feat == feat_name)[0] for feat in ordered_feat_name]).reshape(-1)

# ... on unimportant features
indices = np.flip(ordered_indices.copy()).reshape(1, -1)
global_PGU = prediction_gap_with_dataloader(model, loader, transform, gap, baseline, studied_class, indices, y_true, y_pred)
global_PGU = np.round(np.mean(list(global_PGU.values())) * 100, 2)
print(f'    Average PGU', global_PGU)

# ... on important features
indices = ordered_indices.copy().reshape(1, -1)
global_PGI = prediction_gap_with_dataloader(model, loader, transform, gap, baseline, studied_class, indices, y_true, y_pred)
global_PGI = np.round(np.mean(list(global_PGI.values())) * 100, 2)
print(f'    Average PGI', global_PGI)


# ... random order
# PGRs = []
# for t in range(10):
#     indices = get_features_order(attr, _type="random")
#     PGR = prediction_gap_with_dataloader(model, loader, transform, gap, baseline, studied_class, indices, y_true, y_pred)
#     PGRs.append(np.round(np.mean(list(PGR.values())) * 100, 2))
# print('Average PGR', np.round(np.mean(PGRs), 2))


# Save
with open(os.path.join(save_path, save_name, XAI_method, "figures", f"global_XAI_{set_name}.csv"), "w") as f:
    f.write(f"PGU, {global_PGU}\n")
    f.write(f"PGI, {global_PGI}\n")
    # f.write(f"PGR, {np.round(np.mean(PGRs), 2)}\n")
    # f.write(f"list_PGR, {PGRs}\n")

