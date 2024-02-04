# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
import torch
import argparse
from setting import *
from loader import *
from XAI_method import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, DiffuseLR, DiffuseMLP)")
argParser.add_argument("--set", type=str, help="set (train or test)", default="test")
argParser.add_argument("--exp", type=int, help="experiment number", default=1)
args = argParser.parse_args()
name = args.name
model_name = args.model
set_name = args.set
exp = args.exp
save_name = os.path.join(model_name, f"exp_{exp}")


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Dataset
train_loader, test_loader, n_class, n_feat, class_name, feat_name, transform, n_sample = load_dataloader(data_path, name, device)


# Set
if set_name == 'train':
    loader = train_loader
elif set_name == 'test':
    loader = test_loader

    
# Baseline
base_class, studied_class = get_XAI_hyperparameters(name, n_class)
baseline = get_baseline(train_loader, device, n_feat, transform, base_class).to("cpu")
# print(baseline.shape)


# Attributions
XAI_method = "Integrated_Gradients"
attr, y_pred, y_true, labels, features, baseline, _ = load_attributions(XAI_method, os.path.join(save_path, save_name, XAI_method), set_name=set_name)
attr = transform_data(attr, transform='divide_by_norm')


# Example
index = 1
torch.manual_seed(1)  # Seed needed to load the examples in the same order as in `attr`.
for x, y in loader:
    # Select studied examples
    x = x[sum(y == c for c in studied_class).bool()]
    y = y[sum(y == c for c in studied_class).bool()]
    # Select one example
    x = x[index].reshape(1, -1)
    if transform:
        x = x.to(device)
        x = transform(x).to("cpu")
    y = y[index]
    # Sanity check
    assert y.item() == y_true[index], 'Problem with data order.'
    # Rank features by importance for each example
    local_indices = np.argsort(-np.abs(attr)[index].reshape(1, -1), axis=1)
    break
print("Class", y.item())


# Global ranking
ordered_feat_name = np.load(os.path.join(save_path, "order", f"order_IG_{model_name}_set_{set_name}_exp_{exp}.npy"), allow_pickle=True)
global_indices = np.array([np.argwhere(feat == feat_name)[0] for feat in ordered_feat_name]).reshape(-1)


# Save
torch.save(x, os.path.join(save_path, "x.pt"))
torch.save(y, os.path.join(save_path, "y.pt"))
torch.save(baseline, os.path.join(save_path, save_name, XAI_method, "baseline.pt"))
np.save(os.path.join(save_path, save_name, XAI_method, "global_indices.npy"), global_indices)
np.save(os.path.join(save_path, save_name, XAI_method, "local_indices.npy"), local_indices)
