# Libraries
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from scipy.sparse import load_npz, csc_matrix
from setting import *
from loader import *
from models import *
from XAI_method import *
from XAI_interpret import *
from graphs import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, DiffuseLR, DiffuseMLP)")
argParser.add_argument("--exp", type=int, help="experiment number", default=1)
argParser.add_argument("--set", type=str, help="set (train or test)")
argParser.add_argument("--global_setting", type=str, help="set the function used to infer a global feature agreement metric", default="intersect_rank_scores", choices=["rank_mean_scores", "intersect_rank_scores"])
argParser.add_argument("--diffusion", help="smooth the attributions", action='store_true')
args = argParser.parse_args()
name = args.name
model_name = args.model
exp = args.exp
set_name = args.set
global_setting = args.global_setting
diffusion = args.diffusion
print('Model    ', model_name)
XAI_method = "Integrated_Gradients"


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)
save_name = os.path.join(model_name, f"exp_{exp}", XAI_method)
    

# Dataset
data = np.load(os.path.join(data_path, f'{name}.npy'), allow_pickle=True).item()
if diffusion:
    method = 'pearson_correlation'
    min_value = 0.5
    print(f"Loading of the diffusion matrix stored in {os.path.join(save_path, 'graph', f'{method}_{min_value}_variables_diffusion.npz')}")
    D = load_npz(os.path.join(save_path, 'graph', f'{method}_{min_value}_variables_diffusion.npz'))  # Make sure to load the diffusion matrix that you want.


# XAI scores
attr, y_pred, y_true, labels, features, baseline, _ = load_attributions(XAI_method, os.path.join(save_path, save_name), set_name=set_name)
n_class = len(np.unique(labels))
if diffusion:
    attr = csc_matrix(attr)
    attr = attr.dot(D)
    attr = attr.toarray()
attr = transform_data(attr, transform='divide_by_norm')
correct_indices = np.argwhere((y_pred - y_true) == 0)[:, 0]
print("There are {} uncorrect test examples. We remove them from our study.".format(len(y_pred) - len(correct_indices)))
print("")


# Classes for which the feature agreement (FA) scores will be computed
_, studied_class = get_XAI_hyperparameters(name, n_class)
print('Studied classes', studied_class)


# With two classes, the variables enabling to identify one class also enable to identify the other classes.
# Assumption with more classes: the variables enabling to identify one class do not enable to identify another class from the other classes.  
other_class = {}
if n_class == 2:
    for c in studied_class:
        other_class[c] = [o for o in range(n_class) if o != c]
else:
    for c in studied_class:
        other_class[c] = []
print('Other classes', other_class)


# In particular cases, a class contains several subclasses.
# In this case, by construction, the number of classes used in the classification problem (n_class) is not the same as the number of subclasses in the data (data['n_class']).
# To calculate FA scores, since important variables depend on subclasses, class labels must be replaced by subclass labels.
if n_class != data['n_class']:
    # Warning! The following function has to be adapted to new datasets containing subclasses.
    y_true, studied_class, other_class = from_classes_to_subclasses(data_path, name, set_name, data['n_class'], y_true, studied_class, other_class)
    print('Studied classes', studied_class)
    print('Other classes', other_class)


# For each class, list the variables that are useful for the identification
counts, genes_per_class = get_informative_variables(studied_class, other_class, data['useful_paths'], data['useful_genes'])
print(f"{counts} genes need to be retrieved per class.")
print("")


# Only correctly classified examples are used to compute the scores.           
attr = attr[correct_indices]
y_true = y_true[correct_indices]


# Global FA (ranking)
print("Global FA")
## Most important genes
chosen_g  = {}
for c in studied_class:
    indices = np.argwhere(y_true == c)[:, 0]
    attr_cls = attr[indices]
    if global_setting == "rank_mean_scores":
        chosen_g["C" + str(c)] = np.argsort(-np.sum(attr_cls, axis=0))[:counts[c]]
    elif global_setting == "intersect_rank_scores":
        genes = np.argsort(-attr_cls, axis=1)[:, :counts[c]].reshape(-1)
        list_g, nb_g = np.unique(genes, return_counts=True)
        chosen_g["C" + str(c)] = list_g[np.argsort(-nb_g)[:counts[c]]]
## Intersection
global_score = 0
for c in studied_class:
    global_score += len(set(chosen_g["C"+str(c)]).intersection(set(genes_per_class["C"+str(c)]))) / counts[c]
    print(f'Class {c}  -- {np.round(len(set(chosen_g["C"+str(c)]).intersection(set(genes_per_class["C"+str(c)]))) / counts[c] * 100, 2)}')
global_score = global_score / len(studied_class)
print(f"Global ranking score averaged per class on the {set_name} set: {np.round(global_score * 100, 2)}")  
print(' ')


# Local FA
print("Local FA")
## Most important genes
ranking = 0
for c in studied_class:
    indices = np.argwhere(y_true == c)[:, 0]
    attr_cls = attr[indices]
    genes = np.argsort(-attr_cls, axis=1)[:, :counts[c]].reshape(-1)
    nb_correct = 0
    for g in genes:
        if g in genes_per_class["C"+str(c)]:
            nb_correct += 1
    ranking += nb_correct / len(genes)
    print(f'Class {c}  -- {np.round(nb_correct / len(genes) * 100, 2)}')
local_score = ranking / len(studied_class)
print(f"Average ranking score on the {set_name} examples: {np.round(local_score * 100, 2)}")
print(' ')


# Save
file_name = f"ranking_diffusion_{global_setting}_{set_name}.csv" if diffusion else f"ranking_{global_setting}_{set_name}.csv"
with open(os.path.join(save_path, save_name, "figures", file_name), "w") as f:
    f.write(f"global, {np.round(global_score * 100, 2)}\n")
    f.write(f"local, {np.round(local_score * 100, 2)}\n")
