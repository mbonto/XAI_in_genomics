# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
import argparse
from setting import *
from loader import *
from XAI_method import *
from XAI_interpret import *
from utils import *


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, DiffuseLR, DiffuseMLP)")
argParser.add_argument("--set", type=str, help="set (train, test)")
argParser.add_argument("--exp", type=int, help="experiment number", default=1)
args = argParser.parse_args()
name = args.name
model_name = args.model
set_name = args.set
exp = args.exp
print('Model    ', model_name)
XAI_method = "Integrated_Gradients"


# Path
save_path = get_save_path(name, code_path)
save_name = os.path.join(model_name, f"exp_{exp}", XAI_method)


# Attributions
attr, y_pred, y_true, labels, feat_name, baseline, _ = load_attributions(XAI_method, os.path.join(save_path, save_name), set_name=set_name)
feat_name = np.array(feat_name)
correct_indices = np.argwhere((y_pred - y_true) == 0)[:, 0]
print("There are {} uncorrect examples. We remove the uncorrect examples from our study.".format(len(y_pred) - len(correct_indices)))
attr = attr[correct_indices]
y_true = y_true[correct_indices]
y_pred = y_pred[correct_indices]


# Normalize
attr = transform_data(attr, transform='divide_by_norm')


# Attributions averaged per class
method = 'mean'


# Compute the relevance scores averaged per feature and per class.
classes = np.unique(y_true)
scores = {}
for _class in classes:
    cls_attr = get_attributions_per_class(attr, y_true, _class, method)
    cls_sorted_indices = sort_attribution_scores(cls_attr)
    scores[_class] = {'attr': cls_attr, 'sorted_indices': cls_sorted_indices}


# Same thing without distinguishing the classes.
avg_attr = normalize_attribution_scores(attr, method)
avg_sorted_indices = sort_attribution_scores(avg_attr)
scores["general"] = {'attr': avg_attr, 'sorted_indices': avg_sorted_indices}


# Save
np.save(os.path.join(save_path, save_name, "{}_scores_with_{}_{}.npy".format(XAI_method, method, set_name)), scores)


# Save to compare with other feature selection methods
## When there are more than two classes, several classes can be explained with respect to a certain baseline.
## In that case, the importance given to a feature is defined as the average over all studied classes.
print(f"There is {len(classes)} studied class: {classes}") if len(classes) == 1 else print(f"There are {len(classes)} studied classes: {classes}")
scores_IG = np.zeros(attr.shape[1])
for _class in classes:
    scores_IG += scores[_class]['attr']
scores_IG = scores_IG / len(classes)
print("Shape -", scores_IG.shape, "Min coef -", np.round(np.min(scores_IG), 2), "Max coef -", np.round(np.max(scores_IG), 2))
## Features sorted by decreasing values
order_IG = np.argsort(-scores_IG)
np.save(os.path.join(save_path, save_name, "order_{}_set_{}_exp_{}.npy".format(XAI_method, set_name, exp)), feat_name[order_IG])
if XAI_method == "Integrated_Gradients" and set_name == "train" and exp == 1:
    create_new_folder(os.path.join(save_path, "order")) 
    np.save(os.path.join(save_path, "order", "order_IG_set_{}_exp_{}.npy".format(set_name, exp)), feat_name[order_IG])















































































