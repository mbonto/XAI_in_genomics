# Libraries
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
import random
from sklearn.metrics import accuracy_score
from joblib import load
import argparse
from setting import *
from dataset import *
from loader import *
from evaluate import *
from models import *
from XAI_method import *
from XAI_interpret import *
from utils import create_new_folder


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


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)
save_name = os.path.join(model_name, f"exp_{exp}")


# Seed
random.seed(exp)
np.random.seed(exp)


# Load a dataset
X_train, X_test, y_train, y_test, n_class, n_feat, class_name, feat_name = load_dataset(data_path, name, normalize=True)
feat_name = np.array(feat_name)


# Load a model
clf = load(os.path.join(save_path, save_name, "checkpoint.joblib"))
checkpoint = {}
with open(os.path.join(save_path, save_name, "accuracy.csv"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(', ')
        if line[0] == 'train':
            checkpoint["train_acc"] = float(line[1])
        elif line[0] == 'test':
            checkpoint["test_acc"] = float(line[1])


# Assert that the model and the data are coherent
assert np.round(accuracy_score(y_train, clf.predict(X_train)) * 100, 2) == checkpoint['train_acc']
assert np.round(accuracy_score(y_test, clf.predict(X_test)) * 100, 2) == checkpoint['test_acc']


# Baseline
base_class, studied_class = get_XAI_hyperparameters(name, n_class)
baseline = get_baseline_sklearn(X_train, y_train, n_feat, base_class)
default_output = clf.predict_proba(baseline)
# print("Output of the baseline", default_output)


# Data of interest
if set_name == 'train':
    X = X_train
    y = y_train
elif set_name == 'test':
    X = X_test
    y = y_test
    

# Load the explainability scores
ordered_feat_name = np.load(os.path.join(save_path, "order", f"order_IG_LR_L1_penalty_set_train_exp_{exp}.npy"), allow_pickle=True)
ordered_indices = np.array([np.argwhere(feat == feat_name)[0] for feat in ordered_feat_name]).reshape(-1)


# Global prediction gaps where variables are ordered...
# ... according to the absolute values of their scores
global_PG = {}
for order in ["increasing", "decreasing"]:
    if order == "increasing":
        indices = np.flip(ordered_indices).reshape(1, -1)
    elif order == "decreasing":
        indices = ordered_indices.reshape(1, -1)
    PG = prediction_gap_with_dataset(clf, X, y, gap, baseline, studied_class, indices)
    global_PG[order] = np.round(np.mean(list(PG.values())) * 100, 2)
    print(f'    Average PGU', global_PG[order]) if order == "increasing" else print(f'    Average PGI', global_PG[order])

# ... randomly
# PGRs = []
# for t in range(10):
#     indices = np.arange(n_feat)
#     np.random.shuffle(indices)
#     indices = indices.reshape(1, -1)
#     PGR = prediction_gap_with_dataset(clf, X, y, gap, baseline, studied_class, indices)
#     PGRs.append(np.round(np.mean(list(PGR.values())) * 100, 2))
# print('Average PGR', np.round(np.mean(PGRs), 2))


# Save
create_new_folder(os.path.join(save_path, save_name, "figures"))
with open(os.path.join(save_path, save_name, "figures", f"global_XAI_{set_name}.csv"), "w") as f:
    for order in ["increasing", "decreasing"]:
        if order == "increasing":
            f.write(f"PGU, {global_PG[order]}\n")
        else:
            f.write(f"PGI, {global_PG[order]}\n")
    # f.write(f"PGR, {np.round(np.mean(PGRs), 2)}\n")
    # f.write(f"list_PGR, {PGRs}\n")


