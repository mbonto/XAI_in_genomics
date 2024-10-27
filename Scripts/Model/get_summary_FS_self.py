# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import argparse
import numpy as np
from setting import *


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="Dataset name")
argParser.add_argument("-m", "--model", type=str, help="Model name (LR, MLP, GCN, LR_L1_penalty, LR_L2_penalty, xgboost)")
argParser.add_argument("--n_repet", type=int, help="Results are averaged for all experiments between 1 and `n_repet`")
args = argParser.parse_args()
name = args.name
model_name = args.model
n_repet = args.n_repet


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Variables
exps = np.arange(1, n_repet+1)
n_feats = [1, 5, 10 , 50, 100, 500, 1000]
n_rand = 3
rands = np.arange(1, n_rand + 1)
selections = {"LR": "IG_LR_set_train_", "MLP":"IG_MLP_set_train_", "GCN":"IG_GCN_set_train_", "LR_L1_penalty":"IG_LR_L1_penalty_set_train_", "LR_L2_penalty":"IG_LR_L2_penalty_set_train_", "xgboost":"xgboost_"}


# Summarize results
results = {}


for selection_type in ["best", "worst",]:
    results[selection_type] = {}
    for n_feat in n_feats:
        results[selection_type][n_feat] = {}
        data = []
        for exp in exps:
            save_name = os.path.join(model_name, "FS", f"1_{selections[model_name]}exp_{exp}_{n_feat}_{selection_type}.csv")
            with open(os.path.join(save_path, save_name), "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(', ')
                    if line[0] == 'balanced_test':
                        data.append(float(line[1]))
        assert len(data) == n_repet
        results[selection_type][n_feat]["mean"] = np.round(np.mean(data), 2)
        results[selection_type][n_feat]["std"] = np.round(np.std(data), 2)


selection_type = "random_wo_best"
results[selection_type] = {}
for n_feat in n_feats:
    results[selection_type][n_feat] = {}
    data = []
    for exp in exps:
        data_rand = []
        for rand in rands:
            save_name = os.path.join(model_name, "FS", f"{rand}_{selections[model_name]}exp_{exp}_{n_feat}_{selection_type}.csv")
            with open(os.path.join(save_path, save_name), "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(', ')
                    if line[0] == 'balanced_test':
                        data_rand.append(float(line[1]))
        assert len(data_rand) == n_rand
        data.append(np.mean(data_rand))
    assert len(data) == n_repet
    results[selection_type][n_feat]["mean"] = np.round(np.mean(data), 2)
    results[selection_type][n_feat]["std"] = np.round(np.std(data), 2)



selection_type = None
results[selection_type] = {}
for n_feat in n_feats:
    results[selection_type][n_feat] = {}
    data = []
    for rand in rands:
        save_name = os.path.join(model_name, "FS", f"{rand}_None_{n_feat}_{selection_type}.csv")
        with open(os.path.join(save_path, save_name), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(', ')
                if line[0] == 'balanced_test':
                    data.append(float(line[1]))
    assert len(data) == n_rand
    results[selection_type][n_feat]["mean"] = np.round(np.mean(data), 2)
    results[selection_type][n_feat]["std"] = np.round(np.std(data), 2)


# Save
with open(os.path.join(save_path, model_name, f"classif_perf_with_SF_self.csv"), "w") as f:
    f.write("Features, Important, Random, Random_wo_best, Unimportant, Std_important, Std_random, Std_random_wo_best, Std_unimportant\n")
    for n_feat in n_feats:
        f.write(f"{n_feat}, {results['best'][n_feat]['mean']}, {results[None][n_feat]['mean']}, {results['random_wo_best'][n_feat]['mean']}, {results['worst'][n_feat]['mean']}, {results['best'][n_feat]['std']}, {results[None][n_feat]['std']}, {results['random_wo_best'][n_feat]['std']}, {results['worst'][n_feat]['std']}\n")



