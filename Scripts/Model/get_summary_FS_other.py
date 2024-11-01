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
argParser.add_argument("-m", "--model", type=str, help="Model name (LR, MLP, GCN, LR_L1_penalty, LR_L2_penalty)")
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
selection_types = ["best", "worst"]


# Summarize results
results = {}

for selection_type in selection_types:
    for selection in ["IG_LR_L1_penalty_set_train", "IG_LR_L2_penalty_set_train", "IG_MLP_set_train", "IG_GCN_set_train", "xgboost"]:
        results[selection] = {}
        for n_feat in n_feats:
            results[selection][n_feat] = {}
            data = []
            for exp in exps:
                save_name = os.path.join(model_name, "FS", f"1_{selection}_exp_{exp}_{n_feat}_{selection_type}.csv")
                with open(os.path.join(save_path, save_name), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip().split(', ')
                        if line[0] == 'balanced_test':
                            data.append(float(line[1]))
            assert len(data) == n_repet
            results[selection][n_feat]["mean"] = np.round(np.mean(data), 2)
            results[selection][n_feat]["std"] = np.round(np.std(data), 2)
    
    for selection in ["var", "edgeR", "DESeq2", "PCA_PC1", "MI"]:
        results[selection] = {}
        for n_feat in n_feats:
            results[selection][n_feat] = {}
            data = []
            save_name = os.path.join(model_name, "FS", f"1_{selection}_{n_feat}_{selection_type}.csv")
            with open(os.path.join(save_path, save_name), "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(', ')
                    if line[0] == 'balanced_test':
                        data.append(float(line[1]))
            assert len(data) == 1
            results[selection][n_feat]["mean"] = np.round(np.mean(data), 2)
            results[selection][n_feat]["std"] = np.round(np.std(data), 2)


    # Save
    with open(os.path.join(save_path, model_name, f"classif_perf_with_SF_other_{selection_type}.csv"), "w") as f:
        f.write("Features, VAR, PCA, MI, EdgeR, DESeq2, LR_L1, LR_L2, MLP, GCN, XGBoost, STD_VAR, STD_PCA, STD_MI, STD_EdgeR, STD_DESeq2, STD_LR_L1, STD_LR_L2, STD_MLP, STD_GCN, STD_XGBoost\n")
        for n_feat in n_feats:
            f.write(f"{n_feat}, {results['var'][n_feat]['mean']}, {results['PCA_PC1'][n_feat]['mean']}, {results['MI'][n_feat]['mean']}, {results['edgeR'][n_feat]['mean']}, {results['DESeq2'][n_feat]['mean']}, {results['IG_LR_L1_penalty_set_train'][n_feat]['mean']}, {results['IG_LR_L2_penalty_set_train'][n_feat]['mean']}, {results['IG_MLP_set_train'][n_feat]['mean']}, {results['IG_GCN_set_train'][n_feat]['mean']}, {results['xgboost'][n_feat]['mean']}, {results['var'][n_feat]['std']}, {results['PCA_PC1'][n_feat]['std']}, {results['MI'][n_feat]['std']}, {results['edgeR'][n_feat]['std']}, {results['DESeq2'][n_feat]['std']}, {results['IG_LR_L1_penalty_set_train'][n_feat]['std']}, {results['IG_LR_L2_penalty_set_train'][n_feat]['std']}, {results['IG_MLP_set_train'][n_feat]['std']}, {results['IG_GCN_set_train'][n_feat]['std']}, {results['xgboost'][n_feat]['std']}\n")
