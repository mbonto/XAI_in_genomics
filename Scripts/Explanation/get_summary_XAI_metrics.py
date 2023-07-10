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
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, DiffuseLR, DiffuseMLP)")
argParser.add_argument("--set", type=str, help="set (train or test)")
argParser.add_argument("--n_repet", type=int, help="Results are averaged for all experiments between 1 and `n_repet`")
args = argParser.parse_args()
name = args.name
model_name = args.model
set_name = args.set
n_repet = args.n_repet
print('Model    ', model_name)
print('Dataset    ', name)
XAI_method = "Integrated_Gradients"


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Summarize local PGs
exps = np.arange(1, n_repet+1)
PGU = []
PGU_adjusted = []
PGI = []


for exp in exps:
    save_name = os.path.join(model_name, f"exp_{exp}", XAI_method, "figures")
    with open(os.path.join(save_path, save_name, f"local_XAI_{set_name}.csv"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(', ')
            if line[0] == 'PGU':
                PGU.append(float(line[1]))
            elif line[0] == 'PGU_adjusted':
                PGU_adjusted.append(float(line[1]))
            elif line[0] == 'PGI':
                PGI.append(float(line[1]))
assert len(PGU) == len(exps)
print("Local Prediction Gaps")
print(f"  PGU: {np.round(np.mean(PGU) , 2)} +- {np.round(np.std(PGU) , 2)}")
print(f"      adjusted: {np.round(np.mean(PGU_adjusted) , 2)} +- {np.round(np.std(PGU_adjusted) , 2)}")
print(f"  PGI: {np.round(np.mean(PGI) , 2)} +- {np.round(np.std(PGI) , 2)}")


# Summarize global PGs
PGU = []
PGU_adjusted = []
PGI = []
PGR = []

for exp in exps:
    save_name = os.path.join(model_name, f"exp_{exp}", XAI_method, "figures")   
    with open(os.path.join(save_path, save_name, f"global_XAI_{set_name}.csv"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(', ')
            if line[0] == 'sum':
                if line[1] == 'PGU':
                    PGU.append(float(line[2]))
                elif line[1] == 'PGU_adjusted':
                    PGU_adjusted.append(float(line[2]))
                elif line[1] == 'PGI':
                    PGI.append(float(line[2]))
            if line[0] == 'PGR':
                PGR.append(float(line[1]))
assert len(PGU) == len(exps)
print(' ')
print("Global Prediction Gaps")
print(f"  PGU: {np.round(np.mean(PGU) , 2)} +- {np.round(np.std(PGU) , 2)}")
print(f"      adjusted: {np.round(np.mean(PGU_adjusted) , 2)} +- {np.round(np.std(PGU_adjusted) , 2)}")
print(f"  PGI: {np.round(np.mean(PGI) , 2)} +- {np.round(np.std(PGI) , 2)}")
print(f"  PGR: {np.round(np.mean(PGR) , 2)} +- {np.round(np.std(PGR) , 2)}")


# Summarize FA
if name not in ['BRCA', 'KIRC', 'pancan']:
    print("")
    print("Feature agreement")
    global_setting = "rank_mean_scores"  # "rank_mean_scores", "intersect_rank_scores"
    local = []
    _global = []
    
    for exp in exps:
        save_name = os.path.join(model_name, f"exp_{exp}", XAI_method, "figures")   
        with open(os.path.join(save_path, save_name, f"ranking_{global_setting}_{set_name}.csv"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(', ')
                if line[0] == 'local':
                    local.append(float(line[1]))
                if line[0] == 'global':
                    _global.append(float(line[1]))
    assert len(local) == len(exps)
    print(f"  Local: {np.round(np.mean(local) , 2)} +- {np.round(np.std(local) , 2)}")
    print(f"  Global: {np.round(np.mean(_global) , 2)} +- {np.round(np.std(_global) , 2)}")
    
    print("")
    print("With diffusion")
    local = []
    _global = []
    
    for exp in exps:
        save_name = os.path.join(model_name, f"exp_{exp}", XAI_method, "figures")   
        with open(os.path.join(save_path, save_name, f"ranking_diffusion_{global_setting}_{set_name}.csv"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(', ')
                if line[0] == 'local':
                    local.append(float(line[1]))
                if line[0] == 'global':
                    _global.append(float(line[1]))
    assert len(local) == len(exps)
    print(f"  Local: {np.round(np.mean(local) , 2)} +- {np.round(np.std(local) , 2)}")
    print(f"  Global: {np.round(np.mean(_global) , 2)} +- {np.round(np.std(_global) , 2)}")

