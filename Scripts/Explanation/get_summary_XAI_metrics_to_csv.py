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
argParser.add_argument("--set", type=str, help="set (train or test)")
argParser.add_argument("--n_repet", type=int, help="Results are averaged for all experiments between 1 and `n_repet`")
args = argParser.parse_args()
name = args.name 
set_name = args.set
n_repet = args.n_repet


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Variables
model_names = {"LR+L1": "LR_L1_penalty", "LR+L2": "LR_L2_penalty", "MLP": "MLP", "GCN": "GCN"}
XAI_methods = {"LR+L1": "Integrated_Gradients", "LR+L2": "Integrated_Gradients", "MLP": "Integrated_Gradients", "GCN": "Integrated_Gradients"}
exps = np.arange(1, n_repet+1)


# Results
PGI_local = {}
PGU_local = {}
PGI_global = {}
PGU_global = {}
# PGR_both = {}

# Summarize local PGs for each model
models = ["MLP", "GCN"]
for model in models:
    PGI_local[model] = {}
    PGU_local[model] = {}

    PGU = []
    PGI = []
    for exp in exps:
        save_name = os.path.join(model_names[model], f"exp_{exp}", XAI_methods[model], "figures")
        with open(os.path.join(save_path, save_name, f"local_XAI_{set_name}.csv"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(', ')
                if line[0] == 'PGU':
                    PGU.append(float(line[1]))
                elif line[0] == 'PGI':
                    PGI.append(float(line[1]))
    assert len(PGU) == len(exps)
    assert len(PGI) == len(exps)

    PGI_local[model]["mean"] = np.round(np.mean(PGI), 1)
    PGI_local[model]["std"] = np.round(np.std(PGI), 1)
    PGU_local[model]["mean"] = np.round(np.mean(PGU), 1)
    PGU_local[model]["std"] = np.round(np.std(PGU), 1)


# Summarize global PGs for each model
models = ["LR+L1","LR+L2", "MLP", "GCN"]
for model in models:
    PGI_global[model] = {}
    PGU_global[model] = {}
    # PGR_both[model] = {}

    PGU = []
    PGI = []
    # PGR = []
    for exp in exps:
        save_name = os.path.join(model_names[model], f"exp_{exp}", XAI_methods[model], "figures")   
        with open(os.path.join(save_path, save_name, f"global_XAI_{set_name}.csv"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(', ')
                if line[0] == 'PGU':
                    PGU.append(float(line[1]))
                elif line[0] == 'PGI':
                    PGI.append(float(line[1]))
                # elif line[0] == 'PGR':
                #    PGR.append(float(line[1]))
    assert len(PGU) == len(exps)
    assert len(PGI) == len(exps)
    # assert len(PGR) == len(exps)

    PGI_global[model]["mean"] = np.round(np.mean(PGI), 1)
    PGI_global[model]["std"] = np.round(np.std(PGI), 1)
    PGU_global[model]["mean"] = np.round(np.mean(PGU), 1)
    PGU_global[model]["std"] = np.round(np.std(PGU), 1)
    # PGR_both[model]["mean"] = np.round(np.mean(PGR), 1)
    # PGR_both[model]["std"] = np.round(np.std(PGR), 1)


# Save
models = ["MLP", "GCN"]
with open(os.path.join(save_path, "figures", f"PGI_local_{set_name}.csv"), "w") as f:
    f.write("Index, Model, Value, Std\n")
    for i, model in enumerate(models):
        f.write(f"{i+1}, {model}, {PGI_local[model]['mean']}, {PGI_local[model]['std']}\n")

with open(os.path.join(save_path, "figures", f"PGU_local_{set_name}.csv"), "w") as f:
    f.write("Index, Model, Value, Std\n")
    for i, model in enumerate(models):
        f.write(f"{i+1}, {model}, {PGU_local[model]['mean']}, {PGU_local[model]['std']}\n")

models = ["LR+L1","LR+L2", "MLP", "GCN"]
with open(os.path.join(save_path, "figures", f"PGI_global_{set_name}.csv"), "w") as f:
    f.write("Index, Model, Value, Std\n")
    for i, model in enumerate(models):
        f.write(f"{i+1}, {model}, {PGI_global[model]['mean']}, {PGI_global[model]['std']}\n")

with open(os.path.join(save_path, "figures", f"PGU_global_{set_name}.csv"), "w") as f:
    f.write("Index, Model, Value, Std\n")
    for i, model in enumerate(models):
        f.write(f"{i+1}, {model}, {PGU_global[model]['mean']}, {PGU_global[model]['std']}\n")

# with open(os.path.join(save_path, "figures", f"PGR_both_{set_name}.csv"), "w") as f:
#     f.write("Index, Model, Value, Std\n")
#     for i, model in enumerate(models):
#         f.write(f"{i+1}, {model}, {PGR_both[model]['mean']}, {PGR_both[model]['std']}\n")

